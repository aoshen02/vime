from types import SimpleNamespace

import pytest
import torch

from vime_plugins.models.glm5.glm5 import (
    _DSAKVFP8QAT,
    DSAMLASelfAttention,
    IdentityOp,
    _get_fp8_aligned_absorb_weight,
    _get_vllm_rope_cache,
    _VLLMIndexerHeadWeights,
    _VLLMRoPE,
)

NUM_GPUS = 1

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA kernels from vLLM 0.27.1")


def _cosine_error(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    return 1 - 2 * (actual * expected).sum() / ((actual * actual).sum() + (expected * expected).sum())


def _interleaved_rope_reference(
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    half = value.shape[-1] // 2
    broadcast_shape = (positions.numel(),) + (1,) * (value.ndim - 2) + (half,)
    cos = cos_sin_cache[positions, :half].view(broadcast_shape).to(value.dtype)
    sin = cos_sin_cache[positions, half:].view(broadcast_shape).to(value.dtype)
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


def test_external_deepgemm_bf16_indexer_head_matches_linear_forward_and_backward():
    torch.manual_seed(7)
    hidden_states = torch.randn((128, 1, 7168), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn((256, 7168), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output_grad = torch.randn((128, 1, 256), device="cuda", dtype=torch.float32)

    actual = _VLLMIndexerHeadWeights.apply(hidden_states, weight)
    expected = hidden_states.detach().reshape(-1, hidden_states.shape[-1]).float() @ weight.detach().float().t()

    assert actual.shape == (128, 1, 256)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert _cosine_error(actual, expected.view_as(actual)) < 1.0e-5

    actual.backward(output_grad)
    flat_grad = output_grad.reshape(-1, output_grad.shape[-1])
    expected_input_grad = (flat_grad @ weight.detach().float()).view_as(hidden_states).to(torch.bfloat16)
    expected_weight_grad = (flat_grad.t() @ hidden_states.detach().reshape(-1, hidden_states.shape[-1]).float()).to(
        torch.bfloat16
    )

    torch.testing.assert_close(hidden_states.grad, expected_input_grad, rtol=2.0e-2, atol=2.0e-2)
    torch.testing.assert_close(weight.grad, expected_weight_grad, rtol=2.0e-2, atol=2.0e-2)


def test_fp8_absorb_weight_matches_vllm_ue8m0_helpers_layout_and_ste(monkeypatch):
    from vllm.model_executor.layers.quantization.utils.fp8_utils import requant_weight_ue8m0_inplace
    from vllm.utils import deep_gemm as vllm_deep_gemm

    torch.manual_seed(11)
    linear = torch.nn.Linear(257, 129, bias=False, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(vllm_deep_gemm, "is_deep_gemm_e8m0_used", lambda: True)

    reference_qweight, reference_scale = vllm_deep_gemm.per_block_cast_to_fp8(
        linear.weight.detach().contiguous(),
        block_size=[128, 128],
    )
    requant_weight_ue8m0_inplace(reference_qweight, reference_scale, (128, 128))
    expanded_scale = reference_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expected = (
        reference_qweight.float() * expanded_scale[: reference_qweight.shape[0], : reference_qweight.shape[1]]
    ).to(torch.bfloat16)

    actual = _get_fp8_aligned_absorb_weight(linear)

    assert actual.shape == linear.weight.shape
    assert actual.stride() == linear.weight.stride()
    assert actual.dtype == torch.bfloat16
    assert reference_scale.shape == (2, 3)
    torch.testing.assert_close(torch.log2(reference_scale), torch.log2(reference_scale).round(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    upstream_grad = torch.randn_like(actual)
    actual.backward(upstream_grad)
    torch.testing.assert_close(linear.weight.grad, upstream_grad, rtol=0.0, atol=0.0)


def test_vllm_interleaved_rope_matches_reference_forward_and_inverse_gradient():
    torch.manual_seed(19)
    value = torch.randn((4, 3, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.tensor([5, 0, 3, 1], device="cuda", dtype=torch.long)
    cache = _get_vllm_rope_cache(value.device, value.shape[-1], 10000.0, 6)
    original = value.detach().clone()
    output_grad = torch.randn_like(value)

    actual = _VLLMRoPE.apply(value, cache, positions)
    expected = _interleaved_rope_reference(original, cache, positions)

    assert actual.shape == value.shape
    assert actual.stride() == value.stride()
    assert actual.data_ptr() != value.data_ptr()
    torch.testing.assert_close(value.detach(), original, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-2)

    actual.backward(output_grad)
    expected_grad = _interleaved_rope_reference(
        output_grad, torch.cat((cache[:, :32], -cache[:, 32:]), dim=-1), positions
    )
    torch.testing.assert_close(value.grad, expected_grad, rtol=0.0, atol=2.0e-2)


def test_dsa_kv_fp8_qat_matches_vllm_ds_mla_cache_layout_and_ste():
    from vllm import _custom_ops as ops

    torch.manual_seed(23)
    kv = torch.randn((5, 1, 576), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = _DSAKVFP8QAT.apply(kv)

    kv_cache = torch.zeros((1, kv.shape[0], 656), device="cuda", dtype=torch.uint8)
    slot_mapping = torch.arange(kv.shape[0], device="cuda", dtype=torch.long)
    ops.concat_and_cache_mla(
        kv.detach()[..., :512].reshape(-1, 512),
        kv.detach()[..., 512:].reshape(-1, 64),
        kv_cache,
        slot_mapping,
        "fp8_ds_mla",
        torch.tensor(1.0, device="cuda", dtype=torch.float32),
    )

    expected_rows = []
    for cache_row in kv_cache[0]:
        quantized_nope = cache_row[:512].view(torch.float8_e4m3fn).float().view(4, 128)
        scales = cache_row[512:528].view(torch.float32)
        dequantized_nope = (quantized_nope * scales[:, None]).to(torch.bfloat16).reshape(512)
        rope = cache_row[528:].view(torch.bfloat16)
        expected_rows.append(torch.cat((dequantized_nope, rope)))
    expected = torch.stack(expected_rows).view_as(kv)

    assert actual.shape == kv.shape
    assert actual.stride() == kv.stride()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[..., 512:], kv.detach()[..., 512:], rtol=0.0, atol=0.0)

    upstream_grad = torch.randn_like(actual)
    actual.backward(upstream_grad)
    torch.testing.assert_close(kv.grad, upstream_grad, rtol=0.0, atol=0.0)


class _FusedQUp(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.layer_norm_weight = torch.nn.Parameter(weight.clone())


def _make_attention(weight: torch.Tensor) -> DSAMLASelfAttention:
    attention = DSAMLASelfAttention.__new__(DSAMLASelfAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        normalization="RMSNorm",
        layernorm_epsilon=1.0e-5,
        layernorm_zero_centered_gamma=True,
    )
    attention.q_layernorm = IdentityOp()
    attention.linear_q_up_proj = _FusedQUp(weight)
    return attention


def test_batch_invariant_rmsnorm_matches_fp32_reference_and_row_isolation(monkeypatch):
    torch.manual_seed(29)
    stored_weight = torch.randn((512,), device="cuda", dtype=torch.bfloat16)
    attention = _make_attention(stored_weight)
    monkeypatch.setenv("MEGATRON_USE_VLLM_FUSED_RESIDUAL_RMS", "1")
    q_compressed = torch.randn((7, 2, 512), device="cuda", dtype=torch.bfloat16)

    batched = attention._get_indexer_q_input(q_compressed)
    isolated = attention._get_indexer_q_input(q_compressed[3:4])
    expected = torch.nn.functional.rms_norm(
        q_compressed.float(),
        normalized_shape=(q_compressed.shape[-1],),
        weight=stored_weight.float() + 1.0,
        eps=attention.config.layernorm_epsilon,
    ).to(torch.bfloat16)

    assert batched.shape == q_compressed.shape
    assert batched.stride() == q_compressed.stride()
    assert not batched.requires_grad
    torch.testing.assert_close(batched, expected, rtol=0.0, atol=2.0e-2)
    torch.testing.assert_close(batched[3:4], isolated, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
