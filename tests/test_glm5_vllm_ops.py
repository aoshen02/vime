import math

import pytest
import torch

from vime_plugins.models.glm5.ops import indexer, sparse_mla

NUM_GPUS = 1


def _require_cuda_capability(minimum_major: int = 9) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major < minimum_major:
        pytest.skip(f"SM{minimum_major}0 or newer is required")


def _unpack_indexer_k_cache(packed_k: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, cache_stride = packed_k.shape
    assert cache_stride == 132
    blocks = packed_k.view(num_blocks, -1)
    values = blocks[:, : block_size * 128].reshape(-1, 128)[:num_tokens]
    scales = blocks[:, block_size * 128 :].reshape(-1, 4)[:num_tokens]
    values = values.contiguous().view(torch.float8_e4m3fn)
    scales = scales.contiguous().view(torch.float32).reshape(num_tokens)
    return values, scales


def test_fp8_indexer_vllm_layout_and_numeric_contract(monkeypatch):
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils import deep_gemm

    num_queries, num_heads, num_keys, head_dim = 2, 2, 3, 128
    index_q = (
        torch.arange(num_queries * num_heads * head_dim, dtype=torch.float32).reshape(num_queries, num_heads, head_dim)
        / 256
        - 1
    ).to(torch.bfloat16)
    index_k = (torch.arange(num_keys * head_dim, dtype=torch.float32).reshape(num_keys, 1, head_dim) / 192 - 1).to(
        torch.bfloat16
    )
    weights = torch.tensor([[0.5, -0.25], [1.25, 0.75]], dtype=torch.float32)
    starts = torch.tensor([0, 1], dtype=torch.int64)
    ends = torch.tensor([2, 3], dtype=torch.int64)
    q_scales = torch.tensor([[[0.5], [1.0]], [[1.5], [2.0]]], dtype=torch.float32)
    k_scales = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32)
    captured = {}

    def fake_quantize(x, group_size, *, use_ue8m0):
        captured["quantize_input"] = x.clone()
        captured["quantize_group_size"] = group_size
        captured["quantize_ue8m0"] = use_ue8m0
        return x.to(torch.float8_e4m3fn), q_scales.reshape(-1, 1)

    def fake_quantize_and_cache(k, cache, slot_mapping, quant_block_size, cache_dtype):
        captured["cache_input"] = k.clone()
        captured["cache_shape"] = cache.shape
        captured["slot_mapping"] = slot_mapping.clone()
        captured["quant_block_size"] = quant_block_size
        captured["cache_dtype"] = cache_dtype
        cache.zero_()

    def fake_gather(cache, dst_k, dst_scale, block_table, cu_seq_lens):
        captured["gather_cache_shape"] = cache.shape
        captured["gather_block_table"] = block_table.clone()
        captured["gather_cu_seq_lens"] = cu_seq_lens.clone()
        dst_k.copy_(captured["cache_input"].to(torch.float8_e4m3fn).view(torch.uint8))
        dst_scale.copy_(k_scales.view(torch.uint8).reshape(num_keys, 4))

    def fake_mqa_logits(q, kv, scaled_weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
        q_values, q_scale = q
        k_values, k_scale = kv
        captured["mqa_q_values"] = q_values.clone()
        captured["mqa_q_scale"] = q_scale
        captured["mqa_k_values"] = k_values.clone()
        captured["mqa_k_scale"] = k_scale.clone()
        captured["mqa_weights"] = scaled_weights.clone()
        captured["mqa_starts"] = cu_seqlen_ks.clone()
        captured["mqa_ends"] = cu_seqlen_ke.clone()
        captured["mqa_clean_logits"] = clean_logits
        scores = torch.einsum("mhd,nd->hmn", q_values.float(), k_values.float() * k_scale[:, None])
        return (scores.relu() * scaled_weights.T.unsqueeze(-1)).sum(dim=0)

    monkeypatch.setattr(fp8_utils, "per_token_group_quant_fp8", fake_quantize)
    monkeypatch.setattr(ops, "indexer_k_quant_and_cache", fake_quantize_and_cache)
    monkeypatch.setattr(ops, "cp_gather_indexer_k_quant_cache", fake_gather)
    monkeypatch.setattr(deep_gemm, "fp8_fp4_mqa_logits", fake_mqa_logits)

    actual = indexer._vllm_fp8_indexer_logits(index_q, index_k, weights, starts, ends)

    expected_q = torch.cat((index_q[..., -64:], index_q[..., :-64]), dim=-1).contiguous()
    expected_k = torch.cat((index_k[:, 0, -64:], index_k[:, 0, :-64]), dim=-1).contiguous()
    expected_q_fp8 = expected_q.to(torch.float8_e4m3fn)
    expected_k_fp8 = expected_k.to(torch.float8_e4m3fn)
    expected_weights = weights * q_scales.squeeze(-1) / math.sqrt(head_dim)
    expected_scores = torch.einsum(
        "mhd,nd->hmn",
        expected_q_fp8.float(),
        expected_k_fp8.float() * k_scales[:, None],
    )
    expected = (expected_scores.relu() * expected_weights.T.unsqueeze(-1)).sum(dim=0)
    positions = torch.arange(num_keys).unsqueeze(0)
    valid = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    expected = expected.masked_fill(~valid, float("-inf"))

    torch.testing.assert_close(captured["quantize_input"], expected_q.reshape(-1, head_dim))
    assert captured["quantize_group_size"] == 128
    assert captured["quantize_ue8m0"] is True
    torch.testing.assert_close(captured["cache_input"], expected_k)
    assert captured["cache_shape"] == (1, 64, 132)
    torch.testing.assert_close(captured["slot_mapping"], torch.arange(num_keys, dtype=torch.int64))
    assert captured["quant_block_size"] == 128
    assert captured["cache_dtype"] == "ue8m0"
    assert captured["gather_cache_shape"] == (1, 64, 132)
    torch.testing.assert_close(captured["gather_block_table"], torch.tensor([[0]], dtype=torch.int32))
    torch.testing.assert_close(captured["gather_cu_seq_lens"], torch.tensor([0, num_keys], dtype=torch.int32))
    torch.testing.assert_close(captured["mqa_q_values"].float(), expected_q_fp8.float())
    assert captured["mqa_q_scale"] is None
    torch.testing.assert_close(captured["mqa_k_values"].float(), expected_k_fp8.float())
    torch.testing.assert_close(captured["mqa_k_scale"], k_scales)
    torch.testing.assert_close(captured["mqa_weights"], expected_weights)
    assert captured["mqa_starts"].dtype == torch.int32
    assert captured["mqa_ends"].dtype == torch.int32
    assert captured["mqa_starts"].is_contiguous()
    assert captured["mqa_ends"].is_contiguous()
    assert captured["mqa_clean_logits"] is False
    torch.testing.assert_close(actual, expected)


def test_fp8_indexer_real_vllm_kernels_match_quantized_reference():
    _require_cuda_capability()
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from vllm.utils.deep_gemm import calc_diff
    from vllm.utils.import_utils import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("DeepGEMM is not available")

    torch.manual_seed(7)
    num_queries, num_heads, num_keys, head_dim = 512, 32, 1024, 128
    index_q = torch.randn(num_queries, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(num_keys, head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(num_queries, num_heads, device="cuda", dtype=torch.float32)
    starts = torch.zeros(num_queries, device="cuda", dtype=torch.int32)
    ends = torch.arange(num_queries, device="cuda", dtype=torch.int32) + (num_keys - num_queries)

    actual = indexer._vllm_fp8_indexer_logits(index_q, index_k, weights, starts, ends)

    q_rotated = torch.cat((index_q[..., -64:], index_q[..., :-64]), dim=-1).contiguous()
    k_rotated = torch.cat((index_k[..., -64:], index_k[..., :-64]), dim=-1).contiguous()
    q_fp8, q_scale = per_token_group_quant_fp8(
        q_rotated.view(-1, head_dim),
        head_dim,
        use_ue8m0=True,
    )
    q_fp8 = q_fp8.view_as(q_rotated)
    q_scale = q_scale.view(num_queries, num_heads, 1)
    packed_k = torch.empty(
        ((num_keys + 63) // 64, 64, head_dim + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    ops.indexer_k_quant_and_cache(
        k_rotated,
        packed_k,
        torch.arange(num_keys, dtype=torch.int64, device="cuda"),
        head_dim,
        "ue8m0",
    )
    k_fp8, k_scale = _unpack_indexer_k_cache(packed_k, num_keys)
    scaled_weights = weights * q_scale.squeeze(-1).float() / math.sqrt(head_dim)
    scores = torch.einsum("mhd,nd->hmn", q_fp8.float(), k_fp8.float() * k_scale[:, None])
    expected = (scores.relu() * scaled_weights.T.unsqueeze(-1)).sum(dim=0)
    positions = torch.arange(num_keys, dtype=torch.int32, device="cuda").unsqueeze(0)
    valid = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
    expected = expected.masked_fill(~valid, float("-inf"))

    assert torch.equal(torch.isneginf(actual), ~valid)
    diff = calc_diff(actual.masked_fill(~valid, 0), expected.masked_fill(~valid, 0))
    assert diff < 1e-3, f"relative difference is {diff}"


def test_fp8_indexer_tilelang_ste_backward_matches_pytorch(monkeypatch):
    _require_cuda_capability()
    torch.manual_seed(11)
    num_queries, num_heads, num_keys, head_dim, topk = 4, 8, 64, 128, 32
    index_q = (torch.randn(num_queries, num_heads, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    index_k = (torch.randn(num_keys, head_dim, device="cuda") * 0.1).to(torch.bfloat16)
    weights = torch.randn(num_queries, num_heads, 1, device="cuda", dtype=torch.float32)
    index_q.requires_grad_()
    index_k.requires_grad_()
    weights.requires_grad_()
    starts = torch.zeros(num_queries, dtype=torch.int32, device="cuda")
    ends = torch.full((num_queries,), num_keys, dtype=torch.int32, device="cuda")
    topk_indices = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(num_queries, 1)
    grad_scores = torch.randn(num_queries, topk, dtype=torch.float32, device="cuda")

    def fixed_logits(q, k, current_weights, cu_seqlen_ks, cu_seqlen_ke):
        assert q is index_q
        assert k is index_k
        assert current_weights.shape == (num_queries, num_heads)
        torch.testing.assert_close(cu_seqlen_ks, starts)
        torch.testing.assert_close(cu_seqlen_ke, ends)
        return torch.zeros(num_queries, num_keys, dtype=torch.float32, device="cuda")

    monkeypatch.setenv("MEGATRON_USE_VLLM_FP8_INDEXER", "1")
    monkeypatch.setattr(indexer, "_vllm_fp8_indexer_logits", fixed_logits)
    scores, returned_indices = indexer.lighting_indexer(
        index_q,
        index_k,
        weights,
        starts,
        ends,
        topk,
        topk_indices,
    )
    torch.testing.assert_close(returned_indices, topk_indices)
    torch.autograd.backward(scores, grad_scores)

    q_ref = index_q.detach().float().requires_grad_()
    k_ref = index_k.detach().float().requires_grad_()
    weights_ref = weights.detach().squeeze(-1).float().requires_grad_()
    selected_k = k_ref[topk_indices.long()]
    dense_scores = torch.einsum("mhd,mtd->mth", q_ref, selected_k)
    reference_scores = (dense_scores.relu() * weights_ref[:, None, :]).sum(dim=-1)
    torch.autograd.backward(reference_scores, grad_scores)

    assert index_q.grad.shape == index_q.shape
    assert index_k.grad.shape == index_k.shape
    assert weights.grad.shape == weights.shape
    torch.testing.assert_close(index_q.grad.float(), q_ref.grad, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(index_k.grad.float(), k_ref.grad, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(weights.grad.squeeze(-1), weights_ref.grad, rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize(("capability_major", "num_heads", "padded_heads"), [(9, 32, 64), (10, 64, 128)])
def test_sparse_mla_wrapper_native_forward_contract(monkeypatch, capability_major, num_heads, padded_heads):
    import vllm.v1.attention.ops.flashmla as flashmla

    seq_len, kv_len, head_dim, d_v, topk = 2, 5, 576, 512, 8
    q = torch.randn(seq_len, head_dim, num_heads, dtype=torch.bfloat16).transpose(1, 2)
    kv = torch.randn(kv_len, head_dim, 1, dtype=torch.bfloat16).transpose(1, 2)
    indices = torch.arange(seq_len * topk, dtype=torch.int32).reshape(seq_len, topk, 1).transpose(1, 2)
    scaling = 0.125
    captured = {}

    def fake_flash_mla_sparse_fwd(*, q, kv, indices, sm_scale, d_v):
        captured["q"] = q.clone()
        captured["kv"] = kv.clone()
        captured["indices"] = indices.clone()
        captured["sm_scale"] = sm_scale
        captured["d_v"] = d_v
        head_values = torch.arange(q.shape[1], dtype=torch.bfloat16).view(1, q.shape[1], 1)
        output = head_values.expand(q.shape[0], q.shape[1], d_v).contiguous()
        max_logits = torch.full((q.shape[0], q.shape[1]), -7.0, dtype=torch.float32)
        lse = torch.arange(q.shape[1], dtype=torch.float32).view(1, q.shape[1]).expand(q.shape[0], -1).contiguous()
        return output, max_logits, lse

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (capability_major, 0))
    monkeypatch.setattr(flashmla, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)

    output, lse = sparse_mla.VLLMSparseMLA.apply(q, kv, indices, scaling, d_v)

    assert captured["q"].shape == (seq_len, padded_heads, head_dim)
    assert captured["q"].is_contiguous()
    torch.testing.assert_close(captured["q"][:, :num_heads], q.contiguous())
    assert captured["kv"].is_contiguous()
    assert captured["indices"].is_contiguous()
    torch.testing.assert_close(captured["kv"], kv.contiguous())
    torch.testing.assert_close(captured["indices"], indices.contiguous())
    assert captured["sm_scale"] == scaling
    assert captured["d_v"] == d_v
    assert output.shape == (seq_len, num_heads, d_v)
    assert output.is_contiguous()
    assert lse.shape == (seq_len, num_heads)
    assert lse.is_contiguous()
    assert lse.dtype == torch.bfloat16
    expected_heads = torch.arange(num_heads, dtype=torch.bfloat16).view(1, num_heads, 1)
    torch.testing.assert_close(output, expected_heads.expand_as(output))
    torch.testing.assert_close(lse, torch.arange(num_heads, dtype=torch.bfloat16).view(1, -1).expand_as(lse))


def test_sparse_mla_wrapper_tilelang_backward_contract(monkeypatch):
    import vllm.v1.attention.ops.flashmla as flashmla

    seq_len, num_heads, kv_len, head_dim, d_v, topk = 2, 64, 7, 576, 512, 8
    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(kv_len, 1, head_dim, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.randint(kv_len, (seq_len, 1, topk), dtype=torch.int32)
    scaling = 0.25
    forward_output = torch.randn(seq_len, num_heads, d_v, dtype=torch.bfloat16)
    forward_lse = torch.randn(seq_len, num_heads, dtype=torch.float32)
    grad_output = torch.randn_like(forward_output).transpose(0, 1).contiguous().transpose(0, 1)
    captured = {}

    def fake_flash_mla_sparse_fwd(*, q, kv, indices, sm_scale, d_v):
        captured["forward_q"] = q.clone()
        captured["forward_kv"] = kv.clone()
        captured["forward_indices"] = indices.clone()
        captured["forward_scale"] = sm_scale
        captured["forward_d_v"] = d_v
        return forward_output.clone(), torch.empty_like(forward_lse), forward_lse.clone()

    def fake_sparse_mla_bwd(q, kv, output, current_grad_output, indices, lse, *, sm_scale):
        captured["backward_q"] = q.clone()
        captured["backward_kv"] = kv.clone()
        captured["backward_output"] = output.clone()
        captured["backward_grad_output"] = current_grad_output.clone()
        captured["backward_indices"] = indices.clone()
        captured["backward_lse"] = lse.clone()
        captured["backward_scale"] = sm_scale
        return torch.full_like(q, 2), torch.full_like(kv, 3)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(flashmla, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)
    monkeypatch.setattr(sparse_mla, "sparse_mla_bwd", fake_sparse_mla_bwd)

    output, lse = sparse_mla.VLLMSparseMLA.apply(q, kv, indices, scaling, d_v)
    torch.autograd.backward((output, lse), (grad_output, torch.ones_like(lse)))

    torch.testing.assert_close(captured["forward_q"], q.detach())
    torch.testing.assert_close(captured["forward_kv"], kv.detach())
    torch.testing.assert_close(captured["forward_indices"], indices)
    assert captured["forward_scale"] == scaling
    assert captured["forward_d_v"] == d_v
    torch.testing.assert_close(captured["backward_q"], q.detach())
    torch.testing.assert_close(captured["backward_kv"], kv.detach())
    torch.testing.assert_close(captured["backward_output"], forward_output)
    torch.testing.assert_close(captured["backward_grad_output"], grad_output)
    assert captured["backward_grad_output"].is_contiguous()
    torch.testing.assert_close(captured["backward_indices"], indices)
    torch.testing.assert_close(captured["backward_lse"], forward_lse)
    assert captured["backward_lse"].dtype == torch.float32
    assert captured["backward_scale"] == scaling
    torch.testing.assert_close(q.grad, torch.full_like(q, 2))
    torch.testing.assert_close(kv.grad, torch.full_like(kv, 3))


def test_sparse_mla_wrapper_matches_native_vllm_flashmla_forward():
    _require_cuda_capability()
    import vllm.v1.attention.ops.flashmla as flashmla

    supported, reason = flashmla.is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason)

    torch.manual_seed(13)
    seq_len, num_heads, kv_len, head_dim, d_v, topk = 1, 64, 8, 576, 512, 128
    q = torch.randn(seq_len, head_dim, num_heads, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    kv = torch.randn(kv_len, head_dim, 1, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    indices = torch.randint(kv_len, (seq_len, 1, topk), dtype=torch.int32, device="cuda")
    scaling = head_dim**-0.5
    required_heads = 128 if torch.cuda.get_device_capability()[0] >= 10 else 64
    if num_heads == required_heads:
        native_q = q.contiguous()
    else:
        native_q = q.new_empty((seq_len, required_heads, head_dim))
        native_q[:, :num_heads] = q

    native_output, _, native_lse = flashmla.flash_mla_sparse_fwd(
        q=native_q,
        kv=kv.contiguous(),
        indices=indices.contiguous(),
        sm_scale=scaling,
        d_v=d_v,
    )
    output, lse = sparse_mla.VLLMSparseMLA.apply(q, kv, indices, scaling, d_v)

    torch.testing.assert_close(output, native_output[:, :num_heads].contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(lse, native_lse[:, :num_heads].contiguous().to(torch.bfloat16), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
