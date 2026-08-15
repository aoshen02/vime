import sys
from types import SimpleNamespace

import pytest
import torch

try:
    from vime_plugins.models.gemma4 import Gemma4SelfAttention, VNorm, get_gemma4_attention_layout
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not (missing == "megatron" or missing.startswith("megatron.") or missing == "mbridge"):
        raise
    from tests.gemma4._standalone_imports import load_gemma4_model_module

    _gemma4 = load_gemma4_model_module()
    Gemma4SelfAttention = _gemma4.Gemma4SelfAttention
    VNorm = _gemma4.VNorm
    get_gemma4_attention_layout = _gemma4.get_gemma4_attention_layout


def test_attention_layout_reads_transformers_heterogeneous_per_layer_config():
    local_config = SimpleNamespace(head_dim=256, num_key_value_heads=8, sliding_window=1024)
    global_config = SimpleNamespace(head_dim=512, num_key_value_heads=1, sliding_window=1024)
    hf_text = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        per_layer_config=[local_config, global_config],
    )

    layout = get_gemma4_attention_layout(hf_text)

    assert layout.local_head_dim == 256
    assert layout.global_head_dim == 512
    assert layout.local_num_kv_heads == 8
    assert layout.global_num_kv_heads == 1
    assert layout.sliding_window == 1024
    assert layout.global_attn_layers == frozenset({1})


def test_attention_layout_rejects_inconsistent_same_type_geometry():
    hf_text = SimpleNamespace(
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        per_layer_config=[
            SimpleNamespace(head_dim=256, num_key_value_heads=8, sliding_window=1024),
            SimpleNamespace(head_dim=128, num_key_value_heads=8, sliding_window=1024),
            SimpleNamespace(head_dim=512, num_key_value_heads=1, sliding_window=1024),
        ],
    )

    with pytest.raises(ValueError, match="sliding_attention layers have inconsistent attention geometry"):
        get_gemma4_attention_layout(hf_text)


def _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size):
    attn = object.__new__(Gemma4SelfAttention)
    torch.nn.Module.__init__(attn)

    q_per_kv = num_attention_heads // num_kv_heads
    out_width = num_kv_heads * (q_per_kv + 2) * head_dim
    linear_qkv = torch.nn.Linear(hidden_size, out_width, bias=False)
    torch.nn.init.normal_(linear_qkv.weight, std=0.02)

    def _linear_qkv(h):
        return linear_qkv(h), None

    attn.linear_qkv = _linear_qkv
    attn.num_attention_heads_per_partition = num_attention_heads
    attn.num_query_groups_per_partition = num_kv_heads
    attn.hidden_size_per_attention_head = head_dim
    attn.q_layernorm = torch.nn.LayerNorm(head_dim)
    attn.k_layernorm = torch.nn.LayerNorm(head_dim)
    attn.v_norm = VNorm(head_dim, eps=1e-6)
    attn.config = SimpleNamespace(
        layernorm_epsilon=1e-6,
        attention_k_eq_v=True,
        num_query_groups=num_kv_heads,
    )
    attn.world_size = 1
    attn._is_global = False  # flipped per-test
    return attn, linear_qkv


def test_global_k_eq_v_produces_k_norm_and_v_norm_of_raw_k():
    torch.manual_seed(0)
    num_attention_heads, num_kv_heads, head_dim, hidden_size = 8, 2, 512, 256
    attn, linear_qkv = _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size)
    attn._is_global = True

    seq_len, batch = 4, 1
    hidden = torch.randn(seq_len, batch, hidden_size)

    query, key, value = attn.get_query_key_value_tensors(hidden)

    assert query.shape == (seq_len, batch, num_attention_heads, head_dim)
    assert key.shape == (seq_len, batch, num_kv_heads, head_dim)
    assert value.shape == (seq_len, batch, num_kv_heads, head_dim)

    mixed, _ = attn.linear_qkv(hidden)
    q_per_kv = num_attention_heads // num_kv_heads
    mixed = mixed.view(seq_len, batch, num_kv_heads, (q_per_kv + 2) * head_dim)
    q_width = q_per_kv * head_dim
    raw_q, raw_k, _raw_v = torch.split(mixed, [q_width, head_dim, head_dim], dim=3)
    raw_q = raw_q.reshape(seq_len, batch, -1, head_dim)

    expected_query = attn.q_layernorm(raw_q)
    expected_key = attn.k_layernorm(raw_k)
    expected_value = attn.v_norm(raw_k)

    assert torch.allclose(query, expected_query), "query mismatch"
    assert torch.allclose(key, expected_key), "key must be k_norm(raw_k)"
    assert torch.allclose(value, expected_value), (
        "value must be v_norm(raw_k); if this fails, v is being derived from " "k_norm(raw_k) instead of raw_k"
    )


def test_global_k_eq_v_does_not_mutate_k_layernorm():
    torch.manual_seed(1)
    attn, _ = _stub_attention(8, 2, 512, 256)
    attn._is_global = True

    k_layernorm_before = attn.k_layernorm
    hidden = torch.randn(3, 1, 256)
    _ = attn.get_query_key_value_tensors(hidden)
    assert attn.k_layernorm is k_layernorm_before


def test_global_k_eq_v_gathers_before_splitting_when_kv_heads_are_replicated(monkeypatch):
    torch.manual_seed(3)
    num_attention_heads, num_kv_heads, head_dim, hidden_size = 4, 1, 2, 3
    attn, linear_qkv = _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size)
    attn._is_global = True
    attn.world_size = 2

    hidden = torch.randn(2, 1, hidden_size)
    full_qkv = linear_qkv(hidden)
    local_qkv = full_qkv[..., : full_qkv.size(-1) // 2]
    attn.linear_qkv = lambda _hidden: (local_qkv, None)

    module = sys.modules[Gemma4SelfAttention.__module__]
    monkeypatch.setattr(
        module,
        "all_gather_last_dim_from_tensor_parallel_region",
        lambda _local: full_qkv,
    )
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    query, key, value = attn.get_query_key_value_tensors(hidden)

    mixed = full_qkv.view(2, 1, num_kv_heads, (num_attention_heads + 2) * head_dim)
    raw_query, raw_key, _ = torch.split(
        mixed,
        [num_attention_heads * head_dim, head_dim, head_dim],
        dim=3,
    )
    raw_query = raw_query.reshape(2, 1, num_attention_heads, head_dim)[:, :, :2, :]

    assert query.shape == (2, 1, 2, head_dim)
    assert torch.allclose(query, attn.q_layernorm(raw_query))
    assert torch.allclose(key, attn.k_layernorm(raw_key))
    assert torch.allclose(value, attn.v_norm(raw_key))


def test_global_k_eq_v_rejects_output_gate():
    attn, _ = _stub_attention(8, 2, 512, 256)
    attn._is_global = True
    with pytest.raises(NotImplementedError):
        attn.get_query_key_value_tensors(torch.randn(3, 1, 256), output_gate=True)


def test_sliding_layer_applies_v_norm_to_value():
    torch.manual_seed(2)
    num_attention_heads, num_kv_heads, head_dim, hidden_size = 8, 2, 256, 256
    attn, linear_qkv = _stub_attention(num_attention_heads, num_kv_heads, head_dim, hidden_size)
    attn._is_global = False

    seq_len, batch = 3, 1
    raw_q = torch.randn(seq_len, batch, num_attention_heads, head_dim)
    raw_k = torch.randn(seq_len, batch, num_kv_heads, head_dim)
    raw_v = torch.randn(seq_len, batch, num_kv_heads, head_dim)

    def _fake_parent(*_a, **_k):
        return raw_q, raw_k, raw_v

    import unittest.mock as mock

    _Base = Gemma4SelfAttention.__mro__[1]
    with mock.patch.object(_Base, "get_query_key_value_tensors", _fake_parent):
        query, key, value = attn.get_query_key_value_tensors(torch.randn(seq_len, batch, hidden_size))

    assert torch.equal(query, raw_q)
    assert torch.equal(key, raw_k)
    assert torch.allclose(value, attn.v_norm(raw_v))
