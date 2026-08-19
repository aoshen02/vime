from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils import gemma4_bridge
from vime.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import _replace_task_weights


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(model_type="gemma4_text", enable_moe_block=True), True),
        (SimpleNamespace(model_type="gemma4_text", enable_moe_block=False), False),
        (
            SimpleNamespace(text_config=SimpleNamespace(model_type="gemma4_text", enable_moe_block=True)),
            True,
        ),
        (SimpleNamespace(model_type="qwen3", enable_moe_block=True), False),
    ],
)
def test_identifies_only_moe_gemma4_text_models(config, expected):
    assert gemma4_bridge.is_gemma4_bridge_model(config) is expected


@pytest.mark.unit
def test_provider_uses_runtime_parallelism(monkeypatch):
    provider = SimpleNamespace(finalize=lambda: setattr(provider, "finalized", True))
    bridge = SimpleNamespace(to_megatron_provider=lambda load_weights: provider)
    monkeypatch.setattr(gemma4_bridge, "create_gemma4_bridge", lambda _: bridge)
    args = SimpleNamespace(
        hf_checkpoint="gemma4",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=4,
        decoder_first_pipeline_num_layers=14,
        moe_router_load_balancing_type="none",
    )

    assert gemma4_bridge.create_gemma4_provider(args) is provider
    assert provider.tensor_model_parallel_size == 2
    assert provider.pipeline_model_parallel_size == 2
    assert provider.expert_model_parallel_size == 4
    assert provider.moe_router_load_balancing_type == "none"
    assert provider.num_layers_in_first_pipeline_stage == 14
    assert provider.finalized


@dataclass(frozen=True)
class ConversionTask:
    global_param_name: str
    param_weight: object


class Weight:
    def __init__(self, replacement):
        self.replacement = replacement

    def cuda(self):
        return self.replacement


@pytest.mark.unit
def test_conversion_tasks_use_global_parameter_names():
    replacement = object()
    task = ConversionTask("decoder.layers.16.mlp.weight", torch.nn.Parameter(torch.zeros(1)))

    converted = list(_replace_task_weights([task], {task.global_param_name: Weight(replacement)}))

    assert converted[0].param_weight is replacement


@pytest.mark.unit
def test_conversion_tasks_keep_persistent_buffers_but_reject_missing_parameters():
    buffer_task = ConversionTask("decoder.layers.0.layer_scalar", torch.zeros(1))
    parameter_task = ConversionTask("decoder.layers.0.mlp.weight", torch.nn.Parameter(torch.zeros(1)))

    assert list(_replace_task_weights([buffer_task], {})) == [buffer_task]
    with pytest.raises(KeyError, match="decoder.layers.0.mlp.weight"):
        list(_replace_task_weights([parameter_task], {}))
