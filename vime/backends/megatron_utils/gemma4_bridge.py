from argparse import Namespace
from pathlib import Path

from transformers import AutoConfig, PretrainedConfig


_RUNTIME_FIELDS = (
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "sequence_parallel",
    "context_parallel_size",
    "gradient_accumulation_fusion",
    "calculate_per_token_loss",
    "variable_seq_lengths",
    "attention_softmax_in_fp32",
    "fp32_residual_connection",
    "deterministic_mode",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "recompute_modules",
    "cpu_offloading_num_layers",
    "distribute_saved_activations",
    "cpu_offloading",
    "tp_comm_overlap",
    "fp8",
    "fp8_recipe",
    "attention_backend",
    "moe_token_dispatcher_type",
    "moe_router_load_balancing_type",
)


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return getattr(config, "text_config", config)


def is_gemma4_bridge_model(config_or_path: PretrainedConfig | str | Path) -> bool:
    config = (
        AutoConfig.from_pretrained(config_or_path, trust_remote_code=True)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    text_config = _text_config(config)
    return text_config.model_type == "gemma4_text" and bool(getattr(text_config, "enable_moe_block", False))


def create_gemma4_bridge(path: str | Path):
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    pretrained = PreTrainedCausalLM.from_pretrained(path, trust_remote_code=True)
    text_config = _text_config(pretrained.config)
    if not is_gemma4_bridge_model(text_config):
        raise ValueError("Megatron-Bridge currently supports only MoE Gemma4 text models")

    text_config.architectures = ["Gemma4ForCausalLM"]
    text_config.allow_global_per_layer_attribute_access = True
    pretrained.config = text_config
    return AutoBridge(pretrained)


def create_gemma4_provider(args: Namespace):
    provider = create_gemma4_bridge(args.hf_checkpoint).to_megatron_provider(load_weights=False)
    for field in _RUNTIME_FIELDS:
        if hasattr(args, field):
            setattr(provider, field, getattr(args, field))

    optional_fields = {
        "decoder_first_pipeline_num_layers": "num_layers_in_first_pipeline_stage",
        "decoder_last_pipeline_num_layers": "num_layers_in_last_pipeline_stage",
        "moe_router_bias_update_rate": "moe_router_bias_update_rate",
        "moe_aux_loss_coeff": "moe_aux_loss_coeff",
    }
    for arg_name, provider_name in optional_fields.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(provider, provider_name, value)

    provider.finalize()
    return provider
