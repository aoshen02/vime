# the file to manage all vllm deps in the megatron actor
try:
    from vllm.srt.layers.quantization.fp8_utils import quant_weight_ue8m0, transform_scale_ue8m0
    from vllm.srt.model_loader.utils import should_deepgemm_weight_requant_ue8m0
except ImportError:
    quant_weight_ue8m0 = None
    transform_scale_ue8m0 = None
    should_deepgemm_weight_requant_ue8m0 = None

try:
    from vllm.srt.utils.patch_torch import monkey_patch_torch_reductions
except ImportError:
    from vllm.srt.patch_torch import monkey_patch_torch_reductions


from vllm.srt.utils import MultiprocessingSerializer


try:
    from vllm.srt.weight_sync.tensor_bucket import FlattenedTensorBucket  # type: ignore[import]
except ImportError:
    from vllm.srt.model_executor.model_runner import FlattenedTensorBucket  # type: ignore[import]

__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
    "monkey_patch_torch_reductions",
    "MultiprocessingSerializer",
    "FlattenedTensorBucket",
]
