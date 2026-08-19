import dataclasses
from collections.abc import Callable, Sequence

import torch

from vime.utils.misc import chunk_named_params_by_size
from vime.utils.types import ParamInfo

from ..gemma4_bridge import create_gemma4_bridge
from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bridge = create_gemma4_bridge(self.args.hf_checkpoint)

    def get_hf_weight_chunks(
        self,
        megatron_local_weights,
        progress_desc: str = "Update weights",
        should_convert_chunk: Callable[[int], bool] | None = None,
        param_info_buckets: Sequence[Sequence[ParamInfo]] | None = None,
    ):
        del progress_desc, param_info_buckets
        local_weights = {strip_param_name_prefix(name): weight for name, weight in megatron_local_weights.items()}
        tasks = _replace_task_weights(self._bridge.get_conversion_tasks(self.model), local_weights)
        weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=tasks)

        def converted_weights():
            for hf_name, weight, megatron_name in weights:
                weight = postprocess_hf_param(self.args, megatron_name, hf_name, weight)
                yield from quantize_params(
                    self.args,
                    megatron_name,
                    [(hf_name, weight)],
                    self.quantization_config,
                    self.transform_ue8m0,
                )

        chunks = chunk_named_params_by_size(converted_weights(), self.args.update_weight_buffer_size)
        for index, chunk in enumerate(chunks):
            yield chunk if should_convert_chunk is None or should_convert_chunk(index) else []


def _replace_task_weights(tasks, local_weights):
    def replace(task):
        if task is None or task.param_weight is None:
            return task
        key = task.global_param_name
        if key not in local_weights:
            if not isinstance(task.param_weight, torch.nn.Parameter):
                return task
            raise KeyError(f"Megatron-Bridge conversion weight is missing: {key}")
        return dataclasses.replace(task, param_weight=local_weights[key].cuda())

    return _MappedTasks(replace, tasks)


class _MappedTasks:
    def __init__(self, fn, tasks):
        self.fn = fn
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return map(self.fn, self.tasks)
