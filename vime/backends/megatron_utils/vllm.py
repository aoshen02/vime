"""vLLM FP8 helpers used by Megatron weight conversion."""

import socket
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Any

import torch
import torch.distributed as dist
from vllm.utils.deep_gemm import (
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_tma_aligned_size,
    is_deep_gemm_e8m0_used,
    per_block_cast_to_fp8,
)

from vime.utils.distributed_utils import get_gloo_group


class HfWeightSource:
    def __init__(self, iterator, weights_getter: Callable[[], Mapping[str, torch.Tensor]]) -> None:
        self.iterator = iterator
        self.weights_getter = weights_getter
        self._metadata = None

    def metadata(self):
        if self._metadata is None:
            from vllm.distributed.weight_transfer.base import ParamMeta

            self._metadata = [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self]
        return self._metadata

    def __iter__(self):
        for chunk in self.iterator.get_hf_weight_chunks(self.weights_getter()):
            yield from chunk


class VimeRayWeightSyncClient:
    def __init__(
        self,
        engines: Sequence[Any],
        version_getter: Callable[[], int],
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None:
        self.engines = list(engines)
        self.version_getter = version_getter
        self.engine_gpu_counts = engine_gpu_counts
        self.draft = False

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        import ray

        refs = []
        rank_offset = 1
        for index, engine in enumerate(self.engines):
            engine_info = dict(init_info)
            if self.engine_gpu_counts is not None:
                engine_info["rank_offset"] = rank_offset
                rank_offset += self.engine_gpu_counts[index]
            refs.append(engine.init_weight_transfer_engine.remote({"init_info": engine_info}))
        ray.get(refs)

    def start_weight_update(self) -> None:
        import ray

        method = "start_draft_weight_update" if self.draft else "start_weight_update"
        ray.get([getattr(engine, method).remote() for engine in self.engines])

    def update_weights(self, update_info: dict[str, Any] | list[dict[str, Any] | None]) -> None:
        import ray

        ray.get([engine.update_weights.remote(update_info) for engine in self.engines])

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        import ray

        version = str(self.version_getter()) if weight_version is None else str(weight_version)
        ray.get([engine.finish_weight_update.remote(weight_version=version) for engine in self.engines])


def create_nccl_trainer(
    client: VimeRayWeightSyncClient,
    source: HfWeightSource,
    engine_gpu_counts: Sequence[int],
):
    import ray
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

    rendezvous = [None]
    if dist.get_rank() == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            rendezvous[0] = (ray._private.services.get_node_ip_address(), sock.getsockname()[1])
    dist.broadcast_object_list(rendezvous, src=0, group=get_gloo_group())
    master_address, master_port = rendezvous[0]
    return WeightTransferTrainerFactory.trainer_init(
        NCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=sum(engine_gpu_counts) + 1,
            rank=dist.get_rank(),
        ),
        client=client,
        source=source,
    )


def should_deepgemm_weight_requant_ue8m0(weight_block_size) -> bool:
    return weight_block_size is not None and is_deep_gemm_e8m0_used()


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: list[int],
):
    assert weight_block_size == [128, 128]
    assert weight_dequant.dtype == torch.bfloat16, f"{weight_dequant.dtype=} {weight_dequant.shape=}"
    *batch_dims, n, k = weight_dequant.shape
    flat = weight_dequant.view(-1, k)
    out_w_flat, out_s_flat = per_block_cast_to_fp8(flat, block_size=[128, 128], use_ue8m0=True)
    out_w = out_w_flat.view(*batch_dims, n, k)
    out_s = out_s_flat.view(
        *batch_dims,
        ceil(n / weight_block_size[0]),
        ceil(k / weight_block_size[1]),
    )
    return out_w, out_s


def transform_scale_ue8m0(sf: torch.Tensor, mn: int):
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    if sf.shape[-1] == 1:
        aligned_mn = get_tma_aligned_size(sf.shape[-2], sf.element_size())
        if sf.stride(-1) != aligned_mn:
            new_stride = list(sf.stride())
            new_stride[-1] = aligned_mn
            sf = sf.as_strided(sf.shape, tuple(new_stride))
    return sf


__all__ = [
    "HfWeightSource",
    "VimeRayWeightSyncClient",
    "create_nccl_trainer",
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
]
