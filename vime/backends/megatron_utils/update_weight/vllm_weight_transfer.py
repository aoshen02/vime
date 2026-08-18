import socket
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

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
            packed_num_buffers=1,
        ),
        client=client,
        source=source,
    )


__all__ = ["HfWeightSource", "VimeRayWeightSyncClient", "create_nccl_trainer"]
