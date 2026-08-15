"""
Colocated vLLM weight sync (trainer side)
=========================================

``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC handles
→ ``POST /update_weights`` to vLLM's native ``IPCWeightTransferEngine``.

vLLM handles UUID routing + device_index remapping + layerwise reload
internally; no worker extension or monkey-patch is needed.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/
"""

from __future__ import annotations

import os
from argparse import Namespace
<<<<<<< ours (vime current)
from collections.abc import Callable, Iterable, Mapping, Sequence
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
from collections.abc import Callable, Mapping, Sequence
=======
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

<<<<<<< ours (vime current)
from vime.utils.distributed_utils import get_gloo_group
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
from slime.utils.distributed_utils import get_gloo_group
=======
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.types import ParamInfo
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

<<<<<<< ours (vime current)
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
from ..vllm import FlattenedTensorBucket, MultiprocessingSerializer
=======
from ..megatron_to_hf import convert_to_hf
from ..vllm import FlattenedTensorBucket, MultiprocessingSerializer
from .expert_routing import configure_expert_routing
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

_MAX_COLOCATED_UPDATES_INFLIGHT = 4


def _build_packed_ipc_update_info(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, Any], torch.Tensor]:
    from torch.multiprocessing.reductions import reduce_tensor

    names, dtype_names, shapes, tensor_sizes, byte_tensors = [], [], [], [], []
    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        byte_tensor = tensor.detach().contiguous().view(torch.uint8).flatten()
        tensor_sizes.append(byte_tensor.numel())
        byte_tensors.append(byte_tensor)
    if not byte_tensors:
        raise ValueError("cannot build an empty packed IPC update")

    packed_tensor = torch.cat(byte_tensors)
    _, ipc_args = reduce_tensor(packed_tensor)
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return (
        {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "tensor_sizes": tensor_sizes,
            "ipc_handles": {gpu_uuid: ipc_args},
        },
        packed_tensor,
    )


def _serialize_ipc_update_info(info: dict[str, Any]) -> str:
    """Pickle IPC handles for cross-rank gather (Gloo ``all_gather_object`` cannot carry them)."""
    import base64

    import cloudpickle

    return base64.b64encode(cloudpickle.dumps(info)).decode("ascii")


def _deserialize_ipc_update_info(payload: str) -> dict[str, Any]:
    import base64

    import cloudpickle

    return cloudpickle.loads(base64.b64decode(payload.encode("ascii")))


def _merge_ipc_update_infos(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge the per-rank handles for one packed IPC update."""
    if not infos:
        raise ValueError("no IPC update_info payloads to merge")

    metadata_keys = ("names", "dtype_names", "shapes", "tensor_sizes")
    base = infos[0]
    if "tensor_sizes" not in base or any(
        "tensor_sizes" not in info or any(info[key] != base[key] for key in metadata_keys) for info in infos[1:]
    ):
        raise ValueError("packed IPC metadata must match across all ranks in a slot")
    handles = {}
    for info in infos:
        handles.update(info["ipc_handles"])
    return {**base, "ipc_handles": handles}


def _build_flattened_tensor_data(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    if not named_tensors:
        return {
            "flattened_tensor": torch.empty(0, dtype=torch.uint8, device=torch.cuda.current_device()),
            "metadata": [],
        }

    # Do not reuse the IPC-facing flattened tensor. VLLM returns from the
    # HTTP/Ray request after enqueueing GPU copies into model weights, but it
    # does not guarantee a CUDA-device sync before the response. Reusing and
    # overwriting the same producer buffer immediately after ray.get can race
    # with the consumer-side copy and corrupt weights.
    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    return {
        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
        "metadata": flattened_tensor_bucket.get_metadata(),
    }


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → all_gather_object(Gloo CPU, over the engine
    slot ranks) → Ray IPC to engine.  Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.rank = dist.get_rank()
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )
        param_info_buckets = getattr(self._hf_weight_iterator, "megatron_local_param_info_buckets", None)
        self._full_param_info_buckets = (
            tuple(tuple(bucket) for bucket in param_info_buckets) if param_info_buckets is not None else None
        )
        self._non_expert_param_info_buckets: list[list[ParamInfo]] | None = None

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
<<<<<<< ours (vime current)
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # vLLM IPC handle payloads may use cloudpickle on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
=======
        self._expert_transfer_plan = []
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "vime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]
        colocate_parallel_configs = (
            engine_parallel_configs[:colocate_engine_nums] if engine_parallel_configs is not None else None
        )

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if self.rank in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= self.rank < end:
                self._ipc_engine = engine

<<<<<<< ours (vime current)
        # vLLM #39212: one-time IPC transfer-engine init on each colocated engine.
        if dist.get_rank() == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self.rollout_engines])
            self._ipc_initialized = True

||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
=======
        self._non_expert_param_info_buckets, self._expert_transfer_plan = configure_expert_routing(
            args=self.args,
            full_param_info_buckets=self._full_param_info_buckets,
            get_local_weight_names=self.weights_getter,
            engine_gpu_counts=colocate_gpu_counts,
            engine_gpu_offsets=colocate_gpu_offsets,
            engine_parallel_configs=colocate_parallel_configs,
            use_distribute=self.use_distribute,
        )

>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

<<<<<<< ours (vime current)
    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
=======
    def _prepare_expert_weight_batch(
        self,
        transfers: Sequence[Any],
        megatron_local_weights: Mapping[str, torch.Tensor],
        staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        local_params = []
        p2p_ops = []
        buffer_offsets: dict[tuple[torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        for transfer in transfers:
            for expert_param in transfer.params:
                info = expert_param.info
                if self.rank != transfer.source_rank and self.rank not in transfer.target_ranks:
                    continue
                key = (info.dtype, tuple(info.shape))
                pool = staging_buffers.setdefault(key, [])
                offset = buffer_offsets[key]
                buffer_offsets[key] = offset + 1
                if offset == len(pool):
                    pool.append(torch.empty(info.shape, dtype=info.dtype, device="cuda"))
                tensor = pool[offset]
                if self.rank == transfer.source_rank:
                    source = megatron_local_weights[info.name]
                    if source.shape != info.shape or source.dtype != info.dtype:
                        raise ValueError(f"expert metadata changed for {info.name}")
                    tensor.copy_(source, non_blocking=True)
                    p2p_ops.extend(
                        dist.P2POp(dist.isend, tensor, target_rank)
                        for target_rank in transfer.target_ranks
                        if target_rank != self.rank
                    )
                    if self.rank in expert_param.target_ranks:
                        local_params.append((expert_param, tensor))
                else:
                    p2p_ops.append(dist.P2POp(dist.irecv, tensor, transfer.source_rank))
                    local_params.append((expert_param, tensor))

        for request in dist.batch_isend_irecv(p2p_ops) if p2p_ops else ():
            request.wait()

        hf_named_tensors = []
        for expert_param, tensor in local_params:
            hf_named_tensors.extend(
                convert_to_hf(
                    self.args,
                    self.model_name,
                    expert_param.info.name,
                    tensor,
                    self.quantization_config,
                )
            )
        return hf_named_tensors

    def _update_expert_weights(
        self,
        megatron_local_weights: Mapping[str, torch.Tensor],
    ) -> None:
        dist.barrier(group=get_gloo_group())
        # Initialize WORLD on all ranks before subset batched P2P.
        dist.barrier()
        # Reuse staging across layers instead of fragmenting the CUDA allocator.
        staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]] = {}
        for transfer_group in tqdm(
            self._expert_transfer_plan,
            disable=self.rank != 0,
            desc="Update expert weights",
        ):
            for transfer_batch in transfer_group:
                hf_named_tensors = self._prepare_expert_weight_batch(
                    transfer_batch,
                    megatron_local_weights,
                    staging_buffers,
                )
                refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
                ray.get(refs)
                dist.barrier(group=get_gloo_group())
                torch.cuda.synchronize()
                del refs, long_lived_tensors, hf_named_tensors
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
        del staging_buffers
        torch.cuda.empty_cache()

>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        if self.rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        # vLLM #39212: enter weight-update mode on each slot leader.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
        dist.barrier(group=get_gloo_group())

<<<<<<< ours (vime current)
        megatron_local_weights = self.weights_getter()
        self._send_weight_chunks(megatron_local_weights)
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (vllm engines)
            # have already closed their IPC handles.
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()
=======
        param_info_buckets = (
            self._non_expert_param_info_buckets if self._expert_transfer_plan else self._full_param_info_buckets
        )
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
            megatron_local_weights,
            param_info_buckets=param_info_buckets,
        ):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (vllm engines)
            # have already closed their IPC handles.
            del refs, long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        if self._expert_transfer_plan:
            self._update_expert_weights(megatron_local_weights)
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

        del megatron_local_weights
        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        # vLLM #39212: exit weight-update mode.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        if (
            not self.use_distribute
            and self.args.enable_mtp_training
            and (self.args.vllm_speculative_config or {}).get("method") == "mtp"
        ):
            if self._ipc_engine is not None and rank == self._ipc_gather_src:
                ray.get(self._ipc_engine.start_draft_weight_update.remote())
            dist.barrier(group=get_gloo_group())

            self._send_weight_chunks(megatron_local_weights)

            dist.barrier(group=get_gloo_group())
            torch.cuda.ipc_collect()
            if self._ipc_engine is not None and rank == self._ipc_gather_src:
                ray.get(self._ipc_engine.finish_weight_update.remote())
            dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if self.rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_weight_chunks(self, megatron_local_weights) -> None:
        max_inflight = 1 if self.use_distribute else _MAX_COLOCATED_UPDATES_INFLIGHT
        pending = []
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, weight_refs = self._send_hf_params(hf_named_tensors)
            pending.append((refs, weight_refs))
            if len(pending) >= max_inflight:
                self._drain_ipc_updates(pending)
        self._drain_ipc_updates(pending)

    def _drain_ipc_updates(self, pending) -> None:
        if not pending:
            return
        ray.get([ref for refs, _ in pending for ref in refs])
        if self._ipc_gather_group is not None:
            dist.barrier(group=self._ipc_gather_group)
        pending.clear()
        torch.cuda.ipc_collect()

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # all_gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

<<<<<<< ours (vime current)
    local_info, weight_ref = _build_packed_ipc_update_info(hf_named_tensors)

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        ref = ipc_engine.update_weights_from_tensor.remote(**local_info, weight_version=str(weight_version))
        return [ref], weight_ref

    payload = _serialize_ipc_update_info(local_info)

    gathered_payloads = [None] * slot_size if dist.get_rank() == ipc_gather_src else None
    dist.gather_object(payload, object_gather_list=gathered_payloads, dst=ipc_gather_src, group=ipc_gather_group)
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors} if hf_named_tensors else {}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )
=======
    long_live_tensors = []

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors} if hf_named_tensors else {}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_data = _build_flattened_tensor_data(named_tensors)
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        merged = _merge_ipc_update_infos(slot_infos)
        refs.append(ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version)))

    return refs, weight_ref
