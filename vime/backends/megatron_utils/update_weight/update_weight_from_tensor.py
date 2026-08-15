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
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from vime.utils.distributed_utils import get_gloo_group
from vime.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from .expert_routing import configure_expert_routing
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

_MAX_COLOCATED_UPDATES_INFLIGHT = 4


def _rollout_engine_identity(engine: ActorHandle) -> str:
    actor_id = getattr(engine, "_actor_id", None)
    return actor_id.hex() if actor_id is not None else str(id(engine))


def _native_ipc_buffer_size(args: Namespace, param_info_buckets: Sequence[Sequence[ParamInfo]] | None) -> int:
    buffer_size = args.update_weight_buffer_size
    if not param_info_buckets:
        return buffer_size

    tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
    expert_tensor_parallel_size = mpu.get_expert_tensor_parallel_world_size()
    for bucket in param_info_buckets:
        for info in bucket:
            parallel_size = expert_tensor_parallel_size if ".experts." in info.name else tensor_parallel_size
            buffer_size = max(buffer_size, info.size * parallel_size)
    return buffer_size


class _HfWeightSource:
    def __init__(self, iterator, weights_getter) -> None:
        self._iterator = iterator
        self._weights_getter = weights_getter

    def metadata(self):
        from vllm.distributed.weight_transfer.base import ParamMeta

        return [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self]

    def __iter__(self):
        local_weights = self._weights_getter()
        for chunk in self._iterator.get_hf_weight_chunks(local_weights):
            yield from chunk


class _RayVLLMWeightSyncClient:
    def __init__(self, engine, version_getter) -> None:
        self._engine = engine
        self._version_getter = version_getter

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        ray.get(self._engine.init_weight_transfer_engine.remote({"init_info": init_info}))

    def start_weight_update(self) -> None:
        ray.get(self._engine.start_weight_update.remote())

    def update_weights(self, update_info: dict[str, Any]) -> None:
        ray.get(self._engine.update_weights.remote(update_info))

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        version = str(self._version_getter()) if weight_version is None else str(weight_version)
        ray.get(self._engine.finish_weight_update.remote(weight_version=version))


def _build_packed_ipc_update_info(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, Any], torch.Tensor | None]:
    names, dtype_names, shapes, tensor_sizes, byte_tensors = [], [], [], [], []
    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        byte_tensor = tensor.detach().contiguous().view(torch.uint8).flatten()
        tensor_sizes.append(byte_tensor.numel())
        byte_tensors.append(byte_tensor)
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if not byte_tensors:
        return (
            {
                "names": [],
                "dtype_names": [],
                "shapes": [],
                "tensor_sizes": [],
                "ipc_handles": {},
                "empty_gpu_uuids": [gpu_uuid],
            },
            None,
        )

    from torch.multiprocessing.reductions import reduce_tensor

    packed_tensor = torch.cat(byte_tensors)
    _, ipc_args = reduce_tensor(packed_tensor)
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
    nonempty_infos = [info for info in infos if info["names"]]
    if not nonempty_infos:
        return {}
    base = nonempty_infos[0]
    if "tensor_sizes" not in base or any(
        "tensor_sizes" not in info or any(info[key] != base[key] for key in metadata_keys)
        for info in nonempty_infos[1:]
    ):
        raise ValueError("packed IPC metadata must match across all ranks in a slot")
    handles = {}
    empty_gpu_uuids = []
    for info in infos:
        handles.update(info["ipc_handles"])
        empty_gpu_uuids.extend(info.get("empty_gpu_uuids", []))
    merged = {**base, "ipc_handles": handles}
    if empty_gpu_uuids:
        merged["empty_gpu_uuids"] = empty_gpu_uuids
    return merged


def _group_ipc_update_infos(infos: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group per-rank IPC payloads by metadata for rank-local expert updates."""
    if not infos:
        raise ValueError("no IPC update_info payloads to group")

    metadata_keys = ("names", "dtype_names", "shapes", "tensor_sizes")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    all_gpu_uuids = set()
    for info in infos:
        all_gpu_uuids.update(info["ipc_handles"])
        all_gpu_uuids.update(info.get("empty_gpu_uuids", []))
        if not info["names"]:
            continue
        key = tuple(
            tuple(tuple(value) if isinstance(value, list) else value for value in info[field])
            for field in metadata_keys
        )
        groups[key].append(info)

    merged_groups = []
    for group in groups.values():
        merged = _merge_ipc_update_infos(group)
        empty_gpu_uuids = sorted(all_gpu_uuids - set(merged["ipc_handles"]))
        if empty_gpu_uuids:
            merged["empty_gpu_uuids"] = empty_gpu_uuids
        merged_groups.append(merged)
    return merged_groups


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
        self._all_rollout_engines = []
        self.distributed_rollout_engines = []
        self._expert_transfer_plan = []
        self._native_ipc_trainer = None
        self._ipc_initialized_engine_ids: set[str] = set()
        # vLLM IPC handle payloads may use cloudpickle on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

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
        self._all_rollout_engines = list(rollout_engines)
        self.rollout_engines = rollout_engines
        self.distributed_rollout_engines = []

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

        self._non_expert_param_info_buckets, self._expert_transfer_plan = configure_expert_routing(
            args=self.args,
            full_param_info_buckets=self._full_param_info_buckets,
            get_local_weight_names=self.weights_getter,
            engine_gpu_counts=colocate_gpu_counts,
            engine_gpu_offsets=colocate_gpu_offsets,
            engine_parallel_configs=colocate_parallel_configs,
            use_distribute=self.use_distribute,
        )

        self._native_ipc_trainer = None
        native_ipc_eligible = (
            not self.use_distribute
            and self.args.actor_num_nodes == 1
            and len(self.rollout_engines) == 1
            and colocate_gpu_offsets == [0]
            and colocate_gpu_counts == [dist.get_world_size()]
            and not getattr(self.args, "enable_mtp_training", False)
            and not self._expert_transfer_plan
        )
        if native_ipc_eligible:
            from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
            from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

            self._native_ipc_trainer = WeightTransferTrainerFactory.trainer_init(
                IPCTrainerInitInfo(
                    rank=dist.get_rank(),
                    packed=True,
                    packed_buffer_size_bytes=_native_ipc_buffer_size(self.args, self._full_param_info_buckets),
                ),
                client=_RayVLLMWeightSyncClient(
                    self.rollout_engines[0],
                    lambda: self.weight_version,
                ),
                source=_HfWeightSource(self._hf_weight_iterator, self.weights_getter),
            )

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        if self._native_ipc_trainer is None and dist.get_rank() == 0 and self.rollout_engines:
            engines_to_initialize = [
                engine
                for engine in self.rollout_engines
                if _rollout_engine_identity(engine) not in self._ipc_initialized_engine_ids
            ]
            ray.get(
                [
                    engine.init_weight_transfer_engine.remote({"init_info": {"packed": True}})
                    for engine in engines_to_initialize
                ]
            )
            self._ipc_initialized_engine_ids.update(map(_rollout_engine_identity, engines_to_initialize))

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

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
        dist.barrier()
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

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        previous_version = self.weight_version
        self.weight_version += 1
        try:
            rank = dist.get_rank()
            if rank == 0:
                ray.get([engine.pause_generation.remote() for engine in self._all_rollout_engines])
                ray.get([engine.flush_cache.remote() for engine in self._all_rollout_engines])
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    post_process_weights(
                        restore_weights_before_load=True,
                        post_process_quantization=False,
                        rollout_engines=self._all_rollout_engines,
                    )
            dist.barrier(group=get_gloo_group())

            if self._native_ipc_trainer is not None:
                self._native_ipc_trainer.send_weights()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                if rank == 0:
                    if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                        post_process_weights(
                            restore_weights_before_load=False,
                            post_process_quantization=True,
                            rollout_engines=self._all_rollout_engines,
                        )
                    ray.get([engine.continue_generation.remote() for engine in self._all_rollout_engines])
                dist.barrier(group=get_gloo_group())
                return

            self._start_weight_update(draft=False)

            megatron_local_weights = self.weights_getter()
            self._send_weight_chunks(megatron_local_weights)

            dist.barrier(group=get_gloo_group())
            # After the barrier all engines have returned, so every rank's last-chunk
            # IPC handles are now released by the consumers.  Clean them up.
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            self._finish_weight_update()

            if self.args.enable_mtp_training and (self.args.vllm_speculative_config or {}).get("method") == "mtp":
                self._start_weight_update(draft=True)

                self._send_weight_chunks(megatron_local_weights)

                dist.barrier(group=get_gloo_group())
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
                self._finish_weight_update()

            # int4/fp4 post_process
            if rank == 0:
                if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                    post_process_weights(
                        restore_weights_before_load=False,
                        post_process_quantization=True,
                        rollout_engines=self._all_rollout_engines,
                    )
                ray.get([engine.continue_generation.remote() for engine in self._all_rollout_engines])
            dist.barrier(group=get_gloo_group())
        except Exception:
            self.weight_version = previous_version
            raise

    def _start_weight_update(self, *, draft: bool) -> None:
        rank = dist.get_rank()
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            method = self._ipc_engine.start_draft_weight_update if draft else self._ipc_engine.start_weight_update
            ray.get(method.remote())
        if rank == 0 and self.distributed_rollout_engines:
            if draft:
                refs = [engine.start_draft_weight_update.remote() for engine in self.distributed_rollout_engines]
            else:
                refs = [engine.start_weight_update.remote() for engine in self.distributed_rollout_engines]
            ray.get(refs)
        dist.barrier(group=get_gloo_group())

    def _finish_weight_update(self) -> None:
        rank = dist.get_rank()
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote(weight_version=str(self.weight_version)))
        if rank == 0 and self.distributed_rollout_engines:
            ray.get(
                [
                    engine.finish_weight_update.remote(weight_version=str(self.weight_version))
                    for engine in self.distributed_rollout_engines
                ]
            )
        dist.barrier(group=get_gloo_group())

    def _send_weight_chunks(self, megatron_local_weights) -> None:
        max_inflight = 1 if self.use_distribute else _MAX_COLOCATED_UPDATES_INFLIGHT
        pending = []
        param_info_buckets = (
            self._non_expert_param_info_buckets if self._expert_transfer_plan else self._full_param_info_buckets
        )
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
            megatron_local_weights,
            param_info_buckets=param_info_buckets,
        ):
            refs, weight_refs = self._send_hf_params(hf_named_tensors)
            pending.append((refs, weight_refs))
            if len(pending) >= max_inflight:
                self._drain_ipc_updates(pending)
        self._drain_ipc_updates(pending)
        if self._expert_transfer_plan:
            self._update_expert_weights(megatron_local_weights)

    def _drain_ipc_updates(self, pending) -> None:
        if not pending:
            return
        ray.get([ref for refs, _ in pending for ref in refs])
        if self._ipc_gather_group is not None:
            dist.barrier(group=self._ipc_gather_group)
        pending.clear()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

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

    local_info, weight_ref = _build_packed_ipc_update_info(hf_named_tensors)

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        if not local_info["names"]:
            return [], weight_ref
        ref = ipc_engine.update_weights_from_tensor.remote(**local_info, weight_version=str(weight_version))
        return [ref], weight_ref

    payload = _serialize_ipc_update_info(local_info)

    gathered_payloads = [None] * slot_size if dist.get_rank() == ipc_gather_src else None
    dist.gather_object(payload, object_gather_list=gathered_payloads, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        for merged in _group_ipc_update_infos(slot_infos):
            refs.append(ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version)))

    return refs, weight_ref
