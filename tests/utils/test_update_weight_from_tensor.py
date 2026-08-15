"""CPU unit tests for colocated vLLM IPC weight sync (UpdateWeightFromTensor)."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import torch

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_tensor"

NUM_GPUS = 0


def _install_stubs():
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()

    import torch.distributed as _dist

    dist_stub = MagicMock()
    dist_stub.get_rank.return_value = 0
    dist_stub.get_world_size.return_value = 1
    dist_stub.get_process_group_ranks.return_value = [0, 1]
    dist_stub.barrier = MagicMock()
    dist_stub.all_gather_object = MagicMock()
    _dist.get_rank = dist_stub.get_rank
    _dist.get_world_size = dist_stub.get_world_size
    _dist.get_process_group_ranks = dist_stub.get_process_group_ranks
    _dist.barrier = dist_stub.barrier
    _dist.all_gather_object = dist_stub.all_gather_object

    hf_iter_stub = MagicMock()
    hf_iter_stub.get_hf_weight_chunks.return_value = iter([])

    hf_base_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base")
    hf_base_mod.HfWeightIteratorBase = MagicMock()
    hf_base_mod.HfWeightIteratorBase.create.return_value = hf_iter_stub

    upw_dist_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    upw_dist_mod.connect_rollout_engines_from_distributed = MagicMock(return_value="groups")
    upw_dist_mod.disconnect_rollout_engines_from_distributed = MagicMock()
    upw_dist_mod.post_process_weights = MagicMock()
    upw_dist_mod.update_weights_from_distributed = MagicMock(return_value=[])

    for key, mod in [
        ("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base", hf_base_mod),
        ("vime.backends.megatron_utils.update_weight.update_weight_from_distributed", upw_dist_mod),
    ]:
        sys.modules.setdefault(key, mod)

    return hf_iter_stub, upw_dist_mod


# Placeholder iterator stored on freshly-built instances; every test that drives a real
# update overrides obj._hf_weight_iterator with its own MagicMock, so this only needs to be
# a non-None object.
_HF_ITER_STUB = MagicMock()
_HF_ITER_STUB.get_hf_weight_chunks.return_value = iter([])

# Modules stubbed by _install_stubs(), plus torch.distributed attributes it overwrites.
# These are installed ONLY for this module's tests (inside the fixture) and restored on
# teardown. Installing at import time leaked the stubs into sibling modules' COLLECTION (and
# left MagicMocks on torch.distributed), one source of the cross-test order-pollution.
_STUBBED_MODULES = (
    "megatron",
    "megatron.core",
    "ray",
    "ray.actor",
    "vime.utils.distributed_utils",
    "vime.backends.megatron_utils.update_weight.hf_weight_iterator_base",
    "vime.backends.megatron_utils.update_weight.update_weight_from_distributed",
)
_DIST_ATTRS = ("get_rank", "get_world_size", "get_process_group_ranks", "barrier", "all_gather_object")


@pytest.fixture(scope="module")
def upw_vllm():
    import torch.distributed as _dist

    saved_mods = _unit_stubs.save_sys_modules((*_STUBBED_MODULES, MODULE_PATH))
    saved_dist = {a: getattr(_dist, a, None) for a in _DIST_ATTRS}
    # Pop first so _install_stubs()'s setdefault() actually installs stubs (hermetic).
    for k in _STUBBED_MODULES:
        sys.modules.pop(k, None)
    _install_stubs()
    sys.modules.pop(MODULE_PATH, None)
    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved_mods)
        for a, original in saved_dist.items():
            if original is not None:
                setattr(_dist, a, original)


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self):
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return "ref"


@dataclass
class RecordingVLLMEngine:
    release_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    resume_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    init_weight_transfer_engine: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_draft_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    finish_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights_from_tensor: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    pause_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    flush_cache: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    continue_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)


def _default_args(**kwargs) -> Namespace:
    base = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus_per_engine=2,
        update_weight_buffer_size=1 << 30,
        enable_mtp_training=False,
        vllm_speculative_config=None,
    )
    base.update(kwargs)
    return Namespace(**base)


def _make_instance(upw_vllm, args=None):
    obj = object.__new__(upw_vllm.UpdateWeightFromTensor)
    obj.args = args or _default_args()
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test"
    obj.quantization_config = None
    obj.weight_version = 0
    obj.rank = 0
    obj._hf_weight_iterator = _HF_ITER_STUB
    obj._full_param_info_buckets = None
    obj._non_expert_param_info_buckets = None
    obj._expert_transfer_plan = []
    obj.rollout_engines = []
    obj._all_rollout_engines = []
    obj.distributed_rollout_engines = []
    obj.use_distribute = False
    obj._ipc_engine = None
    obj._ipc_gather_group = None
    obj._ipc_gather_src = None
    obj._model_update_groups = None
    obj._is_distributed_src_rank = False
    obj._group_name = "vime"
    obj._ipc_initialized_engine_ids = set()
    obj._native_ipc_trainer = None
    return obj


def _bind_single_slot(obj, engine, *, src=0):
    """Bind ``obj`` to one colocated engine forming a slot whose leader rank is ``src``."""
    obj.rollout_engines = [engine]
    obj._all_rollout_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_gather_group = "slot_group"
    obj._ipc_gather_src = src


def _chunks(n=1):
    return [[(f"p.{i}", torch.zeros(2, 2)) for i in range(2)] for _ in range(n)]


def _run_update(obj, *, chunks=None, rank=0, slot_size=1) -> dict:
    """Drive ``update_weights`` with controlled rank / slot size.

    ``slot_size`` is what ``dist.get_world_size(self._ipc_gather_group)`` returns,
    so slot_size==1 takes the direct IPC path and slot_size>1 the gather path.
    Returns counters for barriers and CUDA cache cleanup calls.
    """
    chunks = chunks or _chunks(1)
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.side_effect = lambda *args, **kwargs: iter(chunks)

    counters = {"barrier": 0, "ipc_collect": 0, "empty_cache": 0}

    def counting_barrier(*args, **kwargs):
        counters["barrier"] += 1

    def counting_ipc_collect(*args, **kwargs):
        counters["ipc_collect"] += 1

    def counting_empty_cache(*args, **kwargs):
        counters["empty_cache"] += 1

    with (
        patch("torch.distributed.get_rank", return_value=rank),
        patch("torch.distributed.get_world_size", return_value=slot_size),
        patch("torch.distributed.barrier", side_effect=counting_barrier),
        patch("torch.cuda.ipc_collect", side_effect=counting_ipc_collect),
        patch("torch.cuda.empty_cache", side_effect=counting_empty_cache),
    ):
        obj.update_weights()
    return counters


@pytest.mark.unit
def test_colocated_lifecycle_uses_pause_flush_and_weight_transfer_apis(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "tensor_sizes": [8],
        "ipc_handles": {"u": ("f", ())},
    }
    with patch(f"{MODULE_PATH}._build_packed_ipc_update_info", return_value=(dummy_info, [])):
        counters = _run_update(obj, chunks=_chunks(2))

    # Colocate quiesce: pause_generation + flush_cache only, no /sleep round-trip;
    # continue_generation resumes. No release/resume_memory_occupation.
    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1
    assert len(engine.release_memory_occupation.calls) == 0
    assert len(engine.resume_memory_occupation.calls) == 0
    # vLLM #39212: init runs in connect_rollout_engines, not update_weights.
    assert len(engine.init_weight_transfer_engine.calls) == 0
    assert len(engine.start_weight_update.calls) == 1
    assert engine.start_weight_update.calls[0].kwargs == {}
    assert len(engine.finish_weight_update.calls) == 1
    assert engine.finish_weight_update.calls[0].kwargs == {"weight_version": "1"}
    assert len(engine.continue_generation.calls) == 1
    # Both chunks are kept alive until the bounded in-flight batch drains.
    assert counters["ipc_collect"] == 2
    assert counters["empty_cache"] == 2
    # lifecycle barriers (no per-chunk barrier).
    assert counters["barrier"] >= 4


@pytest.mark.unit
def test_colocated_mtp_updates_target_then_draft_from_fresh_weight_stream(upw_vllm):
    obj = _make_instance(
        upw_vllm,
        args=_default_args(
            enable_mtp_training=True,
            vllm_speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        ),
    )
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "tensor_sizes": [8],
        "ipc_handles": {},
    }
    with patch(f"{MODULE_PATH}._build_packed_ipc_update_info", return_value=(dummy_info, [])):
        counters = _run_update(obj, chunks=_chunks(2))

    assert len(engine.start_weight_update.calls) == 1
    assert len(engine.start_draft_weight_update.calls) == 1
    assert len(engine.finish_weight_update.calls) == 2
    assert len(engine.update_weights_from_tensor.calls) == 4
    assert obj._hf_weight_iterator.get_hf_weight_chunks.call_count == 2
    assert counters["empty_cache"] == 4


@pytest.mark.unit
def test_mixed_colocated_and_remote_engines_share_update_lifecycle(upw_vllm):
    obj = _make_instance(upw_vllm)
    colocated_engine = RecordingVLLMEngine()
    remote_engine = RecordingVLLMEngine()
    _bind_single_slot(obj, colocated_engine, src=0)
    obj._all_rollout_engines = [colocated_engine, remote_engine]
    obj.distributed_rollout_engines = [remote_engine]
    obj.use_distribute = True
    obj._send_weight_chunks = MagicMock()

    _run_update(obj)

    for engine in (colocated_engine, remote_engine):
        assert len(engine.pause_generation.calls) == 1
        assert len(engine.flush_cache.calls) == 1
        assert len(engine.start_weight_update.calls) == 1
        assert len(engine.finish_weight_update.calls) == 1
        assert engine.finish_weight_update.calls[0].kwargs == {"weight_version": "1"}
        assert len(engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_failed_tensor_transfer_does_not_commit_weight_version(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)
    obj._send_weight_chunks = MagicMock(side_effect=RuntimeError("transfer failed"))

    with pytest.raises(RuntimeError, match="transfer failed"):
        _run_update(obj)

    assert obj.weight_version == 0
    assert len(engine.finish_weight_update.calls) == 0
    assert len(engine.continue_generation.calls) == 0


@pytest.mark.unit
def test_send_via_ipc_dispatches_update_weights_from_tensor_with_version(upw_vllm):
    """slot_size=1: every HF chunk fires
    ``engine.update_weights_from_tensor.remote(**fields, weight_version=...)`` —
    same name and parameterized fields; the version commits at
    ``finish_weight_update`` after every chunk succeeds."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "tensor_sizes": [8],
        "ipc_handles": {"u": ("f", ())},
    }
    with patch(
        f"{MODULE_PATH}._build_packed_ipc_update_info",
        return_value=(dummy_info, []),
    ):
        _run_update(obj, chunks=_chunks(2))

    # 2 HF chunks → 2 IPC RPCs
    assert len(engine.update_weights_from_tensor.calls) == 2
    kwargs = engine.update_weights_from_tensor.calls[0].kwargs
    # fields are passed as explicit kwargs (** expanded from local_info)
    assert kwargs["names"] == dummy_info["names"]
    assert kwargs["dtype_names"] == dummy_info["dtype_names"]
    assert kwargs["shapes"] == dummy_info["shapes"]
    assert kwargs["ipc_handles"] is dummy_info["ipc_handles"]
    # weight_version is the trainer's post-increment version (0 + 1 = 1) as a str
    assert kwargs["weight_version"] == "1"
    assert len(engine.finish_weight_update.calls) == 1
    assert engine.finish_weight_update.calls[0].kwargs == {"weight_version": "1"}


@pytest.mark.unit
def test_send_via_ipc_dispatches_update_weights_from_tensor_coordinator_multi_gpu(upw_vllm):
    """slot_size > 1: the slot leader (rank == _ipc_gather_src) gathers payloads from
    all slot ranks, merges them, and fires a single update_weights_from_tensor RPC per chunk."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info_0 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "tensor_sizes": [8],
        "ipc_handles": {"uuid-gpu0": ("f", ())},
    }
    dummy_info_1 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "tensor_sizes": [8],
        "ipc_handles": {"uuid-gpu1": ("f", ())},
    }

    def fake_gather_object(payload, object_gather_list=None, dst=None, group=None):
        del payload, dst, group
        gathered_payloads = object_gather_list
        gathered_payloads[0] = "payload0"
        gathered_payloads[1] = "payload1"

    with (
        patch(
            f"{MODULE_PATH}._build_packed_ipc_update_info",
            return_value=(dummy_info_0, []),
        ),
        patch(f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload0"),
        patch(f"{MODULE_PATH}._deserialize_ipc_update_info", side_effect=[dummy_info_0, dummy_info_1] * 2),
        patch("torch.distributed.gather_object", side_effect=fake_gather_object),
    ):
        _run_update(obj, chunks=_chunks(2), rank=0, slot_size=2)

    assert len(engine.update_weights_from_tensor.calls) == 2
    kwargs = engine.update_weights_from_tensor.calls[0].kwargs
    assert kwargs["names"] == dummy_info_0["names"]
    assert kwargs["dtype_names"] == dummy_info_0["dtype_names"]
    assert kwargs["shapes"] == dummy_info_0["shapes"]
    assert set(kwargs["ipc_handles"]) == {"uuid-gpu0", "uuid-gpu1"}
    assert kwargs["weight_version"] == "1"


@pytest.mark.unit
def test_colocated_update_waits_in_bounded_batches(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)
    next_ref = iter(range(5))

    def fake_send(_hf_named_tensors):
        index = next(next_ref)
        return [f"update-{index}"], [torch.zeros(1)]

    obj._send_hf_params = fake_send
    update_batches = []

    def record_get(refs):
        if isinstance(refs, list) and refs and all(str(ref).startswith("update-") for ref in refs):
            update_batches.append(refs)

    with (
        patch(f"{MODULE_PATH}._MAX_COLOCATED_UPDATES_INFLIGHT", 2),
        patch(f"{MODULE_PATH}.ray.get", side_effect=record_get),
    ):
        counters = _run_update(obj, chunks=_chunks(5))

    assert [len(batch) for batch in update_batches] == [2, 2, 1]
    assert counters["ipc_collect"] == 4


@pytest.mark.unit
def test_merge_packed_ipc_update_infos_combines_gpu_uuids(upw_vllm):
    base = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "tensor_sizes": [8],
    }
    info0 = {**base, "ipc_handles": {"uuid-gpu0": ("f0", ())}}
    info1 = {**base, "ipc_handles": {"uuid-gpu1": ("f1", ())}}

    merged = upw_vllm._merge_ipc_update_infos([info0, info1])

    assert set(merged["ipc_handles"]) == {"uuid-gpu0", "uuid-gpu1"}


@pytest.mark.unit
def test_merge_packed_ipc_update_infos_rejects_mismatched_metadata(upw_vllm):
    info0 = {
        "names": ["a"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2]],
        "tensor_sizes": [4],
        "ipc_handles": {"uuid-gpu0": ("f0", ())},
    }
    info1 = {**info0, "names": ["b"], "ipc_handles": {"uuid-gpu1": ("f1", ())}}

    with pytest.raises(ValueError, match="packed IPC metadata must match"):
        upw_vllm._merge_ipc_update_infos([info0, info1])


@pytest.mark.unit
def test_group_packed_ipc_update_infos_routes_different_metadata_to_matching_gpu(upw_vllm):
    info0 = {
        "names": ["experts.0.weight"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2]],
        "tensor_sizes": [4],
        "ipc_handles": {"uuid-gpu0": ("f0", ())},
    }
    info1 = {
        **info0,
        "names": ["experts.1.weight"],
        "ipc_handles": {"uuid-gpu1": ("f1", ())},
    }

    assert upw_vllm._group_ipc_update_infos([info0, info1]) == [
        {**info0, "empty_gpu_uuids": ["uuid-gpu1"]},
        {**info1, "empty_gpu_uuids": ["uuid-gpu0"]},
    ]


@pytest.mark.unit
def test_build_packed_ipc_update_info_preserves_metadata_and_bytes(upw_vllm):
    tensors = [("a", torch.tensor([1, 2], dtype=torch.int16)), ("b", torch.tensor([3.0]))]

    with (
        patch("torch.multiprocessing.reductions.reduce_tensor", return_value=(None, ("rebuild", ()))),
        patch("torch.cuda.current_device", return_value=0),
        patch("torch.cuda.get_device_properties", return_value=MagicMock(uuid="uuid-gpu0")),
    ):
        update_info, packed = upw_vllm._build_packed_ipc_update_info(tensors)

    assert update_info["names"] == ["a", "b"]
    assert update_info["tensor_sizes"] == [4, 4]
    assert update_info["ipc_handles"] == {"uuid-gpu0": ("rebuild", ())}
    assert torch.equal(packed, torch.cat([tensor.view(torch.uint8) for _, tensor in tensors]))


@pytest.mark.unit
def test_colocated_source_has_no_nonpacked_path(upw_vllm):
    source = inspect.getsource(upw_vllm)
    assert "vllm_weight_sync_packed" not in source
    assert "_build_ipc_update_info_from_named_tensors" not in source


@pytest.mark.unit
def test_connect_binds_engine_and_slot_leader_per_gpu_slot(upw_vllm):
    """Each rank binds to its slot's engine; the slot leader (== _ipc_gather_src,
    the lowest trainer rank in the engine GPU range) is the start/finish coordinator."""
    engines = [RecordingVLLMEngine() for _ in range(4)]
    for rank, engine_idx, expected_src in [
        (0, 0, 0),
        (1, 0, 0),
        (2, 1, 2),
        (3, 1, 2),
    ]:
        obj = _make_instance(
            upw_vllm,
            args=_default_args(actor_num_gpus_per_node=8, rollout_num_gpus_per_engine=2),
        )
        with (
            patch("torch.distributed.get_rank", return_value=rank),
            patch("megatron.core.mpu.get_tensor_model_parallel_rank", return_value=rank % 2),
            patch("torch.distributed.new_group", return_value="slot_group"),
        ):
            obj.connect_rollout_engines(
                engines,
                rollout_engine_lock=MagicMock(),
                engine_gpu_counts=[2, 2, 2, 2],
                engine_gpu_offsets=[0, 2, 4, 6],
            )
        assert obj._ipc_engine is engines[engine_idx]
        assert obj._ipc_gather_src == expected_src
        is_coordinator = rank == obj._ipc_gather_src
        assert is_coordinator is (rank in (0, 2))
        assert obj.use_distribute is False
        assert obj.distributed_rollout_engines == []
        # vLLM #39212: init_weight_transfer_engine fires once during connect (rank 0 only).
        if rank == 0:
            assert len(engines[0].init_weight_transfer_engine.calls) == 1
            assert engines[0].init_weight_transfer_engine.calls[0].args[0] == {"init_info": {"packed": True}}


@pytest.mark.unit
def test_non_leader_skips_start_finish_and_merged_rpc(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    # slot leader is rank 0; we drive update_weights as rank 1 (non-leader).
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": [], "dtype_names": [], "shapes": [], "tensor_sizes": [], "ipc_handles": {}}
    with (
        patch(
            f"{MODULE_PATH}._build_packed_ipc_update_info",
            return_value=(dummy_info, []),
        ),
        patch(f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload"),
        patch("torch.distributed.gather_object") as gather_obj,
    ):
        _run_update(obj, chunks=_chunks(1), rank=1, slot_size=2)

    gather_obj.assert_called_once()
    # non-leader: no start/finish, and no merged update_weights_from_tensor RPC
    assert len(engine.start_weight_update.calls) == 0
    assert len(engine.finish_weight_update.calls) == 0
    assert len(engine.update_weights_from_tensor.calls) == 0


@pytest.mark.unit
def test_ipc_init_runs_once_in_connect(upw_vllm):
    """init_weight_transfer_engine fires once in connect_rollout_engines (rank 0),
    not in update_weights. Existing engines are not re-initialized after reconnect."""
    engines = [RecordingVLLMEngine() for _ in range(2)]
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=2),
    )
    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0),
        patch("torch.distributed.new_group", return_value="slot_group"),
    ):
        obj.connect_rollout_engines(
            engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert len(obj._ipc_initialized_engine_ids) == 2
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(engines[1].init_weight_transfer_engine.calls) == 1
    assert engines[0].init_weight_transfer_engine.calls[0].args[0] == {"init_info": {"packed": True}}
    assert engines[1].init_weight_transfer_engine.calls[0].args[0] == {"init_info": {"packed": True}}

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0),
        patch("torch.distributed.new_group", return_value="slot_group"),
    ):
        obj.connect_rollout_engines(
            engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(engines[1].init_weight_transfer_engine.calls) == 1

    replacement = RecordingVLLMEngine()
    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0),
        patch("torch.distributed.new_group", return_value="slot_group"),
    ):
        obj.connect_rollout_engines(
            [engines[0], replacement],
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(replacement.init_weight_transfer_engine.calls) == 1
    assert len(obj._ipc_initialized_engine_ids) == 3


@pytest.mark.unit
def test_single_full_slot_uses_vllm_stateful_ipc_trainer(upw_vllm):
    engine = RecordingVLLMEngine()
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=4),
    )
    trainer = MagicMock()
    trainer_factory = MagicMock()
    trainer_factory.trainer_init.return_value = trainer

    class FakeIPCTrainerInitInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    factory_module = types.ModuleType("vllm.distributed.weight_transfer.factory")
    factory_module.WeightTransferTrainerFactory = trainer_factory
    ipc_module = types.ModuleType("vllm.distributed.weight_transfer.ipc_engine")
    ipc_module.IPCTrainerInitInfo = FakeIPCTrainerInitInfo

    with (
        patch.dict(
            sys.modules,
            {
                "vllm.distributed.weight_transfer.factory": factory_module,
                "vllm.distributed.weight_transfer.ipc_engine": ipc_module,
            },
        ),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.new_group", return_value="slot_group"),
    ):
        obj.connect_rollout_engines(
            [engine],
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[4],
            engine_gpu_offsets=[0],
        )

    assert obj._native_ipc_trainer is trainer
    init_info = trainer_factory.trainer_init.call_args.args[0]
    assert init_info.rank == 0
    assert init_info.packed is True
    assert init_info.packed_buffer_size_bytes == 1 << 30
    assert len(engine.init_weight_transfer_engine.calls) == 0

    counters = _run_update(obj)

    trainer.send_weights.assert_called_once_with()
    assert counters["empty_cache"] == 1
    assert len(engine.pause_generation.calls) == 1
    assert len(engine.continue_generation.calls) == 1
    assert len(engine.start_weight_update.calls) == 0
    assert len(engine.update_weights_from_tensor.calls) == 0


@pytest.mark.unit
def test_native_ipc_buffer_grows_for_largest_reconstructed_tensor(upw_vllm):
    args = _default_args(update_weight_buffer_size=512)
    dense_param = types.SimpleNamespace(name="model.embed_tokens.weight", size=400)
    expert_param = types.SimpleNamespace(name="model.layers.0.mlp.experts.0.weight", size=700)

    with (
        patch("megatron.core.mpu.get_tensor_model_parallel_world_size", return_value=2),
        patch("megatron.core.mpu.get_expert_tensor_parallel_world_size", return_value=1),
    ):
        buffer_size = upw_vllm._native_ipc_buffer_size(args, [[dense_param, expert_param]])

    assert buffer_size == 800


@pytest.mark.unit
def test_expert_routing_keeps_rank_local_fallback(upw_vllm):
    engine = RecordingVLLMEngine()
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=4),
    )
    expert_plan = [("batch",)]

    with (
        patch(
            f"{MODULE_PATH}.configure_expert_routing",
            return_value=([[]], expert_plan),
        ) as configure,
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.new_group", return_value="slot_group"),
    ):
        obj.connect_rollout_engines(
            [engine],
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[4],
            engine_gpu_offsets=[0],
            engine_parallel_configs=[
                {
                    "tp_size": 1,
                    "pp_size": 1,
                    "pcp_size": 1,
                    "dp_size": 4,
                    "enable_expert_parallel": True,
                    "ep_size": 4,
                }
            ],
        )

    assert obj._native_ipc_trainer is None
    assert obj._expert_transfer_plan is expert_plan
    assert len(engine.init_weight_transfer_engine.calls) == 1
    assert configure.call_args.kwargs["engine_parallel_configs"] == [
        {
            "tp_size": 1,
            "pp_size": 1,
            "pcp_size": 1,
            "dp_size": 4,
            "enable_expert_parallel": True,
            "ep_size": 4,
        }
    ]


@pytest.mark.unit
def test_multi_node_full_slot_falls_back_from_cuda_ipc_trainer(upw_vllm):
    engine = RecordingVLLMEngine()
    obj = _make_instance(
        upw_vllm,
        args=_default_args(
            actor_num_nodes=2,
            actor_num_gpus_per_node=2,
            rollout_num_gpus_per_engine=4,
        ),
    )

    with (
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.new_group", return_value="slot_group"),
    ):
        obj.connect_rollout_engines(
            [engine],
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[4],
            engine_gpu_offsets=[0],
        )

    assert obj._native_ipc_trainer is None
    assert len(engine.init_weight_transfer_engine.calls) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
