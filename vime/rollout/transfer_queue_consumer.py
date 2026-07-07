"""TransferQueue readers for vLLM rollout trajectory artifacts."""

from __future__ import annotations

import importlib
import pickle
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import quote

import torch

from vime.rollout.trajectory_artifact import TrajectoryArtifactV1Alpha1

REQUIRED_TRAJECTORY_FIELDS = [
    "prompt_token_ids",
    "response_token_ids",
    "response_logprobs",
]


def _install_numpy2_pickle_compat() -> None:
    """Allow NumPy 1.x consumers to unpickle payloads written by NumPy 2.x."""
    numpy = importlib.import_module("numpy")
    if int(numpy.__version__.split(".", 1)[0]) >= 2:
        return
    core = importlib.import_module("numpy.core")
    sys.modules["numpy._core"] = core
    for module_name in ("multiarray", "numeric"):
        module = importlib.import_module(f"numpy.core.{module_name}")
        sys.modules[f"numpy._core.{module_name}"] = module


def _select_sample(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor) and value.is_nested:
        return value.unbind()[index].unsqueeze(0)
    return value[index : index + 1]


def build_trajectory_partition_id(
    run_id: str,
    policy_version: str | int,
    partition_prefix: str = "rollout",
) -> str:
    run = quote(str(run_id), safe="-_.")
    policy = quote(str(policy_version), safe="-_.")
    prefix = quote(str(partition_prefix), safe="-_.")
    if not run or not policy or not prefix:
        raise ValueError(
            "run_id, policy_version, and partition_prefix must not be empty"
        )
    return f"{prefix}-{run}-{policy}"


@dataclass
class TransferQueueConsumerConfig:
    ray_address: str
    run_id: str
    policy_version: str | int
    batch_size: int = 1
    micro_batch_size: int = 1
    task_name: str = "vime-trainer"
    dp_rank: int = 0
    data_fields: list[str] = field(
        default_factory=lambda: list(REQUIRED_TRAJECTORY_FIELDS)
    )
    poll_interval_s: float = 0.2
    service_config_path: str | None = None
    partition_prefix: str = "rollout"

    @property
    def partition_id(self) -> str:
        return build_trajectory_partition_id(
            self.run_id,
            self.policy_version,
            self.partition_prefix,
        )


class TransferQueueTrajectoryConsumer:
    def __init__(
        self,
        config: TransferQueueConsumerConfig,
        *,
        ray_module: Any | None = None,
        transfer_queue_module: Any | None = None,
        streaming_dataset_cls: Any | None = None,
    ):
        self.config = config
        self._ray = ray_module
        self._tq = transfer_queue_module
        self._streaming_dataset_cls = streaming_dataset_cls
        self._tq_config: Any | None = None
        self._dataset: Any | None = None
        self._owns_ray = False

    def initialize(self) -> None:
        _install_numpy2_pickle_compat()
        if self._tq_config is not None:
            return
        if self.config.service_config_path is not None:
            config_path = Path(self.config.service_config_path)
            with config_path.open("rb") as file:
                self._tq_config = pickle.load(file)
            return
        if self._ray is None:
            self._ray = importlib.import_module("ray")
        if self._tq is None:
            self._tq = importlib.import_module("transfer_queue")

        if not self._ray.is_initialized():
            self._ray.init(address=self.config.ray_address)
            self._owns_ray = True
        self._tq.init()

        controller = self._ray.get_actor(
            "TransferQueueController",
            namespace="transfer_queue",
        )
        self._tq_config = self._ray.get(controller.get_config.remote())

    def get_artifact(
        self,
        *,
        key: str,
        partition_id: str | None = None,
        timeout_s: float = 0,
    ) -> TrajectoryArtifactV1Alpha1:
        self.initialize()
        if self._tq is None:
            raise RuntimeError(
                "Point lookup requires a service-Ray connection; exported "
                "configuration supports streaming reads only"
            )
        partition = partition_id or self.config.partition_id
        deadline = time.monotonic() + timeout_s

        while True:
            try:
                fields = self._tq.kv_batch_get(
                    keys=key,
                    partition_id=partition,
                    select_fields=self.config.data_fields,
                )
                tags = self._tq.kv_list(partition)
                tag = tags[partition][key]
                return TrajectoryArtifactV1Alpha1.from_transfer_queue(
                    fields,
                    tag,
                )
            except (KeyError, ValueError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(self.config.poll_interval_s)

    def get_from_handle(
        self,
        handle: dict[str, Any],
        *,
        timeout_s: float = 0,
    ) -> TrajectoryArtifactV1Alpha1:
        if handle.get("backend") != "transfer_queue":
            raise ValueError(
                f"Unsupported artifact backend: {handle.get('backend')}"
            )
        location = handle.get("location") or {}
        key = location.get("key") or handle.get("artifact_id")
        partition_id = location.get("partition_id")
        if not key or not partition_id:
            raise ValueError(
                "TransferQueue artifact handle requires key and partition_id"
            )
        return self.get_artifact(
            key=str(key),
            partition_id=str(partition_id),
            timeout_s=timeout_s,
        )

    def _create_dataset(self) -> Any:
        self.initialize()
        if self._streaming_dataset_cls is None:
            self._streaming_dataset_cls = importlib.import_module(
                "transfer_queue"
            ).StreamingDataset
        return self._streaming_dataset_cls(
            config=self._tq_config,
            batch_size=self.config.batch_size,
            micro_batch_size=self.config.micro_batch_size,
            data_fields=self.config.data_fields,
            partition_id=self.config.partition_id,
            task_name=self.config.task_name,
            dp_rank=self.config.dp_rank,
            should_check_consumption_status=False,
        )

    def iter_artifacts(self) -> Iterator[TrajectoryArtifactV1Alpha1]:
        if self._dataset is None:
            self._dataset = self._create_dataset()
        for fields, batch_meta in self._dataset:
            batch_size = len(batch_meta.custom_meta)
            for index in range(batch_size):
                sample_fields = {
                    name: _select_sample(value, index)
                    for name, value in fields.items()
                }
                yield TrajectoryArtifactV1Alpha1.from_transfer_queue(
                    sample_fields,
                    batch_meta.custom_meta[index],
                )

    def step(
        self,
        run_id: str,
        policy_version: str | int,
    ) -> None:
        self.config.run_id = run_id
        self.config.policy_version = policy_version
        if self._dataset is not None:
            self._dataset.step(self.config.partition_id)

    def close(self) -> None:
        if self._tq is not None:
            try:
                self._tq.get_client().close()
            except (AssertionError, AttributeError):
                pass
        if self._owns_ray and self._ray is not None:
            self._ray.shutdown()
        self._owns_ray = False
