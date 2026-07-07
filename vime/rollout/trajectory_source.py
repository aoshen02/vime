"""VIME rollout source backed by TransferQueue trajectory artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from vime.rollout.trajectory_sample import trajectory_to_sample
from vime.rollout.transfer_queue_consumer import (
    REQUIRED_TRAJECTORY_FIELDS,
    TransferQueueConsumerConfig,
    TransferQueueTrajectoryConsumer,
)
from vime.utils.types import Sample


class TransferQueueSampleSource:
    def __init__(
        self,
        consumer: TransferQueueTrajectoryConsumer,
        *,
        require_reward: bool,
    ):
        self.consumer = consumer
        self.require_reward = require_reward
        self._iterator: Iterator | None = None

    @classmethod
    def from_args(cls, args: Any) -> "TransferQueueSampleSource":
        data_fields = list(REQUIRED_TRAJECTORY_FIELDS)
        if args.tq_require_rewards:
            data_fields.append("rewards")
        if args.tq_include_loss_mask:
            data_fields.append("loss_mask")
        if args.tq_include_routed_experts:
            data_fields.append("routed_experts")

        consumer = TransferQueueTrajectoryConsumer(
            TransferQueueConsumerConfig(
                ray_address=args.tq_ray_address,
                run_id=args.tq_run_id,
                policy_version=args.tq_policy_version,
                batch_size=args.tq_consumer_batch_size,
                micro_batch_size=args.tq_consumer_micro_batch_size,
                task_name=args.tq_task_name,
                dp_rank=args.tq_dp_rank,
                data_fields=data_fields,
                poll_interval_s=args.tq_poll_interval,
                service_config_path=getattr(
                    args, "tq_service_config_path", None
                ),
                partition_prefix=getattr(
                    args, "tq_partition_prefix", "train"
                ),
            )
        )
        return cls(
            consumer,
            require_reward=args.tq_require_rewards,
        )

    def take(self, count: int) -> list[Sample]:
        if count < 1:
            raise ValueError("TransferQueue sample count must be positive")
        if self._iterator is None:
            self._iterator = self.consumer.iter_artifacts()
        return [
            trajectory_to_sample(
                next(self._iterator),
                require_reward=self.require_reward,
            )
            for _ in range(count)
        ]

    def close(self) -> None:
        self.consumer.close()
