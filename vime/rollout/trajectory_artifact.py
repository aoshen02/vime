"""Consumer-side contract for vLLM rollout trajectory artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

TRAJECTORY_SCHEMA_NAME = "vllm.trajectory"
TRAJECTORY_SCHEMA_VERSION = "v1alpha1"
TRAJECTORY_STATUS_COMPLETE = "complete"


def _unwrap_sample(value: Any, field_name: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            samples = list(value.unbind())
            if len(samples) != 1:
                raise ValueError(
                    f"{field_name} contains {len(samples)} samples; expected one"
                )
            return samples[0]
        if value.ndim > 0 and value.shape[0] == 1:
            return value[0]
    return value


def _tensor(value: Any, field_name: str) -> torch.Tensor:
    try:
        return torch.as_tensor(value).detach().cpu().contiguous()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be tensor-like") from exc


def _vector(
    value: Any,
    field_name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = _tensor(_unwrap_sample(value, field_name), field_name)
    if tensor.ndim != 1:
        raise ValueError(
            f"{field_name} must be one-dimensional, got {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=dtype)


@dataclass(frozen=True)
class TrajectoryArtifactV1Alpha1:
    run_id: str
    request_id: str
    engine_id: str
    model_id: str
    policy_version: str | int
    created_at_ns: int
    prompt_token_ids: torch.Tensor
    response_token_ids: torch.Tensor
    response_logprobs: torch.Tensor
    sample_index: int = 0
    group_id: str | None = None
    finish_reason: str | None = None
    rewards: torch.Tensor | None = None
    values: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    routed_experts: torch.Tensor | None = None

    @classmethod
    def from_transfer_queue(
        cls,
        fields: Mapping[str, Any],
        tag: Mapping[str, Any],
    ) -> "TrajectoryArtifactV1Alpha1":
        if tag.get("schema_name") != TRAJECTORY_SCHEMA_NAME:
            raise ValueError(
                f"Unsupported trajectory schema: {tag.get('schema_name')}"
            )
        if tag.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported trajectory schema version: "
                f"{tag.get('schema_version')}"
            )
        if tag.get("status") != TRAJECTORY_STATUS_COMPLETE:
            raise ValueError(
                f"Trajectory is not complete: status={tag.get('status')}"
            )

        required_tags = (
            "run_id",
            "request_id",
            "engine_id",
            "model_id",
            "policy_version",
            "created_at_ns",
        )
        missing_tags = [name for name in required_tags if name not in tag]
        if missing_tags:
            raise ValueError(f"Missing required trajectory tags: {missing_tags}")

        required_fields = (
            "prompt_token_ids",
            "response_token_ids",
            "response_logprobs",
        )
        missing_fields = [name for name in required_fields if name not in fields]
        if missing_fields:
            raise ValueError(
                f"Missing required trajectory fields: {missing_fields}"
            )

        prompt_token_ids = _vector(
            fields["prompt_token_ids"],
            "prompt_token_ids",
            torch.int64,
        )
        response_token_ids = _vector(
            fields["response_token_ids"],
            "response_token_ids",
            torch.int64,
        )
        response_logprobs = _vector(
            fields["response_logprobs"],
            "response_logprobs",
            torch.float32,
        )
        if len(prompt_token_ids) == 0:
            raise ValueError("prompt_token_ids must not be empty")
        if len(response_token_ids) != len(response_logprobs):
            raise ValueError(
                "response_token_ids and response_logprobs must have equal length"
            )

        loss_mask = None
        if "loss_mask" in fields:
            loss_mask = _vector(fields["loss_mask"], "loss_mask", torch.float32)
            if len(loss_mask) != len(response_token_ids):
                raise ValueError(
                    "loss_mask and response_token_ids must have equal length"
                )

        rewards = None
        if "rewards" in fields:
            rewards = _tensor(
                _unwrap_sample(fields["rewards"], "rewards"),
                "rewards",
            ).to(dtype=torch.float32)

        values = None
        if "values" in fields:
            values = _vector(fields["values"], "values", torch.float32)

        routed_experts = None
        if "routed_experts" in fields:
            routed_experts = _tensor(
                _unwrap_sample(fields["routed_experts"], "routed_experts"),
                "routed_experts",
            )

        sample_index = int(tag.get("sample_index", 0))
        created_at_ns = int(tag["created_at_ns"])
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if created_at_ns <= 0:
            raise ValueError("created_at_ns must be positive")

        return cls(
            run_id=str(tag["run_id"]),
            request_id=str(tag["request_id"]),
            engine_id=str(tag["engine_id"]),
            model_id=str(tag["model_id"]),
            policy_version=tag["policy_version"],
            created_at_ns=created_at_ns,
            sample_index=sample_index,
            group_id=(
                None if tag.get("group_id") is None else str(tag["group_id"])
            ),
            finish_reason=tag.get("finish_reason"),
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            response_logprobs=response_logprobs,
            rewards=rewards,
            values=values,
            loss_mask=loss_mask,
            routed_experts=routed_experts,
        )
