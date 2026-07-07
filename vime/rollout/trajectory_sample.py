"""Conversion from vLLM trajectory artifacts to native VIME samples."""

from __future__ import annotations

import hashlib

import torch

from vime.rollout.trajectory_artifact import TrajectoryArtifactV1Alpha1
from vime.utils.types import Sample


def _stable_int(*components: object) -> int:
    value = "\x1f".join(str(component) for component in components)
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _sample_status(finish_reason: str | None) -> Sample.Status:
    if finish_reason in (None, "stop", "eos"):
        return Sample.Status.COMPLETED
    if finish_reason in ("length", "1"):
        return Sample.Status.TRUNCATED
    if finish_reason in ("abort", "aborted"):
        return Sample.Status.ABORTED
    return Sample.Status.FAILED


def _scalar_reward(
    rewards: torch.Tensor | None,
    *,
    require_reward: bool,
) -> float | None:
    if rewards is None:
        if require_reward:
            raise ValueError("Trajectory is not train-ready: rewards are missing")
        return None
    if rewards.numel() != 1:
        raise ValueError(
            "VIME Sample requires one scalar reward; configure a reward reducer "
            "before consuming vector rewards"
        )
    return float(rewards.reshape(-1)[0].item())


def trajectory_to_sample(
    artifact: TrajectoryArtifactV1Alpha1,
    *,
    require_reward: bool = True,
) -> Sample:
    response_length = len(artifact.response_token_ids)
    if artifact.loss_mask is None:
        loss_mask = [1] * response_length
    else:
        loss_values = artifact.loss_mask.tolist()
        if any(value not in (0, 1, 0.0, 1.0) for value in loss_values):
            raise ValueError("loss_mask values must be binary")
        loss_mask = [int(value) for value in loss_values]

    group_key = artifact.group_id or artifact.request_id
    sample_index = _stable_int(
        artifact.run_id,
        artifact.request_id,
        artifact.sample_index,
    )
    group_index = _stable_int(
        artifact.run_id,
        artifact.policy_version,
        group_key,
    )
    # VIME schedules train steps by unique rollout_id. Keep group_index
    # prompt-scoped, but make rollout_id sample-scoped so n_samples_per_prompt
    # contributes the expected number of trainable rollouts.
    rollout_id = sample_index
    metadata = {
        "artifact_schema": "vllm.trajectory/v1alpha1",
        "run_id": artifact.run_id,
        "request_id": artifact.request_id,
        "group_id": group_key,
        "sample_index": artifact.sample_index,
        "engine_id": artifact.engine_id,
        "model_id": artifact.model_id,
        "policy_version": artifact.policy_version,
        "created_at_ns": artifact.created_at_ns,
        "finish_reason": artifact.finish_reason,
    }

    return Sample(
        group_index=group_index,
        index=sample_index,
        rollout_id=rollout_id,
        tokens=(
            artifact.prompt_token_ids.tolist()
            + artifact.response_token_ids.tolist()
        ),
        response_length=response_length,
        reward=_scalar_reward(
            artifact.rewards,
            require_reward=require_reward,
        ),
        loss_mask=loss_mask,
        weight_versions=[str(artifact.policy_version)],
        rollout_log_probs=artifact.response_logprobs.tolist(),
        rollout_routed_experts=(
            None
            if artifact.routed_experts is None
            else artifact.routed_experts.tolist()
        ),
        status=_sample_status(artifact.finish_reason),
        metadata=metadata,
        train_metadata={
            "model_id": artifact.model_id,
            "policy_version": artifact.policy_version,
        },
    )
