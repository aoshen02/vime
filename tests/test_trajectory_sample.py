import pytest
import torch

from vime.rollout.trajectory_artifact import TrajectoryArtifactV1Alpha1
from vime.rollout.trajectory_sample import trajectory_to_sample
from vime.utils.types import Sample


def make_artifact(**overrides):
    values = {
        "run_id": "run-a",
        "request_id": "request-a",
        "engine_id": "engine-a",
        "model_id": "model-a",
        "policy_version": 3,
        "created_at_ns": 123,
        "sample_index": 2,
        "group_id": "group-a",
        "finish_reason": "stop",
        "prompt_token_ids": torch.tensor([1, 2]),
        "response_token_ids": torch.tensor([3, 4]),
        "response_logprobs": torch.tensor([-0.1, -0.2]),
        "rewards": torch.tensor(1.5),
        "loss_mask": None,
    }
    values.update(overrides)
    return TrajectoryArtifactV1Alpha1(**values)


def test_converts_artifact_to_vime_sample():
    sample = trajectory_to_sample(make_artifact())

    assert sample.tokens == [1, 2, 3, 4]
    assert sample.response_length == 2
    assert sample.rollout_log_probs == pytest.approx([-0.1, -0.2])
    assert sample.loss_mask == [1, 1]
    assert sample.reward == pytest.approx(1.5)
    assert sample.status is Sample.Status.COMPLETED
    assert sample.metadata["request_id"] == "request-a"
    assert sample.weight_versions == ["3"]


def test_group_members_share_group_index_but_not_rollout_id_or_sample_index():
    first = trajectory_to_sample(make_artifact(request_id="request-a"))
    second = trajectory_to_sample(make_artifact(request_id="request-b"))
    assert first.group_index == second.group_index
    assert first.rollout_id != second.rollout_id
    assert first.index != second.index


def test_missing_reward_has_explicit_policy():
    artifact = make_artifact(rewards=None)
    with pytest.raises(ValueError, match="rewards are missing"):
        trajectory_to_sample(artifact)
    assert trajectory_to_sample(artifact, require_reward=False).reward is None


def test_rejects_vector_rewards_without_reducer():
    with pytest.raises(ValueError, match="one scalar reward"):
        trajectory_to_sample(make_artifact(rewards=torch.tensor([1.0, 2.0])))


def test_maps_finish_status_and_loss_mask():
    sample = trajectory_to_sample(
        make_artifact(
            finish_reason="length",
            loss_mask=torch.tensor([1.0, 0.0]),
        )
    )
    assert sample.status is Sample.Status.TRUNCATED
    assert sample.loss_mask == [1, 0]


def test_maps_legacy_numeric_length_finish_reason():
    sample = trajectory_to_sample(make_artifact(finish_reason="1"))
    assert sample.status is Sample.Status.TRUNCATED
