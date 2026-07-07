import pytest
import torch

from vime.rollout.trajectory_artifact import TrajectoryArtifactV1Alpha1


def make_tag(**overrides):
    tag = {
        "schema_name": "vllm.trajectory",
        "schema_version": "v1alpha1",
        "status": "complete",
        "run_id": "run-a",
        "request_id": "request-a",
        "group_id": "group-a",
        "sample_index": 2,
        "engine_id": "engine-a",
        "model_id": "model-a",
        "policy_version": 7,
        "created_at_ns": 123,
        "finish_reason": "stop",
    }
    tag.update(overrides)
    return tag


def make_fields(**overrides):
    fields = {
        "prompt_token_ids": torch.tensor([[1, 2, 3]]),
        "response_token_ids": torch.tensor([[4, 5]]),
        "response_logprobs": torch.tensor([[-0.1, -0.2]]),
    }
    fields.update(overrides)
    return fields


def test_decode_vllm_trajectory():
    artifact = TrajectoryArtifactV1Alpha1.from_transfer_queue(
        make_fields(
            rewards=torch.tensor([1.5]),
            loss_mask=torch.tensor([[1.0, 0.0]]),
        ),
        make_tag(),
    )

    assert artifact.prompt_token_ids.tolist() == [1, 2, 3]
    assert artifact.response_token_ids.tolist() == [4, 5]
    assert artifact.response_logprobs.tolist() == pytest.approx([-0.1, -0.2])
    assert artifact.rewards.item() == pytest.approx(1.5)
    assert artifact.loss_mask.tolist() == [1.0, 0.0]
    assert artifact.group_id == "group-a"
    assert artifact.sample_index == 2


@pytest.mark.parametrize(
    ("tag_update", "error"),
    [
        ({"schema_name": "other"}, "Unsupported trajectory schema"),
        ({"schema_version": "v2"}, "Unsupported trajectory schema version"),
        ({"status": "writing"}, "Trajectory is not complete"),
    ],
)
def test_rejects_incompatible_tags(tag_update, error):
    with pytest.raises(ValueError, match=error):
        TrajectoryArtifactV1Alpha1.from_transfer_queue(
            make_fields(),
            make_tag(**tag_update),
        )


def test_rejects_token_logprob_length_mismatch():
    with pytest.raises(ValueError, match="must have equal length"):
        TrajectoryArtifactV1Alpha1.from_transfer_queue(
            make_fields(response_logprobs=torch.tensor([[-0.1]])),
            make_tag(),
        )


def test_decodes_single_nested_tensor_sample():
    artifact = TrajectoryArtifactV1Alpha1.from_transfer_queue(
        make_fields(
            response_token_ids=torch.nested.nested_tensor(
                [torch.tensor([4, 5])]
            ),
            response_logprobs=torch.nested.nested_tensor(
                [torch.tensor([-0.1, -0.2])]
            ),
        ),
        make_tag(),
    )
    assert artifact.response_token_ids.tolist() == [4, 5]
