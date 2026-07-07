from types import SimpleNamespace

import torch

from vime.rollout.trajectory_artifact import TrajectoryArtifactV1Alpha1
from vime.rollout.trajectory_source import TransferQueueSampleSource
from vime.ray.rollout import RolloutManager


def make_artifact(request_id, reward=1.0):
    return TrajectoryArtifactV1Alpha1(
        run_id="run-a",
        request_id=request_id,
        engine_id="engine-a",
        model_id="model-a",
        policy_version=1,
        created_at_ns=123,
        prompt_token_ids=torch.tensor([1, 2]),
        response_token_ids=torch.tensor([3]),
        response_logprobs=torch.tensor([-0.5]),
        rewards=None if reward is None else torch.tensor(reward),
    )


class FakeConsumer:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.closed = False

    def iter_artifacts(self):
        yield from self.artifacts

    def close(self):
        self.closed = True


def test_takes_exact_native_sample_batch():
    consumer = FakeConsumer(
        [make_artifact("a"), make_artifact("b"), make_artifact("c")]
    )
    source = TransferQueueSampleSource(consumer, require_reward=True)
    samples = source.take(2)

    assert [sample.metadata["request_id"] for sample in samples] == ["a", "b"]
    assert all(sample.tokens == [1, 2, 3] for sample in samples)
    assert source.take(1)[0].metadata["request_id"] == "c"
    source.close()
    assert consumer.closed


def test_from_args_selects_train_ready_fields(monkeypatch):
    captured = {}

    class FakeTrajectoryConsumer:
        def __init__(self, config):
            captured["config"] = config

    monkeypatch.setattr(
        "vime.rollout.trajectory_source.TransferQueueTrajectoryConsumer",
        FakeTrajectoryConsumer,
    )
    args = SimpleNamespace(
        tq_ray_address="ray://trainer:20001",
        tq_run_id="run-a",
        tq_policy_version="1",
        tq_consumer_batch_size=8,
        tq_consumer_micro_batch_size=2,
        tq_task_name="trainer-a",
        tq_dp_rank=0,
        tq_poll_interval=0.1,
        tq_require_rewards=True,
        tq_include_loss_mask=True,
        tq_include_routed_experts=False,
    )
    source = TransferQueueSampleSource.from_args(args)

    assert source.require_reward
    assert captured["config"].data_fields == [
        "prompt_token_ids",
        "response_token_ids",
        "response_logprobs",
        "rewards",
        "loss_mask",
    ]


def test_rollout_manager_reads_transfer_queue_batch():
    class FakeSource:
        def take(self, count):
            assert count == 2
            return ["sample-a", "sample-b"]

    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = SimpleNamespace(
        load_debug_rollout_data=None,
        trajectory_source="transfer_queue",
        rollout_batch_size=2,
        n_samples_per_prompt=1,
    )
    manager.trajectory_sample_source = FakeSource()

    samples, metrics = manager._get_rollout_data(rollout_id=4)
    assert samples == ["sample-a", "sample-b"]
    assert metrics == {"transfer_queue/samples": 2}


def test_transfer_queue_can_keep_managed_rollout_servers(monkeypatch):
    # Closed-loop artifact-transfer training still relies on VIME's normal
    # rollout-engine ownership so actor_model.update_weights() has engines to update.
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    started = {}

    monkeypatch.setattr(
        "vime.ray.rollout.TransferQueueSampleSource.from_args",
        lambda args: object(),
    )
    monkeypatch.setattr(
        "vime.ray.rollout.init_http_client",
        lambda args: started.setdefault("http", True),
    )
    monkeypatch.setattr(
        "vime.ray.rollout.start_rollout_servers",
        lambda args, pg: started.setdefault("servers", {"actor": object()}),
    )
    monkeypatch.setattr(
        "vime.ray.rollout.init_tracking",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vime.ray.rollout.Lock",
        SimpleNamespace(
            options=lambda **kwargs: SimpleNamespace(remote=lambda: object())
        ),
    )

    manager = manager_cls.__new__(manager_cls)
    manager_cls.__init__(
        manager,
        SimpleNamespace(
            trajectory_source="transfer_queue",
            tq_manage_rollout_servers=True,
            debug_train_only=False,
            custom_reward_post_process_path=None,
            custom_convert_samples_to_train_data_path=None,
            use_fault_tolerance=False,
            ci_test=False,
        ),
        pg=object(),
    )

    assert started["http"]
    assert manager.servers == {"actor": started["servers"]["actor"]}

