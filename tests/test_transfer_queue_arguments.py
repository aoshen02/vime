import argparse

from vime.utils.arguments import get_vime_extra_args_provider


def test_transfer_queue_arguments_are_opt_in():
    parser = argparse.ArgumentParser()
    parser = get_vime_extra_args_provider()(parser)
    args = parser.parse_args(["--rollout-batch-size", "1"])

    assert args.trajectory_source == "generated"
    assert args.tq_ray_address == "ray://172.16.1.248:20001"
    assert args.tq_require_rewards
    assert not args.tq_manage_rollout_servers


def test_transfer_queue_argument_values():
    parser = argparse.ArgumentParser()
    parser = get_vime_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--rollout-batch-size",
            "1",
            "--trajectory-source",
            "transfer_queue",
            "--tq-run-id",
            "run-a",
            "--tq-policy-version",
            "7",
            "--tq-consumer-batch-size",
            "8",
            "--tq-allow-missing-rewards",
            "--tq-manage-rollout-servers",
        ]
    )

    assert args.trajectory_source == "transfer_queue"
    assert args.tq_run_id == "run-a"
    assert args.tq_policy_version == "7"
    assert args.tq_consumer_batch_size == 8
    assert not args.tq_require_rewards
    assert args.tq_manage_rollout_servers
