import sys
from types import SimpleNamespace

import numpy
import torch

from vime.rollout.transfer_queue_consumer import (
    TransferQueueConsumerConfig,
    TransferQueueTrajectoryConsumer,
    _install_numpy2_pickle_compat,
    build_trajectory_partition_id,
)


def make_tag(request_id="request-a"):
    return {
        "schema_name": "vllm.trajectory",
        "schema_version": "v1alpha1",
        "status": "complete",
        "run_id": "run-a",
        "request_id": request_id,
        "engine_id": "engine-a",
        "model_id": "model-a",
        "policy_version": 1,
        "created_at_ns": 123,
    }


def make_fields():
    return {
        "prompt_token_ids": torch.tensor([[1, 2]]),
        "response_token_ids": torch.tensor([[3]]),
        "response_logprobs": torch.tensor([[-0.5]]),
    }


class FakeRay:
    def __init__(self):
        self.initialized = False
        self.shutdown_count = 0
        self.controller = SimpleNamespace(
            get_config=SimpleNamespace(remote=lambda: "tq-config")
        )

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True
        self.init_kwargs = kwargs

    def get_actor(self, *args, **kwargs):
        return self.controller

    def get(self, value):
        return value

    def shutdown(self):
        self.shutdown_count += 1
        self.initialized = False


class FakeTQ:
    def __init__(self):
        self.fields = make_fields()
        self.tags = {
            "rollout-run-a-1": {"request-a": make_tag()},
        }
        self.client = SimpleNamespace(close=lambda: None)

    def init(self):
        return None

    def kv_batch_get(self, **kwargs):
        return self.fields

    def kv_list(self, partition_id):
        return self.tags

    def get_client(self):
        return self.client


def make_consumer(**kwargs):
    ray = FakeRay()
    tq = FakeTQ()
    consumer = TransferQueueTrajectoryConsumer(
        TransferQueueConsumerConfig(
            ray_address="ray://trainer:20001",
            run_id="run-a",
            policy_version=1,
            **kwargs,
        ),
        ray_module=ray,
        transfer_queue_module=tq,
    )
    return consumer, ray, tq


def test_numpy2_pickle_compat_aliases_core_modules():
    if int(numpy.__version__.split(".", 1)[0]) >= 2:
        return
    _install_numpy2_pickle_compat()
    assert sys.modules["numpy._core"] is numpy.core
    assert sys.modules["numpy._core.numeric"] is numpy.core.numeric


def test_build_partition_id_escapes_components():
    assert (
        build_trajectory_partition_id("run/a", "policy 1")
        == "rollout-run%2Fa-policy%201"
    )


def test_point_lookup_and_handle_lookup():
    consumer, ray, _ = make_consumer()
    artifact = consumer.get_artifact(key="request-a")
    assert artifact.response_token_ids.tolist() == [3]
    assert ray.init_kwargs == {"address": "ray://trainer:20001"}

    handle = {
        "backend": "transfer_queue",
        "artifact_id": "request-a",
        "location": {
            "partition_id": "rollout-run-a-1",
            "key": "request-a",
        },
    }
    assert consumer.get_from_handle(handle).request_id == "request-a"
    consumer.close()
    assert ray.shutdown_count == 1


def test_streaming_dataset_and_partition_step():
    nested_fields = make_fields()
    nested_fields["response_token_ids"] = torch.nested.nested_tensor(
        [torch.tensor([3])]
    )
    nested_fields["response_logprobs"] = torch.nested.nested_tensor(
        [torch.tensor([-0.5])]
    )
    batches = [
        (
            nested_fields,
            SimpleNamespace(custom_meta=[make_tag()]),
        )
    ]

    class FakeDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.partitions = []

        def __iter__(self):
            return iter(batches)

        def step(self, partition):
            self.partitions.append(partition)

    consumer, _, _ = make_consumer(batch_size=4, micro_batch_size=1)
    consumer._streaming_dataset_cls = FakeDataset
    artifact = next(consumer.iter_artifacts())
    assert artifact.prompt_token_ids.tolist() == [1, 2]
    assert consumer._dataset.kwargs["task_name"] == "vime-trainer"

    consumer.step("run-b", 2)
    assert consumer._dataset.partitions == ["rollout-run-b-2"]
