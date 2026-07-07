import pickle
from pathlib import Path

from transfer_queue import StreamingDataset


def test_exported_config_supports_direct_streaming_dataset():
    config_path = Path(
        "/mnt/data1/yibo/vime-workspace/services/"
        "transfer_queue/client_config.pkl"
    )
    with config_path.open("rb") as file:
        config = pickle.load(file)

    dataset = StreamingDataset(
        config=config,
        batch_size=1,
        micro_batch_size=1,
        data_fields=[
            "prompt_token_ids",
            "response_token_ids",
            "response_logprobs",
        ],
        partition_id="phase5-config-smoke",
        task_name="phase5-config-smoke",
        dp_rank=0,
    )
    assert dataset.config.backend.storage_backend == "SimpleStorage"
