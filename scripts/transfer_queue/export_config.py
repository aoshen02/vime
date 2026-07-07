"""Export resolved TransferQueue client configuration for non-service Ray jobs."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import ray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ray.init(address=args.ray_address)
    controller = ray.get_actor(
        "TransferQueueController",
        namespace="transfer_queue",
    )
    config = ray.get(controller.get_config.remote())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(config, file)
    os.replace(temporary, output)
    print(f"TRANSFER_QUEUE_CONFIG_EXPORTED path={output}")
    ray.shutdown()


if __name__ == "__main__":
    main()
