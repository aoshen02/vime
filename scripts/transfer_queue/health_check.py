"""Non-destructive health check for the shared Ray and TransferQueue services."""

from __future__ import annotations

import argparse

import ray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", required=True)
    args = parser.parse_args()

    ray.init(address=args.ray_address)
    controller = ray.get_actor(
        "TransferQueueController",
        namespace="transfer_queue",
    )
    config = ray.get(controller.get_config.remote())
    print(
        "TRANSFER_QUEUE_HEALTHY "
        f"backend={config.backend.storage_backend} "
        f"ray_address={args.ray_address}"
    )
    ray.shutdown()


if __name__ == "__main__":
    main()
