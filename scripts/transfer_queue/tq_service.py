"""Long-running TransferQueue service driver for the shared training cluster."""

from __future__ import annotations

import argparse
import signal
import threading

import ray
import transfer_queue as tq
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", required=True)
    parser.add_argument(
        "--polling-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return empty fetches instead of blocking the TQ controller request loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init(address=args.ray_address)
    config = tq.init(
        OmegaConf.create(
            {"controller": {"polling_mode": args.polling_mode}},
            flags={"allow_objects": True},
        )
    )
    backend = config.backend.storage_backend if config is not None else "existing"
    print(
        f"TRANSFER_QUEUE_SERVICE_READY backend={backend} "
        f"polling_mode={args.polling_mode}",
        flush=True,
    )

    stopped = threading.Event()

    def request_stop(*_args) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    stopped.wait()

    # Close this process's client only. tq.close() would destroy shared actors.
    tq.get_client().close()
    ray.shutdown()


if __name__ == "__main__":
    main()
