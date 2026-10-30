"""PD and EPD-specific vLLM deployment sequencing."""

import logging
from typing import Any

import ray

from vime.backends.vllm_utils.engine_group import ServerGroup, ServerGroupPlacement
from vime.backends.vllm_utils.vllm_config import ModelConfig

logger = logging.getLogger(__name__)


def start_pd_server_groups(
    model_config: ModelConfig,
    placement: ServerGroupPlacement,
    router_ip: str | None,
    router_port: int | None,
) -> tuple[list[ServerGroup], list[Any]]:
    """Start prefill/decode groups without waiting for final engine initialization."""
    server_groups = []
    init_handles = []
    for group_config in model_config.server_groups:
        group = placement.create(group_config, router_ip, router_port)
        init_handles.extend(placement.start(group))
        server_groups.append(group)
    return server_groups, init_handles


def start_epd_server_groups(
    model_config: ModelConfig,
    placement: ServerGroupPlacement,
    router_ip: str | None,
    router_port: int | None,
) -> tuple[list[ServerGroup], list[Any], list[str]]:
    """Start encoder groups first, then return their endpoints for rollout."""
    server_groups = []
    transfer_overrides = {
        "ec_transfer_config": {
            "ec_connector": "ECCPUConnector",
            "ec_connector_extra_config": {
                "ec_cpu_bytes": 1_000_000_000,
                "ec_enable_nixl": True,
            },
        },
    }

    encoder_endpoints: list[str] = []
    for group_config in model_config.server_groups:
        if group_config.worker_type != "encoder":
            continue
        group = placement.create(
            group_config,
            router_ip,
            router_port,
            overrides_extra=transfer_overrides,
        )
        handles = placement.start(group)
        if handles:
            ray.get(handles)
        endpoints = ray.get([engine.get_url.remote() for engine in group.engines])
        encoder_endpoints.extend(endpoint for endpoint in endpoints if endpoint is not None)
        server_groups.append(group)

    logger.info("EPD phase 1 done: collected %d encoder endpoints", len(encoder_endpoints))

    init_handles = []
    for group_config in model_config.server_groups:
        if group_config.worker_type == "encoder":
            continue
        overrides_extra = (
            transfer_overrides if encoder_endpoints and group_config.worker_type in ("regular", "prefill") else None
        )
        group = placement.create(
            group_config,
            router_ip,
            router_port,
            overrides_extra=overrides_extra,
        )
        init_handles.extend(placement.start(group))
        server_groups.append(group)

    return server_groups, init_handles, encoder_endpoints
