"""Grouped rollout dispatch to an external artifact-producing vLLM server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class RolloutPrompt:
    group_id: str
    prompt: str | list[int]
    reward_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchedRollout:
    request_id: str
    group_id: str
    sample_index: int
    response: dict[str, Any]
    reward_context: dict[str, Any]


@dataclass(frozen=True)
class RolloutDispatcherConfig:
    endpoint: str
    model: str
    run_id: str
    policy_version: str | int
    n_samples_per_prompt: int
    max_tokens: int
    temperature: float = 1.0
    concurrency: int = 32
    timeout_s: float = 300


class ExternalVllmRolloutDispatcher:
    def __init__(
        self,
        config: RolloutDispatcherConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self._client = client
        self._owns_client = client is None

    def _request_id(self, group_id: str, sample_index: int) -> str:
        return (
            f"{self.config.run_id}-"
            f"{self.config.policy_version}-{group_id}-{sample_index}"
        )

    async def _dispatch_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        prompt: RolloutPrompt,
        sample_index: int,
    ) -> DispatchedRollout:
        request_id = self._request_id(prompt.group_id, sample_index)
        payload = {
            "model": self.config.model,
            "prompt": prompt.prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "logprobs": 0,
            "artifact_transfer_params": {
                "run_id": self.config.run_id,
                "policy_version": self.config.policy_version,
                "model_id": self.config.model,
                "group_id": prompt.group_id,
                "sample_index": sample_index,
            },
        }
        async with semaphore:
            response = await client.post(
                self.config.endpoint,
                json=payload,
                headers={"X-Request-Id": request_id},
            )
            response.raise_for_status()
        return DispatchedRollout(
            request_id=request_id,
            group_id=prompt.group_id,
            sample_index=sample_index,
            response=response.json(),
            reward_context=dict(prompt.reward_context),
        )

    async def dispatch(
        self,
        prompts: list[RolloutPrompt],
    ) -> list[DispatchedRollout]:
        if not prompts:
            return []
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self.config.timeout_s)
            self._client = client
        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = [
            self._dispatch_one(client, semaphore, prompt, sample_index)
            for prompt in prompts
            for sample_index in range(self.config.n_samples_per_prompt)
        ]
        try:
            return await asyncio.gather(*tasks)
        finally:
            if self._owns_client:
                await client.aclose()
                self._client = None
