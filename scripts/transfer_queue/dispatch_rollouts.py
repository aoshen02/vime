"""Dispatch JSONL prompts to an external vLLM completions endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from vime.rollout.rollout_dispatcher import (
    ExternalVllmRolloutDispatcher,
    RolloutDispatcherConfig,
    RolloutPrompt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--n-samples-per-prompt", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts = []
    with Path(args.prompts).open() as file:
        for line in file:
            row = json.loads(line)
            prompts.append(
                RolloutPrompt(
                    group_id=str(row["group_id"]),
                    prompt=row["prompt"],
                    reward_context=row.get("reward_context", {}),
                )
            )

    dispatcher = ExternalVllmRolloutDispatcher(
        RolloutDispatcherConfig(
            endpoint=args.endpoint,
            model=args.model,
            run_id=args.run_id,
            policy_version=args.policy_version,
            n_samples_per_prompt=args.n_samples_per_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )
    )
    results = asyncio.run(dispatcher.dispatch(prompts))
    with Path(args.output).open("w") as file:
        for result in results:
            file.write(
                json.dumps(
                    {
                        "request_id": result.request_id,
                        "group_id": result.group_id,
                        "sample_index": result.sample_index,
                        "reward_context": result.reward_context,
                        "response": result.response,
                    }
                )
                + "\n"
            )
    print(f"ROLLOUT_DISPATCH_COMPLETE samples={len(results)}")


if __name__ == "__main__":
    main()
