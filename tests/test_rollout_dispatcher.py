import asyncio
import json

import httpx

from vime.rollout.rollout_dispatcher import (
    ExternalVllmRolloutDispatcher,
    RolloutDispatcherConfig,
    RolloutPrompt,
)


def test_grouped_dispatch_sets_stable_artifact_metadata():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"id": request.headers["X-Request-Id"], "choices": []},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm",
    )
    dispatcher = ExternalVllmRolloutDispatcher(
        RolloutDispatcherConfig(
            endpoint="/v1/completions",
            model="model-a",
            run_id="run-a",
            policy_version=3,
            n_samples_per_prompt=2,
            max_tokens=16,
        ),
        client=client,
    )
    results = asyncio.run(
        dispatcher.dispatch(
            [
                RolloutPrompt(
                    group_id="group-a",
                    prompt="hello",
                    reward_context={"label": "answer"},
                ),
                RolloutPrompt(group_id="group-b", prompt=[1, 2]),
            ]
        )
    )
    asyncio.run(client.aclose())

    assert len(results) == 4
    assert [item.sample_index for item in results] == [0, 1, 0, 1]
    assert results[0].reward_context == {"label": "answer"}
    first_payload = json.loads(requests[0].content)
    assert first_payload["logprobs"] == 0
    assert first_payload["artifact_transfer_params"] == {
        "run_id": "run-a",
        "policy_version": 3,
        "model_id": "model-a",
        "group_id": "group-a",
        "sample_index": 0,
    }
    assert requests[0].headers["X-Request-Id"] == "run-a-3-group-a-0"
