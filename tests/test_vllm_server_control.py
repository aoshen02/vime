import asyncio

import pytest

from vime.backends.vllm_utils import server_control

NUM_GPUS = 0


@pytest.mark.unit
def test_get_inflight_diagnostics(monkeypatch):
    async def fake_get(url, *, timeout=None):
        assert timeout == server_control.DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS
        return {"server_load": 1, "inflight": [{"data_parallel_rank": 0}]}

    monkeypatch.setattr(server_control, "get", fake_get)
    result = asyncio.run(server_control.get_inflight_diagnostics(["http://worker:8000/"]))

    assert result == {"http://worker:8000/": {"server_load": 1, "inflight": [{"data_parallel_rank": 0}]}}


@pytest.mark.unit
def test_get_inflight_diagnostics_keeps_partial_failures(monkeypatch):
    async def fake_get(url, *, timeout=None):
        if "bad" in url:
            raise OSError("unreachable")
        return {"server_load": 0}

    monkeypatch.setattr(server_control, "get", fake_get)
    result = asyncio.run(server_control.get_inflight_diagnostics(["http://good", "http://bad"]))

    assert result["http://good"] == {"server_load": 0}
    assert "unreachable" in result["http://bad"]["error"]
