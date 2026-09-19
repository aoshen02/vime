"""Control-plane helper for aborting in-flight requests on vLLM workers."""

import asyncio
import logging

from vime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

DEFAULT_ABORT_TIMEOUT_SECONDS = 180.0
DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS = 10.0


async def abort_inflight_requests(urls: list[str]) -> None:
    """Abort all in-flight requests on each worker (one best-effort sweep).

    Posts to ``/abort_requests`` with an empty body; failures are logged, not
    raised. Idempotent, so the caller may re-issue it to converge.
    """

    async def _abort_one(url: str) -> None:
        try:
            await post(f"{url.rstrip('/')}/abort_requests", {}, max_retries=3)
        except Exception as e:
            logger.warning(f"Failed to abort requests on {url}: {e}")

    await asyncio.gather(*(_abort_one(url) for url in urls))


async def get_inflight_diagnostics(urls: list[str]) -> dict[str, object]:
    """Return bounded vLLM queue snapshots for abort timeout errors."""

    async def _get_one(url: str) -> object:
        try:
            return await get(
                f"{url.rstrip('/')}/load?include_inflight=true&inflight_limit=100",
                timeout=DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as error:
            return {"error": repr(error)}

    return dict(zip(urls, await asyncio.gather(*(_get_one(url) for url in urls)), strict=True))
