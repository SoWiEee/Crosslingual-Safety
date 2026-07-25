import asyncio
from collections.abc import Awaitable, Callable

from crosslingual_safety.generation.providers import ProviderAdapter
from crosslingual_safety.schemas import GenerationRequest, GenerationResult

RETRYABLE_STATUSES = {"rate_limited", "timeout", "server_error"}


async def execute_with_retry(
    provider: ProviderAdapter,
    request: GenerationRequest,
    *,
    max_attempts: int = 4,
    backoff_base: float = 1.0,
    on_attempt: Callable[[GenerationResult, int, bool], Awaitable[None]] | None = None,
) -> tuple[GenerationResult, int]:
    attempts = 0
    while True:
        attempts += 1
        result = await provider.generate(request)
        empty_retry_allowed = result.status == "empty_response" and attempts < 2
        retry_allowed = (
            result.status in RETRYABLE_STATUSES and attempts < max_attempts
        ) or empty_retry_allowed
        if on_attempt is not None:
            await on_attempt(result, attempts, retry_allowed)
        if not retry_allowed:
            return result, attempts
        await asyncio.sleep(backoff_base * (2 ** (attempts - 1)))
