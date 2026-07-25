import json
import time
from collections.abc import Sequence
from typing import Protocol

import httpx

from crosslingual_safety.schemas import (
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
)


class AuthenticationError(RuntimeError):
    """Authentication errors abort an experiment without exposing credentials."""


class ProviderAdapter(Protocol):
    provider_id: str

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


def _result(
    request: GenerationRequest,
    status: GenerationStatus,
    *,
    response_text: str | None = None,
    actual_model_id: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    latency_ms: float | None = None,
    provider_request_id: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    raw_response_json: str | None = None,
) -> GenerationResult:
    return GenerationResult(
        run_id=request.run_id,
        status=status,
        response_text=response_text,
        actual_model_id=actual_model_id,
        raw_response_path=None,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        provider_request_id=provider_request_id,
        error_type=error_type,
        error_message=error_message,
        raw_response_json=raw_response_json,
    )


class FakeProvider:
    provider_id = "fake"

    def __init__(
        self,
        outcomes: Sequence[GenerationStatus] | None = None,
        response_text: str = "fake response",
    ) -> None:
        self.outcomes: list[GenerationStatus] = (
            list(outcomes) if outcomes is not None else ["success"]
        )
        self.response_text = response_text
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request.model_copy(deep=True))
        status: GenerationStatus = self.outcomes.pop(0) if self.outcomes else "success"
        if status == "success":
            return _result(
                request,
                "success",
                response_text=self.response_text,
                actual_model_id=request.requested_model_id,
                finish_reason="stop",
                prompt_tokens=len(request.rendered_prompt.split()),
                completion_tokens=len(self.response_text.split()),
                latency_ms=0.0,
                provider_request_id=f"fake-{len(self.requests)}",
            )
        return _result(
            request,
            status,
            error_type=status,
            error_message=f"fake {status}",
        )


class _OpenAICompatibleProvider:
    endpoint_path: str

    def __init__(
        self,
        provider_id: str,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._transport = transport

    def _payload(self, request: GenerationRequest) -> dict[str, object]:
        raise NotImplementedError

    def _response_text(self, choice: dict[str, object]) -> str | None:
        raise NotImplementedError

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/{self.endpoint_path}",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=self._payload(request),
                )
        except httpx.TimeoutException:
            return _result(
                request,
                "timeout",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_type="timeout",
                error_message="provider request timed out",
            )
        except httpx.RequestError:
            return _result(
                request,
                "server_error",
                latency_ms=(time.perf_counter() - started) * 1000,
                error_type="connection_error",
                error_message="provider connection failed",
            )

        latency_ms = (time.perf_counter() - started) * 1000
        request_id = response.headers.get("x-request-id")
        safe_response_text = response.text.replace(self._api_key, "[REDACTED]")
        if response.status_code in {401, 403}:
            raise AuthenticationError(f"{self.provider_id} authentication failed")
        if response.status_code == 408:
            return _result(
                request,
                "timeout",
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type="http_408",
                error_message="provider request timed out",
                raw_response_json=safe_response_text,
            )
        if response.status_code == 429:
            return _result(
                request,
                "rate_limited",
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type="http_429",
                error_message="provider rate limit exceeded",
                raw_response_json=safe_response_text,
            )
        if response.status_code >= 500:
            return _result(
                request,
                "server_error",
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type=f"http_{response.status_code}",
                error_message="provider server error",
                raw_response_json=safe_response_text,
            )
        if response.status_code >= 400:
            try:
                error_code = json.loads(safe_response_text).get("error", {}).get("code")
            except (ValueError, AttributeError):
                error_code = None
            status: GenerationStatus = (
                "provider_blocked" if error_code == "content_filter" else "invalid_request"
            )
            return _result(
                request,
                status,
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type=str(error_code or f"http_{response.status_code}"),
                error_message=(
                    "provider content filter blocked request"
                    if status == "provider_blocked"
                    else "provider rejected request parameters"
                ),
                raw_response_json=safe_response_text,
            )

        try:
            body = json.loads(safe_response_text)
            choice = body["choices"][0]
            finish_reason = choice.get("finish_reason")
            actual_model = body.get("model")
            usage = body.get("usage", {})
            response_text = self._response_text(choice)
            request_id = request_id or body.get("id")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return _result(
                request,
                "empty_response",
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type="malformed_response",
                error_message="provider returned no usable choice",
                raw_response_json=safe_response_text,
            )
        if finish_reason == "content_filter":
            return _result(
                request,
                "provider_blocked",
                actual_model_id=actual_model,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type="content_filter",
                error_message="provider content filter blocked response",
                raw_response_json=safe_response_text,
            )
        if not response_text:
            return _result(
                request,
                "empty_response",
                actual_model_id=actual_model,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                provider_request_id=request_id,
                error_type="empty_response",
                error_message="provider returned empty text",
                raw_response_json=safe_response_text,
            )
        return _result(
            request,
            "success",
            response_text=response_text,
            actual_model_id=actual_model,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            provider_request_id=request_id,
            raw_response_json=safe_response_text,
        )


class OpenAICompatibleChatProvider(_OpenAICompatibleProvider):
    endpoint_path = "chat/completions"

    def _payload(self, request: GenerationRequest) -> dict[str, object]:
        messages = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.rendered_prompt})
        payload: dict[str, object] = {
            "model": request.requested_model_id,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    def _response_text(self, choice: dict[str, object]) -> str | None:
        message = choice.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        return content if isinstance(content, str) else None


class OpenAICompatibleCompletionProvider(_OpenAICompatibleProvider):
    endpoint_path = "completions"

    def _payload(self, request: GenerationRequest) -> dict[str, object]:
        prompt = (
            f"{request.system_prompt}\n\n{request.rendered_prompt}"
            if request.system_prompt is not None
            else request.rendered_prompt
        )
        payload: dict[str, object] = {
            "model": request.requested_model_id,
            "prompt": prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    def _response_text(self, choice: dict[str, object]) -> str | None:
        text = choice.get("text")
        return text if isinstance(text, str) else None
