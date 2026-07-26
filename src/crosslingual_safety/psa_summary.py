"""Runtime Paper Summary Attack summarization and immutable artifacts.

The PSA renderer keeps a reviewed, static corpus as its low-level fallback.  A manual run uses
this module to derive one small, auditable summary for each wrapper language before any victim
request is queued.  The service deliberately uses the existing generation provider contract so
authentication and typed provider statuses have the same semantics as victim generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from crosslingual_safety.generation.providers import (
    AuthenticationError,
    OpenAICompatibleChatProvider,
)
from crosslingual_safety.schemas import GenerationRequest, GenerationResult

SUMMARY_LANGUAGES: tuple[str, ...] = ("en", "zh", "vi", "my")
SUMMARY_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Traditional Chinese",
    "vi": "Vietnamese",
    "my": "Burmese",
}
SUMMARY_KEYS: tuple[str, ...] = (
    "attack_methods",
    "mechanism_analysis",
    "related_work",
)
SUMMARY_MODEL_ID = "ais3/gemma-4-12b"
SUMMARY_ENDPOINT_TYPE = "chat"
SUMMARY_GENERATION_CONFIG: dict[str, object] = {
    "max_tokens": 2048,
    "temperature": 0,
}
SUMMARY_MIN_TIMEOUT_SECONDS = 180.0


def canonical_json(value: object) -> str:
    """Serialize an object using the repository's reproducibility contract."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SummaryArtifact(BaseModel):
    """A successful, content-addressed summary response.

    The response text is retained for auditability.  It is provider output, never a credential;
    provider request IDs are safe correlation IDs and may be absent for local test providers.
    """

    model_config = ConfigDict(extra="forbid")

    summary_id: str
    language: str
    source_sha256: str
    request_sha256: str
    provider_id: str
    model_id: str
    endpoint_type: str
    generation_config: dict[str, object]
    response_text: str
    response_sha256: str
    provider_request_id: str | None
    created_at: str

    @field_validator(
        "summary_id",
        "language",
        "source_sha256",
        "request_sha256",
        "provider_id",
        "model_id",
        "endpoint_type",
        "response_text",
        "response_sha256",
        "created_at",
    )
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("summary artifact fields must not be empty")
        return value


class SummaryProvider(Protocol):
    provider_id: str

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


class SummaryError(ValueError):
    """A summary contract or provider response failure."""


def _response_json_text(response_text: str) -> str:
    """Remove one complete JSON markdown fence, rejecting surrounding prose."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise SummaryError("summary provider returned an empty response")
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        if "```" in stripped:
            raise SummaryError("summary response must not contain markdown prose or fences")
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
        raise SummaryError("summary response must use one complete JSON markdown fence")
    if lines[-1].strip() != "```" or "```" in "\n".join(lines[1:-1]):
        raise SummaryError("summary response must use one complete JSON markdown fence")
    body = "\n".join(lines[1:-1]).strip()
    if not body:
        raise SummaryError("summary provider returned an empty response")
    return body


def _parse_summary_object(response_text: str) -> dict[str, str]:
    """Parse the strict three-field JSON response required by PSA.

    ``json.loads`` accepts duplicate keys by default, which would make the persisted text and
    selected values ambiguous. Reject duplicate keys, malformed fences, extra keys, and empty
    values explicitly so malformed output aborts before a victim request can be created.
    """

    stripped = _response_json_text(response_text)

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SummaryError(f"summary response contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(stripped, object_pairs_hook=object_pairs)
    except (json.JSONDecodeError, SummaryError) as error:
        if isinstance(error, SummaryError):
            raise
        raise SummaryError("summary response is not valid JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != set(SUMMARY_KEYS):
        raise SummaryError(
            "summary response must contain exactly attack_methods, mechanism_analysis, and "
            "related_work"
        )
    values: dict[str, str] = {}
    for key in SUMMARY_KEYS:
        value = parsed[key]
        if not isinstance(value, str) or not value.strip():
            raise SummaryError(f"summary response value {key} must be a non-empty string")
        values[key] = value
    return values


def parse_summary_response(response_text: str) -> dict[str, str]:
    """Return the validated three generated PSA fields."""

    return _parse_summary_object(response_text)


def normalize_summary_response(response_text: str) -> tuple[str, dict[str, str]]:
    """Return canonical JSON text and its validated generated fields."""

    parsed = _parse_summary_object(response_text)
    return canonical_json(parsed), parsed


def artifact_sections(artifact: SummaryArtifact) -> dict[str, str]:
    """Return the parsed generated fields from an artifact, validating its response text."""

    parsed = parse_summary_response(artifact.response_text)
    if sha256_text(artifact.response_text) != artifact.response_sha256:
        raise SummaryError("summary artifact response_sha256 does not match response_text")
    return parsed


def artifact_contract(artifact: SummaryArtifact) -> dict[str, object]:
    """Return stable artifact fields suitable for renderer hashes and variant IDs.

    Provider request IDs and creation timestamps are useful audit fields in the JSONL cache but
    are not part of the generated content contract.  Excluding those ephemeral values keeps a
    variant ID tied to the source/request/model/output hashes and not to wall-clock time.
    """

    return {
        "summary_id": artifact.summary_id,
        "language": artifact.language,
        "source_sha256": artifact.source_sha256,
        "request_sha256": artifact.request_sha256,
        "provider_id": artifact.provider_id,
        "model_id": artifact.model_id,
        "endpoint_type": artifact.endpoint_type,
        "generation_config": artifact.generation_config,
        "response_sha256": artifact.response_sha256,
    }


class PaperSummaryService:
    """Create deterministic summary requests and validate immutable four-row caches."""

    def __init__(
        self,
        summary_id: str,
        source_sections: Mapping[str, str],
        provenance: Mapping[str, object],
        summary_prompt: Mapping[str, str],
        provider: SummaryProvider | None = None,
        provider_id: str = "zoolab",
        model_id: str = SUMMARY_MODEL_ID,
        endpoint_type: str = SUMMARY_ENDPOINT_TYPE,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if not summary_id:
            raise ValueError("summary_id must not be empty")
        if set(source_sections) == set():
            raise ValueError("summary source sections must not be empty")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in source_sections.values()
        ):
            raise ValueError("summary source sections must be non-empty strings")
        if set(summary_prompt) != {"system", "user"}:
            raise ValueError("summary_prompt must contain exactly system and user templates")
        if any(
            not isinstance(value, str) or not value.strip() for value in summary_prompt.values()
        ):
            raise ValueError("summary_prompt templates must be non-empty strings")
        if model_id != SUMMARY_MODEL_ID:
            raise ValueError(f"PSA summary model must be {SUMMARY_MODEL_ID}")
        if endpoint_type != SUMMARY_ENDPOINT_TYPE:
            raise ValueError("PSA summary endpoint_type must be chat")

        self.summary_id = summary_id
        self.source_sections = {str(key): str(value) for key, value in source_sections.items()}
        self.provenance = {str(key): value for key, value in provenance.items()}
        self.summary_prompt = {str(key): str(value) for key, value in summary_prompt.items()}
        self.provider_id = provider_id
        self.model_id = model_id
        self.endpoint_type = endpoint_type
        self.timeout_seconds = max(float(timeout_seconds), SUMMARY_MIN_TIMEOUT_SECONDS)
        self.generation_config = {
            **SUMMARY_GENERATION_CONFIG,
            "timeout_seconds": self.timeout_seconds,
        }
        self._clock = clock or _utc_now

        source_contract = {
            "sections": self.source_sections,
            "provenance": self.provenance,
        }
        self.source_json = canonical_json(source_contract)
        self.source_sections_json = canonical_json(self.source_sections)
        self.source_sha256 = sha256_text(self.source_json)
        self.prompt_sha256 = sha256_text(canonical_json(self.summary_prompt))
        if provider is None:
            if base_url is None or api_key is None:
                raise ValueError("summary provider requires base_url and api_key")
            provider = OpenAICompatibleChatProvider(
                provider_id,
                base_url,
                api_key,
                timeout_seconds=self.timeout_seconds,
                transport=transport,
            )
        self.provider = provider
        # Providers may expose the concrete ID (for example a fake test provider); the contract
        # should describe the configured provider, not an authorization value.
        self.provider_id = str(getattr(provider, "provider_id", provider_id))

    @classmethod
    def from_method(
        cls,
        method: object,
        *,
        provider: SummaryProvider | None = None,
        provider_id: str = "zoolab",
        model_id: str = SUMMARY_MODEL_ID,
        endpoint_type: str = SUMMARY_ENDPOINT_TYPE,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], str] | None = None,
    ) -> PaperSummaryService:
        """Build a service from a loaded ``PaperSummaryJailbreak``.

        Keeping this adapter duck-typed lets the low-level renderer remain independent from the
        provider while making the service straightforward to exercise with a fake transport.
        """

        summary_id = getattr(method, "summary_id", None)
        sections = getattr(method, "sections", None)
        provenance = getattr(method, "provenance", None)
        summary_prompt = getattr(method, "summary_prompt", None)
        if not isinstance(summary_id, str) or not isinstance(sections, Mapping):
            raise ValueError("method does not expose Paper Summary configuration")
        if not isinstance(provenance, Mapping) or not isinstance(summary_prompt, Mapping):
            raise ValueError("method does not expose Paper Summary provenance or prompt")
        source_language = str(provenance.get("source_language", "en"))
        if source_language not in sections or not isinstance(sections[source_language], Mapping):
            raise ValueError("Paper Summary source language sections are missing")
        return cls(
            summary_id=summary_id,
            source_sections={
                str(key): str(value) for key, value in sections[source_language].items()
            },
            provenance=provenance,
            summary_prompt={str(key): str(value) for key, value in summary_prompt.items()},
            provider=provider,
            provider_id=provider_id,
            model_id=model_id,
            endpoint_type=endpoint_type,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
            clock=clock,
        )

    def request_contract(self, language: str) -> dict[str, object]:
        if language not in SUMMARY_LANGUAGE_NAMES:
            raise ValueError(f"unsupported summary language: {language}")
        language_name = SUMMARY_LANGUAGE_NAMES[language]
        system_prompt = self.summary_prompt["system"]
        user_prompt = self.summary_prompt["user"].format(
            language_name=language_name,
            canonical_source_sections_json=self.source_sections_json,
        )
        # Keep this object independent of the API key and provider runtime IDs.  It is the exact
        # request contract whose hash is persisted in the artifact and run manifest.
        return {
            "summary_id": self.summary_id,
            "language": language,
            "language_name": language_name,
            "source_sha256": self.source_sha256,
            "source_sections": self.source_sections,
            "source_provenance": self.provenance,
            "source_sections_json": self.source_sections_json,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "endpoint_type": self.endpoint_type,
            "generation_config": self.generation_config,
        }

    def request_sha256(self, language: str) -> str:
        return sha256_text(canonical_json(self.request_contract(language)))

    @property
    def request_hashes(self) -> dict[str, str]:
        return {language: self.request_sha256(language) for language in SUMMARY_LANGUAGES}

    @property
    def contract(self) -> dict[str, object]:
        return {
            "summary_id": self.summary_id,
            "source_sha256": self.source_sha256,
            "prompt_sha256": self.prompt_sha256,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "endpoint_type": self.endpoint_type,
            "generation_config": self.generation_config,
            "request_sha256s": self.request_hashes,
        }

    def _request(self, language: str) -> GenerationRequest:
        contract = self.request_contract(language)
        request_hash = sha256_text(canonical_json(contract))
        return GenerationRequest(
            run_id=request_hash,
            experiment_id=f"psa-summary-{self.summary_id}",
            variant_id=f"summary-{language}",
            provider_id=self.provider_id,
            requested_model_id=self.model_id,
            endpoint_type=cast(Literal["chat", "completion"], self.endpoint_type),
            system_prompt=str(contract["system_prompt"]),
            rendered_prompt=str(contract["user_prompt"]),
            temperature=0.0,
            top_p=None,
            max_tokens=2048,
            seed=None,
            generation_config_hash=request_hash,
        )

    async def summarize_async(self, language: str) -> SummaryArtifact:
        if language not in SUMMARY_LANGUAGE_NAMES:
            raise ValueError(f"unsupported summary language: {language}")
        request = self._request(language)
        try:
            result = await self.provider.generate(request)
        except AuthenticationError:
            raise
        except Exception as error:
            # Do not include provider response bodies or authorization values in propagated errors.
            raise SummaryError("summary provider request failed") from error
        if result.status != "success":
            detail = result.error_type or result.status
            raise SummaryError(f"summary provider returned {detail}")
        response_text = result.response_text
        if not isinstance(response_text, str):
            raise SummaryError("summary provider returned no response text")
        normalized_response_text, _ = normalize_summary_response(response_text)
        return SummaryArtifact(
            summary_id=self.summary_id,
            language=language,
            source_sha256=self.source_sha256,
            request_sha256=request.run_id,
            provider_id=self.provider_id,
            model_id=self.model_id,
            endpoint_type=self.endpoint_type,
            generation_config=dict(self.generation_config),
            response_text=normalized_response_text,
            response_sha256=sha256_text(normalized_response_text),
            provider_request_id=result.provider_request_id,
            created_at=self._clock(),
        )

    def summarize(self, summary_id: str, language: str) -> SummaryArtifact:
        if summary_id != self.summary_id:
            raise ValueError(f"summary_id mismatch: expected {self.summary_id}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.summarize_async(language))
        raise RuntimeError("PaperSummaryService.summarize cannot run inside an event loop")

    def validate_artifacts(
        self, artifacts: Sequence[SummaryArtifact]
    ) -> dict[str, SummaryArtifact]:
        if len(artifacts) != len(SUMMARY_LANGUAGES):
            raise SummaryError("summary cache must contain exactly four language rows")
        by_language: dict[str, SummaryArtifact] = {}
        for artifact in artifacts:
            if artifact.language not in SUMMARY_LANGUAGE_NAMES:
                raise SummaryError(f"unsupported summary cache language: {artifact.language}")
            if artifact.language in by_language:
                raise SummaryError(f"duplicate summary cache language: {artifact.language}")
            if artifact.summary_id != self.summary_id:
                raise SummaryError("summary cache summary_id does not match the contract")
            expected = self.request_sha256(artifact.language)
            if artifact.request_sha256 != expected:
                raise SummaryError(f"summary cache request hash mismatch: {artifact.language}")
            if artifact.source_sha256 != self.source_sha256:
                raise SummaryError(f"summary cache source hash mismatch: {artifact.language}")
            if artifact.provider_id != self.provider_id or artifact.model_id != self.model_id:
                raise SummaryError(f"summary cache provider contract mismatch: {artifact.language}")
            if artifact.endpoint_type != self.endpoint_type:
                raise SummaryError(f"summary cache endpoint contract mismatch: {artifact.language}")
            if artifact.generation_config != self.generation_config:
                raise SummaryError(
                    f"summary cache generation contract mismatch: {artifact.language}"
                )
            artifact_sections(artifact)
            by_language[artifact.language] = artifact
        if set(by_language) != set(SUMMARY_LANGUAGES):
            raise SummaryError("summary cache must contain one row for each en, zh, vi, and my")
        return by_language

    def load_cache(self, path: Path) -> dict[str, SummaryArtifact]:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows: list[SummaryArtifact] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(SummaryArtifact.model_validate_json(line))
        except (OSError, ValueError) as error:
            raise SummaryError(f"invalid summary cache: {path}") from error
        return self.validate_artifacts(rows)

    def dump_cache(self, artifacts: Mapping[str, SummaryArtifact]) -> str:
        if set(artifacts) != set(SUMMARY_LANGUAGES):
            raise SummaryError("summary cache must contain exactly four language rows")
        ordered = self.validate_artifacts([artifacts[language] for language in SUMMARY_LANGUAGES])
        return "".join(
            canonical_json(ordered[language].model_dump(mode="json")) + "\n"
            for language in SUMMARY_LANGUAGES
        )

    def write_cache(self, path: Path, artifacts: Mapping[str, SummaryArtifact]) -> None:
        content = self.dump_cache(artifacts)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            if path.read_text(encoding="utf-8") != content:
                raise SummaryError(f"immutable summary cache conflict: {path}")
            return
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "SUMMARY_GENERATION_CONFIG",
    "SUMMARY_KEYS",
    "SUMMARY_LANGUAGES",
    "SUMMARY_LANGUAGE_NAMES",
    "SUMMARY_MODEL_ID",
    "SUMMARY_MIN_TIMEOUT_SECONDS",
    "PaperSummaryService",
    "SummaryArtifact",
    "SummaryError",
    "artifact_sections",
    "artifact_contract",
    "canonical_json",
    "parse_summary_response",
    "normalize_summary_response",
    "sha256_text",
]
