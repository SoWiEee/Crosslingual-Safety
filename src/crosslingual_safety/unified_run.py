"""The beginner-facing unified experiment facade.

The low-level translation, jailbreak, and generation commands remain useful for debugging and
advanced workflows.  This module deliberately owns the public contract and keeps the records it
emits separate from the legacy ``Manual*`` models: ``zh-tw`` is a public identifier while the
existing jailbreak templates continue to use their internal ``zh`` wrapper language.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NoReturn, cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from crosslingual_safety.generation.commands import generate_pending
from crosslingual_safety.generation.config import (
    ExperimentConfig,
    ExperimentPaths,
    ExperimentSection,
    GenerationConfig,
    ModelConfig,
)
from crosslingual_safety.generation.providers import ProviderAdapter
from crosslingual_safety.generation.queue import JobQueue
from crosslingual_safety.ids import canonicalize_text, stable_id
from crosslingual_safety.jailbreaks import (
    JailbreakContext,
    JailbreakMethod,
    JailbreakResult,
    PaperSummaryJailbreak,
    load_jailbreaks,
)
from crosslingual_safety.psa_papers import extract_paper, load_psa_papers
from crosslingual_safety.psa_summary import (
    SUMMARY_LANGUAGES,
    PaperSummaryService,
    SummaryArtifact,
    artifact_sections,
)
from crosslingual_safety.reporting import write_hierarchical_reports
from crosslingual_safety.schemas import GenerationRequest, GenerationResult, PromptVariant
from crosslingual_safety.translation.bench import (
    BenchTranslation,
    BenchTranslationCatalog,
    NativeBenchTranslator,
)
from crosslingual_safety.translation.languages import load_languages
from crosslingual_safety.translation.paid_ledger import (
    PaidCallLedger,
    PaidCallLedgerError,
    PaidTranslationTask,
    is_proven_preprocessing_rejection,
)
from crosslingual_safety.translation.providers import (
    DatasetTranslationProvider,
    FakeTranslator,
    GoogleCloudAuthenticationError,
    GoogleCloudConfigurationError,
    GoogleCloudIndeterminatePaidAttemptError,
    GoogleCloudInvalidRequestError,
    GoogleCloudNMTTranslator,
    GoogleCloudPermissionError,
    GoogleCloudProviderError,
    GoogleCloudQuotaError,
    GoogleCloudRequestTooLargeError,
    GoogleCloudReservationError,
    GoogleCloudRunBudgetExceededError,
    GoogleCloudTransientError,
    GoogleCloudTranslationClient,
    GoogleCloudTranslationResponseError,
    NLLBTranslator,
    ProviderTranslation,
    TranslationInputTooLongError,
    Translator,
)

PUBLIC_LANGUAGES: tuple[str, ...] = (
    "en",
    "zh-tw",
    "jv",
    "my",
    "th",
    "vi",
    "tl",
    "eo",
)
PUBLIC_JAILBREAKS: tuple[str, ...] = (
    "none",
    "psa_attack_poetry_v2",
    "psa_defense_r2d_v2",
)
LEGACY_JAILBREAKS: tuple[str, ...] = (
    "gra",
    "psa",
    "psa_attack_poetry_v1",
    "psa_defense_r2d_v1",
)
FORMAL_PSA_CONDITIONS: tuple[str, ...] = (
    "psa_attack_poetry_v1",
    "psa_defense_r2d_v1",
    "psa_attack_poetry_v2",
    "psa_defense_r2d_v2",
)
PUBLIC_SOURCES: tuple[str, ...] = ("manual", "bench")
LOCALIZED_WRAPPER_LANGUAGES: dict[str, str] = {
    "en": "en",
    "zh-tw": "zh",
    "vi": "vi",
    "my": "my",
    "eo": "eo",
}
WRAPPER_LANGUAGES: dict[str, str] = {
    **LOCALIZED_WRAPPER_LANGUAGES,
    "jv": "en",
    "th": "en",
    "tl": "en",
}
OUTPUT_LANGUAGE_NAMES: dict[str, str] = {
    "jv": "Javanese",
    "th": "Thai",
    "tl": "Tagalog",
}
FALLBACK_OUTPUT_INSTRUCTIONS: dict[str, tuple[str, str]] = {
    "gra": (
        "Every JSON string value must be in English.",
        "Every JSON string value must be in {language_name}.",
    ),
    "psa": (
        "Requested output language: English.",
        "Requested output language: {language_name}.",
    ),
}
ATTACK_IDS: dict[str, str] = {
    "none": "none",
    "gra": "gra_v1",
    "psa": "psa_static_v1",
    "psa_attack_poetry_v1": "psa_attack_poetry_v1",
    "psa_defense_r2d_v1": "psa_defense_r2d_v1",
    "psa_attack_poetry_v2": "psa_attack_poetry_v2",
    "psa_defense_r2d_v2": "psa_defense_r2d_v2",
}
SUMMARY_WRAPPER_LANGUAGES: tuple[str, ...] = ("en", "zh", "vi", "my", "eo")
GOOGLE_CLOUD_TRANSLATOR = "google-cloud-nmt-v3"
GOOGLE_CLOUD_INDETERMINATE_ERROR = "GoogleCloudIndeterminatePaidAttemptError"
GOOGLE_CLOUD_INDETERMINATE_MESSAGE = (
    "Google Cloud Translation paid attempt outcome is indeterminate; manual review is required"
)
DEFAULT_GOOGLE_CLOUD_SETTINGS: dict[str, object] = {
    "project_id": "gen-lang-client-0036391889",
    "location": "global",
    "model": "general/nmt",
    "max_request_characters": 5000,
    "max_run_characters": 100000,
}
SANITIZED_ERROR_MESSAGES: dict[str, str] = {
    "GoogleCloudAuthenticationError": "Google Cloud Translation authentication failed",
    "GoogleCloudPermissionError": "Google Cloud Translation permission was denied",
    "GoogleCloudQuotaError": "Google Cloud Translation quota was exceeded",
    "GoogleCloudInvalidRequestError": "Google Cloud Translation rejected the request",
    "GoogleCloudTransientError": "Google Cloud Translation is temporarily unavailable",
    "GoogleCloudTranslationResponseError": (
        "Google Cloud Translation returned an unusable response"
    ),
    "GoogleCloudReservationError": (
        "Google Cloud Translation paid-call reservation could not be persisted"
    ),
    GOOGLE_CLOUD_INDETERMINATE_ERROR: GOOGLE_CLOUD_INDETERMINATE_MESSAGE,
    "GoogleCloudProviderError": "Google Cloud Translation provider request failed",
    "GoogleCloudConfigurationError": "Google Cloud Translation configuration is invalid",
    "GoogleCloudRequestTooLargeError": (
        "Google Cloud Translation request character limit exceeded"
    ),
    "GoogleCloudRunBudgetExceededError": ("Google Cloud Translation run character budget exceeded"),
    "TranslationInputTooLongError": "Translation input exceeds the configured token limit",
    "UnexpectedOperationError": "An unexpected operation failed",
}
SANITIZED_ERROR_TYPES: dict[type[BaseException], str] = {
    GoogleCloudAuthenticationError: "GoogleCloudAuthenticationError",
    GoogleCloudPermissionError: "GoogleCloudPermissionError",
    GoogleCloudQuotaError: "GoogleCloudQuotaError",
    GoogleCloudInvalidRequestError: "GoogleCloudInvalidRequestError",
    GoogleCloudTransientError: "GoogleCloudTransientError",
    GoogleCloudTranslationResponseError: "GoogleCloudTranslationResponseError",
    GoogleCloudReservationError: "GoogleCloudReservationError",
    GoogleCloudIndeterminatePaidAttemptError: GOOGLE_CLOUD_INDETERMINATE_ERROR,
    GoogleCloudProviderError: "GoogleCloudProviderError",
    GoogleCloudConfigurationError: "GoogleCloudConfigurationError",
    GoogleCloudRequestTooLargeError: "GoogleCloudRequestTooLargeError",
    GoogleCloudRunBudgetExceededError: "GoogleCloudRunBudgetExceededError",
    TranslationInputTooLongError: "TranslationInputTooLongError",
}
PROVEN_PAID_REJECTION_ERROR_TYPES = frozenset(
    {
        "GoogleCloudAuthenticationError",
        "GoogleCloudPermissionError",
        "GoogleCloudQuotaError",
        "GoogleCloudInvalidRequestError",
    }
)
TRANSLATION_ATTEMPT_FIELDS = frozenset(
    {
        "attempt_id",
        "attempt_number",
        "case_id",
        "status",
        "source",
        "source_language",
        "target_language",
        "provider",
        "source_character_count",
        "source_text_sha256",
        "charged_character_count",
        "provider_reservation_id",
        "error_type",
        "error_message",
        "created_at",
    }
)
PAID_ATTEMPT_LINK_FIELDS = frozenset(
    {
        "audit_reference",
        "task_key",
        "provider_contract_sha256",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sanitized_error(error: BaseException) -> tuple[str, str]:
    """Return only allowlisted project failures or a fixed fail-closed fallback."""

    error_type = SANITIZED_ERROR_TYPES.get(type(error), "UnexpectedOperationError")
    return error_type, SANITIZED_ERROR_MESSAGES[error_type]


def parse_selection(value: str, allowed: tuple[str, ...], option_name: str) -> tuple[str, ...]:
    """Normalize a comma-separated public selection.

    ``all`` expands in the canonical order supplied by ``allowed``.  Every other selection is
    validated before returning, and duplicate values are removed while retaining canonical order.
    """

    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{option_name} must not be empty")
    if "all" in values:
        if len(values) != 1:
            raise ValueError(f"{option_name}=all cannot be combined with other values")
        return tuple(allowed)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"{option_name} must be one of {', '.join(allowed)}")
    selected = set(values)
    return tuple(item for item in allowed if item in selected)


def parse_jailbreak_selection(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if values == ("all",):
        return PUBLIC_JAILBREAKS
    allowed = PUBLIC_JAILBREAKS + LEGACY_JAILBREAKS
    return parse_selection(value, allowed, "--jailbreak")


def _wrapper_language(language: str, jailbreak: str) -> str:
    return language if jailbreak in FORMAL_PSA_CONDITIONS else WRAPPER_LANGUAGES[language]


class ManualSettings(BaseModel):
    input_path: Path = Path("prompts/prompt.txt")
    source_language: str = "zh-tw"


class BenchSettings(BaseModel):
    cases_path: Path = Path("data/normalized/cases.parquet")
    selection_path: Path = Path("data/normalized/variant_case_selection.parquet")
    native_translations_path: Path = Path("data/normalized/native_translations.parquet")
    datasets: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoogleCloudSettings:
    project_id: str
    location: str
    model: str
    max_request_characters: int
    max_run_characters: int

    def contract(self, client_library_version: str) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model": self.model,
            "client_library": "google-cloud-translate",
            "client_library_version": client_library_version,
            "language_codes": dict(GoogleCloudNMTTranslator.language_codes),
            "mime_type": "text/plain",
            "use_language_detection": False,
            "max_request_characters": self.max_request_characters,
            "max_run_characters": self.max_run_characters,
        }


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    manual: ManualSettings = Field(default_factory=ManualSettings)
    bench: BenchSettings = Field(default_factory=BenchSettings)
    models: list[str] = Field(
        default_factory=lambda: [
            "llama31_8b",
            "gemma_4_12b",
            "gemma_4_26b",
            "nemotron_cascade_2_30b",
            "llama33_70b",
        ]
    )
    translator: str = "nllb"
    google_cloud: object = Field(default_factory=lambda: dict(DEFAULT_GOOGLE_CLOUD_SETTINGS))
    wrapper_language_mode: Literal["same-as-payload", "english"] = "same-as-payload"
    gra_role: str = "joker"
    models_config: Path = Path("configs/models.yaml")
    languages_config: Path = Path("configs/languages.yaml")
    jailbreaks_config: Path = Path("configs/jailbreaks.yaml")
    psa_papers_config: Path = Path("configs/psa_papers.yaml")
    runs_dir: Path = Path("runs/experiments")
    temperature: float = 1.0
    top_p: float | None = None
    max_tokens: int = Field(default=4096, gt=0)
    seed: int | None = None
    retry_backoff_base: float = 1.0
    config_path: Path | None = Field(default=None, exclude=True)
    nllb_checkpoint: str = "facebook/nllb-200-distilled-600M"
    nllb_local_files_only: bool = True


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["manual", "bench"] = "manual"
    languages: tuple[str, ...] = PUBLIC_LANGUAGES
    jailbreaks: tuple[str, ...] = ("none",)
    models: tuple[str, ...] = ("all",)
    dry_run: bool = False

    @field_validator("languages", "jailbreaks", "models", mode="before")
    @classmethod
    def split_public_selection(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(","))
        return value


class UnifiedCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    source: Literal["manual", "bench"]
    source_language: str
    source_text: str
    category: str | None = None
    intent: str = "harmful"
    dataset: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def canonical_payload(self) -> str:
        return self.source_text


class RunPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    request: RunRequest
    cases: tuple[UnifiedCase, ...]
    models: tuple[str, ...]
    languages: tuple[str, ...]
    jailbreaks: tuple[str, ...]
    translation_jobs: int
    psa_summary_count: int
    psa_localization_count: int = 0
    victim_request_count: int
    run_id: str
    parent_path: Path
    input_snapshot_sha256: str
    contract: dict[str, object]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def translation_count(self) -> int:
        return self.translation_jobs

    @property
    def summary_count(self) -> int:
        return self.psa_summary_count

    @property
    def selected_cases(self) -> tuple[UnifiedCase, ...]:
        return self.cases

    @property
    def selected_models(self) -> tuple[str, ...]:
        return self.models

    @property
    def prospective_parent_path(self) -> Path:
        return self.parent_path

    @property
    def victim_requests(self) -> int:
        return self.victim_request_count


class ContractConflictError(ValueError):
    """An existing parent or child has a different immutable run contract."""


@dataclass
class RunDependencies:
    """Injectable runtime seams used by tests and offline experiments.

    ``translator``, ``summary_service``, and ``generation`` are the simple fakes most callers use.
    Factories are available when a test needs to inspect the selected settings before constructing
    a fake.  No factory is called by ``plan_run`` or a dry run.
    """

    translator: Translator | None = None
    summary_service: Any | None = None
    generation: Callable[..., Any] | None = None
    translator_factory: Callable[..., Translator] | None = None
    summary_service_factory: Callable[..., Any] | None = None
    provider_factory: Callable[..., ProviderAdapter] | None = None
    google_translation_client: GoogleCloudTranslationClient | None = None
    google_adc_preflight: Callable[[str], None] | None = None
    google_client_library_version: str | None = None
    clock: Callable[[], str] = _utc_now
    emit: Callable[[str], None] = print
    generate: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if self.generation is None:
            self.generation = self.generate
        if self.generate is None:
            self.generate = self.generation


@dataclass
class RunExecution:
    run_id: str
    status: str
    parent_path: Path
    results_path: Path
    manifest: dict[str, object]
    rows: list[dict[str, object]]
    child_statuses: dict[str, str]


def _resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Versioned configs in this repository use project-root-relative paths.  Temporary test
    # configs use paths relative to their own directory.
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return (root / path).resolve()


def load_run_settings(path: Path = Path("configs/run.yaml")) -> RunSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid run configuration: {path}")
    value = dict(raw)
    manual = dict(value.get("manual") or {})
    bench = dict(value.get("bench") or {})
    manual["input_path"] = _resolve_config_path(
        manual.get("input_path", "prompts/prompt.txt"), path
    )
    bench["cases_path"] = _resolve_config_path(
        bench.get("cases_path", "data/normalized/cases.parquet"), path
    )
    bench["selection_path"] = _resolve_config_path(
        bench.get("selection_path", "data/normalized/variant_case_selection.parquet"), path
    )
    bench["native_translations_path"] = _resolve_config_path(
        bench.get(
            "native_translations_path",
            "data/normalized/native_translations.parquet",
        ),
        path,
    )
    value["manual"] = manual
    value["bench"] = bench
    for key, default in (
        ("models_config", "configs/models.yaml"),
        ("languages_config", "configs/languages.yaml"),
        ("jailbreaks_config", "configs/jailbreaks.yaml"),
        ("psa_papers_config", "configs/psa_papers.yaml"),
        ("runs_dir", "runs/experiments"),
    ):
        value[key] = _resolve_config_path(value.get(key, default), path)
    settings = RunSettings.model_validate({**value, "config_path": path.resolve()})
    if settings.manual.source_language not in PUBLIC_LANGUAGES:
        raise ValueError(f"manual.source_language must be one of: {', '.join(PUBLIC_LANGUAGES)}")
    return settings


def _public_language(value: str) -> str:
    return "zh-tw" if value == "zh" else value


def _load_manual_cases(settings: RunSettings) -> tuple[UnifiedCase, ...]:
    path = settings.manual.input_path
    configured_source_language = _public_language(settings.manual.source_language)
    if configured_source_language not in PUBLIC_LANGUAGES:
        raise ValueError(f"invalid manual source language: {configured_source_language}")
    if not path.is_file():
        raise ValueError(f"manual input does not exist: {path}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"manual input must be UTF-8: {path}") from error
    if path.suffix.lower() == ".txt":
        prompt = canonicalize_text(text)
        if not prompt:
            raise ValueError("manual input is empty")
        return (
            UnifiedCase(
                case_id=stable_id("manual-prompt", prompt),
                source="manual",
                source_language=configured_source_language,
                source_text=prompt,
                intent="harmful",
            ),
        )
    if path.suffix.lower() != ".jsonl":
        raise ValueError("manual input must use .txt or .jsonl")
    cases: list[UnifiedCase] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid manual prompt on line {line_number}")
        prompt = canonicalize_text(str(value.get("prompt", value.get("source_text", ""))))
        if not prompt:
            raise ValueError(f"invalid manual prompt on line {line_number}: empty prompt")
        source_language = _public_language(
            str(value.get("source_language", configured_source_language))
        )
        if source_language not in PUBLIC_LANGUAGES:
            raise ValueError(f"invalid manual source language: {source_language}")
        case_id = str(value.get("prompt_id", value.get("case_id", ""))).strip()
        case_id = case_id or stable_id("manual-prompt", prompt)
        cases.append(
            UnifiedCase(
                case_id=case_id,
                source="manual",
                source_language=source_language,
                source_text=prompt,
                category=value.get("category"),
                intent=str(value.get("intent", "harmful")),
                metadata={
                    key: item
                    for key, item in value.items()
                    if key not in {"prompt", "source_text", "source_language"}
                },
            )
        )
    if not cases:
        raise ValueError("manual JSONL input contains no prompts")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate manual case_id")
    return tuple(cases)


def _load_bench_cases(settings: RunSettings) -> tuple[UnifiedCase, ...]:
    if not settings.bench.cases_path.is_file():
        raise ValueError(f"benchmark cases do not exist: {settings.bench.cases_path}")
    if not settings.bench.selection_path.is_file():
        raise ValueError(f"benchmark selection does not exist: {settings.bench.selection_path}")
    rows = pq.read_table(settings.bench.cases_path).to_pylist()
    by_id = {str(row.get("case_id")): row for row in rows}
    selected_rows = pq.read_table(settings.bench.selection_path).to_pylist()
    if settings.bench.datasets:
        available_datasets = {str(row.get("dataset")) for row in selected_rows}
        unknown_datasets = sorted(set(settings.bench.datasets) - available_datasets)
        if unknown_datasets:
            raise ValueError(f"benchmark datasets are unavailable: {', '.join(unknown_datasets)}")
        selected_datasets = set(settings.bench.datasets)
        selected_rows = [
            row for row in selected_rows if str(row.get("dataset")) in selected_datasets
        ]
    selected_ids = [str(row["selected_case_id"]) for row in selected_rows]
    missing = sorted(set(selected_ids) - by_id.keys())
    if missing:
        raise ValueError(f"benchmark selection references unknown cases: {', '.join(missing)}")
    cases: list[UnifiedCase] = []
    seen: set[str] = set()
    for case_id in selected_ids:
        if case_id in seen:
            continue
        seen.add(case_id)
        row = by_id[case_id]
        source_text = canonicalize_text(
            str(row.get("canonical_payload", row.get("source_text", "")))
        )
        if not source_text:
            raise ValueError(f"benchmark case {case_id} has empty payload")
        cases.append(
            UnifiedCase(
                case_id=case_id,
                source="bench",
                source_language=_public_language(str(row.get("source_language", "en"))),
                source_text=source_text,
                category=row.get("category"),
                intent=str(row.get("intent", "harmful")),
                dataset=row.get("dataset"),
                metadata={
                    key: value
                    for key, value in row.items()
                    if key not in {"canonical_payload", "source_text", "source_language"}
                },
            )
        )
    if not cases:
        raise ValueError("benchmark selection contains no cases")
    return tuple(cases)


def _load_cases(request: RunRequest, settings: RunSettings) -> tuple[UnifiedCase, ...]:
    return (
        _load_manual_cases(settings) if request.source == "manual" else _load_bench_cases(settings)
    )


def _bench_translation_catalog(
    request: RunRequest, settings: RunSettings
) -> BenchTranslationCatalog | None:
    if request.source != "bench":
        return None
    return BenchTranslationCatalog.from_parquet(settings.bench.native_translations_path)


def _native_bench_translation(
    catalog: BenchTranslationCatalog | None,
    case: UnifiedCase,
    language: str,
) -> BenchTranslation | None:
    if catalog is None or case.source != "bench":
        return None
    return catalog.resolve(
        case_id=case.case_id,
        source_text=case.source_text,
        target_language=language,
    )


def _file_sha256(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def _raw_models(settings: RunSettings) -> dict[str, dict[str, object]]:
    if not settings.models_config.is_file():
        return {}
    raw = yaml.safe_load(settings.models_config.read_text(encoding="utf-8")) or {}
    models = raw.get("models", raw) if isinstance(raw, dict) else {}
    return {str(name): dict(value) for name, value in models.items() if isinstance(value, dict)}


def _google_cloud_settings(settings: RunSettings) -> GoogleCloudSettings:
    if not isinstance(settings.google_cloud, dict) or not all(
        isinstance(key, str) for key in settings.google_cloud
    ):
        raise GoogleCloudConfigurationError(
            "Google Cloud Translation configuration must be a mapping"
        )
    configured = {
        str(key): value for key, value in cast(dict[object, object], settings.google_cloud).items()
    }
    unknown = set(configured) - set(DEFAULT_GOOGLE_CLOUD_SETTINGS)
    if unknown:
        raise GoogleCloudConfigurationError(
            "Google Cloud Translation configuration contains unsupported settings"
        )
    values = {**DEFAULT_GOOGLE_CLOUD_SETTINGS, **configured}
    project_id = values["project_id"]
    location = values["location"]
    model = values["model"]
    max_request_characters = values["max_request_characters"]
    max_run_characters = values["max_run_characters"]
    if not isinstance(project_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project_id
    ):
        raise GoogleCloudConfigurationError("Google Cloud project ID is invalid")
    if location != "global":
        raise GoogleCloudConfigurationError("Google Cloud Translation location must be global")
    if model != "general/nmt":
        raise GoogleCloudConfigurationError("Google Cloud Translation model must be general/nmt")
    if (
        isinstance(max_request_characters, bool)
        or not isinstance(max_request_characters, int)
        or max_request_characters <= 0
    ):
        raise GoogleCloudConfigurationError(
            "Google Cloud request character limit must be a positive integer"
        )
    if (
        isinstance(max_run_characters, bool)
        or not isinstance(max_run_characters, int)
        or max_run_characters <= 0
    ):
        raise GoogleCloudConfigurationError(
            "Google Cloud run character limit must be a positive integer"
        )
    if max_request_characters > max_run_characters:
        raise GoogleCloudConfigurationError(
            "Google Cloud request character limit must not exceed the run limit"
        )
    return GoogleCloudSettings(
        project_id=project_id,
        location=location,
        model=model,
        max_request_characters=max_request_characters,
        max_run_characters=max_run_characters,
    )


def _google_client_library_version() -> str:
    try:
        return package_version("google-cloud-translate")
    except PackageNotFoundError:
        return "not-installed"


def _translator_contract(settings: RunSettings) -> dict[str, object]:
    if settings.translator == GOOGLE_CLOUD_TRANSLATOR:
        return _google_cloud_settings(settings).contract(_google_client_library_version())
    if settings.translator == "nllb":
        return {
            "provider": "nllb",
            "checkpoint": settings.nllb_checkpoint,
            "local_files_only": settings.nllb_local_files_only,
        }
    return {"provider": settings.translator}


def plan_run(request: RunRequest, settings: RunSettings) -> RunPlan:
    if request.source not in PUBLIC_SOURCES:
        raise ValueError("source must be manual or bench")
    languages = parse_selection(",".join(request.languages), PUBLIC_LANGUAGES, "--language")
    jailbreaks = parse_jailbreak_selection(",".join(request.jailbreaks))
    requested_models = [model.strip() for model in request.models if model.strip()]
    if len(requested_models) != len(set(requested_models)):
        raise ValueError("--model must not contain duplicate values")
    model_names = parse_selection(",".join(requested_models), tuple(settings.models), "--model")
    if "zh" in languages:
        raise ValueError("--language uses zh-tw for Traditional Chinese")
    normalized_request = request.model_copy(
        update={"languages": languages, "jailbreaks": jailbreaks, "models": model_names}
    )
    cases = _load_cases(normalized_request, settings)
    native_catalog = _bench_translation_catalog(normalized_request, settings)
    translation_jobs = sum(
        1
        for case in cases
        for language in languages
        if language != case.source_language
        and _native_bench_translation(native_catalog, case, language) is None
    )
    formal_psa = tuple(item for item in jailbreaks if item in FORMAL_PSA_CONDITIONS)
    psa_summary_count = len(formal_psa) + (len(SUMMARY_LANGUAGES) if "psa" in jailbreaks else 0)
    psa_localization_count = len(formal_psa) * (len(PUBLIC_LANGUAGES) - 1)
    victim_request_count = len(cases) * len(languages) * len(jailbreaks) * len(model_names)
    input_snapshot = "".join(_canonical_json(case.model_dump(mode="json")) + "\n" for case in cases)
    input_snapshot_sha256 = _sha256_text(input_snapshot)
    models = _raw_models(settings)
    selected_models = {name: models.get(name) for name in model_names}
    translator_contract = _translator_contract(settings)
    paper_contracts: dict[str, object] = {}
    if formal_psa:
        paper_specs = load_psa_papers(settings.psa_papers_config)
        for condition in formal_psa:
            spec = paper_specs.get(condition)
            if spec is None:
                raise ValueError(f"PSA paper condition is missing: {condition}")
            extracted = extract_paper(spec)
            paper_contracts[condition] = {
                "summary_id": spec.summary_id,
                "title": spec.title,
                "source_path": spec.source_path.as_posix(),
                "source_sha256": extracted.source_sha256,
                "text_sha256": extracted.text_sha256,
                "page_count": extracted.page_count,
                "chunk_sha256s": [chunk.text_sha256 for chunk in extracted.chunks],
                "summarizer_model": spec.summarizer_model,
            }
    contract: dict[str, object] = {
        "version": settings.version,
        "request": normalized_request.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
        "input_snapshot_sha256": input_snapshot_sha256,
        "models": selected_models,
        "model_names": list(model_names),
        "translator": settings.translator,
        "translator_contract": translator_contract,
        "nllb_checkpoint": settings.nllb_checkpoint,
        "nllb_local_files_only": settings.nllb_local_files_only,
        "wrapper_language_mode": settings.wrapper_language_mode,
        "gra_role": settings.gra_role,
        "generation": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "seed": settings.seed,
            "retry_backoff_base": settings.retry_backoff_base,
        },
        "config_hashes": {
            "run": _file_sha256(settings.config_path or Path("configs/run.yaml")),
            "models": _file_sha256(settings.models_config),
            "languages": _file_sha256(settings.languages_config),
            "jailbreaks": _file_sha256(settings.jailbreaks_config),
            "psa_papers": _file_sha256(settings.psa_papers_config),
            "native_translations": (
                _file_sha256(settings.bench.native_translations_path)
                if normalized_request.source == "bench"
                else None
            ),
        },
        "attack_ids": {name: ATTACK_IDS[name] for name in jailbreaks},
        "psa_papers": paper_contracts,
    }
    run_id = stable_id("experiment-run", _canonical_json(contract))
    return RunPlan(
        request=normalized_request,
        cases=cases,
        models=model_names,
        languages=languages,
        jailbreaks=jailbreaks,
        translation_jobs=translation_jobs,
        psa_summary_count=psa_summary_count,
        psa_localization_count=psa_localization_count,
        victim_request_count=victim_request_count,
        run_id=run_id,
        parent_path=settings.runs_dir / run_id,
        input_snapshot_sha256=input_snapshot_sha256,
        contract=contract,
    )


def _load_model_configs(settings: RunSettings) -> dict[str, ModelConfig]:
    raw = _raw_models(settings)
    missing = [name for name in settings.models if name not in raw]
    if missing:
        raise ValueError(f"unknown models: {', '.join(missing)}")
    return {name: ModelConfig.model_validate(raw[name]) for name in settings.models}


_NLLB_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.model",
    "source.spm",
    "vocab.json",
)
_NLLB_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _usable_nllb_snapshot(path: Path) -> bool:
    """Return whether ``path`` contains the local files needed by ``NLLBTranslator``."""

    if not path.is_dir() or not _nonempty_file(path / "config.json"):
        return False
    if not any(_nonempty_file(path / name) for name in _NLLB_TOKENIZER_ASSETS):
        return False
    try:
        return any(
            child.is_file()
            and child.name.endswith(_NLLB_WEIGHT_SUFFIXES)
            and child.stat().st_size > 0
            for child in path.iterdir()
        )
    except OSError:
        return False


def _nllb_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for variable in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            root = Path(value).expanduser()
            roots.append(root / "hub" if variable == "HF_HOME" else root)
    roots.extend(
        [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".cache" / "huggingface" / "transformers",
        ]
    )
    return roots


def _nllb_snapshot_candidates(repository: Path) -> Iterable[Path]:
    """Yield a cache repository's referenced and discovered snapshot directories."""

    snapshots = repository / "snapshots"
    refs = repository / "refs"
    if refs.is_dir() and snapshots.is_dir():
        try:
            for ref in sorted(refs.iterdir()):
                if not ref.is_file():
                    continue
                revision = ref.read_text(encoding="utf-8").strip()
                if revision:
                    yield snapshots / revision
        except OSError:
            return
    if snapshots.is_dir():
        try:
            yield from (
                candidate for candidate in sorted(snapshots.iterdir()) if candidate.is_dir()
            )
        except OSError:
            return


def _nllb_checkpoint_candidates(settings: RunSettings) -> Iterable[Path]:
    checkpoint = Path(settings.nllb_checkpoint).expanduser()
    local_candidates = [checkpoint]
    if settings.config_path is not None and not checkpoint.is_absolute():
        local_candidates.extend(
            [
                settings.config_path.parent / checkpoint,
                settings.config_path.parent.parent / checkpoint,
            ]
        )
    seen: set[Path] = set()
    for candidate in local_candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved
        yield from _nllb_snapshot_candidates(resolved)
    if not settings.nllb_local_files_only:
        return
    repository_cache_name = f"models--{settings.nllb_checkpoint.replace('/', '--')}"
    for root in _nllb_cache_roots():
        repository = (root / repository_cache_name).resolve()
        if repository in seen:
            continue
        seen.add(repository)
        yield from _nllb_snapshot_candidates(repository)


def _find_nllb_snapshot(settings: RunSettings) -> Path | None:
    for candidate in _nllb_checkpoint_candidates(settings):
        if _usable_nllb_snapshot(candidate):
            return candidate
    return None


def _nllb_checkpoint_available(settings: RunSettings) -> bool:
    """Check for a usable local NLLB snapshot without importing model runtimes."""

    return _find_nllb_snapshot(settings) is not None


def _resolved_nllb_checkpoint(settings: RunSettings) -> str:
    snapshot = _find_nllb_snapshot(settings)
    return str(snapshot) if snapshot is not None else settings.nllb_checkpoint


def _default_google_adc_preflight(project_id: str) -> None:
    configured_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not configured_project:
        raise GoogleCloudAuthenticationError(
            "GOOGLE_CLOUD_PROJECT is required for Google Cloud Translation"
        )
    if configured_project != project_id:
        raise GoogleCloudAuthenticationError(
            "GOOGLE_CLOUD_PROJECT must match the configured translation project"
        )
    credential_value = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    credential_path = Path(credential_value) if credential_value else None
    if credential_path is None or not credential_path.is_absolute():
        raise GoogleCloudAuthenticationError(
            "GOOGLE_APPLICATION_CREDENTIALS must reference an absolute local file"
        )
    if not credential_path.is_file():
        raise GoogleCloudAuthenticationError(
            "Google Cloud application default credential file does not exist"
        )
    try:
        google_auth = import_module("google.auth")
        import_module("google.cloud.translate_v3")
    except (ImportError, ModuleNotFoundError):
        raise GoogleCloudConfigurationError(
            "Google translation support is not installed; run `uv sync --extra translation-google`."
        ) from None
    try:
        credentials, _ = google_auth.default(
            scopes=("https://www.googleapis.com/auth/cloud-platform",),
            quota_project_id=project_id,
        )
    except Exception:
        raise GoogleCloudAuthenticationError(
            "Google Cloud application default credentials are unavailable or invalid"
        ) from None
    if credentials is None:
        raise GoogleCloudAuthenticationError(
            "Google Cloud application default credentials are unavailable or invalid"
        )


def _preflight_google_cloud(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies,
) -> None:
    google = _google_cloud_settings(settings)
    total_characters = 0
    for case in plan.cases:
        for target_language in plan.languages:
            if case.source_language == target_language:
                continue
            source = GoogleCloudNMTTranslator.language_codes.get(case.source_language)
            target = GoogleCloudNMTTranslator.language_codes.get(target_language)
            if source is None or target is None or source == target:
                raise GoogleCloudConfigurationError(
                    "Google Cloud Translation does not support a selected language pair"
                )
            character_count = len(case.source_text)
            if character_count > google.max_request_characters:
                raise GoogleCloudRequestTooLargeError(
                    "Google Cloud Translation request character limit exceeded"
                )
            total_characters += character_count
            if total_characters > google.max_run_characters:
                raise GoogleCloudRunBudgetExceededError(
                    "Google Cloud Translation run character budget exceeded"
                )
    adc_preflight = dependencies.google_adc_preflight or _default_google_adc_preflight
    try:
        adc_preflight(google.project_id)
    except (GoogleCloudAuthenticationError, GoogleCloudConfigurationError):
        raise
    except Exception:
        raise GoogleCloudAuthenticationError(
            "Google Cloud application default credentials are unavailable or invalid"
        ) from None
    if dependencies.google_translation_client is None:
        try:
            dependencies.google_translation_client = GoogleCloudNMTTranslator._default_client()
        except (GoogleCloudAuthenticationError, GoogleCloudConfigurationError):
            raise
        except Exception:
            raise GoogleCloudAuthenticationError(
                "Google Cloud application default credentials are unavailable or invalid"
            ) from None


def preflight_run(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies | None = None,
) -> None:
    """Validate every selected input and runtime contract before making a provider call."""

    dependencies = dependencies or RunDependencies()
    if not plan.cases:
        raise ValueError("run plan contains no cases")
    if plan.request.source == "manual":
        if not settings.manual.input_path.is_file():
            raise ValueError(f"manual input does not exist: {settings.manual.input_path}")
    else:
        if (
            not settings.bench.cases_path.is_file()
            or not settings.bench.selection_path.is_file()
            or not settings.bench.native_translations_path.is_file()
        ):
            raise ValueError(
                "benchmark cases, selection, and native translation snapshots are required"
            )
    model_configs = _load_model_configs(settings)
    languages = load_languages(settings.languages_config)
    for language in plan.languages:
        if language not in PUBLIC_LANGUAGES:
            raise ValueError(f"unsupported target language: {language}")
        if language not in languages:
            raise ValueError(f"language is missing from config: {language}")
    methods = load_jailbreaks(settings.jailbreaks_config)
    for jailbreak in plan.jailbreaks:
        method = methods.get(ATTACK_IDS[jailbreak])
        if method is None:
            raise ValueError(f"jailbreak is missing from config: {ATTACK_IDS[jailbreak]}")
        for language in plan.languages:
            wrapper_language = _wrapper_language(language, jailbreak)
            if not method.supports_language(wrapper_language):
                raise ValueError(
                    f"{jailbreak} does not support wrapper language {wrapper_language}"
                )
        if jailbreak == "gra":
            personas = getattr(method, "personas", {})
            if settings.gra_role not in personas:
                raise ValueError(f"unknown GRA role: {settings.gra_role}")
        if (jailbreak == "psa" or jailbreak in FORMAL_PSA_CONDITIONS) and not isinstance(
            method, PaperSummaryJailbreak
        ):
            raise ValueError("psa configuration must use PaperSummaryJailbreak")

    if (
        plan.translation_jobs
        and dependencies.translator is None
        and dependencies.translator_factory is None
    ):
        if settings.translator == "nllb":
            if not _nllb_checkpoint_available(settings):
                raise ValueError(
                    f"NLLB checkpoint is not available locally: {settings.nllb_checkpoint}"
                )
            try:
                import torch
            except ImportError as error:
                raise ValueError("NLLB support is not installed") from error
            if not torch.cuda.is_available():
                raise ValueError("CUDA is required for NLLB translation but is unavailable")
        elif settings.translator == GOOGLE_CLOUD_TRANSLATOR:
            _preflight_google_cloud(plan, settings, dependencies)
        elif settings.translator not in {"fake", "dataset"}:
            raise ValueError(f"unsupported translator: {settings.translator}")

    # A fake generation seam is sufficient for tests and intentionally bypasses credential checks,
    # but endpoint metadata remains part of the selected configuration contract.
    selected_model_configs = [model_configs[name] for name in plan.models]
    for model in selected_model_configs:
        if model.provider == "fake":
            if not model.test_only:
                raise ValueError("FakeProvider requires test_only=true")
            continue
        if not model.base_url_env or not model.api_key_env:
            raise ValueError(f"provider {model.provider} requires endpoint metadata")
    if dependencies.generation is None and dependencies.provider_factory is None:
        for model in selected_model_configs:
            if model.provider == "fake":
                continue
            assert model.base_url_env is not None and model.api_key_env is not None
            if not os.environ.get(model.base_url_env) or not os.environ.get(model.api_key_env):
                raise ValueError(
                    f"required provider environment variable is unset: {model.base_url_env} or "
                    f"{model.api_key_env}"
                )
    if (
        any(item == "psa" or item in FORMAL_PSA_CONDITIONS for item in plan.jailbreaks)
        and dependencies.summary_service is None
        and dependencies.summary_service_factory is None
    ):
        summary_model = model_configs.get("gemma_4_12b")
        if summary_model is None:
            raise ValueError("PSA summary requires gemma_4_12b model configuration")
        if summary_model.provider != "fake":
            if not summary_model.base_url_env or not summary_model.api_key_env:
                raise ValueError("PSA summary provider requires endpoint metadata")
            if not os.environ.get(summary_model.base_url_env) or not os.environ.get(
                summary_model.api_key_env
            ):
                raise ValueError(
                    "required provider environment variable is unset: "
                    f"{summary_model.base_url_env} or "
                    f"{summary_model.api_key_env}"
                )


def _make_translator(
    settings: RunSettings,
    dependencies: RunDependencies,
    *,
    initial_google_characters: int = 0,
    google_paid_call_reservation: Callable[[int], None] | None = None,
) -> Translator | None:
    if dependencies.translator is not None:
        return dependencies.translator
    if dependencies.translator_factory is not None:
        return dependencies.translator_factory(settings)
    if settings.translator == "fake":
        return FakeTranslator()
    if settings.translator == "nllb":
        return NLLBTranslator(
            load_languages(settings.languages_config),
            checkpoint=_resolved_nllb_checkpoint(settings),
            local_files_only=True,
        )
    if settings.translator == GOOGLE_CLOUD_TRANSLATOR:
        google = _google_cloud_settings(settings)
        return GoogleCloudNMTTranslator(
            project_id=google.project_id,
            location=google.location,
            model=google.model,
            client=dependencies.google_translation_client,
            client_library_version=(
                dependencies.google_client_library_version or _google_client_library_version()
            ),
            max_request_characters=google.max_request_characters,
            max_run_characters=google.max_run_characters,
            initial_characters_used=initial_google_characters,
            paid_call_reservation=google_paid_call_reservation,
        )
    if settings.translator == "dataset":
        path = Path("data/normalized/native_translations.parquet")
        if not path.is_file():
            raise ValueError(f"native translations file does not exist: {path}")
        rows = pq.read_table(path).to_pylist()
        return DatasetTranslationProvider(
            {
                (str(row["source_text"]), str(row["language"])): str(row["translated_text"])
                for row in rows
            }
        )
    raise ValueError(f"unsupported translator: {settings.translator}")


def _make_summary_service(
    settings: RunSettings,
    dependencies: RunDependencies,
    method: PaperSummaryJailbreak,
) -> Any:
    if dependencies.summary_service is not None:
        return dependencies.summary_service
    if dependencies.summary_service_factory is not None:
        return dependencies.summary_service_factory(settings, method)
    models = _load_model_configs(settings)
    model = models["gemma_4_12b"]
    if model.provider == "fake":
        from crosslingual_safety.manual_commands import _ManualFakeSummaryProvider

        return PaperSummaryService.from_method(
            method,
            provider=_ManualFakeSummaryProvider(model.fake_status),
            provider_id="fake",
            timeout_seconds=max(model.timeout_seconds, 180.0),
        )
    if not model.base_url_env or not model.api_key_env:
        raise ValueError("PSA summary provider requires endpoint metadata")
    base_url = os.environ.get(model.base_url_env)
    api_key = os.environ.get(model.api_key_env)
    if not base_url or not api_key:
        raise ValueError("required PSA summary provider environment variable is unset")
    return PaperSummaryService.from_method(
        method,
        provider_id=model.provider,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=max(model.timeout_seconds, 180.0),
    )


def _write_text_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ContractConflictError(f"immutable artifact conflict: {path}")
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_text_replace(path: Path, content: str, *, durable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    if durable:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    if durable and os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory = os.open(path.parent, flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _write_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSONL row: {path}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    content = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    _write_text_immutable(path, content)


def _append_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    key: str,
    *,
    durable: bool = False,
) -> None:
    existing = _read_jsonl(path)
    by_key = {str(row.get(key)): row for row in existing if key in row}
    pending: list[dict[str, object]] = []
    for row in rows:
        row_dict = dict(row)
        row_key = str(row_dict.get(key, stable_id(_canonical_json(row_dict))))
        prior = by_key.get(row_key)
        if prior is not None and _canonical_json(prior) != _canonical_json(row_dict):
            raise ContractConflictError(f"immutable JSONL row conflict: {path} ({row_key})")
        if prior is None:
            pending.append(row_dict)
        by_key[row_key] = row_dict
    if durable:
        if not pending:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as stream:
            for row in pending:
                stream.write(_canonical_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name != "nt":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory = os.open(path.parent, flags)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return
    ordered = sorted(by_key.values(), key=lambda value: str(value.get(key, "")))
    content = "".join(_canonical_json(dict(row)) + "\n" for row in ordered)
    _write_text_replace(path, content, durable=durable)


def _translation_record(
    case: UnifiedCase,
    language: str,
    translated_text: str,
    translator: Translator | None,
    provider_request_id: str | None,
    clock: Callable[[], str],
    provider_reservation_id: str | None = None,
) -> dict[str, object]:
    translator_id = "source" if translator is None else str(translator.translator_id)
    translator_version = "1" if translator is None else str(translator.version)
    decoding = {} if translator is None else dict(translator.decoding_config)
    normalized = canonicalize_text(translated_text)
    translation_id = stable_id(
        "unified-translation",
        case.case_id,
        case.source_language,
        language,
        normalized,
        translator_id,
        translator_version,
        _canonical_json(decoding),
    )
    record: dict[str, object] = {
        "translation_id": translation_id,
        "case_id": case.case_id,
        "source": case.source,
        "source_language": case.source_language,
        "target_language": language,
        "source_text": case.source_text,
        "raw_translated_text": translated_text,
        "normalized_translated_text": normalized,
        "method": "identity"
        if translator is None
        else str(getattr(translator, "method", "translation")),
        "translator_id": translator_id,
        "translator_version": translator_version,
        "provider": translator_id,
        "decoding_config": decoding,
        "source_character_count": len(case.source_text),
        "source_text_sha256": _sha256_text(case.source_text),
        "translated_text_sha256": _sha256_text(normalized),
        "provider_request_id": provider_request_id,
        "provider_reservation_id": provider_reservation_id,
        "created_at": clock(),
        "frozen": False,
        "review_status": "pending",
    }
    if translator is not None and translator_id == GOOGLE_CLOUD_TRANSLATOR:
        provider_contract = getattr(translator, "provider_contract", None)
        if isinstance(provider_contract, Mapping):
            record["provider_contract"] = dict(provider_contract)
        record.update(
            {
                "provider_project_id": str(getattr(translator, "project_id")),
                "provider_location": str(getattr(translator, "location")),
                "provider_model": str(getattr(translator, "model")),
                "provider_client_version": str(getattr(translator, "client_library_version")),
            }
        )
    return record


def _valid_audit_timestamp(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 64
        and "\r" not in value
        and "\n" not in value
    )


def _validate_translation_attempt(
    row: Mapping[str, object],
    *,
    job_contexts: Mapping[tuple[str, str], UnifiedCase],
    reservations: Mapping[str, Mapping[str, object]],
    reservation_contexts: Mapping[str, tuple[PaidTranslationTask, UnifiedCase]],
    google_paid_run: bool,
) -> tuple[tuple[str, str], int, str | None]:
    def reject() -> NoReturn:
        message = (
            "invalid Google Cloud paid-call attempt audit"
            if google_paid_run or row.get("provider_reservation_id") is not None
            else "invalid translation attempt audit"
        )
        raise ContractConflictError(message)

    provider_reservation_id = row.get("provider_reservation_id")
    if provider_reservation_id is not None and not isinstance(provider_reservation_id, str):
        reject()
    status = row.get("status")
    expected_fields = TRANSLATION_ATTEMPT_FIELDS
    if isinstance(provider_reservation_id, str):
        expected_fields |= PAID_ATTEMPT_LINK_FIELDS
        if status == "indeterminate":
            expected_fields |= {"billing_status"}
    if set(row) != expected_fields:
        reject()

    case_id = row.get("case_id")
    target_language = row.get("target_language")
    attempt_number = row.get("attempt_number")
    source_character_count = row.get("source_character_count")
    charged_character_count = row.get("charged_character_count")
    error_type = row.get("error_type")
    error_message = row.get("error_message")
    if (
        not isinstance(case_id, str)
        or not isinstance(target_language, str)
        or isinstance(attempt_number, bool)
        or not isinstance(attempt_number, int)
        or attempt_number <= 0
        or isinstance(source_character_count, bool)
        or not isinstance(source_character_count, int)
        or isinstance(charged_character_count, bool)
        or not isinstance(charged_character_count, int)
        or charged_character_count < 0
        or not isinstance(error_type, str)
        or SANITIZED_ERROR_MESSAGES.get(error_type) != error_message
        or not _valid_audit_timestamp(row.get("created_at"))
    ):
        reject()

    key = (case_id, target_language)
    case = job_contexts.get(key)
    if (
        case is None
        or row.get("source") != case.source
        or row.get("source_language") != case.source_language
        or source_character_count != len(case.source_text)
        or row.get("source_text_sha256") != _sha256_text(case.source_text)
        or not isinstance(row.get("provider"), str)
        or not row.get("provider")
    ):
        reject()

    expected_attempt_id = stable_id(
        "translation-attempt",
        case_id,
        target_language,
        error_type,
        str(attempt_number),
        provider_reservation_id or "none",
    )
    if row.get("attempt_id") != expected_attempt_id:
        reject()

    if provider_reservation_id is None:
        if status != "failed":
            reject()
        if google_paid_run and (
            row.get("provider") != GOOGLE_CLOUD_TRANSLATOR or charged_character_count != 0
        ):
            reject()
        return key, attempt_number, None

    reservation = reservations.get(provider_reservation_id)
    context = reservation_contexts.get(provider_reservation_id)
    if reservation is None or context is None:
        reject()
    task, paid_case = context
    reservation_attempt_number = reservation.get("attempt_number")
    if (
        paid_case != case
        or attempt_number != reservation_attempt_number
        or row.get("task_key") != task.task_key
        or row.get("provider") != task.provider
        or row.get("provider_contract_sha256") != task.provider_contract_sha256
        or row.get("source_text_sha256") != task.source_text_sha256
        or source_character_count != task.source_character_count
        or charged_character_count != task.source_character_count
        or row.get("audit_reference") != f"translation_reservations.jsonl#{provider_reservation_id}"
    ):
        reject()
    if status == "indeterminate":
        if (
            error_type != GOOGLE_CLOUD_INDETERMINATE_ERROR
            or error_message != GOOGLE_CLOUD_INDETERMINATE_MESSAGE
            or row.get("billing_status") != "charged_as_indeterminate"
        ):
            reject()
    elif status != "failed" or error_type not in PROVEN_PAID_REJECTION_ERROR_TYPES:
        reject()
    return key, attempt_number, provider_reservation_id


def _validate_translation_audit_row(
    row: Mapping[str, object],
    *,
    job_contexts: Mapping[tuple[str, str], UnifiedCase],
    reservations: Mapping[str, Mapping[str, object]],
    reservation_contexts: Mapping[str, tuple[PaidTranslationTask, UnifiedCase]],
    provider_contract: Mapping[str, object] | None,
    google_paid_run: bool,
) -> tuple[tuple[str, str], str | None]:
    required_fields = {
        "translation_id",
        "case_id",
        "source",
        "source_language",
        "target_language",
        "source_text",
        "raw_translated_text",
        "normalized_translated_text",
        "method",
        "translator_id",
        "translator_version",
        "provider",
        "decoding_config",
        "source_character_count",
        "source_text_sha256",
        "translated_text_sha256",
        "provider_request_id",
        "provider_reservation_id",
        "created_at",
        "frozen",
        "review_status",
    }
    if not required_fields.issubset(row):
        raise ContractConflictError("invalid persisted translation audit")
    case_id = row.get("case_id")
    target_language = row.get("target_language")
    if not isinstance(case_id, str) or not isinstance(target_language, str):
        raise ContractConflictError("invalid persisted translation audit")
    key = (case_id, target_language)
    case = job_contexts.get(key)
    normalized = row.get("normalized_translated_text")
    raw = row.get("raw_translated_text")
    translator_id = row.get("translator_id")
    translator_version = row.get("translator_version")
    decoding_config = row.get("decoding_config")
    provider_request_id = row.get("provider_request_id")
    if (
        case is None
        or row.get("source") != case.source
        or row.get("source_language") != case.source_language
        or row.get("source_text") != case.source_text
        or row.get("source_character_count") != len(case.source_text)
        or row.get("source_text_sha256") != _sha256_text(case.source_text)
        or not isinstance(raw, str)
        or not raw.strip()
        or not isinstance(normalized, str)
        or not normalized.strip()
        or canonicalize_text(raw) != normalized
        or row.get("translated_text_sha256") != _sha256_text(normalized)
        or not isinstance(translator_id, str)
        or not translator_id
        or not isinstance(translator_version, str)
        or not translator_version
        or not isinstance(decoding_config, Mapping)
        or row.get("provider") != translator_id
        or (
            provider_request_id is not None
            and (
                not isinstance(provider_request_id, str)
                or len(provider_request_id) > 256
                or re.fullmatch(r"[A-Za-z0-9._=-]+", provider_request_id) is None
            )
        )
        or not _valid_audit_timestamp(row.get("created_at"))
        or row.get("frozen") is not False
        or row.get("review_status") != "pending"
    ):
        raise ContractConflictError("invalid persisted translation audit")
    expected_translation_id = stable_id(
        "unified-translation",
        case.case_id,
        case.source_language,
        target_language,
        normalized,
        translator_id,
        translator_version,
        _canonical_json(dict(decoding_config)),
    )
    if row.get("translation_id") != expected_translation_id:
        raise ContractConflictError("invalid persisted translation audit")

    provider_reservation_id = row.get("provider_reservation_id")
    if target_language == case.source_language:
        if (
            provider_reservation_id is not None
            or translator_id != "source"
            or translator_version != "1"
            or row.get("method") != "identity"
            or dict(decoding_config)
            or raw != case.source_text
        ):
            raise ContractConflictError("invalid persisted translation audit")
        return key, None

    if translator_id == "native_dataset":
        if (
            provider_reservation_id is not None
            or translator_version != "multijail-native-v1"
            or row.get("method") != "native_dataset"
            or provider_request_id is not None
            or decoding_config.get("source_record_id") in {None, ""}
            or decoding_config.get("dataset_language") not in {"zh", "jv", "th", "vi"}
            or (target_language == "zh-tw" and decoding_config.get("conversion") != "s2twp")
            or (target_language != "zh-tw" and "conversion" in decoding_config)
        ):
            raise ContractConflictError("invalid persisted native translation audit")
        return key, None

    if not google_paid_run:
        if provider_reservation_id is not None:
            raise ContractConflictError("invalid Google Cloud paid-call outcome identity")
        return key, None

    if not isinstance(provider_reservation_id, str):
        raise ContractConflictError("invalid Google Cloud paid-call outcome identity")
    reservation = reservations.get(provider_reservation_id)
    context = reservation_contexts.get(provider_reservation_id)
    if reservation is None or context is None or provider_contract is None:
        raise ContractConflictError("invalid Google Cloud paid-call outcome identity")
    task, paid_case = context
    if (
        paid_case != case
        or row.get("task_key") != task.task_key
        or row.get("provider_contract_sha256") != task.provider_contract_sha256
        or row.get("source_text_sha256") != task.source_text_sha256
        or row.get("source_character_count") != task.source_character_count
        or translator_id != task.provider
        or row.get("provider") != task.provider
        or row.get("method") != "google_cloud_nmt_v3"
        or row.get("provider_contract") != dict(provider_contract)
        or dict(decoding_config) != dict(provider_contract)
        or translator_version != provider_contract.get("model")
        or row.get("provider_project_id") != provider_contract.get("project_id")
        or row.get("provider_location") != provider_contract.get("location")
        or row.get("provider_model") != provider_contract.get("model")
        or row.get("provider_client_version") != provider_contract.get("client_library_version")
    ):
        raise ContractConflictError("invalid Google Cloud paid-call outcome identity")
    return key, provider_reservation_id


def _translate_cases(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies,
    parent_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    audit = parent_path / "audit"
    translations_path = audit / "translations.jsonl"
    attempts_path = audit / "translation_attempts.jsonl"
    paid_ledger = PaidCallLedger(audit)
    existing = _read_jsonl(translations_path)
    failures: dict[tuple[str, str], dict[str, object]] = {}
    attempts = _read_jsonl(attempts_path)
    try:
        reservation_rows = paid_ledger.reservations()
    except PaidCallLedgerError:
        raise ContractConflictError("invalid Google Cloud paid-call reservation identity") from None
    job_contexts: dict[tuple[str, str], UnifiedCase] = {}
    native_catalog = _bench_translation_catalog(plan.request, settings)
    native_jobs: dict[tuple[str, str], BenchTranslation] = {}
    for case in plan.cases:
        for target_language in plan.languages:
            key = (case.case_id, target_language)
            if key in job_contexts:
                raise ContractConflictError("duplicate translation job identity")
            job_contexts[key] = case
            native = _native_bench_translation(native_catalog, case, target_language)
            if native is not None:
                native_jobs[key] = native
    paid_tasks_by_job: dict[tuple[str, str], PaidTranslationTask] = {}
    paid_task_contexts: dict[str, tuple[PaidTranslationTask, UnifiedCase]] = {}
    provider_contract: Mapping[str, object] | None = None
    if settings.translator == GOOGLE_CLOUD_TRANSLATOR:
        configured_provider_contract = plan.contract.get("translator_contract")
        if not isinstance(configured_provider_contract, Mapping):
            raise ContractConflictError("invalid Google Cloud translator contract")
        provider_contract = cast(Mapping[str, object], configured_provider_contract)
        for case in plan.cases:
            for target_language in plan.languages:
                if target_language == case.source_language:
                    continue
                if (case.case_id, target_language) in native_jobs:
                    continue
                task = PaidTranslationTask.build(
                    case_id=case.case_id,
                    source_text=case.source_text,
                    source_language=case.source_language,
                    target_language=target_language,
                    provider=GOOGLE_CLOUD_TRANSLATOR,
                    provider_contract=provider_contract,
                )
                key = (case.case_id, target_language)
                if task.task_key in paid_task_contexts or key in paid_tasks_by_job:
                    raise ContractConflictError("duplicate Google Cloud paid-call task identity")
                paid_tasks_by_job[key] = task
                paid_task_contexts[task.task_key] = (task, case)
    elif reservation_rows:
        raise ContractConflictError("unexpected Google Cloud paid-call reservation audit")

    reservations: dict[str, dict[str, object]] = {}
    reservation_contexts: dict[str, tuple[PaidTranslationTask, UnifiedCase]] = {}
    for row in reservation_rows:
        task_key = row.get("task_key")
        context = paid_task_contexts.get(task_key) if isinstance(task_key, str) else None
        if context is None:
            raise ContractConflictError("invalid Google Cloud paid-call reservation identity")
        task, case = context
        try:
            validated_reservation_id, _ = paid_ledger.validate_reservation_for_task(row, task)
        except PaidCallLedgerError:
            raise ContractConflictError(
                "invalid Google Cloud paid-call reservation identity"
            ) from None
        prior = reservations.get(validated_reservation_id)
        if prior is not None and _canonical_json(prior) != _canonical_json(row):
            raise ContractConflictError("conflicting Google Cloud paid-call reservation audit")
        reservations[validated_reservation_id] = row
        reservation_contexts[validated_reservation_id] = (task, case)

    attempt_counts: dict[tuple[str, str], int] = {}
    completed_reservations: set[str] = set()
    terminal_indeterminate: set[tuple[str, str]] = set()
    seen_attempt_ids: set[str] = set()
    for attempt_row in attempts:
        key, attempt_number, completed_reservation_id = _validate_translation_attempt(
            attempt_row,
            job_contexts=job_contexts,
            reservations=reservations,
            reservation_contexts=reservation_contexts,
            google_paid_run=settings.translator == GOOGLE_CLOUD_TRANSLATOR,
        )
        attempt_id = cast(str, attempt_row["attempt_id"])
        if attempt_id in seen_attempt_ids:
            raise ContractConflictError("duplicate translation attempt audit")
        seen_attempt_ids.add(attempt_id)
        attempt_counts[key] = max(attempt_counts.get(key, 0), attempt_number)
        if completed_reservation_id is not None:
            if completed_reservation_id in completed_reservations:
                raise ContractConflictError("duplicate Google Cloud paid-call outcome audit")
            completed_reservations.add(completed_reservation_id)
        if attempt_row.get("status") == "indeterminate":
            terminal_indeterminate.add(key)
            failures[key] = attempt_row

    for reservation_id, reservation in reservations.items():
        context = reservation_contexts[reservation_id]
        task, _ = context
        try:
            _, attempt_number = paid_ledger.validate_reservation_for_task(reservation, task)
        except PaidCallLedgerError:
            raise ContractConflictError(
                "invalid Google Cloud paid-call reservation identity"
            ) from None
        key = (task.case_id, task.target_language)
        attempt_counts[key] = max(attempt_counts.get(key, 0), attempt_number)

    successful: dict[tuple[str, str], dict[str, object]] = {}
    seen_translation_ids: set[str] = set()
    for row in existing:
        key, completed_reservation_id = _validate_translation_audit_row(
            row,
            job_contexts=job_contexts,
            reservations=reservations,
            reservation_contexts=reservation_contexts,
            provider_contract=provider_contract,
            google_paid_run=settings.translator == GOOGLE_CLOUD_TRANSLATOR,
        )
        translation_id = cast(str, row["translation_id"])
        if translation_id in seen_translation_ids or key in successful:
            raise ContractConflictError("duplicate persisted translation audit")
        seen_translation_ids.add(translation_id)
        if completed_reservation_id is not None:
            if completed_reservation_id in completed_reservations:
                raise ContractConflictError("duplicate Google Cloud paid-call outcome audit")
            completed_reservations.add(completed_reservation_id)
        successful[key] = row

    try:
        historical_google_characters = paid_ledger.charged_characters()
    except PaidCallLedgerError:
        raise ContractConflictError("invalid Google Cloud paid-call reservation identity") from None

    for row in existing:
        if row.get("provider") != GOOGLE_CLOUD_TRANSLATOR:
            continue
        linked_reservation_id = row.get("provider_reservation_id")
        if isinstance(linked_reservation_id, str) and linked_reservation_id in reservations:
            continue
        try:
            historical_google_characters += max(
                0, int(cast(Any, row.get("source_character_count") or 0))
            )
        except (TypeError, ValueError):
            continue
    for row in attempts:
        if row.get("provider") != GOOGLE_CLOUD_TRANSLATOR:
            continue
        linked_reservation_id = row.get("provider_reservation_id")
        if isinstance(linked_reservation_id, str) and linked_reservation_id in reservations:
            continue
        try:
            historical_google_characters += max(
                0, int(cast(Any, row.get("charged_character_count") or 0))
            )
        except (TypeError, ValueError):
            continue

    for reservation_id, reservation in reservations.items():
        if reservation_id in completed_reservations:
            continue
        task, case = reservation_contexts[reservation_id]
        try:
            _, attempt_number = paid_ledger.validate_reservation_for_task(reservation, task)
        except PaidCallLedgerError:
            raise ContractConflictError(
                "invalid Google Cloud paid-call reservation identity"
            ) from None
        key = (task.case_id, task.target_language)
        indeterminate_error = GoogleCloudIndeterminatePaidAttemptError(
            GOOGLE_CLOUD_INDETERMINATE_MESSAGE
        )
        error_type, error_message = _sanitized_error(indeterminate_error)
        indeterminate_attempt: dict[str, object] = {
            "attempt_id": stable_id(
                "translation-attempt",
                task.case_id,
                task.target_language,
                error_type,
                str(attempt_number),
                reservation_id,
            ),
            "attempt_number": attempt_number,
            "audit_reference": f"translation_reservations.jsonl#{reservation_id}",
            "billing_status": "charged_as_indeterminate",
            "case_id": task.case_id,
            "status": "indeterminate",
            "source": case.source,
            "source_language": task.source_language,
            "target_language": task.target_language,
            "provider": GOOGLE_CLOUD_TRANSLATOR,
            "provider_reservation_id": reservation_id,
            "task_key": task.task_key,
            "provider_contract_sha256": task.provider_contract_sha256,
            "source_character_count": task.source_character_count,
            "source_text_sha256": task.source_text_sha256,
            "charged_character_count": task.source_character_count,
            "error_type": error_type,
            "error_message": error_message,
            "created_at": dependencies.clock(),
        }
        _append_jsonl(
            attempts_path,
            [indeterminate_attempt],
            "attempt_id",
            durable=True,
        )
        attempts.append(indeterminate_attempt)
        failures[key] = indeterminate_attempt
        terminal_indeterminate.add(key)

    pending_paid_task: PaidTranslationTask | None = None
    persisted_reservation_id: str | None = None

    def persist_paid_call_reservation(character_count: int) -> None:
        nonlocal persisted_reservation_id
        task = pending_paid_task
        if task is None or task.source_character_count != character_count:
            raise ContractConflictError("Google Cloud paid-call reservation context is invalid")
        reservation = paid_ledger.make_reservation(
            task,
            character_count=character_count,
            clock=dependencies.clock,
        )
        reservation_id = reservation.get("reservation_id")
        if not isinstance(reservation_id, str):
            raise ContractConflictError("Google Cloud paid-call reservation context is invalid")
        persisted_reservation_id = reservation_id

    translator = (
        _make_translator(
            settings,
            dependencies,
            initial_google_characters=historical_google_characters,
            google_paid_call_reservation=persist_paid_call_reservation,
        )
        if plan.translation_jobs
        else None
    )
    if isinstance(translator, GoogleCloudNMTTranslator):
        translator.paid_call_reservation = persist_paid_call_reservation
        for key, task in paid_tasks_by_job.items():
            case = next(item for item in plan.cases if item.case_id == key[0])
            observed_task = PaidTranslationTask.build(
                case_id=case.case_id,
                source_text=case.source_text,
                source_language=case.source_language,
                target_language=key[1],
                provider=translator.translator_id,
                provider_contract=translator.provider_contract,
            )
            if observed_task != task:
                raise ContractConflictError(
                    "Google Cloud translator does not match the immutable run contract"
                )

    for case in plan.cases:
        for language in plan.languages:
            key = (case.case_id, language)
            if key in successful or key in terminal_indeterminate:
                continue
            if language == case.source_language:
                record = _translation_record(
                    case, language, case.source_text, None, None, dependencies.clock
                )
                _append_jsonl(translations_path, [record], "translation_id", durable=True)
                successful[key] = record
                continue
            native = native_jobs.get(key)
            if native is not None:
                native_translator = NativeBenchTranslator(native)
                record = _translation_record(
                    case,
                    language,
                    native.text,
                    native_translator,
                    None,
                    dependencies.clock,
                )
                _append_jsonl(translations_path, [record], "translation_id", durable=True)
                successful[key] = record
                continue

            attempt_number = attempt_counts.get(key, 0) + 1
            pending_paid_task = None
            persisted_reservation_id = None
            if isinstance(translator, GoogleCloudNMTTranslator):
                pending_paid_task = paid_tasks_by_job.get(key)
                if pending_paid_task is None:
                    raise ContractConflictError(
                        "Google Cloud paid-call task context is unavailable"
                    )

            characters_before = int(getattr(translator, "characters_used", 0))
            try:
                if translator is None or not translator.supports(case.source_language, language):
                    raise ValueError(
                        f"translator does not support {case.source_language}->{language}"
                    )
                output = translator.translate(case.source_text, case.source_language, language)
                if not isinstance(output, ProviderTranslation):
                    output = ProviderTranslation(
                        str(getattr(output, "text", output)),
                        getattr(output, "provider_request_id", None),
                    )
                record = _translation_record(
                    case,
                    language,
                    output.text,
                    translator,
                    output.provider_request_id,
                    dependencies.clock,
                    provider_reservation_id=persisted_reservation_id,
                )
                if pending_paid_task is not None:
                    record["task_key"] = pending_paid_task.task_key
                    record["provider_contract_sha256"] = pending_paid_task.provider_contract_sha256
            except Exception as error:
                paid_task = pending_paid_task
                is_indeterminate = (
                    persisted_reservation_id is not None
                    and paid_task is not None
                    and not is_proven_preprocessing_rejection(error)
                )
                if is_indeterminate:
                    error_type = GOOGLE_CLOUD_INDETERMINATE_ERROR
                    error_message = GOOGLE_CLOUD_INDETERMINATE_MESSAGE
                else:
                    error_type, error_message = _sanitized_error(error)
                characters_after = int(getattr(translator, "characters_used", 0))
                attempt_counts[key] = attempt_number
                failure_attempt: dict[str, object] = {
                    "attempt_id": stable_id(
                        "translation-attempt",
                        case.case_id,
                        language,
                        error_type,
                        str(attempt_number),
                        persisted_reservation_id or "none",
                    ),
                    "attempt_number": attempt_number,
                    "case_id": case.case_id,
                    "status": "indeterminate" if is_indeterminate else "failed",
                    "source": case.source,
                    "source_language": case.source_language,
                    "target_language": language,
                    "provider": (
                        settings.translator if translator is None else str(translator.translator_id)
                    ),
                    "source_character_count": len(case.source_text),
                    "source_text_sha256": _sha256_text(case.source_text),
                    "charged_character_count": (
                        paid_task.source_character_count
                        if persisted_reservation_id is not None and paid_task is not None
                        else max(0, characters_after - characters_before)
                    ),
                    "provider_reservation_id": persisted_reservation_id,
                    "error_type": error_type,
                    "error_message": error_message,
                    "created_at": dependencies.clock(),
                }
                if persisted_reservation_id is not None:
                    failure_attempt["audit_reference"] = (
                        f"translation_reservations.jsonl#{persisted_reservation_id}"
                    )
                    if paid_task is None:
                        raise ContractConflictError(
                            "Google Cloud paid-call attempt context is invalid"
                        )
                    failure_attempt["task_key"] = paid_task.task_key
                    failure_attempt["provider_contract_sha256"] = paid_task.provider_contract_sha256
                if is_indeterminate:
                    failure_attempt["billing_status"] = "charged_as_indeterminate"
                _append_jsonl(
                    attempts_path,
                    [failure_attempt],
                    "attempt_id",
                    durable=True,
                )
                failures[key] = failure_attempt
            else:
                _append_jsonl(translations_path, [record], "translation_id", durable=True)
                successful[key] = record
            finally:
                pending_paid_task = None
                persisted_reservation_id = None

    return successful, failures


def _quarantine_summary_cache(path: Path) -> Path:
    """Preserve invalid cache bytes under a content-addressed audit path."""

    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    quarantine = path.with_name(f"{path.stem}.quarantine.{digest}{path.suffix}")
    if quarantine.is_file():
        if quarantine.read_bytes() != payload:
            raise ContractConflictError(f"summary cache quarantine conflict: {quarantine}")
    else:
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        temporary = quarantine.with_suffix(f"{quarantine.suffix}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(quarantine)
    path.unlink()
    return quarantine


def _summary_cache(
    method: PaperSummaryJailbreak,
    settings: RunSettings,
    dependencies: RunDependencies,
    parent_path: Path,
) -> dict[str, SummaryArtifact] | dict[str, object] | None:
    service = _make_summary_service(settings, dependencies, method)
    cache_path = parent_path / "audit" / "psa_summary_artifacts.jsonl"
    if not hasattr(service, "load_cache") and cache_path.is_file():
        try:
            cached_rows = _read_jsonl(cache_path)
            cached = {
                str(row["language"]): row.get("value")
                for row in cached_rows
                if row.get("language") in SUMMARY_WRAPPER_LANGUAGES
            }
            if set(cached) == set(SUMMARY_WRAPPER_LANGUAGES):
                return cached
            raise ValueError("summary cache must contain all wrapper languages")
        except Exception:
            _quarantine_summary_cache(cache_path)
    try:
        loaded = service.load_cache(cache_path)
        return cast(dict[str, SummaryArtifact], loaded)
    except FileNotFoundError:
        pass
    except Exception:
        if cache_path.is_file():
            _quarantine_summary_cache(cache_path)
    artifacts: dict[str, Any] = {}
    try:
        for language in SUMMARY_WRAPPER_LANGUAGES:
            if hasattr(service, "summarize_async"):
                summarize_async = service.summarize_async
                try:
                    parameter_count = len(inspect.signature(summarize_async).parameters)
                except (TypeError, ValueError):
                    parameter_count = 1
                value = (
                    summarize_async(getattr(method, "summary_id", ""), language)
                    if parameter_count >= 2
                    else summarize_async(language)
                )
                if inspect.isawaitable(value):
                    value = asyncio.run(cast(Any, value))
            elif hasattr(service, "summarize"):
                summarize = service.summarize
                try:
                    parameter_count = len(inspect.signature(summarize).parameters)
                except (TypeError, ValueError):
                    parameter_count = 2
                value = (
                    summarize(getattr(method, "summary_id", ""), language)
                    if parameter_count >= 2
                    else summarize(language)
                )
            elif callable(service):
                value = service(language)
            else:
                raise ValueError("summary service has no summarize method")
            artifacts[language] = value
    except Exception:
        # No partial artifact is written by design.
        raise
    if hasattr(service, "write_cache"):
        service.write_cache(cache_path, artifacts)
    else:
        # Lightweight fakes can return already-validated mapping values.  Persist only values
        # that are JSON serializable; renderers use static sections for non-SummaryArtifact fakes.
        rows = [
            {
                "language": language,
                "summary_id": getattr(method, "summary_id", ""),
                "value": (
                    artifacts[language].model_dump(mode="json")
                    if isinstance(artifacts[language], SummaryArtifact)
                    else artifacts[language]
                ),
            }
            for language in SUMMARY_WRAPPER_LANGUAGES
        ]
        _write_jsonl(cache_path, rows)
    return artifacts


def _load_formal_summary_cache(
    path: Path,
    *,
    summary_id: str,
    source_sha256: str,
    summary_request_contract: Mapping[str, object],
    translator: Translator,
) -> dict[str, SummaryArtifact]:
    artifacts: dict[str, SummaryArtifact] = {}
    for row in _read_jsonl(path):
        artifact = SummaryArtifact.model_validate(row)
        if artifact.language in artifacts:
            raise ContractConflictError("duplicate formal PSA summary language")
        if artifact.summary_id != summary_id or artifact.source_sha256 != source_sha256:
            raise ContractConflictError("formal PSA summary cache contract mismatch")
        artifact_sections(artifact)
        artifacts[artifact.language] = artifact
    if set(artifacts) != set(PUBLIC_LANGUAGES):
        raise ContractConflictError("formal PSA summary cache is incomplete")
    _validate_formal_summary_artifact_provenance(
        artifacts,
        summary_id=summary_id,
        source_sha256=source_sha256,
        summary_request_contract=summary_request_contract,
        translator=translator,
    )
    return artifacts


def _validate_formal_summary_artifact_provenance(
    artifacts: Mapping[str, SummaryArtifact],
    *,
    summary_id: str,
    source_sha256: str,
    summary_request_contract: Mapping[str, object],
    translator: Translator,
) -> None:
    if set(artifacts) != set(PUBLIC_LANGUAGES):
        raise ContractConflictError("formal PSA summary cache is incomplete")
    if any(
        artifact.summary_id != summary_id or artifact.source_sha256 != source_sha256
        for artifact in artifacts.values()
    ):
        raise ContractConflictError("formal PSA summary cache contract mismatch")
    english = artifacts["en"]
    expected_english = {
        "request_sha256": summary_request_contract.get("request_sha256"),
        "provider_id": summary_request_contract.get("provider_id"),
        "model_id": summary_request_contract.get("model_id"),
        "endpoint_type": summary_request_contract.get("endpoint_type"),
        "generation_config": summary_request_contract.get("generation_config"),
    }
    actual_english = {
        "request_sha256": english.request_sha256,
        "provider_id": english.provider_id,
        "model_id": english.model_id,
        "endpoint_type": english.endpoint_type,
        "generation_config": english.generation_config,
    }
    if actual_english != expected_english:
        raise ContractConflictError("formal PSA cache summary request provenance mismatch")
    for language in PUBLIC_LANGUAGES:
        if language == "en":
            continue
        artifact = artifacts[language]
        request_contract = _formal_localization_request_contract(english, language, translator)
        expected_localization = {
            "request_sha256": _sha256_text(_canonical_json(request_contract)),
            "provider_id": translator.translator_id,
            "model_id": translator.version,
            "endpoint_type": "translation",
            "generation_config": _formal_localization_generation_config(english, translator),
        }
        actual_localization = {
            "request_sha256": artifact.request_sha256,
            "provider_id": artifact.provider_id,
            "model_id": artifact.model_id,
            "endpoint_type": artifact.endpoint_type,
            "generation_config": artifact.generation_config,
        }
        if actual_localization != expected_localization:
            raise ContractConflictError("formal PSA cache localization provenance mismatch")


def _summarize_english(service: object, summary_id: str) -> SummaryArtifact:
    if hasattr(service, "summarize_async"):
        value = service.summarize_async("en")
        if inspect.isawaitable(value):
            value = asyncio.run(cast(Any, value))
    elif hasattr(service, "summarize"):
        summarize = service.summarize
        try:
            parameter_count = len(inspect.signature(summarize).parameters)
        except (TypeError, ValueError):
            parameter_count = 2
        value = summarize(summary_id, "en") if parameter_count >= 2 else summarize("en")
    else:
        raise ValueError("formal PSA summary service has no summarize method")
    if not isinstance(value, SummaryArtifact):
        try:
            value = SummaryArtifact.model_validate(value)
        except Exception:
            raise ValueError("formal PSA summary service returned an invalid artifact") from None
    if value.summary_id != summary_id or value.language != "en":
        raise ValueError("formal PSA English summary identity mismatch")
    artifact_sections(value)
    return value


def _formal_localization_request_contract(
    english: SummaryArtifact,
    target_language: str,
    translator: Translator,
) -> dict[str, object]:
    return {
        "canonical_summary_sha256": english.response_sha256,
        "source_language": "en",
        "target_language": target_language,
        "translator_id": translator.translator_id,
        "translator_version": translator.version,
        "decoding_config": translator.decoding_config,
    }


def _formal_localization_generation_config(
    english: SummaryArtifact,
    translator: Translator,
) -> dict[str, object]:
    return {
        "canonical_summary_sha256": english.response_sha256,
        "decoding_config": dict(translator.decoding_config),
    }


def _localized_summary_artifact(
    english: SummaryArtifact,
    target_language: str,
    translator: Translator,
    clock: Callable[[], str],
) -> SummaryArtifact:
    if not translator.supports("en", target_language):
        raise ValueError(f"summary translator does not support en->{target_language}")
    localized: dict[str, str] = {}
    for key, value in artifact_sections(english).items():
        translated = translator.translate(value, "en", target_language)
        if not isinstance(translated, ProviderTranslation):
            translated = ProviderTranslation(str(getattr(translated, "text", translated)))
        localized[key] = canonicalize_text(translated.text)
        if not localized[key]:
            raise ValueError(f"formal PSA summary localization is empty: {target_language}/{key}")
    response_text = _canonical_json(localized)
    request_contract = _formal_localization_request_contract(english, target_language, translator)
    return SummaryArtifact(
        summary_id=english.summary_id,
        language=target_language,
        source_sha256=english.source_sha256,
        request_sha256=_sha256_text(_canonical_json(request_contract)),
        provider_id=translator.translator_id,
        model_id=translator.version,
        endpoint_type="translation",
        generation_config=_formal_localization_generation_config(english, translator),
        response_text=response_text,
        response_sha256=_sha256_text(response_text),
        provider_request_id=None,
        created_at=clock(),
    )


def _formal_summary_request_contract(service: object) -> dict[str, object]:
    raw_contract = getattr(service, "contract", None)
    if not isinstance(raw_contract, Mapping):
        raise ValueError("formal PSA summary service has no request contract")
    request_hashes = raw_contract.get("request_sha256s")
    if not isinstance(request_hashes, Mapping) or not isinstance(request_hashes.get("en"), str):
        raise ValueError("formal PSA summary service has no English request identity")
    required = (
        "summary_id",
        "source_sha256",
        "prompt_sha256",
        "provider_id",
        "model_id",
        "endpoint_type",
        "generation_config",
    )
    if any(key not in raw_contract for key in required):
        raise ValueError("formal PSA summary service request contract is incomplete")
    return {
        "summary_id": raw_contract["summary_id"],
        "source_sha256": raw_contract["source_sha256"],
        "prompt_sha256": raw_contract["prompt_sha256"],
        "request_sha256": request_hashes["en"],
        "provider_id": raw_contract["provider_id"],
        "model_id": raw_contract["model_id"],
        "endpoint_type": raw_contract["endpoint_type"],
        "generation_config": raw_contract["generation_config"],
    }


def _formal_translator_artifact_contract(translator: Translator) -> dict[str, object]:
    return {
        "translator_id": translator.translator_id,
        "translator_version": translator.version,
        "decoding_config": dict(translator.decoding_config),
    }


def _read_formal_cache_sidecar(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ContractConflictError("formal PSA cache is incomplete")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ContractConflictError("formal PSA cache sidecar is invalid") from None
    if not isinstance(value, dict):
        raise ContractConflictError("formal PSA cache sidecar is invalid")
    return cast(dict[str, object], value)


def _formal_summary_cache(
    condition: str,
    method: PaperSummaryJailbreak,
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies,
    parent_path: Path,
) -> dict[str, SummaryArtifact]:
    specs = load_psa_papers(settings.psa_papers_config)
    spec = specs.get(condition)
    if spec is None:
        raise ValueError(f"PSA paper condition is missing: {condition}")
    paper = extract_paper(spec)
    paper_method = SimpleNamespace(
        summary_id=method.summary_id,
        sections={"en": {"paper_text": paper.text}},
        provenance={
            "source_ref": spec.source_path.as_posix(),
            "source_language": "en",
            "source_sha256": paper.source_sha256,
            "text_sha256": paper.text_sha256,
            "chunk_sha256s": [chunk.text_sha256 for chunk in paper.chunks],
        },
        summary_prompt=method.summary_prompt,
    )
    service = _make_summary_service(
        settings,
        dependencies,
        cast(PaperSummaryJailbreak, paper_method),
    )
    source_sha256 = str(getattr(service, "source_sha256", ""))
    if not source_sha256:
        raise ValueError("formal PSA summary service has no source contract")
    summary_request_contract = _formal_summary_request_contract(service)
    summary_settings = settings.model_copy(update={"translator": GOOGLE_CLOUD_TRANSLATOR})
    translator = _make_translator(summary_settings, dependencies)
    if translator is None:
        raise ValueError("formal PSA summary translator is unavailable")
    configured_papers = plan.contract.get("psa_papers")
    paper_contract = (
        configured_papers.get(condition) if isinstance(configured_papers, Mapping) else None
    )
    if not isinstance(paper_contract, Mapping):
        raise ValueError(f"PSA paper contract is missing: {condition}")
    if (
        summary_request_contract.get("summary_id") != method.summary_id
        or summary_request_contract.get("source_sha256") != source_sha256
    ):
        raise ValueError("formal PSA summary request identity is inconsistent")
    cache_contract = {
        "paper": dict(paper_contract),
        "summary_source_sha256": source_sha256,
        "summary_prompt": method.summary_prompt,
        "summary_request_contract": summary_request_contract,
        "translator_contract": _translator_contract(summary_settings),
        "translator_artifact_contract": _formal_translator_artifact_contract(translator),
        "languages": list(PUBLIC_LANGUAGES),
    }
    cache_id = stable_id("formal-psa-cache", _canonical_json(cache_contract))
    cache_dir = settings.runs_dir.parent / "_cache" / "psa" / cache_id
    cache_path = cache_dir / "summary_artifacts.jsonl"
    contract_path = cache_dir / "cache_contract.json"
    extraction_path = cache_dir / "extraction_manifest.json"
    expected_extraction = paper.model_dump(mode="json", exclude={"condition_id"})
    cache_files = (cache_path, contract_path, extraction_path)
    if any(path.exists() for path in cache_files):
        if not all(path.is_file() for path in cache_files):
            raise ContractConflictError("formal PSA cache is incomplete")
        if _read_formal_cache_sidecar(contract_path) != cache_contract:
            raise ContractConflictError("formal PSA cache contract mismatch")
        if _read_formal_cache_sidecar(extraction_path) != expected_extraction:
            raise ContractConflictError("formal PSA cache extraction provenance mismatch")
        artifacts = _load_formal_summary_cache(
            cache_path,
            summary_id=method.summary_id,
            source_sha256=source_sha256,
            summary_request_contract=summary_request_contract,
            translator=translator,
        )
    else:
        english = _summarize_english(service, method.summary_id)
        artifacts = {"en": english}
        for language in PUBLIC_LANGUAGES:
            if language != "en":
                artifacts[language] = _localized_summary_artifact(
                    english,
                    language,
                    translator,
                    dependencies.clock,
                )
        _validate_formal_summary_artifact_provenance(
            artifacts,
            summary_id=method.summary_id,
            source_sha256=source_sha256,
            summary_request_contract=summary_request_contract,
            translator=translator,
        )
        content = "".join(
            _canonical_json(artifacts[language].model_dump(mode="json")) + "\n"
            for language in PUBLIC_LANGUAGES
        )
        _write_json(extraction_path, expected_extraction)
        _write_json(contract_path, cache_contract)
        _write_text_immutable(cache_path, content)
    _write_json(
        parent_path / "audit" / f"{condition}_summary_cache.json",
        {
            "cache_id": cache_id,
            "cache_path": cache_path.as_posix(),
            "condition": condition,
            "source_sha256": source_sha256,
        },
    )
    return artifacts


def _summary_for_language(value: object, language: str) -> SummaryArtifact | None:
    if isinstance(value, SummaryArtifact):
        return value
    if isinstance(value, Mapping):
        # Some test doubles expose a SummaryArtifact mapping directly.
        try:
            artifact = SummaryArtifact.model_validate(value)
            return artifact
        except Exception:
            return None
    return None


def _apply_low_resource_output_contract(
    rendered: JailbreakResult,
    *,
    payload_language: str,
    jailbreak: str,
) -> JailbreakResult:
    language_name = OUTPUT_LANGUAGE_NAMES.get(payload_language)
    instruction_contract = FALLBACK_OUTPUT_INSTRUCTIONS.get(jailbreak)
    if language_name is None or instruction_contract is None:
        return rendered
    english_instruction, target_template = instruction_contract
    if (
        rendered.wrapper_language != "en"
        or rendered.rendered_prompt.count(english_instruction) != 1
    ):
        raise ValueError("low-resource wrapper fallback contract is unavailable")
    target_instruction = target_template.format(language_name=language_name)
    rendered_prompt = rendered.rendered_prompt.replace(english_instruction, target_instruction)
    try:
        metadata = json.loads(rendered.metadata_json)
    except json.JSONDecodeError:
        raise ValueError("low-resource wrapper fallback metadata is invalid") from None
    if not isinstance(metadata, dict):
        raise ValueError("low-resource wrapper fallback metadata is invalid")
    metadata.update(
        {
            "wrapper_fallback": "english",
            "requested_output_language": payload_language,
            "requested_output_language_name": language_name,
        }
    )
    metadata_json = _canonical_json(metadata)
    fallback_contract = _canonical_json(
        {
            "base_template_sha256": rendered.template_sha256,
            "target_instruction": target_instruction,
            "metadata_json": metadata_json,
        }
    )
    return replace(
        rendered,
        rendered_prompt=rendered_prompt,
        template_sha256=_sha256_text(fallback_contract),
        metadata_json=metadata_json,
    )


def _render_variants(
    plan: RunPlan,
    settings: RunSettings,
    method: JailbreakMethod,
    jailbreak: str,
    translations: Mapping[tuple[str, str], dict[str, object]],
    translation_failures: Mapping[tuple[str, str], dict[str, object]],
    summaries: Mapping[str, object] | None,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    variants: list[dict[str, object]] = []
    errors: dict[tuple[str, str], dict[str, object]] = {}
    for case in plan.cases:
        for language in plan.languages:
            key = (case.case_id, language)
            if key in translation_failures:
                continue
            translation = translations.get(key)
            if translation is None:
                errors[key] = {
                    "error_type": "missing_translation",
                    "error_message": "translation record is unavailable",
                }
                continue
            wrapper_language = _wrapper_language(language, jailbreak)
            role = settings.gra_role if jailbreak == "gra" else None
            context = JailbreakContext(
                language=language,
                wrapper_language=wrapper_language,
                intent=case.intent,
                category=case.category,
                role=role,
            )
            try:
                rendered: Any
                summary_artifact = _summary_for_language(
                    summaries.get(wrapper_language) if summaries is not None else None,
                    wrapper_language,
                )
                if isinstance(method, PaperSummaryJailbreak) and summary_artifact is not None:
                    rendered = method.render(
                        str(translation["normalized_translated_text"]),
                        context,
                        summary_sections=artifact_sections(summary_artifact),
                        summary_artifact=summary_artifact,
                    )
                else:
                    rendered = method.render(
                        str(translation["normalized_translated_text"]),
                        context,
                    )
                rendered = _apply_low_resource_output_contract(
                    rendered,
                    payload_language=language,
                    jailbreak=jailbreak,
                )
                localized_wrapper_language = LOCALIZED_WRAPPER_LANGUAGES.get(language)
                language_mode: Literal["no_wrapper", "monolingual", "mixed_language"] = (
                    "no_wrapper"
                    if rendered.wrapper_language is None
                    else "monolingual"
                    if localized_wrapper_language is not None
                    and rendered.wrapper_language == localized_wrapper_language
                    else "mixed_language"
                )
                variant_id = stable_id(
                    "unified-variant",
                    case.case_id,
                    str(translation["translation_id"]),
                    rendered.attack_id,
                    rendered.template_version,
                    rendered.template_sha256,
                    rendered.metadata_json,
                    rendered.wrapper_language or "",
                )
                variants.append(
                    {
                        "variant_id": variant_id,
                        "case_id": case.case_id,
                        "dataset": case.dataset or case.source,
                        "translation_id": str(translation["translation_id"]),
                        "language": language,
                        "intent": case.intent,
                        "payload": str(translation["normalized_translated_text"]),
                        "attack_id": rendered.attack_id,
                        "attack_family": rendered.attack_family,
                        "wrapper_language": rendered.wrapper_language,
                        "language_mode": language_mode,
                        "rendered_prompt": rendered.rendered_prompt,
                        "template_version": rendered.template_version,
                        "template_sha256": rendered.template_sha256,
                        "attack_metadata_json": rendered.metadata_json,
                        "source": case.source,
                    }
                )
            except Exception as error:
                error_type, error_message = _sanitized_error(error)
                errors[key] = {"error_type": error_type, "error_message": error_message}
    return variants, errors


def _write_variants(child_path: Path, variants: Sequence[Mapping[str, object]]) -> Path:
    jsonl_path = child_path / "variants.jsonl"
    _append_jsonl(jsonl_path, variants, "variant_id")
    parquet_path = child_path / "variants.parquet"
    rows = [
        PromptVariant.model_validate(row).model_dump(mode="json") for row in _read_jsonl(jsonl_path)
    ]
    if rows:
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        temporary.replace(parquet_path)
    return parquet_path


def _generation_config(settings: RunSettings) -> GenerationConfig:
    return GenerationConfig(
        system_prompt_file=None,
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        seed=settings.seed,
        retry_backoff_base=settings.retry_backoff_base,
    )


def _build_requests(
    child_id: str,
    variants_path: Path,
    settings: RunSettings,
    model_configs: Mapping[str, ModelConfig],
    model_names: tuple[str, ...],
) -> tuple[ExperimentConfig, list[tuple[str, GenerationRequest]]]:
    generation = _generation_config(settings)
    config = ExperimentConfig(
        experiment=ExperimentSection(
            id=child_id,
            datasets=["unified"],
            languages=list(PUBLIC_LANGUAGES),
            jailbreaks=list(ATTACK_IDS.values()),
            models=list(model_names),
            generation=generation,
        ),
        models=dict(model_configs),
        paths=ExperimentPaths(variants=variants_path, runs_dir=variants_path.parent.parent),
    )
    rows = [PromptVariant.model_validate(row) for row in pq.read_table(variants_path).to_pylist()]
    config_hash = stable_id(
        _canonical_json(
            {
                "models": {
                    name: model_configs[name].model_dump(mode="json") for name in model_names
                },
                "generation": generation.model_dump(mode="json"),
            }
        )
    )
    requests: list[tuple[str, GenerationRequest]] = []
    for model_name in model_names:
        model = model_configs[model_name]
        for variant in rows:
            request_id = stable_id(child_id, variant.variant_id, model_name, config_hash)
            requests.append(
                (
                    model_name,
                    GenerationRequest(
                        run_id=request_id,
                        experiment_id=child_id,
                        variant_id=variant.variant_id,
                        provider_id=model.provider,
                        requested_model_id=model.model_id,
                        endpoint_type=model.endpoint_type,
                        system_prompt=None,
                        rendered_prompt=variant.rendered_prompt,
                        temperature=generation.temperature,
                        top_p=generation.top_p,
                        max_tokens=generation.max_tokens,
                        seed=generation.seed,
                        generation_config_hash=config_hash,
                    ),
                )
            )
    return config, requests


def _read_generation_rows(child_path: Path) -> list[dict[str, object]]:
    path = child_path / "generation_results.parquet"
    if not path.is_file():
        return []
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _write_generation_rows(child_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    by_id = {str(row.get("run_id")): dict(row) for row in rows if row.get("run_id")}
    values = [by_id[key] for key in sorted(by_id)]
    path = child_path / "generation_results.parquet"
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(values), temporary)
    temporary.replace(path)


def _call_generation(
    function: Callable[..., Any],
    child_plan: RunPlan,
    config: ExperimentConfig,
    child_path: Path,
    queue: JobQueue,
    requests: list[tuple[str, GenerationRequest]],
) -> Any:
    try:
        signature = inspect.signature(function)
        count = len(
            [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
            ]
        )
    except (TypeError, ValueError):
        count = 3
    if count >= 5:
        value = function(child_plan, config, child_path, queue, requests)
    elif count == 4:
        value = function(config, child_path, queue, requests)
    elif count == 3:
        value = function(config, child_path, queue)
    elif count == 2:
        value = function(child_path, requests)
    else:
        value = function(requests)
    if inspect.isawaitable(value):
        return asyncio.run(cast(Any, value))
    return value


def _normalize_generation_output(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    values: Sequence[object]
    if isinstance(value, Mapping):
        values = list(value.values()) if "run_id" not in value and value else [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return []
    rows: list[dict[str, object]] = []
    for item in values:
        if isinstance(item, GenerationResult):
            rows.append(item.model_dump(mode="json"))
        elif isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _execute_child(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies,
    parent_path: Path,
    jailbreak: str,
    methods: Mapping[str, JailbreakMethod],
    translations: Mapping[tuple[str, str], dict[str, object]],
    translation_failures: Mapping[tuple[str, str], dict[str, object]],
    summaries: Mapping[str, object] | None,
    summary_failure: Mapping[str, object] | None = None,
) -> tuple[
    str,
    list[dict[str, object]],
    dict[tuple[str, str, str], dict[str, object]],
    str,
]:
    child_id = stable_id("unified-child", plan.run_id, jailbreak)
    child_path = parent_path / "children" / jailbreak
    child_path.mkdir(parents=True, exist_ok=True)
    child_contract = {
        "parent_run_id": plan.run_id,
        "child_id": child_id,
        "jailbreak": jailbreak,
        "attack_id": ATTACK_IDS[jailbreak],
        "models": list(plan.models),
        "languages": list(plan.languages),
        "cases": [case.case_id for case in plan.cases],
        "config": plan.contract,
    }
    contract_text = _canonical_json(child_contract) + "\n"
    contract_path = child_path / "contract.json"
    if contract_path.is_file() and contract_path.read_text(encoding="utf-8") != contract_text:
        raise ContractConflictError(f"child contract conflict: {child_path}")
    _write_text_immutable(contract_path, contract_text)
    method = methods[ATTACK_IDS[jailbreak]]
    if summary_failure is not None and (jailbreak == "psa" or jailbreak in FORMAL_PSA_CONDITIONS):
        variants: list[dict[str, object]] = []
        preparation_errors: dict[tuple[str, str], dict[str, object]] = {
            (case.case_id, language): dict(summary_failure)
            for case in plan.cases
            for language in plan.languages
            if (case.case_id, language) not in translation_failures
        }
    else:
        variants, preparation_errors = _render_variants(
            plan, settings, method, jailbreak, translations, translation_failures, summaries
        )
    variants_path = _write_variants(child_path, variants)
    model_configs = _load_model_configs(settings)
    requests: list[tuple[str, GenerationRequest]] = []
    config: ExperimentConfig | None = None
    if variants:
        config, requests = _build_requests(
            child_id, variants_path, settings, model_configs, plan.models
        )
    existing_rows = _read_generation_rows(child_path)
    generated_rows: list[dict[str, object]] = []
    child_error: dict[str, object] | None = None
    if requests:
        try:
            queue = JobQueue(child_path / "jobs.sqlite")
            try:
                assert config is not None
                queue.enqueue(requests)
                queue.reset_stale()
                request_ids = {request.run_id for _, request in requests}
                for row in existing_rows:
                    run_id = str(row.get("run_id", ""))
                    if run_id in request_ids:
                        queue.reconcile(
                            run_id,
                            str(row.get("status", "permanent_error")),
                            1,
                            cast(str | None, row.get("error_type")),
                            cast(str | None, row.get("error_message")),
                        )
                queue.retry_failed(child_id)
                pending_requests = [
                    (model_name, request) for _, model_name, _, request in queue.pending(child_id)
                ]
                status_counts = queue.status_counts(child_id)
                running = status_counts.get("running", 0)
                completed = len(requests) - len(pending_requests) - running
                dependencies.emit(
                    f"[4/5] Generate jailbreak={jailbreak} pending={len(pending_requests)} "
                    f"running={running} completed={completed}/{len(requests)}"
                )
                if dependencies.generation is not None:
                    if pending_requests:
                        generated = _call_generation(
                            dependencies.generation,
                            plan,
                            config,
                            child_path,
                            queue,
                            pending_requests,
                        )
                        generated_rows = _normalize_generation_output(generated)
                else:
                    # The existing generation service owns provider construction and retry logic.
                    emitted_progress = completed

                    def emit_progress(
                        model_name: str,
                        request: GenerationRequest,
                        result: GenerationResult,
                    ) -> None:
                        del model_name, request
                        nonlocal emitted_progress
                        emitted_progress += 1
                        newly_completed = emitted_progress - completed
                        if (
                            newly_completed == 1
                            or emitted_progress == len(requests)
                            or newly_completed % 25 == 0
                        ):
                            dependencies.emit(
                                f"[4/5] Generate jailbreak={jailbreak} "
                                f"completed={emitted_progress}/{len(requests)} "
                                f"status={result.status}"
                            )

                    asyncio.run(
                        generate_pending(
                            config,
                            child_path,
                            queue,
                            provider_factory=dependencies.provider_factory,
                            on_final_result=emit_progress,
                        )
                    )
                generated_by_id = {
                    str(row.get("run_id")): row for row in generated_rows if row.get("run_id")
                }
                for model_name, request in pending_requests:
                    generated_row = generated_by_id.get(request.run_id)
                    if generated_row is None or not queue.claim(request.run_id):
                        continue
                    queue.complete(
                        request.run_id,
                        str(generated_row.get("status", "permanent_error")),
                        1,
                        0,
                        cast(str | None, generated_row.get("error_type")),
                        cast(str | None, generated_row.get("error_message")),
                    )
                final_counts = queue.status_counts(child_id)
                final_pending = final_counts.get("pending", 0)
                final_running = final_counts.get("running", 0)
                final_completed = len(requests) - final_pending - final_running
                dependencies.emit(
                    f"[4/5] Generate jailbreak={jailbreak} pending={final_pending} "
                    f"running={final_running} completed={final_completed}/{len(requests)}"
                )
            finally:
                queue.close()
        except Exception as error:
            error_type, error_message = _sanitized_error(error)
            child_error = {"error_type": error_type, "error_message": error_message}
            _write_json(
                child_path / "child_error.json",
                {"child_id": child_id, "jailbreak": jailbreak, **child_error},
            )
    request_by_id = {request.run_id: (model_name, request) for model_name, request in requests}
    # A generation callback may return a corrected result while an older parquet snapshot still
    # contains the prior attempt.  Merge persisted rows first and let newly returned rows win.
    all_rows = existing_rows + _read_generation_rows(child_path) + generated_rows
    by_id = {str(row.get("run_id")): row for row in all_rows if row.get("run_id")}
    all_rows = [by_id[key] for key in sorted(by_id)]
    for row in all_rows:
        request_info = request_by_id.get(str(row.get("run_id")))
        if request_info is not None:
            model_name, request = request_info
            row.setdefault("model_name", model_name)
            row.setdefault("variant_id", request.variant_id)
    _write_generation_rows(child_path, all_rows)

    variant_by_id = {str(row["variant_id"]): row for row in variants}
    concrete: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in all_rows:
        request_info = request_by_id.get(str(row.get("run_id")))
        if request_info is None:
            continue
        model_name, request = request_info
        variant = variant_by_id.get(request.variant_id)
        if variant is not None:
            concrete[(str(variant["case_id"]), str(variant["language"]), model_name)] = {
                **row,
                "run_id": str(row.get("run_id")),
                "model_name": model_name,
                "variant_id": request.variant_id,
            }
    expected = len(plan.cases) * len(plan.languages) * len(plan.models)
    synth: dict[tuple[str, str, str], dict[str, object]] = {}
    for case in plan.cases:
        for language in plan.languages:
            for model in plan.models:
                key = (case.case_id, language, model)
                if key in concrete:
                    continue
                translation_failure = translation_failures.get((case.case_id, language))
                prep = preparation_errors.get((case.case_id, language))
                detail = translation_failure or prep or child_error
                if detail is None:
                    detail = {
                        "error_type": "missing_generation",
                        "error_message": "generation row is unavailable",
                    }
                synth[key] = dict(detail)
    success_count = sum(str(row.get("status", "failed")) == "success" for row in concrete.values())
    if expected > 0 and success_count == expected:
        status = "success"
    elif success_count:
        status = "partial"
    else:
        status = "failed"
    child_error_rows = []
    for (case_id, language, model), detail in sorted(synth.items()):
        child_error_rows.append(
            {
                "error_id": stable_id(
                    "child-error", plan.run_id, jailbreak, case_id, language, model
                ),
                "child_id": child_id,
                "jailbreak": jailbreak,
                "case_id": case_id,
                "language": language,
                "model": model,
                "error_type": detail.get("error_type", "failed"),
                "error_message": detail.get("error_message", "run failed"),
            }
        )
    _append_jsonl(child_path / "child_errors.jsonl", child_error_rows, "error_id")
    return status, all_rows, synth, child_id


def _compact_row(
    case: UnifiedCase,
    language: str,
    jailbreak: str,
    model: str,
    concrete: Mapping[str, object] | None,
    failure: Mapping[str, object] | None,
) -> dict[str, object]:
    if concrete is not None:
        status = str(concrete.get("status", "failed"))
        row: dict[str, object] = {
            "case_id": case.case_id,
            "source": case.source,
            "language": language,
            "jailbreak": jailbreak,
            "model": model,
            "status": status,
        }
        if status == "success":
            row["response"] = concrete.get("response_text")
        else:
            row["response"] = None
            row["error_type"] = str(concrete.get("error_type") or status)
            row["error_message"] = str(concrete.get("error_message") or status)
        return row
    return {
        "case_id": case.case_id,
        "source": case.source,
        "language": language,
        "jailbreak": jailbreak,
        "model": model,
        "status": "failed",
        "response": None,
        "error_type": str((failure or {}).get("error_type") or "failed"),
        "error_message": str((failure or {}).get("error_message") or "run failed"),
    }


def _aggregate(
    plan: RunPlan,
    parent_path: Path,
    child_outputs: Mapping[
        str, tuple[str, list[dict[str, object]], dict[tuple[str, str, str], dict[str, object]], str]
    ],
    translation_failures: Mapping[tuple[str, str], dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    child_statuses: dict[str, str] = {}
    for jailbreak in plan.jailbreaks:
        status, generation_rows, synthetic, child_id = child_outputs[jailbreak]
        child_statuses[jailbreak] = status
        variant_rows = _read_jsonl(parent_path / "children" / jailbreak / "variants.jsonl")
        variant_ids_by_tuple = {
            (str(item.get("case_id")), str(item.get("language"))): item.get("variant_id")
            for item in variant_rows
            if item.get("variant_id")
        }
        request_by_id: dict[str, tuple[str, str]] = {}
        for generation in generation_rows:
            request_by_id[str(generation.get("run_id"))] = (
                str(generation.get("model_name", "")),
                str(generation.get("variant_id", "")),
            )
        concrete_by_tuple: dict[tuple[str, str, str], dict[str, object]] = {}
        for generation in generation_rows:
            model = str(generation.get("model_name", ""))
            variant_id = str(generation.get("variant_id", ""))
            variant = next(
                (item for item in variant_rows if str(item.get("variant_id")) == variant_id), None
            )
            if variant is not None:
                concrete_by_tuple[(str(variant["case_id"]), str(variant["language"]), model)] = (
                    generation
                )
        for case in plan.cases:
            for language in plan.languages:
                for model in plan.models:
                    tuple_key = (case.case_id, language, model)
                    concrete = concrete_by_tuple.get(tuple_key)
                    failure = synthetic.get(tuple_key)
                    rows.append(_compact_row(case, language, jailbreak, model, concrete, failure))
                    if concrete is not None:
                        index_rows.append(
                            {
                                "case_id": case.case_id,
                                "source": case.source,
                                "language": language,
                                "jailbreak": jailbreak,
                                "model": model,
                                "variant_id": concrete.get("variant_id"),
                                "generation_run_id": concrete.get("run_id"),
                                "audit_record_type": "generation",
                                "audit_record_id": concrete.get("run_id"),
                            }
                        )
                    elif (case.case_id, language) in translation_failures:
                        attempt = translation_failures[(case.case_id, language)]
                        index_rows.append(
                            {
                                "case_id": case.case_id,
                                "source": case.source,
                                "language": language,
                                "jailbreak": jailbreak,
                                "model": model,
                                "variant_id": None,
                                "generation_run_id": None,
                                "audit_record_type": "translation_attempt",
                                "audit_record_id": attempt.get("attempt_id"),
                            }
                        )
                    else:
                        error_id = stable_id(
                            "child-error", plan.run_id, jailbreak, case.case_id, language, model
                        )
                        index_rows.append(
                            {
                                "case_id": case.case_id,
                                "source": case.source,
                                "language": language,
                                "jailbreak": jailbreak,
                                "model": model,
                                "variant_id": variant_ids_by_tuple.get((case.case_id, language)),
                                "generation_run_id": None,
                                "audit_record_type": "child_error",
                                "audit_record_id": error_id,
                            }
                        )
    rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["source"]),
            str(row["language"]),
            str(row["jailbreak"]),
            str(row["model"]),
        )
    )
    index_rows.sort(
        key=lambda row: (
            str(row["case_id"]),
            str(row["source"]),
            str(row["language"]),
            str(row["jailbreak"]),
            str(row["model"]),
        )
    )
    results_path = parent_path / "results.jsonl"
    results_content = "".join(_canonical_json(row) + "\n" for row in rows)
    _write_text_replace(results_path, results_content)
    index_content = "".join(_canonical_json(row) + "\n" for row in index_rows)
    _write_text_replace(parent_path / "audit" / "result_index.jsonl", index_content)
    parent_status = (
        "success"
        if child_statuses and all(status == "success" for status in child_statuses.values())
        else "partial"
        if any(status in {"success", "partial"} for status in child_statuses.values())
        else "failed"
    )
    manifest: dict[str, object] = {
        "run_id": plan.run_id,
        "status": parent_status,
        "children": child_statuses,
        "child_ids": {name: child_outputs[name][3] for name in plan.jailbreaks},
        "counts": {
            "cases": len(plan.cases),
            "translations": plan.translation_jobs,
            "psa_summaries": plan.psa_summary_count,
            "victim_requests": plan.victim_request_count,
            "results": len(rows),
        },
        "input_snapshot_sha256": plan.input_snapshot_sha256,
        "contract_sha256": _sha256_text(_canonical_json(plan.contract)),
        "fixed_configuration": {
            "models": list(plan.models),
            "languages": list(plan.languages),
            "jailbreaks": list(plan.jailbreaks),
            "translator": plan.contract["translator"],
            "translator_contract": plan.contract["translator_contract"],
            "gra_role": plan.contract["gra_role"],
        },
        "created_at": _utc_now(),
    }
    _write_json(parent_path / "run_manifest.json", manifest)
    write_hierarchical_reports(parent_path)
    return rows, child_statuses, manifest


def _requires_dotenv(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies,
) -> bool:
    if (
        plan.translation_jobs
        and settings.translator == GOOGLE_CLOUD_TRANSLATOR
        and dependencies.translator is None
        and dependencies.translator_factory is None
        and dependencies.google_adc_preflight is None
    ):
        return True
    model_configs = _load_model_configs(settings)
    if dependencies.generation is None and dependencies.provider_factory is None:
        if any(model_configs[name].provider != "fake" for name in plan.models):
            return True
    if (
        any(item == "psa" or item in FORMAL_PSA_CONDITIONS for item in plan.jailbreaks)
        and dependencies.summary_service is None
        and dependencies.summary_service_factory is None
    ):
        summary_model = model_configs.get("gemma_4_12b")
        return summary_model is not None and summary_model.provider != "fake"
    return False


def execute_run(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies | None = None,
) -> RunExecution:
    dependencies = dependencies or RunDependencies()
    # Formal execution reads only the explicitly scoped current-working-directory dotenv file,
    # and only when a selected non-test provider needs environment-backed credentials.
    dotenv_path = Path.cwd() / ".env"
    if _requires_dotenv(plan, settings, dependencies) and dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path)
    emit = dependencies.emit
    emit(
        f"[1/5] Plan cases={len(plan.cases)} translations={plan.translation_jobs} "
        f"summaries={plan.psa_summary_count} victim_requests={plan.victim_request_count}"
    )
    preflight_run(plan, settings, dependencies)
    parent_path = plan.parent_path
    parent_path.mkdir(parents=True, exist_ok=True)
    contract_text = _canonical_json(plan.contract) + "\n"
    _write_text_immutable(parent_path / "run_contract.json", contract_text)
    audit = parent_path / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    input_snapshot = "".join(
        _canonical_json(case.model_dump(mode="json")) + "\n" for case in plan.cases
    )
    _write_text_immutable(audit / "input_snapshot.jsonl", input_snapshot)

    emit(
        f"[2/5] Translate count={plan.translation_jobs}"
        if plan.translation_jobs
        else "[2/5] Translate skipped"
    )
    translations, translation_failures = _translate_cases(plan, settings, dependencies, parent_path)

    methods = load_jailbreaks(settings.jailbreaks_config)
    psa_conditions = tuple(
        item for item in plan.jailbreaks if item == "psa" or item in FORMAL_PSA_CONDITIONS
    )
    summaries_by_condition: dict[str, Mapping[str, object]] = {}
    summary_failures: dict[str, dict[str, object]] = {}
    if psa_conditions:
        emit(
            f"[3/5] Summarize count={plan.psa_summary_count} "
            f"localizations={plan.psa_localization_count}"
        )
        for condition in psa_conditions:
            method = methods[ATTACK_IDS[condition]]
            if not isinstance(method, PaperSummaryJailbreak):
                raise ValueError("psa configuration must use PaperSummaryJailbreak")
            try:
                if condition in FORMAL_PSA_CONDITIONS:
                    summaries_by_condition[condition] = _formal_summary_cache(
                        condition,
                        method,
                        plan,
                        settings,
                        dependencies,
                        parent_path,
                    )
                else:
                    summaries_by_condition[condition] = cast(
                        Mapping[str, object],
                        _summary_cache(method, settings, dependencies, parent_path),
                    )
            except Exception as error:
                error_type, error_message = _sanitized_error(error)
                if condition in FORMAL_PSA_CONDITIONS:
                    raise ContractConflictError(
                        f"formal PSA preparation failed: {condition}: {error_message}"
                    ) from error
                summary_failures[condition] = {
                    "error_type": error_type,
                    "error_message": error_message,
                }
                _write_json(
                    parent_path / "audit" / f"{condition}_summary_error.json",
                    summary_failures[condition],
                )
    else:
        emit("[3/5] Summarize skipped")

    model_configs = _load_model_configs(settings)
    requests_per_model = len(plan.cases) * len(plan.languages) * len(plan.jailbreaks)
    provider_limits: dict[str, tuple[int, int, int]] = {}
    for model_name in plan.models:
        model = model_configs[model_name]
        current = provider_limits.get(model.provider)
        if current is None:
            provider_limits[model.provider] = (
                model.concurrency,
                model.requests_per_minute,
                requests_per_model,
            )
        else:
            provider_limits[model.provider] = (
                min(current[0], model.concurrency),
                min(current[1], model.requests_per_minute),
                current[2] + requests_per_model,
            )
    limit_details = ",".join(
        f"{provider}(concurrency={concurrency},rpm={rpm},lower_bound="
        f"{max(0, request_count - 1) / rpm:.1f}m)"
        for provider, (concurrency, rpm, request_count) in sorted(provider_limits.items())
    )
    emit(f"[4/5] Generate count={plan.victim_request_count} provider_limits={limit_details}")
    child_outputs: dict[
        str, tuple[str, list[dict[str, object]], dict[tuple[str, str, str], dict[str, object]], str]
    ] = {}
    for jailbreak in plan.jailbreaks:
        child_outputs[jailbreak] = _execute_child(
            plan,
            settings,
            dependencies,
            parent_path,
            jailbreak,
            methods,
            translations,
            translation_failures,
            summaries_by_condition.get(jailbreak),
            summary_failures.get(jailbreak),
        )

    emit("[5/5] Aggregate")
    rows, child_statuses, manifest = _aggregate(
        plan, parent_path, child_outputs, translation_failures
    )
    result_path = parent_path / "results.jsonl"
    emit(f"run_id={plan.run_id}")
    emit(f"status={manifest['status']}")
    emit(f"results={result_path}")
    return RunExecution(
        run_id=plan.run_id,
        status=str(manifest["status"]),
        parent_path=parent_path,
        results_path=result_path,
        manifest=manifest,
        rows=rows,
        child_statuses=child_statuses,
    )


__all__ = [
    "ATTACK_IDS",
    "PUBLIC_JAILBREAKS",
    "PUBLIC_LANGUAGES",
    "PUBLIC_SOURCES",
    "RunDependencies",
    "RunExecution",
    "RunPlan",
    "RunRequest",
    "RunSettings",
    "UnifiedCase",
    "WRAPPER_LANGUAGES",
    "ContractConflictError",
    "execute_run",
    "load_run_settings",
    "parse_selection",
    "parse_jailbreak_selection",
    "plan_run",
    "preflight_run",
]
