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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

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
    PaperSummaryJailbreak,
    load_jailbreaks,
)
from crosslingual_safety.psa_summary import (
    SUMMARY_LANGUAGES,
    PaperSummaryService,
    SummaryArtifact,
    artifact_sections,
)
from crosslingual_safety.schemas import GenerationRequest, GenerationResult, PromptVariant
from crosslingual_safety.translation.languages import load_languages
from crosslingual_safety.translation.providers import (
    DatasetTranslationProvider,
    FakeTranslator,
    NLLBTranslator,
    ProviderTranslation,
    Translator,
)

PUBLIC_LANGUAGES: tuple[str, ...] = ("en", "zh-tw", "vi", "my")
PUBLIC_JAILBREAKS: tuple[str, ...] = ("none", "gra", "psa")
PUBLIC_SOURCES: tuple[str, ...] = ("manual", "bench")
WRAPPER_LANGUAGES: dict[str, str] = {
    "en": "en",
    "zh-tw": "zh",
    "vi": "vi",
    "my": "my",
}
ATTACK_IDS: dict[str, str] = {"none": "none", "gra": "gra_v1", "psa": "psa_static_v1"}
SUMMARY_WRAPPER_LANGUAGES: tuple[str, ...] = ("en", "zh", "vi", "my")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sanitized_error(error: BaseException) -> tuple[str, str]:
    """Return an error safe to persist or show in the terminal.

    Provider adapters already redact response bodies.  The facade still uses a conservative
    message because arbitrary exceptions can contain a prompt, URL query, or credential.
    """

    error_type = type(error).__name__
    message = str(error).strip()
    if not message or len(message) > 240:
        message = error_type
    for secret_name in ("ZOOLAB_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(secret_name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    if "http" in message.lower() and "" in message:
        # Do not preserve complete provider URLs in an audit failure.
        message = f"{error_type} provider request failed"
    return error_type, message


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


class ManualSettings(BaseModel):
    input_path: Path = Path("prompts/prompt.txt")
    source_language: str = "zh-tw"


class BenchSettings(BaseModel):
    cases_path: Path = Path("data/normalized/cases.parquet")
    selection_path: Path = Path("data/normalized/variant_case_selection.parquet")


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
    wrapper_language_mode: Literal["same-as-payload", "english"] = "same-as-payload"
    gra_role: str = "joker"
    models_config: Path = Path("configs/models.yaml")
    languages_config: Path = Path("configs/languages.yaml")
    jailbreaks_config: Path = Path("configs/jailbreaks.yaml")
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
    dry_run: bool = False

    @field_validator("languages", "jailbreaks", mode="before")
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
    value["manual"] = manual
    value["bench"] = bench
    for key, default in (
        ("models_config", "configs/models.yaml"),
        ("languages_config", "configs/languages.yaml"),
        ("jailbreaks_config", "configs/jailbreaks.yaml"),
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


def plan_run(request: RunRequest, settings: RunSettings) -> RunPlan:
    if request.source not in PUBLIC_SOURCES:
        raise ValueError("source must be manual or bench")
    languages = parse_selection(",".join(request.languages), PUBLIC_LANGUAGES, "--language")
    jailbreaks = parse_selection(",".join(request.jailbreaks), PUBLIC_JAILBREAKS, "--jailbreak")
    if "zh" in languages:
        raise ValueError("--language uses zh-tw for Traditional Chinese")
    normalized_request = request.model_copy(
        update={"languages": languages, "jailbreaks": jailbreaks}
    )
    cases = _load_cases(normalized_request, settings)
    translation_jobs = sum(
        1 for case in cases for language in languages if language != case.source_language
    )
    psa_summary_count = len(SUMMARY_LANGUAGES) if "psa" in jailbreaks else 0
    victim_request_count = len(cases) * len(languages) * len(jailbreaks) * len(settings.models)
    input_snapshot = "".join(_canonical_json(case.model_dump(mode="json")) + "\n" for case in cases)
    input_snapshot_sha256 = _sha256_text(input_snapshot)
    models = _raw_models(settings)
    selected_models = {name: models.get(name) for name in settings.models}
    contract: dict[str, object] = {
        "version": settings.version,
        "request": normalized_request.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
        "input_snapshot_sha256": input_snapshot_sha256,
        "models": selected_models,
        "model_names": list(settings.models),
        "translator": settings.translator,
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
        },
        "attack_ids": {name: ATTACK_IDS[name] for name in jailbreaks},
    }
    run_id = stable_id("experiment-run", _canonical_json(contract))
    return RunPlan(
        request=normalized_request,
        cases=cases,
        models=tuple(settings.models),
        languages=languages,
        jailbreaks=jailbreaks,
        translation_jobs=translation_jobs,
        psa_summary_count=psa_summary_count,
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
        if not settings.bench.cases_path.is_file() or not settings.bench.selection_path.is_file():
            raise ValueError("benchmark cases and selection snapshots are required")
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
            if not method.supports_language(WRAPPER_LANGUAGES[language]):
                raise ValueError(
                    f"{jailbreak} does not support wrapper language {WRAPPER_LANGUAGES[language]}"
                )
        if jailbreak == "gra":
            personas = getattr(method, "personas", {})
            if settings.gra_role not in personas:
                raise ValueError(f"unknown GRA role: {settings.gra_role}")
        if jailbreak == "psa" and not isinstance(method, PaperSummaryJailbreak):
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
        elif settings.translator not in {"fake", "dataset"}:
            raise ValueError(f"unsupported translator: {settings.translator}")

    # A fake generation seam is sufficient for tests and intentionally bypasses credential checks,
    # but endpoint metadata remains part of the selected configuration contract.
    for model in model_configs.values():
        if model.provider == "fake":
            if not model.test_only:
                raise ValueError("FakeProvider requires test_only=true")
            continue
        if not model.base_url_env or not model.api_key_env:
            raise ValueError(f"provider {model.provider} requires endpoint metadata")
    if dependencies.generation is None and dependencies.provider_factory is None:
        for model in model_configs.values():
            if model.provider == "fake":
                continue
            assert model.base_url_env is not None and model.api_key_env is not None
            if not os.environ.get(model.base_url_env) or not os.environ.get(model.api_key_env):
                raise ValueError(
                    f"required provider environment variable is unset: {model.base_url_env} or "
                    f"{model.api_key_env}"
                )
    if (
        "psa" in plan.jailbreaks
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


def _make_translator(settings: RunSettings, dependencies: RunDependencies) -> Translator | None:
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


def _write_text_replace(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


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


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, object]], key: str) -> None:
    existing = _read_jsonl(path)
    by_key = {str(row.get(key)): row for row in existing if key in row}
    for row in rows:
        row_dict = dict(row)
        row_key = str(row_dict.get(key, stable_id(_canonical_json(row_dict))))
        prior = by_key.get(row_key)
        if prior is not None and _canonical_json(prior) != _canonical_json(row_dict):
            raise ContractConflictError(f"immutable JSONL row conflict: {path} ({row_key})")
        by_key[row_key] = row_dict
    ordered = sorted(by_key.values(), key=lambda value: str(value.get(key, "")))
    content = "".join(_canonical_json(dict(row)) + "\n" for row in ordered)
    _write_text_replace(path, content)


def _translation_record(
    case: UnifiedCase,
    language: str,
    translated_text: str,
    translator: Translator | None,
    provider_request_id: str | None,
    clock: Callable[[], str],
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
    return {
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
        "decoding_config": decoding,
        "source_text_sha256": _sha256_text(case.source_text),
        "translated_text_sha256": _sha256_text(normalized),
        "provider_request_id": provider_request_id,
        "created_at": clock(),
        "frozen": False,
        "review_status": "pending",
    }


def _translate_cases(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies,
    parent_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    audit = parent_path / "audit"
    translations_path = audit / "translations.jsonl"
    attempts_path = audit / "translation_attempts.jsonl"
    existing = _read_jsonl(translations_path)
    successful: dict[tuple[str, str], dict[str, object]] = {
        (str(row.get("case_id")), str(row.get("target_language"))): row
        for row in existing
        if row.get("case_id")
        and row.get("target_language")
        and row.get("normalized_translated_text")
    }
    failures: dict[tuple[str, str], dict[str, object]] = {}
    attempts = _read_jsonl(attempts_path)
    attempt_counts: dict[tuple[str, str], int] = {}
    for row in attempts:
        key = (str(row.get("case_id")), str(row.get("target_language")))
        try:
            sequence = int(cast(Any, row.get("attempt_number", 0)))
        except (TypeError, ValueError):
            sequence = 0
        attempt_counts[key] = max(attempt_counts.get(key, 0), sequence)
    translator = _make_translator(settings, dependencies) if plan.translation_jobs else None
    new_records: list[dict[str, object]] = []
    new_attempts: list[dict[str, object]] = []
    for case in plan.cases:
        for language in plan.languages:
            key = (case.case_id, language)
            if key in successful:
                continue
            if language == case.source_language:
                successful[key] = _translation_record(
                    case, language, case.source_text, None, None, dependencies.clock
                )
                new_records.append(successful[key])
                continue
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
                )
                successful[key] = record
                new_records.append(record)
            except Exception as error:
                error_type, error_message = _sanitized_error(error)
                attempt_number = attempt_counts.get(key, 0) + 1
                attempt_counts[key] = attempt_number
                attempt: dict[str, object] = {
                    "attempt_id": stable_id(
                        "translation-attempt",
                        case.case_id,
                        language,
                        error_type,
                        str(attempt_number),
                    ),
                    "attempt_number": attempt_number,
                    "case_id": case.case_id,
                    "source": case.source,
                    "source_language": case.source_language,
                    "target_language": language,
                    "error_type": error_type,
                    "error_message": error_message,
                    "created_at": dependencies.clock(),
                }
                new_attempts.append(attempt)
                failures[key] = attempt
    _append_jsonl(translations_path, new_records, "translation_id")
    _append_jsonl(attempts_path, new_attempts, "attempt_id")
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
            wrapper_language = WRAPPER_LANGUAGES[language]
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
                language_mode: Literal["no_wrapper", "monolingual", "mixed_language"] = (
                    "no_wrapper"
                    if rendered.wrapper_language is None
                    else "monolingual"
                    if rendered.wrapper_language == WRAPPER_LANGUAGES[language]
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
) -> tuple[ExperimentConfig, list[tuple[str, GenerationRequest]]]:
    generation = _generation_config(settings)
    config = ExperimentConfig(
        experiment=ExperimentSection(
            id=child_id,
            datasets=["unified"],
            languages=list(PUBLIC_LANGUAGES),
            jailbreaks=list(ATTACK_IDS.values()),
            models=list(settings.models),
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
                    name: model_configs[name].model_dump(mode="json") for name in settings.models
                },
                "generation": generation.model_dump(mode="json"),
            }
        )
    )
    requests: list[tuple[str, GenerationRequest]] = []
    for model_name in settings.models:
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
    if summary_failure is not None and jailbreak == "psa":
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
        config, requests = _build_requests(child_id, variants_path, settings, model_configs)
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
                    asyncio.run(
                        generate_pending(
                            config,
                            child_path,
                            queue,
                            provider_factory=dependencies.provider_factory,
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
    report_lines = [f"# Unified Run {plan.run_id}", ""]
    current: tuple[str, str, str] | None = None
    for row in rows:
        group = (str(row["case_id"]), str(row["language"]), str(row["jailbreak"]))
        if group != current:
            report_lines.extend([f"## {group[0]} / {group[1]} / {group[2]}", ""])
            current = group
        report_lines.extend([f"### {row['model']}", "", f"Status: `{row['status']}`", ""])
        response = row.get("response", row.get("error_message", ""))
        report_lines.extend(["```text", str(response or ""), "```", ""])
    _write_text_replace(parent_path / "report.md", "\n".join(report_lines).rstrip() + "\n")
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
            "translator": "nllb",
            "gra_role": "joker",
        },
        "created_at": _utc_now(),
    }
    _write_json(parent_path / "run_manifest.json", manifest)
    return rows, child_statuses, manifest


def execute_run(
    plan: RunPlan,
    settings: RunSettings,
    dependencies: RunDependencies | None = None,
) -> RunExecution:
    dependencies = dependencies or RunDependencies()
    # Formal execution may read credentials from the current working directory.  Planning and
    # dry-run paths never call this function, so they remain side-effect free.
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()
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
    summaries: Mapping[str, object] | None = None
    summary_failure: dict[str, object] | None = None
    if "psa" in plan.jailbreaks:
        emit(f"[3/5] Summarize count={plan.psa_summary_count}")
        psa_method = methods[ATTACK_IDS["psa"]]
        if not isinstance(psa_method, PaperSummaryJailbreak):
            raise ValueError("psa configuration must use PaperSummaryJailbreak")
        try:
            summaries = cast(
                Mapping[str, object],
                _summary_cache(psa_method, settings, dependencies, parent_path),
            )
        except Exception as error:
            error_type, error_message = _sanitized_error(error)
            summary_failure = {
                "error_type": error_type,
                "error_message": error_message,
            }
            _write_json(
                parent_path / "audit" / "psa_summary_error.json",
                summary_failure,
            )
    else:
        emit("[3/5] Summarize skipped")

    emit(f"[4/5] Generate count={plan.victim_request_count}")
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
            summaries,
            summary_failure,
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
    "plan_run",
    "preflight_run",
]
