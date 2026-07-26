from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.jailbreaks import load_jailbreaks
from crosslingual_safety.psa_summary import (
    SUMMARY_KEYS,
    PaperSummaryService,
)
from crosslingual_safety.schemas import GenerationRequest, GenerationResult
from crosslingual_safety.translation.providers import FakeTranslator
from crosslingual_safety.unified_run import (
    PUBLIC_JAILBREAKS,
    PUBLIC_LANGUAGES,
    RunDependencies,
    RunRequest,
    RunSettings,
    _nllb_checkpoint_available,
    _resolved_nllb_checkpoint,
    execute_run,
    load_run_settings,
    parse_selection,
    plan_run,
    preflight_run,
)

runner = CliRunner()


@pytest.mark.parametrize(
    ("value", "allowed", "expected"),
    [
        ("all", PUBLIC_LANGUAGES, PUBLIC_LANGUAGES),
        (" my, en, my ", PUBLIC_LANGUAGES, ("en", "my")),
        ("gra,none", PUBLIC_JAILBREAKS, ("none", "gra")),
    ],
)
def test_parse_selection_normalizes_and_orders(
    value: str, allowed: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert parse_selection(value, allowed, "--selection") == expected


def test_parse_selection_rejects_internal_zh() -> None:
    with pytest.raises(ValueError):
        parse_selection("zh", PUBLIC_LANGUAGES, "--language")


def test_run_help_exposes_only_public_experiment_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output
    assert "--language" in result.output
    assert "--jailbreak" in result.output
    assert "--dry-run" in result.output
    for forbidden in ("--model", "--role", "--translator", "--source-language", "--max-tokens"):
        assert forbidden not in result.output


def _settings(tmp_path: Path) -> RunSettings:
    configs = tmp_path / "configs"
    prompts = tmp_path / "prompts"
    configs.mkdir()
    prompts.mkdir()
    (prompts / "prompt.txt").write_text("Prompt", encoding="utf-8")
    (configs / "run.yaml").write_text(
        """version: 1
manual:
  input_path: prompts/prompt.txt
  source_language: zh-tw
models: [fake_model]
translator: fake
wrapper_language_mode: same-as-payload
gra_role: joker
models_config: configs/models.yaml
languages_config: configs/languages.yaml
jailbreaks_config: configs/jailbreaks.yaml
runs_dir: runs/experiments
""",
        encoding="utf-8",
    )
    (configs / "models.yaml").write_text(
        """models:
  fake_model:
    provider: fake
    model_id: fake-model
    endpoint_type: chat
    concurrency: 1
    requests_per_minute: 6000
    test_only: true
""",
        encoding="utf-8",
    )
    shutil.copy(Path("configs/languages.yaml"), configs / "languages.yaml")
    shutil.copy(Path("configs/jailbreaks.yaml"), configs / "jailbreaks.yaml")
    return load_run_settings(configs / "run.yaml")


def _generation_result(request: GenerationRequest, status: str) -> GenerationResult:
    return GenerationResult(
        run_id=request.run_id,
        status=status,  # type: ignore[arg-type]
        response_text="offline" if status == "success" else None,
        actual_model_id=request.requested_model_id if status == "success" else None,
        raw_response_path=None,
        finish_reason="stop" if status == "success" else None,
        prompt_tokens=1 if status == "success" else None,
        completion_tokens=1 if status == "success" else None,
        latency_ms=0.0,
        provider_request_id=None,
        error_type=None if status == "success" else status,
        error_message=None if status == "success" else f"{status} response",
    )


def _write_minimal_nllb_snapshot(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


class _SummaryProvider:
    provider_id = "summary-test"

    def __init__(self, fail_language: str | None = None) -> None:
        self.fail_language = fail_language
        self.languages: list[str] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        language = request.variant_id.removeprefix("summary-")
        self.languages.append(language)
        if language == self.fail_language:
            raise RuntimeError("synthetic summary provider failure")
        response_text = json.dumps(
            {key: f"{key}-{language}" for key in SUMMARY_KEYS},
            sort_keys=True,
            separators=(",", ":"),
        )
        return GenerationResult(
            run_id=request.run_id,
            status="success",
            response_text=response_text,
            actual_model_id=request.requested_model_id,
            raw_response_path=None,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
            provider_request_id=f"summary-{language}",
            error_type=None,
            error_message=None,
        )


def _summary_service(settings: RunSettings, provider: _SummaryProvider) -> PaperSummaryService:
    method = load_jailbreaks(settings.jailbreaks_config)["psa_static_v1"]
    return PaperSummaryService.from_method(
        method,
        provider=provider,
        clock=lambda: "2026-07-26T00:00:00Z",
    )


def _generate_success(
    config: object,
    path: Path,
    queue: object,
    requests: list[tuple[str, GenerationRequest]],
) -> list[GenerationResult]:
    del config, path, queue
    return [_generation_result(request, "success") for _, request in requests]


def test_manual_plan_counts_identity_separately(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(
        RunRequest(source="manual", languages=("en", "zh-tw"), jailbreaks=("none",)), settings
    )
    assert len(plan.cases) == 1
    assert plan.translation_jobs == 1
    assert plan.victim_request_count == 2
    assert (
        plan.run_id
        == plan_run(
            RunRequest(source="manual", languages=("en", "zh-tw"), jailbreaks=("none",)), settings
        ).run_id
    )


def test_manual_plan_uses_configured_source_language(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.config_path is not None
    settings.config_path.write_text(
        settings.config_path.read_text(encoding="utf-8").replace(
            "source_language: zh-tw", "source_language: en"
        ),
        encoding="utf-8",
    )
    settings = load_run_settings(settings.config_path)

    plan = plan_run(
        RunRequest(source="manual", languages=("en",), jailbreaks=("none",)),
        settings,
    )

    assert plan.cases[0].source_language == "en"
    assert plan.translation_jobs == 0


def test_fake_execution_writes_sparse_public_results(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("zh-tw",), jailbreaks=("none",)), settings)

    def generate(
        config: object,
        path: Path,
        queue: object,
        requests: list[tuple[str, GenerationRequest]],
    ) -> list[GenerationResult]:
        del config, path, queue
        return [
            GenerationResult(
                run_id=request.run_id,
                status="success",
                response_text="offline",
                actual_model_id=request.requested_model_id,
                raw_response_path=None,
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=0.0,
                provider_request_id=None,
                error_type=None,
                error_message=None,
            )
            for _, request in requests
        ]

    result = execute_run(
        plan,
        settings,
        RunDependencies(translator=FakeTranslator(), generation=generate, emit=lambda _: None),
    )
    row = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert set(row) == {
        "case_id",
        "source",
        "language",
        "jailbreak",
        "model",
        "status",
        "response",
    }
    assert row["language"] == "zh-tw"
    translations = (result.parent_path / "audit" / "translations.jsonl").read_text(encoding="utf-8")
    assert '"target_language":"zh"' not in translations


def test_formal_execute_loads_credentials_from_cwd_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.models_config.write_text(
        """models:
  fake_model:
    provider: remote
    base_url_env: TEST_BASE_URL
    api_key_env: TEST_API_KEY
    model_id: remote-model
    endpoint_type: chat
    concurrency: 1
    requests_per_minute: 6000
""",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "TEST_BASE_URL=https://example.invalid/v1\nTEST_API_KEY=local-test-key\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    plan = plan_run(RunRequest(languages=("zh-tw",), jailbreaks=("none",)), settings)

    async def fake_generate(
        config: object, path: Path, queue: object, provider_factory: object = None
    ) -> int:
        del config, path, queue, provider_factory
        return 0

    # The default generation boundary is used so preflight actually checks the remote model
    # environment; the monkeypatched service keeps this test offline.
    monkeypatch.setattr("crosslingual_safety.unified_run.generate_pending", fake_generate)
    result = execute_run(plan, settings, RunDependencies(emit=lambda _: None))
    assert result.status == "failed"


def test_resume_new_result_replaces_prior_generation_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("zh-tw",), jailbreaks=("none",)), settings)
    attempts = 0

    def generate(
        config: object,
        path: Path,
        queue: object,
        requests: list[tuple[str, GenerationRequest]],
    ) -> list[GenerationResult]:
        del config, path, queue
        nonlocal attempts
        attempts += 1
        status = "server_error" if attempts == 1 else "success"
        return [_generation_result(request, status) for _, request in requests]

    dependencies = RunDependencies(
        translator=FakeTranslator(), generation=generate, emit=lambda _: None
    )
    first = execute_run(plan, settings, dependencies)
    first_row = json.loads(first.results_path.read_text(encoding="utf-8"))
    assert first_row["status"] == "server_error"
    assert first_row["response"] is None

    second = execute_run(plan, settings, dependencies)
    second_row = json.loads(second.results_path.read_text(encoding="utf-8"))
    assert second_row["status"] == "success"
    assert second_row["response"] == "offline"
    parquet_rows = pq.read_table(
        second.parent_path / "children" / "none" / "generation_results.parquet"
    ).to_pylist()
    assert [row["status"] for row in parquet_rows] == ["success"]


@pytest.mark.parametrize(
    ("languages", "statuses", "expected_status"),
    [
        (("zh-tw",), ("server_error",), "failed"),
        (("zh-tw", "en"), ("success", "server_error"), "partial"),
    ],
)
def test_child_status_reflects_each_generation_row(
    tmp_path: Path,
    languages: tuple[str, ...],
    statuses: tuple[str, ...],
    expected_status: str,
) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=languages, jailbreaks=("none",)), settings)

    def generate(
        config: object,
        path: Path,
        queue: object,
        requests: list[tuple[str, GenerationRequest]],
    ) -> list[GenerationResult]:
        del config, path, queue
        assert len(requests) == len(statuses)
        return [
            _generation_result(request, status)
            for (_, request), status in zip(requests, statuses, strict=True)
        ]

    result = execute_run(
        plan,
        settings,
        RunDependencies(translator=FakeTranslator(), generation=generate, emit=lambda _: None),
    )
    assert result.child_statuses["none"] == expected_status
    assert result.manifest["status"] == expected_status
    public_rows = [
        json.loads(line) for line in result.results_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sorted(row["status"] for row in public_rows) == sorted(statuses)


def test_malformed_psa_cache_is_quarantined_and_regenerated(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(
        RunRequest(languages=("zh-tw",), jailbreaks=("none", "psa")),
        settings,
    )
    audit = plan.parent_path / "audit"
    audit.mkdir(parents=True)
    malformed = b"{malformed summary cache\n"
    cache_path = audit / "psa_summary_artifacts.jsonl"
    cache_path.write_bytes(malformed)
    provider = _SummaryProvider()
    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=FakeTranslator(),
            summary_service=_summary_service(settings, provider),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )
    digest = hashlib.sha256(malformed).hexdigest()
    quarantine = audit / f"psa_summary_artifacts.quarantine.{digest}.jsonl"
    assert quarantine.read_bytes() == malformed
    assert cache_path.is_file()
    assert len(provider.languages) == 4
    assert result.child_statuses == {"none": "success", "psa": "success"}
    assert result.status == "success"
    assert len(_summary_service(settings, _SummaryProvider()).load_cache(cache_path)) == 4


def test_malformed_psa_cache_stays_quarantined_when_summary_regeneration_fails(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(
        RunRequest(languages=("zh-tw",), jailbreaks=("none", "psa")),
        settings,
    )
    audit = plan.parent_path / "audit"
    audit.mkdir(parents=True)
    malformed = b"not valid summary jsonl\n"
    cache_path = audit / "psa_summary_artifacts.jsonl"
    cache_path.write_bytes(malformed)
    provider = _SummaryProvider(fail_language="vi")
    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=FakeTranslator(),
            summary_service=_summary_service(settings, provider),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )
    digest = hashlib.sha256(malformed).hexdigest()
    quarantine = audit / f"psa_summary_artifacts.quarantine.{digest}.jsonl"
    assert quarantine.read_bytes() == malformed
    assert not cache_path.exists()
    assert result.child_statuses == {"none": "success", "psa": "failed"}
    assert result.status == "partial"


class _AlwaysFailTranslator:
    translator_id = "failing"
    version = "1"
    method = "test"
    decoding_config: dict[str, object] = {}

    def supports(self, source_language: str, target_language: str) -> bool:
        return source_language != target_language

    def translate(self, text: str, source_language: str, target_language: str) -> object:
        del text, source_language, target_language
        raise RuntimeError("synthetic translation failure")


def test_repeated_translation_failure_appends_distinct_attempts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    dependencies = RunDependencies(translator=_AlwaysFailTranslator(), emit=lambda _: None)
    execute_run(plan, settings, dependencies)
    execute_run(plan, settings, dependencies)
    rows = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2
    assert sorted(row["attempt_number"] for row in rows) == [1, 2]
    assert len({row["attempt_id"] for row in rows}) == 2


def test_child_failure_index_retains_variant_id_and_public_response_null(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("zh-tw",), jailbreaks=("gra",)), settings)

    def fail_generation(
        config: object,
        path: Path,
        queue: object,
        requests: list[tuple[str, GenerationRequest]],
    ) -> None:
        del config, path, queue, requests
        raise RuntimeError("child generation failed")

    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=FakeTranslator(), generation=fail_generation, emit=lambda _: None
        ),
    )
    public_row = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert public_row["response"] is None
    assert public_row["error_type"]
    index_row = json.loads(
        (result.parent_path / "audit" / "result_index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert index_row["audit_record_type"] == "child_error"
    assert index_row["variant_id"]
    assert index_row["generation_run_id"] is None


def test_nllb_preflight_rejects_missing_local_checkpoint(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.translator = "nllb"
    settings.nllb_checkpoint = str(tmp_path / "missing-nllb-checkpoint")
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    with pytest.raises(ValueError, match="NLLB checkpoint is not available locally"):
        preflight_run(plan, settings)


def test_nllb_preflight_rejects_empty_checkpoint_directory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.translator = "nllb"
    empty_checkpoint = tmp_path / "empty-nllb-checkpoint"
    empty_checkpoint.mkdir()
    settings.nllb_checkpoint = str(empty_checkpoint)
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    with pytest.raises(ValueError, match="NLLB checkpoint is not available locally"):
        preflight_run(plan, settings)


def test_nllb_preflight_accepts_minimal_local_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.translator = "nllb"
    snapshot = tmp_path / "nllb-snapshot"
    _write_minimal_nllb_snapshot(snapshot)
    settings.nllb_checkpoint = str(snapshot)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    preflight_run(plan, settings)
    assert _nllb_checkpoint_available(settings)
    assert Path(_resolved_nllb_checkpoint(settings)) == snapshot.resolve()


def test_nllb_preflight_resolves_huggingface_cache_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.translator = "nllb"
    settings.nllb_checkpoint = "facebook/nllb-200-distilled-600M"
    cache_root = tmp_path / "hf-hub"
    snapshot = (
        cache_root / "models--facebook--nllb-200-distilled-600M" / "snapshots" / "fixture-revision"
    )
    _write_minimal_nllb_snapshot(snapshot)
    refs = snapshot.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text("fixture-revision\n", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    preflight_run(plan, settings)
    assert Path(_resolved_nllb_checkpoint(settings)) == snapshot.resolve()
