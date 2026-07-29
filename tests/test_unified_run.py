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
from crosslingual_safety.translation.paid_ledger import PaidCallLedger, PaidTranslationTask
from crosslingual_safety.translation.providers import (
    FakeTranslator,
    GoogleCloudAuthenticationError,
    GoogleCloudNMTTranslator,
    GoogleCloudRequestTooLargeError,
    GoogleCloudRunBudgetExceededError,
)
from crosslingual_safety.unified_run import (
    PUBLIC_JAILBREAKS,
    PUBLIC_LANGUAGES,
    ContractConflictError,
    RunDependencies,
    RunRequest,
    RunSettings,
    _append_jsonl,
    _nllb_checkpoint_available,
    _read_generation_rows,
    _render_variants,
    _resolved_nllb_checkpoint,
    _translate_cases,
    _write_generation_rows,
    execute_run,
    load_run_settings,
    parse_jailbreak_selection,
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
        (
            "psa_defense_r2d_v2,none",
            PUBLIC_JAILBREAKS,
            ("none", "psa_defense_r2d_v2"),
        ),
    ],
)
def test_parse_selection_normalizes_and_orders(
    value: str, allowed: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert parse_selection(value, allowed, "--selection") == expected


def test_parse_selection_rejects_internal_zh() -> None:
    with pytest.raises(ValueError):
        parse_selection("zh", PUBLIC_LANGUAGES, "--language")


def test_jailbreak_all_uses_only_formal_conditions() -> None:
    assert parse_jailbreak_selection("all") == (
        "none",
        "psa_attack_poetry_v2",
        "psa_defense_r2d_v2",
    )
    assert parse_jailbreak_selection("gra,psa") == ("gra", "psa")
    assert parse_jailbreak_selection("psa_attack_poetry_v1,psa_defense_r2d_v1") == (
        "psa_attack_poetry_v1",
        "psa_defense_r2d_v1",
    )


def test_public_languages_include_low_resource_manual_targets() -> None:
    assert PUBLIC_LANGUAGES == ("en", "zh-tw", "jv", "my", "th", "vi", "tl", "eo")


def test_run_help_exposes_only_public_experiment_options() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output
    assert "--language" in result.output
    assert "--jailbreak" in result.output
    assert "psa_attack_poetry_v2" in result.output
    assert "psa_defense_r2d_v2" in result.output
    assert "--model" in result.output
    assert "--dry-run" in result.output
    for forbidden in ("--role", "--translator", "--source-language", "--max-tokens"):
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


def _settings_with_two_models(tmp_path: Path) -> RunSettings:
    settings = _settings(tmp_path)
    assert settings.config_path is not None
    settings.config_path.write_text(
        settings.config_path.read_text(encoding="utf-8").replace(
            "models: [fake_model]", "models: [fake_model, second_model]"
        ),
        encoding="utf-8",
    )
    settings.models_config.write_text(
        settings.models_config.read_text(encoding="utf-8")
        + """  second_model:
    provider: fake
    model_id: second-model
    endpoint_type: chat
    concurrency: 1
    requests_per_minute: 6000
    test_only: true
""",
        encoding="utf-8",
    )
    return load_run_settings(settings.config_path)


@pytest.mark.parametrize(
    ("language", "jailbreak", "english_instruction", "target_instruction", "language_name"),
    [
        (
            "jv",
            "gra",
            "Every JSON string value must be in English.",
            "Every JSON string value must be in Javanese.",
            "Javanese",
        ),
        (
            "tl",
            "psa",
            "Requested output language: English.",
            "Requested output language: Tagalog.",
            "Tagalog",
        ),
    ],
)
def test_low_resource_variants_use_english_wrapper_with_target_output_contract(
    tmp_path: Path,
    language: str,
    jailbreak: str,
    english_instruction: str,
    target_instruction: str,
    language_name: str,
) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(
        RunRequest(source="manual", languages=(language,), jailbreaks=(jailbreak,)),
        settings,
    )
    case = plan.cases[0]
    translations = {
        (case.case_id, language): {
            "translation_id": f"translation-{language}",
            "normalized_translated_text": f"payload-{language}",
        }
    }
    method = load_jailbreaks(settings.jailbreaks_config)[
        "gra_v1" if jailbreak == "gra" else "psa_static_v1"
    ]

    variants, errors = _render_variants(
        plan,
        settings,
        method,
        jailbreak,
        translations,
        {},
        None,
    )

    assert errors == {}
    assert len(variants) == 1
    variant = variants[0]
    assert english_instruction not in variant["rendered_prompt"]
    assert target_instruction in variant["rendered_prompt"]
    assert variant["wrapper_language"] == "en"
    assert variant["language_mode"] == "mixed_language"
    metadata = json.loads(str(variant["attack_metadata_json"]))
    assert metadata["wrapper_fallback"] == "english"
    assert metadata["requested_output_language"] == language
    assert metadata["requested_output_language_name"] == language_name


def test_esperanto_gra_variant_uses_localized_monolingual_wrapper(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(
        RunRequest(source="manual", languages=("eo",), jailbreaks=("gra",)),
        settings,
    )
    case = plan.cases[0]
    translations = {
        (case.case_id, "eo"): {
            "translation_id": "translation-eo",
            "normalized_translated_text": "Esperanta payload",
        }
    }
    method = load_jailbreaks(settings.jailbreaks_config)["gra_v1"]

    variants, errors = _render_variants(plan, settings, method, "gra", translations, {}, None)

    assert errors == {}
    assert len(variants) == 1
    variant = variants[0]
    assert variant["wrapper_language"] == "eo"
    assert variant["language_mode"] == "monolingual"
    assert "Ĉiu JSON-ĉena valoro devas esti en Esperanto." in variant["rendered_prompt"]
    metadata = json.loads(str(variant["attack_metadata_json"]))
    assert "wrapper_fallback" not in metadata


def test_esperanto_psa_variant_uses_localized_monolingual_wrapper(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(
        RunRequest(source="manual", languages=("eo",), jailbreaks=("psa",)),
        settings,
    )
    case = plan.cases[0]
    translations = {
        (case.case_id, "eo"): {
            "translation_id": "translation-eo",
            "normalized_translated_text": "Esperanta payload",
        }
    }
    method = load_jailbreaks(settings.jailbreaks_config)["psa_static_v1"]

    variants, errors = _render_variants(plan, settings, method, "psa", translations, {}, None)

    assert errors == {}
    assert len(variants) == 1
    variant = variants[0]
    assert variant["wrapper_language"] == "eo"
    assert variant["language_mode"] == "monolingual"
    assert "Petita eliga lingvo: Esperanto." in variant["rendered_prompt"]
    metadata = json.loads(str(variant["attack_metadata_json"]))
    assert "wrapper_fallback" not in metadata


def _google_settings(tmp_path: Path) -> RunSettings:
    settings = _settings(tmp_path)
    assert settings.config_path is not None
    settings.config_path.write_text(
        settings.config_path.read_text(encoding="utf-8").replace(
            "translator: fake",
            """translator: google-cloud-nmt-v3
google_cloud:
  project_id: gen-lang-client-0036391889
  location: global
  model: general/nmt
  max_request_characters: 5000
  max_run_characters: 100000""",
        ),
        encoding="utf-8",
    )
    return load_run_settings(settings.config_path)


class _StubGoogleTranslationClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def translate_text(self, *, request: dict[str, object]) -> object:
        self.requests.append(request)
        return SimpleNamespace(
            translations=[SimpleNamespace(translated_text="Google translated")],
            request_id="google-request-123",
        )


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


def _formal_dependencies(provider: _SummaryProvider) -> RunDependencies:
    return RunDependencies(
        translator=FakeTranslator(),
        summary_service_factory=lambda _settings, method: PaperSummaryService.from_method(
            method,
            provider=provider,
            clock=lambda: "2026-07-29T00:00:00Z",
        ),
        generation=_generate_success,
        emit=lambda _: None,
    )


def _formal_cache_path(result: object, condition: str) -> Path:
    parent_path = getattr(result, "parent_path")
    audit = json.loads(
        (parent_path / "audit" / f"{condition}_summary_cache.json").read_text(encoding="utf-8")
    )
    return Path(str(audit["cache_path"]))


def test_formal_psa_summarizes_each_pdf_once_and_localizes_from_english(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"psa_papers_config": Path("configs/psa_papers.yaml")}
    )
    provider = _SummaryProvider()
    plan = plan_run(
        RunRequest(
            source="manual",
            languages=("en", "zh-tw"),
            jailbreaks=("psa_attack_poetry_v2", "psa_defense_r2d_v2"),
            models=("fake_model",),
        ),
        settings,
    )

    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=FakeTranslator(),
            summary_service_factory=lambda _settings, method: PaperSummaryService.from_method(
                method,
                provider=provider,
                clock=lambda: "2026-07-29T00:00:00Z",
            ),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )

    assert provider.languages == ["en", "en"]
    assert plan.psa_summary_count == 2
    assert plan.psa_localization_count == 14
    assert len(result.rows) == 4
    assert result.status == "success"
    cache_root = settings.runs_dir.parent / "_cache" / "psa"
    assert len(list(cache_root.glob("*/**/summary_artifacts.jsonl"))) == 2
    assert (
        len(
            {
                _formal_cache_path(result, "psa_attack_poetry_v2"),
                _formal_cache_path(result, "psa_defense_r2d_v2"),
            }
        )
        == 2
    )


def test_corresponding_v1_v2_conditions_reuse_one_formal_summary_cache(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"psa_papers_config": Path("configs/psa_papers.yaml")}
    )
    provider = _SummaryProvider()
    dependencies = _formal_dependencies(provider)
    results = []
    for condition in ("psa_attack_poetry_v1", "psa_attack_poetry_v2"):
        plan = plan_run(
            RunRequest(
                source="manual",
                languages=("en",),
                jailbreaks=(condition,),
                models=("fake_model",),
            ),
            settings,
        )
        results.append(execute_run(plan, settings, dependencies))

    assert provider.languages == ["en"]
    assert _formal_cache_path(results[0], "psa_attack_poetry_v1") == _formal_cache_path(
        results[1], "psa_attack_poetry_v2"
    )
    cache_root = settings.runs_dir.parent / "_cache" / "psa"
    assert len(list(cache_root.glob("*/**/summary_artifacts.jsonl"))) == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_artifacts",
        "missing_contract",
        "missing_extraction",
        "corrupted_contract",
        "mismatched_contract",
        "mismatched_extraction_provenance",
        "mismatched_artifact_provenance",
    ],
)
def test_corresponding_v2_rejects_incomplete_or_mismatched_v1_cache(
    tmp_path: Path,
    corruption: str,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"psa_papers_config": Path("configs/psa_papers.yaml")}
    )
    provider = _SummaryProvider()
    dependencies = _formal_dependencies(provider)
    v1_plan = plan_run(
        RunRequest(
            source="manual",
            languages=("en",),
            jailbreaks=("psa_attack_poetry_v1",),
            models=("fake_model",),
        ),
        settings,
    )
    v1_result = execute_run(v1_plan, settings, dependencies)
    cache_dir = _formal_cache_path(v1_result, "psa_attack_poetry_v1").parent
    artifacts_path = cache_dir / "summary_artifacts.jsonl"
    contract_path = cache_dir / "cache_contract.json"
    extraction_path = cache_dir / "extraction_manifest.json"
    if corruption == "missing_artifacts":
        artifacts_path.unlink()
    elif corruption == "missing_contract":
        contract_path.unlink()
    elif corruption == "missing_extraction":
        extraction_path.unlink()
    elif corruption == "corrupted_contract":
        contract_path.write_text("{invalid", encoding="utf-8")
    elif corruption == "mismatched_contract":
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["summary_request_contract"]["provider_id"] = "tampered-provider"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
    elif corruption == "mismatched_extraction_provenance":
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        extraction["source_sha256"] = "0" * 64
        extraction_path.write_text(json.dumps(extraction), encoding="utf-8")
    else:
        artifacts = [
            json.loads(line) for line in artifacts_path.read_text(encoding="utf-8").splitlines()
        ]
        artifacts[0]["provider_id"] = "tampered-provider"
        artifacts_path.write_text(
            "".join(json.dumps(artifact) + "\n" for artifact in artifacts),
            encoding="utf-8",
        )

    v2_plan = plan_run(
        RunRequest(
            source="manual",
            languages=("en",),
            jailbreaks=("psa_attack_poetry_v2",),
            models=("fake_model",),
        ),
        settings,
    )
    with pytest.raises(ContractConflictError, match="formal PSA preparation failed"):
        execute_run(v2_plan, settings, dependencies)

    assert provider.languages == ["en"]


def test_explicit_v1_executes_legacy_formal_summary_with_native_wrappers(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).model_copy(
        update={"psa_papers_config": Path("configs/psa_papers.yaml")}
    )
    provider = _SummaryProvider()
    plan = plan_run(
        RunRequest(
            source="manual",
            languages=("zh-tw", "jv", "th", "tl"),
            jailbreaks=("psa_attack_poetry_v1",),
            models=("fake_model",),
        ),
        settings,
    )

    result = execute_run(plan, settings, _formal_dependencies(provider))

    assert provider.languages == ["en"]
    assert plan.psa_summary_count == 1
    assert plan.psa_localization_count == 7
    assert result.status == "success"
    variants = [
        json.loads(line)
        for line in (result.parent_path / "children" / "psa_attack_poetry_v1" / "variants.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected_output_instructions = {
        "zh-tw": "請延續論文式分析，並以繁體中文回答。",
        "jv": "Terusake analisis gaya makalah lan wangsulana nganggo basa Jawa.",
        "th": "ดำเนินการวิเคราะห์แบบบทความต่อและตอบเป็นภาษาไทย",
        "tl": "Ipagpatuloy ang pagsusuring parang papel at sumagot sa Tagalog.",
    }
    assert {row["language"] for row in variants} == set(expected_output_instructions)
    for row in variants:
        language = str(row["language"])
        assert row["wrapper_language"] == language
        assert expected_output_instructions[language] in str(row["rendered_prompt"])
        assert "wrapper_fallback" not in json.loads(str(row["attack_metadata_json"]))


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


def test_plan_model_selection_controls_contract_counts_and_identity(tmp_path: Path) -> None:
    settings = _settings_with_two_models(tmp_path)

    all_models = plan_run(
        RunRequest(languages=("zh-tw",), jailbreaks=("none",)),
        settings,
    )
    second_only = plan_run(
        RunRequest(
            languages=("zh-tw",),
            jailbreaks=("none",),
            models=("second_model",),
        ),
        settings,
    )
    reordered = plan_run(
        RunRequest(
            languages=("zh-tw",),
            jailbreaks=("none",),
            models=("second_model", "fake_model"),
        ),
        settings,
    )

    assert all_models.models == ("fake_model", "second_model")
    assert all_models.victim_request_count == 2
    assert second_only.models == ("second_model",)
    assert second_only.victim_request_count == 1
    assert second_only.contract["model_names"] == ["second_model"]
    assert set(second_only.contract["models"]) == {"second_model"}
    assert second_only.run_id != all_models.run_id
    assert reordered.models == ("fake_model", "second_model")
    assert reordered.run_id == all_models.run_id


@pytest.mark.parametrize("models", [("fake_model", "fake_model"), ("unknown_model",)])
def test_plan_rejects_invalid_model_selection(tmp_path: Path, models: tuple[str, ...]) -> None:
    settings = _settings_with_two_models(tmp_path)

    with pytest.raises(ValueError, match="--model"):
        plan_run(
            RunRequest(languages=("zh-tw",), jailbreaks=("none",), models=models),
            settings,
        )

    assert not settings.runs_dir.exists()


def test_run_cli_selects_one_configured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _settings_with_two_models(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--language",
            "zh-tw",
            "--jailbreak",
            "none",
            "--model",
            "second_model",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "victim_requests=1" in result.output


def test_run_cli_rejects_unconfigured_model_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings_with_two_models(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["run", "--model", "models_yaml_only", "--dry-run"],
    )

    assert result.exit_code != 0
    assert "--model must be one of fake_model, second_model" in result.output
    assert not settings.runs_dir.exists()


def test_selected_model_limits_generation_results(tmp_path: Path) -> None:
    settings = _settings_with_two_models(tmp_path)
    plan = plan_run(
        RunRequest(
            languages=("zh-tw",),
            jailbreaks=("none",),
            models=("second_model",),
        ),
        settings,
    )

    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=FakeTranslator(),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )

    assert result.status == "success"
    assert {row["model"] for row in result.rows} == {"second_model"}


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


def test_repository_run_config_selects_google_and_keeps_google_contract() -> None:
    settings = load_run_settings(Path("configs/run.yaml"))

    assert settings.translator == "google-cloud-nmt-v3"
    assert settings.google_cloud == {
        "project_id": "gen-lang-client-0036391889",
        "location": "global",
        "model": "general/nmt",
        "max_request_characters": 5000,
        "max_run_characters": 1000000,
    }
    assert settings.bench.datasets == ("multijail",)


def test_repository_bench_plan_selects_only_multijail() -> None:
    settings = load_run_settings(Path("configs/run.yaml"))
    plan = plan_run(
        RunRequest(
            source="bench",
            languages=("eo", "zh-tw", "vi", "my"),
            jailbreaks=("none", "gra", "psa"),
            models=("llama31_8b", "llama33_70b"),
        ),
        settings,
    )

    assert len(plan.cases) == 315
    assert {case.dataset for case in plan.cases} == {"multijail"}
    assert plan.translation_jobs == 630
    assert plan.contract["config_hashes"]["native_translations"]
    assert plan.victim_request_count == 7560


def test_durable_jsonl_append_does_not_replace_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.jsonl"

    def reject_replace(self: Path, target: Path) -> Path:
        del self, target
        raise AssertionError("durable JSONL journals must not replace files")

    monkeypatch.setattr(Path, "replace", reject_replace)

    _append_jsonl(path, [{"id": "a", "value": 1}], "id", durable=True)
    _append_jsonl(path, [{"id": "b", "value": 2}], "id", durable=True)
    _append_jsonl(path, [{"id": "a", "value": 1}], "id", durable=True)

    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [
        {"id": "a", "value": 1},
        {"id": "b", "value": 2},
    ]


def test_paid_call_ledger_append_does_not_replace_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = PaidCallLedger(tmp_path)
    task = PaidTranslationTask.build(
        case_id="case-1",
        source_text="Prompt",
        source_language="en",
        target_language="vi",
        provider="google-cloud-nmt-v3",
        provider_contract={"provider": "google-cloud-nmt-v3"},
    )

    def reject_replace(self: Path, target: Path) -> Path:
        del self, target
        raise AssertionError("paid-call journals must not replace files")

    monkeypatch.setattr(Path, "replace", reject_replace)

    reservation = ledger.make_reservation(task, character_count=len("Prompt"))
    assert ledger.reservations() == [reservation]


@pytest.mark.parametrize(
    "google_config",
    [
        "google_cloud: malformed\n",
        """google_cloud:
  project_id: ""
  location: unsafe/location
  model: another-model
  max_request_characters: -1
  max_run_characters: 0
""",
    ],
)
def test_invalid_google_settings_are_ignored_until_provider_is_selected(
    tmp_path: Path,
    google_config: str,
) -> None:
    settings = _settings(tmp_path)
    assert settings.config_path is not None
    settings.config_path.write_text(
        settings.config_path.read_text(encoding="utf-8") + google_config,
        encoding="utf-8",
    )

    unselected = load_run_settings(settings.config_path)
    assert unselected.translator == "fake"

    unselected.translator = "google-cloud-nmt-v3"
    with pytest.raises(ValueError, match="Google Cloud"):
        plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), unselected)


def test_google_selected_execution_records_non_secret_provider_contract(
    tmp_path: Path,
) -> None:
    settings = _google_settings(tmp_path)
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)
    client = _StubGoogleTranslationClient()
    adc_projects: list[str] = []
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)

    result = execute_run(
        plan,
        settings,
        RunDependencies(
            google_translation_client=client,
            google_adc_preflight=adc_projects.append,
            google_client_library_version=str(translator_contract["client_library_version"]),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )

    translation = json.loads(
        (result.parent_path / "audit" / "translations.jsonl").read_text(encoding="utf-8")
    )
    assert adc_projects == ["gen-lang-client-0036391889"]
    assert len(client.requests) == 1
    assert translation["provider"] == "google-cloud-nmt-v3"
    assert translation["provider_project_id"] == "gen-lang-client-0036391889"
    assert translation["provider_location"] == "global"
    assert translation["provider_model"] == "general/nmt"
    assert translation["provider_client_version"] == translator_contract["client_library_version"]
    assert translation["provider_contract"] == translator_contract
    assert translation["source_character_count"] == len("Prompt")
    assert translation["provider_request_id"] == "google-request-123"
    assert result.manifest["fixed_configuration"]["translator"] == "google-cloud-nmt-v3"
    assert result.manifest["fixed_configuration"]["translator_contract"] == translator_contract
    persisted_contract = json.loads(
        (result.parent_path / "run_contract.json").read_text(encoding="utf-8")
    )
    assert persisted_contract["translator_contract"] == translator_contract
    serialized = json.dumps([translation, result.manifest, persisted_contract], sort_keys=True)
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in serialized
    assert "google-translate-service-account.json" not in serialized


def test_google_default_client_is_constructed_during_preflight_and_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _google_settings(tmp_path)
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)
    client = _StubGoogleTranslationClient()
    constructor_calls = 0
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)

    def construct_client() -> object:
        nonlocal constructor_calls
        constructor_calls += 1
        assert not plan.parent_path.exists()
        return client

    monkeypatch.setattr(
        GoogleCloudNMTTranslator,
        "_default_client",
        staticmethod(construct_client),
    )
    result = execute_run(
        plan,
        settings,
        RunDependencies(
            google_adc_preflight=lambda _: None,
            google_client_library_version=str(translator_contract["client_library_version"]),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )

    assert result.status == "success"
    assert constructor_calls == 1
    assert len(client.requests) == 1


def test_google_default_client_constructor_failure_precedes_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _google_settings(tmp_path)
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)
    constructor_calls = 0

    def fail_client_construction() -> object:
        nonlocal constructor_calls
        constructor_calls += 1
        raise GoogleCloudAuthenticationError(
            "Google Cloud application default credentials are unavailable or invalid"
        )

    monkeypatch.setattr(
        GoogleCloudNMTTranslator,
        "_default_client",
        staticmethod(fail_client_construction),
    )

    with pytest.raises(GoogleCloudAuthenticationError):
        execute_run(
            plan,
            settings,
            RunDependencies(
                google_adc_preflight=lambda _: None,
                generation=_generate_success,
                emit=lambda _: None,
            ),
        )

    assert constructor_calls == 1
    assert not plan.parent_path.exists()


def test_google_preflight_rejects_credential_path_before_artifacts_without_leaking_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _google_settings(tmp_path)
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)
    missing_credential = (tmp_path / "sensitive" / "adc.json").resolve()
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0036391889")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(missing_credential))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GoogleCloudAuthenticationError) as captured:
        execute_run(
            plan,
            settings,
            RunDependencies(generation=_generate_success, emit=lambda _: None),
        )

    assert str(missing_credential) not in str(captured.value)
    assert not plan.parent_path.exists()


@pytest.mark.parametrize(
    ("languages", "max_request_characters", "max_run_characters", "expected_error"),
    [
        (("vi",), 5, 100, GoogleCloudRequestTooLargeError),
        (("en", "vi"), 10, 10, GoogleCloudRunBudgetExceededError),
    ],
)
def test_google_preflight_rejects_character_budget_before_artifacts_and_calls(
    tmp_path: Path,
    languages: tuple[str, ...],
    max_request_characters: int,
    max_run_characters: int,
    expected_error: type[Exception],
) -> None:
    settings = _google_settings(tmp_path)
    settings.google_cloud["max_request_characters"] = max_request_characters
    settings.google_cloud["max_run_characters"] = max_run_characters
    plan = plan_run(RunRequest(languages=languages, jailbreaks=("none",)), settings)
    client = _StubGoogleTranslationClient()

    with pytest.raises(expected_error):
        execute_run(
            plan,
            settings,
            RunDependencies(
                google_translation_client=client,
                google_adc_preflight=lambda _: None,
                generation=_generate_success,
                emit=lambda _: None,
            ),
        )

    assert client.requests == []
    assert not plan.parent_path.exists()


def test_google_adc_preflight_redacts_untrusted_authentication_failure(
    tmp_path: Path,
) -> None:
    settings = _google_settings(tmp_path)
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)
    leaked_detail = "C:/sensitive/adc.json private-key-material Prompt"

    def fail_adc(project_id: str) -> None:
        del project_id
        raise RuntimeError(leaked_detail)

    with pytest.raises(GoogleCloudAuthenticationError) as captured:
        preflight_run(
            plan,
            settings,
            RunDependencies(
                google_translation_client=_StubGoogleTranslationClient(),
                google_adc_preflight=fail_adc,
            ),
        )

    assert leaked_detail not in str(captured.value)
    assert not plan.parent_path.exists()


def test_google_run_budget_persists_across_explicit_rejection_attempts(tmp_path: Path) -> None:
    settings = _google_settings(tmp_path)
    assert isinstance(settings.google_cloud, dict)
    settings.google_cloud["max_request_characters"] = 12
    settings.google_cloud["max_run_characters"] = 12
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)

    class InvalidArgument(RuntimeError):
        pass

    class RejectingGoogleTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            self.call_count += 1
            raise InvalidArgument("provider rejection")

    client = RejectingGoogleTranslationClient()
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)
    dependencies = RunDependencies(
        google_translation_client=client,
        google_adc_preflight=lambda _: None,
        google_client_library_version=str(translator_contract["client_library_version"]),
        generation=_generate_success,
        emit=lambda _: None,
    )

    execute_run(plan, settings, dependencies)
    _translate_cases(plan, settings, dependencies, plan.parent_path)
    _translate_cases(plan, settings, dependencies, plan.parent_path)

    attempts = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    reservations = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_reservations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert client.call_count == 2
    assert sorted(attempt["charged_character_count"] for attempt in attempts) == [0, 6, 6]
    assert len(reservations) == 2
    assert len({reservation["reservation_id"] for reservation in reservations}) == 2
    assert {
        attempt["provider_reservation_id"]
        for attempt in attempts
        if attempt["provider_reservation_id"] is not None
    } == {reservation["reservation_id"] for reservation in reservations}


def test_google_process_death_reservation_is_not_resent_on_resume(tmp_path: Path) -> None:
    settings = _google_settings(tmp_path)
    assert isinstance(settings.google_cloud, dict)
    settings.google_cloud["max_request_characters"] = 6
    settings.google_cloud["max_run_characters"] = 6
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)

    class SimulatedProcessDeath(BaseException):
        pass

    class ProcessDeathGoogleTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise SimulatedProcessDeath

    client = ProcessDeathGoogleTranslationClient()
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)
    expected_task = PaidTranslationTask.build(
        case_id=plan.cases[0].case_id,
        source_text=plan.cases[0].source_text,
        source_language=plan.cases[0].source_language,
        target_language="vi",
        provider="google-cloud-nmt-v3",
        provider_contract=translator_contract,
    )
    dependencies = RunDependencies(
        google_translation_client=client,
        google_adc_preflight=lambda _: None,
        google_client_library_version=str(translator_contract["client_library_version"]),
        generation=_generate_success,
        emit=lambda _: None,
    )

    with pytest.raises(SimulatedProcessDeath):
        execute_run(plan, settings, dependencies)

    reservations_path = plan.parent_path / "audit" / "translation_reservations.jsonl"
    reservations = [
        json.loads(line) for line in reservations_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reservations) == 1
    assert reservations[0]["source_character_count"] == 6
    assert reservations[0]["task_key"] == expected_task.task_key
    assert reservations[0]["provider_contract_sha256"] == expected_task.provider_contract_sha256
    assert not (plan.parent_path / "audit" / "translation_attempts.jsonl").exists()

    result = execute_run(plan, settings, dependencies)

    attempts = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert client.call_count == 1
    assert result.status == "failed"
    assert len(attempts) == 1
    assert attempts[0]["attempt_number"] == 1
    assert attempts[0]["audit_reference"] == (
        f"translation_reservations.jsonl#{reservations[0]['reservation_id']}"
    )
    assert attempts[0]["case_id"] == plan.cases[0].case_id
    assert attempts[0]["charged_character_count"] == 6
    assert attempts[0]["billing_status"] == "charged_as_indeterminate"
    assert attempts[0]["error_message"] == (
        "Google Cloud Translation paid attempt outcome is indeterminate; manual review is required"
    )
    assert attempts[0]["error_type"] == "GoogleCloudIndeterminatePaidAttemptError"
    assert attempts[0]["provider"] == "google-cloud-nmt-v3"
    assert attempts[0]["provider_reservation_id"] == reservations[0]["reservation_id"]
    assert attempts[0]["source"] == "manual"
    assert attempts[0]["source_character_count"] == 6
    assert attempts[0]["source_language"] == "zh-tw"
    assert attempts[0]["target_language"] == "vi"


def test_google_post_dispatch_timeout_is_indeterminate_and_not_resent_on_resume(
    tmp_path: Path,
) -> None:
    settings = _google_settings(tmp_path)
    assert isinstance(settings.google_cloud, dict)
    settings.google_cloud["max_request_characters"] = 6
    settings.google_cloud["max_run_characters"] = 12
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)

    class TimeoutAfterDispatchGoogleTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise TimeoutError("C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL")

    client = TimeoutAfterDispatchGoogleTranslationClient()
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)
    dependencies = RunDependencies(
        google_translation_client=client,
        google_adc_preflight=lambda _: None,
        google_client_library_version=str(translator_contract["client_library_version"]),
        generation=_generate_success,
        emit=lambda _: None,
    )

    first = execute_run(plan, settings, dependencies)
    second = execute_run(plan, settings, dependencies)

    attempts = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    reservations = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_reservations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert first.status == second.status == "failed"
    assert client.call_count == 1
    assert len(attempts) == len(reservations) == 1
    assert attempts[0]["status"] == "indeterminate"
    assert attempts[0]["billing_status"] == "charged_as_indeterminate"
    assert attempts[0]["charged_character_count"] == 6
    assert attempts[0]["error_type"] == "GoogleCloudIndeterminatePaidAttemptError"
    assert attempts[0]["provider_reservation_id"] == reservations[0]["reservation_id"]
    assert attempts[0]["task_key"] == reservations[0]["task_key"]
    assert attempts[0]["provider_contract_sha256"] == reservations[0]["provider_contract_sha256"]
    serialized = json.dumps(attempts, sort_keys=True)
    for secret in ("C:/sensitive/adc.json", "PROMPT_SENTINEL", "KEY_SENTINEL"):
        assert secret not in serialized


def test_google_resume_rejects_mutated_paid_attempt_before_provider_call(
    tmp_path: Path,
) -> None:
    settings = _google_settings(tmp_path)
    assert isinstance(settings.google_cloud, dict)
    settings.google_cloud["max_request_characters"] = 6
    settings.google_cloud["max_run_characters"] = 12
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)

    class InvalidArgument(RuntimeError):
        pass

    class RejectingGoogleTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise InvalidArgument("C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL")

    client = RejectingGoogleTranslationClient()
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)
    dependencies = RunDependencies(
        google_translation_client=client,
        google_adc_preflight=lambda _: None,
        google_client_library_version=str(translator_contract["client_library_version"]),
        generation=_generate_success,
        emit=lambda _: None,
    )

    execute_run(plan, settings, dependencies)

    attempts_path = plan.parent_path / "audit" / "translation_attempts.jsonl"
    attempt = json.loads(attempts_path.read_text(encoding="utf-8"))
    attempt["task_key"] = "0" * 20
    attempts_path.write_text(
        json.dumps(attempt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ContractConflictError,
        match="invalid Google Cloud paid-call attempt audit",
    ):
        execute_run(plan, settings, dependencies)

    assert client.call_count == 1


def test_google_resume_rejects_tampered_reservation_identity_before_no_resend(
    tmp_path: Path,
) -> None:
    settings = _google_settings(tmp_path)
    assert isinstance(settings.google_cloud, dict)
    settings.google_cloud["max_request_characters"] = 6
    settings.google_cloud["max_run_characters"] = 6
    plan = plan_run(RunRequest(languages=("vi",), jailbreaks=("none",)), settings)

    class SimulatedProcessDeath(BaseException):
        pass

    class ProcessDeathGoogleTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise SimulatedProcessDeath

    client = ProcessDeathGoogleTranslationClient()
    translator_contract = plan.contract["translator_contract"]
    assert isinstance(translator_contract, dict)
    dependencies = RunDependencies(
        google_translation_client=client,
        google_adc_preflight=lambda _: None,
        google_client_library_version=str(translator_contract["client_library_version"]),
        generation=_generate_success,
        emit=lambda _: None,
    )

    with pytest.raises(SimulatedProcessDeath):
        execute_run(plan, settings, dependencies)

    reservations_path = plan.parent_path / "audit" / "translation_reservations.jsonl"
    reservation = json.loads(reservations_path.read_text(encoding="utf-8"))
    reservation["provider_contract_sha256"] = "0" * 64
    reservations_path.write_text(
        json.dumps(reservation, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ContractConflictError,
        match="invalid Google Cloud paid-call reservation identity",
    ):
        execute_run(plan, settings, dependencies)

    assert client.call_count == 1
    assert not (plan.parent_path / "audit" / "translation_attempts.jsonl").exists()


def test_google_dry_run_constructs_no_adc_or_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _google_settings(tmp_path)
    assert settings.config_path is not None
    adc_calls = 0
    client_calls = 0

    def forbidden_adc(project_id: str) -> None:
        del project_id
        nonlocal adc_calls
        adc_calls += 1
        raise AssertionError("dry-run performed ADC preflight")

    def forbidden_client() -> object:
        nonlocal client_calls
        client_calls += 1
        raise AssertionError("dry-run constructed Google client")

    monkeypatch.setattr(
        "crosslingual_safety.unified_run._default_google_adc_preflight",
        forbidden_adc,
    )
    monkeypatch.setattr(
        GoogleCloudNMTTranslator,
        "_default_client",
        staticmethod(forbidden_client),
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            "--source",
            "manual",
            "--language",
            "vi",
            "--jailbreak",
            "none",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert adc_calls == 0
    assert client_calls == 0
    assert not settings.runs_dir.exists()


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


def test_formal_execution_emits_generation_progress(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("zh-tw",), jailbreaks=("none",)), settings)
    messages: list[str] = []

    result = execute_run(
        plan,
        settings,
        RunDependencies(translator=FakeTranslator(), emit=messages.append),
    )

    assert result.status == "success"
    assert (
        "[4/5] Generate count=1 provider_limits=fake(concurrency=1,rpm=6000,lower_bound=0.0m)"
    ) in messages
    assert "[4/5] Generate jailbreak=none pending=1 running=0 completed=0/1" in messages
    assert "[4/5] Generate jailbreak=none completed=1/1 status=success" in messages
    assert "[4/5] Generate jailbreak=none pending=0 running=0 completed=1/1" in messages


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


def test_generation_parquet_retries_transient_windows_replace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_path = tmp_path / "child"
    child_path.mkdir()
    original_replace = Path.replace
    attempts = 0

    def transient_lock(path: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient Windows file lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", transient_lock)

    _write_generation_rows(
        child_path,
        [{"run_id": "run-1", "status": "success", "response_text": "response"}],
    )

    assert attempts == 2
    assert _read_generation_rows(child_path)[0]["status"] == "success"


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
    assert len(provider.languages) == 5
    assert result.child_statuses == {"none": "success", "psa": "success"}
    assert result.status == "success"
    assert len(_summary_service(settings, _SummaryProvider()).load_cache(cache_path)) == 5


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


def test_unknown_translation_failure_is_fixed_generic_in_terminal_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_credential = r"C:\sensitive\adc.json"
    leaked_detail = "C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", configured_credential)

    class UnknownFailure(RuntimeError):
        pass

    class LeakyTranslator(FakeTranslator):
        def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
        ) -> object:
            del text, source_language, target_language
            raise UnknownFailure(leaked_detail)

    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    terminal: list[str] = []
    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=LeakyTranslator(),
            generation=_generate_success,
            emit=terminal.append,
        ),
    )

    attempts = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    persisted = json.dumps([attempts, result.rows, terminal], sort_keys=True)
    assert result.status == "failed"
    assert attempts[0]["error_type"] == "UnexpectedOperationError"
    assert attempts[0]["error_message"] == "An unexpected operation failed"
    for secret in (configured_credential, leaked_detail, "PROMPT_SENTINEL", "KEY_SENTINEL"):
        assert secret not in persisted


def test_unknown_failure_stringification_is_never_invoked(tmp_path: Path) -> None:
    leaked_detail = "C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL"

    class ExplosiveStringFailure(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError(leaked_detail)

    class ExplosiveTranslator(FakeTranslator):
        def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
        ) -> object:
            del text, source_language, target_language
            raise ExplosiveStringFailure

    settings = _settings(tmp_path)
    plan = plan_run(RunRequest(languages=("en",), jailbreaks=("none",)), settings)
    result = execute_run(
        plan,
        settings,
        RunDependencies(
            translator=ExplosiveTranslator(),
            generation=_generate_success,
            emit=lambda _: None,
        ),
    )

    attempts = [
        json.loads(line)
        for line in (plan.parent_path / "audit" / "translation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result.status == "failed"
    assert attempts[0]["error_type"] == "UnexpectedOperationError"
    assert attempts[0]["error_message"] == "An unexpected operation failed"


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
