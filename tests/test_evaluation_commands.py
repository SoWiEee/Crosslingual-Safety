import json
from pathlib import Path

from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.evaluation.commands import _multilingual_judge, _response_translator
from crosslingual_safety.evaluation.models import EvaluationConfig, StrongRejectConfig
from crosslingual_safety.generation.config import ModelConfig
from crosslingual_safety.translation.providers import GoogleCloudNMTTranslator
from crosslingual_safety.unified_run import RunSettings

runner = CliRunner()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _workspace(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "run.yaml").write_text(
        "version: 1\nruns_dir: runs/experiments\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "runs/experiments/run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    _write_jsonl(
        run_dir / "audit/input_snapshot.jsonl",
        [
            {
                "case_id": "case-1",
                "source": "bench",
                "source_language": "en",
                "source_text": "goal",
                "intent": "harmful",
            }
        ],
    )
    _write_jsonl(
        run_dir / "results.jsonl",
        [
            {
                "case_id": "case-1",
                "source": "bench",
                "language": "en",
                "jailbreak": "none",
                "model": "llama31_8b",
                "status": "success",
                "response": "I cannot help.",
            }
        ],
    )
    _write_jsonl(
        run_dir / "children/none/variants.jsonl",
        [{"case_id": "case-1", "language": "en", "payload": "goal"}],
    )
    return run_dir


def test_report_command_rebuilds_hierarchy(tmp_path: Path, monkeypatch: object) -> None:
    run_dir = _workspace(tmp_path)
    getattr(monkeypatch, "chdir")(tmp_path)

    result = runner.invoke(app, ["report", "--run-id", run_dir.name])

    assert result.exit_code == 0, result.output
    assert f"report={run_dir / 'report.md'}" in result.output
    assert (run_dir / "children/none/report.md").is_file()


def test_evaluate_rejects_unknown_run_without_credentials(
    tmp_path: Path, monkeypatch: object
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/run.yaml").write_text(
        "version: 1\nruns_dir: runs/experiments\n",
        encoding="utf-8",
    )
    getattr(monkeypatch, "chdir")(tmp_path)

    result = runner.invoke(app, ["evaluate", "--run-id", "missing"])

    assert result.exit_code != 0
    assert "run does not exist" in result.output
    assert "ZOOLAB_API_KEY" not in result.output


def test_response_translator_uses_evaluation_request_limit() -> None:
    config = EvaluationConfig(
        version=1,
        multilingual_judge_model="gemma_4_12b",
        multilingual_confidence_threshold=0.7,
        strongreject_threshold=0.5,
        response_translator="google-cloud-nmt-v3",
        response_translation_max_request_characters=30_000,
        response_translation_max_run_characters=20_000_000,
        strongreject=StrongRejectConfig(
            adapter_id="adapter",
            adapter_revision="revision",
            base_model_id="base",
            max_response_tokens=512,
            batch_size=8,
        ),
        report_refresh_records=100,
    )
    settings = RunSettings(
        translator="google-cloud-nmt-v3",
        google_cloud={
            "project_id": "valid-project-123",
            "location": "global",
            "model": "general/nmt",
            "max_request_characters": 5000,
            "max_run_characters": 1_000_000,
        },
    )

    translator = _response_translator(config, settings)

    assert isinstance(translator, GoogleCloudNMTTranslator)
    assert translator.max_request_characters == 30_000


def test_multilingual_judge_uses_configured_completion_budget() -> None:
    model = ModelConfig(
        provider="fake",
        model_id="fake-judge",
        context_size=8192,
        endpoint_type="chat",
        concurrency=1,
        requests_per_minute=6000,
        test_only=True,
        fake_response="{}",
    )

    judge = _multilingual_judge(model, max_tokens=4096)

    assert judge.max_tokens == 4096
