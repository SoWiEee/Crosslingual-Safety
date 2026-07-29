import json
from pathlib import Path

from typer.testing import CliRunner

from crosslingual_safety.cli import app

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
