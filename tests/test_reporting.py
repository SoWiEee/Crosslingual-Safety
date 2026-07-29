import json
from pathlib import Path

from crosslingual_safety.evaluation.artifacts import load_evaluation_cases
from crosslingual_safety.evaluation.models import ConsensusEvaluation
from crosslingual_safety.reporting import write_hierarchical_reports


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_report_run(tmp_path: Path, *, evaluated: bool = False) -> Path:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
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
    rows = []
    for jailbreak in ("none", "gra"):
        rows.append(
            {
                "case_id": "case-1",
                "source": "bench",
                "language": "vi",
                "jailbreak": jailbreak,
                "model": "llama31_8b",
                "status": "success",
                "response": "model response with ``` inside",
            }
        )
        _write_jsonl(
            run_dir / f"children/{jailbreak}/variants.jsonl",
            [{"case_id": "case-1", "language": "vi", "payload": "translated goal"}],
        )
    _write_jsonl(run_dir / "results.jsonl", rows)
    if evaluated:
        cases = load_evaluation_cases(run_dir)
        _write_jsonl(
            run_dir / "evaluation/evaluations.jsonl",
            [
                ConsensusEvaluation(
                    consensus_id=f"consensus-{case.evaluation_id}",
                    evaluation_id=case.evaluation_id,
                    config_sha256="config-1",
                    verdict="bypass" if case.jailbreak == "gra" else "not_bypass",
                    review_reason=None,
                    multilingual_strict_bypass=case.jailbreak == "gra",
                    multilingual_confidence=0.9,
                    strongreject_score=0.8 if case.jailbreak == "gra" else 0.2,
                    strongreject_threshold=0.5,
                    multilingual_confidence_threshold=0.7,
                ).model_dump(mode="json")
                for case in cases
            ],
        )
    return run_dir


def test_parent_is_index_and_children_hold_responses(tmp_path: Path) -> None:
    run_dir = _write_report_run(tmp_path, evaluated=True)

    summary = write_hierarchical_reports(run_dir)

    parent = (run_dir / "report.md").read_text(encoding="utf-8")
    child = (run_dir / "children/gra/report.md").read_text(encoding="utf-8")
    assert "[gra](children/gra/report.md)" in parent
    assert "model response" not in parent
    assert "model response" in child
    assert "Strict ASR" in parent
    assert "| gra | vi | llama31_8b | 1 | 1 |" in parent
    assert summary.evaluated == 2


def test_request_success_is_not_rendered_as_bypass_before_evaluation(tmp_path: Path) -> None:
    run_dir = _write_report_run(tmp_path)

    write_hierarchical_reports(run_dir)

    child = (run_dir / "children/none/report.md").read_text(encoding="utf-8")
    parent = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Generation: `success`" in child
    assert "Verdict: `pending`" in child
    assert "Pending: **1**" in child
    assert "n/a (0/0)" in parent
    assert "````text\nmodel response with ``` inside\n````" in child
