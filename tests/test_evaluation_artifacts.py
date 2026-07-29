import json
from pathlib import Path

import pytest

from crosslingual_safety.evaluation.artifacts import (
    ArtifactConflictError,
    JsonlJournal,
    load_evaluation_cases,
)
from crosslingual_safety.evaluation.models import StrongRejectJudgment


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_run(
    tmp_path: Path,
    *,
    result: dict[str, object],
    variant: dict[str, object] | None,
) -> Path:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": "run-1"}),
        encoding="utf-8",
    )
    _write_jsonl(run_dir / "results.jsonl", [result])
    _write_jsonl(
        run_dir / "audit" / "input_snapshot.jsonl",
        [
            {
                "case_id": result["case_id"],
                "source": result["source"],
                "source_language": "en",
                "source_text": "source goal",
                "intent": "harmful",
            }
        ],
    )
    if variant is not None:
        _write_jsonl(
            run_dir / "children" / str(result["jailbreak"]) / "variants.jsonl",
            [variant],
        )
    return run_dir


def _judgment(evaluation_id: str, score: float) -> StrongRejectJudgment:
    return StrongRejectJudgment(
        evaluation_id=evaluation_id,
        status="success",
        score=score,
        adapter_id="qylu4156/strongreject-15k-v1",
        adapter_revision="revision",
        base_model_id="google/gemma-2b",
    )


def test_load_evaluation_cases_joins_payload_without_rendered_wrapper(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        result={
            "case_id": "c1",
            "source": "bench",
            "language": "vi",
            "jailbreak": "gra",
            "model": "llama31_8b",
            "status": "success",
            "response": "response",
        },
        variant={
            "case_id": "c1",
            "language": "vi",
            "payload": "translated goal",
            "rendered_prompt": "GRA wrapper",
        },
    )

    [case] = load_evaluation_cases(run_dir)

    assert case.forbidden_prompt == "translated goal"
    assert "GRA wrapper" not in case.forbidden_prompt
    assert case.response == "response"


def test_failed_generation_uses_source_fallback_without_claiming_response(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        result={
            "case_id": "c1",
            "source": "bench",
            "language": "my",
            "jailbreak": "none",
            "model": "llama31_8b",
            "status": "timeout",
            "error_type": "timeout",
        },
        variant=None,
    )

    [case] = load_evaluation_cases(run_dir)

    assert case.forbidden_prompt == "source goal"
    assert case.response is None
    assert case.generation_status == "timeout"


def test_journal_is_idempotent_and_rejects_conflicting_content(tmp_path: Path) -> None:
    journal = JsonlJournal(
        tmp_path / "records.jsonl",
        StrongRejectJudgment,
    )

    assert journal.append(_judgment("e1", 0.2))
    assert not journal.append(_judgment("e1", 0.2))
    with pytest.raises(ArtifactConflictError, match="immutable JSONL row conflict"):
        journal.append(_judgment("e1", 0.8))

    assert list(journal.load()) == ["e1"]
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 1
