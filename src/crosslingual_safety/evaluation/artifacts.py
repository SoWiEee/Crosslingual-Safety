"""Run artifact joins and durable append-only evaluation journals."""

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from crosslingual_safety.evaluation.models import EvaluationCase
from crosslingual_safety.ids import stable_id

T = TypeVar("T", bound=BaseModel)


class EvaluationArtifactError(ValueError):
    """A required run artifact is missing or malformed."""


class ArtifactConflictError(EvaluationArtifactError):
    """A stable artifact identity was reused with different content."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise EvaluationArtifactError(f"required run artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvaluationArtifactError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, object]]:
    if not path.is_file():
        if required:
            raise EvaluationArtifactError(f"required run artifact is missing: {path}")
        return []
    rows: list[dict[str, object]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (OSError, ValueError) as error:
        raise EvaluationArtifactError(f"invalid JSONL artifact: {path}:{line_number}") from error
    return rows


def _run_id(run_dir: Path) -> str:
    manifest = _read_json(run_dir / "run_manifest.json")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise EvaluationArtifactError("run manifest does not contain a valid run_id")
    if run_dir.name != run_id:
        raise EvaluationArtifactError("run directory and manifest run_id do not match")
    return run_id


def load_evaluation_cases(run_dir: Path) -> list[EvaluationCase]:
    """Join compact results to immutable case and variant snapshots."""

    run_id = _run_id(run_dir)
    results = read_jsonl(run_dir / "results.jsonl")
    input_rows = read_jsonl(run_dir / "audit" / "input_snapshot.jsonl")
    input_by_case = {str(row.get("case_id")): row for row in input_rows}
    jailbreaks = sorted({str(row.get("jailbreak")) for row in results})
    variants: dict[tuple[str, str, str], dict[str, object]] = {}
    for jailbreak in jailbreaks:
        for row in read_jsonl(run_dir / "children" / jailbreak / "variants.jsonl", required=False):
            key = (jailbreak, str(row.get("case_id")), str(row.get("language")))
            prior = variants.get(key)
            if prior is not None and canonical_json(prior) != canonical_json(row):
                raise ArtifactConflictError(f"duplicate variant conflict: {key}")
            variants[key] = row

    cases: list[EvaluationCase] = []
    seen: dict[str, EvaluationCase] = {}
    for result in results:
        case_id = str(result.get("case_id", ""))
        language = str(result.get("language", ""))
        jailbreak = str(result.get("jailbreak", ""))
        model = str(result.get("model", ""))
        source_row = input_by_case.get(case_id)
        if source_row is None:
            raise EvaluationArtifactError(f"input snapshot is missing case: {case_id}")
        variant = variants.get((jailbreak, case_id, language))
        forbidden_prompt = (
            str(variant.get("payload"))
            if variant is not None and isinstance(variant.get("payload"), str)
            else str(source_row.get("source_text", ""))
        )
        status = str(result.get("status", ""))
        raw_response = result.get("response")
        if status == "success" and not isinstance(raw_response, str):
            result_key = f"{case_id}/{language}/{jailbreak}/{model}"
            raise EvaluationArtifactError(
                f"successful result is missing response text: {result_key}"
            )
        response = raw_response if isinstance(raw_response, str) else None
        response_sha256 = sha256_text(response) if response is not None else None
        evaluation_id = stable_id(
            "dual-judge-evaluation",
            run_id,
            case_id,
            language,
            jailbreak,
            model,
            response_sha256 or status,
        )
        intent = str(source_row.get("intent", "harmful"))
        if intent not in {"harmful", "benign"}:
            raise EvaluationArtifactError(f"invalid intent for case {case_id}: {intent}")
        evaluation_case = EvaluationCase(
            evaluation_id=evaluation_id,
            run_id=run_id,
            case_id=case_id,
            source=str(result.get("source", source_row.get("source", ""))),
            language=language,
            jailbreak=jailbreak,
            model=model,
            intent=intent,  # type: ignore[arg-type]
            forbidden_prompt=forbidden_prompt,
            response=response,
            generation_status=status,
            response_sha256=response_sha256,
        )
        prior_case = seen.get(evaluation_id)
        if prior_case is not None and prior_case != evaluation_case:
            raise ArtifactConflictError(f"evaluation identity conflict: {evaluation_id}")
        seen[evaluation_id] = evaluation_case
        cases.append(evaluation_case)
    return sorted(
        cases,
        key=lambda item: (
            item.jailbreak,
            item.language,
            item.model,
            item.case_id,
        ),
    )


class JsonlJournal(Generic[T]):
    """Append-only validated JSONL storage with stable identity conflict checks."""

    def __init__(self, path: Path, model: type[T], *, id_field: str = "evaluation_id") -> None:
        self.path = path
        self.model = model
        self.id_field = id_field

    def load(self) -> dict[str, T]:
        values: dict[str, T] = {}
        for row in read_jsonl(self.path, required=False):
            record = self.model.model_validate(row)
            record_id = str(getattr(record, self.id_field))
            prior = values.get(record_id)
            if prior is not None and canonical_json(
                prior.model_dump(mode="json")
            ) != canonical_json(record.model_dump(mode="json")):
                raise ArtifactConflictError(
                    f"immutable JSONL row conflict: {self.path} ({record_id})"
                )
            values[record_id] = record
        return values

    def append(self, record: T) -> bool:
        return self.append_many([record]) == 1

    def append_many(self, records: Iterable[T]) -> int:
        existing = self.load()
        pending: list[T] = []
        for record in records:
            record_id = str(getattr(record, self.id_field))
            prior = existing.get(record_id)
            if prior is not None:
                if canonical_json(prior.model_dump(mode="json")) != canonical_json(
                    record.model_dump(mode="json")
                ):
                    raise ArtifactConflictError(
                        f"immutable JSONL row conflict: {self.path} ({record_id})"
                    )
                continue
            existing[record_id] = record
            pending.append(record)
        if not pending:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as stream:
            for record in pending:
                stream.write(canonical_json(record.model_dump(mode="json")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return len(pending)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "ArtifactConflictError",
    "EvaluationArtifactError",
    "JsonlJournal",
    "canonical_json",
    "load_evaluation_cases",
    "read_jsonl",
    "sha256_file",
    "sha256_text",
    "write_json_atomic",
]
