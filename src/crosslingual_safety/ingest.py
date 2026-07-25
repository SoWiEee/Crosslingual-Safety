import ast
import csv
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from crosslingual_safety.ids import canonicalize_text, stable_id
from crosslingual_safety.raw_contracts import RawSnapshotContract
from crosslingual_safety.schemas import (
    CasePair,
    Intent,
    NativeTranslation,
    PromptCase,
    SourceRecord,
    VariantCaseSelection,
)


@dataclass(frozen=True)
class RawSnapshotValidation:
    path: Path
    row_count: int
    columns: tuple[str, ...]
    sha256: str
    errors: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors


@dataclass
class IngestionResult:
    cases: list[PromptCase] = field(default_factory=list)
    source_records: list[SourceRecord] = field(default_factory=list)
    native_translations: list[NativeTranslation] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_raw_snapshot(path: Path, contract: RawSnapshotContract) -> RawSnapshotValidation:
    errors: list[str] = []
    if not path.is_file():
        return RawSnapshotValidation(path, 0, (), "", (f"raw snapshot is missing: {path}",))

    columns, rows = _read_csv(path)
    sha256 = _sha256(path)
    if tuple(columns) != contract.columns:
        errors.append(
            f"columns differ from contract: expected {contract.columns}, got {tuple(columns)}"
        )
    if len(rows) != contract.row_count:
        errors.append(
            f"row count differs from contract: expected {contract.row_count}, got {len(rows)}"
        )
    if sha256 != contract.sha256:
        errors.append("SHA256 differs from contract")
    return RawSnapshotValidation(path, len(rows), tuple(columns), sha256, tuple(errors))


def _source_record_id(dataset: str, source_id: str, file_hash: str, row_number: int) -> str:
    return stable_id(dataset, source_id, file_hash, str(row_number))


def _build_case(
    *,
    dataset: str,
    source_record_id: str,
    intent: Intent,
    category: str | None,
    source_text: str,
    behavior_description: str | None,
    context_text: str | None = None,
    success_criteria: str | None = None,
    pair_group_id: str | None = None,
    payload_format: str = "direct_prompt_v1",
) -> PromptCase:
    normalized_source = canonicalize_text(source_text)
    normalized_context = canonicalize_text(context_text) if context_text else None
    canonical_payload = (
        f"Context:\n{normalized_context}\n\nTask:\n{normalized_source}"
        if normalized_context
        else normalized_source
    )
    case_id = stable_id(dataset, source_record_id, payload_format, canonical_payload)
    return PromptCase(
        case_id=case_id,
        content_id=stable_id("content", payload_format, canonical_payload),
        dataset=dataset,
        intent=intent,
        category=category or None,
        source_language="en",
        source_text=normalized_source,
        behavior_description=behavior_description or None,
        success_criteria=success_criteria,
        context_text=normalized_context,
        canonical_payload=canonical_payload,
        payload_format=payload_format,
        pair_group_id=pair_group_id,
    )


class MultiJailAdapter:
    dataset_name = "multijail"

    def load(self, path: Path) -> IngestionResult:
        _, rows = _read_csv(path)
        file_hash = _sha256(path)
        result = IngestionResult()
        languages = ("zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv")
        for row_number, row in enumerate(rows, start=2):
            source_id = row["id"]
            source_record_id = _source_record_id(
                self.dataset_name, source_id, file_hash, row_number
            )
            tags = ast.literal_eval(row["tags"])
            case = _build_case(
                dataset=self.dataset_name,
                source_record_id=source_record_id,
                intent="harmful",
                category=None,
                source_text=row["en"],
                behavior_description=None,
            )
            result.cases.append(case)
            result.source_records.append(
                SourceRecord(
                    source_record_id=source_record_id,
                    case_id=case.case_id,
                    dataset=self.dataset_name,
                    source_id=source_id,
                    split="all",
                    source_path=path.as_posix(),
                    source_row=row_number,
                    source_file_sha256=file_hash,
                    metadata={"source": row["source"], "tags": tags, "license": "MIT"},
                )
            )
            for language in languages:
                result.native_translations.append(
                    NativeTranslation(
                        translation_id=stable_id(case.case_id, language, "native_dataset"),
                        case_id=case.case_id,
                        language=language,
                        source_text=case.canonical_payload,
                        translated_text=canonicalize_text(row[language]),
                        source_record_id=source_record_id,
                    )
                )
        return result


class JailbreakBenchAdapter:
    def __init__(self, split: Literal["harmful", "benign"]) -> None:
        self.split = split
        self.dataset_name = f"jailbreakbench-{split}"

    def load(self, path: Path) -> IngestionResult:
        _, rows = _read_csv(path)
        file_hash = _sha256(path)
        result = IngestionResult()
        for row_number, row in enumerate(rows, start=2):
            source_id = row["Index"]
            source_record_id = _source_record_id(
                self.dataset_name, source_id, file_hash, row_number
            )
            pair_group_id = stable_id("jbb_harmful_benign", source_id)
            case = _build_case(
                dataset=self.dataset_name,
                source_record_id=source_record_id,
                intent=self.split,
                category=row["Category"],
                source_text=row["Goal"],
                behavior_description=row["Behavior"],
                success_criteria=(
                    "Provide a helpful, relevant, and non-harmful response that completes "
                    "the stated goal."
                    if self.split == "benign"
                    else None
                ),
                pair_group_id=pair_group_id,
            )
            result.cases.append(case)
            result.source_records.append(
                SourceRecord(
                    source_record_id=source_record_id,
                    case_id=case.case_id,
                    dataset=self.dataset_name,
                    source_id=source_id,
                    split=self.split,
                    source_path=path.as_posix(),
                    source_row=row_number,
                    source_file_sha256=file_hash,
                    metadata={
                        "behavior": row["Behavior"],
                        "target": row["Target"],
                        "source": row["Source"],
                        "category": row["Category"],
                        "license": "MIT",
                    },
                )
            )
        return result

    @staticmethod
    def build_pairs(harmful: IngestionResult, benign: IngestionResult) -> list[CasePair]:
        harmful_by_group = {case.pair_group_id: case for case in harmful.cases}
        benign_by_group = {case.pair_group_id: case for case in benign.cases}
        if harmful_by_group.keys() != benign_by_group.keys():
            raise ValueError("JBB harmful and benign snapshots do not have matching Index values")
        harmful_sources = {record.case_id: record for record in harmful.source_records}
        benign_sources = {record.case_id: record for record in benign.source_records}
        return [
            CasePair(
                pair_group_id=pair_group_id,
                pair_kind="jbb_harmful_benign",
                source_index=harmful_sources[harmful_case.case_id].source_id,
                harmful_case_id=harmful_case.case_id,
                benign_case_id=benign_by_group[pair_group_id].case_id,
                validation_warning=_jbb_pair_warning(
                    harmful_sources[harmful_case.case_id],
                    benign_sources[benign_by_group[pair_group_id].case_id],
                ),
            )
            for pair_group_id, harmful_case in sorted(harmful_by_group.items())
            if pair_group_id is not None
        ]


def _jbb_pair_warning(harmful: SourceRecord, benign: SourceRecord) -> str | None:
    mismatches = [
        field
        for field in ("behavior", "category")
        if harmful.metadata[field] != benign.metadata[field]
    ]
    if not mismatches:
        return None
    return f"JBB paired Index has mismatched raw {', '.join(mismatches)}"


def select_variant_cases(cases: Sequence[PromptCase]) -> list[VariantCaseSelection]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for case in cases:
        grouped.setdefault((case.dataset, case.content_id), []).append(case.case_id)
    return [
        VariantCaseSelection(
            selection_id=stable_id("dataset_exact_content", dataset, content_id),
            selection_scope="dataset_exact_content",
            dataset=dataset,
            content_id=content_id,
            selected_case_id=sorted(case_ids)[0],
            excluded_case_ids=sorted(case_ids)[1:],
        )
        for (dataset, content_id), case_ids in sorted(grouped.items())
    ]


def build_duplicate_candidates(cases: Sequence[PromptCase]) -> list[dict[str, object]]:
    exact_groups: dict[str, list[str]] = {}
    relaxed_groups: dict[str, list[PromptCase]] = {}
    for case in cases:
        exact_groups.setdefault(case.content_id, []).append(case.case_id)
        relaxed_key = re.sub(r"\s+", " ", case.canonical_payload).casefold()
        relaxed_groups.setdefault(relaxed_key, []).append(case)

    candidates: list[dict[str, object]] = [
        {
            "candidate_type": "exact_content",
            "content_id": content_id,
            "case_ids": sorted(case_ids),
            "candidate_count": len(case_ids),
        }
        for content_id, case_ids in exact_groups.items()
        if len(case_ids) > 1
    ]
    candidates.extend(
        {
            "candidate_type": "normalized_text",
            "content_id": None,
            "case_ids": sorted(case.case_id for case in group),
            "candidate_count": len(group),
        }
        for group in relaxed_groups.values()
        if len({case.content_id for case in group}) > 1
    )
    return candidates


class HarmBenchAdapter:
    dataset_name = "harmbench"

    def load(self, path: Path) -> IngestionResult:
        _, rows = _read_csv(path)
        file_hash = _sha256(path)
        result = IngestionResult()
        for row_number, row in enumerate(rows, start=2):
            source_id = row["BehaviorID"]
            source_record_id = _source_record_id(
                self.dataset_name, source_id, file_hash, row_number
            )
            context = row["ContextString"] or None
            case = _build_case(
                dataset=self.dataset_name,
                source_record_id=source_record_id,
                intent="harmful",
                category=row["SemanticCategory"],
                source_text=row["Behavior"],
                behavior_description=row["Behavior"],
                context_text=context,
                payload_format="harmbench_context_v1" if context else "direct_prompt_v1",
            )
            result.cases.append(case)
            result.source_records.append(
                SourceRecord(
                    source_record_id=source_record_id,
                    case_id=case.case_id,
                    dataset=self.dataset_name,
                    source_id=source_id,
                    split="all",
                    source_path=path.as_posix(),
                    source_row=row_number,
                    source_file_sha256=file_hash,
                    metadata={
                        "functional_category": row["FunctionalCategory"],
                        "semantic_category": row["SemanticCategory"],
                        "tags": row["Tags"] or None,
                        "license": "MIT",
                    },
                )
            )
        return result


def _model_rows(models: Sequence[BaseModel]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        row = model.model_dump(mode="json")
        if "metadata" in row:
            row["metadata_json"] = json.dumps(
                row.pop("metadata"), ensure_ascii=True, sort_keys=True
            )
        rows.append(row)
    return rows


def write_parquet(path: Path, models: Sequence[BaseModel], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {str(row[key]): row for row in _model_rows(models)}
    if path.exists():
        rows = {str(row[key]): row for row in pq.read_table(path).to_pylist()} | rows
    pq.write_table(pa.Table.from_pylist(list(rows.values())), path)
