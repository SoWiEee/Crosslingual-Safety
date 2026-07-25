import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.ingest import (
    HarmBenchAdapter,
    JailbreakBenchAdapter,
    MultiJailAdapter,
    build_duplicate_candidates,
    select_variant_cases,
    validate_raw_snapshot,
)
from crosslingual_safety.raw_contracts import RAW_SNAPSHOT_CONTRACTS
from crosslingual_safety.schemas import PromptCase

REPO_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.mark.parametrize("contract", RAW_SNAPSHOT_CONTRACTS.values())
def test_current_raw_snapshots_match_contract(contract) -> None:
    validation = validate_raw_snapshot(REPO_ROOT / contract.relative_path, contract)

    assert validation.is_valid, validation.errors
    assert validation.row_count == contract.row_count
    assert validation.sha256 == contract.sha256


def test_raw_snapshot_contract_rejects_tampered_file(tmp_path: Path) -> None:
    contract = RAW_SNAPSHOT_CONTRACTS["jbb_harmful"]
    tampered = tmp_path / "harmful-behaviors.csv"
    tampered.write_text("Index,Goal\n0,changed\n", encoding="utf-8")

    validation = validate_raw_snapshot(tampered, contract)

    assert not validation.is_valid
    assert any("SHA256" in error for error in validation.errors)
    assert any("row count" in error for error in validation.errors)


def test_multijail_preserves_parallel_native_translations() -> None:
    result = MultiJailAdapter().load(REPO_ROOT / "data/raw/MultiJail/MultiJail.csv")

    assert len(result.cases) == 315
    assert len(result.native_translations) == 315 * 9
    assert {translation.language for translation in result.native_translations} >= {"zh", "jv"}
    assert all(case.intent == "harmful" for case in result.cases)


def test_jbb_uses_index_for_harmful_benign_pairs() -> None:
    harmful = JailbreakBenchAdapter("harmful").load(
        REPO_ROOT / "data/raw/JBB-Behaviors/data/harmful-behaviors.csv"
    )
    benign = JailbreakBenchAdapter("benign").load(
        REPO_ROOT / "data/raw/JBB-Behaviors/data/benign-behaviors.csv"
    )

    pairs = JailbreakBenchAdapter.build_pairs(harmful, benign)

    assert len(harmful.cases) == len(benign.cases) == 100
    assert len(pairs) == 100
    assert all(pair.harmful_case_id != pair.benign_case_id for pair in pairs)
    assert all(case.success_criteria for case in benign.cases)
    index_67 = next(pair for pair in pairs if pair.source_index == "67")
    assert index_67.validation_warning == "JBB paired Index has mismatched raw behavior"


def test_harmbench_context_is_part_of_canonical_payload() -> None:
    result = HarmBenchAdapter().load(
        REPO_ROOT / "data/raw/Harmbench/harmbench_behaviors_text_all.csv"
    )
    contextual = [case for case in result.cases if case.context_text]

    assert len(result.cases) == 400
    assert len(contextual) == 100
    assert all(case.context_text in case.canonical_payload for case in contextual)
    assert all(case.source_text in case.canonical_payload for case in contextual)
    assert all(case.payload_format == "harmbench_context_v1" for case in contextual)


def test_deduplicate_reports_relaxed_matches_without_excluding_them() -> None:
    first = PromptCase(
        case_id="one",
        content_id="content-one",
        dataset="example",
        intent="harmful",
        category=None,
        source_language="en",
        source_text="Prompt",
        behavior_description=None,
        success_criteria=None,
        context_text=None,
        canonical_payload="A  Prompt",
        payload_format="direct_prompt_v1",
    )
    second = first.model_copy(
        update={"case_id": "two", "content_id": "content-two", "canonical_payload": "a prompt"}
    )

    candidates = build_duplicate_candidates([first, second])

    assert candidates == [
        {
            "candidate_type": "normalized_text",
            "content_id": None,
            "case_ids": ["one", "two"],
            "candidate_count": 2,
        }
    ]


def test_variant_selection_is_scoped_by_dataset() -> None:
    first = PromptCase(
        case_id="one",
        content_id="shared-content",
        dataset="first-dataset",
        intent="harmful",
        category=None,
        source_language="en",
        source_text="Prompt",
        behavior_description=None,
        success_criteria=None,
        context_text=None,
        canonical_payload="Prompt",
        payload_format="direct_prompt_v1",
    )
    second = first.model_copy(update={"case_id": "two", "dataset": "second-dataset"})

    selections = select_variant_cases([first, second])

    assert len(selections) == 2
    assert {selection.dataset for selection in selections} == {"first-dataset", "second-dataset"}
    assert len({selection.selection_id for selection in selections}) == 2


def test_ingest_cli_writes_idempotent_normalized_snapshots(tmp_path: Path) -> None:
    output_dir = tmp_path / "normalized"

    first = runner.invoke(
        app,
        ["ingest", "--repo-root", str(REPO_ROOT), "--output-dir", str(output_dir)],
    )
    second = runner.invoke(
        app,
        ["ingest", "--repo-root", str(REPO_ROOT), "--output-dir", str(output_dir)],
    )

    assert first.exit_code == second.exit_code == 0, first.output
    assert "ingested 915 cases" in first.output
    assert (output_dir / "cases.parquet").is_file()
    assert (output_dir / "source_records.parquet").is_file()
    assert (output_dir / "native_translations.parquet").is_file()
    assert (output_dir / "case_pairs.parquet").is_file()
    assert (output_dir / "raw_snapshot_inventory.json").is_file()
    assert (output_dir / "variant_case_selection.parquet").is_file()
    assert pq.read_table(output_dir / "cases.parquet").num_rows == 915
    assert pq.read_table(output_dir / "case_pairs.parquet").num_rows == 100
    assert pq.read_table(output_dir / "variant_case_selection.parquet").num_rows <= 915
    inventory = json.loads((output_dir / "raw_snapshot_inventory.json").read_text(encoding="utf-8"))
    assert {entry["dataset"] for entry in inventory} == {
        "multijail",
        "jailbreakbench-harmful",
        "jailbreakbench-benign",
        "harmbench",
    }
    assert next(entry for entry in inventory if entry["dataset"] == "multijail")[
        "upstream_revision"
    ]


def test_jbb_split_commands_only_persist_the_requested_split(tmp_path: Path) -> None:
    output_dir = tmp_path / "normalized"

    harmful = runner.invoke(
        app,
        [
            "ingest",
            "--dataset",
            "jailbreakbench-harmful",
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert harmful.exit_code == 0, harmful.output
    assert pq.read_table(output_dir / "cases.parquet").num_rows == 100
    assert not (output_dir / "case_pairs.parquet").exists()

    benign = runner.invoke(
        app,
        [
            "ingest",
            "--dataset",
            "jailbreakbench-benign",
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert benign.exit_code == 0, benign.output
    assert pq.read_table(output_dir / "cases.parquet").num_rows == 200
    assert pq.read_table(output_dir / "case_pairs.parquet").num_rows == 100
    inventory = json.loads((output_dir / "raw_snapshot_inventory.json").read_text(encoding="utf-8"))
    assert {entry["dataset"] for entry in inventory} == {
        "jailbreakbench-harmful",
        "jailbreakbench-benign",
    }
