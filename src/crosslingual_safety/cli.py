import json
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from crosslingual_safety.generation.commands import register_generation_commands
from crosslingual_safety.ingest import (
    HarmBenchAdapter,
    IngestionResult,
    JailbreakBenchAdapter,
    MultiJailAdapter,
    build_duplicate_candidates,
    select_variant_cases,
    validate_raw_snapshot,
    write_parquet,
)
from crosslingual_safety.jailbreaks import register_jailbreak_commands
from crosslingual_safety.manual_commands import register_manual_commands
from crosslingual_safety.raw_contracts import RAW_SNAPSHOT_CONTRACTS
from crosslingual_safety.schemas import PromptCase
from crosslingual_safety.translation.commands import register_translation_commands

app = typer.Typer(no_args_is_help=True)
register_translation_commands(app)
register_jailbreak_commands(app)
register_generation_commands(app)
register_manual_commands(app)


def _validate_or_fail(repo_root: Path, contract_name: str, input_path: Path | None = None) -> Path:
    contract = RAW_SNAPSHOT_CONTRACTS[contract_name]
    path = input_path or repo_root / contract.relative_path
    validation = validate_raw_snapshot(path, contract)
    if not validation.is_valid:
        raise typer.BadParameter("; ".join(validation.errors), param_hint="--dataset")
    return path


def _merge_results(*results: IngestionResult) -> IngestionResult:
    merged = IngestionResult()
    for result in results:
        merged.cases.extend(result.cases)
        merged.source_records.extend(result.source_records)
        merged.native_translations.extend(result.native_translations)
    return merged


@app.command()
def ingest(
    dataset: Annotated[
        str,
        typer.Option(
            help="all, multijail, jailbreakbench-harmful, jailbreakbench-benign, or harmbench"
        ),
    ] = "all",
    repo_root: Annotated[Path, typer.Option(file_okay=False)] = Path("."),
    input_path: Annotated[
        Path | None, typer.Option("--input", file_okay=True, dir_okay=False)
    ] = None,
    output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data/normalized"),
) -> None:
    """Validate pinned raw snapshots and create idempotent normalized Parquet outputs."""
    supported = {"all", "multijail", "jailbreakbench-harmful", "jailbreakbench-benign", "harmbench"}
    if dataset not in supported:
        raise typer.BadParameter(f"dataset must be one of: {', '.join(sorted(supported))}")

    selected = {dataset} if dataset != "all" else supported - {"all"}
    if input_path is not None and len(selected) != 1:
        raise typer.BadParameter("--input requires exactly one --dataset")
    results: list[IngestionResult] = []
    pairs = []
    if "multijail" in selected:
        results.append(
            MultiJailAdapter().load(_validate_or_fail(repo_root, "multijail", input_path))
        )
    jbb_results: dict[str, IngestionResult] = {}
    if "jailbreakbench-harmful" in selected:
        harmful = JailbreakBenchAdapter("harmful").load(
            _validate_or_fail(repo_root, "jbb_harmful", input_path)
        )
        jbb_results["harmful"] = harmful
        results.append(harmful)
    if "jailbreakbench-benign" in selected:
        benign = JailbreakBenchAdapter("benign").load(
            _validate_or_fail(repo_root, "jbb_benign", input_path)
        )
        jbb_results["benign"] = benign
        results.append(benign)
    if jbb_results.keys() == {"harmful", "benign"}:
        pairs = JailbreakBenchAdapter.build_pairs(harmful, benign)
    elif jbb_results and (output_dir / "cases.parquet").is_file():
        counterpart = (
            JailbreakBenchAdapter("benign").load(_validate_or_fail(repo_root, "jbb_benign"))
            if "harmful" in jbb_results
            else JailbreakBenchAdapter("harmful").load(_validate_or_fail(repo_root, "jbb_harmful"))
        )
        existing_case_ids = {
            row["case_id"] for row in pq.read_table(output_dir / "cases.parquet").to_pylist()
        }
        if {case.case_id for case in counterpart.cases} <= existing_case_ids:
            harmful_result = jbb_results.get("harmful", counterpart)
            benign_result = jbb_results.get("benign", counterpart)
            pairs = JailbreakBenchAdapter.build_pairs(harmful_result, benign_result)
    if "harmbench" in selected:
        results.append(
            HarmBenchAdapter().load(_validate_or_fail(repo_root, "harmbench", input_path))
        )

    merged = _merge_results(*results)
    write_parquet(output_dir / "cases.parquet", merged.cases, "case_id")
    write_parquet(output_dir / "source_records.parquet", merged.source_records, "source_record_id")
    if merged.native_translations:
        write_parquet(
            output_dir / "native_translations.parquet", merged.native_translations, "translation_id"
        )
    if pairs:
        write_parquet(output_dir / "case_pairs.parquet", pairs, "pair_group_id")
    normalized_cases = [
        PromptCase.model_validate(row)
        for row in pq.read_table(output_dir / "cases.parquet").to_pylist()
    ]
    write_parquet(
        output_dir / "variant_case_selection.parquet",
        select_variant_cases(normalized_cases),
        "selection_id",
    )
    _write_raw_snapshot_inventory(output_dir, selected)
    typer.echo(
        f"ingested {len(merged.cases)} cases, {len(merged.source_records)} source records, "
        f"and {len(pairs)} JBB pairs"
    )


def _write_raw_snapshot_inventory(output_dir: Path, selected: set[str]) -> None:
    contract_names = {
        "multijail": "multijail",
        "jailbreakbench-harmful": "jbb_harmful",
        "jailbreakbench-benign": "jbb_benign",
        "harmbench": "harmbench",
    }
    inventory_path = output_dir / "raw_snapshot_inventory.json"
    existing = {
        entry["dataset"]: entry
        for entry in (
            json.loads(inventory_path.read_text(encoding="utf-8"))
            if inventory_path.exists()
            else []
        )
    }
    updates = [
        {
            "dataset": RAW_SNAPSHOT_CONTRACTS[contract_name].dataset,
            "relative_path": str(RAW_SNAPSHOT_CONTRACTS[contract_name].relative_path),
            "row_count": RAW_SNAPSHOT_CONTRACTS[contract_name].row_count,
            "columns": RAW_SNAPSHOT_CONTRACTS[contract_name].columns,
            "sha256": RAW_SNAPSHOT_CONTRACTS[contract_name].sha256,
            "upstream_revision": RAW_SNAPSHOT_CONTRACTS[contract_name].upstream_revision,
        }
        for dataset, contract_name in contract_names.items()
        if dataset in selected
    ]
    existing.update({entry["dataset"]: entry for entry in updates})
    inventory_path.write_text(
        json.dumps([existing[key] for key in sorted(existing)], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


@app.command()
def deduplicate(
    input_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
        "data/normalized/cases.parquet"
    ),
    output_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
        "data/normalized/duplicate_candidates.parquet"
    ),
    selection_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
        "data/normalized/variant_case_selection.parquet"
    ),
) -> None:
    """Emit exact-content groups; variants are not generated from duplicate members."""
    rows = pq.read_table(input_path).to_pylist()
    cases = [PromptCase.model_validate(row) for row in rows]
    candidates = build_duplicate_candidates(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(candidates), output_path)
    write_parquet(
        selection_path,
        select_variant_cases(cases),
        "selection_id",
    )
    typer.echo(f"wrote {len(candidates)} duplicate candidate groups")
