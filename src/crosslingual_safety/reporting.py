"""Hierarchical Markdown reports derived from immutable run artifacts."""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from crosslingual_safety.evaluation.artifacts import (
    load_evaluation_cases,
    read_jsonl,
)
from crosslingual_safety.evaluation.models import (
    ConsensusEvaluation,
    MultilingualJudgment,
    StrongRejectJudgment,
)


@dataclass(frozen=True)
class ReportSummary:
    run_id: str
    jailbreaks: tuple[str, ...]
    results: int
    evaluated: int


def _fence(value: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    return "`" * max(3, longest + 1)


def _percentage(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.1%}"


def _evaluation_maps(
    run_dir: Path,
) -> tuple[
    dict[str, ConsensusEvaluation],
    dict[str, MultilingualJudgment],
    dict[str, StrongRejectJudgment],
]:
    evaluation_dir = run_dir / "evaluation"
    active_config_sha256: str | None = None
    manifest_path = evaluation_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and isinstance(manifest.get("config_sha256"), str):
            active_config_sha256 = manifest["config_sha256"]
    consensus = {
        row.evaluation_id: row
        for row in (
            ConsensusEvaluation.model_validate(value)
            for value in read_jsonl(evaluation_dir / "evaluations.jsonl", required=False)
            if active_config_sha256 is None or value.get("config_sha256") == active_config_sha256
        )
    }
    multilingual = {
        row.evaluation_id: row
        for row in (
            MultilingualJudgment.model_validate(value)
            for value in read_jsonl(evaluation_dir / "multilingual_judge.jsonl", required=False)
        )
    }
    strongreject = {
        row.evaluation_id: row
        for row in (
            StrongRejectJudgment.model_validate(value)
            for value in read_jsonl(evaluation_dir / "strongreject.jsonl", required=False)
        )
    }
    return consensus, multilingual, strongreject


def _metrics(
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
) -> tuple[Counter[str], int, int, str]:
    counts: Counter[str] = Counter()
    for row in rows:
        evaluation = consensus.get(str(row["evaluation_id"]))
        counts[evaluation.verdict if evaluation is not None else "pending"] += 1
    bypass = counts["bypass"]
    denominator = bypass + counts["not_bypass"]
    return counts, bypass, denominator, _percentage(bypass, denominator)


def _conditional_asr(
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
    multilingual: Mapping[str, MultilingualJudgment],
) -> tuple[int, int, str]:
    eligible_ids = {
        evaluation_id
        for row in rows
        if (evaluation_id := str(row["evaluation_id"])) in multilingual
        and multilingual[evaluation_id].status == "success"
        and multilingual[evaluation_id].prompt_understood == "yes"
        and evaluation_id in consensus
        and consensus[evaluation_id].verdict in {"bypass", "not_bypass"}
    }
    bypass = sum(consensus[evaluation_id].verdict == "bypass" for evaluation_id in eligible_ids)
    return bypass, len(eligible_ids), _percentage(bypass, len(eligible_ids))


def _parent_report(
    run_id: str,
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
    multilingual: Mapping[str, MultilingualJudgment],
) -> str:
    jailbreaks = sorted({str(row["jailbreak"]) for row in rows})
    lines = [
        f"# Unified Run {run_id}",
        "",
        "This file is an index. Full responses and Judge results are in each jailbreak report.",
        "",
        "## Jailbreak Summary",
        "",
        "| Jailbreak | Generation | Evaluated | Bypass | Not bypass | Uncertain | "
        "Not evaluable | Pending | Strict ASR | Conditional ASR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for jailbreak in jailbreaks:
        selected = [row for row in rows if row["jailbreak"] == jailbreak]
        generated = sum(str(row["status"]) == "success" for row in selected)
        counts, bypass, denominator, asr = _metrics(selected, consensus)
        conditional_bypass, conditional_denominator, conditional_asr = _conditional_asr(
            selected, consensus, multilingual
        )
        evaluated = len(selected) - counts["pending"]
        link = f"[{jailbreak}](children/{jailbreak}/report.md)"
        lines.append(
            f"| {link} | {generated} / {len(selected)} | {evaluated} / {len(selected)} | "
            f"{bypass} | {counts['not_bypass']} | {counts['uncertain']} | "
            f"{counts['not_evaluable']} | {counts['pending']} | {asr} ({bypass}/{denominator}) | "
            f"{conditional_asr} ({conditional_bypass}/{conditional_denominator}) |"
        )

    lines.extend(
        [
            "",
            "## Language and Model Summary",
            "",
            "| Jailbreak | Language | Model | Bypass | Determinate | Uncertain | Pending | "
            "Strict ASR | Conditional ASR |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["jailbreak"]), str(row["language"]), str(row["model"]))].append(row)
    for (jailbreak, language, model), selected in sorted(groups.items()):
        counts, bypass, denominator, asr = _metrics(selected, consensus)
        conditional_bypass, conditional_denominator, conditional_asr = _conditional_asr(
            selected, consensus, multilingual
        )
        lines.append(
            f"| {jailbreak} | {language} | {model} | {bypass} | {denominator} | "
            f"{counts['uncertain']} | {counts['pending']} | {asr} ({bypass}/{denominator}) | "
            f"{conditional_asr} ({conditional_bypass}/{conditional_denominator}) |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _child_report(
    run_id: str,
    jailbreak: str,
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
    multilingual: Mapping[str, MultilingualJudgment],
    strongreject: Mapping[str, StrongRejectJudgment],
) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["language"]),
            str(row["model"]),
            str(row["case_id"]),
        ),
    )
    counts, bypass, denominator, asr = _metrics(ordered, consensus)
    conditional_bypass, conditional_denominator, conditional_asr = _conditional_asr(
        ordered, consensus, multilingual
    )
    lines = [
        f"# {jailbreak} Results",
        "",
        f"Run: `{run_id}`",
        "",
        f"Generation success: **{sum(str(row['status']) == 'success' for row in ordered)} / "
        f"{len(ordered)}**  ",
        f"Evaluated: **{len(ordered) - counts['pending']} / {len(ordered)}**  ",
        f"Bypass: **{bypass}**  ",
        f"Not bypass: **{counts['not_bypass']}**  ",
        f"Uncertain: **{counts['uncertain']}**  ",
        f"Not evaluable: **{counts['not_evaluable']}**  ",
        f"Pending: **{counts['pending']}**  ",
        f"Strict ASR: **{asr} ({bypass}/{denominator})**  ",
        f"Comprehension-conditioned ASR: **{conditional_asr} "
        f"({conditional_bypass}/{conditional_denominator})**",
        "",
    ]
    current_language: str | None = None
    current_model: tuple[str, str] | None = None
    for row in ordered:
        language = str(row["language"])
        model = str(row["model"])
        if language != current_language:
            lines.extend([f"## {language}", ""])
            current_language = language
            current_model = None
        if (language, model) != current_model:
            lines.extend([f"### {model}", ""])
            current_model = (language, model)
        evaluation_id = str(row["evaluation_id"])
        evaluated = consensus.get(evaluation_id)
        verdict = evaluated.verdict if evaluated is not None else "pending"
        lines.extend(
            [
                f"#### {row['case_id']}",
                "",
                f"Generation: `{row['status']}`  ",
                f"Verdict: `{verdict}`  ",
            ]
        )
        if evaluated is not None:
            lines.append(f"Review reason: `{evaluated.review_reason or 'none'}`  ")
        multilingual_row = multilingual.get(evaluation_id)
        if multilingual_row is not None:
            lines.append(
                f"Multilingual Judge: `{multilingual_row.status}`; "
                f"strict=`{multilingual_row.strict_bypass}`; "
                f"confidence=`{multilingual_row.confidence}`  "
            )
        strongreject_row = strongreject.get(evaluation_id)
        if strongreject_row is not None:
            lines.append(
                f"StrongREJECT: `{strongreject_row.status}`; score=`{strongreject_row.score}`  "
            )
        lines.append("")
        response = str(row.get("response") or row.get("error_message") or "")
        fence = _fence(response)
        lines.extend([f"{fence}text", response, fence, ""])
    return "\n".join(lines).rstrip() + "\n"


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_hierarchical_reports(run_dir: Path) -> ReportSummary:
    cases = load_evaluation_cases(run_dir)
    consensus, multilingual, strongreject = _evaluation_maps(run_dir)
    rows = [
        {
            "evaluation_id": case.evaluation_id,
            "case_id": case.case_id,
            "source": case.source,
            "language": case.language,
            "jailbreak": case.jailbreak,
            "model": case.model,
            "status": case.generation_status,
            "response": case.response,
        }
        for case in cases
    ]
    run_id = cases[0].run_id
    _write_atomic(
        run_dir / "report.md",
        _parent_report(run_id, rows, consensus, multilingual),
    )
    jailbreaks = sorted({case.jailbreak for case in cases})
    for jailbreak in jailbreaks:
        selected = [row for row in rows if row["jailbreak"] == jailbreak]
        _write_atomic(
            run_dir / "children" / jailbreak / "report.md",
            _child_report(
                run_id,
                jailbreak,
                selected,
                consensus,
                multilingual,
                strongreject,
            ),
        )
    return ReportSummary(
        run_id=run_id,
        jailbreaks=tuple(jailbreaks),
        results=len(rows),
        evaluated=len(consensus),
    )


__all__ = ["ReportSummary", "write_hierarchical_reports"]
