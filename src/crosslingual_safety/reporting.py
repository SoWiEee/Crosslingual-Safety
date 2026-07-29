"""Hierarchical Markdown reports derived from immutable run artifacts."""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

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


@dataclass(frozen=True)
class MetricSummary:
    counts: Counter[str]
    eligible: int
    bypass: int
    determinate: int
    conservative_asr: float | None
    determinate_asr: float | None
    determinate_coverage: float | None
    dual_judge_count: int
    dual_judge_coverage: float | None
    uncertain: int
    uncertain_rate: float | None
    refusal_count: int
    refusal_denominator: int
    refusal_rate: float | None
    strongreject_count: int
    strongreject_mean: float | None
    strongreject_median: float | None
    strongreject_pass_count: int
    strongreject_pass_denominator: int
    strongreject_pass_rate: float | None
    strongreject_truncated_count: int
    strongreject_truncation_rate: float | None


def _fence(value: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    return "`" * max(3, longest + 1)


def _percentage(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.1%}"


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_metric(value: float | None, numerator: int, denominator: int) -> str:
    return f"{_format_ratio(value)} ({numerator}/{denominator})"


def _format_uplift(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f} pp"


def _research_cells(summary: MetricSummary, uplift: float | None) -> list[str]:
    return [
        str(summary.eligible),
        _format_metric(summary.conservative_asr, summary.bypass, summary.eligible),
        _format_metric(
            summary.determinate_coverage,
            summary.determinate,
            summary.eligible,
        ),
        _format_metric(
            summary.dual_judge_coverage,
            summary.dual_judge_count,
            summary.eligible,
        ),
        _format_metric(summary.uncertain_rate, summary.uncertain, summary.eligible),
        _format_metric(
            summary.refusal_rate,
            summary.refusal_count,
            summary.refusal_denominator,
        ),
        f"{summary.strongreject_count} / {summary.eligible}",
        _format_score(summary.strongreject_mean),
        _format_score(summary.strongreject_median),
        _format_metric(
            summary.strongreject_pass_rate,
            summary.strongreject_pass_count,
            summary.strongreject_pass_denominator,
        ),
        _format_metric(
            summary.strongreject_truncation_rate,
            summary.strongreject_truncated_count,
            summary.strongreject_count,
        ),
        _format_uplift(uplift),
    ]


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


def _metric_summary(
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
    multilingual: Mapping[str, MultilingualJudgment],
    strongreject: Mapping[str, StrongRejectJudgment],
) -> MetricSummary:
    counts, _, _, _ = _metrics(rows, consensus)
    eligible_rows = [
        row
        for row in rows
        if str(row["status"]) == "success" and str(row.get("intent")) == "harmful"
    ]
    eligible_ids = [str(row["evaluation_id"]) for row in eligible_rows]
    verdicts = [
        consensus[evaluation_id].verdict
        for evaluation_id in eligible_ids
        if evaluation_id in consensus
    ]
    bypass = verdicts.count("bypass")
    determinate = bypass + verdicts.count("not_bypass")
    uncertain = verdicts.count("uncertain")

    multilingual_rows = [
        multilingual[evaluation_id]
        for evaluation_id in eligible_ids
        if evaluation_id in multilingual and multilingual[evaluation_id].status == "success"
    ]
    dual_judge_count = sum(
        evaluation_id in multilingual
        and multilingual[evaluation_id].status == "success"
        and evaluation_id in strongreject
        and strongreject[evaluation_id].status == "success"
        for evaluation_id in eligible_ids
    )
    refusal_count = sum(row.is_refusal == "yes" for row in multilingual_rows)

    strongreject_rows = [
        strongreject[evaluation_id]
        for evaluation_id in eligible_ids
        if evaluation_id in strongreject and strongreject[evaluation_id].status == "success"
    ]
    scores = [row.score for row in strongreject_rows if row.score is not None]
    threshold_pairs = [
        (strongreject[evaluation_id].score, consensus[evaluation_id].strongreject_threshold)
        for evaluation_id in eligible_ids
        if evaluation_id in strongreject
        and strongreject[evaluation_id].status == "success"
        and strongreject[evaluation_id].score is not None
        and evaluation_id in consensus
    ]
    strongreject_pass_count = sum(
        score >= threshold for score, threshold in threshold_pairs if score is not None
    )
    truncated = sum(row.prompt_truncated or row.response_truncated for row in strongreject_rows)

    return MetricSummary(
        counts=counts,
        eligible=len(eligible_rows),
        bypass=bypass,
        determinate=determinate,
        conservative_asr=_ratio(bypass, len(eligible_rows)),
        determinate_asr=_ratio(bypass, determinate),
        determinate_coverage=_ratio(determinate, len(eligible_rows)),
        dual_judge_count=dual_judge_count,
        dual_judge_coverage=_ratio(dual_judge_count, len(eligible_rows)),
        uncertain=uncertain,
        uncertain_rate=_ratio(uncertain, len(eligible_rows)),
        refusal_count=refusal_count,
        refusal_denominator=len(multilingual_rows),
        refusal_rate=_ratio(refusal_count, len(multilingual_rows)),
        strongreject_count=len(scores),
        strongreject_mean=fmean(scores) if scores else None,
        strongreject_median=median(scores) if scores else None,
        strongreject_pass_count=strongreject_pass_count,
        strongreject_pass_denominator=len(threshold_pairs),
        strongreject_pass_rate=_ratio(strongreject_pass_count, len(threshold_pairs)),
        strongreject_truncated_count=truncated,
        strongreject_truncation_rate=_ratio(truncated, len(strongreject_rows)),
    )


def _matched_none_rows(
    selected: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    matched_keys = {
        (
            str(row["source"]),
            str(row["case_id"]),
            str(row["language"]),
            str(row["model"]),
        )
        for row in selected
    }
    return [
        row
        for row in rows
        if str(row["jailbreak"]) == "none"
        and (
            str(row["source"]),
            str(row["case_id"]),
            str(row["language"]),
            str(row["model"]),
        )
        in matched_keys
    ]


def _psa_uplift(
    jailbreak: str,
    selected: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
    multilingual: Mapping[str, MultilingualJudgment],
    strongreject: Mapping[str, StrongRejectJudgment],
) -> float | None:
    if not jailbreak.startswith("psa_"):
        return None
    attack = _metric_summary(selected, consensus, multilingual, strongreject)
    baseline_rows = _matched_none_rows(selected, rows)
    baseline = _metric_summary(baseline_rows, consensus, multilingual, strongreject)
    if attack.conservative_asr is None or baseline.conservative_asr is None:
        return None
    return attack.conservative_asr - baseline.conservative_asr


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
    strongreject: Mapping[str, StrongRejectJudgment],
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
            "## Research Metrics",
            "",
            "| Jailbreak | Eligible | Conservative ASR | Determinate Coverage | "
            "Dual-Judge Coverage | Uncertain Rate | Refusal Rate | StrongREJECT Successful | "
            "StrongREJECT Mean | StrongREJECT Median | StrongREJECT >= Threshold | "
            "StrongREJECT Truncated | PSA Uplift vs none |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |",
        ]
    )
    for jailbreak in jailbreaks:
        selected = [row for row in rows if row["jailbreak"] == jailbreak]
        research = _metric_summary(selected, consensus, multilingual, strongreject)
        uplift = _psa_uplift(
            jailbreak,
            selected,
            rows,
            consensus,
            multilingual,
            strongreject,
        )
        link = f"[{jailbreak}](children/{jailbreak}/report.md)"
        lines.append(f"| {link} | {' | '.join(_research_cells(research, uplift))} |")

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

    lines.extend(
        [
            "",
            "## Language and Model Research Metrics",
            "",
            "| Jailbreak | Language | Model | Eligible | Conservative ASR | "
            "Determinate Coverage | Dual-Judge Coverage | Uncertain Rate | Refusal Rate | "
            "StrongREJECT Successful | StrongREJECT Mean | StrongREJECT Median | "
            "StrongREJECT >= Threshold | StrongREJECT Truncated | PSA Uplift vs none |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |",
        ]
    )
    for (jailbreak, language, model), selected in sorted(groups.items()):
        research = _metric_summary(selected, consensus, multilingual, strongreject)
        uplift = _psa_uplift(
            jailbreak,
            selected,
            rows,
            consensus,
            multilingual,
            strongreject,
        )
        prefix = f"{jailbreak} | {language} | {model}"
        lines.append(f"| {prefix} | {' | '.join(_research_cells(research, uplift))} |")
    return "\n".join(lines).rstrip() + "\n"


def _child_report(
    run_id: str,
    jailbreak: str,
    rows: Sequence[Mapping[str, object]],
    consensus: Mapping[str, ConsensusEvaluation],
    multilingual: Mapping[str, MultilingualJudgment],
    strongreject: Mapping[str, StrongRejectJudgment],
    all_rows: Sequence[Mapping[str, object]],
    paper: Mapping[str, object] | None = None,
    translation_provenance: Mapping[tuple[str, str], int] | None = None,
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
    research = _metric_summary(ordered, consensus, multilingual, strongreject)
    uplift = _psa_uplift(
        jailbreak,
        ordered,
        all_rows,
        consensus,
        multilingual,
        strongreject,
    )
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
        f"({conditional_bypass}/{conditional_denominator})**  ",
        f"Conservative ASR: **{_format_ratio(research.conservative_asr)} "
        f"({research.bypass}/{research.eligible})**  ",
        f"Determinate Coverage: **{_format_ratio(research.determinate_coverage)} "
        f"({research.determinate}/{research.eligible})**  ",
        f"Dual-Judge Coverage: **{_format_ratio(research.dual_judge_coverage)} "
        f"({research.dual_judge_count}/{research.eligible})**  ",
        f"Uncertain Rate: **{_format_ratio(research.uncertain_rate)} "
        f"({research.uncertain}/{research.eligible})**  ",
        f"Refusal Rate: **{_format_ratio(research.refusal_rate)} "
        f"({research.refusal_count}/{research.refusal_denominator})**  ",
        f"StrongREJECT Successful: **{research.strongreject_count} / {research.eligible}**  ",
        f"StrongREJECT Mean: **{_format_score(research.strongreject_mean)}**  ",
        f"StrongREJECT Median: **{_format_score(research.strongreject_median)}**  ",
        f"StrongREJECT >= Threshold: **{_format_ratio(research.strongreject_pass_rate)} "
        f"({research.strongreject_pass_count}/{research.strongreject_pass_denominator})**  ",
        f"StrongREJECT Truncated: **{_format_ratio(research.strongreject_truncation_rate)} "
        f"({research.strongreject_truncated_count}/{research.strongreject_count})**  ",
        f"PSA Uplift vs none: **{_format_uplift(uplift)}**",
        "",
    ]
    if paper is not None:
        lines.extend(
            [
                "## Paper Source",
                "",
                f"Title: {paper.get('title', 'unknown')}  ",
                f"SHA-256: `{paper.get('source_sha256', 'unknown')}`",
                "",
            ]
        )
    if translation_provenance:
        lines.extend(
            [
                "## Translation Provenance",
                "",
                "| Language | Method | Records |",
                "| --- | --- | ---: |",
            ]
        )
        for (language, method), count in sorted(translation_provenance.items()):
            lines.append(f"| {language} | {method} | {count} |")
        lines.append("")
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
            "intent": case.intent,
            "response": case.response,
        }
        for case in cases
    ]
    run_id = cases[0].run_id
    paper_contracts: Mapping[str, object] = {}
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        configured_papers = contract.get("psa_papers") if isinstance(contract, dict) else None
        if isinstance(configured_papers, dict):
            paper_contracts = configured_papers
    translation_provenance: Counter[tuple[str, str]] = Counter()
    for row in read_jsonl(run_dir / "audit" / "translations.jsonl", required=False):
        language = row.get("target_language")
        method = row.get("method")
        if isinstance(language, str) and isinstance(method, str):
            translation_provenance[(language, method)] += 1
    _write_atomic(
        run_dir / "report.md",
        _parent_report(run_id, rows, consensus, multilingual, strongreject),
    )
    jailbreaks = sorted({case.jailbreak for case in cases})
    for jailbreak in jailbreaks:
        selected = [row for row in rows if row["jailbreak"] == jailbreak]
        configured_paper = paper_contracts.get(jailbreak)
        paper = configured_paper if isinstance(configured_paper, Mapping) else None
        _write_atomic(
            run_dir / "children" / jailbreak / "report.md",
            _child_report(
                run_id,
                jailbreak,
                selected,
                consensus,
                multilingual,
                strongreject,
                rows,
                paper=paper,
                translation_provenance=translation_provenance,
            ),
        )
    return ReportSummary(
        run_id=run_id,
        jailbreaks=tuple(jailbreaks),
        results=len(rows),
        evaluated=len(consensus),
    )


__all__ = ["ReportSummary", "write_hierarchical_reports"]
