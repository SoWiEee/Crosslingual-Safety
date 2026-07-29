"""Resumable orchestration for offline dual-Judge evaluation."""

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from crosslingual_safety.evaluation.artifacts import (
    JsonlJournal,
    canonical_json,
    load_evaluation_cases,
    sha256_file,
    write_json_atomic,
)
from crosslingual_safety.evaluation.models import (
    ConsensusEvaluation,
    EvaluationCase,
    EvaluationConfig,
    MultilingualJudgment,
    ResponseTranslation,
    StrongRejectJudgment,
    derive_consensus,
)
from crosslingual_safety.evaluation.multilingual import MultilingualJudge
from crosslingual_safety.evaluation.strongreject import (
    StrongRejectJudge,
    translate_case_to_english,
)
from crosslingual_safety.ids import stable_id
from crosslingual_safety.translation.providers import Translator


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class EvaluationDependencies:
    translator: Translator
    multilingual_judge: MultilingualJudge
    strongreject_judge: StrongRejectJudge
    clock: Callable[[], str] = _utc_now
    emit: Callable[[str], None] = print
    on_progress: Callable[[Path], None] | None = None
    multilingual_batch_size: int = 2


@dataclass(frozen=True)
class EvaluationExecution:
    run_id: str
    status: str
    total: int
    completed: int
    verdict_counts: dict[str, int]
    evaluations_path: Path


def _config_hash(config: EvaluationConfig) -> str:
    return hashlib.sha256(
        canonical_json(config.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _judge_config_hash(config: EvaluationConfig) -> str:
    value = {
        "version": config.version,
        "multilingual_judge_model": config.multilingual_judge_model,
        "response_translator": config.response_translator,
        "strongreject": config.strongreject.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _not_evaluable(
    case: EvaluationCase,
    config: EvaluationConfig,
    config_sha256: str,
    reason: str,
) -> ConsensusEvaluation:
    return ConsensusEvaluation(
        consensus_id=stable_id("consensus", case.evaluation_id, config_sha256),
        evaluation_id=case.evaluation_id,
        config_sha256=config_sha256,
        verdict="not_evaluable",
        review_reason=reason,
        multilingual_strict_bypass=None,
        multilingual_confidence=None,
        strongreject_score=None,
        strongreject_threshold=config.strongreject_threshold,
        multilingual_confidence_threshold=config.multilingual_confidence_threshold,
    )


def _consensus(
    case: EvaluationCase,
    config: EvaluationConfig,
    multilingual: MultilingualJudgment | None,
    strongreject: StrongRejectJudgment | None,
    config_sha256: str,
) -> ConsensusEvaluation:
    decision = derive_consensus(
        multilingual,
        strongreject,
        strongreject_threshold=config.strongreject_threshold,
        confidence_threshold=config.multilingual_confidence_threshold,
    )
    return ConsensusEvaluation(
        consensus_id=stable_id("consensus", case.evaluation_id, config_sha256),
        evaluation_id=case.evaluation_id,
        config_sha256=config_sha256,
        verdict=decision.verdict,
        review_reason=decision.review_reason,
        multilingual_strict_bypass=(
            multilingual.strict_bypass
            if multilingual is not None and multilingual.status == "success"
            else None
        ),
        multilingual_confidence=(
            multilingual.confidence
            if multilingual is not None and multilingual.status == "success"
            else None
        ),
        strongreject_score=(
            strongreject.score
            if strongreject is not None and strongreject.status == "success"
            else None
        ),
        strongreject_threshold=config.strongreject_threshold,
        multilingual_confidence_threshold=config.multilingual_confidence_threshold,
    )


async def _evaluate_multilingual(
    cases: Sequence[EvaluationCase],
    judge: MultilingualJudge,
) -> list[MultilingualJudgment]:
    return list(await asyncio.gather(*(judge.evaluate(case) for case in cases)))


def evaluate_run(
    run_dir: Path,
    config: EvaluationConfig,
    dependencies: EvaluationDependencies,
) -> EvaluationExecution:
    cases = load_evaluation_cases(run_dir)
    if not cases:
        raise ValueError("run contains no result rows")
    config_sha256 = _config_hash(config)
    judge_config_sha256 = _judge_config_hash(config)
    input_results_sha256 = sha256_file(run_dir / "results.jsonl")
    evaluation_dir = run_dir / "evaluation"
    translation_journal = JsonlJournal(
        evaluation_dir / "response_translations.jsonl", ResponseTranslation
    )
    multilingual_journal = JsonlJournal(
        evaluation_dir / "multilingual_judge.jsonl", MultilingualJudgment
    )
    strongreject_journal = JsonlJournal(evaluation_dir / "strongreject.jsonl", StrongRejectJudgment)
    consensus_journal = JsonlJournal(
        evaluation_dir / "evaluations.jsonl",
        ConsensusEvaluation,
        id_field="consensus_id",
    )
    started_at = dependencies.clock()
    manifest_path = evaluation_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("invalid existing evaluation manifest") from error
        if not isinstance(prior_manifest, dict) or (
            prior_manifest.get("input_results_sha256") != input_results_sha256
            or prior_manifest.get("judge_config_sha256") != judge_config_sha256
        ):
            raise ValueError("evaluation input or Judge configuration conflicts with existing run")
    write_json_atomic(
        manifest_path,
        {
            "run_id": cases[0].run_id,
            "status": "running",
            "config_sha256": config_sha256,
            "judge_config_sha256": judge_config_sha256,
            "input_results_sha256": input_results_sha256,
            "configuration": config.model_dump(mode="json"),
            "total": len(cases),
            "started_at": started_at,
        },
    )

    consensus_records = consensus_journal.load()
    consensus = {
        record.evaluation_id: record
        for record in consensus_records.values()
        if record.config_sha256 == config_sha256
    }
    for case in cases:
        if case.evaluation_id in consensus:
            continue
        if case.generation_status != "success":
            skipped = _not_evaluable(
                case,
                config,
                config_sha256,
                f"generation_{case.generation_status}",
            )
            consensus_journal.append(skipped)
            consensus[case.evaluation_id] = skipped
        elif case.intent != "harmful":
            skipped = _not_evaluable(case, config, config_sha256, "benign_intent")
            consensus_journal.append(skipped)
            consensus[case.evaluation_id] = skipped

    eligible = [
        case for case in cases if case.generation_status == "success" and case.intent == "harmful"
    ]
    translations = translation_journal.load()
    for index, case in enumerate(eligible, 1):
        if case.evaluation_id in translations:
            continue
        try:
            translated = translate_case_to_english(case, dependencies.translator)
        except Exception:
            dependencies.emit(
                f"translation_failed evaluation_id={case.evaluation_id} language={case.language}"
            )
            continue
        translation_journal.append(translated)
        translations[case.evaluation_id] = translated
        if index % config.report_refresh_records == 0 and dependencies.on_progress is not None:
            dependencies.on_progress(run_dir)

    multilingual = {
        evaluation_id: judgment
        for evaluation_id, judgment in multilingual_journal.load().items()
        if judgment.status == "success"
    }
    multilingual_pending = [case for case in eligible if case.evaluation_id not in multilingual]
    if multilingual_pending:
        dependencies.emit(f"multilingual_judge pending={len(multilingual_pending)}")
        remote_batch_size = max(1, dependencies.multilingual_batch_size)
        for offset in range(0, len(multilingual_pending), remote_batch_size):
            batch = multilingual_pending[offset : offset + remote_batch_size]
            for judgment in asyncio.run(
                _evaluate_multilingual(batch, dependencies.multilingual_judge)
            ):
                if judgment.status != "success":
                    dependencies.emit(
                        "multilingual_judge_failed "
                        f"evaluation_id={judgment.evaluation_id} status={judgment.status}"
                    )
                    continue
                multilingual_journal.append(judgment)
                multilingual[judgment.evaluation_id] = judgment
            if dependencies.on_progress is not None:
                dependencies.on_progress(run_dir)

    strongreject = {
        evaluation_id: judgment
        for evaluation_id, judgment in strongreject_journal.load().items()
        if judgment.status == "success"
    }
    strongreject_pending = [
        (case, translations[case.evaluation_id])
        for case in eligible
        if case.evaluation_id in translations and case.evaluation_id not in strongreject
    ]
    batch_size = config.strongreject.batch_size
    for offset in range(0, len(strongreject_pending), batch_size):
        strong_batch = strongreject_pending[offset : offset + batch_size]
        for strong_judgment in dependencies.strongreject_judge.evaluate_batch(strong_batch):
            if strong_judgment.status != "success":
                dependencies.emit(
                    "strongreject_failed "
                    f"evaluation_id={strong_judgment.evaluation_id} "
                    f"status={strong_judgment.status}"
                )
                continue
            strongreject_journal.append(strong_judgment)
            strongreject[strong_judgment.evaluation_id] = strong_judgment

    for index, case in enumerate(eligible, 1):
        if case.evaluation_id in consensus:
            continue
        if (
            case.evaluation_id not in translations
            or case.evaluation_id not in multilingual
            or case.evaluation_id not in strongreject
        ):
            continue
        evaluated = _consensus(
            case,
            config,
            multilingual.get(case.evaluation_id),
            strongreject.get(case.evaluation_id),
            config_sha256,
        )
        consensus_journal.append(evaluated)
        consensus[case.evaluation_id] = evaluated
        if index % config.report_refresh_records == 0 and dependencies.on_progress is not None:
            dependencies.on_progress(run_dir)

    counts = Counter(record.verdict for record in consensus.values())
    completed = len(consensus)
    status = "success" if completed == len(cases) else "partial"
    write_json_atomic(
        manifest_path,
        {
            "run_id": cases[0].run_id,
            "status": status,
            "config_sha256": config_sha256,
            "judge_config_sha256": judge_config_sha256,
            "input_results_sha256": input_results_sha256,
            "configuration": config.model_dump(mode="json"),
            "total": len(cases),
            "completed": completed,
            "counts": dict(sorted(counts.items())),
            "artifact_counts": {
                "response_translations": len(translations),
                "multilingual_judgments": len(multilingual),
                "strongreject_judgments": len(strongreject),
                "consensus_evaluations": len(consensus),
            },
            "started_at": started_at,
            "completed_at": dependencies.clock(),
        },
    )
    if dependencies.on_progress is not None:
        dependencies.on_progress(run_dir)
    return EvaluationExecution(
        run_id=cases[0].run_id,
        status=status,
        total=len(cases),
        completed=completed,
        verdict_counts=dict(sorted(counts.items())),
        evaluations_path=evaluation_dir / "evaluations.jsonl",
    )


__all__ = [
    "EvaluationDependencies",
    "EvaluationExecution",
    "evaluate_run",
]
