import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TypeVar

import pyarrow.parquet as pq
import typer
import yaml

from crosslingual_safety.generation.commands import _generate_pending
from crosslingual_safety.generation.config import (
    ExperimentConfig,
    ExperimentPaths,
    ExperimentSection,
    GenerationConfig,
    ModelConfig,
)
from crosslingual_safety.generation.queue import JobQueue
from crosslingual_safety.ids import stable_id
from crosslingual_safety.jailbreaks import load_jailbreaks
from crosslingual_safety.manual import (
    ManualLanguage,
    ManualRole,
    ManualTranslation,
    ManualVariant,
    build_manual_generation_requests,
    build_manual_variants,
    load_manual_prompts,
    translate_manual_prompts,
)
from crosslingual_safety.schemas import GenerationRequest
from crosslingual_safety.translation.commands import _translator

DEFAULT_MANUAL_MODELS = (
    "llama31_8b",
    "gemma_4_12b",
    "gemma_4_26b",
    "nemotron_cascade_2_30b",
    "llama33_70b",
)
MANUAL_LANGUAGES: tuple[ManualLanguage, ...] = ("en", "zh", "vi", "my")
MANUAL_ROLES: tuple[ManualRole, ...] = (
    "joker",
    "lex_luthor",
    "riddler",
    "scarecrow",
)
ManualRecordT = TypeVar("ManualRecordT", ManualTranslation, ManualVariant)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_immutable(path: Path, content: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"immutable manual run conflict: {path}")
        return
    _atomic_write(path, content)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@contextmanager
def _preparation_lock(run_dir: Path) -> Iterator[None]:
    connection = sqlite3.connect(run_dir / "preparation.sqlite", timeout=600)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS preparation_lock (lock_id INTEGER PRIMARY KEY)"
        )
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        yield
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _dump_jsonl(values: Sequence[object]) -> str:
    rows = []
    for value in values:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        rows.append(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    return "".join(rows)


def _read_jsonl(path: Path, model: type[ManualRecordT]) -> list[ManualRecordT]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_models(path: Path) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise ValueError(f"invalid models configuration: {path}")
    return {str(name): ModelConfig.model_validate(value) for name, value in raw["models"].items()}


def _split_names(value: str | None) -> list[str]:
    return [] if value is None else [item.strip() for item in value.split(",") if item.strip()]


def _select_models(
    available: dict[str, ModelConfig],
    models: str | None,
    add_models: str | None,
) -> dict[str, ModelConfig]:
    selected = _split_names(models) or list(DEFAULT_MANUAL_MODELS)
    additions = _split_names(add_models)
    invalid_additions = [name for name in additions if name != "nemotron_3_ultra_550b"]
    if invalid_additions:
        raise ValueError("--add-model only supports nemotron_3_ultra_550b")
    for name in additions:
        if name not in selected:
            selected.append(name)
    unknown = sorted(set(selected) - available.keys())
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    return {name: available[name] for name in selected}


def _manual_experiment(
    experiment_id: str,
    models: dict[str, ModelConfig],
    run_dir: Path,
    *,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    seed: int | None,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(
            id=experiment_id,
            datasets=["manual"],
            languages=list(MANUAL_LANGUAGES),
            jailbreaks=["manual"],
            models=list(models),
            generation=GenerationConfig(
                system_prompt_file=None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                retry_backoff_base=1.0,
            ),
        ),
        models=models,
        paths=ExperimentPaths(
            variants=run_dir / "variants.jsonl",
            runs_dir=run_dir.parent,
        ),
    )


def _result_rows(
    run_dir: Path,
    requests: list[tuple[str, GenerationRequest]],
    variants: list[ManualVariant],
) -> list[dict[str, object]]:
    results_path = run_dir / "generation_results.parquet"
    if not results_path.is_file():
        return []
    results = {row["run_id"]: row for row in pq.read_table(results_path).to_pylist()}
    variants_by_id = {variant.variant_id: variant for variant in variants}
    rows: list[dict[str, object]] = []
    for model_name, request in requests:
        result = results.get(request.run_id)
        if result is None:
            continue
        variant = variants_by_id[request.variant_id]
        rows.append(
            {
                "run_id": request.run_id,
                "prompt_id": variant.prompt_id,
                "language": variant.language,
                "role": variant.role,
                "model_name": model_name,
                "requested_model_id": request.requested_model_id,
                "payload": variant.payload,
                "rendered_prompt": request.rendered_prompt,
                "system_prompt": request.system_prompt,
                "attack_id": variant.attack_id,
                "status": result["status"],
                "response_text": result["response_text"],
                "actual_model_id": result["actual_model_id"],
                "finish_reason": result["finish_reason"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "latency_ms": result["latency_ms"],
                "provider_request_id": result["provider_request_id"],
                "error_type": result["error_type"],
                "error_message": result["error_message"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["prompt_id"]),
            str(row["language"]),
            str(row["model_name"]),
        ),
    )


def _report(rows: list[dict[str, object]], run_id: str) -> str:
    lines = [f"# Manual Run {run_id}", ""]
    current: tuple[str, str, str] | None = None
    for row in rows:
        group = (str(row["prompt_id"]), str(row["language"]), str(row["role"]))
        if group != current:
            lines.extend([f"## {group[0]} / {group[1]} / {group[2]}", ""])
            current = group
        response = str(row["response_text"] or row["error_message"] or "")
        response = response.replace("```", "'''")
        lines.extend(
            [
                f"### {row['model_name']}",
                "",
                f"Status: `{row['status']}`",
                "",
                "```text",
                response,
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def register_manual_commands(app: typer.Typer) -> None:
    @app.command("manual-run")
    def manual_run(
        input_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
        source_language: Annotated[str | None, typer.Option()] = None,
        translator_name: Annotated[str, typer.Option("--translator")] = "nllb",
        jailbreak_id: Annotated[str, typer.Option("--jailbreak")] = "none",
        role: Annotated[str, typer.Option()] = "joker",
        wrapper_language_mode: Annotated[str, typer.Option()] = "english",
        models: Annotated[str | None, typer.Option()] = None,
        add_model: Annotated[str | None, typer.Option()] = None,
        models_config: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "configs/models.yaml"
        ),
        languages_config: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "configs/languages.yaml"
        ),
        jailbreaks_config: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "configs/jailbreaks.yaml"
        ),
        runs_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("runs/manual"),
        run_id: Annotated[str | None, typer.Option()] = None,
        temperature: Annotated[float, typer.Option()] = 1.0,
        top_p: Annotated[float | None, typer.Option()] = None,
        max_tokens: Annotated[int, typer.Option(min=1)] = 4096,
        seed: Annotated[int | None, typer.Option()] = None,
    ) -> None:
        if source_language is not None and source_language not in MANUAL_LANGUAGES:
            raise typer.BadParameter("--source-language must be en, zh, vi, or my")
        if role not in MANUAL_ROLES:
            raise typer.BadParameter(f"--role must be one of: {', '.join(MANUAL_ROLES)}")
        if wrapper_language_mode not in {"english", "same-as-payload"}:
            raise typer.BadParameter("--wrapper-language-mode must be english or same-as-payload")
        resolved_wrapper_mode: Literal["english", "same-as-payload"] = (
            "english" if wrapper_language_mode == "english" else "same-as-payload"
        )
        try:
            batch = load_manual_prompts(
                input_path,
                source_language,
            )
            selected_models = _select_models(_load_models(models_config), models, add_model)
            methods = load_jailbreaks(jailbreaks_config)
            if jailbreak_id not in methods:
                raise ValueError(f"unknown jailbreak: {jailbreak_id}")
            method = methods[jailbreak_id]
            translator = _translator(
                translator_name,
                Path("data/normalized/native_translations.parquet"),
                None,
                languages_config,
            )
            fingerprint = {
                "input_sha256": batch.input_sha256,
                "input_snapshot_sha256": _sha256_bytes(batch.snapshot_jsonl.encode("utf-8")),
                "languages": list(MANUAL_LANGUAGES),
                "languages_config_sha256": _file_sha256(languages_config),
                "translator": {
                    "id": translator.translator_id,
                    "version": translator.version,
                    "decoding_config": translator.decoding_config,
                },
                "jailbreak": jailbreak_id,
                "jailbreak_version": method.version,
                "jailbreak_config_sha256": _file_sha256(jailbreaks_config),
                "persona_catalog_sha256": getattr(method, "catalog_sha256", None),
                "role": role,
                "wrapper_language_mode": wrapper_language_mode,
                "models": {
                    name: model.model_dump(mode="json") for name, model in selected_models.items()
                },
                "generation": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "seed": seed,
                },
            }
            fingerprint_json = json.dumps(
                fingerprint,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            resolved_run_id = run_id or stable_id("manual-run", fingerprint_json)
            run_dir = runs_dir / resolved_run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            with _preparation_lock(run_dir):
                _write_immutable(run_dir / "input_snapshot.jsonl", batch.snapshot_jsonl)
                manifest_path = run_dir / "run_manifest.json"
                existing_manifest = (
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.is_file()
                    else None
                )
                if (
                    existing_manifest is not None
                    and existing_manifest.get("contract") != fingerprint
                ):
                    raise ValueError(f"manual run contract conflict: {resolved_run_id}")

                translations_path = run_dir / "translations.jsonl"
                if translations_path.is_file():
                    translations = _read_jsonl(translations_path, ManualTranslation)
                else:
                    translations = translate_manual_prompts(batch.prompts, translator)
                    _write_immutable(translations_path, _dump_jsonl(translations))
                variants_path = run_dir / "variants.jsonl"
                if variants_path.is_file():
                    variants = _read_jsonl(variants_path, ManualVariant)
                else:
                    variants = build_manual_variants(
                        batch.prompts,
                        translations,
                        method,
                        default_role=role,
                        wrapper_language_mode=resolved_wrapper_mode,
                    )
                    _write_immutable(variants_path, _dump_jsonl(variants))

                catalog_hashes = sorted(
                    {
                        str(metadata["catalog_sha256"])
                        for variant in variants
                        if (metadata := json.loads(variant.attack_metadata_json)).get(
                            "catalog_sha256"
                        )
                    }
                )
                manifest = {
                    "run_id": resolved_run_id,
                    "created_at": (
                        existing_manifest["created_at"]
                        if existing_manifest is not None
                        else _utc_now()
                    ),
                    "contract": fingerprint,
                    "planned_jobs": len(batch.prompts)
                    * len(MANUAL_LANGUAGES)
                    * len(selected_models),
                    "template_sha256s": sorted({variant.template_sha256 for variant in variants}),
                    "catalog_sha256s": catalog_hashes,
                }
                manifest_content = (
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                )
                _write_immutable(manifest_path, manifest_content)

            experiment_id = f"manual-{resolved_run_id}"
            requests = build_manual_generation_requests(
                experiment_id,
                variants,
                selected_models,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
            queue = JobQueue(run_dir / "jobs.sqlite")
            try:
                queue.enqueue(requests)
                queue.retry_failed(experiment_id)
                processed = asyncio.run(
                    _generate_pending(
                        _manual_experiment(
                            experiment_id,
                            selected_models,
                            run_dir,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            seed=seed,
                        ),
                        run_dir,
                        queue,
                    )
                )
            finally:
                queue.close()
            rows = _result_rows(run_dir, requests, variants)
            _atomic_write(run_dir / "results.jsonl", _dump_jsonl(rows))
            _atomic_write(run_dir / "report.md", _report(rows, resolved_run_id))
        except ValueError as error:
            raise typer.BadParameter(str(error)) from None
        typer.echo(
            f"run_id={resolved_run_id} planned_jobs={len(requests)} "
            f"processed_jobs={processed} results={len(rows)}"
        )
