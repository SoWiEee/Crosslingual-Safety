import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml
from dotenv import load_dotenv

from crosslingual_safety.generation.config import (
    ExperimentConfig,
    ModelConfig,
    build_generation_requests,
    load_experiment_config,
)
from crosslingual_safety.generation.providers import (
    AuthenticationError,
    FakeProvider,
    OpenAICompatibleChatProvider,
    OpenAICompatibleCompletionProvider,
    ProviderAdapter,
)
from crosslingual_safety.generation.queue import JobQueue
from crosslingual_safety.generation.runner import execute_with_retry
from crosslingual_safety.schemas import GenerationRequest, GenerationResult


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.next_request_at = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_request_at - now)
            if delay:
                await asyncio.sleep(delay)
            self.next_request_at = time.monotonic() + self.interval


class ThrottledProvider:
    def __init__(self, provider: ProviderAdapter, limiter: RateLimiter) -> None:
        self.provider_id = provider.provider_id
        self.provider = provider
        self.limiter = limiter

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        await self.limiter.wait()
        return await self.provider.generate(request)


@dataclass
class ProviderRuntime:
    semaphore: asyncio.Semaphore
    limiter: RateLimiter


def _run_dir(runs_dir: Path, experiment: str) -> Path:
    return runs_dir / experiment


def _load_run_config(run_dir: Path) -> ExperimentConfig:
    path = run_dir / "experiment.yaml"
    if not path.is_file():
        raise typer.BadParameter(f"experiment config does not exist: {path}")
    return load_experiment_config(path)


def _provider(model: ModelConfig) -> ProviderAdapter:
    if model.provider == "fake":
        if not model.test_only:
            raise ValueError("FakeProvider requires test_only=true")
        status = model.fake_status or "success"
        return FakeProvider(outcomes=[status] * 4, response_text=model.fake_response)
    if model.base_url_env is None or model.api_key_env is None:
        raise ValueError(f"provider {model.provider} requires base_url_env and api_key_env")
    load_dotenv()
    try:
        base_url = os.environ[model.base_url_env]
        api_key = os.environ[model.api_key_env]
    except KeyError as error:
        raise ValueError(
            f"required provider environment variable is unset: {error.args[0]}"
        ) from None
    provider_type = (
        OpenAICompatibleChatProvider
        if model.endpoint_type == "chat"
        else OpenAICompatibleCompletionProvider
    )
    return provider_type(model.provider, base_url, api_key)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_attempt(
    run_dir: Path,
    result: GenerationResult,
    *,
    local_attempt: int,
    attempts_before: int,
    will_retry: bool,
) -> GenerationResult:
    attempt_dir = run_dir / "generation_attempts" / result.run_id
    raw_dir = run_dir / "raw_responses" / result.run_id
    existing = [
        int(path.stem.split("-")[-1])
        for directory in (attempt_dir, raw_dir)
        for path in directory.glob("attempt-*.json")
        if path.stem.split("-")[-1].isdigit()
    ]
    sequence = max(existing, default=0) + 1
    raw_path = raw_dir / f"attempt-{sequence:04d}.json"
    attempt_path = attempt_dir / f"attempt-{sequence:04d}.json"
    if raw_path.exists() or attempt_path.exists():
        raise RuntimeError(f"attempt sequence already exists: {result.run_id}/{sequence}")
    stored = result.model_copy(
        update={"raw_response_path": str(raw_path), "raw_response_json": None}
    )
    _atomic_write_text(
        raw_path,
        result.raw_response_json
        or json.dumps(stored.model_dump(mode="json"), indent=2, sort_keys=True),
    )
    attempt_record = {
        "attempt_sequence": sequence,
        "queue_attempt_number": attempts_before + local_attempt,
        "will_retry": will_retry,
        "result": stored.model_dump(mode="json"),
    }
    _atomic_write_text(
        attempt_path,
        json.dumps(attempt_record, indent=2, sort_keys=True),
    )
    if not will_retry:
        projection = {
            "total_attempts": attempts_before + local_attempt,
            "result": stored.model_dump(mode="json"),
        }
        _atomic_write_text(
            run_dir / "generation_results" / f"{stored.run_id}.json",
            json.dumps(projection, indent=2, sort_keys=True),
        )
    return stored


def _final_projections(run_dir: Path) -> list[tuple[int, GenerationResult]]:
    journal_dir = run_dir / "generation_results"
    if not journal_dir.is_dir():
        return []
    projections = []
    for path in journal_dir.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        projections.append(
            (
                int(value["total_attempts"]),
                GenerationResult.model_validate(value["result"]),
            )
        )
    return projections


def _reconcile_results(run_dir: Path, queue: JobQueue) -> None:
    for total_attempts, result in _final_projections(run_dir):
        queue.reconcile(
            result.run_id,
            result.status,
            total_attempts,
            result.error_type,
            result.error_message,
        )


def _flush_results(run_dir: Path) -> None:
    results_path = run_dir / "generation_results.parquet"
    projections = {result.run_id: result for _, result in _final_projections(run_dir)}
    if not projections:
        return
    rows = [
        item.model_dump(mode="json")
        for item in sorted(projections.values(), key=lambda value: value.run_id)
    ]
    temporary = results_path.with_suffix(f"{results_path.suffix}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(results_path)


async def _generate_pending(config: ExperimentConfig, run_dir: Path, queue: JobQueue) -> int:
    _reconcile_results(run_dir, queue)
    queue.reset_stale()
    jobs = queue.pending(config.experiment.id)
    if not jobs:
        _flush_results(run_dir)
        return 0
    provider_limits: dict[str, tuple[int, int]] = {}
    for model_name in config.experiment.models:
        model = config.models[model_name]
        current = provider_limits.get(model.provider)
        limits = (model.concurrency, model.requests_per_minute)
        provider_limits[model.provider] = (
            limits if current is None else (min(current[0], limits[0]), min(current[1], limits[1]))
        )
    runtimes = {
        provider_id: ProviderRuntime(
            semaphore=asyncio.Semaphore(limits[0]),
            limiter=RateLimiter(limits[1]),
        )
        for provider_id, limits in provider_limits.items()
    }
    providers = {
        model_name: _provider(config.models[model_name]) for model_name in config.experiment.models
    }
    persistence_lock = asyncio.Lock()

    async def execute(
        run_id: str,
        model_name: str,
        attempts_before: int,
        request: GenerationRequest,
    ) -> bool:
        if not queue.claim(run_id):
            return False
        provider = providers[model_name]
        runtime = runtimes[provider.provider_id]
        throttled = ThrottledProvider(provider, runtime.limiter)

        async def persist_attempt(
            result: GenerationResult,
            local_attempt: int,
            will_retry: bool,
        ) -> None:
            async with persistence_lock:
                _write_attempt(
                    run_dir,
                    result,
                    local_attempt=local_attempt,
                    attempts_before=attempts_before,
                    will_retry=will_retry,
                )

        async with runtime.semaphore:
            result, attempts = await execute_with_retry(
                throttled,
                request,
                backoff_base=config.experiment.generation.retry_backoff_base,
                on_attempt=persist_attempt,
            )
        async with persistence_lock:
            queue.complete(
                run_id,
                result.status,
                attempts_before + attempts,
                attempts_before,
                result.error_type,
                result.error_message,
            )
        return True

    try:
        completed = await asyncio.gather(
            *(
                execute(run_id, model_name, attempts_before, request)
                for run_id, model_name, attempts_before, request in jobs
            )
        )
    except AuthenticationError as error:
        raise typer.BadParameter(str(error)) from None
    _flush_results(run_dir)
    return sum(completed)


def register_generation_commands(app: typer.Typer) -> None:
    @app.command("plan")
    def plan(
        config_path: Annotated[Path, typer.Option("--config", file_okay=True, dir_okay=False)],
    ) -> None:
        config = load_experiment_config(config_path)
        requests = build_generation_requests(config)
        estimated_tokens = sum(
            max(1, len(request.rendered_prompt.split())) for _, request in requests
        )
        typer.echo(
            f"planned_jobs={len(requests)} estimated_prompt_tokens={estimated_tokens} "
            f"models={len(config.experiment.models)}"
        )

    @app.command("enqueue")
    def enqueue(
        config_path: Annotated[Path, typer.Option("--config", file_okay=True, dir_okay=False)],
    ) -> None:
        config = load_experiment_config(config_path)
        run_dir = _run_dir(config.paths.runs_dir, config.experiment.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "experiment.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        queue = JobQueue(run_dir / "jobs.sqlite")
        try:
            created = queue.enqueue(build_generation_requests(config))
        finally:
            queue.close()
        typer.echo(f"enqueued_jobs={created}")

    @app.command("generate")
    def generate(
        experiment: Annotated[str, typer.Option()],
        runs_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("runs"),
    ) -> None:
        run_dir = _run_dir(runs_dir, experiment)
        config = _load_run_config(run_dir)
        queue = JobQueue(run_dir / "jobs.sqlite")
        try:
            completed = asyncio.run(_generate_pending(config, run_dir, queue))
        finally:
            queue.close()
        typer.echo(f"processed_jobs={completed}")

    @app.command("generation-status")
    def generation_status(
        experiment: Annotated[str, typer.Option()],
        runs_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("runs"),
    ) -> None:
        queue = JobQueue(_run_dir(runs_dir, experiment) / "jobs.sqlite")
        try:
            counts = queue.status_counts(experiment)
        finally:
            queue.close()
        typer.echo(" ".join(f"{key}={value}" for key, value in sorted(counts.items())))

    @app.command("retry-failed")
    def retry_failed(
        experiment: Annotated[str, typer.Option()],
        only: Annotated[str, typer.Option()] = "retryable",
        runs_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("runs"),
    ) -> None:
        if only != "retryable":
            raise typer.BadParameter("--only currently supports retryable")
        queue = JobQueue(_run_dir(runs_dir, experiment) / "jobs.sqlite")
        try:
            retried = queue.retry_failed(experiment)
        finally:
            queue.close()
        typer.echo(f"retried_jobs={retried}")
