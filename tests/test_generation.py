import json
import sqlite3
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.generation.commands import generate_pending
from crosslingual_safety.generation.config import (
    build_generation_requests,
    load_experiment_config,
)
from crosslingual_safety.generation.providers import (
    FakeProvider,
    OpenAICompatibleChatProvider,
    OpenAICompatibleCompletionProvider,
)
from crosslingual_safety.generation.queue import JobQueue
from crosslingual_safety.generation.runner import execute_with_retry
from crosslingual_safety.schemas import GenerationRequest, GenerationResult, PromptVariant

runner = CliRunner()


def test_repository_zoolab_pilot_rate_limits() -> None:
    models = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))["models"]
    victim_models = yaml.safe_load(Path("configs/run.yaml").read_text(encoding="utf-8"))["models"]

    assert victim_models == [
        "llama31_8b",
        "gemma_4_12b",
        "gemma_4_26b",
        "nemotron_cascade_2_30b",
        "llama33_70b",
    ]
    assert {
        name: (models[name]["concurrency"], models[name]["requests_per_minute"])
        for name in victim_models
    } == {
        "llama31_8b": (4, 60),
        "gemma_4_12b": (2, 30),
        "gemma_4_26b": (2, 30),
        "nemotron_cascade_2_30b": (2, 30),
        "llama33_70b": (4, 60),
    }
    assert (
        models["llama_guard_3_8b"]["concurrency"],
        models["llama_guard_3_8b"]["requests_per_minute"],
    ) == (2, 20)
    assert (
        models["nemotron_3_ultra_550b"]["concurrency"],
        models["nemotron_3_ultra_550b"]["requests_per_minute"],
    ) == (1, 10)


def _request() -> GenerationRequest:
    return GenerationRequest(
        run_id="run-1",
        experiment_id="pilot",
        variant_id="variant-1",
        provider_id="fake",
        requested_model_id="fake-model",
        endpoint_type="chat",
        system_prompt=None,
        rendered_prompt="Prompt",
        temperature=0.0,
        top_p=None,
        max_tokens=64,
        seed=42,
        generation_config_hash="config-hash",
    )


def test_retry_keeps_request_unchanged_and_stops_after_success() -> None:
    provider = FakeProvider(outcomes=["rate_limited", "server_error", "success"])

    result, attempts = __import__("asyncio").run(
        execute_with_retry(provider, _request(), max_attempts=4, backoff_base=0)
    )

    assert result.status == "success"
    assert attempts == 3
    assert provider.requests == [_request(), _request(), _request()]


def test_chat_provider_keeps_content_filter_separate_from_model_refusal() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "id": "blocked",
                    "model": "actual-model",
                    "choices": [
                        {
                            "message": {"content": None},
                            "finish_reason": "content_filter",
                        }
                    ],
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "refusal",
                    "model": "actual-model",
                    "choices": [
                        {
                            "message": {"content": "I cannot help with that."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 6},
                },
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    provider = OpenAICompatibleChatProvider(
        provider_id="remote",
        base_url="https://example.invalid/v1",
        api_key="secret-value",
        transport=httpx.MockTransport(handler),
    )

    blocked = __import__("asyncio").run(provider.generate(_request()))
    refusal = __import__("asyncio").run(provider.generate(_request()))

    assert blocked.status == "provider_blocked"
    assert blocked.response_text is None
    assert refusal.status == "success"
    assert refusal.response_text == "I cannot help with that."
    assert "secret-value" not in json.dumps(refusal.model_dump(mode="json"))


def test_completion_provider_uses_completion_shape() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "model": "actual-model",
                "choices": [{"text": "Completed", "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleCompletionProvider(
        provider_id="remote",
        base_url="https://example.invalid/v1",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    request = _request().model_copy(
        update={"endpoint_type": "completion", "system_prompt": "System"}
    )

    result = __import__("asyncio").run(provider.generate(request))

    assert result.status == "success"
    assert result.response_text == "Completed"
    assert captured["prompt"] == "System\n\nPrompt"
    assert "messages" not in captured


def test_provider_redacts_api_key_if_response_echoes_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "actual-model",
                "choices": [
                    {
                        "message": {"content": "echo secret-value"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider = OpenAICompatibleChatProvider(
        provider_id="remote",
        base_url="https://example.invalid/v1",
        api_key="secret-value",
        transport=httpx.MockTransport(handler),
    )

    result = __import__("asyncio").run(provider.generate(_request()))

    assert result.response_text == "echo [REDACTED]"
    assert result.raw_response_json is not None
    assert "secret-value" not in result.raw_response_json


def test_chat_provider_applies_model_timeout() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={
                "model": "actual-model",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            },
        )

    provider = OpenAICompatibleChatProvider(
        provider_id="remote",
        base_url="https://example.invalid/v1",
        api_key="secret",
        timeout_seconds=180,
        transport=httpx.MockTransport(handler),
    )

    result = __import__("asyncio").run(provider.generate(_request()))

    assert result.status == "success"
    assert captured == {"connect": 180, "read": 180, "write": 180, "pool": 180}


def _write_experiment(
    tmp_path: Path,
    fake_status: str = "success",
    *,
    variant_count: int = 1,
) -> Path:
    variants_path = tmp_path / "variants.parquet"
    variants = [
        PromptVariant(
            variant_id=f"variant-{index}",
            case_id=f"case-{index}",
            dataset="test",
            translation_id=f"source-{index}",
            language="en",
            intent="harmful",
            payload="Prompt",
            attack_id="none",
            attack_family="baseline",
            wrapper_language=None,
            language_mode="no_wrapper",
            rendered_prompt="Prompt",
            template_version="1",
            template_sha256="template-hash",
        )
        for index in range(variant_count)
    ]
    pq.write_table(
        pa.Table.from_pylist([variant.model_dump(mode="json") for variant in variants]),
        variants_path,
    )
    config = {
        "experiment": {
            "id": "pilot",
            "datasets": ["test"],
            "languages": ["en"],
            "jailbreaks": ["none"],
            "models": ["fake_model"],
            "generation": {
                "system_prompt_file": None,
                "temperature": 0.0,
                "top_p": None,
                "max_tokens": 64,
                "seed": 42,
                "retry_backoff_base": 0,
            },
        },
        "models": {
            "fake_model": {
                "provider": "fake",
                "model_id": "fake-model",
                "endpoint_type": "chat",
                "concurrency": 2,
                "requests_per_minute": 6000,
                "test_only": True,
                "fake_status": fake_status,
                "fake_response": "offline response",
            }
        },
        "paths": {
            "variants": str(variants_path),
            "runs_dir": str(tmp_path / "runs"),
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_plan_enqueue_generate_is_resumable_and_counts_match(tmp_path: Path) -> None:
    config_path = _write_experiment(tmp_path)
    runs_dir = tmp_path / "runs"

    planned = runner.invoke(app, ["plan", "--config", str(config_path)])
    enqueued = runner.invoke(app, ["enqueue", "--config", str(config_path)])
    generated = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )
    (runs_dir / "pilot" / "generation_results.parquet").write_bytes(b"corrupt")
    resumed = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )
    status = runner.invoke(
        app,
        ["generation-status", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )

    assert all(result.exit_code == 0 for result in (planned, enqueued, generated, resumed, status))
    assert "planned_jobs=1" in planned.output
    assert "enqueued_jobs=1" in enqueued.output
    assert "success=1" in status.output
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        row = connection.execute("SELECT status, attempts FROM jobs").fetchone()
    assert row == ("success", 1)
    results = pq.read_table(runs_dir / "pilot" / "generation_results.parquet").to_pylist()
    assert len(results) == 1
    assert results[0]["response_text"] == "offline response"


def test_generate_pending_calls_final_result_callback_after_persistence(tmp_path: Path) -> None:
    config = load_experiment_config(_write_experiment(tmp_path))
    run_dir = tmp_path / "callback-run"
    run_dir.mkdir()
    queue = JobQueue(run_dir / "jobs.sqlite")
    requests = build_generation_requests(config)
    queue.enqueue(requests)
    observations: list[tuple[str, str, bool, dict[str, int]]] = []

    def observe(model_name: str, request: GenerationRequest, result: object) -> None:
        attempt = run_dir / "generation_attempts" / request.run_id / "attempt-0001.json"
        observations.append(
            (
                model_name,
                getattr(result, "status"),
                attempt.is_file(),
                queue.status_counts(config.experiment.id),
            )
        )

    try:
        __import__("asyncio").run(generate_pending(config, run_dir, queue, on_final_result=observe))
    finally:
        queue.close()

    assert observations == [("fake_model", "success", True, {"success": 1})]


def test_generate_pending_claims_only_jobs_with_execution_slots(tmp_path: Path) -> None:
    asyncio = __import__("asyncio")
    config = load_experiment_config(_write_experiment(tmp_path, variant_count=5))
    run_dir = tmp_path / "bounded-claims"
    run_dir.mkdir()
    queue = JobQueue(run_dir / "jobs.sqlite")
    queue.enqueue(build_generation_requests(config))
    release = asyncio.Event()
    two_started = asyncio.Event()

    class BlockingProvider:
        provider_id = "fake"

        def __init__(self) -> None:
            self.started = 0

        async def generate(self, request: GenerationRequest) -> GenerationResult:
            self.started += 1
            if self.started == 2:
                two_started.set()
            await release.wait()
            return await FakeProvider().generate(request)

    provider = BlockingProvider()

    async def observe_claims() -> None:
        task = asyncio.create_task(
            generate_pending(config, run_dir, queue, provider_factory=lambda _: provider)
        )
        await asyncio.wait_for(two_started.wait(), timeout=1)
        assert queue.status_counts(config.experiment.id) == {"pending": 3, "running": 2}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(observe_claims())
    finally:
        queue.close()


def test_retry_failed_requeues_only_retryable_jobs(tmp_path: Path) -> None:
    config_path = _write_experiment(tmp_path, "rate_limited")
    runs_dir = tmp_path / "runs"
    assert runner.invoke(app, ["enqueue", "--config", str(config_path)]).exit_code == 0
    generated = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )
    retried = runner.invoke(
        app,
        [
            "retry-failed",
            "--experiment",
            "pilot",
            "--only",
            "retryable",
            "--runs-dir",
            str(runs_dir),
        ],
    )

    assert generated.exit_code == retried.exit_code == 0
    assert "retried_jobs=1" in retried.output
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        row = connection.execute("SELECT status, attempts FROM jobs").fetchone()
    assert row == ("pending", 4)
    generated_again = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )
    assert generated_again.exit_code == 0
    attempt_dir = next((runs_dir / "pilot" / "generation_attempts").iterdir())
    raw_dir = next((runs_dir / "pilot" / "raw_responses").iterdir())
    assert len(list(attempt_dir.glob("attempt-*.json"))) == 8
    assert len(list(raw_dir.glob("attempt-*.json"))) == 8
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        retried_row = connection.execute("SELECT status, attempts FROM jobs").fetchone()
    assert retried_row == ("retryable_error", 8)


def test_stale_running_job_is_recovered_on_generate(tmp_path: Path) -> None:
    config_path = _write_experiment(tmp_path)
    runs_dir = tmp_path / "runs"
    assert runner.invoke(app, ["enqueue", "--config", str(config_path)]).exit_code == 0
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        connection.execute(
            "UPDATE jobs SET status = 'running', claimed_at = '2000-01-01T00:00:00Z'"
        )
        connection.commit()

    generated = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )

    assert generated.exit_code == 0, generated.output
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        row = connection.execute("SELECT status, attempts FROM jobs").fetchone()
    assert row == ("success", 1)


def test_final_journal_reconciles_crash_before_sqlite_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_experiment(tmp_path)
    runs_dir = tmp_path / "runs"
    assert runner.invoke(app, ["enqueue", "--config", str(config_path)]).exit_code == 0
    original_complete = JobQueue.complete

    def crash_before_complete(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(JobQueue, "complete", crash_before_complete)
    crashed = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )
    monkeypatch.setattr(JobQueue, "complete", original_complete)
    resumed = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )

    assert crashed.exit_code != 0
    assert resumed.exit_code == 0
    assert "processed_jobs=0" in resumed.output
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        row = connection.execute("SELECT status, attempts FROM jobs").fetchone()
    assert row == ("success", 1)


def test_orphaned_raw_attempt_is_never_overwritten(tmp_path: Path) -> None:
    config_path = _write_experiment(tmp_path)
    runs_dir = tmp_path / "runs"
    assert runner.invoke(app, ["enqueue", "--config", str(config_path)]).exit_code == 0
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        run_id = connection.execute("SELECT run_id FROM jobs").fetchone()[0]
    raw_dir = runs_dir / "pilot" / "raw_responses" / run_id
    raw_dir.mkdir(parents=True)
    orphan = raw_dir / "attempt-0001.json"
    orphan.write_text("orphaned raw response", encoding="utf-8")

    generated = runner.invoke(
        app,
        ["generate", "--experiment", "pilot", "--runs-dir", str(runs_dir)],
    )

    assert generated.exit_code == 0, generated.output
    assert orphan.read_text(encoding="utf-8") == "orphaned raw response"
    assert (raw_dir / "attempt-0002.json").is_file()
    assert (runs_dir / "pilot" / "generation_attempts" / run_id / "attempt-0002.json").is_file()


def test_complete_is_compare_and_set_after_reconciliation(tmp_path: Path) -> None:
    config_path = _write_experiment(tmp_path)
    runs_dir = tmp_path / "runs"
    assert runner.invoke(app, ["enqueue", "--config", str(config_path)]).exit_code == 0
    queue = JobQueue(runs_dir / "pilot" / "jobs.sqlite")
    try:
        run_id = queue.pending("pilot")[0][0]
        assert queue.claim(run_id)
        assert queue.reconcile(run_id, "success", 1, None, None)
        assert not queue.complete(run_id, "success", 1, 0, None, None)
        assert queue.status_counts("pilot") == {"success": 1}
    finally:
        queue.close()
    with sqlite3.connect(runs_dir / "pilot" / "jobs.sqlite") as connection:
        attempts = connection.execute("SELECT attempts FROM jobs").fetchone()[0]
    assert attempts == 1
