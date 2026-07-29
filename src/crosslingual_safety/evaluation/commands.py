"""Typer commands for evaluating and reporting persisted unified runs."""

import os
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import yaml
from dotenv import load_dotenv

from crosslingual_safety.evaluation.models import EvaluationConfig
from crosslingual_safety.evaluation.multilingual import ZooLabMultilingualJudge
from crosslingual_safety.evaluation.service import EvaluationDependencies, evaluate_run
from crosslingual_safety.evaluation.strongreject import (
    LocalStrongRejectJudge,
    TransformersStrongRejectBackend,
)
from crosslingual_safety.generation.commands import RateLimiter, ThrottledProvider
from crosslingual_safety.generation.config import ModelConfig
from crosslingual_safety.generation.providers import (
    FakeProvider,
    OpenAICompatibleChatProvider,
    ProviderAdapter,
)
from crosslingual_safety.reporting import write_hierarchical_reports
from crosslingual_safety.translation.languages import load_languages
from crosslingual_safety.translation.providers import (
    FakeTranslator,
    GoogleCloudNMTTranslator,
    NLLBTranslator,
    Translator,
)
from crosslingual_safety.unified_run import RunSettings, load_run_settings


def load_evaluation_config(path: Path) -> EvaluationConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid evaluation configuration: {path}")
    return EvaluationConfig.model_validate(value)


def _load_model(path: Path, name: str) -> ModelConfig:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise ValueError(f"invalid models configuration: {path}")
    raw = cast(dict[object, object], value["models"])
    if name not in raw:
        raise ValueError(f"unknown evaluation model: {name}")
    return ModelConfig.model_validate(raw[name])


def _multilingual_judge(model: ModelConfig) -> ZooLabMultilingualJudge:
    provider: ProviderAdapter
    if model.provider == "fake":
        if not model.test_only:
            raise ValueError("FakeProvider requires test_only=true")
        provider = FakeProvider(response_text=model.fake_response)
    else:
        if model.endpoint_type != "chat" or not model.base_url_env or not model.api_key_env:
            raise ValueError("multilingual Judge requires chat endpoint metadata")
        base_url = os.environ.get(model.base_url_env)
        api_key = os.environ.get(model.api_key_env)
        if not base_url or not api_key:
            raise ValueError("required multilingual Judge environment variable is unset")
        raw_provider = OpenAICompatibleChatProvider(
            model.provider,
            base_url,
            api_key,
            timeout_seconds=model.timeout_seconds,
        )
        provider = ThrottledProvider(raw_provider, RateLimiter(model.requests_per_minute))
    return ZooLabMultilingualJudge(provider, model.model_id)


def _response_translator(
    config: EvaluationConfig,
    settings: RunSettings,
) -> Translator:
    if config.response_translator == "google-cloud-nmt-v3":
        if not isinstance(settings.google_cloud, dict):
            raise ValueError("Google Cloud Translation configuration must be a mapping")
        google = cast(dict[str, Any], settings.google_cloud)
        return GoogleCloudNMTTranslator(
            project_id=str(google.get("project_id", "")),
            location=str(google.get("location", "global")),
            model=str(google.get("model", "general/nmt")),
            max_request_characters=int(google.get("max_request_characters", 5000)),
            max_run_characters=config.response_translation_max_run_characters,
        )
    if config.response_translator == "nllb":
        return NLLBTranslator(
            load_languages(settings.languages_config),
            checkpoint=settings.nllb_checkpoint,
            local_files_only=settings.nllb_local_files_only,
        )
    return FakeTranslator()


def _dependencies(
    config: EvaluationConfig,
    settings: RunSettings,
) -> EvaluationDependencies:
    model = _load_model(settings.models_config, config.multilingual_judge_model)
    backend = TransformersStrongRejectBackend(config.strongreject)

    def refresh_report(run_dir: Path) -> None:
        write_hierarchical_reports(run_dir)

    return EvaluationDependencies(
        translator=_response_translator(config, settings),
        multilingual_judge=_multilingual_judge(model),
        strongreject_judge=LocalStrongRejectJudge(backend, config.strongreject),
        on_progress=refresh_report,
        multilingual_batch_size=model.concurrency,
    )


def register_evaluation_commands(app: typer.Typer) -> None:
    @app.command("evaluate")
    def evaluate_command(
        run_id: Annotated[str, typer.Option("--run-id")],
        config_path: Annotated[
            Path,
            typer.Option("--config", file_okay=True, dir_okay=False),
        ] = Path("configs/evaluation.yaml"),
    ) -> None:
        try:
            settings = load_run_settings(Path("configs/run.yaml"))
            run_dir = settings.runs_dir / run_id
            if not run_dir.is_dir():
                raise ValueError(f"run does not exist: {run_id}")
            dotenv_path = Path.cwd() / ".env"
            if dotenv_path.is_file():
                load_dotenv(dotenv_path=dotenv_path)
            config = load_evaluation_config(config_path)
            execution = evaluate_run(run_dir, config, _dependencies(config, settings))
        except (OSError, ValueError) as error:
            raise typer.BadParameter(str(error)) from None
        except Exception as error:
            raise typer.BadParameter(f"evaluation failed: {type(error).__name__}") from None
        typer.echo(
            f"run_id={execution.run_id} status={execution.status} "
            f"completed={execution.completed}/{execution.total}"
        )
        typer.echo(f"evaluations={execution.evaluations_path}")
        typer.echo(f"report={run_dir / 'report.md'}")

    @app.command("report")
    def report_command(
        run_id: Annotated[str, typer.Option("--run-id")],
    ) -> None:
        try:
            settings = load_run_settings(Path("configs/run.yaml"))
            run_dir = settings.runs_dir / run_id
            if not run_dir.is_dir():
                raise ValueError(f"run does not exist: {run_id}")
            summary = write_hierarchical_reports(run_dir)
        except (OSError, ValueError) as error:
            raise typer.BadParameter(str(error)) from None
        typer.echo(
            f"run_id={summary.run_id} results={summary.results} evaluated={summary.evaluated}"
        )
        typer.echo(f"report={run_dir / 'report.md'}")


__all__ = ["load_evaluation_config", "register_evaluation_commands"]
