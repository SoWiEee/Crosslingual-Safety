import json
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field

from crosslingual_safety.ids import stable_id
from crosslingual_safety.schemas import GenerationRequest, GenerationStatus, PromptVariant


class ModelConfig(BaseModel):
    provider: str
    base_url_env: str | None = None
    api_key_env: str | None = None
    model_id: str
    endpoint_type: Literal["chat", "completion"]
    concurrency: int = Field(gt=0)
    requests_per_minute: int = Field(gt=0)
    test_only: bool = False
    fake_status: GenerationStatus | None = None
    fake_response: str = "fake response"


class GenerationConfig(BaseModel):
    system_prompt_file: Path | None = None
    temperature: float
    top_p: float | None = None
    max_tokens: int
    seed: int | None = None
    retry_backoff_base: float = 1.0


class ExperimentSection(BaseModel):
    id: str
    datasets: list[str]
    languages: list[str]
    jailbreaks: list[str]
    models: list[str]
    generation: GenerationConfig


class ExperimentPaths(BaseModel):
    variants: Path = Path("data/variants/prompt_variants.parquet")
    runs_dir: Path = Path("runs")


class ExperimentConfig(BaseModel):
    experiment: ExperimentSection
    models: dict[str, ModelConfig]
    paths: ExperimentPaths = ExperimentPaths()


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "models" not in raw:
        models_path = path.with_name("models.yaml")
        if not models_path.is_file():
            raise ValueError(f"models configuration does not exist: {models_path}")
        raw["models"] = yaml.safe_load(models_path.read_text(encoding="utf-8"))["models"]
    base = path.resolve().parent
    paths = raw.setdefault("paths", {})
    for key, default in (
        ("variants", "data/variants/prompt_variants.parquet"),
        ("runs_dir", "runs"),
    ):
        value = Path(paths.get(key, default))
        paths[key] = str(value if value.is_absolute() else (base / value).resolve())
    system_prompt = raw["experiment"]["generation"].get("system_prompt_file")
    if system_prompt is not None:
        value = Path(system_prompt)
        raw["experiment"]["generation"]["system_prompt_file"] = str(
            value if value.is_absolute() else (base / value).resolve()
        )
    return ExperimentConfig.model_validate(raw)


def build_generation_requests(config: ExperimentConfig) -> list[tuple[str, GenerationRequest]]:
    rows = pq.read_table(config.paths.variants).to_pylist()
    variants = [PromptVariant.model_validate(row) for row in rows]
    selected = [
        variant
        for variant in variants
        if variant.dataset in config.experiment.datasets
        and variant.language in config.experiment.languages
        and variant.attack_id in config.experiment.jailbreaks
    ]
    generation = config.experiment.generation
    system_prompt = (
        generation.system_prompt_file.read_text(encoding="utf-8")
        if generation.system_prompt_file is not None
        else None
    )
    requests = []
    for model_name in config.experiment.models:
        model = config.models[model_name]
        config_json = json.dumps(
            {
                "model": model.model_dump(mode="json"),
                "generation": generation.model_dump(mode="json"),
                "system_prompt": system_prompt,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        generation_hash = stable_id(config_json)
        for variant in selected:
            run_id = stable_id(
                config.experiment.id,
                variant.variant_id,
                model_name,
                generation_hash,
            )
            requests.append(
                (
                    model_name,
                    GenerationRequest(
                        run_id=run_id,
                        experiment_id=config.experiment.id,
                        variant_id=variant.variant_id,
                        provider_id=model.provider,
                        requested_model_id=model.model_id,
                        endpoint_type=model.endpoint_type,
                        system_prompt=system_prompt,
                        rendered_prompt=variant.rendered_prompt,
                        temperature=generation.temperature,
                        top_p=generation.top_p,
                        max_tokens=generation.max_tokens,
                        seed=generation.seed,
                        generation_config_hash=generation_hash,
                    ),
                )
            )
    return requests
