import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from crosslingual_safety.generation.config import ModelConfig
from crosslingual_safety.ids import canonicalize_text, stable_id
from crosslingual_safety.jailbreaks import JailbreakContext, JailbreakMethod, PaperSummaryJailbreak
from crosslingual_safety.psa_summary import SummaryArtifact
from crosslingual_safety.schemas import GenerationRequest
from crosslingual_safety.translation.providers import Translator

ManualLanguage = Literal["en", "zh", "vi", "my"]
ManualRole = Literal["joker", "lex_luthor", "riddler", "scarecrow"]


class ManualPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    prompt: str
    source_language: ManualLanguage
    role: ManualRole | None = None
    category: str | None = None
    system_prompt: str | None = None

    @field_validator("prompt_id", "prompt")
    @classmethod
    def nonempty(cls, value: str) -> str:
        normalized = canonicalize_text(value)
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("category", "system_prompt")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = canonicalize_text(value)
        return normalized or None


@dataclass(frozen=True)
class ManualInputBatch:
    prompts: tuple[ManualPrompt, ...]
    input_sha256: str
    snapshot_jsonl: str


class ManualTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translation_id: str
    prompt_id: str
    source_language: ManualLanguage
    language: ManualLanguage
    source_text: str
    translated_text: str
    translator_id: str
    translator_version: str
    decoding_config: dict[str, object]
    provider_request_id: str | None


class ManualVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str
    prompt_id: str
    translation_id: str
    language: ManualLanguage
    role: ManualRole | None
    payload: str
    system_prompt: str | None
    attack_id: str
    attack_family: str
    wrapper_language: str | None
    language_mode: Literal["no_wrapper", "monolingual", "mixed_language"]
    rendered_prompt: str
    template_version: str
    template_sha256: str
    attack_metadata_json: str


def _snapshot(prompts: tuple[ManualPrompt, ...]) -> str:
    return "".join(
        json.dumps(
            prompt.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for prompt in prompts
    )


def load_manual_prompts(
    path: Path,
    source_language: ManualLanguage | None = None,
) -> ManualInputBatch:
    raw = path.read_bytes()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"manual input must be UTF-8: {path}") from error

    prompts: tuple[ManualPrompt, ...]
    if path.suffix.lower() == ".txt":
        if source_language is None:
            raise ValueError("source_language is required for .txt input")
        prompt_text = canonicalize_text(text)
        prompt = ManualPrompt(
            prompt_id=stable_id("manual-prompt", prompt_text),
            prompt=prompt_text,
            source_language=source_language,
        )
        prompts = (prompt,)
    elif path.suffix.lower() == ".jsonl":
        parsed: list[ManualPrompt] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            try:
                parsed.append(ManualPrompt.model_validate(value))
            except ValueError as error:
                raise ValueError(f"invalid manual prompt on line {line_number}: {error}") from error
        if not parsed:
            raise ValueError("manual JSONL input contains no prompts")
        prompt_ids = [prompt.prompt_id for prompt in parsed]
        duplicates = sorted(
            prompt_id for prompt_id in set(prompt_ids) if prompt_ids.count(prompt_id) > 1
        )
        if duplicates:
            raise ValueError(f"duplicate prompt_id: {', '.join(duplicates)}")
        prompts = tuple(parsed)
    else:
        raise ValueError("manual input must use .txt or .jsonl")

    return ManualInputBatch(
        prompts=prompts,
        input_sha256=input_sha256,
        snapshot_jsonl=_snapshot(prompts),
    )


def translate_manual_prompts(
    prompts: tuple[ManualPrompt, ...],
    translator: Translator,
    languages: tuple[ManualLanguage, ...] = ("en", "zh", "vi", "my"),
) -> list[ManualTranslation]:
    records: list[ManualTranslation] = []
    for prompt in prompts:
        for language in languages:
            if language == prompt.source_language:
                translated_text = prompt.prompt
                translator_id = "source"
                translator_version = "1"
                decoding_config: dict[str, object] = {}
                provider_request_id = None
            else:
                if not translator.supports(prompt.source_language, language):
                    raise ValueError(
                        f"translator {translator.translator_id} does not support "
                        f"{prompt.source_language}->{language}"
                    )
                output = translator.translate(
                    prompt.prompt,
                    prompt.source_language,
                    language,
                )
                translated_text = canonicalize_text(output.text)
                translator_id = translator.translator_id
                translator_version = translator.version
                decoding_config = translator.decoding_config
                provider_request_id = output.provider_request_id
            translation_id = stable_id(
                prompt.prompt_id,
                prompt.source_language,
                language,
                translated_text,
                translator_id,
                translator_version,
                json.dumps(decoding_config, sort_keys=True, separators=(",", ":")),
            )
            records.append(
                ManualTranslation(
                    translation_id=translation_id,
                    prompt_id=prompt.prompt_id,
                    source_language=prompt.source_language,
                    language=language,
                    source_text=prompt.prompt,
                    translated_text=translated_text,
                    translator_id=translator_id,
                    translator_version=translator_version,
                    decoding_config=decoding_config,
                    provider_request_id=provider_request_id,
                )
            )
    return records


def build_manual_variants(
    prompts: tuple[ManualPrompt, ...],
    translations: list[ManualTranslation],
    jailbreak: JailbreakMethod,
    *,
    default_role: ManualRole = "joker",
    wrapper_language_mode: Literal["english", "same-as-payload"] = "english",
    summary_artifacts: Mapping[str, SummaryArtifact] | None = None,
) -> list[ManualVariant]:
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    variants: list[ManualVariant] = []
    for translation in translations:
        prompt = prompts_by_id[translation.prompt_id]
        role = prompt.role or default_role if jailbreak.attack_id == "gra_v1" else None
        wrapper_language = "en" if wrapper_language_mode == "english" else translation.language
        if jailbreak.attack_id == "none":
            wrapper_language = translation.language
        context = JailbreakContext(
            language=translation.language,
            wrapper_language=wrapper_language,
            intent="harmful",
            category=prompt.category,
            role=role,
        )
        if isinstance(jailbreak, PaperSummaryJailbreak) and summary_artifacts is not None:
            try:
                summary_artifact = summary_artifacts[wrapper_language]
            except KeyError as error:
                raise ValueError(
                    f"missing dynamic PSA summary artifact for {wrapper_language}"
                ) from error
            from crosslingual_safety.psa_summary import artifact_sections

            rendered = jailbreak.render(
                translation.translated_text,
                context,
                summary_sections=artifact_sections(summary_artifact),
                summary_artifact=summary_artifact,
            )
        else:
            rendered = jailbreak.render(translation.translated_text, context)
        language_mode: Literal["no_wrapper", "monolingual", "mixed_language"] = (
            "no_wrapper"
            if rendered.wrapper_language is None
            else "monolingual"
            if rendered.wrapper_language == translation.language
            else "mixed_language"
        )
        variant_id = stable_id(
            prompt.prompt_id,
            translation.translation_id,
            rendered.attack_id,
            rendered.template_version,
            rendered.template_sha256,
            rendered.metadata_json,
            rendered.wrapper_language or "",
        )
        variants.append(
            ManualVariant(
                variant_id=variant_id,
                prompt_id=prompt.prompt_id,
                translation_id=translation.translation_id,
                language=translation.language,
                role=role,
                payload=translation.translated_text,
                system_prompt=prompt.system_prompt,
                attack_id=rendered.attack_id,
                attack_family=rendered.attack_family,
                wrapper_language=rendered.wrapper_language,
                language_mode=language_mode,
                rendered_prompt=rendered.rendered_prompt,
                template_version=rendered.template_version,
                template_sha256=rendered.template_sha256,
                attack_metadata_json=rendered.metadata_json,
            )
        )
    return variants


def build_manual_generation_requests(
    experiment_id: str,
    variants: list[ManualVariant],
    models: Mapping[str, ModelConfig],
    *,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    seed: int | None,
) -> list[tuple[str, GenerationRequest]]:
    requests: list[tuple[str, GenerationRequest]] = []
    for model_name, raw_model in models.items():
        model = ModelConfig.model_validate(raw_model)
        generation_config = {
            "model": model.model_dump(mode="json"),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        generation_config_hash = stable_id(
            json.dumps(generation_config, sort_keys=True, separators=(",", ":"))
        )
        for variant in variants:
            run_id = stable_id(
                experiment_id,
                variant.variant_id,
                model_name,
                generation_config_hash,
            )
            requests.append(
                (
                    model_name,
                    GenerationRequest(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        variant_id=variant.variant_id,
                        provider_id=model.provider,
                        requested_model_id=model.model_id,
                        endpoint_type=model.endpoint_type,
                        system_prompt=variant.system_prompt,
                        rendered_prompt=variant.rendered_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=seed,
                        generation_config_hash=generation_config_hash,
                    ),
                )
            )
    return requests
