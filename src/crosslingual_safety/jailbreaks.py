import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Annotated, Literal, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml

from crosslingual_safety.ids import stable_id
from crosslingual_safety.schemas import PromptCase, PromptVariant


@dataclass(frozen=True)
class JailbreakContext:
    language: str
    wrapper_language: str
    intent: str
    category: str | None


@dataclass(frozen=True)
class JailbreakResult:
    rendered_prompt: str
    attack_id: str
    attack_family: str
    template_version: str
    template_sha256: str
    wrapper_language: str | None


class JailbreakMethod(Protocol):
    attack_id: str
    attack_family: str
    version: str

    def supports_language(self, language: str) -> bool: ...

    def render(self, payload: str, context: JailbreakContext) -> JailbreakResult: ...


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class IdentityJailbreak:
    def __init__(
        self,
        attack_id: str = "none",
        family: str = "baseline",
        version: str = "1",
    ) -> None:
        self.attack_id = attack_id
        self.attack_family = family
        self.version = version

    def supports_language(self, language: str) -> bool:
        return True

    def render(self, payload: str, context: JailbreakContext) -> JailbreakResult:
        return JailbreakResult(
            rendered_prompt=payload,
            attack_id=self.attack_id,
            attack_family=self.attack_family,
            template_version=self.version,
            template_sha256=_sha256("identity"),
            wrapper_language=None,
        )


class TemplateJailbreak:
    def __init__(
        self,
        attack_id: str,
        family: str,
        version: str,
        wrapper_name: str,
        templates: dict[str, str],
        wrappers: dict[str, str],
    ) -> None:
        self.attack_id = attack_id
        self.attack_family = family
        self.version = version
        self.wrapper_name = wrapper_name
        self.templates = templates
        self.wrappers = wrappers

    def supports_language(self, language: str) -> bool:
        return language in self.templates and language in self.wrappers

    def render(self, payload: str, context: JailbreakContext) -> JailbreakResult:
        if not self.supports_language(context.wrapper_language):
            raise ValueError(f"unsupported wrapper language: {context.wrapper_language}")
        template = self.templates[context.wrapper_language]
        rendered = Template(template).substitute(
            payload=payload,
            **{self.wrapper_name: self.wrappers[context.wrapper_language]},
        )
        return JailbreakResult(
            rendered_prompt=rendered,
            attack_id=self.attack_id,
            attack_family=self.attack_family,
            template_version=self.version,
            template_sha256=_sha256(f"{template}\x1f{self.wrappers[context.wrapper_language]}"),
            wrapper_language=context.wrapper_language,
        )


JAILBREAK_REGISTRY: dict[str, type[IdentityJailbreak] | type[TemplateJailbreak]] = {
    "none": IdentityJailbreak,
    "academic_authority_v1": TemplateJailbreak,
    "roleplay_v1": TemplateJailbreak,
}


def load_jailbreaks(path: Path) -> dict[str, JailbreakMethod]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    wrapper_values = raw.get("wrappers", {})
    methods: dict[str, JailbreakMethod] = {}
    for attack_id, config in raw["jailbreaks"].items():
        registered = JAILBREAK_REGISTRY.get(attack_id)
        if registered is None:
            raise ValueError(f"unregistered jailbreak: {attack_id}")
        declared_class = str(config.get("class", ""))
        if declared_class != registered.__name__:
            raise ValueError(f"jailbreak {attack_id} must declare class {registered.__name__}")
        if registered is IdentityJailbreak:
            methods[attack_id] = IdentityJailbreak(
                attack_id=attack_id,
                family=str(config["family"]),
                version=str(config["version"]),
            )
            continue
        wrapper_name = str(config["wrapper"])
        if wrapper_name not in wrapper_values:
            raise ValueError(f"missing wrapper configuration: {wrapper_name}")
        methods[attack_id] = TemplateJailbreak(
            attack_id=attack_id,
            family=str(config["family"]),
            version=str(config["version"]),
            wrapper_name=wrapper_name,
            templates={str(key): str(value) for key, value in config["templates"].items()},
            wrappers={str(key): str(value) for key, value in wrapper_values[wrapper_name].items()},
        )
    return methods


def _translation_payloads(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    if not path.is_file():
        return {}
    payloads = {}
    for row in pq.read_table(path).to_pylist():
        if row["frozen"] and row["review_status"] == "accepted":
            payloads[(row["case_id"], row["target_language"])] = (
                row["translation_id"],
                row["normalized_translated_text"],
            )
    return payloads


def register_jailbreak_commands(app: typer.Typer) -> None:
    @app.command("build-variants")
    def build_variants(
        languages: Annotated[str, typer.Option(help="Comma-separated payload languages")],
        jailbreak: Annotated[str, typer.Option()],
        wrapper_language_mode: Annotated[
            str,
            typer.Option(help="same-as-payload or english"),
        ] = "same-as-payload",
        cases_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "data/normalized/cases.parquet"
        ),
        translations_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "data/translated/translations.parquet"
        ),
        config_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "configs/jailbreaks.yaml"
        ),
        selection_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "data/normalized/variant_case_selection.parquet"
        ),
        output: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "data/variants/prompt_variants.parquet"
        ),
    ) -> None:
        if wrapper_language_mode not in {"same-as-payload", "english"}:
            raise typer.BadParameter("--wrapper-language-mode must be same-as-payload or english")
        methods = load_jailbreaks(config_path)
        if jailbreak not in methods:
            raise typer.BadParameter(f"unknown jailbreak: {jailbreak}")
        method = methods[jailbreak]
        requested_languages = [value.strip() for value in languages.split(",") if value.strip()]
        cases = [PromptCase.model_validate(row) for row in pq.read_table(cases_path).to_pylist()]
        if selection_path.is_file():
            selected_case_ids = {
                row["selected_case_id"] for row in pq.read_table(selection_path).to_pylist()
            }
            cases = [case for case in cases if case.case_id in selected_case_ids]
        translations = _translation_payloads(translations_path)
        variants: list[PromptVariant] = []
        for case in cases:
            for language in requested_languages:
                if language == case.source_language:
                    translation_id = stable_id(case.case_id, "source", language)
                    payload = case.canonical_payload
                else:
                    translated = translations.get((case.case_id, language))
                    if translated is None:
                        raise typer.BadParameter(
                            f"no frozen accepted translation for {case.case_id} in {language}"
                        )
                    translation_id, payload = translated
                wrapper_language = language if wrapper_language_mode == "same-as-payload" else "en"
                if jailbreak == "none":
                    wrapper_language = language
                context = JailbreakContext(
                    language=language,
                    wrapper_language=wrapper_language,
                    intent=case.intent,
                    category=case.category,
                )
                try:
                    rendered = method.render(payload, context)
                except ValueError as error:
                    raise typer.BadParameter(str(error)) from error
                language_mode: Literal["no_wrapper", "monolingual", "mixed_language"] = (
                    "no_wrapper"
                    if rendered.wrapper_language is None
                    else "monolingual"
                    if rendered.wrapper_language == language
                    else "mixed_language"
                )
                variant_id = stable_id(
                    case.case_id,
                    translation_id,
                    language,
                    payload,
                    rendered.attack_id,
                    rendered.template_version,
                    rendered.template_sha256,
                    rendered.wrapper_language or "",
                )
                variants.append(
                    PromptVariant(
                        variant_id=variant_id,
                        case_id=case.case_id,
                        dataset=case.dataset,
                        translation_id=translation_id,
                        language=language,
                        intent=case.intent,
                        payload=payload,
                        attack_id=rendered.attack_id,
                        attack_family=rendered.attack_family,
                        wrapper_language=rendered.wrapper_language,
                        language_mode=language_mode,
                        rendered_prompt=rendered.rendered_prompt,
                        template_version=rendered.template_version,
                        template_sha256=rendered.template_sha256,
                    )
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        existing = (
            {
                row["variant_id"]: PromptVariant.model_validate(row)
                for row in pq.read_table(output).to_pylist()
            }
            if output.is_file()
            else {}
        )
        existing.update({variant.variant_id: variant for variant in variants})
        rows = [
            variant.model_dump(mode="json")
            for variant in sorted(existing.values(), key=lambda item: item.variant_id)
        ]
        pq.write_table(pa.Table.from_pylist(rows), output)
        typer.echo(f"built {len(variants)} prompt variants; snapshot_rows={len(rows)}")
