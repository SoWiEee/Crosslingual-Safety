import csv
import json
from pathlib import Path
from typing import Annotated, cast

import pyarrow.parquet as pq
import typer
from dotenv import load_dotenv

from crosslingual_safety.ids import stable_id
from crosslingual_safety.schemas import PromptCase, TranslationReview
from crosslingual_safety.translation.languages import load_languages
from crosslingual_safety.translation.paid_ledger import (
    LedgeredGoogleCloudTranslator,
    PaidCallLedger,
)
from crosslingual_safety.translation.providers import (
    DatasetTranslationProvider,
    DeepTranslatorGoogleTranslator,
    FakeTranslator,
    GoogleCloudAuthenticationError,
    GoogleCloudNMTTranslator,
    GoogleCloudProviderError,
    GoogleCloudRequestTooLargeError,
    GoogleCloudRunBudgetExceededError,
    ManualTranslationProvider,
    NLLBTranslator,
    TranslationInputTooLongError,
    Translator,
)
from crosslingual_safety.translation.service import TranslationService
from crosslingual_safety.translation.storage import TranslationStore, utc_now

REVIEW_FIELDS = (
    "translation_id",
    "case_id",
    "source_language",
    "target_language",
    "source_text",
    "normalized_translated_text",
    "revised_translated_text",
    "reviewer_id",
    "rubric_version",
    "intent_preserved",
    "requested_action_preserved",
    "target_preserved",
    "constraints_preserved",
    "fluency",
    "ambiguity_added",
    "notes",
)


def _native_provider(path: Path) -> DatasetTranslationProvider:
    if not path.is_file():
        raise typer.BadParameter(f"native translations file does not exist: {path}")
    rows = pq.read_table(path).to_pylist()
    return DatasetTranslationProvider(
        {(row["source_text"], row["language"]): row["translated_text"] for row in rows}
    )


def _translator(
    name: str,
    native_translations_path: Path,
    manual_input: Path | None,
    languages_config: Path,
) -> Translator:
    if name == "fake":
        return FakeTranslator()
    if name == "dataset":
        return _native_provider(native_translations_path)
    if name == "manual":
        if manual_input is None:
            raise typer.BadParameter("--manual-input is required for translator=manual")
        return ManualTranslationProvider.from_csv(manual_input)
    if name == "deep-translator-google":
        return DeepTranslatorGoogleTranslator()
    if name == "google-cloud-nmt-v3":
        return GoogleCloudNMTTranslator()
    if name == "nllb":
        return NLLBTranslator(load_languages(languages_config))
    raise typer.BadParameter(
        "--translator must be one of: dataset, manual, deep-translator-google, "
        "google-cloud-nmt-v3, nllb, fake"
    )


def _load_google_dotenv() -> None:
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.is_file():
        try:
            load_dotenv(dotenv_path=dotenv_path)
        except Exception:
            raise GoogleCloudAuthenticationError(
                "Google Cloud application default credentials are unavailable or invalid"
            ) from None


def register_translation_commands(app: typer.Typer) -> None:
    @app.command("translate")
    def translate_command(
        languages: Annotated[str, typer.Option(help="Comma-separated target language codes")],
        translator: Annotated[str, typer.Option()] = "nllb",
        cases_path: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "data/normalized/cases.parquet"
        ),
        native_translations_path: Annotated[
            Path, typer.Option(file_okay=True, dir_okay=False)
        ] = Path("data/normalized/native_translations.parquet"),
        output_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data/translated"),
        languages_config: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)] = Path(
            "configs/languages.yaml"
        ),
        manual_input: Annotated[Path | None, typer.Option(file_okay=True, dir_okay=False)] = None,
        candidate_set: Annotated[str, typer.Option()] = "default",
        force_retranslate: Annotated[bool, typer.Option()] = False,
    ) -> None:
        """Create immutable candidate translations, preferring native dataset records."""
        target_languages = [value.strip() for value in languages.split(",") if value.strip()]
        configured_languages = load_languages(languages_config)
        unknown = sorted(set(target_languages) - configured_languages.keys())
        if unknown:
            raise typer.BadParameter(f"unsupported target languages: {', '.join(unknown)}")
        native_provider = (
            _native_provider(native_translations_path)
            if native_translations_path.is_file()
            else None
        )
        cases = [PromptCase.model_validate(row) for row in pq.read_table(cases_path).to_pylist()]
        store = TranslationStore(output_dir)
        service = TranslationService(store)
        provider = (
            None
            if translator == "google-cloud-nmt-v3"
            else _translator(translator, native_translations_path, manual_input, languages_config)
        )
        created = 0
        preserved_native = 0
        failures: list[dict[str, object]] = []
        try:
            for case in cases:
                for target_language in target_languages:
                    if target_language == case.source_language:
                        continue
                    existing = store.find(case.case_id, target_language)
                    if (
                        any(record.translator_id == "native_dataset" for record in existing)
                        and not force_retranslate
                    ):
                        preserved_native += 1
                        continue
                    selected_provider = provider
                    if (
                        native_provider is not None
                        and (case.canonical_payload, target_language) in native_provider.values
                    ):
                        selected_provider = native_provider
                    elif provider is None:
                        _load_google_dotenv()
                        google_translator = cast(
                            GoogleCloudNMTTranslator,
                            _translator(
                                translator,
                                native_translations_path,
                                manual_input,
                                languages_config,
                            ),
                        )
                        provider = LedgeredGoogleCloudTranslator(
                            google_translator,
                            PaidCallLedger(output_dir / "audit"),
                        )
                        selected_provider = provider
                    assert selected_provider is not None
                    try:
                        if isinstance(selected_provider, LedgeredGoogleCloudTranslator):
                            selected_provider.begin_task(
                                case_id=case.case_id,
                                source_text=case.canonical_payload,
                                source_language=case.source_language,
                                target_language=target_language,
                            )
                        before = len(store.translations())
                        service.translate_case(
                            case,
                            target_language,
                            selected_provider,
                            candidate_set,
                            defer_snapshot=True,
                            force_retranslate=force_retranslate,
                        )
                        created += len(store.translations()) - before
                    except TranslationInputTooLongError as error:
                        failures.append(
                            {
                                "case_id": case.case_id,
                                "error_code": "input_too_long",
                                "max_input_tokens": error.max_input_tokens,
                                "requires_manual_translation": True,
                                "source_language": case.source_language,
                                "target_language": target_language,
                                "token_count": error.token_count,
                                "translator_id": selected_provider.translator_id,
                            }
                        )
                        continue
                    except (
                        GoogleCloudProviderError,
                        GoogleCloudRequestTooLargeError,
                        GoogleCloudRunBudgetExceededError,
                    ) as error:
                        outcome = (
                            selected_provider.current_failure()
                            if isinstance(selected_provider, LedgeredGoogleCloudTranslator)
                            else None
                        )
                        if outcome is not None:
                            error_type = outcome["error_type"]
                            error_message = outcome["error_message"]
                            charged_character_count = outcome["charged_character_count"]
                            audit_reference = outcome["audit_reference"]
                        elif type(error) is GoogleCloudRequestTooLargeError:
                            error_type = "GoogleCloudRequestTooLargeError"
                            error_message = (
                                "Google Cloud Translation request character limit exceeded"
                            )
                            charged_character_count = 0
                            audit_reference = None
                        elif type(error) is GoogleCloudRunBudgetExceededError:
                            error_type = "GoogleCloudRunBudgetExceededError"
                            error_message = "Google Cloud Translation run character budget exceeded"
                            charged_character_count = 0
                            audit_reference = None
                        else:
                            error_type = "GoogleCloudProviderError"
                            error_message = "Google Cloud Translation provider request failed"
                            charged_character_count = 0
                            audit_reference = None
                        failure = {
                            "case_id": case.case_id,
                            "error_code": "google_translation_failed",
                            "error_type": error_type,
                            "error_message": error_message,
                            "charged_character_count": charged_character_count,
                            "source_language": case.source_language,
                            "target_language": target_language,
                            "translator_id": selected_provider.translator_id,
                        }
                        if audit_reference is not None:
                            failure["audit_reference"] = audit_reference
                        failures.append(failure)
                        continue
                    except ValueError as error:
                        if (
                            selected_provider.translator_id == "native_dataset"
                            and "unavailable" in str(error)
                        ):
                            continue
                        raise
        finally:
            store.flush()
        failures_path = output_dir / "translation_failures.jsonl"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.write_text(
            "".join(f"{json.dumps(failure, sort_keys=True)}\n" for failure in failures),
            encoding="utf-8",
        )
        typer.echo(
            f"created {created} translations; preserved {preserved_native} native translations; "
            f"failed {len(failures)}"
        )

    @app.command("export-translation-review")
    def export_translation_review(
        output: Annotated[Path, typer.Option(file_okay=True, dir_okay=False)],
        translations_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data/translated"),
    ) -> None:
        store = TranslationStore(translations_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for record in store.translations():
                row = {field: "" for field in REVIEW_FIELDS}
                row.update(
                    {
                        "translation_id": record.translation_id,
                        "case_id": record.case_id,
                        "source_language": record.source_language,
                        "target_language": record.target_language,
                        "source_text": record.source_text,
                        "normalized_translated_text": record.normalized_translated_text,
                    }
                )
                writer.writerow(row)
        typer.echo(f"exported {len(store.translations())} translations")

    @app.command("import-translation-review")
    def import_translation_review(
        input_path: Annotated[Path, typer.Option("--input", file_okay=True, dir_okay=False)],
        translations_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data/translated"),
    ) -> None:
        with input_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        store = TranslationStore(translations_dir)
        service = TranslationService(store)
        reviews = []
        for row in rows:
            translation_id = row["translation_id"]
            revised_text = row.get("revised_translated_text", "").strip()
            if revised_text:
                parent = store.get(translation_id)
                if parent is None:
                    raise typer.BadParameter(f"unknown translation: {translation_id}")
                translation_id = service.add_human_revision(
                    parent,
                    revised_text,
                    row["reviewer_id"],
                    row["rubric_version"],
                ).translation_id
            review_id = stable_id(
                translation_id,
                row["reviewer_id"],
                row["rubric_version"],
                row["intent_preserved"],
                row["requested_action_preserved"],
                row["target_preserved"],
                row["constraints_preserved"],
                row["fluency"],
                row["ambiguity_added"],
                row.get("notes", ""),
            )
            reviews.append(
                TranslationReview.model_validate(
                    {
                        "review_id": review_id,
                        "translation_id": translation_id,
                        "reviewer_id": row["reviewer_id"],
                        "rubric_version": row["rubric_version"],
                        "intent_preserved": row["intent_preserved"],
                        "requested_action_preserved": row["requested_action_preserved"],
                        "target_preserved": row["target_preserved"],
                        "constraints_preserved": row["constraints_preserved"],
                        "fluency": row["fluency"],
                        "ambiguity_added": row["ambiguity_added"],
                        "notes": row.get("notes") or None,
                        "created_at": utc_now(),
                    }
                )
            )
        store.add_reviews(reviews)
        typer.echo(f"imported {len(reviews)} translation reviews")

    @app.command("validate-translations")
    def validate_translations(
        translations_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data/translated"),
    ) -> None:
        records = TranslationStore(translations_dir).validate_reviews()
        counts = {
            status: sum(record.review_status == status for record in records)
            for status in ("accepted", "rejected", "needs_revision", "pending")
        }
        typer.echo("; ".join(f"{key}={value}" for key, value in counts.items()))

    @app.command("freeze-translations")
    def freeze_translations(
        experiment: Annotated[str, typer.Option()],
        translations_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("data/translated"),
        runs_dir: Annotated[Path, typer.Option(file_okay=False)] = Path("runs"),
    ) -> None:
        manifest = TranslationStore(translations_dir).freeze(experiment, runs_dir / experiment)
        typer.echo(f"froze {len(manifest.translation_hashes)} accepted translations")
