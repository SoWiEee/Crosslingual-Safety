import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.schemas import PromptCase, TranslationReview
from crosslingual_safety.translation.commands import REVIEW_FIELDS
from crosslingual_safety.translation.providers import FakeTranslator
from crosslingual_safety.translation.service import TranslationService
from crosslingual_safety.translation.storage import TranslationStore

runner = CliRunner()


def _case(case_id: str = "case-1") -> PromptCase:
    return PromptCase(
        case_id=case_id,
        content_id=f"content-{case_id}",
        dataset="test",
        intent="harmful",
        category=None,
        source_language="en",
        source_text="Original",
        behavior_description=None,
        success_criteria=None,
        context_text=None,
        canonical_payload="Original",
        payload_format="direct_prompt_v1",
    )


def _accepted_review(translation_id: str) -> TranslationReview:
    return TranslationReview(
        review_id=f"review-{translation_id}",
        translation_id=translation_id,
        reviewer_id="reviewer-a",
        rubric_version="v1",
        intent_preserved="yes",
        requested_action_preserved="yes",
        target_preserved="yes",
        constraints_preserved="yes",
        fluency="good",
        ambiguity_added="no",
        notes=None,
        created_at="2026-01-01T00:00:00Z",
    )


def test_translation_cache_never_calls_provider_twice(tmp_path: Path) -> None:
    store = TranslationStore(tmp_path / "translated")
    translator = FakeTranslator(outputs={("Original", "zh"): "翻譯"})
    service = TranslationService(store)

    first = service.translate_case(_case(), "zh", translator)
    second = service.translate_case(_case(), "zh", translator)

    assert first.translation_id == second.translation_id
    assert translator.call_count == 1
    assert pq.read_table(store.translations_path).num_rows == 1


def test_force_retranslate_preserves_original_and_calls_provider(tmp_path: Path) -> None:
    store = TranslationStore(tmp_path / "translated")
    translator = FakeTranslator(outputs={("Original", "zh"): "翻譯"})
    service = TranslationService(store)

    first = service.translate_case(_case(), "zh", translator)
    second = service.translate_case(
        _case(),
        "zh",
        translator,
        force_retranslate=True,
    )

    assert first.translation_id != second.translation_id
    assert second.revision_id is not None
    assert translator.call_count == 2
    assert len(store.translations()) == 2


def test_provider_cache_is_shared_by_cases_with_identical_payload(tmp_path: Path) -> None:
    store = TranslationStore(tmp_path / "translated")
    translator = FakeTranslator(outputs={("Original", "zh"): "翻譯"})
    service = TranslationService(store)

    first = service.translate_case(_case("case-1"), "zh", translator)
    second = service.translate_case(_case("case-2"), "zh", translator)

    assert first.translation_id != second.translation_id
    assert first.provider_cache_key == second.provider_cache_key
    assert translator.call_count == 1
    assert len(store.translations()) == 2


def test_frozen_translation_never_calls_provider_again(tmp_path: Path) -> None:
    store = TranslationStore(tmp_path / "translated")
    translator = FakeTranslator(outputs={("Original", "zh"): "翻譯"})
    service = TranslationService(store)
    record = service.translate_case(_case(), "zh", translator)
    store.add_reviews([_accepted_review(record.translation_id)])
    store.freeze("pilot", tmp_path / "runs" / "pilot")

    cached = service.translate_case(_case(), "zh", translator)

    assert cached.frozen
    assert translator.call_count == 1
    with pytest.raises(ValueError, match="cannot translate frozen"):
        service.translate_case(_case(), "zh", translator, force_retranslate=True)
    assert translator.call_count == 1

    other_translator = FakeTranslator()
    other_translator.translator_id = "other"
    with pytest.raises(ValueError, match="cannot translate frozen"):
        service.translate_case(_case(), "zh", other_translator)
    assert other_translator.call_count == 0


def test_review_gate_and_freeze_manifest(tmp_path: Path) -> None:
    store = TranslationStore(tmp_path / "translated")
    translator = FakeTranslator(outputs={("Original", "my"): "ဘာသာပြန်"})
    record = TranslationService(store).translate_case(_case(), "my", translator)

    store.add_reviews([_accepted_review(record.translation_id)])
    validated = store.validate_reviews()
    manifest = store.freeze("pilot", tmp_path / "runs" / "pilot")

    assert validated[0].review_status == "accepted"
    assert store.translations()[0].frozen
    assert manifest.translation_hashes == {record.translation_id: record.translated_text_sha256}
    assert (tmp_path / "runs" / "pilot" / "translation_manifest.json").is_file()


def test_rejected_translation_cannot_be_frozen(tmp_path: Path) -> None:
    store = TranslationStore(tmp_path / "translated")
    translator = FakeTranslator(outputs={("Original", "jv"): "terjemahan"})
    record = TranslationService(store).translate_case(_case(), "jv", translator)
    review = _accepted_review(record.translation_id).model_copy(
        update={"review_id": "rejected", "intent_preserved": "no"}
    )

    store.add_reviews([review])
    validated = store.validate_reviews()

    assert validated[0].review_status == "rejected"
    assert store.freeze("pilot", tmp_path / "runs" / "pilot").translation_hashes == {}


def test_phase2_cli_runs_offline_with_fake_translator(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    translated = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "zh,my",
            "--translator",
            "fake",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert translated.exit_code == 0, translated.output
    assert pq.read_table(output_dir / "translations.parquet").num_rows == 2

    review_csv = tmp_path / "reviews.csv"
    exported = runner.invoke(
        app,
        [
            "export-translation-review",
            "--translations-dir",
            str(output_dir),
            "--output",
            str(review_csv),
        ],
    )
    assert exported.exit_code == 0, exported.output

    rows = list(csv.DictReader(review_csv.open(encoding="utf-8", newline="")))
    for row in rows:
        row.update(
            {
                "reviewer_id": "reviewer-a",
                "rubric_version": "v1",
                "intent_preserved": "yes",
                "requested_action_preserved": "yes",
                "target_preserved": "yes",
                "constraints_preserved": "yes",
                "fluency": "acceptable",
                "ambiguity_added": "no",
            }
        )
    with review_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    imported = runner.invoke(
        app,
        [
            "import-translation-review",
            "--translations-dir",
            str(output_dir),
            "--input",
            str(review_csv),
        ],
    )
    validated = runner.invoke(
        app,
        ["validate-translations", "--translations-dir", str(output_dir)],
    )
    frozen = runner.invoke(
        app,
        [
            "freeze-translations",
            "--translations-dir",
            str(output_dir),
            "--experiment",
            "pilot",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )

    assert imported.exit_code == validated.exit_code == frozen.exit_code == 0
    manifest = json.loads(
        (tmp_path / "runs" / "pilot" / "translation_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["translation_hashes"]) == 2


def test_translate_defaults_to_google_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    translator = FakeTranslator(outputs={("Original", "my"): "default-google"})
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)
    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        lambda: translator,
    )

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "my",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert TranslationStore(output_dir).translations()[0].normalized_translated_text == (
        "default-google"
    )


def test_native_dataset_translation_has_priority(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.parquet"
    native_path = tmp_path / "native.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_text": "Original",
                    "language": "zh",
                    "translated_text": "資料集翻譯",
                }
            ]
        ),
        native_path,
    )

    dataset_run = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "zh",
            "--translator",
            "dataset",
            "--cases-path",
            str(cases_path),
            "--native-translations-path",
            str(native_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    fake_run = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "zh",
            "--translator",
            "fake",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert dataset_run.exit_code == fake_run.exit_code == 0
    records = TranslationStore(output_dir).translations()
    assert len(records) == 1
    assert records[0].translator_id == "native_dataset"
    assert records[0].normalized_translated_text == "資料集翻譯"


def test_native_dataset_is_used_before_requested_machine_translator(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.parquet"
    native_path = tmp_path / "native.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "source_text": "Original",
                    "language": "zh",
                    "translated_text": "資料集優先",
                }
            ]
        ),
        native_path,
    )

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "zh",
            "--translator",
            "fake",
            "--cases-path",
            str(cases_path),
            "--native-translations-path",
            str(native_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    record = TranslationStore(output_dir).translations()[0]
    assert record.translator_id == "native_dataset"
    assert record.normalized_translated_text == "資料集優先"


def test_review_import_creates_human_revision_without_overwriting_source(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "translated"
    store = TranslationStore(output_dir)
    original = TranslationService(store).translate_case(_case(), "zh", FakeTranslator())
    review_csv = tmp_path / "review.csv"
    with review_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "translation_id": original.translation_id,
                "case_id": original.case_id,
                "source_language": original.source_language,
                "target_language": original.target_language,
                "source_text": original.source_text,
                "normalized_translated_text": original.normalized_translated_text,
                "revised_translated_text": "人工修訂",
                "reviewer_id": "reviewer-a",
                "rubric_version": "v1",
                "intent_preserved": "yes",
                "requested_action_preserved": "yes",
                "target_preserved": "yes",
                "constraints_preserved": "yes",
                "fluency": "good",
                "ambiguity_added": "no",
                "notes": "",
            }
        )

    result = runner.invoke(
        app,
        [
            "import-translation-review",
            "--input",
            str(review_csv),
            "--translations-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    records = TranslationStore(output_dir).translations()
    assert len(records) == 2
    assert next(record for record in records if record.translation_id == original.translation_id)
    revision = next(
        record for record in records if record.translation_id != original.translation_id
    )
    assert revision.translator_id == "fake+human_revision"
    assert revision.normalized_translated_text == "人工修訂"
    assert store.reviews()[0].translation_id == revision.translation_id
