import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from crosslingual_safety.schemas import (
    TranslationManifest,
    TranslationRecord,
    TranslationReview,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_models(path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.is_file():
        return []
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if "decoding_config_json" in row:
            row["decoding_config"] = json.loads(row.pop("decoding_config_json"))
    return [model.model_validate(row) for row in rows]


def _write_models(path: Path, models: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [model.model_dump(mode="json") for model in models]
    for row in rows:
        if "decoding_config" in row:
            row["decoding_config_json"] = json.dumps(
                row.pop("decoding_config"), sort_keys=True, separators=(",", ":")
            )
    if not rows:
        return
    pq.write_table(pa.Table.from_pylist(rows), path)


def review_status(review: TranslationReview) -> str:
    values = (
        review.intent_preserved,
        review.requested_action_preserved,
        review.target_preserved,
        review.constraints_preserved,
    )
    if "no" in values or review.fluency == "poor" or review.ambiguity_added == "yes":
        return "rejected"
    if (
        "uncertain" in values
        or review.fluency == "uncertain"
        or review.ambiguity_added == "uncertain"
    ):
        return "needs_revision"
    if review.fluency in {"good", "acceptable"} and review.ambiguity_added == "no":
        return "accepted"
    return "needs_revision"


class TranslationStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.translations_path = root / "translations.parquet"
        self.reviews_path = root / "translation_reviews.parquet"
        self.rejected_path = root / "rejected_translations.parquet"
        self.candidates_dir = root / "candidates"
        self.provider_cache_dir = root / "provider_cache"
        self._translation_cache: dict[str, TranslationRecord] | None = None

    def translations(self) -> list[TranslationRecord]:
        if self._translation_cache is None:
            records = {
                record.translation_id: record
                for record in _read_models(self.translations_path, TranslationRecord)
            }
            if self.candidates_dir.is_dir():
                for path in self.candidates_dir.glob("*.json"):
                    record = TranslationRecord.model_validate_json(path.read_text(encoding="utf-8"))
                    records.setdefault(record.translation_id, record)
            self._translation_cache = records
        return list(self._translation_cache.values())

    def reviews(self) -> list[TranslationReview]:
        return _read_models(self.reviews_path, TranslationReview)

    def get(self, translation_id: str) -> TranslationRecord | None:
        return next(
            (record for record in self.translations() if record.translation_id == translation_id),
            None,
        )

    def find(self, case_id: str, target_language: str) -> list[TranslationRecord]:
        return [
            record
            for record in self.translations()
            if record.case_id == case_id and record.target_language == target_language
        ]

    def get_provider_response(self, cache_key: str) -> tuple[str, str | None] | None:
        path = self.provider_cache_dir / f"{cache_key}.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value["text"]), value.get("provider_request_id")

    def add_provider_response(
        self,
        cache_key: str,
        text: str,
        provider_request_id: str | None,
    ) -> None:
        self.provider_cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.provider_cache_dir / f"{cache_key}.json"
        value = {
            "cache_key": cache_key,
            "text": text,
            "provider_request_id": provider_request_id,
        }
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != value:
                raise ValueError(f"immutable provider cache conflict: {cache_key}")
            return
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

    def add_translation(
        self, record: TranslationRecord, *, defer_snapshot: bool = False
    ) -> TranslationRecord:
        records = {item.translation_id: item for item in self.translations()}
        existing = records.get(record.translation_id)
        if existing is not None:
            if existing.model_dump() != record.model_dump():
                immutable_fields = (
                    existing.normalized_translated_text,
                    existing.translated_text_sha256,
                    existing.source_text_sha256,
                )
                incoming_fields = (
                    record.normalized_translated_text,
                    record.translated_text_sha256,
                    record.source_text_sha256,
                )
                if immutable_fields != incoming_fields:
                    raise ValueError(f"immutable translation conflict: {record.translation_id}")
            return existing
        records[record.translation_id] = record
        self._translation_cache = records
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = self.candidates_dir / f"{record.translation_id}.json"
        candidate_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        if not defer_snapshot:
            self.flush()
        return record

    def flush(self) -> None:
        _write_models(self.translations_path, self.translations())

    def add_reviews(self, reviews: list[TranslationReview]) -> None:
        known_translations = {record.translation_id for record in self.translations()}
        records = {review.review_id: review for review in self.reviews()}
        for review in reviews:
            if review.translation_id not in known_translations:
                raise ValueError(f"review references unknown translation: {review.translation_id}")
            existing = records.get(review.review_id)
            if existing is not None:
                existing_values = existing.model_dump(exclude={"created_at"})
                incoming_values = review.model_dump(exclude={"created_at"})
                if existing_values != incoming_values:
                    raise ValueError(f"immutable review conflict: {review.review_id}")
                continue
            records[review.review_id] = review
        _write_models(self.reviews_path, list(records.values()))

    def validate_reviews(self) -> list[TranslationRecord]:
        reviews_by_translation: dict[str, list[TranslationReview]] = {}
        for review in self.reviews():
            reviews_by_translation.setdefault(review.translation_id, []).append(review)

        validated: list[TranslationRecord] = []
        rejected: list[TranslationRecord] = []
        for record in self.translations():
            statuses = [
                review_status(review)
                for review in reviews_by_translation.get(record.translation_id, [])
            ]
            if not statuses:
                status = "pending"
            elif all(value == "accepted" for value in statuses):
                status = "accepted"
            elif "rejected" in statuses:
                status = "rejected"
            else:
                status = "needs_revision"
            updated = record.model_copy(update={"review_status": status})
            validated.append(updated)
            if status == "rejected":
                rejected.append(updated)
        _write_models(self.translations_path, validated)
        self._translation_cache = {record.translation_id: record for record in validated}
        if rejected:
            _write_models(self.rejected_path, rejected)
        return validated

    def freeze(self, experiment_id: str, experiment_dir: Path) -> TranslationManifest:
        validated = self.validate_reviews()
        frozen = [
            record.model_copy(update={"frozen": True})
            if record.review_status == "accepted"
            else record
            for record in validated
        ]
        _write_models(self.translations_path, frozen)
        self._translation_cache = {record.translation_id: record for record in frozen}
        accepted = [
            record for record in frozen if record.frozen and record.review_status == "accepted"
        ]
        provider_language_support = {
            record.translator_id: sorted(str(value) for value in supported)
            for record in accepted
            if isinstance(supported := record.decoding_config.get("supported_languages"), list)
        }
        manifest = TranslationManifest(
            experiment_id=experiment_id,
            created_at=utc_now(),
            translation_hashes={
                record.translation_id: record.translated_text_sha256 for record in accepted
            },
            provider_language_support=provider_language_support,
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "translation_manifest.json").write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_models(experiment_dir / "frozen_translations.parquet", accepted)
        return manifest
