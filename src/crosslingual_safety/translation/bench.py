"""Deterministic selection of dataset-native benchmark translations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from opencc import OpenCC

from crosslingual_safety.ids import canonicalize_text
from crosslingual_safety.schemas import NativeTranslation
from crosslingual_safety.translation.providers import ProviderTranslation

NATIVE_PUBLIC_LANGUAGES = frozenset({"jv", "th", "vi", "zh-tw"})
TRADITIONAL_CHINESE_CONVERSION = "s2twp"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchTranslation:
    case_id: str
    target_language: str
    text: str
    source_record_id: str
    native_language: str
    conversion: str | None

    @property
    def decoding_config(self) -> dict[str, object]:
        value: dict[str, object] = {
            "dataset_language": self.native_language,
            "source_record_id": self.source_record_id,
        }
        if self.conversion is not None:
            value["conversion"] = self.conversion
        return value

    @property
    def provenance(self) -> dict[str, object]:
        return {
            **self.decoding_config,
            "output_text_sha256": _sha256_text(self.text),
        }


class NativeBenchTranslator:
    """Translator protocol adapter for one already-selected native record."""

    translator_id = "native_dataset"
    version = "multijail-native-v1"
    method = "native_dataset"

    def __init__(self, selection: BenchTranslation) -> None:
        self.selection = selection
        self.decoding_config = selection.decoding_config

    def supports(self, source_language: str, target_language: str) -> bool:
        return source_language == "en" and target_language == self.selection.target_language

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        if not self.supports(source_language, target_language):
            raise ValueError(
                f"native translation does not support {source_language}->{target_language}"
            )
        return ProviderTranslation(self.selection.text)


class BenchTranslationCatalog:
    """Validated immutable lookup over ``native_translations.parquet``."""

    def __init__(
        self,
        records: dict[tuple[str, str], NativeTranslation],
        *,
        converter: OpenCC | None = None,
    ) -> None:
        self._records = records
        self._converter = converter or OpenCC(TRADITIONAL_CHINESE_CONVERSION)

    @classmethod
    def from_parquet(cls, path: Path) -> BenchTranslationCatalog:
        if not path.is_file():
            raise ValueError(f"native translations do not exist: {path}")
        records: dict[tuple[str, str], NativeTranslation] = {}
        for raw in pq.read_table(path).to_pylist():
            record = NativeTranslation.model_validate(raw)
            if not canonicalize_text(record.source_text):
                raise ValueError("native translation has blank source text")
            if not canonicalize_text(record.translated_text):
                raise ValueError("native translation has blank translated text")
            key = (record.case_id, record.language)
            if key in records:
                raise ValueError(
                    f"duplicate native translation: {record.case_id}/{record.language}"
                )
            records[key] = record
        return cls(records)

    def resolve(
        self,
        *,
        case_id: str,
        source_text: str,
        target_language: str,
    ) -> BenchTranslation | None:
        if target_language not in NATIVE_PUBLIC_LANGUAGES:
            return None
        native_language = "zh" if target_language == "zh-tw" else target_language
        record = self._records.get((case_id, native_language))
        if record is None:
            return None
        if canonicalize_text(record.source_text) != canonicalize_text(source_text):
            raise ValueError(f"native translation source mismatch: {case_id}/{native_language}")
        text = canonicalize_text(record.translated_text)
        conversion = None
        if target_language == "zh-tw":
            text = canonicalize_text(self._converter.convert(text))
            conversion = TRADITIONAL_CHINESE_CONVERSION
        return BenchTranslation(
            case_id=case_id,
            target_language=target_language,
            text=text,
            source_record_id=record.source_record_id,
            native_language=native_language,
            conversion=conversion,
        )
