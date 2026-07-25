import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from crosslingual_safety.translation.languages import LanguageConfig


@dataclass(frozen=True)
class ProviderTranslation:
    text: str
    provider_request_id: str | None = None


class Translator(Protocol):
    translator_id: str
    version: str
    method: str
    decoding_config: dict[str, object]

    def supports(self, source_language: str, target_language: str) -> bool: ...

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation: ...


class FakeTranslator:
    translator_id = "fake"
    version = "1"
    method = "test_only"
    decoding_config: dict[str, object] = {}

    def __init__(self, outputs: dict[tuple[str, str], str] | None = None) -> None:
        self.outputs = outputs or {}
        self.call_count = 0

    def supports(self, source_language: str, target_language: str) -> bool:
        return source_language != target_language

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        self.call_count += 1
        return ProviderTranslation(
            self.outputs.get((text, target_language), f"[{target_language}] {text}")
        )


class DatasetTranslationProvider:
    translator_id = "native_dataset"
    version = "phase1_snapshot_v1"
    method = "native_dataset"
    decoding_config: dict[str, object] = {}

    def __init__(self, values: dict[tuple[str, str], str]) -> None:
        self.values = values

    def supports(self, source_language: str, target_language: str) -> bool:
        return source_language == "en" and any(key[1] == target_language for key in self.values)

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        try:
            return ProviderTranslation(self.values[(text, target_language)])
        except KeyError as error:
            raise ValueError(
                f"native dataset translation is unavailable for {target_language}"
            ) from error


class ManualTranslationProvider(DatasetTranslationProvider):
    translator_id = "manual"
    method = "manual"

    def __init__(self, values: dict[tuple[str, str], str], version: str = "manual_v1") -> None:
        super().__init__(values)
        self.version = version

    @classmethod
    def from_csv(cls, path: Path, version: str = "manual_v1") -> "ManualTranslationProvider":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {"source_text", "target_language", "translated_text"}
        if not rows or not required <= rows[0].keys():
            raise ValueError(f"manual translation CSV must contain {sorted(required)}")
        return cls(
            {(row["source_text"], row["target_language"]): row["translated_text"] for row in rows},
            version,
        )


class GoogleCloudNMTTranslator:
    translator_id = "google_cloud_nmt_v3"
    version = "general/nmt"
    method = "google_cloud_nmt_v3"
    decoding_config: dict[str, object] = {
        "model": "general/nmt",
        "mime_type": "text/plain",
        "use_language_detection": False,
    }

    def __init__(self, project_id: str | None = None, location: str = "global") -> None:
        try:
            from google.cloud import translate_v3
        except ImportError as error:
            raise RuntimeError(
                "Google translation support is not installed; run "
                "`uv sync --extra translation-google`."
            ) from error
        self.project_id = project_id or os.environ["GOOGLE_CLOUD_PROJECT"]
        self.location = location
        self.client = translate_v3.TranslationServiceClient()
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        response = self.client.get_supported_languages(
            request={"parent": self.parent, "display_language_code": "en"}
        )
        self.supported_languages = {language.language_code for language in response.languages}
        self.decoding_config = {
            **type(self).decoding_config,
            "supported_languages": sorted(self.supported_languages),
        }

    def supports(self, source_language: str, target_language: str) -> bool:
        return source_language != target_language and target_language in self.supported_languages

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        response = self.client.translate_text(
            request={
                "parent": self.parent,
                "contents": [text],
                "mime_type": "text/plain",
                "source_language_code": source_language,
                "target_language_code": target_language,
                "model": f"{self.parent}/models/general/nmt",
            }
        )
        request_id = getattr(response, "request_id", None)
        return ProviderTranslation(response.translations[0].translated_text, request_id)


class NLLBTranslator:
    translator_id = "nllb"
    method = "nllb"

    def __init__(
        self,
        languages: dict[str, LanguageConfig],
        checkpoint: str = "facebook/nllb-200-distilled-600M",
        num_beams: int = 5,
        max_input_tokens: int = 512,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "NLLB support is not installed; run `uv sync --extra translation-nllb`."
            ) from error
        self.languages = languages
        self.version = checkpoint
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        self.decoding_config: dict[str, object] = {
            "do_sample": False,
            "num_beams": num_beams,
            "max_input_tokens": max_input_tokens,
            "local_files_only": local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )

    def supports(self, source_language: str, target_language: str) -> bool:
        return (
            source_language != target_language
            and source_language in self.languages
            and target_language in self.languages
        )

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        source_code = self.languages[source_language].nllb_code
        target_code = self.languages[target_language].nllb_code
        self.tokenizer.src_lang = source_code
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        token_count = int(encoded["input_ids"].shape[-1])
        if token_count > self.max_input_tokens:
            raise ValueError(
                f"NLLB input has {token_count} tokens; limit is {self.max_input_tokens}"
            )
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
            num_beams=self.num_beams,
            do_sample=False,
        )
        return ProviderTranslation(
            self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        )
