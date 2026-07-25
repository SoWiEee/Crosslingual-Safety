import csv
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from time import sleep
from typing import Protocol

from crosslingual_safety.translation.languages import LanguageConfig


@dataclass(frozen=True)
class ProviderTranslation:
    text: str
    provider_request_id: str | None = None


class TranslationInputTooLongError(ValueError):
    def __init__(self, token_count: int, max_input_tokens: int) -> None:
        self.token_count = token_count
        self.max_input_tokens = max_input_tokens
        super().__init__(f"NLLB input has {token_count} tokens; limit is {max_input_tokens}")


class Translator(Protocol):
    translator_id: str
    version: str
    method: str
    decoding_config: dict[str, object]

    def supports(self, source_language: str, target_language: str) -> bool: ...

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation: ...


class DeepTranslatorBackend(Protocol):
    def get_supported_languages(self, as_dict: bool = False) -> dict[str, str]: ...

    def translate(self, text: str) -> str: ...


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


class DeepTranslatorGoogleTranslator:
    """Free Google Translate web frontend exposed through deep-translator."""

    translator_id = "deep_translator_google"
    method = "deep_translator_google"
    language_code_overrides = {
        "jv": "jw",
        "zh": "zh-CN",
    }

    def __init__(
        self,
        translator_factory: Callable[..., DeepTranslatorBackend] | None = None,
        retry_delays: tuple[float, ...] = (1.0, 2.0),
    ) -> None:
        if translator_factory is None:
            try:
                from deep_translator import GoogleTranslator
            except ImportError as error:
                raise RuntimeError(
                    "Free translation support is not installed; run `uv sync`."
                ) from error
            translator_factory = GoogleTranslator

        self.translator_factory = translator_factory
        self.retry_delays = retry_delays
        self.version = package_version("deep-translator")
        probe = translator_factory(source="en", target="zh-CN")
        supported = probe.get_supported_languages(as_dict=True)
        self.supported_language_codes = set(supported.values())
        self.decoding_config: dict[str, object] = {
            "backend": "google",
            "language_code_overrides": dict(self.language_code_overrides),
            "retry_delays_seconds": list(retry_delays),
            "supported_languages": sorted(self.supported_language_codes),
        }

    def _provider_code(self, language: str) -> str:
        return self.language_code_overrides.get(language, language)

    def supports(self, source_language: str, target_language: str) -> bool:
        return (
            source_language != target_language
            and self._provider_code(source_language) in self.supported_language_codes
            and self._provider_code(target_language) in self.supported_language_codes
        )

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        if not self.supports(source_language, target_language):
            raise ValueError(
                f"deep-translator does not support {source_language}->{target_language}"
            )
        translated: str | None = None
        last_error: Exception | None = None
        for attempt in range(len(self.retry_delays) + 1):
            try:
                translator = self.translator_factory(
                    source=self._provider_code(source_language),
                    target=self._provider_code(target_language),
                )
                translated = translator.translate(text)
                break
            except Exception as error:
                last_error = error
                if attempt < len(self.retry_delays):
                    sleep(self.retry_delays[attempt])
        if last_error is not None and translated is None:
            raise ValueError(
                "deep-translator request failed for "
                f"{source_language}->{target_language}: {type(last_error).__name__}"
            ) from None
        if not isinstance(translated, str) or not translated.strip():
            raise ValueError(
                f"deep-translator returned empty text for {source_language}->{target_language}"
            )
        return ProviderTranslation(translated)


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
        max_input_tokens: int = 1024,
        local_files_only: bool = False,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "NLLB support is not installed; run `uv sync --extra translation-nllb`."
            ) from error
        if not device.startswith("cuda"):
            raise ValueError("NLLB translation device must be CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for NLLB translation but is unavailable")
        if dtype != "float16":
            raise ValueError("NLLB translation dtype must be float16")
        self.languages = languages
        self.version = checkpoint
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        self.device = device
        self.dtype = dtype
        self.torch = torch
        self.decoding_config: dict[str, object] = {
            "checkpoint": checkpoint,
            "device": device,
            "dtype": dtype,
            "do_sample": False,
            "num_beams": num_beams,
            "max_input_tokens": max_input_tokens,
            "local_files_only": local_files_only,
            "language_codes": {
                code: config.nllb_code for code, config in sorted(languages.items())
            },
            "torch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint,
            local_files_only=local_files_only,
            dtype=torch.float16,
        )
        self.model.to(device)
        self.model.eval()

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
            raise TranslationInputTooLongError(token_count, self.max_input_tokens)
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        try:
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_code),
                    num_beams=self.num_beams,
                    do_sample=False,
                )
        except RuntimeError as error:
            raise RuntimeError(
                f"NLLB translation failed for {source_language}->{target_language} "
                f"with {self.version}: {type(error).__name__}"
            ) from None
        translated = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        if not translated.strip():
            raise ValueError(f"NLLB returned empty text for {source_language}->{target_language}")
        return ProviderTranslation(translated)
