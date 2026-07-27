import csv
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from threading import Lock
from time import sleep
from types import MappingProxyType
from typing import Any, Protocol, cast

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


class GoogleCloudTranslationClient(Protocol):
    def translate_text(self, *, request: dict[str, object]) -> object: ...


class GoogleCloudConfigurationError(ValueError):
    """A non-secret Cloud Translation setting is invalid."""


class GoogleCloudProviderError(RuntimeError):
    """A sanitized Cloud Translation provider failure."""


class GoogleCloudAuthenticationError(GoogleCloudProviderError):
    """Application Default Credentials could not be loaded safely."""


class GoogleCloudPermissionError(GoogleCloudProviderError):
    """The authenticated principal is not authorized to translate."""


class GoogleCloudQuotaError(GoogleCloudProviderError):
    """The Cloud Translation request was rejected by a quota control."""


class GoogleCloudInvalidRequestError(GoogleCloudProviderError):
    """Cloud Translation rejected a sanitized request."""


class GoogleCloudTransientError(GoogleCloudProviderError):
    """Cloud Translation encountered a retryable provider failure."""


class GoogleCloudTranslationResponseError(GoogleCloudProviderError):
    """Cloud Translation returned no usable translated text."""


class GoogleCloudReservationError(GoogleCloudProviderError):
    """A paid-call reservation could not be persisted safely."""


class GoogleCloudIndeterminatePaidAttemptError(GoogleCloudProviderError):
    """A paid request has no durable success or failure outcome."""


class GoogleCloudRequestTooLargeError(ValueError):
    """A translation exceeds the configured request character budget."""


class GoogleCloudRunBudgetExceededError(ValueError):
    """A translation would exceed the configured run character budget."""


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
    translator_id = "google-cloud-nmt-v3"
    method = "google_cloud_nmt_v3"
    language_codes: Mapping[str, str] = MappingProxyType(
        {
            "en": "en",
            "id": "id",
            "jv": "jv",
            "my": "my",
            "th": "th",
            "tl": "tl",
            "vi": "vi",
            "zh": "zh-CN",
            "zh-tw": "zh-TW",
        }
    )

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "global",
        model: str = "general/nmt",
        *,
        client: GoogleCloudTranslationClient | None = None,
        client_library_version: str | None = None,
        max_request_characters: int = 5000,
        max_run_characters: int = 100000,
        initial_characters_used: int = 0,
        paid_call_reservation: Callable[[int], None] | None = None,
    ) -> None:
        resolved_project = (project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", resolved_project):
            raise GoogleCloudConfigurationError("Google Cloud project ID is invalid")
        if location != "global":
            raise GoogleCloudConfigurationError("Google Cloud Translation location must be global")
        if model != "general/nmt":
            raise GoogleCloudConfigurationError(
                "Google Cloud Translation model must be general/nmt"
            )
        if (
            isinstance(max_request_characters, bool)
            or not isinstance(max_request_characters, int)
            or max_request_characters <= 0
        ):
            raise GoogleCloudConfigurationError(
                "Google Cloud request character limit must be a positive integer"
            )
        if (
            isinstance(max_run_characters, bool)
            or not isinstance(max_run_characters, int)
            or max_run_characters <= 0
        ):
            raise GoogleCloudConfigurationError(
                "Google Cloud run character limit must be a positive integer"
            )
        if max_request_characters > max_run_characters:
            raise GoogleCloudConfigurationError(
                "Google Cloud request character limit must not exceed the run limit"
            )
        if (
            isinstance(initial_characters_used, bool)
            or not isinstance(initial_characters_used, int)
            or not 0 <= initial_characters_used <= max_run_characters
        ):
            raise GoogleCloudConfigurationError("Google Cloud initial character usage is invalid")
        if client_library_version is None:
            try:
                client_library_version = package_version("google-cloud-translate")
            except PackageNotFoundError:
                client_library_version = "unknown"

        self.project_id = resolved_project
        self.location = location
        self.model = model
        self.client = client
        self.client_library_version = client_library_version
        self.max_request_characters = max_request_characters
        self.max_run_characters = max_run_characters
        self.characters_used = initial_characters_used
        self.paid_call_reservation = paid_call_reservation
        self._budget_lock = Lock()
        self._client_lock = Lock()
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        self.version = model
        self.decoding_config: dict[str, object] = {
            "client_library": "google-cloud-translate",
            "client_library_version": client_library_version,
            "language_codes": dict(self.language_codes),
            "location": location,
            "mime_type": "text/plain",
            "model": model,
            "max_request_characters": max_request_characters,
            "max_run_characters": max_run_characters,
            "project_id": resolved_project,
            "use_language_detection": False,
        }
        self.provider_contract = dict(self.decoding_config)

    @staticmethod
    def _default_client() -> GoogleCloudTranslationClient:
        try:
            from google.cloud import translate_v3
        except ImportError:
            raise GoogleCloudConfigurationError(
                "Google translation support is not installed; run "
                "`uv sync --extra translation-google`."
            ) from None
        try:
            return cast(
                GoogleCloudTranslationClient,
                translate_v3.TranslationServiceClient(),
            )
        except Exception:
            raise GoogleCloudAuthenticationError(
                "Google Cloud application default credentials are unavailable or invalid"
            ) from None

    def _client_for_request(self) -> GoogleCloudTranslationClient:
        with self._client_lock:
            if self.client is None:
                try:
                    self.client = self._default_client()
                except (GoogleCloudAuthenticationError, GoogleCloudConfigurationError):
                    raise
                except Exception:
                    raise GoogleCloudAuthenticationError(
                        "Google Cloud application default credentials are unavailable or invalid"
                    ) from None
            return self.client

    def _provider_language(self, language: str) -> str | None:
        return self.language_codes.get(language)

    def supports(self, source_language: str, target_language: str) -> bool:
        source = self._provider_language(source_language)
        target = self._provider_language(target_language)
        return source is not None and target is not None and source != target

    @staticmethod
    def _categorized_error(error: Exception) -> GoogleCloudProviderError:
        try:
            error_type = type(error)
            error_mro = type.__getattribute__(error_type, "__mro__")
            if not isinstance(error_mro, tuple):
                return GoogleCloudProviderError("Google Cloud Translation provider request failed")
            error_names: set[str] = set()
            for candidate in error_mro:
                if not isinstance(candidate, type):
                    return GoogleCloudProviderError(
                        "Google Cloud Translation provider request failed"
                    )
                name = type.__getattribute__(candidate, "__name__")
                if not isinstance(name, str):
                    return GoogleCloudProviderError(
                        "Google Cloud Translation provider request failed"
                    )
                error_names.add(name)
        except BaseException:
            return GoogleCloudProviderError("Google Cloud Translation provider request failed")
        if error_names & {"DefaultCredentialsError", "RefreshError", "Unauthenticated"}:
            return GoogleCloudAuthenticationError("Google Cloud Translation authentication failed")
        if "PermissionDenied" in error_names:
            return GoogleCloudPermissionError("Google Cloud Translation permission was denied")
        if error_names & {"ResourceExhausted", "TooManyRequests"}:
            return GoogleCloudQuotaError("Google Cloud Translation quota was exceeded")
        if error_names & {"BadRequest", "InvalidArgument"}:
            return GoogleCloudInvalidRequestError("Google Cloud Translation rejected the request")
        if error_names & {
            "Aborted",
            "ConnectionError",
            "DeadlineExceeded",
            "InternalServerError",
            "RetryError",
            "ServiceUnavailable",
            "TimeoutError",
        }:
            return GoogleCloudTransientError("Google Cloud Translation is temporarily unavailable")
        return GoogleCloudProviderError("Google Cloud Translation provider request failed")

    def _reserve_characters(self, character_count: int) -> None:
        if character_count > self.max_request_characters:
            raise GoogleCloudRequestTooLargeError(
                "Google Cloud Translation request character limit exceeded"
            )
        with self._budget_lock:
            if self.characters_used + character_count > self.max_run_characters:
                raise GoogleCloudRunBudgetExceededError(
                    "Google Cloud Translation run character budget exceeded"
                )
            # A provider failure may still consume quota or incur cost, so reservation is not
            # rolled back after the paid-call boundary is crossed.
            self.characters_used += character_count

    def _release_characters(self, character_count: int) -> None:
        with self._budget_lock:
            self.characters_used -= character_count

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> ProviderTranslation:
        if not self.supports(source_language, target_language):
            raise ValueError("Google Cloud Translation does not support the selected language pair")
        source = self.language_codes[source_language]
        target = self.language_codes[target_language]
        character_count = len(text)
        self._reserve_characters(character_count)
        try:
            client = self._client_for_request()
        except Exception:
            self._release_characters(character_count)
            raise
        if self.paid_call_reservation is not None:
            try:
                self.paid_call_reservation(character_count)
            except Exception:
                self._release_characters(character_count)
                raise GoogleCloudReservationError(
                    "Google Cloud Translation paid-call reservation could not be persisted"
                ) from None
        try:
            response = client.translate_text(
                request={
                    "parent": self.parent,
                    "contents": [text],
                    "mime_type": "text/plain",
                    "source_language_code": source,
                    "target_language_code": target,
                    "model": f"{self.parent}/models/{self.model}",
                }
            )
        except Exception as error:
            raise self._categorized_error(error) from None
        try:
            request_id = getattr(response, "request_id", None)
            translations = getattr(response, "translations", None)
            if not translations:
                raise GoogleCloudTranslationResponseError(
                    "Google Cloud Translation returned an unusable response"
                )
            translated_text = getattr(cast(Any, translations)[0], "translated_text", None)
            if not isinstance(translated_text, str) or not translated_text.strip():
                raise GoogleCloudTranslationResponseError(
                    "Google Cloud Translation returned an unusable response"
                )
            safe_request_id = (
                request_id
                if isinstance(request_id, str)
                and len(request_id) <= 256
                and re.fullmatch(r"[A-Za-z0-9._=-]+", request_id)
                else None
            )
        except GoogleCloudTranslationResponseError:
            raise
        except Exception:
            raise GoogleCloudTranslationResponseError(
                "Google Cloud Translation returned an unusable response"
            ) from None
        return ProviderTranslation(translated_text, safe_request_id)


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
