import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from crosslingual_safety.cli import app
from crosslingual_safety.schemas import PromptCase, TranslationReview
from crosslingual_safety.translation.commands import REVIEW_FIELDS
from crosslingual_safety.translation.languages import load_languages
from crosslingual_safety.translation.paid_ledger import PaidCallLedgerError
from crosslingual_safety.translation.providers import (
    DeepTranslatorGoogleTranslator,
    FakeTranslator,
    GoogleCloudAuthenticationError,
    GoogleCloudInvalidRequestError,
    GoogleCloudNMTTranslator,
    GoogleCloudPermissionError,
    GoogleCloudProviderError,
    GoogleCloudQuotaError,
    GoogleCloudRequestTooLargeError,
    GoogleCloudRunBudgetExceededError,
    GoogleCloudTransientError,
    GoogleCloudTranslationResponseError,
    NLLBTranslator,
    ProviderTranslation,
    TranslationInputTooLongError,
)
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


def test_deep_translator_google_maps_project_language_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class StubGoogleTranslator:
        def __init__(self, source: str, target: str) -> None:
            calls.append((source, target))

        def get_supported_languages(self, as_dict: bool = False) -> dict[str, str]:
            assert as_dict
            return {
                "english": "en",
                "chinese (simplified)": "zh-CN",
                "javanese": "jw",
                "myanmar (burmese)": "my",
                "vietnamese": "vi",
            }

        def translate(self, text: str) -> str:
            return f"translated: {text}"

    monkeypatch.setattr(
        "crosslingual_safety.translation.providers.package_version",
        lambda _: "test-version",
    )
    translator = DeepTranslatorGoogleTranslator(StubGoogleTranslator)

    assert translator.supports("en", "zh")
    assert translator.supports("en", "jv")
    assert translator.supports("en", "vi")
    assert translator.supports("en", "my")
    assert translator.translate("Original", "en", "jv").text == "translated: Original"
    assert calls[-1] == ("en", "jw")
    assert translator.translate("Original", "en", "zh").text == "translated: Original"
    assert calls[-1] == ("en", "zh-CN")
    assert translator.translate("Original", "en", "my").text == "translated: Original"
    assert calls[-1] == ("en", "my")
    assert translator.translate("Original", "en", "vi").text == "translated: Original"
    assert calls[-1] == ("en", "vi")
    assert translator.version == "test-version"


def test_deep_translator_google_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class FlakyGoogleTranslator:
        def __init__(self, source: str, target: str) -> None:
            pass

        def get_supported_languages(self, as_dict: bool = False) -> dict[str, str]:
            return {"english": "en", "vietnamese": "vi"}

        def translate(self, text: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError
            return "Nghiên cứu đánh giá an toàn."

    monkeypatch.setattr(
        "crosslingual_safety.translation.providers.package_version",
        lambda _: "test-version",
    )
    translator = DeepTranslatorGoogleTranslator(FlakyGoogleTranslator, retry_delays=(0,))

    assert translator.translate("Safety evaluation research.", "en", "vi").text
    assert attempts == 2


@pytest.mark.parametrize(
    ("target_language", "provider_language", "translated_text"),
    [
        ("zh-tw", "zh-TW", "安全評估研究。"),
        ("vi", "vi", "Nghiên cứu đánh giá an toàn."),
        ("my", "my", "ဘေးကင်းရေး အကဲဖြတ် သုတေသန။"),
    ],
)
def test_google_cloud_nmt_uses_exact_v3_request_and_static_language_map(
    target_language: str,
    provider_language: str,
    translated_text: str,
) -> None:
    requests: list[dict[str, object]] = []

    class StubTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            requests.append(request)
            return SimpleNamespace(
                translations=[SimpleNamespace(translated_text=translated_text)],
                request_id="request-123",
            )

    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        location="global",
        model="general/nmt",
        client=StubTranslationClient(),
        client_library_version="3.test",
    )

    result = translator.translate("Safety evaluation research.", "en", target_language)

    assert result == ProviderTranslation(translated_text, "request-123")
    assert requests == [
        {
            "parent": "projects/gen-lang-client-0036391889/locations/global",
            "contents": ["Safety evaluation research."],
            "mime_type": "text/plain",
            "source_language_code": "en",
            "target_language_code": provider_language,
            "model": ("projects/gen-lang-client-0036391889/locations/global/models/general/nmt"),
        }
    ]
    assert translator.supports("zh", target_language)
    assert translator.decoding_config["language_codes"]["zh"] == "zh-CN"
    assert translator.decoding_config["language_codes"]["zh-tw"] == "zh-TW"
    assert "supported_languages" not in translator.decoding_config


def test_google_cloud_nmt_validates_source_and_target_before_request() -> None:
    class StubTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            self.call_count += 1
            return SimpleNamespace(translations=[SimpleNamespace(translated_text="translated")])

    client = StubTranslationClient()
    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=client,
        client_library_version="3.test",
    )

    with pytest.raises(ValueError, match="does not support the selected language pair"):
        translator.translate("text", "unknown", "vi")
    with pytest.raises(ValueError, match="does not support the selected language pair"):
        translator.translate("text", "en", "unknown")

    assert client.call_count == 0


def test_google_cloud_nmt_enforces_request_and_run_budgets_before_calls() -> None:
    class StubTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            self.call_count += 1
            return SimpleNamespace(translations=[SimpleNamespace(translated_text="translated")])

    client = StubTranslationClient()
    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=client,
        client_library_version="3.test",
        max_request_characters=5,
        max_run_characters=8,
    )

    with pytest.raises(GoogleCloudRequestTooLargeError):
        translator.translate("123456", "en", "vi")
    translator.translate("12345", "en", "vi")
    with pytest.raises(GoogleCloudRunBudgetExceededError):
        translator.translate("1234", "en", "my")

    assert client.call_count == 1
    assert translator.characters_used == 5


def test_google_cloud_nmt_reserves_failed_requests_against_run_budget() -> None:
    class FailingTranslationClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            self.call_count += 1
            raise RuntimeError("untrusted provider detail")

    client = FailingTranslationClient()
    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=client,
        client_library_version="3.test",
        max_request_characters=5,
        max_run_characters=8,
    )

    with pytest.raises(GoogleCloudProviderError):
        translator.translate("12345", "en", "vi")
    with pytest.raises(GoogleCloudRunBudgetExceededError):
        translator.translate("1234", "en", "my")

    assert client.call_count == 1
    assert translator.characters_used == 5


@pytest.mark.parametrize(
    ("provider_error_name", "expected_error"),
    [
        ("Unauthenticated", GoogleCloudAuthenticationError),
        ("PermissionDenied", GoogleCloudPermissionError),
        ("ResourceExhausted", GoogleCloudQuotaError),
        ("InvalidArgument", GoogleCloudInvalidRequestError),
        ("DeadlineExceeded", GoogleCloudTransientError),
        ("UnknownProviderFailure", GoogleCloudProviderError),
    ],
)
def test_google_cloud_nmt_categorizes_and_redacts_provider_failures(
    provider_error_name: str,
    expected_error: type[Exception],
) -> None:
    credential_path = "C:/sensitive/google-translate-service-account.json"
    key_material = "private-key-material"
    prompt_text = "untrusted prompt text"
    provider_error = type(provider_error_name, (RuntimeError,), {})(
        f"{credential_path} {key_material} {prompt_text}"
    )

    class FailingTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            raise provider_error

    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=FailingTranslationClient(),
        client_library_version="3.test",
    )

    with pytest.raises(expected_error) as captured:
        translator.translate("Safety evaluation research.", "en", "vi")

    message = str(captured.value)
    assert credential_path not in message
    assert key_material not in message
    assert prompt_text not in message


def test_google_cloud_nmt_hostile_exception_metaclass_fails_closed() -> None:
    leaked_detail = "HOSTILE_EXCEPTION_NAME_SENTINEL"

    class HostileExceptionMeta(type):
        def __getattribute__(cls, name: str) -> object:
            if name == "__name__":
                raise RuntimeError(leaked_detail)
            return super().__getattribute__(name)

    class HostileProviderError(RuntimeError, metaclass=HostileExceptionMeta):
        pass

    class FailingTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            raise HostileProviderError

    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=FailingTranslationClient(),
        client_library_version="3.test",
    )

    with pytest.raises(GoogleCloudProviderError) as captured:
        translator.translate("Safety evaluation research.", "en", "vi")

    assert type(captured.value) is GoogleCloudProviderError
    assert str(captured.value) == "Google Cloud Translation provider request failed"
    assert leaked_detail not in str(captured.value)


def test_google_cloud_nmt_discards_unsafe_request_correlation() -> None:
    class StubTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            return SimpleNamespace(
                translations=[SimpleNamespace(translated_text="translated")],
                request_id="C:/sensitive/adc.json",
            )

    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=StubTranslationClient(),
        client_library_version="3.test",
    )

    result = translator.translate("Safe text", "en", "vi")

    assert result.provider_request_id is None


def test_google_cloud_nmt_categorizes_response_property_failures() -> None:
    leaked_detail = "C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL"

    class MaliciousResponse:
        @property
        def translations(self) -> object:
            raise RuntimeError(leaked_detail)

    class StubTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            return MaliciousResponse()

    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=StubTranslationClient(),
        client_library_version="3.test",
    )

    with pytest.raises(GoogleCloudTranslationResponseError) as captured:
        translator.translate("Safe text", "en", "vi")

    assert str(captured.value) == "Google Cloud Translation returned an unusable response"
    assert leaked_detail not in str(captured.value)


def test_translate_google_response_failure_is_sanitized_for_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_detail = "C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL"
    cases_path = tmp_path / "cases.parquet"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class MaliciousResponse:
        @property
        def translations(self) -> object:
            raise RuntimeError(leaked_detail)

    class StubTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            return MaliciousResponse()

    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=StubTranslationClient(),
        client_library_version="3.test",
    )
    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        lambda: translator,
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "vi",
            "--translator",
            "google-cloud-nmt-v3",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(tmp_path / "translated"),
            "--languages-config",
            str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    terminal = result.output
    assert "failed 1" in terminal
    assert leaked_detail not in terminal
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            tmp_path / "translated" / "audit" / "translation_reservations.jsonl",
            tmp_path / "translated" / "audit" / "translation_reservation_outcomes.jsonl",
            tmp_path / "translated" / "translation_failures.jsonl",
        )
    )
    assert (
        "Google Cloud Translation paid attempt outcome is indeterminate; manual review is required"
    ) in persisted
    assert leaked_detail not in persisted


@pytest.mark.live_google
@pytest.mark.skipif(
    os.environ.get("RUN_GOOGLE_TRANSLATION_LIVE") != "1",
    reason=(
        "set RUN_GOOGLE_TRANSLATION_LIVE=1 and select -m live_google "
        "to incur an explicit Google Cloud request"
    ),
)
def test_live_google_cloud_nmt_translates_harmless_sentence() -> None:
    translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        location="global",
        model="general/nmt",
        max_request_characters=5000,
        max_run_characters=100000,
    )
    sentence = "This is a harmless translation smoke test."

    results = {
        target: translator.translate(sentence, "en", target).text
        for target in ("zh-tw", "vi", "my")
    }

    assert all(result.strip() for result in results.values())
    assert translator.characters_used == len(sentence) * 3


def test_nllb_uses_cuda_fp16_and_project_language_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forced_bos_codes: list[str] = []

    class StubTensor:
        shape = (1, 579)

        def __init__(self) -> None:
            self.device: str | None = None

        def to(self, device: str) -> "StubTensor":
            self.device = device
            return self

    class StubTokenizer:
        src_lang = ""
        encoded = StubTensor()

        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object) -> "StubTokenizer":
            return cls()

        def __call__(self, text: str, **kwargs: object) -> dict[str, StubTensor]:
            return {"input_ids": self.encoded}

        def convert_tokens_to_ids(self, code: str) -> int:
            forced_bos_codes.append(code)
            return 101

        def batch_decode(self, generated: object, **kwargs: object) -> list[str]:
            return ["translated"]

    class StubModel:
        loaded_dtype: object = None
        device: str | None = None
        evaluated = False

        @classmethod
        def from_pretrained(cls, checkpoint: str, **kwargs: object) -> "StubModel":
            cls.loaded_dtype = kwargs.get("dtype")
            return cls()

        def to(self, device: str) -> "StubModel":
            type(self).device = device
            return self

        def eval(self) -> "StubModel":
            type(self).evaluated = True
            return self

        def generate(self, **kwargs: object) -> list[list[int]]:
            return [[101, 102]]

    transformers = SimpleNamespace(
        AutoModelForSeq2SeqLM=StubModel,
        AutoTokenizer=StubTokenizer,
        __version__="test-transformers",
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float16="float16",
        inference_mode=nullcontext,
        __version__="test-torch",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    languages = load_languages(Path("configs/languages.yaml"))

    translator = NLLBTranslator(languages, device="cuda", dtype="float16")
    results = [
        translator.translate("Safety evaluation research.", "en", target)
        for target in ("zh", "vi", "my")
    ]

    assert StubModel.loaded_dtype == "float16"
    assert StubModel.device == "cuda"
    assert StubModel.evaluated
    assert StubTokenizer.encoded.device == "cuda"
    assert forced_bos_codes == ["zho_Hans", "vie_Latn", "mya_Mymr"]
    assert all(result.text == "translated" for result in results)


def test_nllb_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = SimpleNamespace(
        AutoModelForSeq2SeqLM=object,
        AutoTokenizer=object,
        __version__="test-transformers",
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        __version__="test-torch",
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(
        RuntimeError,
        match="CUDA is required for NLLB translation but is unavailable",
    ):
        NLLBTranslator(load_languages(Path("configs/languages.yaml")))


def test_translate_defaults_to_local_nllb_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    translator = FakeTranslator(outputs={("Original", "my"): "default-local-nllb"})
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)
    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.NLLBTranslator",
        lambda languages: translator,
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
        "default-local-nllb"
    )


def test_translate_keeps_explicit_google_cloud_advanced_provider_option(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    factory_calls = 0
    dotenv_calls: list[Path] = []
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)
    (tmp_path / ".env").write_text("GOOGLE_CLOUD_PROJECT=test-project\n", encoding="utf-8")

    class StubTranslationClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            return SimpleNamespace(
                translations=[SimpleNamespace(translated_text="Google stub translation")]
            )

    def google_translator_factory() -> GoogleCloudNMTTranslator:
        nonlocal factory_calls
        factory_calls += 1
        assert dotenv_calls == [tmp_path / ".env"]
        return GoogleCloudNMTTranslator(
            project_id="gen-lang-client-0036391889",
            client=StubTranslationClient(),
            client_library_version="3.test",
        )

    def scoped_load_dotenv(*, dotenv_path: Path) -> None:
        dotenv_calls.append(dotenv_path)

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        google_translator_factory,
    )
    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.load_dotenv",
        scoped_load_dotenv,
        raising=False,
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "vi",
            "--translator",
            "google-cloud-nmt-v3",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(output_dir),
            "--languages-config",
            str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert factory_calls == 1
    assert dotenv_calls == [tmp_path / ".env"]
    assert TranslationStore(output_dir).translations()[0].normalized_translated_text == (
        "Google stub translation"
    )


@pytest.mark.parametrize("target_language", ["en", ""])
def test_translate_google_zero_paid_work_constructs_no_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_language: str,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)
    constructor_calls = 0

    def forbidden_google_translator() -> FakeTranslator:
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("zero paid work constructed a Google client")

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        forbidden_google_translator,
    )

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            target_language,
            "--translator",
            "google-cloud-nmt-v3",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert constructor_calls == 0
    assert not (output_dir / "audit").exists()


def test_translate_google_cached_work_constructs_no_sdk_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class SeedClient:
        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            return SimpleNamespace(
                translations=[SimpleNamespace(translated_text="cached translation")]
            )

    seed_translator = GoogleCloudNMTTranslator(
        project_id="gen-lang-client-0036391889",
        client=SeedClient(),
    )
    TranslationService(TranslationStore(output_dir)).translate_case(
        _case(),
        "vi",
        seed_translator,
    )
    constructor_calls = 0

    def forbidden_sdk_client() -> object:
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("cached work constructed a Google SDK client")

    monkeypatch.setattr(
        GoogleCloudNMTTranslator,
        "_default_client",
        staticmethod(forbidden_sdk_client),
    )
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0036391889")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "vi",
            "--translator",
            "google-cloud-nmt-v3",
            "--cases-path",
            str(cases_path),
            "--output-dir",
            str(output_dir),
            "--languages-config",
            str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert constructor_calls == 0
    assert not (output_dir / "audit").exists()


def test_translate_google_process_death_is_indeterminate_and_never_resent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class SimulatedProcessDeath(BaseException):
        pass

    class AcceptedThenProcessDeathClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise SimulatedProcessDeath

    client = AcceptedThenProcessDeathClient()

    def google_translator_factory() -> GoogleCloudNMTTranslator:
        return GoogleCloudNMTTranslator(
            project_id="gen-lang-client-0036391889",
            client=client,
            client_library_version="3.test",
            max_request_characters=8,
            max_run_characters=8,
        )

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        google_translator_factory,
    )
    monkeypatch.chdir(tmp_path)
    arguments = [
        "translate",
        "--languages",
        "vi",
        "--translator",
        "google-cloud-nmt-v3",
        "--cases-path",
        str(cases_path),
        "--output-dir",
        str(output_dir),
        "--languages-config",
        str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
    ]

    with pytest.raises(SimulatedProcessDeath):
        runner.invoke(app, arguments)

    reservations_path = output_dir / "audit" / "translation_reservations.jsonl"
    reservations = [
        json.loads(line) for line in reservations_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(reservations) == 1
    assert reservations[0]["source_character_count"] == len("Original")
    assert "Original" not in reservations_path.read_text(encoding="utf-8")

    second = runner.invoke(app, arguments)

    outcomes_path = output_dir / "audit" / "translation_reservation_outcomes.jsonl"
    outcomes = [json.loads(line) for line in outcomes_path.read_text(encoding="utf-8").splitlines()]
    failures = [
        json.loads(line)
        for line in (output_dir / "translation_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert second.exit_code == 0, second.output
    assert client.call_count == 1
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "indeterminate"
    assert outcomes[0]["reservation_id"] == reservations[0]["reservation_id"]
    assert outcomes[0]["charged_character_count"] == len("Original")
    assert failures[0]["error_type"] == "GoogleCloudIndeterminatePaidAttemptError"
    assert failures[0]["audit_reference"] == (
        f"translation_reservations.jsonl#{reservations[0]['reservation_id']}"
    )
    assert TranslationStore(output_dir).translations() == []


def test_translate_google_post_dispatch_timeout_is_indeterminate_and_never_resent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class TimeoutAfterDispatchClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise TimeoutError("C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL")

    client = TimeoutAfterDispatchClient()

    def google_translator_factory() -> GoogleCloudNMTTranslator:
        return GoogleCloudNMTTranslator(
            project_id="gen-lang-client-0036391889",
            client=client,
            client_library_version="3.test",
            max_request_characters=8,
            max_run_characters=16,
        )

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        google_translator_factory,
    )
    monkeypatch.chdir(tmp_path)
    arguments = [
        "translate",
        "--languages",
        "vi",
        "--translator",
        "google-cloud-nmt-v3",
        "--cases-path",
        str(cases_path),
        "--output-dir",
        str(output_dir),
        "--languages-config",
        str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
    ]

    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)

    outcomes = [
        json.loads(line)
        for line in (output_dir / "audit" / "translation_reservation_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    failure = json.loads((output_dir / "translation_failures.jsonl").read_text(encoding="utf-8"))
    assert first.exit_code == second.exit_code == 0
    assert client.call_count == 1
    assert [outcome["status"] for outcome in outcomes] == ["indeterminate"]
    assert failure["error_type"] == "GoogleCloudIndeterminatePaidAttemptError"
    serialized = json.dumps([outcomes, failure], sort_keys=True)
    for secret in ("C:/sensitive/adc.json", "PROMPT_SENTINEL", "KEY_SENTINEL"):
        assert secret not in serialized


def test_translate_google_rejects_forged_outcome_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class SimulatedProcessDeath(BaseException):
        pass

    class AcceptedThenProcessDeathClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise SimulatedProcessDeath

    client = AcceptedThenProcessDeathClient()

    def google_translator_factory() -> GoogleCloudNMTTranslator:
        return GoogleCloudNMTTranslator(
            project_id="gen-lang-client-0036391889",
            client=client,
            client_library_version="3.test",
            max_request_characters=8,
            max_run_characters=16,
        )

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        google_translator_factory,
    )
    monkeypatch.chdir(tmp_path)
    arguments = [
        "translate",
        "--languages",
        "vi",
        "--translator",
        "google-cloud-nmt-v3",
        "--cases-path",
        str(cases_path),
        "--output-dir",
        str(output_dir),
        "--languages-config",
        str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
    ]

    with pytest.raises(SimulatedProcessDeath):
        runner.invoke(app, arguments)

    reservation = json.loads(
        (output_dir / "audit" / "translation_reservations.jsonl").read_text(encoding="utf-8")
    )
    forged_outcome = {
        "outcome_id": "0" * 20,
        "reservation_id": reservation["reservation_id"],
        "task_key": reservation["task_key"],
        "status": "success",
        "charged_character_count": reservation["source_character_count"],
        "audit_reference": (f"translation_reservations.jsonl#{reservation['reservation_id']}"),
        "created_at": "2026-01-01T00:00:00Z",
        "translated_text": "forged translation",
        "provider_request_id": None,
        "error_type": None,
        "error_message": None,
    }
    (output_dir / "audit" / "translation_reservation_outcomes.jsonl").write_text(
        json.dumps(forged_outcome, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    second = runner.invoke(app, arguments)

    assert second.exit_code != 0
    assert isinstance(second.exception, PaidCallLedgerError)
    assert str(second.exception) == "paid-call outcome identity is invalid"
    assert client.call_count == 1
    assert TranslationStore(output_dir).translations() == []


def test_translate_google_replays_persisted_success_after_local_process_death(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class SimulatedLocalProcessDeath(BaseException):
        pass

    class SuccessfulClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            return SimpleNamespace(
                translations=[SimpleNamespace(translated_text="persisted translation")]
            )

    client = SuccessfulClient()

    def google_translator_factory() -> GoogleCloudNMTTranslator:
        return GoogleCloudNMTTranslator(
            project_id="gen-lang-client-0036391889",
            client=client,
            client_library_version="3.test",
        )

    original_add_provider_response = TranslationStore.add_provider_response
    should_crash = True

    def crash_after_persisted_outcome(
        store: TranslationStore,
        cache_key: str,
        text: str,
        provider_request_id: str | None,
    ) -> None:
        nonlocal should_crash
        if should_crash:
            should_crash = False
            raise SimulatedLocalProcessDeath
        original_add_provider_response(store, cache_key, text, provider_request_id)

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        google_translator_factory,
    )
    monkeypatch.setattr(TranslationStore, "add_provider_response", crash_after_persisted_outcome)
    monkeypatch.chdir(tmp_path)
    arguments = [
        "translate",
        "--languages",
        "vi",
        "--translator",
        "google-cloud-nmt-v3",
        "--cases-path",
        str(cases_path),
        "--output-dir",
        str(output_dir),
        "--languages-config",
        str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
    ]

    with pytest.raises(SimulatedLocalProcessDeath):
        runner.invoke(app, arguments)

    second = runner.invoke(app, arguments)

    outcomes = [
        json.loads(line)
        for line in (output_dir / "audit" / "translation_reservation_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert second.exit_code == 0, second.output
    assert client.call_count == 1
    assert [outcome["status"] for outcome in outcomes] == ["success"]
    assert (
        TranslationStore(output_dir).translations()[0].normalized_translated_text
        == "persisted translation"
    )


def test_translate_google_explicit_rejections_retry_with_distinct_bounded_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist([_case().model_dump(mode="json")]), cases_path)

    class InvalidArgument(RuntimeError):
        pass

    class RejectingClient:
        def __init__(self) -> None:
            self.call_count = 0

        def translate_text(self, *, request: dict[str, object]) -> object:
            del request
            self.call_count += 1
            raise InvalidArgument("C:/sensitive/adc.json PROMPT_SENTINEL KEY_SENTINEL")

    client = RejectingClient()

    def google_translator_factory() -> GoogleCloudNMTTranslator:
        return GoogleCloudNMTTranslator(
            project_id="gen-lang-client-0036391889",
            client=client,
            client_library_version="3.test",
            max_request_characters=16,
            max_run_characters=16,
        )

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        google_translator_factory,
    )
    monkeypatch.chdir(tmp_path)
    arguments = [
        "translate",
        "--languages",
        "vi",
        "--translator",
        "google-cloud-nmt-v3",
        "--cases-path",
        str(cases_path),
        "--output-dir",
        str(output_dir),
        "--languages-config",
        str(Path(__file__).parents[1] / "configs" / "languages.yaml"),
    ]

    results = [runner.invoke(app, arguments) for _ in range(3)]

    reservations = [
        json.loads(line)
        for line in (output_dir / "audit" / "translation_reservations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    outcomes = [
        json.loads(line)
        for line in (output_dir / "audit" / "translation_reservation_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    final_failure = json.loads(
        (output_dir / "translation_failures.jsonl").read_text(encoding="utf-8")
    )
    assert all(result.exit_code == 0 for result in results)
    assert client.call_count == 2
    assert len(reservations) == 2
    assert len({reservation["reservation_id"] for reservation in reservations}) == 2
    assert [outcome["status"] for outcome in outcomes] == ["failed", "failed"]
    assert final_failure["error_type"] == "GoogleCloudRunBudgetExceededError"
    assert final_failure["charged_character_count"] == 0
    serialized = json.dumps([reservations, outcomes, final_failure], sort_keys=True)
    for secret in ("C:/sensitive/adc.json", "PROMPT_SENTINEL", "KEY_SENTINEL", "Original"):
        assert secret not in serialized


def test_translate_google_missing_input_constructs_no_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_calls = 0

    def forbidden_google_translator() -> FakeTranslator:
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("missing input constructed a Google client")

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        forbidden_google_translator,
    )
    missing_cases = tmp_path / "missing.parquet"

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "vi",
            "--translator",
            "google-cloud-nmt-v3",
            "--cases-path",
            str(missing_cases),
            "--output-dir",
            str(tmp_path / "translated"),
        ],
    )

    assert result.exit_code != 0
    assert constructor_calls == 0
    assert not (tmp_path / "translated" / "audit").exists()


def test_translate_reports_overlong_nllb_input_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PartiallyOverlongTranslator(FakeTranslator):
        translator_id = "nllb"
        version = "facebook/nllb-200-distilled-600M"
        method = "nllb"

        def translate(
            self,
            text: str,
            source_language: str,
            target_language: str,
        ) -> ProviderTranslation:
            if text == "Too long":
                raise TranslationInputTooLongError(1046, 1024)
            return super().translate(text, source_language, target_language)

    cases_path = tmp_path / "cases.parquet"
    output_dir = tmp_path / "translated"
    translator = PartiallyOverlongTranslator(outputs={("Original", "my"): "translated"})
    import pyarrow as pa

    overlong = _case("case-overlong").model_copy(
        update={"source_text": "Too long", "canonical_payload": "Too long"}
    )
    pq.write_table(
        pa.Table.from_pylist([_case().model_dump(mode="json"), overlong.model_dump(mode="json")]),
        cases_path,
    )
    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.NLLBTranslator",
        lambda languages: translator,
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
    assert "failed 1" in result.output
    assert len(TranslationStore(output_dir).translations()) == 1
    failures = [
        json.loads(line)
        for line in (output_dir / "translation_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures == [
        {
            "case_id": "case-overlong",
            "error_code": "input_too_long",
            "max_input_tokens": 1024,
            "requires_manual_translation": True,
            "source_language": "en",
            "target_language": "my",
            "token_count": 1046,
            "translator_id": "nllb",
        }
    ]


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


def test_native_dataset_avoids_google_client_and_paid_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    constructor_calls = 0

    def forbidden_google_translator() -> FakeTranslator:
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("native translation constructed a Google client")

    monkeypatch.setattr(
        "crosslingual_safety.translation.commands.GoogleCloudNMTTranslator",
        forbidden_google_translator,
    )

    result = runner.invoke(
        app,
        [
            "translate",
            "--languages",
            "zh",
            "--translator",
            "google-cloud-nmt-v3",
            "--cases-path",
            str(cases_path),
            "--native-translations-path",
            str(native_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert constructor_calls == 0
    assert not (output_dir / "audit").exists()
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
