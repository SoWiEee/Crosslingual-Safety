import json
from pathlib import Path

import httpx
import pytest

from crosslingual_safety.jailbreaks import JailbreakContext, load_jailbreaks
from crosslingual_safety.psa_summary import (
    SUMMARY_KEYS,
    SUMMARY_LANGUAGES,
    PaperSummaryService,
    SummaryError,
    normalize_summary_response,
    parse_summary_response,
    sha256_text,
)


def _method():
    return load_jailbreaks(Path("configs/jailbreaks.yaml"))["psa_static_v1"]


def _service(requests: list[dict[str, object]]) -> PaperSummaryService:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        language_names = {
            "en": "English",
            "zh": "Traditional Chinese",
            "vi": "Vietnamese",
            "my": "Burmese",
            "eo": "Esperanto",
        }
        language = next(
            language
            for language in SUMMARY_LANGUAGES
            if f"into {language_names[language]}" in str(body["messages"][-1]["content"])
        )
        response_json = json.dumps(
            {key: f"{key}-{language}" for key in SUMMARY_KEYS},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        response = f"```json\n{response_json}\n```"
        return httpx.Response(
            200,
            headers={"x-request-id": f"req-{language}"},
            json={
                "id": f"id-{language}",
                "model": "ais3/gemma-4-12b",
                "choices": [{"message": {"content": response}, "finish_reason": "stop"}],
            },
        )

    return PaperSummaryService.from_method(
        _method(),
        base_url="https://summary.test/v1",
        api_key="secret-value",
        transport=httpx.MockTransport(handler),
        clock=lambda: "2026-07-26T00:00:00Z",
    )


def test_five_language_summary_contract_and_cache_reuse(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []
    service = _service(requests)
    artifacts = {
        language: service.summarize(service.summary_id, language) for language in SUMMARY_LANGUAGES
    }

    assert service.timeout_seconds == 180.0
    assert service.generation_config["timeout_seconds"] == 180.0
    assert all(
        artifact.generation_config["timeout_seconds"] == 180.0 for artifact in artifacts.values()
    )
    assert len(requests) == 5
    assert [request["model"] for request in requests] == ["ais3/gemma-4-12b"] * 5
    assert all(request["temperature"] == 0.0 for request in requests)
    assert all(request["max_tokens"] == 2048 for request in requests)
    assert all("secret-value" not in json.dumps(request) for request in requests)
    assert all(artifact.response_text.startswith("{") for artifact in artifacts.values())
    assert all("```" not in artifact.response_text for artifact in artifacts.values())
    cache_path = tmp_path / "summary_artifacts.jsonl"
    service.write_cache(cache_path, artifacts)
    loaded = service.load_cache(cache_path)
    assert loaded == artifacts
    assert len(service.validate_artifacts(list(loaded.values()))) == 5


@pytest.mark.parametrize(
    "response",
    [
        'prose\n```json\n{"attack_methods":"a","mechanism_analysis":"m","related_work":"r"}\n```',
        '```json\n{"attack_methods":"a","mechanism_analysis":"m","related_work":"r"}',
        '```json\n{"attack_methods":"a","mechanism_analysis":"m","related_work":"r"}\n```\nprose',
        '{"attack_methods":"a","mechanism_analysis":"m"}',
        '{"attack_methods":"a","mechanism_analysis":"m","related_work":"r","extra":"x"}',
        '{"attack_methods":"a","mechanism_analysis":"m","related_work":""}',
        '{"attack_methods":"a","attack_methods":"b","mechanism_analysis":"m","related_work":"r"}',
    ],
)
def test_summary_response_is_strict_json(response: str) -> None:
    with pytest.raises(SummaryError):
        parse_summary_response(response)


def test_json_markdown_fence_is_normalized_for_artifacts() -> None:
    fenced = '```json\n{"related_work":"r","attack_methods":"a","mechanism_analysis":"m"}\n```'
    normalized, sections = normalize_summary_response(fenced)
    assert sections == {
        "attack_methods": "a",
        "mechanism_analysis": "m",
        "related_work": "r",
    }
    assert normalized == '{"attack_methods":"a","mechanism_analysis":"m","related_work":"r"}'


def test_dynamic_renderer_replaces_only_generated_sections() -> None:
    requests: list[dict[str, object]] = []
    service = _service(requests)
    artifact = service.summarize(service.summary_id, "zh")
    method = _method()
    sections = parse_summary_response(artifact.response_text)
    result = method.render(
        "PAYLOAD",
        JailbreakContext("zh", "zh", "harmful", None),
        summary_sections=sections,
        summary_artifact=artifact,
    )
    assert "title" not in result.rendered_prompt
    assert "PAYLOAD" in result.rendered_prompt
    assert "attack_methods-zh" in result.rendered_prompt
    metadata = json.loads(result.metadata_json)
    assert metadata["summary_method"] == "llm_generated"
    assert metadata["translation_provenance"] == "llm_translation"
    assert metadata["summary_artifact"]["response_sha256"] == sha256_text(artifact.response_text)


def test_cache_rejects_duplicate_language(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []
    service = _service(requests)
    artifacts = [service.summarize(service.summary_id, language) for language in SUMMARY_LANGUAGES]
    duplicate = artifacts[:-1] + [artifacts[0]]
    with pytest.raises(SummaryError, match="duplicate"):
        service.validate_artifacts(duplicate)
