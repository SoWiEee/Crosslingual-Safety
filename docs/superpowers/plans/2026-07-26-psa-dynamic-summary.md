# PSA Dynamic Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `psa_static_v1` manual flow summarize the GRA paper with `ais3/gemma-4-12b` before sending the final PSA prompt to the five victim models.

**Architecture:** Add a small pre-victim summary service around the existing OpenAI-compatible chat provider. It renders one localized summary request per language, persists immutable summary artifacts, and passes the cached text into `PaperSummaryJailbreak`; the existing generation queue remains responsible only for victim requests.

**Tech Stack:** Python 3.11, httpx, Pydantic, PyYAML, Typer, pytest, uv

## Global Constraints

- Keep attack ID `psa_static_v1` for CLI and artifact compatibility.
- Use summary model `ais3/gemma-4-12b` and the same `ZOOLAB_BASE_URL`/`ZOOLAB_API_KEY` endpoint.
- Support `en`, `zh`, `vi`, and `my` summaries.
- Persist no API key or raw authorization header.
- Abort before victim generation if any required summary fails.
- Preserve existing static YAML sections as the low-level fallback/reference corpus.
- Send the English `paper_summaries.gra_attack_summary_v1.sections` snapshot as the only summary source corpus; do not have the remote model fetch the PDF.
- Compute the source hash as SHA-256 over sorted UTF-8 JSON of the source sections plus shared provenance.
- Store the literal summary system/user templates in YAML under `psa_static_v1.summary_prompt`; canonicalize the request object with `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
- Use the fixed language mapping `en=English`, `zh=Traditional Chinese`, `vi=Vietnamese`, `my=Burmese`; include the resolved language name in the canonical request object and hash.

---

### Task 1: Dynamic Summary Service and PSA Renderer

**Files:**
- Create: `src/crosslingual_safety/psa_summary.py`
- Modify: `src/crosslingual_safety/jailbreaks.py`
- Modify: `configs/jailbreaks.yaml`
- Test: `tests/test_psa_summary.py`
- Modify: `tests/test_variants.py`

**Interfaces:**
- `PaperSummaryService.summarize(summary_id: str, language: str) -> SummaryArtifact`
- `SummaryArtifact` fields: `summary_id`, `language`, `source_sha256`, `request_sha256`, `provider_id`, `model_id`, `endpoint_type`, `generation_config`, `response_text`, `response_sha256`, `provider_request_id`, `created_at`.
- `PaperSummaryJailbreak.render(payload: str, context: JailbreakContext, summary_sections: Mapping[str, str] | None = None, summary_artifact: SummaryArtifact | None = None) -> JailbreakResult`

- [ ] Add fake-transport tests for four canonical summary payloads, strict JSON parsing, deterministic artifacts, cache reuse, and dynamic insertion.
- [ ] Implement the service with the existing OpenAI-compatible chat payload shape, model `ais3/gemma-4-12b`, `max_tokens=2048`, `temperature=0`, safe response parsing, SHA-256 cache keys, and immutable JSON artifacts.
- [ ] Require exactly the JSON keys `attack_methods`, `mechanism_analysis`, and `related_work`; reject markdown fences, missing keys, duplicate language rows, and empty values.
- [ ] Extend PSA rendering metadata/hash to include the selected summary sections and summary model contract when an override is supplied; dynamic metadata uses `llm_generated` and `translation_provenance=llm_translation`.
- [ ] Make the three generated keys replace only static `attack_methods`, `mechanism_analysis`, and `related_work`; title, author, attack-scenario boundary, and both payload positions remain static.
- [ ] Enforce renderer invariants: dynamic `summary_sections` and `summary_artifact` must be supplied together, artifact language must equal wrapper language, and parsing the artifact response must reproduce the three supplied sections; otherwise use static fallback with both omitted.
- [ ] Keep direct render fallback behavior using static configured sections.

### Task 2: Manual Preflight Integration

**Files:**
- Modify: `src/crosslingual_safety/manual_commands.py`
- Modify: `src/crosslingual_safety/manual.py`
- Test: `tests/test_manual.py`

**Interfaces:**
- `manual-run` creates `summary_artifacts.jsonl` before `variants.jsonl` and victim jobs.

- [ ] Add tests proving four summary calls happen before victim calls, a failed summary creates no victim request or `variants.jsonl`/`jobs.sqlite`, and a second exact-contract run makes zero summary calls.
- [ ] Add summary model/provider/source/request hashes to the run fingerprint. Keep generated response hashes out of the pre-request run ID to avoid circularity; record them in `summary_artifacts.jsonl`, manifest hashes, renderer metadata, template hashes, and variant IDs.
- [ ] Load an immutable cache only when it contains exactly four unique language rows with matching contract; reject partial/mismatched caches without overwriting them.
- [ ] Generate all four summaries once, select by `context.wrapper_language`, pass structured sections into variant construction, and keep `role: null` for PSA.
- [ ] Use the existing OpenAI-compatible provider's typed statuses; authentication aborts safely, while any non-success or malformed summary aborts without blind retry. Validate only that each artifact row's declared language matches its requested language; do not infer linguistic quality with a script heuristic.

### Task 3: Documentation and Verification

**Files:**
- Modify: `docs/spec.md`
- Modify: `README.md`

- [ ] Document that `psa_static_v1` now performs a pre-victim LLM summary stage and list the cache artifact.
- [ ] Document the shared endpoint and `ais3/gemma-4-12b` requirement without exposing credentials.
- [ ] Run focused tests, full pytest, Ruff, and mypy.
- [ ] Before live smoke, require non-empty `ZOOLAB_BASE_URL` and `ZOOLAB_API_KEY`; on authentication/configuration failure stop without retry and report the run as externally blocked.
- [ ] Automated tests cover both wrapper modes; run one live four-language five-victim smoke test with `--wrapper-language-mode same-as-payload` and verify four summary artifacts precede 20 victim results.
