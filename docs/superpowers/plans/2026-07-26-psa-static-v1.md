# PSA Static V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a four-language `psa_static_v1` jailbreak wrapper that faithfully preserves the Paper Summary Attack prompt structure while using a versioned, human-authored GRA paper summary.

**Architecture:** A dedicated `PaperSummaryJailbreak` loads localized static paper sections and prompt templates from `configs/jailbreaks.yaml`, inserts the runtime payload, and records source and summary provenance in deterministic metadata. The existing variant and single-turn manual CLI paths consume the new method through the jailbreak registry; non-role attacks omit the irrelevant GRA role from artifacts and reports.

**Tech Stack:** Python 3.11, PyYAML, Pydantic, Typer, pytest, Ruff, mypy, uv

## Global Constraints

- Preserve the official PSA `attack.py` section order: title, author, attack-method summary, mechanism analysis, attack-scenario example, and related work.
- Treat Attack Scenario Example as one logical insertion boundary; match the official skeleton's two `${payload}` references inside that section.
- Identify this method as a static, human-authored adaptation; do not claim runtime GPT-4o paper summarization.
- Support wrapper languages `en`, `zh`, `vi`, and `my`, each with an explicit output-language instruction.
- Use `refs/GRA_Jailbreak.pdf` as the summarized source and `refs/Paper_Summary_Attacks.pdf` as the method reference.
- Do not expose `.env` values, API keys, or remote response content in tests.
- Do not revert existing uncommitted work or create a commit unless the user explicitly requests one.

---

### Task 1: Paper Summary Jailbreak Contract

**Files:**
- Modify: `tests/test_variants.py`
- Modify: `src/crosslingual_safety/jailbreaks.py`
- Modify: `configs/jailbreaks.yaml`

**Interfaces:**
- Consumes: `JailbreakMethod.render(payload: str, context: JailbreakContext) -> JailbreakResult`
- Produces: `PaperSummaryJailbreak`, registered as `psa_static_v1`, with deterministic template hashing and provenance metadata

- [ ] **Step 1: Add failing contract and snapshot tests**

Add tests that assert `psa_static_v1` loads, supports exactly `en/zh/vi/my`, preserves the required section ordering, has one logical Attack Scenario Example insertion with two payload references, emits a locked full rendered raw snapshot for every language, and returns metadata containing:

```python
{
    "summary_id": "gra_attack_summary_v1",
    "source_ref": "refs/GRA_Jailbreak.pdf",
    "source_doi": "10.1109/LSP.2026.3677330",
    "psa_reference": "refs/Paper_Summary_Attacks.pdf",
    "section_order": [
        "title",
        "author",
        "attack_methods",
        "mechanism_analysis",
        "attack_scenario_example",
        "related_work",
    ],
    "insertion_index": "attack_scenario_example",
    "payload_occurrences": 2,
    "source_language": "en",
    "summary_language": "<wrapper language>",
    "summary_method": (
        "human_authored_from_source"
        if wrapper_language == "en"
        else "human_translated_from_english_summary"
    ),
}
```

The metadata test must also assert `translation_provenance` is `"none"` for English and `"human_translation"` for `zh`, `vi`, and `my`.

- [ ] **Step 2: Run focused tests and confirm the red state**

Run: `uv run pytest tests/test_variants.py -q`

Expected: failures because `psa_static_v1` is not registered.

- [ ] **Step 3: Implement the dedicated loader and renderer**

Add a `PaperSummaryJailbreak` that validates localized sections at construction, substitutes `${payload}` plus localized section fields with `string.Template`, and serializes metadata with sorted JSON keys. Its `template_sha256` input is the UTF-8 SHA-256 of this canonical JSON object:

```python
{
    "template": localized_template,
    "sections": localized_sections,
    "provenance": shared_provenance,
    "summary_id": summary_id,
}
```

Canonical JSON uses `ensure_ascii=False`, `sort_keys=True`, and `separators=(",", ":")`.

- [ ] **Step 4: Add the four localized static templates**

Define `gra_attack_summary_v1` and the `psa_static_v1` entry in YAML. Each localized template must name its requested output language and keep the same headings and scenario-example continuation contract. Correct the research note's “insert once” wording to distinguish one logical insertion boundary from the official skeleton's two payload references.

- [ ] **Step 5: Prove identity changes when source material changes**

In a temporary config fixture, mutate one localized summary field and one provenance field separately. Assert each mutation changes `template_sha256`; build a variant and manual-run fingerprint fixture to assert it also changes `variant_id` and the derived manual run ID.

- [ ] **Step 6: Run focused quality checks**

Run:

```powershell
uv run pytest tests/test_variants.py -q
uv run ruff check src/crosslingual_safety/jailbreaks.py tests/test_variants.py
uv run mypy src
```

Expected: all pass.

### Task 2: Manual CLI and Documentation

**Files:**
- Modify: `tests/test_manual.py`
- Modify: `src/crosslingual_safety/manual.py`
- Modify: `src/crosslingual_safety/manual_commands.py`
- Modify: `docs/spec.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: `psa_static_v1` through `load_jailbreaks`
- Produces: manual variants whose `role` is `None` for non-GRA attacks and reports that label those rows with the attack ID

- [ ] **Step 1: Add failing manual artifact tests**

Test that a PSA manual variant has `role is None`, its result row serializes `"role": null`, its manifest contract serializes `"role": null`, and its Markdown heading uses `psa_static_v1` rather than `joker`. Assert changing CLI `--role` for PSA does not change the derived run ID. Retain existing assertions that `gra_v1` validates, resolves, fingerprints, and reports the selected role.

- [ ] **Step 2: Run focused tests and confirm the red state**

Run: `uv run pytest tests/test_manual.py -q`

Expected: failures because manual variants currently always resolve a role.

- [ ] **Step 3: Make role semantics attack-specific**

Change `ManualVariant.role` to `ManualRole | None`; resolve and fingerprint a role only when `jailbreak_id == "gra_v1"`. For non-GRA attacks, `contract.role` and result-row `role` are JSON `null`, a valid CLI or JSONL role is semantically ignored, and report headings use `attack_id`. Preserve the raw input snapshot in run identity, so changing a JSONL input record still changes identity even when its role field is not applied to PSA.

- [ ] **Step 4: Document the contract and commands**

Add the PSA static fidelity boundary and provenance fields to `docs/spec.md`. Add a Getting Started example:

```powershell
uv run crosslingual-safety manual-run prompts\prompt.txt --source-language zh --jailbreak psa_static_v1 --wrapper-language-mode same-as-payload
```

State that `--role` applies to `gra_v1`, while PSA uses localized static GRA paper sections and performs no runtime summarization.

- [ ] **Step 5: Run focused and full local verification**

Run:

```powershell
uv run pytest tests/test_manual.py tests/test_variants.py -q
uv run pytest -q
uv run ruff check .
uv run mypy src
```

Expected: all checks pass.

### Task 3: Five-Model Manual Smoke Test

**Files:**
- Generate: `runs/manual/<run-id>/run_manifest.json`
- Generate: `runs/manual/<run-id>/results.jsonl`
- Generate: `runs/manual/<run-id>/report.md`

**Interfaces:**
- Consumes: `prompts/prompt.txt`, local translation artifacts, `.env` API configuration, and the default five-model set
- Produces: a complete manual run report with one PSA response per payload language and model

- [ ] **Step 1: Execute the remote batch**

Preflight without printing secrets: load `.env`, require non-empty `ZOOLAB_BASE_URL` and `ZOOLAB_API_KEY`, import the installed NLLB/Transformers dependencies, verify the local NLLB model is available, and confirm `torch.cuda.is_available()` is true. Confirm `prompts/prompt.txt` resolves to exactly one prompt before expecting 20 jobs.

Run:

```powershell
uv run crosslingual-safety manual-run prompts\prompt.txt --source-language zh --jailbreak psa_static_v1 --wrapper-language-mode same-as-payload
```

Expected: `runs/manual/<run-id>` containing `run_manifest.json`, JSONL results, and Markdown report.

- [ ] **Step 2: Validate artifact completeness**

Require `planned_jobs == 20` and exactly 20 unique terminal `(prompt, language, model)` tuples for one prompt × four languages × five default models. Check that each successful row includes the final model response, failed rows include transport error details, and report headings show `psa_static_v1` rather than a role.

- [ ] **Step 3: Assess multilingual behavior**

Record per-language transport success, whether the response follows the requested output language, and whether it refuses or complies. Treat transport success separately from jailbreak effectiveness.

- [ ] **Step 4: Apply the bounded retry rule**

If any tuple is missing or has a retryable transport failure, resume the same run at most two times so the queue retries failed work without duplicating successful rows. After two resumes, stop and report the remaining tuple/error as a blocker; do not claim the remote verification passed.
