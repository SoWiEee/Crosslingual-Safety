# Low-Resource Wrapper Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one manual prompt through GRA and PSA in seven requested languages across five
remote models.

**Architecture:** Extend the unified runner's public language registry while keeping reviewed
localized templates unchanged. New low-resource languages select the English wrapper, replace its
English-only output instruction with an explicit target-language instruction, and record the
fallback in immutable variant metadata.

**Tech Stack:** Python 3.11, Typer, Pydantic, pytest, Google Cloud Translation v3, ZooLab
OpenAI-compatible API.

## Global Constraints

- Keep `en`, `zh-tw`, `vi`, and `my` localized rendering unchanged.
- Use English GRA/PSA wrapper content for `jv`, `th`, `id`, and `tl`.
- Translate payloads with `google-cloud-nmt-v3` for the formal experiment.
- Do not expand PSA summary generation beyond `en`, `zh`, `vi`, and `my`.
- Preserve deterministic IDs, hashes, sanitized failures, and resumability.

---

### Task 1: Public Language And Fallback Contract

**Files:**
- Modify: `src/crosslingual_safety/unified_run.py`
- Test: `tests/test_unified_run.py`

**Interfaces:**
- Produces: expanded `PUBLIC_LANGUAGES`
- Produces: deterministic English-wrapper fallback rendering in `_render_variants`

- [ ] **Step 1: Write failing parsing and rendering tests**

Add tests asserting that `PUBLIC_LANGUAGES` contains
`("en", "zh-tw", "jv", "my", "th", "vi", "id", "tl")`, that Javanese GRA replaces the
English-only JSON language sentence, and that Tagalog PSA replaces `Requested output language:
English.`. Assert fallback metadata and `mixed_language`.

- [ ] **Step 2: Run tests to verify failure**

Run:
`uv run pytest tests/test_unified_run.py -q -k "low_resource or parse_selection"`

Expected: the new languages are rejected or missing from wrapper mappings.

- [ ] **Step 3: Implement minimal fallback**

Add target-language names and fallback wrapper mappings. For fallback variants, replace the exact
English output instruction, derive a new SHA-256 template hash, merge canonical fallback metadata,
and classify the variant as mixed-language. Raise a sanitized rendering error if the expected
versioned instruction is absent.

- [ ] **Step 4: Run focused tests**

Run:
`uv run pytest tests/test_unified_run.py -q -k "low_resource or parse_selection"`

Expected: all selected tests pass.

### Task 2: Documentation And Regression Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/spec.md`
- Test: `tests/test_unified_run.py`

**Interfaces:**
- Consumes: expanded public language and fallback contract from Task 1

- [ ] **Step 1: Update user-facing language documentation**

Document all public language codes and explain that `jv`, `th`, `id`, and `tl` use an English
attack wrapper with target-language output instructions.

- [ ] **Step 2: Run complete offline verification**

Run:
`uv run pytest -q -m "not live_google" --basetemp .pytest-tmp-low-resource`

Run:
`uv run ruff format src tests`

Run:
`uv run ruff check src tests`

Run:
`uv run mypy src`

Expected: all checks pass.

### Task 3: Formal Google And Victim-Model Run

**Files:**
- Modify: `configs/run.yaml`
- Output: `runs/experiments/<run-id>/`

**Interfaces:**
- Consumes: `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `ZOOLAB_BASE_URL`, and
  `ZOOLAB_API_KEY` from the local environment

- [ ] **Step 1: Select Google Cloud Translation**

Set `translator: google-cloud-nmt-v3` in `configs/run.yaml`.

- [ ] **Step 2: Verify deterministic dry-run counts**

Run:
`uv run crosslingual-safety run --source manual --language zh-tw,jv,my,th,vi,id,tl --jailbreak gra,psa --dry-run`

Expected: `cases=1 translations=6 psa_summaries=4 victim_requests=70`.

- [ ] **Step 3: Execute the formal run**

Run the same command without `--dry-run`. Completion requires a terminal parent manifest and
child reports for both GRA and PSA. Provider failures remain recorded rather than discarded.

- [ ] **Step 4: Summarize artifacts**

Report the run ID, path, translation status by language, model request status totals, and locations
of report and JSONL result files without reproducing harmful response text.

