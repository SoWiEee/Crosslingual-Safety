# Native MultiJail and Dual PSA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox markers so execution can be resumed safely.

**Goal:** Run the final multilingual benchmark with dataset-native MultiJail translations where
available and two independently attributable Paper Summary Attack conditions based on the two
versioned PDFs in `refs/`.

**Architecture:** Keep `unified_run.py` as the orchestration facade, but move native translation
selection and PSA paper preparation into focused modules. Translation resolution is deterministic:
source text, native MultiJail text, deterministic Traditional Chinese conversion, then configured
provider fallback. PSA preparation creates one canonical English summary per PDF, translates the
summary to the selected languages with Google Cloud, and binds every source and derived hash into
the immutable run contract before victim requests begin.

**Tech Stack:** Python 3.11, Pydantic v2, PyArrow, pypdf, OpenCC, Google Cloud Translation v3,
pytest, Ruff, mypy, uv.

## Global Constraints

- Formal languages are `en,zh-tw,jv,my,th,vi,tl,eo`.
- Formal conditions are `none`, `psa_attack_poetry_v1`, and `psa_defense_r2d_v1`.
- `gra_v1` and `psa_static_v1` remain loadable for historical artifacts, but `--jailbreak all`
  selects only the three formal conditions.
- MultiJail native translations must never be silently replaced by machine translations.
- PDF summaries are generated once in English with `ais3/gemma-4-12b`; the seven non-English
  summary artifacts use the configured Google Cloud translator.
- A victim template contains the case payload exactly once.
- Preparation is fail-closed: missing PDFs, malformed extraction, incomplete translations, or
  mismatched hashes stop the run before any victim-model request.
- Tests use fakes and fixtures. They do not call paid APIs or remote models.
- Preserve unrelated worktree changes and generated run artifacts.

---

## Task 1: Add deterministic native MultiJail translation resolution

**Files:**
- Create: `src/crosslingual_safety/translation/bench.py`
- Modify: `src/crosslingual_safety/unified_run.py`
- Modify: `configs/run.yaml`
- Modify: `pyproject.toml`
- Test: `tests/test_translation.py`
- Test: `tests/test_unified_run.py`

- [ ] Add failing unit tests for loading `data/normalized/native_translations.parquet` by
  `(case_id, language)` and rejecting duplicate keys, blank text, mismatched source text, and
  unsupported provenance.
- [ ] Add a failing test for the exact resolution table:

  | Requested language | Resolution |
  | --- | --- |
  | `en` | case source text |
  | `zh-tw` | native `zh`, then OpenCC `s2twp` |
  | `jv`, `th`, `vi` | native MultiJail row |
  | `my`, `tl`, `eo` | configured provider fallback |

- [ ] Add `opencc-python-reimplemented` to project dependencies and
  `native_translations_path: data/normalized/native_translations.parquet` to `configs/run.yaml`.
- [ ] Implement immutable `BenchTranslation` and `BenchTranslationCatalog` types in
  `translation/bench.py`. Expose a resolver that returns text plus `method`, `provider`,
  `source_record_id`, `source_text_sha256`, `output_text_sha256`, and conversion metadata.
- [ ] Update `RunSettings` validation and the immutable run contract to include the native
  snapshot path/hash and OpenCC configuration.
- [ ] Update `_translate_cases` so bench cases consult the catalog before creating paid tasks.
  Only unresolved jobs may enter the Google Cloud reservation ledger or local NLLB provider.
- [ ] Update `RunPlan.translation_jobs` and dry-run output to count provider calls, not native or
  identity resolutions.
- [ ] Persist native rows in `audit/translations.jsonl` using the same stable identity checks as
  provider translations, while omitting paid-call fields.
- [ ] Run:

  ```sh
  uv run pytest tests/test_translation.py tests/test_unified_run.py -q
  uv run ruff check src tests
  uv run mypy src
  ```

- [ ] Commit: `Use native MultiJail translations first`

## Task 2: Extract and fingerprint the two PSA source papers

**Files:**
- Create: `src/crosslingual_safety/psa_papers.py`
- Create: `configs/psa_papers.yaml`
- Modify: `pyproject.toml`
- Add:
  `refs/Adversarial Poetry as a Universal Single-Turn.pdf`
- Add:
  `refs/Reasoning-to-Defend, Safety-Aware Reasoning.pdf`
- Test: `tests/test_psa_summary.py`

- [ ] Add `pypdf` to dependencies.
- [ ] Define two immutable paper specs in `configs/psa_papers.yaml`:
  `psa_attack_poetry_v1` and `psa_defense_r2d_v1`, each with title, condition ID, source path,
  expected PDF SHA-256, and summarizer model.
- [ ] Add failing tests for source-config validation, exact PDF hashes, deterministic extraction,
  page provenance, and chunks capped at 1,000 words.
- [ ] Implement `PsaPaperSpec`, `ExtractedPaper`, and `PaperChunk` Pydantic models.
- [ ] Implement section-aware PDF extraction. Normalize repeated whitespace without rewriting
  content; preserve page numbers and compute hashes for the PDF, normalized text, and each chunk.
- [ ] Reject encrypted, empty, unexpectedly hashed, or unextractable PDFs with a sanitized
  configuration error.
- [ ] Test extraction against small generated PDF fixtures and verify the two real PDFs through a
  metadata-only/hash test.
- [ ] Run:

  ```sh
  uv run pytest tests/test_psa_summary.py -q
  uv run ruff check src tests
  uv run mypy src
  ```

- [ ] Commit: `Add versioned PSA paper sources`

## Task 3: Build canonical summary and multilingual localization artifacts

**Files:**
- Create: `src/crosslingual_safety/psa_preparation.py`
- Modify: `src/crosslingual_safety/psa_summary.py`
- Modify: `src/crosslingual_safety/unified_run.py`
- Test: `tests/test_psa_summary.py`
- Test: `tests/test_unified_run.py`

- [ ] Replace the legacy three-field summary contract for new conditions with:
  `title`, `authors`, `paper_methods`, `mechanism_analysis`, and `related_work`.
- [ ] Add failing tests proving each selected PDF causes exactly one English summarizer request
  even when eight languages are selected.
- [ ] Add failing tests proving the seven localized artifacts are derived from the canonical
  English artifact, preserve all field keys, and record translator/provider contract hashes.
- [ ] Implement map-reduce summarization: summarize each 1,000-word chunk, then synthesize one
  canonical English JSON artifact using `ais3/gemma-4-12b`. Validate strict JSON, required keys,
  nonblank values, and paper identity.
- [ ] Implement hash-addressed preparation storage under
  `runs/_cache/psa/<condition_id>/<artifact_hash>/`. Store extraction manifest, chunk summaries,
  canonical summary, and one localization JSON per language.
- [ ] Use Google Cloud Translation v3 to localize summary field values for
  `zh-tw,jv,my,th,vi,tl,eo`. Do not translate identifiers or hashes.
- [ ] Bind PDF, extraction, chunk, summarizer contract, canonical summary, translator contract,
  localization, and template hashes into `RunPlan.contract`.
- [ ] Make preparation atomic. A run may reuse a fully matching cache, but partial or conflicting
  artifacts raise `ContractConflictError` before generation.
- [ ] Extend dry-run/stage output to report `paper_summaries` and `summary_localizations`
  separately from case translations.
- [ ] Preserve legacy `PaperSummaryService.from_method()` behavior for historical
  `psa_static_v1` artifacts.
- [ ] Run:

  ```sh
  uv run pytest tests/test_psa_summary.py tests/test_unified_run.py -q
  uv run ruff check src tests
  uv run mypy src
  ```

- [ ] Commit: `Cache multilingual PSA paper summaries`

## Task 4: Add the two formal PSA conditions

**Files:**
- Modify: `configs/jailbreaks.yaml`
- Modify: `src/crosslingual_safety/jailbreaks.py`
- Modify: `src/crosslingual_safety/unified_run.py`
- Modify: `src/crosslingual_safety/run_commands.py`
- Test: `tests/test_variants.py`
- Test: `tests/test_unified_run.py`

- [ ] Add locked prompt snapshot tests for `psa_attack_poetry_v1` and
  `psa_defense_r2d_v1` in all eight formal languages.
- [ ] Assert the user payload appears exactly once and target-language output is requested
  explicitly.
- [ ] Add both templates to `configs/jailbreaks.yaml`. Keep their wording structurally identical;
  only the prepared paper summary and condition identity differ.
- [ ] Generalize `PaperSummaryJailbreak` to accept the five-field summary contract and eight
  localized templates while retaining a legacy parser for `psa_static_v1`.
- [ ] Change public selection so exact formal IDs are accepted. Define `all` as:
  `none,psa_attack_poetry_v1,psa_defense_r2d_v1`.
- [ ] Retain explicit legacy aliases only for replaying older runs; do not include `gra` in
  `all`, CLI examples, or the formal experiment manifest.
- [ ] Ensure request/variant IDs distinguish source paper, summary localization, language,
  victim model, and case.
- [ ] Run:

  ```sh
  uv run pytest tests/test_variants.py tests/test_unified_run.py -q
  uv run ruff check src tests
  uv run mypy src
  ```

- [ ] Commit: `Add independent poetry and defense PSA conditions`

## Task 5: Update the operator workflow and reports

**Files:**
- Modify: `README.md`
- Modify: `docs/spec.md`
- Modify: `src/crosslingual_safety/reporting.py`
- Test: `tests/test_reporting.py`

- [ ] Add report tests showing separate sections and aggregate tables for `none`,
  `psa_attack_poetry_v1`, and `psa_defense_r2d_v1`.
- [ ] Include paper source title/hash and translation provenance in each condition report without
  duplicating full prompts in aggregate tables.
- [ ] Update `docs/spec.md` with the formal language matrix, native/fallback resolution policy,
  active condition set, and immutable PSA preparation contract.
- [ ] Rewrite README Getting Started commands for the final workflow:

  ```sh
  uv sync --all-groups --extra translation-google --extra evaluation-local
  uv run crosslingual-safety run --source bench \
    --language en,zh-tw,jv,my,th,vi,tl,eo \
    --jailbreak all --model llama --translator google
  uv run crosslingual-safety evaluate --run-dir runs/<run-id>
  ```

- [ ] Document that MultiJail supplies `en,zh,jv,th,vi`, `zh-tw` is deterministic `s2twp`, and
  `my,tl,eo` use Google Cloud fallback.
- [ ] Document both source PDFs as separate experimental conditions, the summarize-once/cache
  behavior, preparation artifact paths, resumability, and how hashes protect comparability.
- [ ] Mark GRA and `psa_static_v1` as legacy replay-only methods.
- [ ] Document report locations: root `report.md` index, condition reports, evaluation artifacts,
  bypass labels, dual-judge disagreement, and manual-review queue.
- [ ] Run:

  ```sh
  uv run pytest tests/test_reporting.py -q
  uv run ruff format src tests
  uv run ruff check src tests
  uv run mypy src
  ```

- [ ] Commit: `Document final multilingual PSA workflow`

## Task 6: Verify the complete pipeline before spending the remote budget

**Files:**
- Test: `tests/test_unified_run.py`
- Test: `tests/test_evaluation_commands.py`
- Test: `tests/test_reporting.py`

- [ ] Add an end-to-end fake-provider test with two MultiJail cases, eight languages, three formal
  conditions, and one victim model. Assert native/GCP resolution counts, two summarizer calls,
  fourteen summary-localization groups, forty-eight victim requests, and three reports.
- [ ] Verify fail-closed behavior by corrupting one cached localization and asserting zero victim
  calls.
- [ ] Run the complete offline quality gate:

  ```sh
  uv run pytest -q
  uv run ruff format --check src tests
  uv run ruff check src tests
  uv run mypy src
  ```

- [ ] Run a no-cost dry run and inspect the printed counts:

  ```sh
  uv run crosslingual-safety run --source bench \
    --language en,zh-tw,jv,my,th,vi,tl,eo \
    --jailbreak all --model llama --translator google --dry-run
  ```

- [ ] Run one explicitly authorized live smoke case using the smallest supported bench selection.
  Confirm both PSA caches, translations, victim responses, evaluation artifacts, and hierarchical
  reports are readable before starting the full experiment.
- [ ] Commit: `Verify final multilingual experiment pipeline`

