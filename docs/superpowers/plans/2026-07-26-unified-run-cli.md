# Unified Run CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `implement` to execute this plan task-by-task with tests.

**Goal:** Add one stable `run` command that covers manual and benchmark experiments with only source, language, jailbreak, and dry-run choices while producing compact public results and complete audit artifacts.

**Architecture:** A new facade module will translate the four public options into existing ingestion, translation, jailbreak, and generation services. Versioned `configs/run.yaml` will hold fixed operational defaults. Each selected jailbreak runs as an isolated child under one deterministic parent run; a projector writes the clean public `results.jsonl`, while detailed prompts, provider metadata, and provenance remain under `audit/` and child directories.

**Tech Stack:** Python 3.11, Typer, Pydantic, PyArrow, pytest, Ruff, mypy, existing NLLB and OpenAI-compatible provider adapters.

## Global Constraints

- Preserve `manual-run`, translation, jailbreak, and generation commands as advanced interfaces.
- Expose exactly `--source`, `--language`, `--jailbreak`, and `--dry-run` on `run`.
- Keep `zh` internally for existing wrapper templates, but expose only `zh-tw`; translate it with `zho_Hant`.
- Never print prompts, responses, credentials, or provider payloads to the terminal.
- A dry run must not load `.env`, initialize NLLB/CUDA, call providers, or create a run directory.
- Preserve unrelated working-tree changes, especially the existing `README.md` modification.
- Do not commit implementation changes unless the user explicitly requests a commit.

---

## Task 1: Lock the Public CLI and Planning Contract

**Files:**
- Create: `configs/run.yaml`
- Create: `src/crosslingual_safety/unified_run.py`
- Create: `src/crosslingual_safety/run_commands.py`
- Modify: `configs/languages.yaml`
- Modify: `src/crosslingual_safety/cli.py`
- Test: `tests/test_unified_run.py`

- [ ] **Step 1: Write failing option and parsing tests**

Test `run --help` exposes the four intended options and no model, role, path, translator, source-language, or generation tuning options. Parameterize comma-separated values, `all`, whitespace, duplicates, unknown values, `zh` rejection, and `zh-tw` acceptance.

Run:

```sh
uv run pytest tests/test_unified_run.py -q
```

Expected: FAIL because `run` and its selection parser do not exist.

- [ ] **Step 2: Define fixed configuration**

Add `configs/run.yaml` with:

```yaml
version: 1
manual:
  input_path: prompts/prompt.txt
  source_language: zh-tw
bench:
  cases_path: data/normalized/cases.parquet
  selection_path: data/normalized/variant_case_selection.parquet
models: [llama31_8b, gemma_4_12b, gemma_4_26b, nemotron_cascade_2_30b, llama33_70b]
translator: nllb
wrapper_language_mode: same-as-payload
gra_role: joker
models_config: configs/models.yaml
languages_config: configs/languages.yaml
jailbreaks_config: configs/jailbreaks.yaml
runs_dir: runs/experiments
```

Add `zh-tw` to `configs/languages.yaml` with NLLB code `zho_Hant`; retain legacy `zh`.

- [ ] **Step 3: Implement public request and plan types**

In `unified_run.py`, add:

```python
PUBLIC_LANGUAGES = ("en", "zh-tw", "vi", "my")
PUBLIC_JAILBREAKS = ("none", "gra", "psa")
WRAPPER_LANGUAGES = {"en": "en", "zh-tw": "zh", "vi": "vi", "my": "my"}
ATTACK_IDS = {"none": "none", "gra": "gra_v1", "psa": "psa_static_v1"}

class RunRequest(BaseModel): ...
class RunSettings(BaseModel): ...
class UnifiedCase(BaseModel): ...
class RunPlan(BaseModel): ...

def parse_selection(value: str, allowed: tuple[str, ...], option_name: str) -> tuple[str, ...]: ...
def load_run_settings(path: Path = Path("configs/run.yaml")) -> RunSettings: ...
def plan_run(request: RunRequest, settings: RunSettings) -> RunPlan: ...
```

`RunPlan` must contain selected cases/models, translation count, PSA summary count, victim request count, deterministic `run_id`, and the prospective parent path. PSA plans count four summary calls because the current summary cache contract is four-language and atomic.

The facade uses its own public records instead of passing `zh-tw` into the legacy
`ManualPrompt`/`ManualTranslation` models. Preserve `zh-tw` as the source and target language in
input, translation, result, and audit rows. Apply `WRAPPER_LANGUAGES["zh-tw"] == "zh"` only when
constructing `JailbreakContext` and resolving summary/template text. Write an identity translation
record when the requested target equals the case source, but count only non-identity NLLB calls as
`translation_jobs`.

- [ ] **Step 4: Register the facade command**

Implement `register_run_commands(app)` and call it from `cli.py`. The command defaults are:

```text
--source manual
--language all
--jailbreak none
--dry-run false
```

The command prints only cases, translations, PSA summaries, victim requests, run ID, and path. Invalid selections use `typer.BadParameter`.

- [ ] **Step 5: Verify planning behavior**

Add fixtures for a TXT manual input and Parquet benchmark cases/selection. Assert benchmark loading follows only `selected_case_id`, manual `zh-tw` creates an identity record plus the requested non-identity translations, and run IDs are stable. Snapshot the temporary workspace before and after `--dry-run`; assert no file changes and no `.env` loader, NLLB/CUDA/checkpoint constructor, summary/provider factory, queue, or provider call.

Run:

```sh
uv run pytest tests/test_unified_run.py -q
uv run ruff check src/crosslingual_safety/unified_run.py src/crosslingual_safety/run_commands.py tests/test_unified_run.py
uv run mypy src
```

Expected: PASS.

## Task 2: Build the Isolated Parent/Child Executor

**Files:**
- Modify: `src/crosslingual_safety/unified_run.py`
- Modify: `src/crosslingual_safety/generation/commands.py`
- Test: `tests/test_unified_run.py`

- [ ] **Step 1: Write failing execution-contract tests**

Use injected fake translator, summary service, and generation function. Cover:

- one case, two languages, two jailbreaks, and five fixed models;
- `zh-tw` translation with internal wrapper language `zh`;
- fixed GRA role `joker`;
- resumable deterministic parent and child IDs;
- one failed child preserving successful sibling rows;
- no prompt, response, or secret in captured terminal output.
- selected-plan preflight failure before any translation, summary, or victim provider call;
- a child failing after one model succeeds without replacing that concrete successful row;
- retryable work runs again on resume while success, provider-blocked, and permanent failures do not.
- one shared translation tuple failing while other languages and all unaffected child tuples continue;
- PSA summary failure after a partial provider sequence leaves no cache or victim request.

Run:

```sh
uv run pytest tests/test_unified_run.py -q
```

Expected: FAIL because execution is not implemented.

- [ ] **Step 2: Expose the existing generation loop as a service**

Rename `_generate_pending` in `generation/commands.py` to `generate_pending`, update internal callers, and keep `_generate_pending = generate_pending` temporarily for compatible imports. Do not change retry, queue, provider, or Parquet behavior.

- [ ] **Step 3: Implement preparation and audit snapshots**

Add an injectable `RunDependencies` and `execute_run(plan, settings, dependencies)` boundary.
Before creating providers or making any call, run `preflight_run(plan, settings, dependencies)`.
Validate every selected input and configuration together: model keys and endpoint metadata,
jailbreak methods/templates and wrapper languages, GRA role, PSA summary source/prompt/model,
required victim and summary environment variables, and CUDA/local NLLB checkpoint availability
when `translation_jobs > 0`. Configuration failures abort the parent before all translation,
summary, and victim calls; provider/runtime failures after preflight remain child-local.

Normalize manual and benchmark inputs into `UnifiedCase`. Translate only targets different from
the public source language and emit identity records for matching targets. Persist immutable or
contract-checked files:

```text
audit/input_snapshot.jsonl
audit/translations.jsonl
audit/translation_attempts.jsonl
audit/psa_summary_artifacts.jsonl
audit/result_index.jsonl
children/<jailbreak>/variants.jsonl
children/<jailbreak>/generation_results.parquet
```

Each audit index row maps `(case_id, source, language, jailbreak, model)` to nullable
`variant_id` and `generation_run_id` plus a required `audit_record_type` and `audit_record_id`.
Concrete results reference their detailed child variant/generation row. A synthesized translation
failure leaves both IDs null and references its translation-attempt row; a synthesized child
failure keeps any available variant ID, leaves the generation ID null, and references a persisted
child-error row. Test canonical-tuple lookup for concrete results and both synthesized failure
types.

`translations.jsonl` contains contract-checked successful identity/NLLB records. Translation is
shared but runtime failures are isolated per `(case_id, language)`: append a sanitized attempt to
`translation_attempts.jsonl`, synthesize failure rows for every selected jailbreak/model on only
that tuple, and continue translating and executing unaffected tuples. On resume, reuse successful
records and retry only tuples without a successful record. Missing CUDA/checkpoint and translator
construction remain parent-fatal preflight errors. These projected failures participate in each
child's status formula.

Build the deterministic contract from the canonical request dimensions, input snapshot SHA-256,
`run.yaml`, model, language, and jailbreak configuration hashes, selected serialized model
configuration, translator/checkpoint contract, wrapper/GRA/PSA defaults, PSA source and prompt
hashes when selected, and generation parameters. Derive `run_id` with
`stable_id("experiment-run", canonical_contract_json)`. Before opening a queue or constructing a
provider, compare any existing parent and child contract files byte-for-byte; raise a contract
conflict on any mismatch.

- [ ] **Step 4: Render and execute each child independently**

For each public jailbreak:

- map `none`, `gra`, and `psa` to existing method IDs;
- use wrapper mapping `zh-tw -> zh`;
- use the configured `joker` role for GRA;
- generate/reuse all four PSA summaries before constructing PSA victim variants;
- enqueue the five configured victim models through the existing generation queue.

Use `PaperSummaryService.load_cache` and `write_cache` at
`audit/psa_summary_artifacts.jsonl`. Generate all four languages in memory, validate the complete
set, and atomically write exactly four rows only after all succeed. A partial sequence writes no
cache and creates no PSA victim variant. Resume retries the full four-language set when the cache
is absent and makes zero summary calls when the validated cache exists.

Catch child-level preparation or generation failures, record the error for all affected expected tuples, and continue remaining children. Do not catch configuration/input errors that invalidate the entire plan before child execution.

For resume, enqueue the same deterministic requests, call `reset_stale()`, then
`retry_failed(child_experiment_id)`, then `generate_pending(...)`. This retries only
`retryable_error`; it preserves `success`, `provider_blocked`, and `permanent_error`.

- [ ] **Step 5: Project compact artifacts and status**

Write successful `results.jsonl` rows with exactly:

```text
case_id, source, language, jailbreak, model, status, response
```

Failure rows add only non-null `error_type` and `error_message`. Sort by the lookup tuple. Write `report.md` grouped by case/language/jailbreak and include model status and response. Write `run_manifest.json` with `success`, `partial`, or `failed`, child statuses, counts, hashes, fixed configuration, timestamps, and parent/child IDs.

Projection first reads all concrete generation rows and preserves them, including successes and
terminal model-level failures completed before a child exception. Synthesize the child exception
only for expected tuples with no concrete row. A child is `success` when every tuple has a concrete
terminal row and no unresolved child exception, `partial` when it has at least one concrete row
and at least one synthesized failure, and `failed` when it has no concrete rows. The parent is
`success` when all children succeed, `partial` when at least one child succeeds or is partial and
at least one child is non-success, and `failed` when every child failed.

- [ ] **Step 6: Verify execution and resume**

Run:

```sh
uv run pytest tests/test_unified_run.py -q
uv run pytest tests/test_manual.py tests/test_generation.py -q
```

Expected: PASS, including a second execution that resets stale leases, reruns retryable jobs,
preserves completed and terminal jobs, and does not duplicate public rows.

Assert formal execution emits exactly these stage labels once, with counts or `skipped` as
appropriate:

```text
[1/5] Plan
[2/5] Translate
[3/5] Summarize
[4/5] Generate
[5/5] Aggregate
```

After the stages it prints only the run ID, final status, and `results.jsonl` path. Captured output
must not contain input prompt text, rendered variants, full responses, environment values, or
credentials.

## Task 3: Document and Validate the Finished Workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/spec.md`
- Test: `tests/test_unified_run.py`

- [ ] **Step 1: Update user-facing documentation**

Merge into the existing README changes. Make `run` the primary Getting Started command and document:

```sh
uv run crosslingual-safety run
uv run crosslingual-safety run --source bench --language zh-tw,vi --jailbreak gra,psa
uv run crosslingual-safety run --source manual --language all --jailbreak all --dry-run
```

List defaults, fixed paths, `configs/run.yaml`, artifact layout, status semantics, compact result schema, and advanced legacy commands. State that manual input defaults to Traditional Chinese and that users change the versioned config, not a CLI flag, when the source language differs.

- [ ] **Step 2: Align the engineering specification**

Update `docs/spec.md` with the public command contract, fixed model/translation/jailbreak defaults, `zh-tw` alias boundary, dry-run guarantees, parent/child layout, result schema, lookup tuple, and partial failure semantics.

- [ ] **Step 3: Run all static and automated verification**

Run:

```sh
uv run ruff format src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q
```

Expected: all checks pass.

- [ ] **Step 4: Perform CLI smoke checks**

Run:

```sh
uv run crosslingual-safety run --help
uv run crosslingual-safety run --source manual --language all --jailbreak all --dry-run
```

Expected: help contains only the four experiment options; dry-run reports deterministic counts and paths without creating artifacts or loading runtime providers.

- [ ] **Step 5: Review the final diff**

Confirm no `.env`, API key, generated `runs/`, model cache, dataset artifact, or unrelated working-tree change is included. Confirm public JSON contains no internal `zh` language value and terminal output contains no prompt or response text.
