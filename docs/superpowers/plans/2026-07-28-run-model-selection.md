# Unified Run Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a validated `--model` selector to `crosslingual-safety run`.

**Architecture:** The CLI parses the public string form, while `plan_run` remains the authoritative
validator and normalizer against `RunSettings.models`. The normalized model tuple flows through
`RunPlan`, its immutable contract, request counts, generation, and aggregation.

**Tech Stack:** Python 3.11, Typer, Pydantic, pytest, Ruff, mypy.

## Global Constraints

- Model values are configuration names already listed in `configs/run.yaml`.
- Accept one name, comma-separated names, or standalone `all`; omitted means `all`.
- Normalize selections to `configs/run.yaml` order and reject duplicates or unknown names.
- Selection changes the run contract and `run_id`; translation and PSA summary counts do not change.
- Do not change provider-level concurrency or rate limits.
- Preserve unrelated uncommitted work; do not commit implementation without an explicit request.

---

### Task 1: Planning Contract

**Files:**
- Modify: `src/crosslingual_safety/unified_run.py`
- Test: `tests/test_unified_run.py`

**Interfaces:**
- Consumes: `RunSettings.models: list[str]`
- Produces: `RunRequest.models: tuple[str, ...]` and normalized `RunPlan.models`

- [x] Add failing tests for default `all`, single/multiple selection, configured-order
  normalization, duplicate rejection, unknown-name rejection, request counts, and distinct
  contract identity.
- [x] Run the focused planning tests and confirm failures come from the missing model selection.
- [x] Add `models` to `RunRequest`; validate and normalize it in `plan_run` against
  `settings.models`; use the normalized tuple for contract models, counts, and `RunPlan.models`.
- [x] Run `uv run pytest tests/test_unified_run.py -q`.

### Task 2: CLI Surface

**Files:**
- Modify: `src/crosslingual_safety/run_commands.py`
- Test: `tests/test_unified_run.py`

**Interfaces:**
- Consumes: `--model NAME[,NAME...]|all`
- Produces: `RunRequest(models=...)`

- [x] Add failing CLI tests for help text, default behavior, a single configured model, and an
  invalid model that fails before side effects.
- [x] Add the Typer `--model` option with default `all`; pass the raw selection through
  `RunRequest` so `plan_run` performs authoritative validation.
- [x] Run focused CLI tests and confirm dry-run victim request counts reflect selected models.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Verify: `src/crosslingual_safety/unified_run.py`
- Verify: `src/crosslingual_safety/run_commands.py`

**Interfaces:**
- Documents single, comma-separated, and `all` model selection.

- [x] Update Getting Started with `--model gemma_4_12b`,
  `--model gemma_4_12b,llama31_8b`, and the default/explicit `all` behavior.
- [x] Run `uv run ruff format src tests` and `uv run ruff check src tests`.
- [x] Run `uv run mypy src`.
- [x] Run `uv run pytest -q`.
- [x] Run the repository bench command with `--model gemma_4_12b --dry-run` and verify its victim
  request count is one fifth of the same `--model all --dry-run` plan.
