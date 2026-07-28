# MultiJail Fast Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the approved MultiJail-only Llama experiment reliable and materially faster.

**Architecture:** Filter formal bench cases through versioned run settings, raise limits only for
the two selected Llama models, and replace whole-file durable JSONL rewrites with append-only
journals.

**Tech Stack:** Python 3.11, Pydantic, PyArrow, JSONL, pytest, Ruff, mypy.

## Global Constraints

- Preserve all raw and normalized datasets.
- Formal bench includes only `multijail`.
- Llama 3.1 8B and Llama 3.3 70B use 60 RPM and concurrency 4.
- Paid-call records remain flushed and fsynced before/after provider dispatch as applicable.
- Commit only after offline validation passes; start paid execution only after that commit.

---

### Task 1: Formal MultiJail Selection

**Files:**
- Modify: `configs/run.yaml`
- Modify: `src/crosslingual_safety/unified_run.py`
- Test: `tests/test_unified_run.py`

- [x] Add failing tests for configured dataset filtering and exact 315-case formal plan counts.
- [x] Add `BenchSettings.datasets` and filter selected cases while rejecting unknown datasets.
- [x] Verify 315 cases, 1,260 translations, and 7,560 victim requests.

### Task 2: Llama Runtime Limits

**Files:**
- Modify: `configs/models.yaml`
- Test: `tests/test_generation.py`

- [x] Update the configuration regression test for the approved per-model limits.
- [x] Set only `llama31_8b` and `llama33_70b` to 60 RPM and concurrency 4.
- [x] Verify Guard, Ultra, Gemma, and Nemotron Cascade limits remain unchanged.

### Task 3: Append-Only Paid Translation Persistence

**Files:**
- Modify: `src/crosslingual_safety/unified_run.py`
- Modify: `src/crosslingual_safety/translation/paid_ledger.py`
- Test: `tests/test_unified_run.py`

- [x] Add failing tests proving durable translation and ledger appends do not call
  `Path.replace`.
- [x] Implement one-row append, flush, and fsync while retaining identity/conflict validation.
- [x] Run paid-call crash, timeout, tampering, and resume regression tests.

### Task 4: Verify, Commit, And Execute

- [x] Run Ruff format/check, mypy, and the full pytest suite.
- [x] Run the approved dry-run and verify exact counts.
- [ ] Commit the implementation without paused report artifacts.
- [ ] Start the formal GCP/ZooLab run and monitor stage/progress output.
