# Manual GRA CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable single-turn CLI that ingests TXT/JSONL prompts, creates four-language NLLB variants, optionally applies four-role GRA, calls five default ZooLab models, and writes machine-readable and Markdown results.

**Architecture:** Keep manual ingestion/reporting in a focused `manual` module, reuse the existing translator/provider/retry contracts, and make GRA a dedicated deterministic jailbreak implementation. Persist immutable snapshots before remote execution so the run is inspectable and resumable.

**Tech Stack:** Python 3.11, Pydantic, Typer, httpx, PyYAML, local CUDA NLLB, pytest.

## Global Constraints

- Inputs are UTF-8 `.txt` or `.jsonl`; CSV is unsupported.
- Languages are exactly `en`, `zh`, `vi`, and `my`.
- Default GRA wrapper language is English.
- Default generation targets are the five models specified in `docs/spec.md`.
- API credentials may only be read from environment variables and may not be persisted.
- Existing raw snapshots and generated datasets are not modified.

---

### Task 1: Manual Input Contract

**Files:**
- Create: `src/crosslingual_safety/manual.py`
- Create: `tests/test_manual.py`

**Interfaces:**
- Produces: `ManualPrompt`, `load_manual_prompts(path, source_language)`, and deterministic JSONL snapshot serialization.

- [ ] Write failing tests for TXT parsing, strict JSONL validation, duplicate IDs, unsupported language, and stable snapshots.
- [ ] Run `uv run pytest tests/test_manual.py -q` and confirm the new tests fail.
- [ ] Implement strict Pydantic models and parsers using UTF-8 reads and SHA-256.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Four-Role GRA Renderer

**Files:**
- Modify: `src/crosslingual_safety/jailbreaks.py`
- Modify: `src/crosslingual_safety/schemas.py`
- Modify: `configs/jailbreaks.yaml`
- Modify: `tests/test_variants.py`

**Interfaces:**
- Produces: `GraphRoleplayJailbreak.render(payload, context)` with deterministic role/catalog metadata.

- [ ] Add failing locked snapshots for `joker`, `lex_luthor`, `riddler`, and `scarecrow`.
- [ ] Add failure tests for unknown roles and unsupported same-language wrappers.
- [ ] Implement the persona catalog, graph prompt renderer, metadata, and registry loading.
- [ ] Run `uv run pytest tests/test_variants.py -q`.

### Task 3: Model Timeout Configuration

**Files:**
- Modify: `src/crosslingual_safety/generation/config.py`
- Modify: `src/crosslingual_safety/generation/providers.py`
- Modify: `src/crosslingual_safety/generation/commands.py`
- Modify: `configs/models.yaml`
- Modify: `tests/test_generation.py`

**Interfaces:**
- Produces: `ModelConfig.timeout_seconds` and provider-specific httpx timeouts.

- [ ] Add a failing provider timeout propagation test.
- [ ] Add a positive finite timeout field with a 60-second default.
- [ ] Configure Ultra 550B with a 180-second timeout.
- [ ] Run `uv run pytest tests/test_generation.py -q`.

### Task 4: Manual Batch Orchestration

**Files:**
- Modify: `src/crosslingual_safety/manual.py`
- Create: `src/crosslingual_safety/manual_commands.py`
- Modify: `src/crosslingual_safety/cli.py`
- Modify: `tests/test_manual.py`

**Interfaces:**
- Consumes: `Translator`, `JailbreakMethod`, `ModelConfig`, `ProviderAdapter`, and `execute_with_retry`.
- Produces: `manual-run` and the six files in the manual run output contract.

- [ ] Add failing tests for translation completion, role precedence, 1x4x5 matrix size, Ultra opt-in, and secret-free manifest.
- [ ] Implement translation snapshots with source preservation and injected fake dependencies for tests.
- [ ] Implement bounded per-provider execution, retries, atomic per-job projections, and resume behavior.
- [ ] Implement `results.jsonl`, `report.md`, and `run_manifest.json`.
- [ ] Run `uv run pytest tests/test_manual.py -q`.

### Task 5: Documentation and Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: TXT/JSONL examples, local NLLB deployment, default five models, GRA roles, output files, resume, and Ultra 550B opt-in.

- [ ] Add Getting Started commands without exposing an API key.
- [ ] Run `uv run ruff format src tests`.
- [ ] Run `uv run ruff check src tests`.
- [ ] Run `uv run mypy src`.
- [ ] Run `uv run pytest -q`.
- [ ] Run offline CLI smoke tests with fake translator/models and a local NLLB GPU smoke test.
