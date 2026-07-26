# Unified Run CLI Design

## Goal

Provide one beginner-facing command for cross-lingual safety experiments while preserving the
existing low-level commands for advanced operation and debugging.

```powershell
uv run crosslingual-safety run `
  --source manual `
  --language en,zh-tw,vi,my `
  --jailbreak none,gra,psa
```

`run --help` exposes only `--source`, `--language`, `--jailbreak`, and `--dry-run`.

## Public Contract

| Option | Accepted values | Default |
|---|---|---|
| `--source` | `manual`, `bench` | `manual` |
| `--language` | one value, comma-separated values, `all` | `all` |
| `--jailbreak` | `none`, `gra`, `psa`, comma-separated values, `all` | `none` |
| `--dry-run` | flag | disabled |

Lists are normalized, deduplicated, and expanded in canonical order. The public language ID is
`zh-tw`; it maps to Traditional Chinese translation and wrapper settings without leaking an
internal `zh` alias into reports or analysis data.

`manual` always reads `prompts/prompt.txt`. `bench` reads normalized cases selected by
`data/normalized/variant_case_selection.parquet`. Benchmark membership and sample size are
configuration concerns, not CLI options.

## Fixed Defaults

The simplified command uses the configured five victim models, local CUDA NLLB translation,
same-as-payload wrapper language, the configured default GRA role (`joker`), and
`ais3/gemma-4-12b` for PSA paper summarization. Model selection, role selection, generation
parameters, configuration paths, and retry controls remain available only through existing
advanced commands and configuration files.

## Architecture

The new `run` command is a facade over existing ingestion, translation, jailbreak, summary, and
generation services. It must call Python interfaces rather than spawn nested CLI processes.

The facade first creates an immutable `RunPlan` containing normalized dimensions, selected cases,
fixed model configuration, expected translations, PSA summaries, victim requests, and artifact
paths. `--dry-run` prints this plan and stops before loading NLLB, reading provider credentials, or
creating a run directory.

Formal execution follows:

```text
options -> normalize -> resolve cases -> RunPlan -> translate
        -> build variants -> generate -> aggregate report
```

Multiple jailbreaks are isolated child experiments that share translations:

```text
runs/experiments/<run-id>/
  run_manifest.json
  report.md
  results.jsonl
  audit/
  children/
    none/
    gra/
    psa/
```

A failed child does not invalidate completed siblings. The parent status is `success`, `partial`,
or `failed`.

## Records

`results.jsonl` is the analysis-facing dataset. Successful rows contain exactly:

```json
{
  "case_id": "case-id",
  "source": "manual",
  "language": "zh-tw",
  "jailbreak": "psa",
  "model": "gemma_4_12b",
  "status": "success",
  "response": "model response"
}
```

Failed rows additionally contain non-null `error_type` and `error_message`. Successful rows omit
those keys. Provider IDs, concrete model IDs, prompts, payloads, role, timing, token counts,
request IDs, translation IDs, template IDs, and hashes are excluded from `results.jsonl`.

Reproducibility data remains under `audit/`: immutable input snapshots, rendered prompts, full
generation records, translation and summary artifacts, model configuration, provenance, and
hashes. A result row is resolved to its audit record by the canonical tuple
`(case_id, source, language, jailbreak, model)`.

## Validation and Failure Handling

Preflight validates only what the selected plan needs:

- `manual` requires a non-empty `prompts/prompt.txt`.
- `bench` requires normalized cases and the selection snapshot.
- Translation requires CUDA and the local NLLB checkpoint.
- Remote execution requires the Zoolab endpoint and API key.
- PSA additionally requires its summarizer configuration.

Input and configuration failures stop before remote calls. Child failures are isolated and
reported. Re-running the same contract resumes pending or retryable work without duplicating
successful jobs. Terminal output contains plan counts, five stage-level progress messages, the
run ID, final status, and result path; it never prints credentials, prompts, or complete responses.

## Testing

Tests must cover:

- the four-option `run --help` surface;
- single, comma-separated, duplicate, and `all` parsing;
- stable `zh-tw` naming and Traditional Chinese mapping;
- fixed manual and benchmark sources;
- a side-effect-free `--dry-run`;
- translation, summary, and victim job counts;
- clean sparse JSON records and audit lookup;
- sibling preservation after a PSA failure;
- same-contract resume behavior;
- secret, prompt, and response absence from terminal output.

The full pytest suite, Ruff, and strict mypy must pass.

## Documentation

README Getting Started leads with dry-run, manual GRA, and baseline-versus-PSA examples. It then
documents the four public options, fixed defaults, artifact layout, partial-run behavior, and an
Advanced Commands section for the existing low-level workflow.

