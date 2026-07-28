# Incremental Experiment Reports

## Goal

Formal unified runs must expose readable Markdown results while generation is still in progress.
Researchers should not need to wait for every jailbreak child to finish, inspect Parquet files, or
repeat provider calls after an interrupted process.

## Report Layout

Every parent run writes `runs/experiments/<run-id>/report.md`. This root report is an index only. It
contains the run ID, current aggregate status, per-jailbreak completed and expected result counts,
and relative links to the selected child reports. It never duplicates model response text.

Each selected jailbreak writes `children/<jailbreak>/report.md`, including `none`. A child report
contains:

- run and jailbreak identity;
- completed, successful, failed, and expected counts;
- results ordered by case, language, then configured model order;
- status and complete response text for each persisted result;
- sanitized error type and message when no successful response exists.

Response blocks use a fence longer than any backtick run in the response so model-produced Markdown
cannot corrupt the report.

## Update Lifecycle

The parent index and an empty child report are created before victim generation begins. Real
generation updates the child report after each final result is durably persisted and its queue entry
is completed. The parent index is updated at the same boundary. Child completion, resume startup,
and final aggregation also rebuild both report levels.

Writes use the existing atomic replacement helper. Reports are derived views: authoritative JSONL,
Parquet, queue, contract, and error artifacts remain unchanged. A report failure must not discard or
rewrite experiment data, trigger a provider retry, or expose provider response bodies outside the
existing result contract.

Injected test generation callbacks that return a batch may update once after the callback returns;
the production generation path updates per final result.

## Offline Recovery

Add:

```text
crosslingual-safety report-run <run-dir>
```

The command rebuilds the parent index and every selected child report from the persisted
`run_contract.json`, child contracts, variants, generation result artifacts, and child errors. It
does not load `.env`, credentials, translation models, generation providers, or PSA summarizers and
does not make network requests. Missing or malformed authoritative artifacts fail with a concise
error rather than inventing results.

The command is idempotent and may be used for completed, partial, or interrupted runs, including
older compatible unified-run directories.

## Tests And Documentation

Tests lock the root index shape, child grouping/order, complete response inclusion, errors, safe
Markdown fences, per-result production updates, batch-test updates, resume rebuilds, and offline
recovery without provider construction. Existing `results.jsonl` remains the clean machine-readable
result contract.

README Getting Started documents where to find the root index and child reports, how partial reports
behave, how to resume a run, and how to rebuild reports with `report-run`.
