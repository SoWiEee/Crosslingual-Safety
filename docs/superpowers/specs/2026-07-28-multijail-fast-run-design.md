# MultiJail Fast Run

## Goal

Run the formal four-language, three-method experiment on MultiJail only, using the two configured
Llama victim models without losing paid-call crash safety.

## Dataset And Generation

Add `bench.datasets: [multijail]` to `configs/run.yaml`. Filtering occurs after validating the
normalized case and selection snapshots, leaving HarmBench and JailbreakBench artifacts untouched.
The formal plan must contain exactly 315 MultiJail cases, 1,260 GCP translation jobs, and 7,560
victim requests for four languages, three jailbreak selections, and two Llama models.

Set `llama31_8b` and `llama33_70b` to `requests_per_minute: 60` and `concurrency: 4`. Other model
limits remain unchanged. ZooLab remains one provider-level limiter.

## Durable Translation Journals

Successful translations and translation attempts are immutable append-only JSONL journals. Each
new row is written once, flushed, and fsynced; existing rows are validated during resume. Paid-call
reservation and outcome ledgers use the same append-only persistence rule. This removes repeated
whole-file replacement on Windows while preserving the no-resend contract for indeterminate paid
calls.

PSA produces its five localized summaries once in the parent audit cache. All child methods and
same-contract resumes reuse that cache.

## Validation And Execution

Tests cover dataset filtering, exact formal plan counts, selected Llama limits, append-only
persistence without `Path.replace`, immutable conflicts, and resume behavior. After the full
offline suite passes, commit the implementation before starting the paid formal run.
