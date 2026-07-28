# ZooLab Pilot Rate Limit

## Goal

Reduce formal experiment runtime with a conservative local throttle increase.

## Configuration

Set `requests_per_minute: 30` in `configs/models.yaml` for the five victim models selected by
`configs/run.yaml`:

- `llama31_8b`
- `gemma_4_12b`
- `gemma_4_26b`
- `nemotron_cascade_2_30b`
- `llama33_70b`

Keep their concurrency at `2`. Keep `llama_guard_3_8b` at `20 RPM` because it is not part of the
formal victim-model selection, and keep `nemotron_3_ultra_550b` at `10 RPM` with concurrency `1`.

The rate remains provider-level: when several selected models share ZooLab, the generation runtime
uses the lowest selected ZooLab limit. Existing 429, timeout, server-error retry, persistence, and
resume behavior does not change.

## Validation

Add a configuration regression test that locks the five formal victim models to `30 RPM`, the
guard model to `20 RPM`, and Ultra to `10 RPM`. Verify a single-model bench dry-run still plans the
same number of requests; only the displayed theoretical lower bound changes from approximately
`411.7m` to `274.5m` for 8,235 requests.
