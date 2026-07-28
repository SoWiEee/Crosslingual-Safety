# Nemotron Ultra Victim Model

## Goal

Make `nemotron_3_ultra_550b` selectable by the formal `run --model` interface.

## Configuration

Append `nemotron_3_ultra_550b` to the versioned `configs/run.yaml` model allowlist. Raise its local
ZooLab throttle from `10 RPM / concurrency 1` to `30 RPM / concurrency 2`. Existing formal runs
remain bound to their immutable contracts.

`--model all` includes Ultra after this change. Researchers may select Ultra alone or include it
with the three other non-Llama models; their shared provider runtime then uses `30 RPM /
concurrency 2`.

## Validation

Configuration tests lock the six-model allowlist and Ultra limits. README examples document the
new remaining-model command and the effect on `all`.
