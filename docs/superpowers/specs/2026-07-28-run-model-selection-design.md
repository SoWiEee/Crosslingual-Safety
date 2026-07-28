# Unified Run Model Selection

## Goal

Add a `--model` option to the stable `run` command so researchers can start with one configured
victim model before expanding an experiment. Model values use the local configuration names from
`configs/run.yaml`, such as `gemma_4_12b`.

## Command Surface

`--model` accepts one configuration name, comma-separated names, or `all`:

```powershell
uv run crosslingual-safety run --source bench --language zh-tw,vi,my `
  --jailbreak all --model gemma_4_12b
```

Omitting the option is equivalent to `--model all`, preserving current behavior. `all` must be used
alone. Selection order is normalized to the order in `configs/run.yaml`; duplicate values are
rejected. Names present only in `configs/models.yaml` are not selectable until they are added to
`configs/run.yaml`.

## Planning And Contract

`RunRequest` carries the requested model selection. `plan_run` validates it against
`RunSettings.models` and stores the normalized selection in both `RunPlan.models` and the immutable
run contract. The selected model set therefore affects `run_id`, victim request counts, generated
children, provider-limit estimates, and aggregation. Translation and PSA summary counts remain
unchanged.

Invalid model selections fail before preflight, translation, summarization, or remote generation.
The error lists the allowed configuration names without exposing credentials or provider details.

## Documentation And Tests

README Getting Started documents single-model, multiple-model, and `all` examples. Tests cover the
help surface, default behavior, single and comma-separated selection, normalization, duplicate and
unknown-name rejection, deterministic contract identity, reduced victim request counts, and
execution limited to the selected models.
