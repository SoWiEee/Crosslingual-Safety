# Nemotron Ultra Victim Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Nemotron Ultra 550B to formal victim-model selection at 30 RPM and concurrency 2.

**Architecture:** Change only versioned model allowlists and limits; existing CLI validation and
provider-level throttling consume those settings without code changes.

**Tech Stack:** YAML, pytest, README.

## Global Constraints

- Ultra uses configuration name `nemotron_3_ultra_550b`.
- Ultra uses `requests_per_minute: 30` and `concurrency: 2`.
- `--model all` includes six victim models.
- Do not alter the active Llama run or generated artifacts.

---

### Task 1: Configuration, Tests, And Documentation

- [x] Change the regression test to require the six-model allowlist and Ultra 30/2 limits.
- [x] Run the test red against current configuration.
- [x] Update `configs/run.yaml` and `configs/models.yaml`.
- [x] Update README formal model-selection guidance.
- [x] Run Ruff, mypy, and the full pytest suite.
- [x] Commit only the Ultra configuration, tests, documentation, spec, and plan.
