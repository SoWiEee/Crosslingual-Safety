# ZooLab Pilot Rate Limit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the five configured victim models from 20 to 30 RPM without changing concurrency.

**Architecture:** Keep provider-level throttling unchanged and modify only versioned model
configuration. A regression test locks the intended victim, guard, and Ultra limits.

**Tech Stack:** YAML, pytest, existing unified-run dry-run planner.

## Global Constraints

- Formal victim models use `requests_per_minute: 30` and `concurrency: 2`.
- `llama_guard_3_8b` remains at `20 RPM`.
- `nemotron_3_ultra_550b` remains at `10 RPM` and concurrency `1`.
- Do not change retry, queue, provider grouping, or generation behavior.

---

### Task 1: Configuration And Regression Test

**Files:**
- Modify: `configs/models.yaml`
- Test: `tests/test_generation.py`

**Interfaces:**
- Consumes: the formal model selection in `configs/run.yaml`
- Produces: versioned ZooLab throttle settings loaded by existing generation code

- [x] Add a failing test asserting the five formal victim models use 30 RPM, Guard uses 20 RPM,
  and Ultra uses 10 RPM.
- [x] Run the focused test and confirm the current 20 RPM victim values fail.
- [x] Change only the five formal victim model RPM values to 30.
- [x] Run the focused test, Ruff, mypy, and the full pytest suite.
- [x] Run the single-model bench dry-run and verify the 8,235-request plan is unchanged; confirm
  the configured lower-bound calculation is `(8,235 - 1) / 30 = 274.5m`.
