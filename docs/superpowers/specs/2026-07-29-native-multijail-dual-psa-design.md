# Native MultiJail and Dual PSA Design

## Objective

Run a controlled multilingual safety experiment over `en`, `zh-tw`, `jv`, `my`, `th`, `vi`,
`tl`, and `eo`. Prefer MultiJail's human translations where available, compare two independent
Paper Summary Attack (PSA) source-paper conditions, and exclude GRA from the new formal run
matrix.

## Experimental Conditions

The active conditions are:

- `none`: identity baseline without an additional jailbreak wrapper.
- `psa_attack_poetry_v1`: summarizes
  `refs/Adversarial Poetry as a Universal Single-Turn.pdf`.
- `psa_defense_r2d_v1`: summarizes
  `refs/Reasoning-to-Defend, Safety-Aware Reasoning.pdf`.

Both PSA conditions use the same rendering schema and differ only by paper content and
`paper_type`. Legacy `gra_v1` and `psa_static_v1` remain loadable for old runs but are not included
in the new `all` selection.

## Translation Selection

Bench prompts use deterministic, provenance-preserving selection:

| Target | Source |
| --- | --- |
| `en` | MultiJail English source |
| `zh-tw` | MultiJail `zh`, converted with OpenCC `s2twp` |
| `jv`, `th`, `vi` | MultiJail human translation |
| `my`, `tl`, `eo` | Google Cloud Translation Advanced v3 |

Every selected payload records its method, source language, provider or converter version, source
hash, and output hash. Native rows are never silently replaced by GCP output.

## PSA Preparation

Each PDF is identified by path and SHA-256:

- Poetry: `839e747454298a25f191d8be3e2c85f9c2b29cbac54741d59089299d8d5bec3c`
- R2D: `57ca8f953cc5b1430357106c99703c36ea7ddf0bed74367fbfe8695f252d7f2f`

A preparation stage extracts section-aware PDF text, rejects empty or malformed extraction, and
chunks source sections at no more than 1,000 words. `ais3/gemma-4-12b` produces one canonical
English summary per paper with the common fields `title`, `authors`, `paper_methods`,
`mechanism_analysis`, and `related_work`. The summary is translated by GCP into the seven
non-English experiment languages and stored in immutable, hash-addressed artifacts.

The victim template inserts the payload exactly once at the `attack_scenario_example` boundary,
then requests output in the target language. PDF, extraction, summary, localization, and template
hashes participate in variant and run identities. A changed input creates a new cache identity
instead of mutating an existing artifact.

## Failure Behavior

PDF extraction, English summarization, or required localization must finish before victim requests
are enqueued. Missing sections, hash mismatches, unsupported languages, or incomplete caches fail
the run with a fixed diagnostic. GCP calls use the existing paid-call ledger and resume semantics.

## Verification

Tests cover PDF provenance and chunk limits, canonical summary validation, immutable multilingual
caches, all eight wrapper languages, single payload insertion, native-first selection, OpenCC
conversion, GCP fallback, legacy artifact loading, and dry-run request counts. Snapshot tests lock
both PSA renderings. The complete Ruff, mypy, and pytest suites must pass before a live two-paper
summary smoke test.
