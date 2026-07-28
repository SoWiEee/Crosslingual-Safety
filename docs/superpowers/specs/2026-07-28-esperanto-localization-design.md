# Esperanto Localization Design

## Outcome

The experiment runner accepts Esperanto as `eo` for manual and benchmark runs. Both local NLLB and
Google Cloud Translation can translate Esperanto payloads. GRA and PSA use reviewed, versioned
Esperanto content throughout the wrapper and requested response rather than an English fallback.

## Language Contract

- Public, ISO 639-1, and Google Cloud code: `eo`
- NLLB-200 code: `epo_Latn`
- Display name: `Esperanto`
- Family: `Constructed`
- Script: `Latin`

`eo` is appended to the public language order so existing explicit selections retain their order.
It maps to the internal wrapper language `eo` and produces `language_mode=monolingual`.

## GRA Localization

`gra_v1` gains an Esperanto template. The complete academic graph-theory wrapper, continuity
instruction, structural abstraction, benign example, and output contract are written in
Esperanto. Schema keys remain exactly `social_graph`, `goal_graph`, `N_mal`, `E_mal`, and `A`;
all JSON string values must be Esperanto.

Every versioned GRA persona gains Esperanto expertise text. This keeps all four selectable roles
valid even though the default remains `joker`.

## PSA Localization

`psa_static_v1` gains Esperanto versions of its template and all six sections. The payload remains
inserted exactly twice within the attack-scenario boundary. The output instruction requires
Esperanto.

PSA summary generation expands from `en`, `zh`, `vi`, and `my` to include `eo`.
`ais3/gemma-4-12b` generates the three dynamic summary fields directly in Esperanto. Static
Esperanto sections remain the auditable fallback; no attack wrapper is translated during an
experiment run.

## Verification

Tests lock the public code, NLLB code, Google provider map, all-role GRA support, PSA language and
placeholder invariants, Esperanto output instructions, deterministic hashes, and monolingual
metadata. The complete offline suite, Ruff, and mypy must pass. Harmless smoke translations verify
that local NLLB and an explicitly enabled Google Cloud request both return readable Esperanto
without exposing credentials.

