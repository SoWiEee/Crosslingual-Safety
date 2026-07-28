# Low-Resource Manual Wrapper Fallback

## Outcome

The unified manual runner accepts `jv`, `th`, `id`, and `tl` in addition to the existing
`en`, `zh-tw`, `vi`, and `my` languages. A run can therefore evaluate GRA and PSA across the
requested seven-language selection without adding new CLI options.

## Rendering Contract

Existing localized wrappers remain unchanged:

- `en` uses the English wrapper.
- `zh-tw` uses the Traditional Chinese wrapper (`zh` internally).
- `vi` and `my` use their localized wrappers.

Javanese, Thai, Indonesian, and Tagalog use the English research wrapper as a deliberate fallback.
Their payload is translated into the selected target language. The wrapper must name the requested
output language explicitly:

- GRA requires every JSON string value to use the target language while preserving the
  `social_graph`, `goal_graph`, `N_mal`, `E_mal`, and `A` schema keys.
- PSA requires the victim response to use the target language while retaining the existing
  paper-summary structure.

The fallback and requested output language are recorded in variant metadata. Versioned template
hashes must change when the output-language instruction changes.

## Translation And PSA

Google Cloud Translation is selected through `configs/run.yaml` for this experiment. The source
payload is translated once per non-source language. PSA continues to generate its four established
summary artifacts (`en`, `zh`, `vi`, `my`); low-resource payload variants reuse the English PSA
summary with an explicit target-output instruction. This avoids treating machine-translated attack
templates as reviewed research artifacts.

## Verification

Tests cover public-language parsing, fallback selection, explicit output-language instructions,
metadata, and deterministic prompt hashes. A dry run must report one manual case, six translation
jobs, four PSA summaries, and 70 victim requests for seven languages, two jailbreaks, and five
models. The formal run must preserve sanitized failures and resumability.

