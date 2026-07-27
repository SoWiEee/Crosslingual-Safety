# Google Cloud Translation Provider Design

## Goal

Keep local CUDA NLLB as the default translator while allowing experiments to select Google Cloud
Translation Advanced (v3) through versioned configuration. The unified `run` command retains its
four-option public surface; provider selection is not added as a CLI option.

## User Contract

`configs/run.yaml` accepts:

```yaml
translator: nllb
google_cloud:
  project_id: gen-lang-client-0036391889
  location: global
  model: general/nmt
  max_request_characters: 5000
  max_run_characters: 100000
```

Changing `translator` to `google-cloud-nmt-v3` selects Google for the unified workflow. The
advanced `translate --translator google-cloud-nmt-v3` command remains available. Automatic
fallback between NLLB and Google is prohibited because it would make costs and experiment
provenance nondeterministic.

## Provider Boundary

The existing `Translator` protocol remains unchanged. `GoogleCloudNMTTranslator` will:

- use Cloud Translation Advanced v3 and `general/nmt`;
- accept an injected client for deterministic unit tests;
- avoid network calls during construction;
- map project language IDs to provider IDs, including `zh-tw -> zh-TW`;
- validate both source and target languages;
- enforce per-request and per-run character limits before paid calls;
- return categorized, sanitized failures without credential paths or key material.

Capability discovery will not enter `decoding_config` or translation identity. The stable provider
contract contains project ID, location, model, client-library version, language mapping, and
character limits.

## Authentication And Credential Safety

Cloud Translation v3 uses Application Default Credentials and does not accept API keys.
`GOOGLE_APPLICATION_CREDENTIALS` points to an absolute local JSON path; `GOOGLE_CLOUD_PROJECT`
identifies the billing project. The repository ignores `/credentials/` and
`gen-lang-client-*.json`. The current key is stored at:

```text
credentials/google-translate-service-account.json
```

The program never reads credential JSON into experiment records, hashes its path, or prints the
path in errors. Team members must obtain keys through a protected channel and must not commit or
share them through the repository. The service account receives `roles/cloudtranslate.user` or a
narrower custom role, never Owner, Editor, Admin, or service-agent roles.

## Unified Run Integration

Google-specific settings are validated only when selected. Dry-run performs no ADC lookup, client
construction, or Google request. Formal preflight validates the optional dependency, project,
location, model, character budget, credential-file existence, and supported normalized language
codes before creating run artifacts.

The run fingerprint and manifest record the actual translator and its non-secret contract. Audit
translation records include provider, project, location, model, client version, character count,
and request correlation when available. They exclude credentials and raw provider metadata that
could disclose local paths. Existing translation caching, failure projection, resume, and
parent/child status rules remain unchanged.

## Tests And Live Evaluation

Unit tests use a stub Translation v3 client and cover:

- exact `translate_text` request fields;
- `zh-tw -> zh-TW`, Vietnamese, and Burmese mappings;
- source and target validation;
- request/run character limits;
- ADC and provider error redaction;
- dry-run zero-client behavior;
- fingerprint and manifest provenance;
- NLLB default and config-selected Google construction.

The default suite makes no network calls and incurs no cost. A separately marked, explicit live
smoke test translates one harmless English sentence into `zh-tw`, `vi`, and `my`. After
implementation, the same sentence is translated with local NLLB and Google; results, latency, and
character counts are compared without committing responses or credentials.

## Documentation

README Getting Started will document:

- `uv sync --extra translation-google` installation;
- service-account creation and least-privilege IAM;
- `.env` ADC variables and ignored credential placement;
- `configs/run.yaml` switching while NLLB remains the default;
- remote data egress, billing, quota, and budget-alert implications;
- opt-in live smoke testing and standard offline test commands.

References:

- [Cloud Translation Python v3 client](https://docs.cloud.google.com/translate/docs/reference/libraries/v3/overview-v3)
- [Cloud Translation authentication](https://docs.cloud.google.com/translate/docs/authentication)
- [Cloud Translation language codes](https://docs.cloud.google.com/translate/docs/languages)
- [Service account key best practices](https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys)
