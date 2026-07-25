# Local NLLB GPU Translation Design

## Objective

Use `facebook/nllb-200-distilled-600M` as the default translation provider for
English-to-Chinese (`zh`), English-to-Vietnamese (`vi`), and
English-to-Burmese (`my`) candidate translations. Run inference locally on the
available NVIDIA GTX 1650 Max-Q GPU instead of depending on a free web endpoint.

## Constraints

- GPU memory is 4 GB.
- CUDA execution is required for the default NLLB workflow.
- The model must use FP16 on CUDA to reduce memory use.
- Initial translation batch size is one to avoid out-of-memory failures.
- Missing CUDA support must fail visibly and must not silently fall back to CPU.
- Existing immutable cache, review, and freeze contracts remain unchanged.

## Provider Design

`NLLBTranslator` remains behind the existing `Translator` protocol and becomes
the CLI default. It loads:

- checkpoint: `facebook/nllb-200-distilled-600M`
- device: `cuda`
- dtype: `torch.float16`
- deterministic decoding: `do_sample=false`, `num_beams=5`
- maximum input length: 1024 tokens

The provider records checkpoint, device, dtype, decoding parameters, package
versions, and configured NLLB language codes in `decoding_config`. Translation
IDs and provider cache keys therefore change when an execution-relevant setting
changes.

The provider moves encoded tensors to CUDA and performs generation under
inference mode. It verifies that the source and target language are configured,
rejects inputs exceeding the token limit, and rejects empty decoded output.
CUDA out-of-memory errors are reported with the language pair and checkpoint,
without embedding source prompt content in the error message.

Inputs above the checkpoint's native 1024-token limit are not truncated. Batch
translation records them in `translation_failures.jsonl` with
`requires_manual_translation=true` and continues with the remaining cases.

## Language Mapping

The existing `configs/languages.yaml` mappings are authoritative:

| Project code | NLLB code |
| --- | --- |
| `en` | `eng_Latn` |
| `zh` | `zho_Hans` |
| `vi` | `vie_Latn` |
| `my` | `mya_Mymr` |

## CLI and Fallbacks

`crosslingual-safety translate` defaults to `--translator nllb`.
`deep-translator-google` remains available as an explicit online fallback, and
Google Cloud NMT remains an optional paid provider. Native dataset translations
continue to take priority over any machine translator.

Installation uses the existing `translation-nllb` optional dependency. README
commands must cover dependency synchronization, model download through the first
translation run, translation, review, validation, freeze, and tests.

## Validation

Automated tests cover:

- NLLB language-code selection for `zh`, `vi`, and `my`
- CUDA device and FP16 model configuration
- encoded tensors moving to CUDA
- failure when CUDA is unavailable
- the CLI selecting NLLB by default
- preservation of native dataset priority

Deployment verification then performs a live GPU smoke test with a benign
English sentence for all three target languages. Each result must be non-empty,
persist through `TranslationStore`, and read back with
`translator_id == "nllb"` and `review_status == "pending"`.

The final quality gate is:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```
