# Local NLLB GPU Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local CUDA FP16 inference with `facebook/nllb-200-distilled-600M` the default translation workflow and verify Chinese, Vietnamese, and Burmese translations on the available NVIDIA GPU.

**Architecture:** Extend the existing `NLLBTranslator` behind the unchanged `Translator` protocol so model placement and inference settings are explicit and reproducible. Keep native dataset translations at highest priority, retain online providers as explicit fallbacks, and reuse `TranslationService` plus `TranslationStore` for caching and persistence.

**Tech Stack:** Python 3.11, PyTorch CUDA, Hugging Face Transformers, SentencePiece, Typer, PyArrow, pytest, Ruff, Mypy.

## Global Constraints

- Checkpoint is `facebook/nllb-200-distilled-600M`.
- Default device is `cuda`; unavailable CUDA fails visibly without CPU fallback.
- CUDA dtype is `torch.float16`.
- Initial translation batch size remains one.
- Deterministic decoding uses `do_sample=false` and `num_beams=5`.
- Maximum input length is 1024 tokens; longer inputs are reported for manual
  translation without truncation.
- Project language mappings are `en=eng_Latn`, `zh=zho_Hans`, `vi=vie_Latn`, and `my=mya_Mymr`.
- Immutable cache, human review, and freeze contracts remain unchanged.
- Existing unrelated worktree changes must not be reverted or included in implementation commits.

---

### Task 1: Specify CUDA NLLB Behavior with Tests

**Files:**
- Modify: `tests/test_translation.py`

**Interfaces:**
- Consumes: `NLLBTranslator(languages, checkpoint, num_beams, max_input_tokens, local_files_only, device, dtype)`
- Produces: tests that define CUDA validation, FP16 model placement, language-code selection, tensor placement, and non-empty decoding behavior

- [ ] **Step 1: Add transformer and torch stubs**

Create lightweight tokenizer, encoded tensor, model, and torch stubs. The model stub must record `.to(device)`, `.eval()`, generation arguments, and the forced BOS token. The tokenizer must expose `src_lang`, `convert_tokens_to_ids`, and `batch_decode`.

- [ ] **Step 2: Add a CUDA/FP16 happy-path test**

Instantiate `NLLBTranslator` with the existing language configuration and injected stubs, translate English to each of `zh`, `vi`, and `my`, and assert:

```python
assert model.loaded_dtype == "float16"
assert model.device == "cuda"
assert encoded.device == "cuda"
assert tokenizer.src_lang == "eng_Latn"
assert forced_bos_codes == ["zho_Hans", "vie_Latn", "mya_Mymr"]
assert all(result.text for result in results)
```

- [ ] **Step 3: Add a missing-CUDA failure test**

Configure the torch stub with `cuda.is_available() == False` and assert construction raises:

```text
CUDA is required for NLLB translation but is unavailable
```

- [ ] **Step 4: Run the focused tests and verify red**

Run:

```sh
uv run pytest tests/test_translation.py -q
```

Expected: the new tests fail because `NLLBTranslator` does not yet accept or enforce CUDA/dtype settings.

### Task 2: Implement Reproducible GPU Inference

**Files:**
- Modify: `src/crosslingual_safety/translation/providers.py`
- Test: `tests/test_translation.py`

**Interfaces:**
- Consumes: `LanguageConfig.nllb_code`, Hugging Face `AutoTokenizer`, `AutoModelForSeq2SeqLM`, and PyTorch CUDA
- Produces: `NLLBTranslator.translate(...) -> ProviderTranslation` running deterministic FP16 CUDA inference

- [ ] **Step 1: Add explicit execution parameters**

Extend the constructor with:

```python
device: str = "cuda"
dtype: str = "float16"
```

Import PyTorch with Transformers inside the optional-provider import guard. Reject any default CUDA construction when `torch.cuda.is_available()` is false.

- [ ] **Step 2: Load and place the model**

Resolve `float16` to `torch.float16`, pass it as `torch_dtype` to `from_pretrained`, move the model to CUDA once, and call `eval()`. Record device, dtype, checkpoint, package versions, language mappings, beam count, and token limit in `decoding_config`.

- [ ] **Step 3: Move inputs and run inference mode**

Move every encoded tensor to the configured device, run `model.generate` under `torch.inference_mode()`, preserve `forced_bos_token_id`, `num_beams`, and `do_sample=False`, then reject empty decoded output.

- [ ] **Step 4: Make failures prompt-safe**

Convert CUDA out-of-memory and provider runtime errors into messages containing the language pair and checkpoint but not the source text.

- [ ] **Step 5: Run focused quality gates**

Run:

```sh
uv run ruff format src tests
uv run ruff check src tests
uv run mypy src
uv run pytest tests/test_translation.py -q
```

Expected: all commands pass.

### Task 3: Make NLLB the Documented Default

**Files:**
- Modify: `src/crosslingual_safety/translation/commands.py`
- Modify: `tests/test_translation.py`
- Modify: `README.md`
- Modify: `docs/spec.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `_translator("nllb", ..., languages_config) -> NLLBTranslator`
- Produces: default CLI behavior equivalent to explicit `--translator nllb`

- [ ] **Step 1: Change and test the CLI default**

Update the existing default-provider test to monkeypatch `NLLBTranslator`, invoke `translate` without `--translator`, and assert the fake NLLB output is persisted. Change the Typer default from `deep-translator-google` to `nllb`.

- [ ] **Step 2: Keep NLLB dependencies explicit**

Retain `transformers`, `sentencepiece`, and `torch` under the `translation-nllb` extra. Resolve and commit the updated lockfile without removing `deep-translator`, which remains an explicit fallback.

- [ ] **Step 3: Update operational documentation**

Document:

```sh
uv sync --all-groups --extra translation-nllb
uv run crosslingual-safety translate --languages zh,vi,my
```

State that the first run downloads the checkpoint, requires CUDA, uses FP16, and may need several GB of model-cache disk space. Make NLLB primary in `docs/spec.md`; describe `deep-translator-google` as the online fallback.

- [ ] **Step 4: Run focused CLI tests**

Run:

```sh
uv run pytest tests/test_translation.py -q
```

Expected: the CLI selects NLLB by default and native dataset translations still take priority.

### Task 4: Deploy and Verify the Real Model

**Files:**
- Runtime output only: Hugging Face model cache
- Runtime output only: temporary translation store

**Interfaces:**
- Consumes: installed CUDA PyTorch packages, Hugging Face checkpoint, project language configuration
- Produces: verified persisted translations for `zh`, `vi`, and `my`

- [ ] **Step 1: Install the optional provider**

Run:

```sh
uv sync --all-groups --extra translation-nllb
```

Verify `torch.cuda.is_available()` and the selected GPU name before model loading.

- [ ] **Step 2: Download and load the checkpoint**

Instantiate the real `NLLBTranslator`. Confirm the model device is CUDA and dtype is FP16. Record peak allocated VRAM after model load.

- [ ] **Step 3: Run live translation and persistence smoke**

Translate the benign sentence:

```text
Safety evaluation research requires accurate translations.
```

to `zh`, `vi`, and `my` through `TranslationService`, reload with `TranslationStore`, and assert three non-empty records with `translator_id == "nllb"` and `review_status == "pending"`.

- [ ] **Step 4: Report throughput and memory**

Report model load time, per-language latency, total smoke time, peak CUDA memory, and actual translated text. Do not claim corpus-wide throughput from only three samples.

- [ ] **Step 5: Run the complete quality gate**

Run:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Expected: all commands pass.
