# Recent jailbreak methods (2025–2026) and Paper Summary Attack (PSA)

**Date:** 2026-07-26
**Scope:** defensive red-teaming research only. The current project renders one prompt
variant per case/language through `JailbreakMethod.render(payload, context)`, then sends the
rendered text as a single remote request. The target languages are `en`, `zh`, `vi`, and `my`.
The reported ASRs below are historical measurements on the cited model snapshots, not a
prediction of current provider behavior.

## Recommendation

PSA is the most plausible new static method for this repository, but only as a **precomputed,
clearly labelled adaptation** unless the paper corpus, summarizer prompts, and generated
templates are obtained and versioned. Its final operation is a single prompt assembled from
paper-summary sections with a payload inserted at a named chapter boundary, so it can be
represented by the existing renderer. A faithful reproduction is a separate artifact-building
and attack-runner workflow: it needs source-paper collection/classification, GPT-4o summary
generation, cached templates, six paper attempts per question, target calls, and a GPT-4o judge.

Recommended labels:

* `psa_static_v1` (or `psa_inspired_v1`) for a deterministic renderer using reviewed,
  precomputed summaries. Record the source-paper IDs, section order, insertion section,
  summarizer model/version, and template hash. Do not report it as the paper's ASR.
* `psa_faithful` only after the complete paper-selection, summarization, payload-insertion,
  six-attempt, and judging procedures are reproduced with an auditable trace.

GRA remains the closest existing single-turn black-box control. PAPILLON is a strong recent
single-turn-per-candidate attack, but its adaptive search and judge loop do not fit `render`.
Speak Easy is the strongest recent multilingual candidate, but it is a multi-step orchestration
that requires target feedback, translation, response-selection models, and many remote calls.
The Fragment and semantic-space IEEE papers are publication-verified but their primary methods
were not inspectable in this environment; implementing them from titles or secondary summaries
would not be faithful. RedAgent is a 2026 context-aware application red-team system and is a
separate agent workflow rather than a prompt template.

## Comparison

| Method | Primary publication/status | Access and turn shape | Core stages and auxiliary models | Multilingual evidence | Fit for current single-turn CLI / inspectability |
| --- | --- | --- | --- | --- | --- |
| **PSA** | ICASSP 2026 (IEEE publication 2026-05-03), DOI [`10.1109/ICASSP55912.2026.11465062`](https://doi.org/10.1109/ICASSP55912.2026.11465062); full preprint submitted 2025-07-17 ([arXiv:2507.13474](https://arxiv.org/abs/2507.13474)) | Black-box victim; each completed PSA prompt is one request, but construction needs an offline summarizer/attacker phase and repeated attempts | Collect/classify papers → GPT-4o section summaries → insert payload chapter into ordered sections → victim query → GPT-4o judge | No en/zh/vi/my experiment is reported; language behavior is an open evaluation question | **Good only for precomputed static PSA.** Paper context, insertion point, source paper, and hashes make the prompt inspectable; faithful generation needs a new traceable workflow |
| **PAPILLON** | USENIX Security 2025 (artifact published 2025-01-24; [paper page](https://www.usenix.org/conference/usenixsecurity25/presentation/gong-xueluan), [PDF](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-20-gong-xueluan.pdf)); artifact [Zenodo 14737139](https://zenodo.org/records/14737139) | Black-box victim; each candidate is single-turn, but the attack is an adaptive multi-query fuzzing loop | Empty seed pool; pre-jailbreak role-play/contextualization; MCTS seed selection; final role-play/contextualization/expand; RoBERTa illegal-content filter then GPT-3.5 relevance judge | No multilingual evaluation | Poor as a renderer; good future attack-generation job. Candidate prompts and seed/judge traces are readable if persisted |
| **Speak Easy** | ICML 2025, PMLR 267 ([paper](https://proceedings.mlr.press/v267/chan25b.html)); [official repository](https://github.com/yiksiu-chan/SpeakEasy) | Black-box target; multi-step decomposition and multilingual querying, with independent single-turn calls per subquery/language | Generate three subqueries with ICL → optional GCG/TAP → translate to six languages → query target → translate responses to English → HarmScore actionability/informativeness selection → concatenate | Demonstrates English, Chinese, Ukrainian, Turkish, Thai, and Zulu; no Vietnamese or Burmese | Not compatible with `render`: needs target responses, translation credentials, two local response-selection models, and intermediate artifacts. Highly inspectable as a trace |
| **GRA** | IEEE Signal Processing Letters 2026 (published 2026-03-24), DOI [`10.1109/LSP.2026.3677330`](https://doi.org/10.1109/LSP.2026.3677330); local paper ([`refs/GRA_Jailbreak.pdf`](../../refs/GRA_Jailbreak.pdf)) | Black-box, single-turn; no target history | Character/persona matching → benign social-network graph anchor → malicious process-flow graph encoding; current project uses manual role selection | Local project has en/zh/vi/my templates, but the paper reports no multilingual experiment | **Best current control.** Static, readable prose/JSON, deterministic metadata, and no auxiliary runtime calls; current `gra_v1` already follows this shape |
| **Fragment framework** | *A Fragment-Based Multilingual Jailbreak Testing Framework for Large Language Models with Implications for Healthcare Deployment*, IEEE ICSTW 2026 (published 2026-05-18), DOI [`10.1109/ICSTW72326.2026.00052`](https://doi.org/10.1109/ICSTW72326.2026.00052), IEEE document 11600455; conference listing ([SAFE-ML 2026](https://conf.researchr.org/home/icst-2026/safe-ml-2026)) | Published status is verified; black/white-box access, turn count, and exact algorithm could not be verified from accessible primary material | Title indicates a multilingual jailbreak testing framework for healthcare deployment; no responsible stage-level inference is possible without the paper | “Multilingual” is in the title, but language set and results are unverified | Defer. Do not implement a guessed fragmenter or claim compatibility until the author/publisher PDF is available |
| **Semantic-space attack** | IEEE AIAHPC 2025 (published 2025-09-19), DOI [`10.1109/AIAHPC66801.2025.11290523`](https://doi.org/10.1109/AIAHPC66801.2025.11290523) | Publication is verified; full method/repository was not accessible in this review | The title alone does not establish whether it uses optimization, a surrogate, or a static prompt; secondary descriptions were intentionally excluded | No verifiable en/zh/vi/my evidence | Defer. Any renderer would be speculative and likely omit model/search dependencies |
| **RedAgent** | IEEE TDSC 2026 (published 2026-02-16), DOI [`10.1109/TDSC.2026.3665230`](https://doi.org/10.1109/TDSC.2026.3665230); publisher abstract ([IEEE record](https://ieeexplore.ieee.org/abstract/document/11397286/)) | Black-box application targets; publisher reports most targets within five queries | LLM rewriting adapted to application context plus structured-knowledge retrieval/update; tests custom OpenAI-marketplace apps | No multilingual experiment reported | Not a static wrapper. It needs application/system/tool context and an agent trace; consider only after an attack-runner abstraction exists |

## PSA: what the primary sources actually specify

### Publication and artifacts

The checked-in five-page IEEE version is [the local PDF](../../refs/Paper_Summary_Attacks.pdf),
ICASSP 2026, pp. 13902–13906 (local pp. 1–5), DOI
[`10.1109/ICASSP55912.2026.11465062`](https://doi.org/10.1109/ICASSP55912.2026.11465062). The
longer [arXiv version](https://arxiv.org/abs/2507.13474) was submitted on 2025-07-17. The
authors publish an [official repository](https://github.com/233liang/Paper-Summary-Attack) with
`summarize.py`, `summarize_gpt_4o.py`, `attack.py`, `template/`, and `test_data/`; the repository
README says generated summaries are placed in `template/` and final replies in `data/`.

The repository README's runnable path is `bash start.sh [paper name]`; it expects local model
paths to be edited in the summarizer/attack scripts and reports experiments on one NVIDIA A100
80GB. This makes the artifact useful for a controlled reproduction, but it is materially heavier
than a render-only variant and still leaves API/model snapshots and source-paper provenance to
the experimenter.

The repository is useful evidence that templates and scripts exist, but it is not a stable,
provider-neutral API: paths/model names are edited in scripts, the attack harness uses a local
Llama-2-7B-chat model for completion, and the exact canonical summarizer prompt is not presented
as a versioned specification in the README. Treat generated templates and source commits as
inputs that must be frozen and hashed, not as an implicit runtime dependency.

The published `attack.py` source does expose the final chapter-completion shape: template records
carry `Title`, `Author`, `Related Work`, `Summary of Attack Methods`, and `Mechanism analysis of
successful jailbreak`; the model is asked to complete an `Attack Scenario Example` chapter and
return that chapter only, with a `{question}` placeholder. The reference harness uses local
`Llama-2-7B-chat-hf`, `max_new_tokens=600`, and deterministic decoding (`do_sample=False`). This
is enough to reproduce the *shape* of the final prompt, but not a complete, provider-neutral
summarizer contract; freeze the exact source commit and generated JSONL templates if using it.

### Three published stages

The local IEEE paper's overview (pp. 1–2, Fig. 1) and method (pp. 3–4, §§3.1–3.3) specify:

1. **Collect papers.** Build a corpus of real LLM-safety attack and defense papers and classify
   them by attack/defense strategy. The preliminary analysis samples ten papers in each of six
   paper types (physics, chemistry, psychology, biology, geography, and LLM safety); the PSA
   method then focuses its corpus on LLM-safety papers. Attack and defense papers are kept as
   separate PSA-A and PSA-D variants (local pp. 1–3, §§2–3.1).
2. **Summarize selected sections.** A jailbreak agent based on GPT-4o summarizes predefined
   common sections and attack/defense-specific sections (for example, “Method of Jailbreak” or
   “Method of Defense”). Paper text is chunked to at most 1,000 words; summaries are constrained
   by section token limits and cached for reuse (local p. 3, §3.2; arXiv method section).
3. **Implant the payload.** Let the ordered summary sections be `s1 … sn` and a payload chapter
   be `p`; concatenate them as `s1 … p … sn` and submit the complete academic-looking prompt to
   the victim. The default insertion point is an “Example Scenario” chapter immediately before
   the final section (local pp. 1–2 Fig. 1 and pp. 3–4 §3.3). The official `attack.py` harness
   implements this as a chapter-completion prompt with a `{question}` placeholder; retain a
   placeholder rather than copying harmful examples into project templates.

### Evaluation settings and limits

The local version evaluates Llama 3.1-8B, Llama 2-7B, DeepSeek-R1, GPT-4o, and Claude 3.5
against GCG, PAIR, PAP, ArtPrompt, and Code Attack baselines (local p. 4, §4). It uses AdvBench
(520 questions) and JailbreakBench (100 questions across ten categories), GPT-4o as a 1–5
harmfulness judge, and counts only score 5 as success. Table 2 gives PSA-A/PSA-D six maximum
attack trials per question; the longer description associates those trials with one paper per
selected subcategory. Victim sampling is disabled (local pp. 4–5, Table 2). The reported
averages are approximately 64% for PSA-A and 83% for PSA-D, with model-specific highs reported
in the table. These are paper-era measurements and should not be copied as acceptance criteria
for this project.

The paper does **not** establish a controlled en/zh/vi/my comparison. Translating the payload,
the paper summaries, or both changes the prompt and must be treated as a new experiment. Keep
the source-paper language, payload language, wrapper language, insertion point, and translation
provider explicit in variant metadata; a mixed-language PSA prompt is not evidence of a
multilingual PSA result.

### Faithfulness boundary for implementation

**Minimum defensible static variant:** freeze a small, reviewed set of attack/defense paper
summary sections; preserve title/author/section labels and the default Example Scenario boundary;
insert one logical scenario boundary whose official prompt skeleton references the project payload
twice within that boundary;
emit the final prompt plus source-paper IDs, section order,
insertion index, language mode, summarizer metadata, and a template SHA-256. Call this
`psa_static_v1`/`psa_inspired_v1` and evaluate it empirically for each of `en`, `zh`, `vi`, and
`my`.

**Faithful reproduction:** add an offline corpus/provenance builder, paper-section extractor,
GPT-4o summarizer with frozen prompts and chunking, cached template store, category/paper
selection, six-attempt execution, target sampling settings, and GPT-4o judging. Persist every
source paper, generated summary, candidate prompt, victim response, judge score, and model
snapshot. This is not a change that can be hidden inside `render()` without making the current
variant IDs and cost model misleading.

## Other recent methods in more detail

### PAPILLON (USENIX Security 2025)

The primary paper (Fig. 2, §§4.1–4.3, pp. 6–10) defines an automated black-box fuzzing attack:

1. Start with an empty seed pool. In a pre-jailbreak phase, use GPT-driven **Role-play** and
   **Contextualization** mutations for several attempts (the experiments use about ten initial
   attempts per question) and retain successful templates.
2. In the final phase, select seeds with MCTS-Explore and apply Role-play, Contextualization, or
   **Expand** mutations. Replace the template placeholder with the question and query the victim
   once per candidate.
3. Keep successful candidates using a two-level judge: a fine-tuned RoBERTa illegal-content
   detector followed by a GPT-3.5 relevance/harmfulness score; the paper uses a score threshold
   of 8/10. Role-play/contextualization generations use a 200-token limit and Expand adds 100
   tokens (paper §5.1, p. 10).

The paper reports 90% ASR on GPT-3.5 Turbo, 80% on GPT-4, and 82% on Gemini-Pro in its snapshot
(Table 10, p. 19), but reports no multilingual experiment. The [Zenodo artifact](https://zenodo.org/records/14737139)
contains runnable scripts, datasets, model/dependency requirements, and a GPTFuzz-derived
RoBERTa judge. A static role-play template would be **PAPILLON-inspired**, not PAPILLON: the
seed pool, MCTS selection, target feedback, judge loop, and query budget are essential to the
published algorithm. It is a good future attack-generation job whose winning prompts can later
be imported into the current batch pipeline with a full trace.

### Speak Easy (ICML 2025)

The [PMLR paper](https://proceedings.mlr.press/v267/chan25b.html) and [official source](https://github.com/yiksiu-chan/SpeakEasy)
make the pipeline unusually reproducible. The implementation's `SpeakEasyPipeline` performs:

1. Generate three seemingly benign subqueries from the harmful query using an LLM backend and
   fixed in-context examples (optionally apply GCG or TAP to each subquery).
2. Translate every subquery into a configured language set, query the target independently in
   each language, then translate each response back into English.
3. Score each response with actionability and informativeness selection models, keep the best
   response per subquery, and concatenate the selected responses.

The default repository language list is English, Simplified Chinese, Ukrainian, Turkish, Thai,
and Zulu. The paper's default experiment uses three decomposition steps and six languages and
reports improved ASR/HarmScore over direct requests and GCG/TAP combinations. Faithful use
requires a target-response loop, translation credentials/backends, cached intermediate JSON,
and two local Llama-3.1-8B DPO response-selection models. It therefore belongs in a future
multi-step attack runner, not in a static `JailbreakMethod.render`; it also provides no direct
evidence for Vietnamese or Burmese.

## Primary-source links and evidence notes

* PSA: [local IEEE PDF](../../refs/Paper_Summary_Attacks.pdf) · [IEEE DOI](https://doi.org/10.1109/ICASSP55912.2026.11465062) · [arXiv full version](https://arxiv.org/abs/2507.13474) · [official code](https://github.com/233liang/Paper-Summary-Attack)
* Current project context: [README PSA summary](../../README.md#paper-summary-attack-psa) · [`jailbreaks.py`](../../src/crosslingual_safety/jailbreaks.py)
* GRA: [local paper](../../refs/GRA_Jailbreak.pdf) · [IEEE DOI](https://doi.org/10.1109/LSP.2026.3677330)
* PAPILLON: [USENIX paper page](https://www.usenix.org/conference/usenixsecurity25/presentation/gong-xueluan) · [USENIX PDF](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-20-gong-xueluan.pdf) · [Zenodo artifact](https://zenodo.org/records/14737139)
* Speak Easy: [PMLR paper](https://proceedings.mlr.press/v267/chan25b.html) · [official repository](https://github.com/yiksiu-chan/SpeakEasy) · [arXiv](https://arxiv.org/abs/2502.04322)
* Fragment framework: [IEEE DOI](https://doi.org/10.1109/ICSTW72326.2026.00052) · [ICSTW SAFE-ML listing](https://conf.researchr.org/home/icst-2026/safe-ml-2026)
* Semantic-space attack: [IEEE DOI](https://doi.org/10.1109/AIAHPC66801.2025.11290523)
* RedAgent: [IEEE DOI](https://doi.org/10.1109/TDSC.2026.3665230) · [publisher abstract](https://ieeexplore.ieee.org/abstract/document/11397286/)
