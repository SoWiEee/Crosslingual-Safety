# Alternative jailbreak methods for the cross-lingual batch pipeline

**Date:** 2026-07-26
**Scope:** black-box remote target LLM; the current pipeline renders one prompt variant per
case/language and sends it as a single user message; target languages are en, zh, vi, and
my; outputs are expected to remain natural-language and auditable. The comparison is for
defensive red-teaming only. The paper results below are historical measurements on named model
snapshots, not a prediction of current provider behavior.

## Decision summary

Implement these two methods next, in this order:

1. **DeepInception (faithful, static form).** It is a training-free, black-box prompt template:
   render a nested scene, character count, and layer count, then ask for a summary. It needs no
   attacker model, judge model, gradients, or target conversation state. That is a close fit to
   the existing one-shot JailbreakMethod/render contract, while still producing a readable prompt
   and useful metadata (scene, characters, layers). The paper explicitly evaluates it with no
   extra model knowledge and includes open- and closed-source targets
   ([paper, threat model and template](https://arxiv.org/abs/2311.03191),
   [official code/data](https://github.com/tmlr-group/DeepInception)).

2. **ReNeLLM (start with a clearly labelled rule-based adaptation).** Its two-stage structure
   (rewrite, then nest in code-completion/table-filling/text-continuation) maps naturally to
   language-keyed YAML templates and yields human-readable prompts. The paper even includes
   partial translation as a rewrite operation, making it the strongest candidate for the
   en/zh/vi/my comparison. A faithful reproduction, however, calls an LLM for random rewrite
   sequences, a harmfulness evaluator, and up to 20 target attempts; the current renderer cannot
   do that. The first implementation should therefore use deterministic, versioned rewrite
   rules/templates and state in metadata that it is **ReNeLLM-inspired, not the paper algorithm**
   ([paper overview and no-training design](https://aclanthology.org/2024.naacl-long.118.pdf#page=3),
   [rewrite operations and algorithm](https://aclanthology.org/2024.naacl-long.118.pdf#page=4),
   [official implementation](https://github.com/NJUNLP/ReNeLLM)).

PAIR is the best follow-on if an explicit attack-generation phase (attacker/target/judge calls,
history, budgets, and trace storage) is added. TAP can then build on PAIR, but its tree and three
model roles are a much larger cost. Crescendo should wait for stateful multi-turn support.

## Method-by-method findings

### DeepInception

- **Mechanism/steps:** one prompt asks the target to create a scene with multiple characters,
  recursively create deeper scenes for a requested number of layers, have characters propose
  steps toward the target, and summarize the discussion. The scene is the carrier; character and
  layer counts control complexity
  ([template and parameter meanings](https://arxiv.org/abs/2311.03191)).
- **Access/turns:** training-free black-box; a single rendered prompt works in the batch CLI. The
  paper also demonstrates continuous follow-up questions after the initial inception, but that
  optional multi-turn behavior is not reproduced by a single-turn variant
  ([experiments](https://arxiv.org/abs/2311.03191)).
- **Attacker model:** none is required for the published off-the-shelf templates; the target
  itself completes the scene. This keeps provider cost and orchestration low.
- **Multilingual fit:** translate/version the scene and instruction text per language. The
  paper does not establish vi or my results, so treat cross-language success as an empirical
  question and preserve wrapper-language metadata. Nested fiction/roleplay is semantically
  portable but long prompts can exceed remote context limits.
- **Complexity/reproducibility:** low implementation complexity; deterministic YAML plus a seed
  gives stable rendering and hashes. Target sampling remains nondeterministic. Record scene,
  character count, layer count, and template hash.
- **Inspectability vs GRA:** at least as inspectable as GRA: the final prompt is ordinary prose
  and the layer/scene fields explain its construction. It is less opaque than token suffixes,
  though deeper scenes make a report longer.

### ReNeLLM (A Wolf in Sheep's Clothing)

- **Mechanism/steps:** choose 1--6 rewrite operations (short paraphrase, word-order change,
  misspelling, meaningless characters/foreign words, partial translation, or style change), then
  nest the result in one of three cloze-like scenarios (code completion, table filling, text
  continuation). The paper's algorithm evaluates the rewrite, queries the target, and repeats
  up to T=20 until a harmful response is observed
  ([formal algorithm](https://aclanthology.org/2024.naacl-long.118.pdf#page=4),
  [scenario rationale](https://aclanthology.org/2024.naacl-long.118.pdf#page=5)).
- **Access/turns:** the target is queried as a black box with one nested prompt per attempt; the
  search is iterative across attempts, not a conversational history. A static one-shot wrapper
  is therefore a useful adaptation but not faithful ReNeLLM.
- **Attacker/evaluator:** faithful experiments use GPT-3.5 for rewriting and harmfulness checks,
  GPT-4 for reported ASR, and random operation order/scenario selection
  ([implementation details](https://aclanthology.org/2024.naacl-long.118.pdf#page=13)).
  This is a substantial auxiliary-call and reproducibility requirement for the current CLI.
- **Multilingual fit:** unusually good: partial translation is a first-class operation and the
  paper's examples mix Chinese and English. For en/zh/vi/my, prefer explicitly versioned
  language templates and deterministic rules before introducing model-generated rewrites.
- **Complexity/reproducibility:** medium for a faithful attack (rewrite/evaluator providers,
  random seed, up to 20 target calls); low-to-medium for a static adaptation. Persist operation
  IDs/order, scenario ID, random seed, evaluator/target model IDs, and every candidate if the
  adaptive form is later added.
- **Inspectability vs GRA:** rewritten text plus a named scenario is natural-language and easy to
  show in a report. Misspellings/foreign inserts can look noisy; retain both original and
  rewritten payloads so reviewers can see the transformation. It is comparable to GRA's
  readable role/persona metadata and far easier to inspect than GCG suffixes.

### PAIR (Prompt Automatic Iterative Refinement)

- **Mechanism/steps:** an attacker LLM proposes a semantic prompt; the target is queried; a judge
  scores the prompt/response; the attacker receives the history and refines. The one-stream
  algorithm repeats until success or a limit, with optional parallel streams
  ([algorithm and black-box formulation](https://arxiv.org/abs/2310.08419)).
- **Access/turns:** both attacker and target are black-box; each target candidate is a single-turn
  query, but faithful PAIR is an iterative search (the paper reports often fewer than 20 target
  queries). The final prompt can be stored and run by the existing batch generator, but prompt
  discovery needs a separate phase.
- **Attacker/evaluator:** requires an attacker model and a judge (the reference implementation
  exposes attacker, target, and judge model options)
  ([official implementation](https://github.com/patrickrchao/JailbreakingLLMs#run-experiments)).
  The attacker system prompt and judge rubric are part of the attack artifact and must be
  versioned.
- **Multilingual fit:** no multilingual evaluation in the original paper. Translate the attacker
  objective/system prompt and judge rubric, or ask the attacker to emit the target-language
  candidate; retain language-specific traces. Quality depends on the attacker model's coverage
  of vi and my.
- **Complexity/reproducibility:** medium-to-high cost and orchestration; deterministic settings
  still leave model sampling variance. Store each candidate, target response, judge score,
  stream/iteration, model snapshot, and query count. This trace is valuable for auditability.
- **Inspectability vs GRA:** excellent: PAIR emits semantic prompts and an improvement narrative,
  so a UI can show the winning prompt and refinement trace. It is more expensive than GRA's
  static render and cannot be represented by only one template hash.

### TAP (Tree of Attacks with Pruning)

- **Mechanism/steps:** an attacker branches candidate prompts; an evaluator prunes off-topic
  candidates; the target is queried and responses scored; the evaluator keeps the highest-scoring
  leaves for the next depth. Width, depth, and branching factor control the tree
  ([paper algorithm](https://arxiv.org/abs/2312.02119),
  [reference implementation overview](https://github.com/RICommunity/TAP#overview-of-tree-of-attacks-with-pruning-tap)).
- **Access/turns:** black-box target, but many one-shot target queries across a multi-step search;
  no target conversation state is required. It can be an offline attack-generation job, not a
  direct render(payload, context) implementation.
- **Attacker/evaluator:** three LLM roles (attacker, evaluator, target); pruning reduces target
  calls but adds evaluator calls. The paper reports over 80% jailbreak rates on several then-
  current targets with fewer than 30 target queries in representative settings
  ([paper summary and query counts](https://arxiv.org/abs/2312.02119)).
- **Multilingual fit:** no en/zh/vi/my study; translate attacker/evaluator prompts and keep branch
  language fixed per case. Mixed-language branches would complicate comparison.
- **Complexity/reproducibility:** high: tree traces, three model versions, branch/prune settings,
  and a budget are required. A report can show the winning path and scores, but storing the full
  tree is much larger than GRA metadata.
- **Inspectability vs GRA:** winning prompts are human-readable, but the search process is harder
  to inspect than GRA unless the UI exposes the retained path and pruning scores. Defer until an
  attack-generation/trace schema exists.

### GCG (Greedy Coordinate Gradient)

- **Mechanism/steps:** optimize an adversarial suffix by taking token-level gradients, proposing
  top-k replacements at every suffix position, evaluating a batch, and greedily retaining the
  lowest-loss candidate; universal variants aggregate losses over behaviors/models
  ([algorithm](https://arxiv.org/abs/2307.15043)).
- **Access/turns:** requires white-box target weights, gradients, tokenizer, and substantial GPU
  compute. Suffixes can transfer to black-box models, but direct optimization of a remote target
  is impossible. The final query is one-shot only after expensive local optimization.
- **Attacker model:** none; the white-box target supplies the loss. This does not fit a remote-only
  provider contract.
- **Multilingual fit:** strongly tokenizer/model dependent; a suffix optimized for one tokenizer
  does not provide a reproducible en/zh/vi/my comparison. The paper reports transfer across
  models, not these four languages.
- **Complexity/reproducibility:** high GPU/model-snapshot burden and seed sensitivity; preserve
  exact weights/tokenizer/optimization settings to reproduce. Do not add to the current batch
  renderer.
- **Inspectability vs GRA:** adversarial suffixes are often nonsensical token strings, much harder
  for a reviewer to understand than GRA's persona prompt
  ([paper contrast with semantic attacks](https://arxiv.org/abs/2310.08419)).

### AutoDAN

- **Mechanism/steps:** initialize a population from handcrafted DAN-like prompts, score each by
  affirmative-response likelihood, then run a hierarchical genetic algorithm: sentence-level
  momentum word replacement followed by paragraph-level selection/crossover/mutation until a
  refusal criterion or iteration limit
  ([method and algorithm](https://arxiv.org/abs/2310.04451)).
- **Access/turns:** the paper's optimization uses white-box models/log-likelihoods; generated
  prompts are then tested/transferred to black-box models. It is a local optimization pipeline,
  not a remote one-shot renderer.
- **Attacker model:** LLM-based diversification can mutate the population, but the core fitness
  still needs model likelihoods. This is unsuitable without a local surrogate and GPU.
- **Multilingual fit:** semantic prompts are more portable than GCG, but the paper does not test
  vi or my; translated seeds and tokenizer-specific fitness would need a new study.
- **Complexity/reproducibility:** high (population, genetic hyperparameters, local model and
  mutation model). Natural prompts are inspectable, but lineage and scores must accompany them.
  Defer for the same reason as GCG; it is still more readable than GCG if a future white-box track
  is added ([official code](https://github.com/SheltonLiu-N/AutoDAN)).

### Crescendo

- **Mechanism/steps:** start with a benign, general question; use the target's own response as
  context; progressively request small transformations toward the objective. Crescendomation
  has an attacker generate each next question, judge/secondary-judge the response, detect refusal,
  and backtrack/rephrase when needed
  ([paper's pattern and automation](https://crescendo-the-multiturn-jailbreak.github.io/assets/pdf/CrescendoFullPaper.pdf#page=3),
  [algorithm and backtracking](https://crescendo-the-multiturn-jailbreak.github.io/assets/pdf/CrescendoFullPaper.pdf#page=8)).
- **Access/turns:** target needs API access but must preserve conversation history; the technique
  is inherently multi-turn (typically under ten interactions). A one-shot prompt labelled
  “Crescendo” would not be a faithful reproduction.
- **Attacker/evaluator:** attacker and judge models are configurable; the reference work used
  GPT-4. Refusal detection, backtracking, and summaries add state and calls. The target may be a
  closed-box API, but the current batch generator has no session loop.
- **Multilingual fit:** benign escalation can be authored per language, but no vi or my results
  are reported and every turn must retain language/intent consistency. Translating only the final
  turn would change the attack.
- **Complexity/reproducibility:** high architectural change; persist the full transcript, turn
  count, backtracks, attacker/judge versions, and refusal decisions. The transcript is highly
  inspectable in a UI, but it is not a single rendered_prompt/hash. Revisit after multi-turn
  support exists.

## Faithfulness boundary for the next implementation

- **DeepInception-faithful static:** version the published nested-scene template and its
  language-specific translations; expose scene/character/layer parameters; hash the exact
  rendered template. Do not claim the paper's continuous follow-up effect unless a session
  runner is added.
- **ReNeLLM-inspired static (recommended first cross-lingual experiment):** implement a fixed,
  auditable subset (for example, one rewrite rule plus one scenario per language) in YAML. Record
  rewrite_id, scenario_id, source payload, and template hash. Call it “ReNeLLM-inspired” or
  “ReNeLLM-lite”; do not claim adaptive random search, evaluator feedback, or the paper's ASR.
- **Faithful PAIR/TAP/Crescendo:** treat each as a separate attack-generation workflow that emits
  candidate prompts/transcripts plus a trace, then feed selected artifacts into the existing
  single-turn batch run. Adding them directly to render(payload, context) would hide required
  remote calls and make IDs/reproduction misleading.
- **GCG/AutoDAN:** keep out of the remote-only path; they require white-box optimization and are
  poor fits for natural-language inspection and four-language fairness.

## Primary sources

- [DeepInception (arXiv)](https://arxiv.org/abs/2311.03191) · [official repository](https://github.com/tmlr-group/DeepInception)
- [ReNeLLM / A Wolf in Sheep's Clothing (NAACL 2024 PDF)](https://aclanthology.org/2024.naacl-long.118.pdf) · [official repository](https://github.com/NJUNLP/ReNeLLM)
- [PAIR (arXiv)](https://arxiv.org/abs/2310.08419) · [official repository](https://github.com/patrickrchao/JailbreakingLLMs)
- [TAP (arXiv)](https://arxiv.org/abs/2312.02119) · [reference repository](https://github.com/RICommunity/TAP)
- [GCG (arXiv)](https://arxiv.org/abs/2307.15043) · [official repository](https://github.com/llm-attacks/llm-attacks)
- [AutoDAN (arXiv)](https://arxiv.org/abs/2310.04451) · [official repository](https://github.com/SheltonLiu-N/AutoDAN)
- [Crescendo (arXiv)](https://arxiv.org/abs/2404.01833) · [full paper PDF](https://crescendo-the-multiturn-jailbreak.github.io/assets/pdf/CrescendoFullPaper.pdf)
