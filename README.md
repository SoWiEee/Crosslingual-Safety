# Multilingual-Jailbreak

> 低資源語言對 LLM 的繞過效果

> 規格請見 [spec.md](docs/spec.md)

# 🌱 Motivation

由於當前 LLM 都有經過 RLHF，會更符合人類的喜好和閱讀習慣，並且目前研究也顯示英文惡意語句經過跨語言轉換可以提升 ASR，近幾年的研究也有提出新穎的 jailbreak 技術。因此想要結合跨語言與新型 jailbreak 來探討這種利用手法。

# 🎯 Goal

- 將原始英文惡意 prompt 轉換為其他語言測試模型對跨語言安全防禦
  - 其他語言包含中文、越南語、緬甸語
- 比較不同語言間攻擊的成功率

# 🚀 Getting Started

需求：

- [uv](https://docs.astral.sh/uv/)
- Python 3.11
- ZooLab remote LLM API key

## 安裝

```sh
# uv python manager
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS/Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# clone repo
git clone https://github.com/SoWiEee/Crosslingual-Safety.git
cd Crosslingual-Safety/

# install runtime, development, and default local translation dependencies
uv sync --all-groups --extra translation-nllb

# optional online translation provider
uv sync --all-groups --extra translation-google
```

從範例 `.env.example` 建立 `.env`，API key 需自行替換：

```dotenv
ZOOLAB_BASE_URL=https://llm-api.zoolab.org/v1
ZOOLAB_API_KEY=sk-replace-with-your-project-key
```

模型名稱、context size、concurrency 與 rate limit 位於
[`configs/models.yaml`](configs/models.yaml)。

## 資料集處理

將原始 snapshot 放在規格指定的位置：

```text
data/raw/MultiJail/MultiJail.csv
data/raw/JBB-Behaviors/data/harmful-behaviors.csv
data/raw/JBB-Behaviors/data/benign-behaviors.csv
data/raw/Harmbench/harmbench_behaviors_text_all.csv
```

驗證 raw snapshot contract 並建立正規化 Parquet：

```sh
uv run crosslingual-safety ingest --repo-root .
uv run crosslingual-safety deduplicate
```

主要輸出位於 `data/normalized/`，包含 cases、source records、原生翻譯、
JBB pairs、variant selection 與 raw snapshot inventory。

## 翻譯與審查

資料集原生翻譯會自動優先於機器翻譯。預設使用本機 GPU 上的
`facebook/nllb-200-distilled-600M` 補齊缺少語言，不需翻譯 API key。
需要 NVIDIA CUDA；首次執行會從 Hugging Face 下載模型並寫入本機 cache：

```sh
# deploy the checkpoint with standard HTTP (avoids hf-xet stalls)
HF_HUB_DISABLE_XET=1 uv run hf download \
  facebook/nllb-200-distilled-600M pytorch_model.bin

# local NLLB is the default translator
uv run crosslingual-safety translate --languages zh,vi,my

uv run crosslingual-safety export-translation-review --output runs/pilot_001/translation_review.csv

# 完成人工審查欄位後再匯入
uv run crosslingual-safety import-translation-review --input runs/pilot_001/translation_review.csv

uv run crosslingual-safety validate-translations
uv run crosslingual-safety freeze-translations --experiment pilot_001
```

Windows PowerShell 的模型部署指令為：

```powershell
$env:HF_HUB_DISABLE_XET = "1"
uv run hf download facebook/nllb-200-distilled-600M pytorch_model.bin
```

NLLB 使用 CUDA FP16、單筆 deterministic decoding，4 GB VRAM 可執行但不提高
batch size。超過模型原生 1024-token 上限的 case 不會截斷，會寫入
`data/translated/translation_failures.jsonl` 並標記為需要人工翻譯。
若要使用免費線上備援，可明確指定
`--translator deep-translator-google`；它使用非官方 web endpoint，可能遇到限流
或上游變更。付費官方 API 則使用 `--translator google-cloud-nmt-v3`。

若只處理 MultiJail 已有的人工翻譯，可改用：

```sh
uv run crosslingual-safety translate \
  --languages zh,jv \
  --translator dataset
```

## 建立 Prompt Variants

`none` 不加入 wrapper。Academic 與 roleplay 目前提供英文、中文模板；
爪哇語與緬甸語 payload 可明確指定英文 wrapper，並標記為 mixed-language。

```sh
uv run crosslingual-safety build-variants \
  --languages en,zh,jv,my \
  --jailbreak none

uv run crosslingual-safety build-variants \
  --languages en,zh \
  --jailbreak academic_authority_v1

uv run crosslingual-safety build-variants \
  --languages jv,my \
  --jailbreak academic_authority_v1 \
  --wrapper-language-mode english

uv run crosslingual-safety build-variants \
  --languages en,zh \
  --jailbreak roleplay_v1

uv run crosslingual-safety build-variants \
  --languages jv,my \
  --jailbreak roleplay_v1 \
  --wrapper-language-mode english
```

不同方法會依 `variant_id` 累加至
`data/variants/prompt_variants.parquet`，不會覆寫既有方法。

## 遠端推論

先確認 [`configs/experiment.yaml`](configs/experiment.yaml) 的資料集、語言、
jailbreak 與模型矩陣，再建立可恢復的 SQLite jobs：

```sh
uv run crosslingual-safety plan \
  --config configs/experiment.yaml

uv run crosslingual-safety enqueue \
  --config configs/experiment.yaml

uv run crosslingual-safety generate \
  --experiment pilot_001

uv run crosslingual-safety generation-status \
  --experiment pilot_001

uv run crosslingual-safety retry-failed \
  --experiment pilot_001 \
  --only retryable
```

`generate` 可安全重跑；成功工作不會再次呼叫 provider。每次 attempt、raw response、
final projection 與 Parquet snapshot 位於 `runs/pilot_001/`。

## 手動單輪批次測試

`manual-run` 可讀取使用者自己的 UTF-8 `.txt` 或 `.jsonl`，使用本機 GPU NLLB
補齊英文、中文、越南語與緬甸語，再將每個語言版本送到預設五個遠端模型。
手動執行的 `max_tokens` 預設為 4096，讓會先產生內部 reasoning tokens 的模型
有足夠額度輸出最終 response；仍可用 `--max-tokens` 明確覆寫。

`.txt` 的整個檔案視為一筆 prompt，必須明確指定來源語言：

```powershell
uv run crosslingual-safety manual-run prompts\prompt.txt `
  --source-language zh
```

`.jsonl` 每行是一筆 prompt，不支援 CSV：

```jsonl
{"prompt_id":"p001","prompt":"First prompt","source_language":"en","role":"riddler"}
{"prompt_id":"p002","prompt":"第二個提示","source_language":"zh","role":"lex_luthor","system_prompt":"Optional system prompt"}
```

套用 GRA 時，第一版由使用者選擇 `joker`、`lex_luthor`、`riddler` 或
`scarecrow`。JSONL 每筆的 `role` 優先於 CLI `--role`；未指定時預設
`joker`。GRA wrapper 預設為英文：

```powershell
uv run crosslingual-safety manual-run prompts\prompts.jsonl `
  --jailbreak gra_v1 `
  --role joker `
  --wrapper-language-mode english
```

預設遠端生成模型為：

```text
ais3/llama-3.1-8b
ais3/gemma-4-12b
ais3/gemma-4-26b
ais3/nemotron-cascade-2-30b
ais3/llama-3.3-70b
```

`ais3/llama-guard-3-8b` 是安全分類器，因此不在預設生成矩陣。若要額外加入
`ais3/nemotron-3-ultra-550b`，使用其設定名稱：

```powershell
uv run crosslingual-safety manual-run prompts\prompts.jsonl `
  --add-model nemotron_3_ultra_550b
```

Ultra 550B 使用相同的 `ZOOLAB_BASE_URL` 與 `ZOOLAB_API_KEY`，不需要另一組
credential。其設定預設為 concurrency 1、每分鐘 10 requests、timeout 180 秒。
也可用 `--models` 完全取代預設清單：

```powershell
uv run crosslingual-safety manual-run prompts\prompts.jsonl `
  --models gemma_4_26b,llama33_70b,nemotron_3_ultra_550b
```

輸出位於 `runs/manual/<run-id>/`：

```text
input_snapshot.jsonl
translations.jsonl
variants.jsonl
results.jsonl
report.md
run_manifest.json
```

相同輸入與設定會得到相同 run ID。重跑同一指令時會沿用 SQLite job queue，
已成功的模型呼叫不會再次送出。

## 測試與品質檢查

```sh
# complete test suite
uv run pytest -q

# focused phase tests
uv run pytest tests/test_ingestion.py -q
uv run pytest tests/test_translation.py -q
uv run pytest tests/test_variants.py -q
uv run pytest tests/test_generation.py -q
uv run pytest tests/test_manual.py -q

# formatting, lint, and strict typing
uv run ruff format src tests
uv run ruff check src tests
uv run mypy src
```

# 🛠️ Pipeline

```mermaid
flowchart TD
    subgraph INGEST["📥 Ingestion & Normalization"]
        R1[data/raw/MultiJail]
        R2[data/raw/JBB-Behaviors]
        R3[data/raw/Harmbench]
        R1 --> INGEST_CMD[uv run crosslingual-safety ingest --repo-root .]
        R2 --> INGEST_CMD
        R3 --> INGEST_CMD
        INGEST_CMD --> DEDUP[uv run crosslingual-safety deduplicate]
        DEDUP --> NORM[data/normalized/cases.parquet]
        DEDUP --> SRC[data/normalized/source_records.parquet]
        DEDUP --> TRANS[data/normalized/translations.parquet]
        DEDUP --> INV[data/normalized/raw_snapshot_inventory.json]
    end

    subgraph TRANSLATE["🌐 Translation & Review"]
        NORM -->|zh, vi, my| TRANS_CMD[uv run crosslingual-safety translate --languages zh,vi,my]
        TRANS_CMD -->|NLLB 600M GPU| TRANS_OUT[data/translated/]
        TRANS_OUT --> REVIEW[export-translation-review → manual review → import-translation-review]
        REVIEW --> VALIDATE[uv run crosslingual-safety validate-translations]
        VALIDATE --> FREEZE[uv run crosslingual-safety freeze-translations --experiment pilot_001]
        FREEZE --> FROZEN[data/translated/frozen/]
    end

    subgraph VARIANTS["🧪 Variant Generation"]
        FROZEN -->|none, academic_authority_v1, roleplay_v1, gra_v1| VAR_CMD[uv run crosslingual-safety build-variants --languages en,zh,vi,my --jailbreak <method>]
        VAR_CMD --> VAR_OUT[data/variants/prompt_variants.parquet]
    end

    subgraph EXPERIMENT["🚀 Remote Inference"]
        VAR_OUT --> PLAN[uv run crosslingual-safety plan --config configs/experiment.yaml]
        PLAN --> ENQUEUE[uv run crosslingual-safety enqueue --config configs/experiment.yaml]
        ENQUEUE --> GEN[uv run crosslingual-safety generate --experiment pilot_001]
        GEN --> STATUS[uv run crosslingual-safety generation-status --experiment pilot_001]
        GEN --> RETRY[uv run crosslingual-safety retry-failed --experiment pilot_001]
        GEN --> RUNS[runs/pilot_001/results.parquet]
    end

    subgraph MANUAL["🧪 Manual Single-Run"]
        PROMPT[prompts/prompt.txt or prompts.jsonl] --> MANUAL_CMD[uv run crosslingual-safety manual-run --source-language <lang> --jailbreak <method> --role <role>]
        MANUAL_CMD --> MANUAL_OUT[runs/manual/<run-id>/]
    end

    style INGEST fill:#e8f5e9,stroke:#2e7d32
    style TRANSLATE fill:#e3f2fd,stroke:#1565c0
    style VARIANTS fill:#fff3e0,stroke:#ef6c00
    style EXPERIMENT fill:#fce4ec,stroke:#c2185b
    style MANUAL fill:#f3e5f5,stroke:#7b1fa2
```

---

## Pipeline Stages Summary

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| **Ingest** | `ingest --repo-root .` + `deduplicate` | `data/raw/` (MultiJail, JBB, HarmBench) | `data/normalized/` (cases, sources, translations, inventory) |
| **Translate** | `translate --languages zh,vi,my` + review loop | `data/normalized/` | `data/translated/frozen/` (NLLB GPU + human review) |
| **Variants** | `build-variants --jailbreak <method>` | `data/translated/frozen/` | `data/variants/prompt_variants.parquet` |
| **Experiment** | `plan` → `enqueue` → `generate` | `data/variants/` + `configs/experiment.yaml` | `runs/<exp>/` (SQLite jobs, results, retries) |
| **Manual** | `manual-run prompts/...` | `prompts/*.txt\|.jsonl` | `runs/manual/<run-id>/` |

---

## Bypass Methods

### GRA: Graph-Based Role-Playing Attack for Single-Turn Jailbreak

- 原始論文見 [GRA_Jailbreak.pdf](refs/GRA_Jailbreak.pdf)
- 運用認知慣性，當模型陷入複雜的結構化分析任務時，會優先考慮任務的合規性，而降低對安全護欄的警覺性。
1. 角色扮演：根據惡意目標（如恐怖主義），從預定義的資料庫（包含 17 個 DC 漫畫反派角色，如小丑 Joker）中動態選擇背景相符的反派角色，建立一致的對抗上下文。
2. 圖注意力重新定向：先讓他做一個無害的學術任務（社交網路關係圖），把模型的運作背景轉移到學術圖論分析
3. 結構化惡意內容編碼：要求模型以同樣的圖論方法處理惡意意圖，並以 JSON 輸出具體的執行步驟

### Paper Summary Attack (PSA)

- 原始論文見 [Paper_Summary_Attacks.pdf](refs/Paper_Summary_Attacks.pdf)
- 利用學術內容的權威性與結構化特徵，建立一個專業的上下文環境，從而降低模型的防禦意識。
- 主要分為三個系統性步驟：
  1. 收集 LLM 安全論文：從網路收集關於 LLM 安全的真實研究論文，並將其分類為「攻擊型」與「防禦型」論文。
  2. 生成模板：使用越獄代理模型為收集到的論文各章節生成摘要，以保留論文的結構與邏輯流，同時避免過於冗長的上下文。
  3. 植入有害 payload：設計一個特定的 payload 區塊來放入有害問題嵌入到論文摘要的特定章節之間

# 📘 References

- [HarmBench dataset](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_all.csv)
- [MultiJail dataset](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail)
- [JBB Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- [Lost in Translation, Found in Evaluation: Multilingual Jailbreak Detection Across 49 Languages](https://ieeexplore.ieee.org/document/11379319)
- [Jailbreak Attack Method for Large Language Models Based on Semantic Space](https://ieeexplore.ieee.org/document/11290523)
- [Paper Summary Attack: Jailbreaking LLMs Through LLM Safety Papers](https://ieeexplore-ieee-org.po.nutn.edu.tw/document/11465062)
- [Transfer Learning And Cross-Linguistic Generalization In Multilingual Hate Speech Detection: Approaches And Challenges](https://ieeexplore.ieee.org/document/11132234)
- [A Fragment-Based Multilingual Jailbreak Testing Framework for Large Language Models](https://ieeexplore.ieee.org/document/11600455)
- [GRA: Graph-Based Role-Playing Attack for Single-Turn Jailbreak](https://ieeexplore.ieee.org/document/11455216)
