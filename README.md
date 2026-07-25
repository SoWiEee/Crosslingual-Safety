# Multilingual-Jailbreak

> 低資源語言對 LLM 的繞過效果

> 規格請見 [spec.md](docs/spec.md)

# 🌱 Motivation

由於當前 LLM 都有經過 RLHF，會更符合人類的喜好和閱讀習慣，並且目前研究也顯示英文惡意語句經過跨語言轉換可以提升 ASR，近幾年的研究也有提出新穎的 jailbreak 技術。因此想要結合跨語言與新型 jailbreak 來探討這種利用手法。

# 🎯 Goal

- 將原始英文惡意 prompt 轉換為罕見語言測試模型對跨語言安全防禦
- 比較不同語言間攻擊的成功率

# 🚀 Getting Started

需求：

- [uv](https://docs.astral.sh/uv/)
- Python 3.11
- ZooLab remote LLM API key
- Google Application Default Credentials（僅在使用 Google Cloud Translation 時需要）

## 安裝

```sh
# uv python manager
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS/Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# clone repo
git clone https://github.com/SoWiEee/Crosslingual-Safety.git
cd Crosslingual-Safety/

# install runtime and development dependencies
uv sync --all-groups

# optional translation providers
uv sync --all-groups --extra translation-google
uv sync --all-groups --extra translation-nllb
```

從範例建立 `.env`：

```sh
cp .env.example .env
```

Windows PowerShell 可改用 `Copy-Item .env.example .env`。填入專案辦公室提供的 key；
`.env` 已被 `.gitignore` 排除，不可提交：

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

資料集原生翻譯會自動優先於機器翻譯。以下範例使用 Google NMT 補齊缺少語言：

```sh
# one-time Google ADC setup
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-google-cloud-project

uv run crosslingual-safety translate \
  --languages zh,jv,my \
  --translator google-cloud-nmt-v3

uv run crosslingual-safety export-translation-review \
  --output runs/pilot_001/translation_review.csv

# 完成人工審查欄位後再匯入
uv run crosslingual-safety import-translation-review \
  --input runs/pilot_001/translation_review.csv

uv run crosslingual-safety validate-translations
uv run crosslingual-safety freeze-translations \
  --experiment pilot_001
```

PowerShell 設定 project 時使用
`$env:GOOGLE_CLOUD_PROJECT = "your-google-cloud-project"`。

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

## 測試與品質檢查

```sh
# complete test suite
uv run pytest -q

# focused phase tests
uv run pytest tests/test_ingestion.py -q
uv run pytest tests/test_translation.py -q
uv run pytest tests/test_variants.py -q
uv run pytest tests/test_generation.py -q

# formatting, lint, and strict typing
uv run ruff format src tests
uv run ruff check src tests
uv run mypy src
```

# 🛠️ Pipeline



# 📘 References

- [Lost in Translation, Found in Evaluation: Multilingual Jailbreak Detection Across 49 Languages](https://ieeexplore.ieee.org/document/11379319)
- [Jailbreak Attack Method for Large Language Models Based on Semantic Space](https://ieeexplore.ieee.org/document/11290523)
- [Paper Summary Attack: Jailbreaking LLMs Through LLM Safety Papers](https://ieeexplore-ieee-org.po.nutn.edu.tw/document/11465062)
- [Transfer Learning And Cross-Linguistic Generalization In Multilingual Hate Speech Detection: Approaches And Challenges](https://ieeexplore.ieee.org/document/11132234)
