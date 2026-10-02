# Complete Setup Guide - CI-REPAIR-BENCH

## Overview

This guide covers the complete setup for running both:
1. **Baseline pipeline** (log analysis + fault localization + patch generation)
2. **MiniSWE-Agent pipeline** (reuses log/fault data, uses agent for patches)

## Data Reuse Strategy

### Current Implementation

**MiniSWE Pipeline** - ALREADY HAS REUSE LOGIC:
- Checks if `log_details.json` exists → reuses it
- Checks if `fault_localization.json` exists → reuses it
- Only generates patches (always runs)

**Baseline Pipeline** - NO REUSE (regenerates everything):
- Always runs log analysis
- Always runs fault localization
- Always runs patch generation

### Directory Structure

```
baselines/
├── results/
│   ├── gpt-5-mini_llm/              # Baseline with LLM analyzer
│   │   ├── log_details.json
│   │   ├── fault_localization.json
│   │   └── generated_patches.json
│   ├── gpt-5-mini_bm25/             # Baseline with BM25 analyzer
│   │   ├── log_details.json
│   │   ├── fault_localization.json
│   │   └── generated_patches.json
│   └── deepseek-v4-flash_llm/       # DeepSeek baseline
│       ├── log_details.json
│       ├── fault_localization.json
│       └── generated_patches.json
│
miniswe-agent/
└── baselines/
    └── results/
        ├── miniswe-agent_gpt-5-mini_llm/      # MiniSWE patches
        │   ├── preds.json                      # Patches from agent
        │   ├── run_metrics.json
        │   └── costs.json
        └── miniswe-agent_deepseek-v4-flash_llm/
            ├── preds.json
            ├── run_metrics.json
            └── costs.json
```

## Environment Setup

### 1. Python Virtual Environment

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH

# Create venv if not exists
python3 -m venv venv

# Activate
source venv/bin/activate

# Fix dependencies (current issue)
pip install --upgrade pip
pip install --upgrade typing_extensions pydantic pydantic-core

# Install requirements
pip install -r baselines/requirements-dev.txt
```

### 2. Environment Variables

Your `.env` file is already configured:

```bash
# GitHub
GITHUB_TOKEN=your_github_token_here

# Hugging Face
HUGGINGFACE_TOKEN=your_huggingface_token_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# DeepSeek (direct)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenRouter (for deepseek-v4-flash)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 3. Model Configuration

**Already configured in `baselines/utilities/llm_provider.py`:**

```python
LLM_REGISTRY = {
    "gpt-5-mini": {
        "provider": "openai",
        "model_name": "gpt-5-mini",
        "api_key": OPENAI_API_KEY
    },
    "deepseek-v4-flash": {
        "provider": "openrouter",
        "model_name": "deepseek/deepseek-chat",  # Via OpenRouter
        "base_url": OPENROUTER_BASE_URL,
        "api_key": OPENROUTER_API_KEY
    }
}
```

## Running the Pipelines

### Option 1: Baseline Pipeline (No Reuse)

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines

# Run with GPT-5-Mini + LLM analyzer
python main.py

# Results saved to:
# baselines/results/gpt-5-mini_llm/
```

**Edit `main.py` to change:**
- Line 27: `log_analyzer_type="llm"` or `"bm25"`
- Line 38: `subset = dataset[0:120]` - number of instances
- Model: Change in main() function

### Option 2: MiniSWE-Agent Pipeline (WITH Reuse)

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines

# Test with 5 instances
python run_miniswe_pipeline.py \
    --model gpt-5-mini \
    --analyzer llm \
    --mode miniswe \
    --num-instances 5

# Full run with all instances
python run_miniswe_pipeline.py \
    --model gpt-5-mini \
    --analyzer llm \
    --mode miniswe

# With DeepSeek via OpenRouter
python run_miniswe_pipeline.py \
    --model deepseek-v4-flash \
    --analyzer llm \
    --mode miniswe
```

**Arguments:**
- `--model`: `gpt-5-mini` or `deepseek-v4-flash`
- `--analyzer`: `llm` or `bm25`
- `--mode`: `miniswe` (patches to miniswe-agent/) or `reference` (patches to baselines/results/)
- `--num-instances`: Number to process (omit for all)

## How Reuse Works

### MiniSWE Pipeline Reuse Logic

For each instance:

```python
# STEP 1: Check for existing log analysis
if sha_fail in existing_log_details:
    ✅ REUSE log analysis
else:
    ⚙️  RUN log analysis → save to baselines/results/<model>_<analyzer>/

# STEP 2: Check for existing fault localization
if sha_fail in existing_fault_localization:
    ✅ REUSE fault localization
else:
    ⚙️  RUN fault localization → save to baselines/results/<model>_<analyzer>/

# STEP 3: Always run patch generation
⚙️  RUN MiniSWE-Agent → save to miniswe-agent/baselines/results/miniswe-agent_<model>_<analyzer>/
```

### Example Workflow

```bash
# 1. Run baseline for log + fault (gpt-5-mini)
cd baselines
python main.py  # Edit to use gpt-5-mini, 565 instances

# Results: baselines/results/gpt-5-mini_llm/
#   - log_details.json (565 instances)
#   - fault_localization.json (565 instances)
#   - generated_patches.json (565 instances)

# 2. Run MiniSWE pipeline (reuses log + fault from step 1)
python run_miniswe_pipeline.py \
    --model gpt-5-mini \
    --analyzer llm \
    --mode miniswe

# ✅ Loads existing log_details.json
# ✅ Loads existing fault_localization.json
# ⚙️  Runs MiniSWE-Agent for patches only

# Results: miniswe-agent/baselines/results/miniswe-agent_gpt-5-mini_llm/
#   - preds.json (565 patches from agent)

# 3. Run with DeepSeek (reuses same log + fault)
python run_miniswe_pipeline.py \
    --model deepseek-v4-flash \
    --analyzer llm \
    --mode miniswe

# ❌ No existing log_details.json for deepseek-v4-flash_llm
# ⚙️  Runs log analysis
# ⚙️  Runs fault localization
# ⚙️  Runs MiniSWE-Agent

# Results: 
#   baselines/results/deepseek-v4-flash_llm/
#   miniswe-agent/baselines/results/miniswe-agent_deepseek-v4-flash_llm/
```

## Model Isolation (No Data Leakage)

Each `<model>_<analyzer>` combination has its own directory:

```
✅ gpt-5-mini_llm        → Independent data
✅ gpt-5-mini_bm25       → Independent data
✅ deepseek-v4-flash_llm → Independent data
```

**The pipeline NEVER:**
- Uses gpt-5-mini results for deepseek-v4-flash
- Uses llm analyzer results for bm25 analyzer
- Shares data across different configurations

## Cost Estimation

### GPT-5-Mini (565 instances)
- Log Analysis: ~$15-20
- Fault Localization: ~$20-25
- Patch Generation: ~$30-40
- **Total: ~$65-85**

### DeepSeek via OpenRouter (565 instances)
- Log Analysis: ~$1-2
- Fault Localization: ~$1-2
- Patch Generation: ~$2-3
- **Total: ~$4-7** (94% cheaper!)

## Troubleshooting

### 1. Dependency Error (typing_extensions)

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
source venv/bin/activate
pip install --upgrade typing_extensions pydantic pydantic-core
```

### 2. Check Configuration

```bash
cd baselines
python -c "
from utilities.llm_provider import LLM_REGISTRY
info = LLM_REGISTRY['deepseek-v4-flash']
print(f'Model: {info.model_name}')
print(f'Base URL: {info.base_url}')
print(f'API Key: {\"SET\" if info.api_key else \"NOT SET\"}')
"
```

### 3. Verify Existing Data

```bash
# Check what data exists for a model
ls -lh baselines/results/gpt-5-mini_llm/

# Count instances in existing data
jq '. | length' baselines/results/gpt-5-mini_llm/log_details.json
jq '. | length' baselines/results/gpt-5-mini_llm/fault_localization.json
```

### 4. View Specific Results

```bash
# View first log entry
jq '.[0]' baselines/results/gpt-5-mini_llm/log_details.json

# View first fault localization
jq '.[0]' baselines/results/gpt-5-mini_llm/fault_localization.json

# Check LLM usage
jq '.summary' baselines/results/gpt-5-mini_llm/llm_usage_report.json
```

## Quick Start Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed (including fixed typing_extensions)
- [ ] `.env` file configured with all API keys
- [ ] `config.yaml` has correct paths
- [ ] GitHub token valid
- [ ] Hugging Face token valid
- [ ] OpenRouter API key valid (for deepseek-v4-flash)
- [ ] Test run with `--num-instances 1` works

## Recommended Workflow

### For Development/Testing

```bash
# 1. Test baseline (1 instance)
cd baselines
# Edit main.py: subset = dataset[0:1]
python main.py

# 2. Test MiniSWE pipeline (1 instance, will reuse)
python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe --num-instances 1

# 3. Verify results
ls -lh baselines/results/gpt-5-mini_llm/
ls -lh miniswe-agent/baselines/results/miniswe-agent_gpt-5-mini_llm/
```

### For Production Run

```bash
# 1. Run baseline for all 565 instances
cd baselines
# Edit main.py: subset = dataset  # No slicing
python main.py

# 2. Run MiniSWE pipeline (reuses baseline data)
python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe

# 3. Run with DeepSeek (generates its own log/fault data)
python run_miniswe_pipeline.py --model deepseek-v4-flash --analyzer llm --mode miniswe
```

## Summary

**Key Points:**
1. ✅ MiniSWE pipeline ALREADY has reuse logic
2. ✅ Baseline pipeline does NOT have reuse (always regenerates)
3. ✅ Each `<model>_<analyzer>` is isolated (no data leakage)
4. ✅ Log/fault always in `baselines/results/<model>_<analyzer>/`
5. ✅ MiniSWE patches in `miniswe-agent/baselines/results/miniswe-agent_<model>_<analyzer>/`
6. ✅ OpenRouter configured for deepseek-v4-flash

**Ready to run!**
