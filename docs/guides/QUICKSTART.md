# Quick Start Guide

## One-Time Setup (Run Once)

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH

# Run the unified setup script
./setup.sh

# Activate the virtual environment
source venv/bin/activate

# Verify everything is working
python verify_setup.py
```

That's it! One virtual environment for everything.

## Running the Pipelines

### Test Run (1 instance)

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines

# Make sure venv is active
source ../venv/bin/activate

# Test with 1 instance
python run_miniswe_pipeline.py \
    --model gpt-5-mini \
    --analyzer llm \
    --mode miniswe \
    --num-instances 1
```

### Full Production Run

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines
source ../venv/bin/activate

# Run with all instances
python run_miniswe_pipeline.py \
    --model gpt-5-mini \
    --analyzer llm \
    --mode miniswe
```

### With DeepSeek (via OpenRouter)

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines
source ../venv/bin/activate

python run_miniswe_pipeline.py \
    --model deepseek-v4-flash \
    --analyzer llm \
    --mode miniswe
```

## How Data Reuse Works

The MiniSWE pipeline automatically reuses existing data:

1. **Checks** for existing `log_details.json` → reuses if found
2. **Checks** for existing `fault_localization.json` → reuses if found  
3. **Always runs** MiniSWE-Agent for patch generation

Example:
```bash
# First run: generates everything
python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe --num-instances 10

# Output:
# [1/3] RUN log analysis (not found)
# [2/3] RUN fault localization (not found)
# [3/3] RUN MiniSWE-Agent

# Second run: reuses log + fault data
python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe --num-instances 20

# Output:
# [1/3] REUSE log analysis (10 found, 10 new to generate)
# [2/3] REUSE fault localization (10 found, 10 new to generate)
# [3/3] RUN MiniSWE-Agent (all 20 instances)
```

## Directory Structure

```
CI-REPAIR-BENCH/
├── venv/                    # Single virtual environment
├── baselines/
│   ├── results/
│   │   ├── gpt-5-mini_llm/           # Log + fault data
│   │   └── deepseek-v4-flash_llm/    # Log + fault data
│   ├── main.py                        # Baseline pipeline
│   └── run_miniswe_pipeline.py       # MiniSWE pipeline (with reuse)
└── miniswe-agent/
    └── baselines/
        └── results/
            ├── miniswe-agent_gpt-5-mini_llm/        # MiniSWE patches
            └── miniswe-agent_deepseek-v4-flash_llm/ # MiniSWE patches
```

## Troubleshooting

If something doesn't work:

```bash
# 1. Re-run verification
python verify_setup.py

# 2. Check if venv is active
which python
# Should show: .../CI-REPAIR-BENCH/venv/bin/python

# 3. Check model configuration
cd baselines
python -c "from utilities.llm_provider import LLM_REGISTRY; print(LLM_REGISTRY.keys())"

# 4. Test a simple LLM call
python test_openrouter.py  # If dependencies are fixed
```

## Files Reference

- `setup.sh` - One-time setup script
- `verify_setup.py` - Verification script
- `requirements.txt` - Unified dependencies
- `SETUP_GUIDE.md` - Detailed documentation
- `QUICKSTART.md` - This file

## Model Configuration

Already configured in `baselines/utilities/llm_provider.py`:

- **gpt-5-mini**: via OpenAI API
- **deepseek-v4-flash**: via OpenRouter API (cheaper!)

Both use the same unified virtual environment.
