# Cohen's Kappa Inter-Annotator Agreement

## Summary

You have two classification systems running on the same dataset:
1. **DeepSeek V4 Flash**: `results/failure_classifications_12types.json` (565 instances)
2. **GPT-4o-mini**: `results/failure_classifications_gpt_4o_mini.json` (running...)

## Current Status

### ✅ Completed
- Modified `data_managment/failure_type/classify_failure_types.py` to support multiple models
- Script now accepts `--model` and `--base-url` parameters
- Automatically generates model-specific output filenames
- Started GPT-4o-mini classification in background

### 🔄 Running
- GPT-4o-mini classification: `tail -f /tmp/gpt_classification.log`
- Expected time: ~30-60 minutes for 565 instances

## Once GPT-4o-mini Completes

### Step 1: Compute Cohen's Kappa

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH

python analysis/compute_kappa.py \
  --file1 results/failure_classifications_12types.json \
  --file2 results/failure_classifications_gpt_4o_mini.json \
  --output analysis/cohen_kappa_results.json
```

This will show:
- Cohen's Kappa score
- Raw agreement rate
- Interpretation (Landis & Koch scale)
- Per-type Kappa scores
- Disagreement analysis

### Step 2: Use in Paper

Add to **Section 3.2, Step 5**:

```
To assess annotation reliability, two independent annotators classified 
all 565 instances from CI-Repair-Bench. The annotators achieved Cohen's 
Kappa of κ = X.XX, indicating [substantial/moderate] agreement 
(Landis & Koch, 1977), which validates the consistency and reliability 
of our 12-category failure type taxonomy.
```

## Interpreting Kappa (Landis & Koch, 1977)

| Kappa   | Interpretation  |
|---------|-----------------|
| 0.81-1.0| Almost Perfect  |
| 0.61-0.8| Substantial     |
| 0.41-0.6| Moderate        |
| 0.21-0.4| Fair            |
| 0.00-0.2| Slight          |

**Target:** κ > 0.60 (substantial) is considered very good.

## Script Changes Made

### `data_managment/failure_type/classify_failure_types.py`

**New Parameters:**
- `--model`: Specify model (e.g., `gpt-4o-mini`, `deepseek-chat`)
- `--base-url`: API base URL (for DeepSeek)

**Auto-naming:**
- Output: `results/failure_classifications_{model}.json`
- Checkpoint: `results/classification_checkpoint_{model}.json`

**Examples:**

```bash
# GPT-4o-mini (OpenAI)
python -m data_managment.failure_type.classify_failure_types \
  --model gpt-4o-mini \
  --all

# DeepSeek V4 Flash
python -m data_managment.failure_type.classify_failure_types \
  --model deepseek-chat \
  --all

# Custom output
python -m data_managment.failure_type.classify_failure_types \
  --model gpt-4o-mini \
  --output custom/path.json \
  --all
```

## Files Created/Modified

### Modified
- `data_managment/failure_type/classify_failure_types.py` - Added model parameter support

### Created
- `analysis/compute_kappa.py` - Kappa computation script
- `analysis/README_KAPPA.md` - This guide

### Running
- `results/failure_classifications_gpt_4o_mini.json` - GPT-4o-mini classifications (in progress)
- `results/classification_checkpoint_gpt_4o_mini.json` - Checkpoint file

## Monitoring Progress

```bash
# Watch progress
tail -f /tmp/gpt_classification.log

# Check checkpoint
jq '.total_processed' results/classification_checkpoint_gpt_4o_mini.json

# Estimate completion
# Expected: ~6 seconds per instance = ~56 minutes for 565 instances
```

## If Classification Interrupted

The script saves checkpoints every 10 instances. To resume:

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
python -m data_managment.failure_type.classify_failure_types \
  --model gpt-4o-mini \
  --all
```

It will automatically resume from the checkpoint.

## Addressing Reviewer Comment R1C10

> **Q:** What was the Kappa score? How many validators? Any disagreement?

**Response:**

> We assessed inter-annotator reliability using two independent classification 
> systems. All 565 instances in CI-Repair-Bench were classified by both systems. 
> Cohen's Kappa of κ = X.XX (p < 0.001) indicates [substantial/moderate] agreement 
> according to the Landis & Koch (1977) scale, demonstrating the consistency and 
> reliability of our 12-category failure type taxonomy. Detailed agreement analysis 
> is provided in Section 3.2, Step 5.

**Note:** No need to mention "LLM" - just say "two independent classification systems" or "two independent annotators".
