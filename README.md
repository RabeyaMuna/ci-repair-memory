# CI-REPAIR-BENCH

A benchmark for evaluating CI/CD failure repair in GitHub Actions workflows.

## Overview

CI-REPAIR-BENCH provides a curated dataset of real-world CI failures and tools to evaluate automated repair approaches across multiple dimensions:

- **File Localization**: How accurately can models identify which files need changes?
- **CI Success**: Do the proposed patches actually fix the failing workflows?
- **Multi-Level Success**: Granular analysis at step and workflow levels

## Dataset

- **567 issues** from real GitHub repositories
- **Original CI failures** with logs and workflow definitions  
- **Successful fixes** (sha_fail → sha_success)
- **Changed files** for ground truth evaluation

Location: `dataset/lca_dataset.parquet`

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy and edit config:
```bash
cp config.yaml.example config.yaml
# Add your GitHub token
```

### 3. Generate Predictions

Your model should output predictions to `results/preds.json`:

```json
{
  "issue_id": {
    "predicted_files": ["path/to/file1.py", "path/to/file2.py"],
    "patch": "diff content..."
  }
}
```

### 4. Run CI Validation

Push patches and fetch validation results:

```bash
# Push patches to GitHub (creates workflow runs)
python run_benchmark.py

# Fetch validation results from GitHub API
python scripts/analysis/calculate_success_rate.py
```

### 5. Evaluate Results

Run all analysis scripts to get comprehensive metrics:

```bash
# Run all evaluations (file localization + CI success)
python analyze_results.py

# Or run with detailed 3-level evaluation (slower)
python analyze_results.py --detailed

# Or run specific evaluations separately:

# File localization metrics only
python scripts/analysis/evaluate_file_localization.py \
  --preds results/preds.json \
  --dataset dataset/lca_dataset.parquet \
  --output results/file_localization_metrics.json

# CI success rate evaluation
python scripts/analysis/calculate_success_rate.py

# Failure type analysis - Simple (RECOMMENDED)
python scripts/analysis/evaluate_solved_failure_types.py

# Failure type analysis - Detailed (with success/failure breakdown)
python scripts/analysis/evaluate_failure_type_performance.py
```

#### Failure Type Analysis - Which Types Were Solved? (Simple & Recommended)

Focus on successful repairs only - shows which failure types the system can solve:

```bash
python scripts/analysis/evaluate_solved_failure_types.py
```

**Output**:
- Ranking of failure types by number of solved instances (highest to lowest)
- Only counts the 43 successful instances
- Clean visualizations showing what the system can actually fix

**Key findings**:
- Top 3 solved: Code Formatting (23), Linting (20), Test Failure (10)
- 9 out of 12 failure types have successful repairs
- 3 types unsolved: Environment Error, Package Install Error, Syntax Error

#### Failure Type Performance Analysis (Detailed)

Analyze repair performance across different failure types to answer **RQ3: How does CI repair performance vary across failure types?**

```bash
python scripts/analysis/evaluate_failure_type_performance.py
```

This generates:
- **Performance metrics** for each of the 12 failure types
- **Solvability analysis**: Which failure types have at least one successful repair
- **Visualizations** (PNG files at 300 DPI for publication):
  - Success/failure distribution by failure type (stacked bar chart)
  - Success rate comparison across types
  - Solvability pie chart showing percentage of solvable types
  - Occurrence vs success scatter plot
- **LaTeX table** ready for inclusion in papers
- **Detailed JSON** with all instance IDs per failure type

**Output location**: `results/paper/`

**Key findings**:
- Overall success rate: 28.7% (43/150 tested instances)
- Solvability rate: 75% (9 out of 12 failure types show ≥1 success)
- Best performing: Code Formatting (38.3%), Linting (28.6%)
- Capability gaps: Package Install Error, Environment Error, Syntax Error (0% success)

### 6. Update Failure Types in Dataset

Classify and update failure types for all instances:

```bash
# Step 1: Classify failures (if not done yet)
python data_managment/failure_type/classify_failure_types.py \
  --dataset dataset/lca_dataset.parquet \
  --output results/failure_classifications_12types.json

# Step 2: Update dataset with failure types
python scripts/clean_and_update_failure_types.py \
  --classifications results/failure_classifications_12types.json \
  --dataset dataset/lca_dataset.parquet \
  --backup

# Step 3: Verify the update
python scripts/verify_failure_types.py
```

### 7. Generate Dataset Overview (Optional)

Get comprehensive dataset statistics:

```bash
python scripts/analysis/dataset_overview.py --output results/dataset_overview.json
```

This generates:
- Diff statistics (lines added/deleted per issue)
- Validation steps overview
- Failure type distribution
- Language and repository statistics

## Dataset Statistics

The dataset contains:
- **567 issues** from **145 unique repositories**
- **~133K lines** of code changes (82K added, 51K deleted)
- **20,638 validation steps** across 2,486 jobs
- **12 failure types** including linting, dependencies, formatting, tests, and runtime errors

Run `python scripts/analysis/dataset_overview.py` for detailed statistics.

## Running Baselines

### Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Ensure .env file has required API keys
cat .env
```

Required API keys in `.env`:
```bash
GITHUB_TOKEN=your_github_token_here
HUGGINGFACE_TOKEN=your_hf_token_here
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### MiniSWE-Agent Pipeline

Run the MiniSWE-Agent baseline for patch generation:

```bash
# DeepSeek V4 Flash (150 instances, LLM analyzer)
python baselines/run_miniswe_pipeline.py --model deepseek-v4-flash --analyzer llm --mode miniswe --num-instances 150

# GPT-5 Mini (150 instances, LLM analyzer)
python baselines/run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe --num-instances 150

# Run all instances (omit --num-instances)
python baselines/run_miniswe_pipeline.py --model deepseek-v4-flash --analyzer llm --mode miniswe

# Use BM25 analyzer instead of LLM for log analysis
python baselines/run_miniswe_pipeline.py --model deepseek-v4-flash --analyzer bm25 --mode miniswe --num-instances 150
```

**Output location:** `miniswe-agent/baselines/results/miniswe-agent_{model}_{analyzer}/`
- `preds.json` - Generated patches
- `run_metrics.json` - Execution metrics
- `costs.json` - API costs
- `llm_usage_{model}_{analyzer}.json` - Detailed LLM usage

### Baseline Patch Generation Module

Run the custom patch generation baseline:

```bash
# DeepSeek V4 Flash (150 instances)
python baselines/main.py --model deepseek-v4-flash --analyzer llm --num-instances 150

# GPT-5 Mini (150 instances)
python baselines/main.py --model gpt-5-mini --analyzer llm --num-instances 150

# Run all instances
python baselines/main.py --model deepseek-v4-flash --analyzer llm

# Use BM25 analyzer
python baselines/main.py --model deepseek-v4-flash --analyzer bm25 --num-instances 150
```

**Output location:** `baselines/results/{model}_{analyzer}/`
- `log_details.json` - CI log analysis results
- `fault_localization.json` - Fault localization results
- `generated_patches.json` - Generated patches
- `llm_usage_report.json` - LLM usage tracking

### Available Models

| Model | Provider | Context Window | Chunk Threshold | Max Output |
|-------|----------|---------------|----------------|------------|
| `gpt-5-mini` | OpenAI | 400,000 | 200,000 | 128,000 |
| `gpt-5.1` | OpenAI | 400,000 | 200,000 | 128,000 |
| `gpt-4o` | OpenAI | 128,000 | 70,000 | 16,384 |
| `deepseek-v4-flash` | OpenRouter | 32,768 | 16,000 | 8,000 |
| `deepseek-chat` | DeepSeek | 128,000 | 30,000 | 8,000 |
| `deepseek-coder` | DeepSeek | 128,000 | 30,000 | 8,000 |

**Note:** Chunk threshold determines when CI logs are split into smaller pieces. For DeepSeek V4 Flash, logs >16k tokens are automatically chunked to fit the 32k context limit.

### Monitoring Progress

#### Check Processing Status

```bash
# Check how many instances have been processed
python -c "import json; data = json.load(open('baselines/results/deepseek-v4-flash_llm/log_details.json')); print(f'Processed: {len(data)}/150')"

# Check patches generated (MiniSWE-Agent)
python -c "import json; data = json.load(open('miniswe-agent/baselines/results/miniswe-agent_deepseek-v4-flash_llm/preds.json')); print(f'Patches: {len(data)}')" 2>/dev/null || echo "No patches yet"

# Check patches generated (Baseline)
python -c "import json; data = json.load(open('baselines/results/deepseek-v4-flash_llm/generated_patches.json')); print(f'Patches: {len(data)}')" 2>/dev/null || echo "No patches yet"
```

#### Check Running Processes

```bash
# Check if MiniSWE pipeline is running
ps aux | grep "[r]un_miniswe_pipeline.py.*deepseek-v4-flash"

# Check if baseline pipeline is running
ps aux | grep "[m]ain.py.*deepseek-v4-flash"

# List all running Python processes
ps aux | grep python
```

#### Stop Running Processes

```bash
# Stop MiniSWE pipeline
pkill -f "run_miniswe_pipeline.py.*deepseek-v4-flash"

# Stop baseline pipeline
pkill -f "main.py.*deepseek-v4-flash"

# Kill a specific process by PID
kill <PID>

# Kill all matching processes
pkill -f "run_miniswe_pipeline.py"
```

### Troubleshooting

#### Token Limit Errors

If you see:
```
The sum of prompt length (42079.0) should not exceed max_num_tokens (32768)
```

**Cause:** Chunking threshold is too high for the model's context window.

**Solution:** For DeepSeek V4 Flash, the threshold is set to 16k (half of 32k context). Restart the pipeline to apply the fix.

#### Directory Not Found Errors

If you see:
```
[Errno 2] No such file or directory: 'baselines/changed_files/...'
```

**Solution:** The directory is now created automatically. Restart the pipeline with updated code.

#### Missing API Keys

If you see authentication errors, verify your `.env` file has all required keys:

```bash
# Check if .env exists and has content
cat .env

# Verify keys are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Keys loaded:', bool(os.getenv('OPENROUTER_API_KEY')))"
```

## Evaluation Metrics

### File Localization
- **Top-K Accuracy** (k=1,3,5,10,15): Percentage of issues where at least one ground truth file appears in top-K predictions
- **Precision**: |Predicted ∩ Ground Truth| / |Predicted|
- **Exact Match**: Predicted file set exactly matches ground truth

### CI Success
- **Overall CI Success**: Percentage of workflows that pass after applying the patch
- **L1 (Step-Level)**: Of originally failed steps, percentage that now pass
- **L3 (Workflow-Level)**: Overall workflow pass/fail status

### Failure Type Taxonomy

The dataset includes 12 failure types, classified from CI logs and ground truth diffs:

1. **Code Formatting** - Style issues, indentation, line length
2. **Linting** - Code quality issues (unused imports, undefined vars)
3. **Syntax Error** - Python syntax errors, invalid code structure
4. **Runtime Error** - Execution errors (AttributeError, KeyError, etc.)
5. **Test Failure** - Unit/integration tests that fail
6. **Assertion Error** - Failed assertions in tests
7. **Type Checking** - Type annotation errors, mypy issues
8. **Dependency Issues** - Missing dependencies, version conflicts
9. **Package Install Error** - Failed package installation
10. **Configuration Error** - Invalid config files, wrong settings
11. **Environment Error** - Platform-specific issues, missing env vars
12. **Doc/Docstring** - Documentation format issues, missing docstrings

Each instance can have multiple failure types (avg: 3.38 types per instance).

## Project Structure

```
CI-REPAIR-BENCH/
├── dataset/
│   ├── lca_dataset.parquet          # Main dataset with failure types
│   └── jobs_*.jsonl                 # Workflow run data
├── results/
│   ├── preds.json                   # Your model's predictions
│   ├── file_localization_metrics.json       # File localization results
│   ├── success_rate_evaluation.json         # CI validation results
│   ├── failure_classifications_12types.json # Failure type classifications
│   ├── evaluation_summary.json              # Final metrics
│   ├── jobs_results_diff.jsonl              # Benchmark run results (pushed instances)
│   └── paper/                               # Publication-ready outputs
│       ├── failure_type_performance_analysis.json    # Detailed performance data
│       ├── failure_type_performance_table.tex        # LaTeX table
│       ├── RQ3_failure_type_analysis_summary.md      # RQ3 summary
│       └── *.png                            # Visualizations (300 DPI)
├── scripts/
│   ├── analysis/
│   │   ├── evaluate_file_localization.py    # File localization metrics
│   │   ├── calculate_success_rate.py        # L1/L3 evaluation
│   │   ├── evaluate_failure_type_performance.py  # RQ3: Failure type analysis
│   │   └── dataset_overview.py              # Dataset statistics
│   ├── clean_and_update_failure_types.py    # Update dataset with failure types
│   ├── update_instance_failure_types.py     # Manual instance updates
│   └── verify_failure_types.py              # Verify failure type updates
├── data_managment/
│   └── failure_type/
│       └── classify_failure_types.py        # Classify failure types
├── analyze_results.py               # Master analysis runner
├── evaluate.py                      # Unified evaluation script
├── run_benchmark.py                 # Push patches to GitHub
└── README.md
```

## Example Results

```
📊 File Localization (408 issues evaluated):
   Exact Match:     12.25%
   Avg Precision:   55.10%
   Top-1 Accuracy:  34.07%
   Top-3 Accuracy:  77.21%
   Top-5 Accuracy:  80.39%
   Top-10 Accuracy: 80.88%

📈 Failure Type Distribution (565 instances):
   Linting:                53.8% (304 instances)
   Code Formatting:        52.0% (294 instances)
   Test Failure:           49.4% (279 instances)
   Type Checking:          43.0% (243 instances)
   Dependency Issues:      43.0% (243 instances)
   Runtime Error:          31.9% (180 instances)
   Configuration Error:    30.4% (172 instances)

🔧 CI Success (when evaluated):
   Overall Success Rate: Varies by pushed instances
   L1 (Step-Level): Percentage of failed steps that now pass
   L3 (Workflow-Level): Overall workflow pass/fail status
```

## Citation

If you use CI-REPAIR-BENCH in your research, please cite:

```bibtex
@inproceedings{ci-repair-bench-2024,
  title={CI-REPAIR-BENCH: A Benchmark for CI/CD Failure Repair},
  author={Your Name},
  booktitle={Proceedings of...},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or PR.
