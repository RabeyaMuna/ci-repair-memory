---
pretty_name: CI-REPAIR-BENCH
tags:
  - benchmark
  - ci
  - github-actions
  - program-repair
  - software-engineering
---

# CI-REPAIR-BENCH

CI-REPAIR-BENCH is a benchmark for evaluating automated repair of failing CI builds under real GitHub Actions workflows.
It collects real CI failure instances, applies candidate patches, and re-runs the original CI pipeline to verify correctness.
A repair is successful only if the full CI workflow transitions from failure to pass.

---

## Prerequisites

- Python 3.9 or later
- GitHub account
- Hugging Face account
- Access to the benchmark GitHub organization

---

## Required Setup Files

You must provide both of the following before running anything:

1. `.env` — stores secrets (tokens, API keys)
2. `config.yaml` — stores benchmark runtime configuration

A template config is available at `config.example.yaml`.

---

## 1. Create `.env`

Create `.env` in the repository root:

```text
# GitHub
GITHUB_USERNAME=your_github_username
GITHUB_TOKEN=your_github_personal_access_token

# Hugging Face
HF_TOKEN=your_huggingface_token

# OpenAI (required if using GPT models)
OPENAI_API_KEY=your_openai_api_key

# MiniMax via OpenRouter (use this for MiniMax-M2.5)
MINIMAX_API_KEY=your_openrouter_api_key
MINIMAX_BASE_URL=https://openrouter.ai/api/v1

# Default model used by the pipeline (no --model-key flag needed if set here)
MEMCI_LLM_MODEL=MiniMax-M2.5
```

Supported model keys: `gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`, `MiniMax-M2.5`, `deepseek-chat`

---

## 2. Create `config.yaml`

```bash
cp config.example.yaml config.yaml
```

Fill in all `<PLACEHOLDER>` values. Key fields:

```yaml
baseline_repo_folder: "/path/to/cloned/repos"   # where repos will be cloned
project_result_dir:   "/path/to/baselines/results"
changed_files_folder: "/path/to/baselines/changed_files"
benchmark_owner:      "your_github_username"
username_gh:          "your_github_username"

# Memory settings (start with false; enable for memory run in Step 5)
memory_enabled: false
memory_writeback_enabled: false
memory_top_k: 3
memory_similarity_threshold: 0.55
```

---

## 3. One-Time Setup: Install Dependencies

Use a **separate virtual environment** inside `baselines/`:

```bash
cd baselines
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cd ..
```

---

## 4. One-Time Setup: Fork Repositories

Fork all benchmark repositories into your GitHub account:

```bash
baselines/.venv/bin/python setup_github/bulk_fork_repositories.py
```

---

## Required Dataset Files

The following files must exist before running the pipeline:

```text
dataset/lca_dataset.parquet
generated_patches_list/generated_patches_success_only.json
```

---

## Memory-Guided Experiment: Full Step-by-Step

The experiment runs the evaluation issues **twice** — once without memory (baseline) and once with memory — to measure the performance difference.

### How the Memory System Works

Memory is organized into three levels:

| Level | File | Scope | Content |
|-------|------|-------|---------|
| L1 | `failure_memory.json` | Per-file | Exact error, fix strategy, and file-level failure reason |
| L2 | `repo_memory.json` | Per-repo | Recurring failure patterns grouped by (repo, error_type) |
| L3 | `cross_memory.json` | Cross-repo | Generalizable principles that apply across repositories |

**Retrieval** uses a two-stage process:
1. **Lexical similarity** (bag-of-words cosine + Jaccard) selects candidates per level
2. **Single LLM call** reranks all candidates together, assigns semantic scores, and synthesizes a diagnostic summary
3. **Score blending**: `final_score = 0.40 × lexical + 0.60 × LLM`
4. **Global gate**: `weighted = 0.60×L1 + 0.30×L2 + 0.10×L3`; if below `0.55`, memory injection is suppressed

Memory is used **only in fault localization** — not in patch generation.

---

### Step 1 — Select Seed and Eval Issues

Selects 5 diverse seed issues per repo (used to build memory) and saves the rest as the evaluation split.

```bash
baselines/.venv/bin/python baselines/scripts/prepare_memory_seed_split.py
```

Optional arguments:

```bash
baselines/.venv/bin/python baselines/scripts/prepare_memory_seed_split.py \
  --per-repo 5 \
  --dataset dataset/lca_dataset.parquet \
  --patches generated_patches_list/generated_patches_success_only.json \
  --output-dir baselines/results
```

**Outputs to `baselines/results/`:**

```text
memory_seed_issues.json    ← 5 seed issues per repo (used to build memory)
memory_eval_issues.json    ← remaining issues (used for evaluation)
summary.json               ← seed/eval counts per repo
```

---

### Step 2 — Analyze Seed Issues with CILogAnalyzerLLM

Runs `CILogAnalyzerLLM` on each seed issue to produce structured error context.
Automatically clones each repo at the failing commit. **Resumable** — already-analyzed issues are skipped on re-run.

```bash
baselines/.venv/bin/python baselines/scripts/analyze_memory_seed_issues.py
```

Optional arguments:

```bash
baselines/.venv/bin/python baselines/scripts/analyze_memory_seed_issues.py \
  --seed-file baselines/results/memory_seed_issues.json \
  --dataset dataset/lca_dataset.parquet \
  --model-key MiniMax-M2.5 \
  --output-dir baselines/results
```

**Outputs to `baselines/results/`:**

```text
seed_log_details.json             ← structured error context per seed (20 records: 5 per repo × 4 repos)
seed_log_analysis_manifest.json   ← run summary and any failures
```

---

### Step 3 — Build L1 / L2 / L3 Memory Bank

Processes each seed issue end-to-end in sequence. For **each issue**:
- **CILogAnalyzerLLM** → structured error context
- **L1** (one LLM call per changed file) → per-file failure record with `dependent_files`
- **L2** (one LLM call per issue) → repo-level view of this specific failure (all files + relationships)
- **L3** (one LLM call per issue) → generalizable principle applicable to any repo with this error type

Results are written to disk after **every issue** — the script is fully resumable.

```bash
baselines/.venv/bin/python baselines/scripts/build_memory_bank.py
```

Optional arguments:

```bash
baselines/.venv/bin/python baselines/scripts/build_memory_bank.py \
  --seed-file baselines/results/memory_seed_issues.json \
  --dataset dataset/lca_dataset.parquet \
  --model-key MiniMax-M2.5 \
  --output-dir baselines/results
```

**Outputs to `baselines/results/`:**

```text
failure_memory.json       ← L1: per-file failure records (multiple per issue)
repo_memory.json          ← L2: one repo-level record per issue
cross_memory.json         ← L3: one generalizable principle per issue
memory_bank_summary.json  ← record counts
```

For 20 seed issues you will get approximately:
- ~40–60 L1 records (multiple files per issue)
- 20 L2 records (one per issue)
- 20 L3 records (one per issue)

> **Note:** Steps 1 and 3 only need to run **once**. Step 2 (`analyze_memory_seed_issues.py`) is no longer required — Step 3 runs `CILogAnalyzerLLM` internally. Steps 4–5 can be re-run with different models without rebuilding the memory bank.

---

### Step 4 — Run Evaluation WITHOUT Memory (Baseline)

Make sure `config.yaml` has `memory_enabled: false` (this is the default).

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode baseline
```

Quick test — limit to 5 issues per repo:

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode baseline \
  --max-per-repo 5
```

With a specific model:

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode baseline \
  --model-key MiniMax-M2.5
```

**Results written to:**

```text
baselines/results/<model>_llm_baseline/
  log_details.json          ← CI log analysis output per issue
  fault_localization.json   ← suspected files per issue
  generated_patches.json    ← generated fix patches
  fl_evaluation.json        ← fault localization evaluation scores
  token_report.json         ← LLM token usage and cost
```

---

### Step 5 — Run Evaluation WITH Memory

First update `config.yaml`:

```yaml
memory_enabled: true
memory_writeback_enabled: false   # keep false — prevents data leakage into memory during eval
memory_top_k: 3
memory_similarity_threshold: 0.55
```

Then run:

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode memory
```

Quick test:

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode memory \
  --max-per-repo 5
```

**Results written to:**

```text
baselines/results/<model>_llm_memory/
  log_details.json
  fault_localization.json
  generated_patches.json
  fl_evaluation.json
  token_report.json
  memory_retrieval_log.jsonl    ← per-issue memory retrieval log
```

---

### Step 6 — Compare Baseline vs Memory Results

Replace `MiniMax-M2.5` with whichever model key was used:

```bash
baselines/.venv/bin/python baselines/scripts/compare_memory_runs.py \
  baselines/results/MiniMax-M2.5_llm_baseline \
  baselines/results/MiniMax-M2.5_llm_memory
```

For other models:

```bash
# GPT-4o-mini
baselines/.venv/bin/python baselines/scripts/compare_memory_runs.py \
  baselines/results/gpt-4o-mini_llm_baseline \
  baselines/results/gpt-4o-mini_llm_memory

# DeepSeek
baselines/.venv/bin/python baselines/scripts/compare_memory_runs.py \
  baselines/results/deepseek-chat_llm_baseline \
  baselines/results/deepseek-chat_llm_memory
```

---

### Step 7 — Ablation Study (L1 / L1+L2 / L1+L2+L3)

Run the repair pipeline three times with increasing memory levels to measure the contribution of each level.

| Ablation | Levels active | Result dir |
|----------|---------------|------------|
| L1 only | file-level memory only | `<model>_llm_memory_L1/` |
| L1+L2 | file + repo-level memory | `<model>_llm_memory_L1L2/` |
| L1+L2+L3 | full memory (baseline comparison) | `<model>_llm_memory/` |

> **Note:** Level weights are automatically renormalized per configuration so the similarity threshold gate behaves consistently across all three runs.

If L1+L2+L3 has already been run (Step 5), the CI log analysis output (`log_details.json`) is **identical** for all three ablations. Use `run_ablation_from_log_details.py` to reuse it and only re-run Fault Localization + Patch Generation:

**Run L1 only (reusing existing log_details.json):**

```bash
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1 \
  --source-log-details baselines/results/MiniMax-M2.5_llm_memory/log_details.json
```

**Run L1+L2 (reusing existing log_details.json):**

```bash
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1+L2 \
  --source-log-details baselines/results/MiniMax-M2.5_llm_memory/log_details.json
```

Quick test with a limit:

```bash
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1 \
  --source-log-details baselines/results/MiniMax-M2.5_llm_memory/log_details.json \
  --max-issues 5
```

If you need to run from scratch (without an existing log_details.json):

**Run L1 only:**

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode memory \
  --ablation-levels L1
```

**Run L1+L2:**

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode memory \
  --ablation-levels L1+L2
```

**Run L1+L2+L3 (full memory — same as Step 5):**

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode memory \
  --ablation-levels L1+L2+L3
```

With a specific model or issue limit:

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --memory-mode memory \
  --ablation-levels L1 \
  --model-key MiniMax-M2.5 \
  --max-per-repo 5
```

**Results written to** (replace `MiniMax-M2.5` with your model key):

```text
baselines/results/MiniMax-M2.5_llm_memory_L1/
  fault_localization.json
  generated_patches.json
  fl_evaluation.json
  token_report.json

baselines/results/MiniMax-M2.5_llm_memory_L1L2/
  fault_localization.json
  generated_patches.json
  fl_evaluation.json
  token_report.json

baselines/results/MiniMax-M2.5_llm_memory/        ← L1+L2+L3 (from Step 5)
  fault_localization.json
  generated_patches.json
  fl_evaluation.json
  token_report.json
```

---

### Optional: Run Steps 4–6 in One Command

```bash
baselines/.venv/bin/python baselines/scripts/run_memory_experiment.py
```

Useful options:

```bash
# Quick test (5 issues per repo)
baselines/.venv/bin/python baselines/scripts/run_memory_experiment.py \
  --max-per-repo 5

# With a specific model
baselines/.venv/bin/python baselines/scripts/run_memory_experiment.py \
  --model-key MiniMax-M2.5 \
  --log-analyzer-type llm
```

Final report: `baselines/results/memory_experiment/experiment_report.json`

---

### Full Pipeline Summary

| Step | Script | Purpose | Reads | Writes |
|------|--------|---------|-------|--------|
| 1 | `prepare_memory_seed_split.py` | Select 5 seed issues per repo | dataset + patches | `memory_seed_issues.json`, `memory_eval_issues.json` |
| 2 | `analyze_memory_seed_issues.py` | CILogAnalyzerLLM on seeds (clones repos, resumable) | seed issues + dataset | `seed_log_details.json` |
| 3 | `build_memory_bank_from_seed_analysis.py` | LLM-based L1/L2/L3 extraction | seed issues + log analysis | `failure_memory.json`, `repo_memory.json`, `cross_memory.json` |
| 4 | `run_repo_eval_subset.py --memory-mode baseline` | Repair pipeline without memory | eval issues | `<model>_llm_baseline/` |
| 5 | `run_repo_eval_subset.py --memory-mode memory` | Repair pipeline with memory | eval issues + memory bank | `<model>_llm_memory/` |
| 6 | `compare_memory_runs.py` | Compare baseline vs memory | both result dirs | comparison report |

### Important Notes

- Memory is used **only in fault localization**, not in patch generation.
- `memory_writeback_enabled` must be `false` during evaluation to prevent data leakage back into the memory bank.
- Steps 1–3 only need to run **once**. Steps 4–5 can be re-run with different models.
- Steps 1–3 require LLM API access. Approximate call counts: `(seed_issues × avg_gt_files)` for L1, `(repo × error_type groups)` for L2, `(distinct error_types)` for L3.
- Step 2 is resumable: re-running it skips already-analyzed seed issues.
- If `MEMCI_LLM_MODEL` is set in `.env`, no `--model-key` flag is needed on any command.

---

## Run Without Memory (Standalone Baseline)

To run the repair pipeline directly without any memory bank (skip Steps 1–3):

Make sure `config.yaml` has `memory_enabled: false`, then:

```bash
baselines/.venv/bin/python baselines/main.py
```

Results written to `baselines/results/<model>_llm_baseline/`.

---

## Benchmark Execution (CI Verification)

### One-Time Setup: Forking Repositories

```bash
python setup_github/bulk_fork_repositories.py
```

### Install benchmark dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the Benchmark

```bash
python run_benchmark.py
```

Results are written to `out_folder` (set in `config.yaml`):

```text
jobs_ids.jsonl       ← job identifiers sent to GitHub
jobs_results.jsonl   ← results for each job
jobs_awaiting.jsonl  ← jobs still running (normally empty)
jobs_invalid.jsonl   ← invalid jobs (normally empty)
```

### Re-check CI Outcome

Sometimes GitHub Actions runs slowly. Recheck outstanding jobs without pushing again:

```bash
python recheck_waiting_jobs.py
```
