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

The system supports a **hierarchical-memory-augmented** repair pipeline where past failure experiences (stored in L1/L2/L3 memory) guide fault localization for new failures.

---

## Repository Structure

```
CI-REPAIR-BENCH/
├── baselines/                        # Memory-augmented repair pipeline
│   ├── ci_repair/                    # Core repair modules
│   │   ├── ci_log_analyzer_llm.py    # LLM-based CI log analyzer
│   │   ├── ci_log_analyzer_bm25.py   # BM25-based CI log analyzer
│   │   ├── fault_localization.py     # Fault localization with memory
│   │   └── patch_generation.py       # Patch generation
│   ├── utilities/                    # Shared utilities
│   │   ├── memory_plugin.py          # 3-level memory retrieval plugin
│   │   ├── llm_provider.py           # LLM API abstraction
│   │   ├── snippet_extractor.py      # Code snippet extraction
│   │   ├── symbols_outline.py        # File outline extraction
│   │   ├── token_tracker.py          # Token usage tracking
│   │   └── ...
│   ├── scripts/                      # Pipeline orchestration scripts
│   │   ├── analyze_memory_seed_issues.py      # Step 3: LLM log analysis on memory issues
│   │   ├── build_memory_bank.py               # Step 4: build L1/L2/L3 memory bank
│   │   ├── run_repo_eval_subset.py            # Step 5 & 6: run baseline / memory eval
│   │   ├── compare_memory_runs.py             # Step 7: compare results
│   │   ├── run_ablation_from_log_details.py   # Ablation study
│   │   ├── run_memory_experiment.py           # Run steps 5–7 in one command
│   │   ├── ablation_comparison.py
│   │   ├── prepare_memory_seed_split.py       # Legacy: random seed selection
│   │   └── build_memory_bank_from_seed_analysis.py
│   ├── changed_files/                # Per-issue changed file metadata (JSON)
│   ├── results/                      # Pipeline output (memory bank + eval results)
│   │   └── trs/                      # TRS split run outputs
│   ├── main.py                       # Standalone baseline runner (no memory)
│   └── requirements-dev.txt          # Baselines dependencies
│
├── scripts/                          # Dataset split & analysis scripts
│   ├── analyze_repo_similarity.py    # Step 0: preview similarity before splitting
│   ├── temporal_recurrence_split.py  # Step 1: RCSS-detected + temporal split (primary)
│   ├── enrich_split_for_pipeline.py  # Step 2: join splits with parquet fields
│   ├── rcss_split.py                 # Legacy: recurrence-ranked split (reference)
│   ├── similarity_group_split.py     # Legacy: within-cluster chronological split
│   ├── sacs_split.py                 # Legacy: agglomerative clustering split
│   ├── overall_fl_evaluator.py       # Fault localization evaluation
│   ├── token_analysis.py             # Token usage analysis
│   └── comparison.py                 # Result comparison utilities
│
├── dataset/                          # Dataset files and preparation scripts
│   └── lca_dataset.parquet           # Main dataset (567 issues, 103 repos)
│
├── results/                          # Split outputs and benchmark results
│   ├── trs_split/                    # TRS split — primary experimental split
│   │   ├── memory_issues.json        #   103 issues → memory bank (older, verified recurrent)
│   │   ├── eval_issues.json          #   189 issues → evaluation (newer, have memory peer)
│   │   ├── excluded_issues.json      #   183 issues → singletons (112) + no cross-link (71)
│   │   ├── trs_per_issue.csv         #   per-issue: rank, rcss_score, future/past counts, role
│   │   ├── trs_per_repo.csv          #   per-repo: recurrent count, pool sizes, coverage %
│   │   └── split_summary.txt
│   ├── similarity_analysis/          # Step 0 output: read-only similarity preview
│   │   ├── per_repo_summary.csv      #   per-repo similarity stats (no split)
│   │   ├── per_issue_similarity.csv  #   per-issue similar_count, avg_sim, groups
│   │   ├── similarity_pairs.csv      #   all qualifying pairs above θ
│   │   └── analysis_summary.txt
│   ├── rcss_split/                   # Legacy RCSS split outputs
│   ├── similarity_group_split/       # Legacy SGS split outputs
│   ├── sacs_split/                   # Legacy SACS split outputs
│   └── generated_patches.json        # Benchmark patch results
│
├── evaluation_plot/                  # Evaluation visualizations
├── repo/                             # Cloned repositories (one per repo)
├── setup/                            # GitHub org setup scripts
├── run_benchmark.py                  # CI benchmark runner
├── benchmark.py / benchmark_utils.py
├── config.example.yaml               # Config template
├── pyproject.toml
├── ARCHITECTURE.md                   # Full system architecture
└── README.md
```

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

# MiniMax via OpenRouter
MINIMAX_API_KEY=your_openrouter_api_key
MINIMAX_BASE_URL=https://openrouter.ai/api/v1

# Default model (no --model-key flag needed if set here)
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
baseline_repo_folder: "/path/to/cloned/repos"
project_result_dir:   "/path/to/baselines/results/trs"
changed_files_folder: "/path/to/baselines/changed_files"
benchmark_owner:      "your_github_username"
username_gh:          "your_github_username"

# Memory settings (false for baseline run, true for memory run)
memory_enabled: false
memory_writeback_enabled: false
memory_top_k: 3
memory_similarity_threshold: 0.55
```

---

## 3. Install Dependencies

### Baseline pipeline (memory-augmented repair)

```bash
cd baselines
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cd ..
```

### Split and analysis scripts

```bash
python -m venv venv
source venv/bin/activate
pip install sentence-transformers pandas pyarrow tqdm numpy
```

---

## 4. One-Time Setup: Fork Repositories

```bash
baselines/.venv/bin/python setup/repo_setup.py
```

---

## Dataset

The main dataset is loaded from Hugging Face Hub:
`ci-benchmark-user/ci-repair-bench/ci_repair_dataset.parquet`.

| Field | Description |
|-------|-------------|
| `id` | Sequential integer — used as temporal ordering proxy (lower = older commit) |
| `sha_fail` | Failing commit SHA (join key with error analysis data) |
| `repo_owner`, `repo_name` | Repository identity |
| `workflow_name`, `workflow_path` | GitHub Actions workflow |
| `diff` | Ground-truth fix diff |
| `changed_files` | Files changed in the fix |
| `error_type` | High-level error category labels |

**Stats:** 567 issues across 103 repositories.

---

## Similarity Signal

The primary split and analysis scripts use a two-signal weighted cosine similarity focused on failure context and repair pattern:

```
sim(i, j) = 0.3125 × cos(error_doc)
           + 0.6875 × cos(fix_doc)
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalised, O-CRD §3.1)

**Error doc priority — WHY > WHAT > HOW > WHERE:**

| Priority | Field | Reason |
|----------|-------|--------|
| 1st | `overall_failure_reasons` | Primary failure narrative — WHY it failed |
| 2nd | Per-file `reason` | Per-file failure narrative |
| 3rd | `overall_error_types`, `issue_type`, `error_subtype` | WHAT type of failure |
| 4th | `failed_tools`, `failed_command` | HOW the build broke |
| 5th | File paths | WHERE — supporting context only |

**Threshold:** θ = 0.50 — default cutoff for linking recurrent issues within a repository. Pairs above θ are treated as plausible older/newer analogues for benchmark construction.

---

## Memory-Guided Experiment — Full Pipeline

Research question: **"Does similar past CI failure experience in memory help repair future similar failures?"**

The pipeline has two phases of data preparation (Steps 0–2) followed by the LLM-based memory build and evaluation (Steps 3–7).

**Pipeline intent:**
- Fetch only repository-local issues that have at least one similar recurrent peer.
- Split recurrent issues into older `30%` memory issues and newer `70%` eval issues.
- Analyze the `30%` memory issues and save `L1`, `L2`, and `L3` memories.
- Evaluate only the remaining issues that have similarity links to the saved memory issues.

**Minimal command sequence:**
```bash
venv/bin/python3 scripts/analyze_repo_similarity.py
venv/bin/python3 scripts/temporal_recurrence_split.py
venv/bin/python3 scripts/enrich_split_for_pipeline.py \
  --split results/trs_split/memory_issues.json \
  --output baselines/results/trs/trs_memory_seed_issues.json
venv/bin/python3 scripts/enrich_split_for_pipeline.py \
  --split results/trs_split/eval_issues.json \
  --output baselines/results/trs/trs_eval_issues.json
baselines/.venv/bin/python baselines/scripts/analyze_memory_seed_issues.py \
  --seed-file baselines/results/trs/trs_memory_seed_issues.json \
  --model-key MiniMax-M2.5 \
  --output-dir baselines/results/trs
baselines/.venv/bin/python baselines/scripts/build_memory_bank.py \
  --seed-file baselines/results/trs/trs_memory_seed_issues.json \
  --analysis-file baselines/results/trs/seed_log_details.json \
  --model-key MiniMax-M2.5 \
  --output-dir baselines/results/trs
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file baselines/results/trs/trs_eval_issues.json \
  --memory-mode baseline \
  --model-key MiniMax-M2.5
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file baselines/results/trs/trs_eval_issues.json \
  --memory-mode memory \
  --model-key MiniMax-M2.5
```

---

### How the Memory System Works

Memory is organised into three levels:

| Level | File | Scope | Content |
|-------|------|-------|---------|
| L1 | `failure_memory.json` | Per-file | Exact error, fix strategy, and file-level failure reason |
| L2 | `repo_memory.json` | Per-repo | Recurring failure patterns grouped by (repo, error_type) |
| L3 | `cross_memory.json` | Cross-repo | Generalizable principles that apply across repositories |

**Retrieval** (in `baselines/utilities/memory_plugin.py`):
1. **Lexical similarity** (bag-of-words cosine + Jaccard) selects candidates per level
2. **Single LLM call** reranks all candidates, assigns semantic scores, synthesises a diagnostic summary
3. **Score blending**: `final_score = 0.40 × lexical + 0.60 × LLM`
4. **Global gate**: `weighted = 0.60×L1 + 0.30×L2 + 0.10×L3`; if below `0.55`, memory injection is suppressed

Memory is used **only in fault localization** — not in patch generation.

---

### Step 0 — Preview Similarity (read-only, no split yet)

Run this **first** to see which repos have recurring issues, how many similar pairs exist, and validate the threshold before committing to a split.

```bash
venv/bin/python3 scripts/analyze_repo_similarity.py
```

**What it shows:**
- Per-repo table: total issues, similar issues count, eligible %, groups, qualifying pairs, avg/max similarity
- Per-issue table: which issues have the most similar peers
- Distribution stats: p10 / p25 / p50 / p75 / p90 of within-repo cosine similarity

**Outputs → `results/similarity_analysis/`:**

```text
per_repo_summary.csv       ← per-repo similarity stats
per_issue_similarity.csv   ← per-issue: similar_count, avg_sim, group, has_similar
similarity_pairs.csv       ← every qualifying pair (id_a, id_b, sim_score, error_types)
analysis_summary.txt       ← summary stats and next-step hint
```

> If the threshold looks wrong after reviewing the table, change `SIMILARITY_THRESHOLD` in `scripts/temporal_recurrence_split.py` before proceeding.

---

### Step 1 — Split: TRS (Temporal Recurrence Split)

The primary split for the memory experiment. Two explicit phases:

**Phase 1 — RCSS Detection:** Computes full pairwise similarity within each repo. An issue is *recurrent* if it has ≥ 1 similar peer above θ (any direction). Issues with zero similar peers are **singletons** — excluded entirely, as they have nothing to learn from and cannot benefit from memory.

**Phase 2 — Temporal Split within recurrent pool:** Takes only the recurrent issues for each repo. Sorts them chronologically (ascending parquet `id`). Oldest 30% → memory pool; newest 70% → eval pool. Verifies bidirectional temporal links: every eval issue must have a memory peer, every memory issue must have an eval user.

```bash
venv/bin/python3 scripts/temporal_recurrence_split.py
```

**Why this design?**

| Property | Guarantee |
|----------|-----------|
| Only recurrent issues used | Singletons excluded — no noise from unrelated failures |
| Memory older than eval | Temporal ordering enforced — no data leakage |
| Every eval issue has a memory peer | Memory can always potentially help |
| Every memory issue has an eval user | No dead memory entries |

**Results:**

| Category | Count |
|----------|-------|
| Total issues (475, repos ≥3 issues) | — |
| Recurrent (≥1 similar peer, θ=0.60) | 363 |
| Singleton (no peer) — excluded | 112 |
| **Memory** (verified, older 30%) | **103** |
| **Eval** (verified, newer 70%) | **189** |
| Excluded (recurrent but no cross-link) | 71 |
| Repos with ≥3 issues | 47 |

> **Similarity:** `sim = 0.40 × cos(error_doc) + 0.60 × cos(fix_doc)` with **θ=0.60**.
> θ=0.60 is the p80 of the actual within-repo cosine distribution — it captures the top 20% of
> most similar pairs. Genuinely matching issues (same error codes + same fix) score 0.75–0.95,
> safely above this threshold. Unrelated failures score 0.25–0.45, safely below.
>
> `error_doc` uses: error codes (F821, ImportError…) + tools + subtypes — not long narratives.
> `fix_doc` uses: fix category + error codes + actual changed lines from the diff.

**Outputs → `results/trs_split/`:**

```text
memory_issues.json    ← 103 issues (older, proven recurrent, go to memory bank)
eval_issues.json      ← 189 issues (newer, guaranteed similar memory peer)
excluded_issues.json  ← 183 issues (singletons 112 + recurrent no cross-link 71)
trs_per_issue.csv     ← full table: temporal_rank, rcss_score, future_count,
                         past_count, avg_sim, pool, role
trs_per_repo.csv      ← per-repo: recurrent count, mem/eval pool sizes, coverage %
split_summary.txt
```

---

### Step 2 — Enrich: Join Splits with Parquet

The split files only contain similarity metadata. The pipeline scripts (`analyze_memory_seed_issues.py`, `build_memory_bank.py`, `run_repo_eval_subset.py`) additionally need `repo_name`, `diff`, `ground_truth_files`, `changed_files`, and `logs_summary` from the parquet.

```bash
# Memory issues
venv/bin/python3 scripts/enrich_split_for_pipeline.py \
  --split  results/trs_split/memory_issues.json \
  --output baselines/results/trs/trs_memory_seed_issues.json

# Eval issues
venv/bin/python3 scripts/enrich_split_for_pipeline.py \
  --split  results/trs_split/eval_issues.json \
  --output baselines/results/trs/trs_eval_issues.json
```

**Outputs → `baselines/results/trs/`:**

```text
trs_memory_seed_issues.json   ← 103 memory issues, fully enriched
trs_eval_issues.json          ← 189 eval issues, fully enriched
```

---

### Step 3 — Analyze Memory Issues with CILogAnalyzerLLM

Runs the LLM CI log analyser on each memory issue. Automatically clones each repo at the failing commit. **Fully resumable** — re-running skips already-analysed issues.

```bash
baselines/.venv/bin/python baselines/scripts/analyze_memory_seed_issues.py \
  --seed-file  baselines/results/trs/trs_memory_seed_issues.json \
  --model-key  MiniMax-M2.5 \
  --output-dir baselines/results/trs
```

**Output → `baselines/results/trs/`:**

```text
seed_log_details.json            ← structured error context per memory issue
seed_log_analysis_manifest.json  ← run summary and any failures
```

---

### Step 4 — Build L1 / L2 / L3 Memory Bank

Processes each memory issue to extract L1 (per-file), L2 (repo-level), and L3 (cross-repo) records. Writes after every issue — **fully resumable**.

```bash
baselines/.venv/bin/python baselines/scripts/build_memory_bank.py \
  --seed-file     baselines/results/trs/trs_memory_seed_issues.json \
  --analysis-file baselines/results/trs/seed_log_details.json \
  --model-key     MiniMax-M2.5 \
  --output-dir    baselines/results/trs
```

**Outputs → `baselines/results/trs/`:**

```text
failure_memory.json      ← L1: per-file failure records (multiple per issue)
repo_memory.json         ← L2: one repo-level record per issue
cross_memory.json        ← L3: one generalizable principle per issue
memory_bank_summary.json ← record counts
```

---

### Step 5 — Run Eval WITHOUT Memory (Baseline)

Set in `config.yaml`:
```yaml
memory_enabled: false
memory_writeback_enabled: false
project_result_dir: "/full/path/to/CI-REPAIR-BENCH/baselines/results/trs"
```

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file  baselines/results/trs/trs_eval_issues.json \
  --memory-mode baseline \
  --model-key   MiniMax-M2.5
```

By default this runs all repositories present in `trs_eval_issues.json`.

Quick test (5 issues per repo for a few repos):
```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file  baselines/results/trs/trs_eval_issues.json \
  --repos       agno axolotl litellm flower \
  --memory-mode baseline \
  --max-per-repo 5
```

**Results → `baselines/results/trs/MiniMax-M2.5_llm_baseline/`:**

```text
log_details.json          ← CI log analysis per eval issue
fault_localization.json   ← suspected files per issue
generated_patches.json    ← generated fix patches
fl_evaluation.json        ← fault localization evaluation scores
token_report.json         ← LLM token usage and cost
```

---

### Step 6 — Run Eval WITH Memory

Update `config.yaml`:
```yaml
memory_enabled: true
memory_writeback_enabled: false   # must stay false — prevents eval leaking into memory
project_result_dir: "/full/path/to/CI-REPAIR-BENCH/baselines/results/trs"
memory_top_k: 3
memory_similarity_threshold: 0.55  # fallback only; standard ablations use built-in thresholds
```

```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file  baselines/results/trs/trs_eval_issues.json \
  --memory-mode memory \
  --model-key   MiniMax-M2.5
```

**Results → `baselines/results/trs/MiniMax-M2.5_llm_memory/`:**

```text
log_details.json
fault_localization.json
generated_patches.json
fl_evaluation.json
token_report.json
memory_retrieval_log.jsonl   ← per-issue memory retrieval log (what was injected)
```

---

### Step 7 — Compare Baseline vs Memory

```bash
baselines/.venv/bin/python baselines/scripts/compare_memory_runs.py \
  baselines/results/trs/MiniMax-M2.5_llm_baseline \
  baselines/results/trs/MiniMax-M2.5_llm_memory
```

For other models (replace `MiniMax-M2.5`):
```bash
baselines/.venv/bin/python baselines/scripts/compare_memory_runs.py \
  baselines/results/trs/gpt-4o-mini_llm_baseline \
  baselines/results/trs/gpt-4o-mini_llm_memory
```

---

### Full Pipeline at a Glance

```
Step 0  venv/bin/python3 scripts/analyze_repo_similarity.py
          ↳ preview: which repos are recurrent, similarity distribution
          ↳ output: results/similarity_analysis/

Step 1  venv/bin/python3 scripts/temporal_recurrence_split.py
          Phase 1 — RCSS Detection: find recurrent issues (≥1 similar peer)
          Phase 2 — Temporal Split: oldest 30% of recurrent → memory
                                    newest 70% of recurrent → eval
          ↳ output: results/trs_split/memory_issues.json   (103)
                    results/trs_split/eval_issues.json      (189)

Step 2  venv/bin/python3 scripts/enrich_split_for_pipeline.py --split ... --output ...
          ↳ joins with parquet: adds diff, ground_truth_files, repo_name, logs_summary
          ↳ output: baselines/results/trs/trs_memory_seed_issues.json
                    baselines/results/trs/trs_eval_issues.json

Step 3  baselines/.venv/bin/python baselines/scripts/analyze_memory_seed_issues.py \
          --seed-file baselines/results/trs/trs_memory_seed_issues.json ...
          ↳ CILogAnalyzerLLM on each memory issue (resumable)
          ↳ output: baselines/results/trs/seed_log_details.json

Step 4  baselines/.venv/bin/python baselines/scripts/build_memory_bank.py \
          --seed-file ... --analysis-file ... --output-dir ...
          ↳ extract L1/L2/L3 from each memory issue (resumable)
          ↳ output: baselines/results/trs/failure_memory.json
                    baselines/results/trs/repo_memory.json
                    baselines/results/trs/cross_memory.json

Step 5  run_repo_eval_subset.py --memory-mode baseline    ← eval without memory
          ↳ output: baselines/results/trs/MiniMax-M2.5_llm_baseline/

Step 6  run_repo_eval_subset.py --memory-mode memory      ← eval with memory
          ↳ output: baselines/results/trs/MiniMax-M2.5_llm_memory/

Step 7  compare_memory_runs.py baseline_dir memory_dir    ← compare results
```

---

### Pipeline Step Summary

| Step | Script | What it does | Reads | Writes |
|------|--------|-------------|-------|--------|
| 0 | `analyze_repo_similarity.py` | Preview similarity — no split | error_details.json + parquet | `results/similarity_analysis/` |
| 1 | `temporal_recurrence_split.py` | RCSS detect + temporal 30/70 split | error_details.json + parquet | `results/trs_split/` |
| 2 | `enrich_split_for_pipeline.py` | Join splits with parquet fields | split JSONs + parquet | `baselines/results/trs/*.json` |
| 3 | `analyze_memory_seed_issues.py` | CILogAnalyzerLLM on memory issues | enriched memory JSON + parquet | `seed_log_details.json` |
| 4 | `build_memory_bank.py` | Extract L1/L2/L3 records | seed JSON + log details | `failure/repo/cross_memory.json` |
| 5 | `run_repo_eval_subset.py --memory-mode baseline` | Repair pipeline without memory | eval JSON + parquet | `<model>_llm_baseline/` |
| 6 | `run_repo_eval_subset.py --memory-mode memory` | Repair pipeline with memory | eval JSON + memory bank | `<model>_llm_memory/` |
| 7 | `compare_memory_runs.py` | Compare baseline vs memory | both result dirs | comparison report |

### Important Notes

- **Steps 0–2 are already complete** for the TRS split (outputs in `results/trs_split/` and `baselines/results/trs/`).
- **Steps 3–4 are resumable** — re-running skips already-processed issues. Safe to stop and restart.
- **`memory_writeback_enabled` must stay `false`** during evaluation — prevents eval issues from writing back into the memory bank (data leakage).
- Steps 3–4 only need to run **once**. Steps 5–6 can be re-run with different models without rebuilding the memory bank.
- If `MEMCI_LLM_MODEL` is set in `.env`, no `--model-key` flag is needed on any command.

---

### Ablation Study: L1 / L1+L2 / L1+L2+L3

Measure the contribution of each memory level by running the eval three times:

| Ablation | Active levels | Output dir |
|----------|--------------|------------|
| L1 only | File-level memory | `<model>_llm_memory_L1/` |
| L1+L2 | File + repo-level | `<model>_llm_memory_L1L2/` |
| L1+L2+L3 | Full memory | `<model>_llm_memory/` |

Reuse the baseline `log_details.json` to avoid re-running CI log analysis. By default the ablation script reads:
`baselines/results/MiniMax-M2.5_llm_baseline/log_details.json`

```bash
# L1 only
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1

# L1+L2
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1+L2

# L1+L2+L3
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1+L2+L3
```

If you want to use another existing analysis file, pass it explicitly:

```bash
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \
  --ablation-levels L1 \
  --source-log-details baselines/results/MiniMax-M2.5_llm_baseline/log_details.json
```

Or run from scratch, which will run CI log analysis again:
```bash
baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file baselines/results/trs/trs_eval_issues.json \
  --memory-mode memory --ablation-levels L1

baselines/.venv/bin/python baselines/scripts/run_repo_eval_subset.py \
  --split-file baselines/results/trs/trs_eval_issues.json \
  --memory-mode memory --ablation-levels L1+L2
```

---

## Run Without Memory (Standalone Baseline)

To run the repair pipeline directly without any memory bank:

Make sure `config.yaml` has `memory_enabled: false`, then:

```bash
baselines/.venv/bin/python baselines/main.py
```

Results written to `baselines/results/<model>_llm_baseline/`.

---

## Benchmark Execution (CI Verification)

### Install benchmark dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt    # or: pip install pandas pyarrow requests PyYAML
```

### Run the Benchmark

```bash
python run_benchmark.py
```

Results written to the `out_folder` set in `config.yaml`:

```text
jobs_ids.jsonl       ← job identifiers sent to GitHub
jobs_results.jsonl   ← results for each job
jobs_awaiting.jsonl  ← jobs still running (normally empty)
jobs_invalid.jsonl   ← invalid jobs (normally empty)
```

### Re-check CI Outcome

```bash
python recheck_waiting_jobs.py
```

---

## Legacy Splits (Reference)

Earlier split strategies are kept in `scripts/` for comparison. They are superseded by TRS for the primary experiment.

### RCSS — Recurrence-Coherence Stratified Split

Ranks all qualifying issues by recurrence count and average similarity (no temporal ordering). Top 30% → memory.

```bash
venv/bin/python3 scripts/rcss_split.py
# → results/rcss_split/memory_30pct.json   (145 issues)
# → results/rcss_split/eval_70pct.json     (422 issues)
```

### SGS — Similarity-Group Split

Clusters issues via Union-Find, then splits each group 70/30 chronologically. Issues with no similar peer excluded.

```bash
venv/bin/python3 scripts/similarity_group_split.py
# → results/similarity_group_split/memory_issues.json  (317 issues)
# → results/similarity_group_split/test_issues.json    (166 issues)
```

### SACS — Semantic Agglomerative Clustering Split (oldest)

Agglomerative clustering with θ = 0.60.

```bash
venv/bin/python3 scripts/sacs_split.py
```
