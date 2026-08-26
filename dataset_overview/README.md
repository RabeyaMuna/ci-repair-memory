# Dataset Overview

**Everything you need for the CI-REPAIR-BENCH paper in 6 files.**

---

## 📄 **Main Document** (START HERE!)

**`DATASET_OVERVIEW_FOR_PAPER.md`** ⭐
- All statistics, tables, and text for paper
- LaTeX tables ready to copy-paste
- Citation-ready text for abstract/intro/methods
- Complete methodology
- **Read this first!**

---

## 🔧 **Scripts** (Run These to Generate Data)

### 1. **`generate_detailed_overview.py`**
Generate complete dataset statistics.

```bash
python generate_detailed_overview.py
```

**Outputs:** `detailed_paper_statistics.json`  
**Contains:** Repository stats, failure types, code changes, multi-problem analysis

---

### 2. **`analyze_validation_jobs_steps.py`**
Analyze validation jobs and steps from filtered data.

```bash
python analyze_validation_jobs_steps.py
```

**Outputs:** `validation_jobs_steps_analysis.json`  
**Contains:** Jobs/steps counts, failed jobs/steps, distributions

**Prerequisites:** Run `data_managment/scripts/extract_validation_jobs_and_steps.py` first

---

### 3. **`compute_benchmark_recurrence.py`** (Optional)
Analyze recurrence using TWO complementary views (matches paper §\ref{sec:instance-similarity}).

```bash
# Install dependencies
pip install scikit-learn

# Run full dataset (recommended for paper)
python compute_benchmark_recurrence.py

# Or sample for quick testing
python compute_benchmark_recurrence.py --sample-size 100
```

**Outputs:** `benchmark_recurrence_analysis.json`, `recurrence_pairs.csv`, LaTeX table  
**Contains:** Jaccard + TF-IDF similarities, per-repository statistics, chronological analysis

**Two Views:**
- **Structural** (Jaccard): failure type, tools, packages, changed files
- **Lexical** (TF-IDF + Cosine): failure + workflow + repair text

**Table Metrics:**
- Overall nearest neighbor
- Within repository
- Cross repository
- Historical predecessor (chronologically earlier only)

**Note:** This is offline benchmark characterization (includes ground-truth repairs)

---

## 📊 **Generated Data**

### 1. **`detailed_paper_statistics.json`**
Complete dataset statistics including:
- Dataset composition (repos, languages, instances)
- Failure type distribution
- Multi-problem analysis
- Code change statistics
- Repository distribution

### 2. **`validation_jobs_steps_analysis.json`**
Validation metrics including:
- Total jobs/steps across all commits
- Failed jobs/steps
- Per-instance distributions
- Top failed job/step names

---

## ⚡ **Quick Start**

```bash
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset_overview

# 1. Read the main document
cat DATASET_OVERVIEW_FOR_PAPER.md

# 2. Generate latest statistics
python generate_detailed_overview.py
python analyze_validation_jobs_steps.py

# 3. Done! Use numbers from the main document for your paper
```

---

## 📋 **Workflow**

```
1. Extract validation (from data_managment)
   ↓
2. Generate statistics (this directory)
   ↓
3. Use DATASET_OVERVIEW_FOR_PAPER.md
   → Copy LaTeX tables to paper
   → Use citation-ready text
   → Reference numbers
```

---

## 🎯 **For Your Paper**

1. Open `DATASET_OVERVIEW_FOR_PAPER.md`
2. Copy Table 1 (Dataset Overview) → paper
3. Copy Table 2 (Failure Types) → paper
4. Use citation-ready text for abstract/methods
5. Reference validation numbers (~7,500 jobs, ~52,000 steps)

---

**That's it! Just 6 files - clean and organized.**

---

**Contact:** rabeya@beaverly.ai  
**Last Updated:** August 25, 2026
