# Dataset Overview

This directory contains documentation and a single script to generate complete benchmark statistics.

## 📊 Generate All Statistics

**Single command to generate everything:**

```bash
python dataset_overview/generate_complete_statistics.py
```

### What it generates:

All outputs are saved to `results/data/`:

1. **`complete_benchmark_statistics.json`** - Complete statistics in JSON format
   - Overall benchmark statistics (565 instances, 101 repos, 12 failure types)
   - Repository aggregate statistics
   - Per-repository detailed statistics (all 101 repositories)
   - Failure type distribution

2. **`overall_statistics.json`** - Quick reference for paper
   - Scale, CI validation, code changes, failure types
   - All with total, mean, and median values

3. **`repository_statistics.json`** - Per-repository breakdown
   - Aggregate stats across all 101 repos
   - Individual stats for each repository

4. **`benchmark_overview_table.tex`** - LaTeX table for appendix

5. **`STATISTICS_REPORT.txt`** - Human-readable complete report

Additional failure type data saved to `results/paper/failure_type_distribution.json`

## 📈 Key Statistics

- **101 unique repositories** (by repo_name)
- **565 repair instances** total
- **12 failure categories**
- **83.7%** multi-failure instances (mean 3.38 types per instance)
- **10.6 files** and **243.7 lines** changed on average (median: 3 files, 24 lines)
- **5.0 CI jobs** and **27.5 steps** on average (median: 2 jobs, 14 steps)
- **~3.75 commits** per repair trajectory

## 📁 Directory Contents

### Scripts
- **`generate_complete_statistics.py`** - Main statistics generation script

### Documentation
- **`README_STATISTICS.md`** - Detailed documentation of generated statistics
- **`PAPER_STATISTICS_SUMMARY.md`** - Paper-ready statistics summary
- **`DATASET_OVERVIEW_FOR_PAPER.md`** - Original dataset overview
- **`SIMILARITY_ALGORITHM.md`** - Similarity computation details
- **`RECURRENCE_ANALYSIS_SUMMARY.md`** - Recurrence analysis
- **`PAPER_SECTION_RECURRENCE_FINAL.tex`** - LaTeX section on recurrence

### Legacy Files (from previous analyses)
- `detailed_paper_statistics.json`
- `recurrence_analysis_final.json`
- `similarity_distribution.json`
- `validation_jobs_steps_analysis.json`

## 🎯 For Your Paper

See `README_STATISTICS.md` and `PAPER_STATISTICS_SUMMARY.md` for:
- Complete statistical breakdown
- Paper-ready text and LaTeX
- Key numbers for abstract/introduction/results

---

**Note**: All generated statistics files are in `results/data/` directory, not in this directory.
