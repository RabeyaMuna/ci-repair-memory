# BRaIn Fault Localization for CI-REPAIR-BENCH

Automated fault localization using BRaIn (Bug Localization with Intelligent Relevance Feedback).

## Quick Start

```bash
# 1. Activate venv
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
source venv/bin/activate

# 2. Start Elasticsearch
brew services start elasticsearch

# 3. Run BRaIn FL (150 instances, ~1-2 hours)
cd baselines/BRaIn
python auto_brain_fl.py gpt-5-mini_llm --limit 150
```

## What It Does

1. Identifies required repositories from log_details.json
2. Fetches missing commits (git fetch --all)
3. Indexes files into Elasticsearch
4. Runs 4-stage BRaIn FL:
   - BM25 search (top-50 candidates)
   - LLM method-level relevance
   - Query expansion with PageRank
   - Re-scoring and final ranking

## Output

Results: `../results/gpt-5-mini_llm/brain_fl_results.json`

## Files

- `auto_brain_fl.py` - Main script (use this!)
- `index_repositories.py` - Elasticsearch indexing
- `run_brain_fl_full.py` - BRaIn FL algorithm
- `check_missing_repos.py` - Diagnostic tool
- `src/` - BRaIn source code

## Prerequisites

- Python 3.10+
- Elasticsearch running on localhost:9200
- OpenAI API key in `../../.env`
- Virtual environment with dependencies installed

## Troubleshooting

**Elasticsearch not running:**
```bash
brew services start elasticsearch
curl http://localhost:9200
```

**Missing commits:**
```bash
cd ../repo_cloned
for dir in */; do (cd "$dir" && git fetch --all); done
```

**API key issues:**
```bash
echo "OPENAI_API_KEY=sk-..." >> ../../.env
```

## About BRaIn

BRaIn (ICPC 2025) improves bug localization by 87.6% (MAP), 89.5% (MRR), and 48.8% (HIT@K) over baselines using Intelligent Relevance Feedback with LLMs.

Paper: https://arxiv.org/abs/2501.10542
