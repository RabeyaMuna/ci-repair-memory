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

You must provide both of the following:

1. `.env`  
   Stores secrets (tokens, API keys)

2. `config.yaml`  
   Stores benchmark runtime configuration (paths, usernames, language)

A template config is available at:

```text
config.example.yaml
```


## 1. Create `.env` for secrets

Create a file named `.env` in the repository root and add the following values:

```text
# GitHub
GITHUB_USERNAME=your_github_username
GITHUB_TOKEN=your_github_personal_access_token

# Hugging Face
HF_TOKEN=your_huggingface_token

# OpenAI (required for deepeval)
OPENAI_API_KEY=your_openai_api_key
```

## 2. Create `config.yaml` for benchmark configuration

Copy the example configuration file:

```bash
cp config.example.yaml config.yaml
```

### Notes:

- username_gh should match GITHUB_USERNAME
- Environment Setup
  Use separate virtual environments for:
  1. Benchmark execution
  2. Baseline experiments
 

## Benchmark execution:
### One-Time Setup: Forking Repositories
To run the benchmark using repositories own GitHub account, fork everything first:

```bash
python setup_github/bulk_fork_repositories.py
```
Commands to set up and run benchmark:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## One Time Setup: Forking Repositories
To run the benchmark using repositories own GitHub account, fork everything first:

```bash
python setup/bulk_fork_repositories.py
```

- Run the Benchmark

```bash
python run_benchmark.py
```

Results are written to out_folder:

```text
jobs_ids.jsonl
Job identifiers sent to GitHub

jobs_results.jsonl
Results for each job

jobs_awaiting.jsonl
Jobs still running (normally empty)

jobs_invalid.jsonl
Invalid jobs (normally empty)
```

 ### Re-check CI outcome and Re-evaluate:

Sometimes GitHub Actions runs slowly, and not all jobs finish in the initial time window. For this reason, the outcome can be rechecked for the pushed commits and update results without pushing again.

```bash
recheck_waiting_jobs.py
```

## To execute baseline(CI Repair System):

- To run the baseline project:
```bash
cd baselines
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python main.py
```
