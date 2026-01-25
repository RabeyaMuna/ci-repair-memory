# Long Code Arena CI Builds Repair Benchmark

This repository provides a benchmark for evaluating automated CI build repair methods.
It downloads failing repositories, applies repair strategies, runs GitHub Actions,
and evaluates whether the CI passes.

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
/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/config.example.yaml
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

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## One Time Setup: Forking Repositories
To run the benchmark using repositories own GitHub account, fork everything first:

```bash
python repo_setup/bulk_fork_repositories.py

Running the Benchmark

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


## To the baseline(CI Repair System):

- Baselines are located in:
```bash
cd baselines
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

- Re-check Waiting CI outcome and Re-evaluate:

Sometimes GitHub Actions runs slowly and not all jobs finish in the initial time window. For this reason, outcome can be rechecked for the pushed commits and update results without pushing again.

```bash
recheck_waiting_jobs.py
```
