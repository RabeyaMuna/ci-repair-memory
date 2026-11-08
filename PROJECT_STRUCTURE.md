# CI-REPAIR-BENCH Project Structure

## Core Execution
- `run_benchmark.py` - Main benchmark runner
- `evaluate.py` - Evaluation script
- `config.example.yaml` - Example configuration
- `config.yaml` - User configuration (gitignored)

## Main Folders

### `/benchmark/`
Core benchmark implementation
- `benchmark.py` - Main benchmark class
- `benchmark_functions.py` - Helper functions (repo cloning, patching, etc.)
- `workflow_patch_validator.py` - Validates workflow patches
- `load_config.py` - Configuration loader

### `/baselines/`
Baseline implementations and comparisons
- `main.py` - Baseline runner
- `BRaIn/` - BRaIn baseline implementation
- `ci_repair/` - CI-Repair baseline

### `/analysis/`
Result analysis and statistics
- `analyze_failure_type_success_rates.py` - Success rate analysis
- `analyze_results.py` - General results analysis
- `analyze_unsuccessful_causes.py` - Failure analysis
- `statistical_significance_test.py` - Statistical tests
- `generate_section6_statistics.py` - Paper statistics
- `export_commit_counts_csv.py` - Export statistics

### `/maintenance/`
Infrastructure and monitoring
- `monitor_and_track_workflows.py` - Workflow monitoring
- `recheck_waiting_jobs.py` - Job recheck utility
- `enable_all_workflows.py` - Workflow enabler
- `extract_patches.py` - Patch extraction
- `generate_inspection_report.py` - Inspection reports
- `run_evaluation.py` - Evaluation runner

### `/setup/`
Setup and verification
- `setup.sh` - Installation script
- `verify_setup.py` - Setup verification
- `repo_setup.py` - Repository setup
- `add_collab.py` - Collaborator management

### `/scripts/`
Utility scripts for data processing
- Analysis and verification scripts
- Failure type classification
- Ground truth verification

### `/docs/`
Documentation
- `guides/QUICKSTART.md` - Quick start guide
- `guides/SETUP_GUIDE.md` - Detailed setup instructions
- Additional documentation

### `/dataset/` (gitignored)
Dataset management scripts
- Fetch, process, and manage benchmark datasets

### `/evaluation/`
Evaluation utilities and visualization

### `/data_managment/`
Data management utilities

## Gitignored Folders
- `/repo/` - Cloned repositories
- `/results/` - Benchmark results
- `/venv/` - Python virtual environment
- `/generated_patches_list/` - Generated patches
- `/dataset/` - Dataset files

## Configuration Files
- `pyproject.toml` - Poetry project configuration
- `poetry.lock` - Poetry dependencies lock
- `requirements.txt` - Pip requirements
- `.gitignore` - Git ignore rules
- `config.yaml` - User configuration (create from config.example.yaml)
