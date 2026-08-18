# Validation Steps Analysis Scripts

This directory contains scripts to analyze validation steps and measure patch effectiveness in the CI-REPAIR-BENCH.

## Overview

The analysis workflow has 3 main steps:

```
1. fetch_validation_steps.py    → Fetch detailed validation steps from GitHub API
2. map_ci_logs_to_steps.py      → Map CI log failures to validation steps
3. compare_patch_effectiveness.py → Compare before/after patch results
```

## Complete Workflow

```bash
# 1. Run your benchmark to generate patched workflow results
cd /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH
python run_benchmark.py

# 2. Fetch detailed validation steps from GitHub API
python scripts/analysis/fetch_validation_steps.py

# 3. Map CI logs to validation steps  
python scripts/analysis/map_ci_logs_to_steps.py

# 4. Compare patch effectiveness
python scripts/analysis/compare_patch_effectiveness.py
```

## Key Features

- **Handles multiple formats**: Job names, step names, matrix jobs
- **Flexible matching**: Exact, partial, and fuzzy matching for CI logs
- **Comprehensive tracking**: Job-level and step-level validation data
- **Effectiveness metrics**: Before/after comparison with resolution rates
