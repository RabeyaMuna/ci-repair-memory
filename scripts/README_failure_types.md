# Failure Type Cleaning and Dataset Update

## Overview

This directory contains scripts to clean duplicate failure types from classification results and update the main dataset with unique failure types.

## Problem Solved

The failure classification process sometimes generates duplicate failure types for the same issue. For example:
- **Before**: `['Code Formatting', 'Linting', 'Code Formatting', 'Linting']`
- **After**: `['Code Formatting', 'Linting']`

The script removes these duplicates while preserving:
- Different failure types (e.g., both "Linting" and "Code Formatting" are kept)
- Order of first occurrence
- Associated subtypes and details

## Script: `clean_and_update_failure_types.py`

### What it does:
1. ✅ Loads `failure_classifications.json`
2. ✅ Identifies and removes duplicate failure types for each issue
3. ✅ Saves cleaned results to `failure_classifications_cleaned.json`
4. ✅ Updates the main dataset (`lca_dataset.parquet`) with unique failure types
5. ✅ Adds three new columns to the dataset:
   - `failure_types`: List of unique failure types
   - `failure_subtypes`: Corresponding subtypes
   - `num_failure_types`: Count of failure types per issue

### Usage:

```bash
# Basic usage (overwrites original dataset)
python3 scripts/clean_and_update_failure_types.py

# With backup (recommended - creates backup before overwriting)
python3 scripts/clean_and_update_failure_types.py --backup

# Custom input/output paths
python3 scripts/clean_and_update_failure_types.py \
  --classifications results/failure_classifications.json \
  --dataset dataset/lca_dataset.parquet \
  --output-classifications results/failure_classifications_cleaned.json \
  --output-dataset dataset/lca_dataset_updated.parquet \
  --backup
```

### Results from Latest Run (2024-09-11):

**Cleaning Statistics:**
- Total issues processed: 565
- Issues with duplicates found: 364 (64.4%)
- Duplicate entries removed: 3,561
- Verification: ✅ 0 duplicates remaining

**Dataset Statistics:**
- Issues with failure types: 565 (100%)
- Issues with multiple types: 474 (83.9%)
- Average failure types per issue: 3.41 (down from 9.71)

**Top Failure Types:**
1. Linting: 303 issues
2. Code Formatting: 294 issues
3. Test Failure: 278 issues
4. Type Checking: 243 issues
5. Dependency Issues: 242 issues
6. Runtime Error: 180 issues
7. Configuration Error: 167 issues
8. Doc/Docstring: 81 issues
9. Assertion Error: 48 issues
10. Package Install Error: 37 issues

### Files Generated:

1. **`results/failure_classifications_cleaned.json`**: Cleaned classification results with no duplicates
2. **`dataset/lca_dataset.parquet`**: Updated main dataset with unique failure types
3. **`dataset/lca_dataset_backup.parquet`**: Backup of original dataset (if `--backup` used)

### Example Transformations:

```
ID 64:
  Before: ['Code Formatting', 'Linting', 'Code Formatting', 'Linting']
  After:  ['Code Formatting', 'Linting']

ID 27:
  Before: ['Doc/Docstring', 'Doc/Docstring', 'Doc/Docstring', 'Runtime Error']
  After:  ['Doc/Docstring', 'Runtime Error']

ID 11:
  Before: ['Code Formatting', 'Code Formatting', 'Code Formatting']
  After:  ['Code Formatting']
```

## Integration with Existing Scripts

This script is designed to work **after** running the failure classification:

```bash
# Step 1: Classify failures (existing script)
python3 data_managment/failure_type/classify_failure_types.py \
  --dataset dataset/lca_dataset.parquet \
  --output results/failure_classifications.json

# Step 2: Clean duplicates and update dataset (new script)
python3 scripts/clean_and_update_failure_types.py --backup
```

## Notes

- The script preserves the order of first occurrence when removing duplicates
- Different failure types are kept (e.g., "Linting" AND "Code Formatting")
- Only exact duplicate failure types are removed (e.g., "Code Formatting" appearing twice)
- The main dataset is updated in-place (unless a different output path is specified)
- Use `--backup` to create a safety backup before overwriting the original dataset
