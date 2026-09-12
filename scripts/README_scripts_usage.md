# Failure Type Management Scripts

This directory contains scripts for managing and analyzing failure types in the CI-REPAIR-BENCH dataset.

---

## 📜 Available Scripts

### 1. **`update_instance_failure_types.py`**
Update failure types for individual instances in the dataset.

### 2. **`analyze_failure_type_statistics.py`**
Generate comprehensive statistics about failure types across the entire dataset.

### 3. **`clean_and_update_failure_types.py`**
Clean duplicate failure types from classifications and update the dataset.

---

## 🔧 Script 1: Update Instance Failure Types

### Purpose
Allows you to modify failure types for specific instances in the dataset.

### Features
- ✅ Replace failure types for a single instance
- ✅ Add new failure types to an instance
- ✅ Remove specific failure types from an instance
- ✅ Batch update multiple instances from CSV/JSON file
- ✅ Automatic backup creation

### Usage

#### Single Instance Update

**Replace failure types:**
```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 64 \
  --failure-types "Code Formatting,Linting" \
  --mode replace \
  --backup
```

**Add failure types:**
```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 64 \
  --failure-types "Test Failure,Assertion Error" \
  --mode add \
  --backup
```

**Remove failure types:**
```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 64 \
  --failure-types "Code Formatting" \
  --mode remove \
  --backup
```

#### Batch Update from File

**CSV format** (`updates.csv`):
```csv
issue_id,failure_types,mode
64,"Code Formatting,Linting",replace
27,"Doc/Docstring,Runtime Error",replace
11,"Code Formatting",replace
```

**Run batch update:**
```bash
python3 scripts/update_instance_failure_types.py \
  --batch updates.csv \
  --backup
```

**JSON format** (`updates.json`):
```json
[
  {
    "issue_id": "64",
    "failure_types": ["Code Formatting", "Linting"],
    "mode": "replace"
  },
  {
    "issue_id": "27",
    "failure_types": ["Doc/Docstring", "Runtime Error"],
    "mode": "add"
  }
]
```

### Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--classifications` | Path to classifications JSON | `results/failure_classifications_cleaned.json` | No |
| `--dataset` | Path to dataset parquet | `dataset/lca_dataset.parquet` | No |
| `--issue-id` | Issue ID to update | - | Yes (single) |
| `--failure-types` | Comma-separated failure types | - | Yes (single) |
| `--subtypes` | Comma-separated subtypes | - | No |
| `--mode` | Update mode: `replace`, `add`, `remove` | `replace` | No |
| `--batch` | CSV/JSON file for batch updates | - | Yes (batch) |
| `--backup` | Create backup before updating | `False` | No |

---

## 📊 Script 2: Analyze Failure Type Statistics

### Purpose
Generate comprehensive statistics and insights about failure types across the entire dataset.

### Features
- ✅ Overall failure type distribution
- ✅ Failure type co-occurrence analysis
- ✅ Repository-level statistics
- ✅ Temporal trends (monthly)
- ✅ Complexity distribution (single vs multiple types)
- ✅ Top subtypes analysis
- ✅ Both JSON and text report formats

### Usage

**Generate both JSON and text reports:**
```bash
python3 scripts/analyze_failure_type_statistics.py
```

**Generate only JSON:**
```bash
python3 scripts/analyze_failure_type_statistics.py --format json
```

**Generate only text report:**
```bash
python3 scripts/analyze_failure_type_statistics.py --format txt
```

**Custom paths:**
```bash
python3 scripts/analyze_failure_type_statistics.py \
  --classifications results/failure_classifications_cleaned.json \
  --dataset dataset/lca_dataset.parquet \
  --output-dir results \
  --format both
```

### Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| `--classifications` | Path to classifications JSON | `results/failure_classifications_cleaned.json` | No |
| `--dataset` | Path to dataset parquet | `dataset/lca_dataset.parquet` | No |
| `--output-dir` | Output directory | `results` | No |
| `--format` | Output format: `json`, `txt`, `both` | `both` | No |

### Output Files

1. **`failure_type_statistics.json`**: Complete statistics in JSON format
2. **`failure_type_statistics_report.txt`**: Human-readable text report

### Statistics Included

#### 1. **Overall Statistics**
- Total instances
- Unique failure types and subtypes
- Single vs multiple failure type distribution
- Average and maximum types per instance

#### 2. **Failure Type Distribution**
- Complete ranking of all failure types
- Counts and percentages for each type

#### 3. **Complexity Distribution**
- Breakdown by number of failure types per instance
- Shows how many instances have 1, 2, 3... types

#### 4. **Co-occurrence Analysis**
- Which failure types commonly appear together
- Top 20 failure type pairs

#### 5. **Repository Statistics**
- Failure type distribution per repository
- Top repositories by instance count

#### 6. **Temporal Trends**
- Monthly breakdown of failure types
- Trends over time

#### 7. **Top Subtypes**
- Most common specific failure subtypes
- Top 20 subtypes across all instances

### Sample Output (Latest Run)

```
Total instances: 565
Unique failure types: 25
Average types per instance: 3.41

Top 5 failure types:
  1. Linting: 303 (53.6%)
  2. Code Formatting: 294 (52.0%)
  3. Test Failure: 278 (49.2%)
  4. Type Checking: 243 (43.0%)
  5. Dependency Issues: 242 (42.8%)

Top 3 co-occurrences:
  1. Code Formatting + Linting: 201 instances
  2. Dependency Issues + Test Failure: 166 instances
  3. Linting + Type Checking: 146 instances
```

---

## 🧹 Script 3: Clean and Update Failure Types

### Purpose
Remove duplicate failure types from classifications and update the dataset.

### Usage

```bash
# With backup (recommended)
python3 scripts/clean_and_update_failure_types.py --backup

# Without backup
python3 scripts/clean_and_update_failure_types.py
```

### What It Does
- Removes duplicate failure types (e.g., `['Test Failure', 'Test Failure']` → `['Test Failure']`)
- Keeps different types together (e.g., `['Linting', 'Code Formatting']` stays unchanged)
- Updates both classifications JSON and dataset parquet

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--classifications` | Input classifications JSON | `results/failure_classifications.json` |
| `--dataset` | Input dataset parquet | `dataset/lca_dataset.parquet` |
| `--output-classifications` | Output classifications JSON | `results/failure_classifications_cleaned.json` |
| `--output-dataset` | Output dataset parquet | `dataset/lca_dataset.parquet` |
| `--backup` | Create backup before overwriting | `False` |

---

## 🔄 Complete Workflow

### Initial Setup and Classification

```bash
# Step 1: Classify failure types (if not already done)
python3 data_managment/failure_type/classify_failure_types.py \
  --dataset dataset/lca_dataset.parquet \
  --output results/failure_classifications.json

# Step 2: Clean duplicates and update dataset
python3 scripts/clean_and_update_failure_types.py --backup

# Step 3: Generate statistics
python3 scripts/analyze_failure_type_statistics.py
```

### Update Specific Instances

```bash
# Update a single instance
python3 scripts/update_instance_failure_types.py \
  --issue-id 64 \
  --failure-types "Code Formatting,Linting,Test Failure" \
  --mode replace \
  --backup

# Re-generate statistics after updates
python3 scripts/analyze_failure_type_statistics.py
```

### Batch Updates

```bash
# Create updates.csv with your changes
# Then run batch update
python3 scripts/update_instance_failure_types.py \
  --batch updates.csv \
  --backup

# Re-generate statistics
python3 scripts/analyze_failure_type_statistics.py
```

---

## 📁 File Structure

```
CI-REPAIR-BENCH/
├── scripts/
│   ├── update_instance_failure_types.py      # Update instances
│   ├── analyze_failure_type_statistics.py    # Generate statistics
│   ├── clean_and_update_failure_types.py     # Clean duplicates
│   └── README_scripts_usage.md               # This file
├── results/
│   ├── failure_classifications.json          # Original classifications
│   ├── failure_classifications_cleaned.json  # Cleaned classifications
│   ├── failure_type_statistics.json          # Statistics (JSON)
│   └── failure_type_statistics_report.txt    # Statistics (text)
└── dataset/
    ├── lca_dataset.parquet                   # Main dataset
    └── lca_dataset_backup.parquet            # Backup (if --backup used)
```

---

## 🎯 Common Use Cases

### Use Case 1: Fix Incorrect Classification
```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 123 \
  --failure-types "Type Checking,Linting" \
  --mode replace \
  --backup
```

### Use Case 2: Add Missing Failure Type
```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 456 \
  --failure-types "Documentation" \
  --mode add \
  --backup
```

### Use Case 3: Analyze Dataset After Updates
```bash
python3 scripts/analyze_failure_type_statistics.py --format both
```

### Use Case 4: Batch Correct Multiple Instances
```bash
# Create updates.csv
# Then run:
python3 scripts/update_instance_failure_types.py --batch updates.csv --backup
python3 scripts/analyze_failure_type_statistics.py
```

---

## ⚠️ Important Notes

1. **Always use `--backup`** when updating the dataset to create safety backups
2. **Regenerate statistics** after any updates to keep insights current
3. **Validate updates** by checking the cleaned data before proceeding
4. **Keep backups** of original files before major changes
5. **Use batch mode** for updating multiple instances efficiently

---

## 📞 Support

For issues or questions:
1. Check this documentation
2. Review the script's `--help` output
3. Examine the generated statistics reports for insights

---

**Last Updated**: 2024-09-11
