# Quick Start Guide - Failure Type Scripts

## 🚀 Quick Commands

### 1. Update a Single Instance

```bash
# Replace failure types for issue #64
python3 scripts/update_instance_failure_types.py \
  --issue-id 64 \
  --failure-types "Code Formatting,Linting" \
  --mode replace \
  --backup
```

### 2. Generate Statistics Report

```bash
# Generate both JSON and text reports
python3 scripts/analyze_failure_type_statistics.py
```

### 3. View Statistics Report

```bash
# View the generated text report
cat results/failure_type_statistics_report.txt
```

---

## 📝 Common Tasks

### Task 1: Fix an Instance with Wrong Failure Type

**Problem**: Instance #123 has "Test Failure" but should be "Type Checking"

```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 123 \
  --failure-types "Type Checking" \
  --mode replace \
  --backup
```

### Task 2: Add a Missing Failure Type

**Problem**: Instance #456 needs "Documentation" added to its existing types

```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 456 \
  --failure-types "Documentation" \
  --mode add \
  --backup
```

### Task 3: Remove an Incorrect Failure Type

**Problem**: Instance #789 has "Runtime Error" that shouldn't be there

```bash
python3 scripts/update_instance_failure_types.py \
  --issue-id 789 \
  --failure-types "Runtime Error" \
  --mode remove \
  --backup
```

### Task 4: Update Multiple Instances at Once

**Step 1**: Create `updates.csv`:
```csv
issue_id,failure_types,mode
64,"Code Formatting,Linting",replace
27,"Doc/Docstring,Runtime Error",replace
11,"Code Formatting",replace
```

**Step 2**: Run batch update:
```bash
python3 scripts/update_instance_failure_types.py \
  --batch updates.csv \
  --backup
```

### Task 5: Get Current Statistics

```bash
# Generate fresh statistics
python3 scripts/analyze_failure_type_statistics.py --format both

# View the summary
head -100 results/failure_type_statistics_report.txt
```

---

## 📊 Check Current Dataset State

### View Overall Statistics

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('dataset/lca_dataset.parquet')
print(f'Total instances: {len(df)}')
print(f'With failure types: {(df[\"num_failure_types\"] > 0).sum()}')
print(f'Average types per instance: {df[\"num_failure_types\"].mean():.2f}')
"
```

### View Specific Instance

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('dataset/lca_dataset.parquet')
instance = df[df['id'] == 64].iloc[0]
print(f'ID: {instance[\"id\"]}')
print(f'Failure types: {instance[\"failure_types\"]}')
print(f'Subtypes: {instance[\"failure_subtypes\"]}')
print(f'Count: {instance[\"num_failure_types\"]}')
"
```

### Search for Instances with Specific Failure Type

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('dataset/lca_dataset.parquet')

# Find all instances with 'Test Failure'
test_failures = df[df['failure_types'].apply(lambda x: 'Test Failure' in x)]
print(f'Instances with Test Failure: {len(test_failures)}')
print(f'IDs: {test_failures[\"id\"].tolist()[:10]}...')
"
```

---

## 🔍 Analyze Results

### Top 10 Failure Types

```bash
python3 -c "
import json
with open('results/failure_type_statistics.json') as f:
    stats = json.load(f)
    
counts = stats['basic_statistics']['failure_type_counts']
sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

print('Top 10 Failure Types:')
for rank, (ftype, count) in enumerate(sorted_types, 1):
    print(f'{rank:2}. {ftype:30} {count:4} instances')
"
```

### Most Common Co-occurrences

```bash
python3 -c "
import json
with open('results/failure_type_statistics.json') as f:
    stats = json.load(f)
    
pairs = stats['cooccurrence']['top_pairs'][:10]

print('Top 10 Failure Type Pairs:')
for rank, (type1, type2, count) in enumerate(pairs, 1):
    print(f'{rank:2}. {type1:25} + {type2:25} = {count:3} instances')
"
```

### Repository Breakdown

```bash
python3 -c "
import json
with open('results/failure_type_statistics.json') as f:
    stats = json.load(f)
    
repos = stats['repository_statistics']
sorted_repos = sorted(repos.items(), 
                     key=lambda x: x[1]['total_instances'], 
                     reverse=True)[:10]

print('Top 10 Repositories by Instance Count:')
for rank, (repo, data) in enumerate(sorted_repos, 1):
    print(f'{rank:2}. {repo:40} {data[\"total_instances\"]:3} instances')
"
```

---

## ⚙️ Advanced Usage

### Export Statistics to CSV

```bash
python3 -c "
import json
import pandas as pd

with open('results/failure_type_statistics.json') as f:
    stats = json.load(f)

# Convert to DataFrame
counts = stats['basic_statistics']['failure_type_counts']
df = pd.DataFrame(list(counts.items()), columns=['Failure_Type', 'Count'])
df = df.sort_values('Count', ascending=False)

# Save to CSV
df.to_csv('results/failure_type_distribution.csv', index=False)
print('✓ Saved to results/failure_type_distribution.csv')
"
```

### Find Instances with Multiple Types

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('dataset/lca_dataset.parquet')

# Get instances with 5+ failure types
complex = df[df['num_failure_types'] >= 5].sort_values('num_failure_types', ascending=False)

print(f'Instances with 5+ failure types: {len(complex)}')
print('\nTop 10 most complex instances:')
for idx, row in complex.head(10).iterrows():
    print(f'  ID {row[\"id\"]}: {row[\"num_failure_types\"]} types - {list(row[\"failure_types\"])}')
"
```

### Compare Before/After Cleaning

```bash
python3 -c "
import json

# Before (original)
with open('results/failure_classifications.json') as f:
    before = json.load(f)
    
# After (cleaned)
with open('results/failure_classifications_cleaned.json') as f:
    after = json.load(f)

print('COMPARISON:')
print(f'Before: {before[\"total_classified\"]} instances')
print(f'After:  {after[\"total_classified\"]} instances')
print(f'\nUnique types before: {len(before[\"unique_failure_types\"])}')
print(f'Unique types after:  {len(after[\"unique_failure_types\"])}')
"
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Check current state
python3 scripts/analyze_failure_type_statistics.py --format txt
cat results/failure_type_statistics_report.txt | head -50

# 2. Update specific instances
python3 scripts/update_instance_failure_types.py \
  --issue-id 64 \
  --failure-types "Code Formatting,Linting" \
  --mode replace \
  --backup

# 3. Re-generate statistics to see changes
python3 scripts/analyze_failure_type_statistics.py

# 4. View updated report
cat results/failure_type_statistics_report.txt | head -50

# 5. Export for further analysis
python3 -c "
import json
import pandas as pd
with open('results/failure_type_statistics.json') as f:
    stats = json.load(f)
df = pd.DataFrame(list(stats['basic_statistics']['failure_type_counts'].items()), 
                  columns=['Type', 'Count'])
df.to_csv('results/export.csv', index=False)
"
```

---

## 📂 File Locations

```
results/
├── failure_type_statistics.json          # Full statistics (JSON)
├── failure_type_statistics_report.txt    # Human-readable report
└── failure_classifications_cleaned.json  # Clean classifications

dataset/
├── lca_dataset.parquet                   # Main dataset (updated)
└── lca_dataset_backup.parquet            # Backup (if --backup used)

scripts/
├── update_instance_failure_types.py      # Update script
├── analyze_failure_type_statistics.py    # Statistics script
├── README_scripts_usage.md               # Full documentation
└── QUICK_START.md                        # This file
```

---

## 💡 Tips

1. **Always use `--backup`** when updating to create safety backups
2. **Re-generate statistics** after any updates to see changes
3. **Check the report** in `results/failure_type_statistics_report.txt`
4. **Use batch mode** for updating multiple instances efficiently
5. **Validate changes** by comparing before/after statistics

---

**Need more help?** Check [README_scripts_usage.md](README_scripts_usage.md) for complete documentation.
