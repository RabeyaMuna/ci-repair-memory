# Benchmark Recurrence Similarity Algorithms

## Overview
This script computes **TWO** independent similarity measures between CI repair instances to characterize benchmark recurrence patterns.

---

## 1. STRUCTURAL SIMILARITY (Jaccard Index)

### Algorithm
**Jaccard Similarity** = |A ∩ B| / |A ∪ B|

For each pair of instances, compute Jaccard similarity across multiple structural features, then average:

```
structural_sim = average(
    jaccard(failure_types_A, failure_types_B),
    jaccard(tools_A, tools_B),
    jaccard(changed_files_A, changed_files_B)
)
```

### Dataset Fields Used

| Feature | Dataset Field(s) | Processing |
|---------|-----------------|------------|
| **Failure Types** | `error_type` | Extract as set of lowercase strings |
| **CI Tools** | `logs`, `workflow` | Regex extraction of tool names (pytest, npm, pip, cargo, gradle, maven, jest, eslint, mypy, flake8, etc.) |
| **Packages** | `logs`, `workflow` | Regex extraction filtered to common packages (numpy, pandas, flask, django, react, express, pytest, jest) |
| **Changed Files** | `changed_files` | • **Within-repo**: Full paths (e.g., `src/utils/parser.py`)<br>• **Cross-repo**: Basenames only (e.g., `parser.py`) |

### Key Logic
- **Within-repository pairs**: Use full file paths (path structure is meaningful)
- **Cross-repository pairs**: Use basenames only (only filenames are comparable across repos)

---

## 2. LEXICAL SIMILARITY (TF-IDF + Cosine)

### Algorithm
1. **Build text representation** for each instance by concatenating:
   - Failure type description
   - Last 10k characters of logs (where errors appear)
   - Workflow configuration
   - Ground-truth repair (diff + changed files)

2. **Vectorize** all texts using TF-IDF:
   ```
   TF-IDF(term, doc) = TF(term, doc) × IDF(term, corpus)
   
   where:
   TF = term frequency in document
   IDF = log(total_docs / docs_containing_term)
   ```

3. **Compute cosine similarity** between TF-IDF vectors:
   ```
   cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
   
   Range: [0, 1]
   - 0 = completely different
   - 1 = identical
   ```

### Dataset Fields Used

| Component | Dataset Field(s) | Extraction Details |
|-----------|-----------------|-------------------|
| **Failure Type** | `error_type` | Full error category list |
| **Logs** | `logs` | **Last 10,000 characters** (errors appear at tail of logs) |
| **Workflow** | `workflow` | First 1000 characters of CI config |
| **Repair** | `diff` | First 2000 characters of ground-truth patch |
| **Changed Files** | `changed_files` | Full list of modified file paths |

### TF-IDF Vectorizer Settings
```python
TfidfVectorizer(
    max_features=1000,      # Keep top 1000 most important terms
    stop_words='english',   # Remove common words (the, is, at, etc.)
    ngram_range=(1, 2),     # Use unigrams and bigrams
    min_df=2                # Term must appear in at least 2 documents
)
```

---

## 3. COMPUTED METRICS

For the paper table, we compute:

### 3.1 Overall Nearest Neighbor
For each instance, find its **maximum similarity** to any other instance, then average across all instances.

### 3.2 Within Repository
Average similarity for pairs where **both instances are from the same repository**.

### 3.3 Cross Repository
Average similarity for pairs where **instances are from different repositories**.

### 3.4 Historical Predecessor
For each instance, find its maximum similarity to **chronologically earlier instances only** (based on `commit_date`), then average.

---

## 4. IMPORTANT DISTINCTIONS

### Offline vs Retrieval-Time

This is **OFFLINE BENCHMARK CHARACTERIZATION**:
- ✅ Includes ground-truth repairs in similarity calculation
- ✅ Uses last 10k characters of logs (where errors appear)
- ✅ Purpose: Measure similarity patterns in the dataset

This is **NOT** retrieval-time evaluation:
- ❌ At retrieval time, ground-truth repair is not available
- ❌ MemRepair evaluation uses different similarity calculation
- ❌ This analysis is for dataset statistics, not method evaluation

---

## 5. DATASET SCHEMA

Required fields from `lca_dataset.parquet`:

```python
{
    'id': str,                    # Unique instance identifier
    'repo_owner': str,            # GitHub owner
    'repo_name': str,             # Repository name
    'commit_date': str,           # ISO timestamp for temporal ordering
    'error_type': list[str],      # Failure categories
    'logs': list[dict] | str,     # CI logs (list of {log: str} or string)
    'workflow': str,              # CI workflow configuration
    'diff': str,                  # Ground-truth repair patch
    'changed_files': list[str],   # Modified file paths
}
```

---

## 6. OUTPUT

### Files Generated
1. `benchmark_recurrence_analysis.json` - Aggregated metrics
2. `recurrence_pairs.csv` - All pairwise similarities

### CSV Schema
```csv
id_a,id_b,repo_a,repo_b,same_repo,jaccard,tfidf_cosine
instance_1,instance_2,repo_x,repo_y,false,0.1234,0.5678
```
