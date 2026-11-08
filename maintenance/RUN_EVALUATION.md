# Evaluation Workflow

## Step-by-Step Process:

### **Step 1: Extract patches from dataset**
```bash
python3 maintenance/extract_patches.py \
  --dataset dataset/lca_dataset.parquet
```
Output: `maintenance/reports/generated_patches.json`

---

### **Step 2: Push all patches and queue them**

Use the main benchmark to push all patches:

```bash
python3 run_benchmark.py
```

This will:
- Push all patches to GitHub
- Queue all workflows
- Save job info to results files

---

### **Step 3: Monitor and wait for completion**

After all patches are pushed, poll for results:

```bash
python3 recheck_waiting_jobs.py
```

This will:
- Check all waiting jobs
- Update status (pass/fail/cancelled/timeout)
- Record completion times

---

### **Step 4: Generate final report with metrics**

After all jobs complete:

```bash
python3 maintenance/generate_inspection_report.py \
  --results results/job_results.jsonl
```

This generates:
- **Accuracy**: Pass rate %
- **Execution duration**: Mean, Median, Max
- **Queueing delay**: Mean, Median, Max  
- **Flagged instances**: For manual inspection

---

## Alternative: Use Maintenance Scripts (Simplified)

If you want self-contained maintenance workflow:

```bash
# 1. Extract
python3 maintenance/extract_patches.py --dataset dataset/lca_dataset.parquet

# 2. Push and evaluate
python3 maintenance/push_and_evaluate.py \
  --patches maintenance/reports/generated_patches.json \
  --config config.yaml

# 3. Report
python3 maintenance/generate_inspection_report.py \
  --results maintenance/reports/evaluations/job_results_*.json
```
