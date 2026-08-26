#!/usr/bin/env python3
"""
Update dataset logs from freshly fetched failed_job_logs.json entries.

Matches each fetched entry to its row in lca_dataset.parquet by `id` and
`sha_fail`, then overwrites that row's `logs` column.

Note: failed_job_logs.json stores the re-run commit SHA under `sha_fail`
(matching jobs_failure_diff.jsonl's `commit` field), while the dataset's
`sha_fail` column holds the original failing commit SHA
(jobs_failure_diff.jsonl's `sha_original`). jobs_failure_diff.jsonl is used
to translate between the two.
"""
import json
import pandas as pd

FAILED_JOB_LOGS_FILE = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/data_managment/results/logs/failed_job_logs.json'
FAILURE_DIFF_FILE = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/jobs_failure_diff.jsonl'
DATASET_FILE = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet'


def load_id_to_sha_original(path: str) -> dict:
    """Map instance id -> original failing commit SHA (dataset's sha_fail)."""
    id_to_sha_original = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                id_to_sha_original[entry['id']] = entry['sha_original']
    return id_to_sha_original


def main():
    print(f"Loading fetched logs from {FAILED_JOB_LOGS_FILE}...")
    with open(FAILED_JOB_LOGS_FILE, 'r') as f:
        fetched_logs = json.load(f)
    print(f"Loaded {len(fetched_logs)} fetched log entries")

    print(f"Loading commit mapping from {FAILURE_DIFF_FILE}...")
    id_to_sha_original = load_id_to_sha_original(FAILURE_DIFF_FILE)

    print(f"Loading dataset from {DATASET_FILE}...")
    df = pd.read_parquet(DATASET_FILE)

    updated_count = 0
    skipped_count = 0

    for entry in fetched_logs:
        instance_id = entry['id']
        sha_original = id_to_sha_original.get(instance_id)

        if sha_original is None:
            print(f"  ✗ ID {instance_id}: no sha_original mapping in {FAILURE_DIFF_FILE}")
            skipped_count += 1
            continue

        mask = (df['id'] == instance_id) & (df['sha_fail'] == sha_original)
        matched_indices = df.index[mask]

        if len(matched_indices) == 0:
            print(f"  ✗ ID {instance_id}: no dataset row matches sha_fail {sha_original[:7]}")
            skipped_count += 1
            continue

        row_index = matched_indices[0]
        df.at[row_index, 'logs'] = entry['logs']
        print(f"  ✓ ID {instance_id}: updated {len(entry['logs'])} log(s)")
        updated_count += 1

    print('=' * 80)
    print(f"Saving dataset to {DATASET_FILE}...")
    df.to_parquet(DATASET_FILE, index=False)

    print(f"✓ Updated {updated_count} dataset row(s)")
    print(f"✗ Skipped {skipped_count} entry(ies)")


if __name__ == '__main__':
    main()
