#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd


def update_logs_with_failed_jobs_in_place(
    dataset_path: Path,
    failed_logs_path: Path,
) -> None:
    # 1. Load the main dataset
    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    # Ensure 'id' column exists
    if "id" not in df.columns:
        raise KeyError("'id' column not found in dataset")

    # Normalize id type to string for matching
    df["id"] = df["id"].astype(str)

    # 2. Load the new failed job logs
    print(f"[INFO] Loading failed job logs from: {failed_logs_path}")
    with failed_logs_path.open("r", encoding="utf-8") as f:
        failed_entries = json.load(f)

    # Expect: { "id": 302, "failed_jobs": [ { "step_name": "...", "log": "..." }, ... ] }
    id_to_failed_jobs = {
        str(rec["id"]): rec["failed_jobs"] for rec in failed_entries
    }

    print(f"[INFO] Loaded failed logs for {len(id_to_failed_jobs)} ids")

    total_ids_updated = 0
    matched_ids = []

    # 3. Replace logs in the dataset BY id
    for rec_id, failed_jobs in id_to_failed_jobs.items():
        mask = df["id"] == rec_id
        count = int(mask.sum())

        if count == 0:
            print(f"[WARN] No row found in dataset with id={rec_id}")
            continue

        if count > 1:
            print(f"[WARN] Dataset has {count} rows with id={rec_id}. "
                  f"Updating 'logs' for all of them.")

        # Replace logs for that id with failed_jobs
        df.loc[mask, "logs"] = df.loc[mask, "logs"].apply(
            lambda _old: failed_jobs
        )

        total_ids_updated += 1
        matched_ids.append(rec_id)
        print(f"[INFO] Updated logs for id={rec_id} (rows affected: {count})")

    print(f"[INFO] Total distinct ids updated: {total_ids_updated}")
    print(f"[INFO] Matched ids: {', '.join(matched_ids) if matched_ids else 'None'}")

    # 4. Overwrite the SAME parquet file
    print(f"[INFO] Overwriting dataset at: {dataset_path}")
    df.to_parquet(dataset_path, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    dataset_path = base_dir.parent / "dataset" / "lca_dataset.parquet"
    failed_logs_path = base_dir / "results" / "logs" / "failed_job_logs.json"

    update_logs_with_failed_jobs_in_place(dataset_path, failed_logs_path)
