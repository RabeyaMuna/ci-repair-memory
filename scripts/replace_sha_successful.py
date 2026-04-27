#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "dataset" / "lca_dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace a row's sha_success value in the parquet dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to lca_dataset.parquet (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument("--record-id", type=int, required=True, help="Target row id")
    parser.add_argument(
        "--new-sha-success",
        required=True,
        help="Replacement sha_success value",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    record_id = args.record_id
    new_sha_success = args.new_sha_success

    df = pd.read_parquet(dataset_path)

    if "id" not in df.columns:
        raise SystemExit("[ERROR] Dataset has no 'id' column.")

    mask = df["id"] == record_id
    if not mask.any():
        raise SystemExit(f"[ERROR] No row with id={record_id}")

    # Ensure the correct column exists
    if "sha_success" not in df.columns:
        df["sha_success"] = pd.NA

    old_value = df.loc[mask, "sha_success"].iloc[0]

    df.loc[mask, "sha_success"] = new_sha_success

    # Atomic save
    tmp = str(dataset_path) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dataset_path)

    print(f"[INFO] Updated row id={record_id}")
    print(f"  - old sha_success: {old_value}")
    print(f"  - new sha_success: {new_sha_success}")


if __name__ == "__main__":
    main()
