#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Set

import pandas as pd

DATASET_PATH = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet")
LOG_PATH = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/deepseek-chat_llm/generated_patches.json")


def collect_sha_fail_values(obj: Any, out: Set[str]) -> None:
    """
    Collect ONLY values of keys named exactly 'sha_fail' from parsed JSON.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "sha_fail" and isinstance(v, str):
                s = v.strip()
                if s:
                    out.add(s)
            else:
                collect_sha_fail_values(v, out)
    elif isinstance(obj, list):
        for x in obj:
            collect_sha_fail_values(x, out)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log JSON not found: {LOG_PATH}")

    df = pd.read_parquet(DATASET_PATH)
    if "id" not in df.columns or "sha_fail" not in df.columns:
        raise KeyError(f"Dataset must contain 'id' and 'sha_fail'. Found: {list(df.columns)}")

    df["id"] = df["id"].apply(lambda x: str(x).strip() if isinstance(x, (int, str)) else None)
    df["sha_fail"] = df["sha_fail"].apply(lambda x: x.strip() if isinstance(x, str) else None)
    df = df[df["sha_fail"].notna() & (df["sha_fail"] != "")].copy()

    with LOG_PATH.open("r", encoding="utf-8") as f:
        log_data = json.load(f)

    logged_shas: Set[str] = set()
    collect_sha_fail_values(log_data, logged_shas)

    # Missing = dataset sha_fail not present as a sha_fail field in log JSON
    missing_df = df[~df["sha_fail"].isin(logged_shas)]
    missing_ids = sorted(
        set(missing_df["id"].dropna().tolist()),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    # Debug counts to prove what’s happening
    print("dataset rows with sha_fail:", len(df))
    print("dataset unique sha_fail:", df["sha_fail"].nunique())
    print("log unique sha_fail:", len(logged_shas))
    print("missing unique sha_fail:", missing_df["sha_fail"].nunique())

    print(json.dumps(missing_ids))


if __name__ == "__main__":
    main()
