#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv

# =========================================================
# FIX: Make imports work for your repo layout
# =========================================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASELINES_ROOT = os.path.join(REPO_ROOT, "baselines")

for p in (REPO_ROOT, BASELINES_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from ci_repair.fault_localization import FaultLocalization
from utilities.llm_provider import get_llm
from utilities.ensure_repo import ensure_repo_at_commit
from utilities.fetch_failed_commit_changed_files import (
    collect_changed_files_for_fail_and_parent,
)

load_dotenv()

# =========================================================
# EDIT THESE VALUES DIRECTLY (no argparse)
# =========================================================

# --- selection mode (CHOOSE ONLY ONE) ---
TARGET_IDS = ["512", "513", "514", "515", "516", "517", "518", "519", "520", "521", "522", "523", "524", "525", "526", "527", "528", "529", "530", "531", "532", "533", "534", "535", "536", "537", "538", "539", "540", "541", "542", "543", "544", "545", "547", "548", "549", "551", "552", "553", "554", "555", "556", "557", "558", "559", "560", "562", "563", "564", "565", "566", "567", "568", "569", "570", "571", "572", "573", "574"]
ID_RANGE = None

# --- model + folder naming ---
MODEL_KEY = "gpt-4o-mini"
LOG_ANALYZER_TYPE = "llm"  # "llm" or "bm25"

# --- project base path (resolved from this file) ---
PROJECT_ROOT = REPO_ROOT

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "lca_dataset.parquet")

LOG_DETAILS_PATH = os.path.join(
    PROJECT_ROOT,
    "baselines/results",
    f"{MODEL_KEY}_{LOG_ANALYZER_TYPE}",
    "log_details.json",
)

CHANGED_FILES_DIR = os.path.join(PROJECT_ROOT, "baselines", "changed_files")

FAULT_LOCALIZATION_PATH = os.path.join(
    PROJECT_ROOT,
    "baselines/results",
    f"{MODEL_KEY}_{LOG_ANALYZER_TYPE}",
    "fault_localization.json",
)

REPLACE_EXISTING = True

# =========================================================


def load_results_index(path: str, key_field: str = "sha_fail") -> dict:
    """Load list-of-dicts JSON into {sha_fail: record} index."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    index = {}
    for item in data:
        if isinstance(item, dict) and key_field in item:
            index[item[key_field]] = item
    return index


def save_results_index_ordered(path: str, index: dict, dataset: list, key_field: str = "sha_fail") -> None:
    """Save index to list JSON ordered by dataset sha_fail order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sha_order = []
    seen = set()
    for dp in dataset:
        sha = dp.get(key_field)
        if sha and sha not in seen:
            sha_order.append(sha)
            seen.add(sha)

    ordered = []
    for sha in sha_order:
        if sha in index:
            ordered.append(index[sha])

    for sha, rec in index.items():
        if sha not in seen:
            ordered.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4)


def select_datapoints(dataset: list) -> list:
    """Select datapoints by TARGET_IDS (list) or ID_RANGE (start,end), keeping dataset order."""
    if TARGET_IDS is not None and ID_RANGE is not None:
        raise ValueError("Set only one: TARGET_IDS or ID_RANGE (set the other to None).")

    if TARGET_IDS is not None:
        wanted = set(int(x) for x in TARGET_IDS)
        return [dp for dp in dataset if int(dp.get("id")) in wanted]

    if ID_RANGE is not None:
        start, end = ID_RANGE
        return [dp for dp in dataset if int(start) <= int(dp.get("id")) <= int(end)]

    raise ValueError("Set either TARGET_IDS or ID_RANGE at the top of the script.")


def load_changed_files_for_sha(sha_fail: str) -> dict | None:
    """Loads baselines/changed_files/{sha_fail}.json. Returns dict or None if missing/invalid."""
    path = os.path.join(CHANGED_FILES_DIR, f"{sha_fail}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def get_or_collect_changed_files(
    *,
    dp: dict,
    config,
    repo_name: str,
    repo_path: str,
    sha_fail: str,
    workflow,
) -> dict:
    """
    Do NOT skip if changed_files is missing:
    - try to load baselines/changed_files/{sha_fail}.json
    - if missing: ensure repo at sha_fail, collect via collect_changed_files_for_fail_and_parent(...)
    - save it to baselines/changed_files/{sha_fail}.json for future runs
    - if collection fails: return {"changed_files": []} (still proceed)
    """
    cached = load_changed_files_for_sha(sha_fail)
    if cached is not None:
        return cached

    print(f"[WARN] Missing changed_files for sha_fail={sha_fail}. Fetching now...")

    benchmark_owner = getattr(config, "benchmark_owner", None) or dp.get("repo_owner")
    workflow_path = dp.get("workflow_path")

    # ensure repo is ready before collecting
    try:
        if benchmark_owner:
            repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
            ensure_repo_at_commit(repo_url, repo_path, sha_fail)
        else:
            os.makedirs(repo_path, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] ensure_repo_at_commit failed before changed_files collection for {sha_fail}: {e}")
        return {"changed_files": []}

    try:
        changed_files_info = collect_changed_files_for_fail_and_parent(
            owner=benchmark_owner,
            repo=repo_name,
            repo_path=repo_path,
            sha_fail=sha_fail,
            workflow_rel_path=workflow_path,
            workflow_yaml_from_dataset=workflow,
        )

        os.makedirs(CHANGED_FILES_DIR, exist_ok=True)
        out_path = os.path.join(CHANGED_FILES_DIR, f"{sha_fail}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(changed_files_info, f, indent=4)

        print(f"[MAIN] Saved changed files info -> {out_path}")
        return changed_files_info if isinstance(changed_files_info, dict) else {"changed_files": []}

    except Exception as e:
        print(f"[ERROR] Failed to collect changed_files for {sha_fail}: {e}")
        print(f"[WARN] Continuing with empty changed_files for sha_fail={sha_fail}")
        return {"changed_files": []}


def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"config.yaml not found: {CONFIG_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"dataset parquet not found: {DATASET_PATH}")
    if not os.path.exists(LOG_DETAILS_PATH):
        raise FileNotFoundError(f"log_details.json not found: {LOG_DETAILS_PATH}")

    config = OmegaConf.load(CONFIG_PATH)

    dataset_df = pd.read_parquet(DATASET_PATH)
    dataset = dataset_df.to_dict(orient="records")

    dps = select_datapoints(dataset)
    if not dps:
        print("[MAIN] No datapoints matched your selection.")
        return

    log_index = load_results_index(LOG_DETAILS_PATH, key_field="sha_fail")
    fault_index = load_results_index(FAULT_LOCALIZATION_PATH, key_field="sha_fail")

    llm = get_llm(MODEL_KEY)

    for dp in dps:
        task_id = int(dp["id"])
        sha_fail = dp["sha_fail"]

        repo_name = dp["repo_name"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)

        workflow = dp.get("workflow")

        print(f"\n[BUGREPORT-ONLY] id={task_id} sha_fail={sha_fail}")

        if not REPLACE_EXISTING and sha_fail in fault_index:
            print(f"[SKIP] Existing fault localization found for {sha_fail} (REPLACE_EXISTING=False).")
            continue

        # log_details is required
        if sha_fail not in log_index:
            print(f"[SKIP] Missing log_details for sha_fail={sha_fail}")
            continue

        error_details = log_index[sha_fail]

        # changed_files is NOT required to exist on disk; fetch if missing
        changed_files_info = get_or_collect_changed_files(
            dp=dp,
            config=config,
            repo_name=repo_name,
            repo_path=repo_path,
            sha_fail=sha_fail,
            workflow=workflow,
        )

        # Ensure repo exists at sha_fail (safe even if already ensured)
        try:
            benchmark_owner = getattr(config, "benchmark_owner", None) or dp.get("repo_owner")
            if benchmark_owner:
                repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
                ensure_repo_at_commit(repo_url, repo_path, sha_fail)
            else:
                os.makedirs(repo_path, exist_ok=True)
        except Exception as e:
            print(f"[ERROR] ensure_repo_at_commit failed for {sha_fail}: {e}")
            continue

        try:
            bug_report = FaultLocalization(
                sha_fail=sha_fail,
                repo_path=repo_path,
                error_logs=error_details,
                workflow=workflow,
                llm=llm,
                model_name=MODEL_KEY,
                changed_files_info=changed_files_info,
            ).run()
        except Exception as e:
            print(f"[ERROR] FaultLocalization failed for {sha_fail}: {e}")
            continue

        if not isinstance(bug_report, dict):
            print(f"[SKIP] FaultLocalization returned non-dict for {sha_fail}")
            continue

        bug_report["sha_fail"] = sha_fail  # force correct key

        if not bug_report.get("fault_localization_data"):
            print(f"[MAIN] No fault_localization_data for {sha_fail} (saving anyway).")

        fault_index[sha_fail] = bug_report
        save_results_index_ordered(FAULT_LOCALIZATION_PATH, fault_index, dataset, key_field="sha_fail")
        print(f"[MAIN] Saved/updated bug report for {sha_fail} -> {FAULT_LOCALIZATION_PATH}")


if __name__ == "__main__":
    main()
