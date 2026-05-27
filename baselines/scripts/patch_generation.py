#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
from typing import Dict, Any, Optional

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

from ci_repair.patch_generation import PatchGeneration
from utilities.llm_provider import get_llm

load_dotenv()

# =========================================================
# EDIT THESE VALUES DIRECTLY (no argparse)
# =========================================================

# --- selection mode (CHOOSE ONLY ONE) ---
TARGET_IDS = [ "69"]  # or set to None
ID_RANGE = None  # e.g., (159, 301), or None

# --- model + folder naming ---
MODEL_KEY = "gpt-4o-mini"  # must match keys in llm_provider and model_token_limits
LOG_ANALYZER_TYPE = "llm"  # "llm" or "bm25"

# --- project base path (resolved from this file) ---
PROJECT_ROOT = REPO_ROOT

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "lca_dataset.parquet")

LOG_DETAILS_PATH = os.path.join(
    PROJECT_ROOT, "baselines/results", f"{MODEL_KEY}_{LOG_ANALYZER_TYPE}", "log_details.json"
)
FAULT_LOCALIZATION_PATH = os.path.join(
    PROJECT_ROOT, "baselines/results", f"{MODEL_KEY}_{LOG_ANALYZER_TYPE}", "fault_localization.json"
)

PATCHES_PATH = os.path.join(
    PROJECT_ROOT, "baselines/results", f"{MODEL_KEY}_{LOG_ANALYZER_TYPE}", "generated_patches.json"
)

WRITE_GLOBAL = True
GLOBAL_PATCHES_PATH = os.path.join(PROJECT_ROOT, "baselines/results", "generated_patches.json")

# =========================================================


# ---------------- Git helpers (NO FETCH, clean sha_fail) ----------------

def _git(repo_path: str, args: list[str]) -> str:
    """Run git command and return stdout, raise on error."""
    res = subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        text=True,
        capture_output=True,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Git failed: git {' '.join(args)}\n"
            f"cwd={repo_path}\nstdout={res.stdout}\nstderr={res.stderr}"
        )
    return res.stdout.strip()


def reset_worktree_clean(repo_path: str) -> None:
    """Make workspace clean WITHOUT fetching and WITHOUT caring what branch/commit you are on."""
    _git(repo_path, ["reset", "--hard"])
    _git(repo_path, ["clean", "-fdx"])


def checkout_sha_clean(repo_path: str, sha: str) -> None:
    """
    Checkout a specific commit SHA and ensure workspace is pristine.
    NO FETCH: commit must already exist in local object database.
    """
    # Ensure commit exists locally (fails fast if shallow/missing)
    _git(repo_path, ["cat-file", "-e", f"{sha}^{{commit}}"])

    # First make sure checkout can't be blocked by local changes
    reset_worktree_clean(repo_path)

    # Force checkout commit (detached head is fine)
    _git(repo_path, ["checkout", "-f", sha])

    # Ensure we are EXACTLY at sha and totally clean
    _git(repo_path, ["reset", "--hard", sha])
    _git(repo_path, ["clean", "-fdx"])


# ---------------- JSON helpers ----------------

def load_results_index(path: str, key_field: str = "sha_fail") -> Dict[str, Dict[str, Any]]:
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
    index: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and key_field in item and isinstance(item[key_field], str):
            index[item[key_field]] = item
    return index


def save_results_index_ordered(
    path: str,
    index: Dict[str, Dict[str, Any]],
    dataset: list,
    key_field: str = "sha_fail",
) -> None:
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

    # Any extra entries not in dataset order
    for sha, rec in index.items():
        if sha not in seen:
            ordered.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4)


def select_datapoints(dataset: list) -> list:
    """
    Select datapoints by TARGET_IDS (list) or ID_RANGE (start,end).
    Keeps dataset order.
    """
    if TARGET_IDS is not None and ID_RANGE is not None:
        raise ValueError("Set only one: TARGET_IDS or ID_RANGE (set the other to None).")

    if TARGET_IDS is not None:
        wanted = set(int(x) for x in TARGET_IDS)
        return [dp for dp in dataset if int(dp.get("id")) in wanted]

    if ID_RANGE is not None:
        start, end = ID_RANGE
        return [dp for dp in dataset if int(start) <= int(dp.get("id")) <= int(end)]

    raise ValueError("Set either TARGET_IDS or ID_RANGE at the top of the script.")


def main():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"config.yaml not found: {CONFIG_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"dataset parquet not found: {DATASET_PATH}")

    if not os.path.exists(LOG_DETAILS_PATH):
        raise FileNotFoundError(f"log_details.json not found: {LOG_DETAILS_PATH}")
    if not os.path.exists(FAULT_LOCALIZATION_PATH):
        raise FileNotFoundError(f"fault_localization.json not found: {FAULT_LOCALIZATION_PATH}")

    config = OmegaConf.load(CONFIG_PATH)

    dataset_df = pd.read_parquet(DATASET_PATH)
    dataset = dataset_df.to_dict(orient="records")

    dps = select_datapoints(dataset)
    if not dps:
        print("[MAIN] No datapoints matched your selection.")
        return

    log_index = load_results_index(LOG_DETAILS_PATH, key_field="sha_fail")
    fault_index = load_results_index(FAULT_LOCALIZATION_PATH, key_field="sha_fail")
    patches_index = load_results_index(PATCHES_PATH, key_field="sha_fail")  # ok if missing

    llm = get_llm(MODEL_KEY)

    for dp in dps:
        task_id = int(dp["id"])
        sha_fail = dp["sha_fail"]

        workflow = dp.get("workflow")
        workflow_path = dp.get("workflow_path")

        repo_name = dp["repo_name"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)

        print(f"\n[PATCH-ONLY] id={task_id} sha_fail={sha_fail} repo={repo_name}")

        # Require precomputed analysis inputs
        missing = []
        if sha_fail not in log_index:
            missing.append("log_details")
        if sha_fail not in fault_index:
            missing.append("fault_localization")
        if missing:
            print(f"[SKIP] Missing data for sha_fail={sha_fail}: {', '.join(missing)}")
            continue

        error_details = log_index[sha_fail]
        bug_report = fault_index[sha_fail]

        if not bug_report.get("fault_localization_data"):
            print(f"[SKIP] No fault_localization_data for {sha_fail}")
            continue

        patch_result: Optional[Dict[str, Any]] = None

        # ---- REQUIRED: clean sha_fail -> run patch generation -> clean again ----
        try:
            checkout_sha_clean(repo_path, sha_fail)

            patch_result = PatchGeneration(
                bug_report=bug_report,
                repo_path=repo_path,
                task_id=task_id,
                error_details=error_details,
                workflow_path=workflow_path,
                workflow=workflow,
                llm=llm,
                model_name=MODEL_KEY,
            ).run()

        except Exception as e:
            print(f"[ERROR] PatchGeneration failed for sha_fail={sha_fail}: {e}")
            continue

        finally:
            # Always leave repo clean after each task (even if PatchGeneration fails)
            try:
                reset_worktree_clean(repo_path)
            except Exception as e:
                print(f"[WARN] Failed to clean repo after task {task_id}: {e}")

        # Validate result
        if not isinstance(patch_result, dict):
            print(f"[SKIP] PatchGeneration returned non-dict for {sha_fail}")
            continue

        if not patch_result.get("diff"):
            print(f"[MAIN] No patch generated for {sha_fail}")
            continue

        patch_result.setdefault("sha_fail", sha_fail)
        patches_index[sha_fail] = patch_result

        # Save per-model generated_patches.json in dataset order (replace if exists)
        save_results_index_ordered(PATCHES_PATH, patches_index, dataset, key_field="sha_fail")
        print(f"[MAIN] Saved/updated patch for {sha_fail} -> {PATCHES_PATH}")

    # Optional global output
    if WRITE_GLOBAL:
        global_index = load_results_index(GLOBAL_PATCHES_PATH, key_field="sha_fail")
        for sha, rec in patches_index.items():
            global_index[sha] = rec
        save_results_index_ordered(GLOBAL_PATCHES_PATH, global_index, dataset, key_field="sha_fail")
        print(f"[MAIN] Global results saved in {GLOBAL_PATCHES_PATH}")


if __name__ == "__main__":
    main()
