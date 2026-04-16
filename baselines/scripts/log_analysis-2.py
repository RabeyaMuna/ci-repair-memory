#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# Fix Python import paths so "utilities", "ci_repair", etc. can be imported
# File is: REPO_ROOT/baselines/scripts/log_analysis.py
# ----------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                     # .../baselines/scripts
BASELINES_DIR = os.path.dirname(SCRIPT_DIR)                                 # .../baselines
REPO_ROOT = os.path.dirname(BASELINES_DIR)                                  # .../CI-REPAIR-BENCH

for p in (REPO_ROOT, BASELINES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ----------------------------------------------------------------------
# Imports that rely on sys.path being set
# ----------------------------------------------------------------------
from utilities.fetch_failed_commit_changed_files import collect_changed_files_for_fail_and_parent
from utilities.llm_provider import get_llm
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()

# ----------------------------
# Only run these task IDs
# ----------------------------
TARGET_IDS = ["300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318"]



TARGET_ID_SET = set(TARGET_IDS)

# ----------------------------------------------------------------------
# Helper functions: load + ordered save (by dataset sha_fail order)
# ----------------------------------------------------------------------
def _load_results_index(path: str, key_field: str = "sha_fail") -> dict:
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


def _save_results_index_ordered(
    path: str,
    index: dict,
    dataset: list,
    key_field: str = "sha_fail",
) -> None:
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


def _coerce_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm"):
    changed_files_dir = config.changed_files_folder
    os.makedirs(changed_files_dir, exist_ok=True)

    result_dir = os.path.join(config.project_result_dir, f"{model_key}_{log_analyzer_type}")
    os.makedirs(result_dir, exist_ok=True)

    log_details_path = os.path.join(result_dir, "log_details.json")
    fault_loc_path = os.path.join(result_dir, "fault_localization.json")
    patches_path = os.path.join(result_dir, "generated_patches.json")

    log_details_index = _load_results_index(log_details_path, key_field="sha_fail")
    fault_loc_index = _load_results_index(fault_loc_path, key_field="sha_fail")
    patches_index = _load_results_index(patches_path, key_field="sha_fail")

    # --- robust subset selection ---
        # --- robust subset selection (works for int or str ids) ---
    target_id_str_set = {str(x) for x in TARGET_IDS}

    subset = []
    found_ids = set()
    for dp in dataset:
        dp_id_raw = dp.get("id")
        if dp_id_raw is None:
            continue
        dp_id_str = str(dp_id_raw)

        if dp_id_str in target_id_str_set:
            subset.append(dp)
            found_ids.add(dp_id_str)

    missing_ids = [str(i) for i in TARGET_IDS if str(i) not in found_ids]


    print(f"[MAIN] Dataset size: {len(dataset)}")
    print(f"[MAIN] Will process {len(subset)} datapoints from TARGET_IDS")
    if missing_ids:
        print(f"[WARN] {len(missing_ids)} TARGET_IDS not found. Example: {missing_ids[:20]}")
    if len(subset) == 0:
        print("[ERROR] subset is empty -> check dataset id column/type or dataset file path.")
        return []

    for datapoint in subset:
        task_id = datapoint["id"]
        repo_name = datapoint["repo_name"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)
        sha_fail = datapoint["sha_fail"]
        benchmark_owner = config.benchmark_owner
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]

        print(f"\n[MAIN] Proceeding with task_id={task_id} sha_fail={sha_fail}")

        # 0) Ensure repo at sha_fail
        try:
            ensure_repo_at_commit(repo_url, repo_path, sha_fail)
            
        except Exception as e:
            print(f"[ERROR] ensure_repo_at_commit failed for {sha_fail}: {e}")
            continue

        # 1) CI LOG ANALYSIS
        try:
            if log_analyzer_type == "llm":
                log_analysis_result = CILogAnalyzerLLM(
                    repo_path,
                    logs,
                    sha_fail,
                    workflow,
                    workflow_path,
                    llm=llm,
                    model_name=model_key,
                    task_id=task_id,
                ).run()
            else:
                log_analysis_result = CILogAnalyzerBM25(
                    repo_path,
                    logs,
                    sha_fail,
                    workflow,
                    workflow_path,
                    llm=llm,
                    model_name=model_key,
                    task_id=task_id,
                ).run()

            if isinstance(log_analysis_result, dict):
                log_analysis_result.setdefault("sha_fail", sha_fail)
                log_details_index[sha_fail] = log_analysis_result
                _save_results_index_ordered(log_details_path, log_details_index, dataset)
            else:
                print(f"[WARN] log_analysis_result not dict for {sha_fail}; skipping.")
                continue

        except Exception as e:
            print(f"[ERROR] CI log analysis failed for {sha_fail}: {e}")
            continue

        # 2) CHANGED FILES COLLECTION
        try:
            changed_files_info = collect_changed_files_for_fail_and_parent(
                owner=benchmark_owner,
                repo=repo_name,
                repo_path=repo_path,
                sha_fail=sha_fail,
                workflow_rel_path=workflow_path,
                workflow_yaml_from_dataset=workflow,
            )

            changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
            with open(changed_files_path, "w", encoding="utf-8") as f:
                json.dump(changed_files_info, f, indent=4)

            print(f"[MAIN] Saved changed files info to {changed_files_path}")

        except Exception as e:
            print(f"[ERROR] Changed files collection failed for {sha_fail}: {e}")
            continue

        # 3) FAULT LOCALIZATION
        try:
            fault_localizer = FaultLocalization(
                sha_fail=sha_fail,
                repo_path=repo_path,
                error_logs=log_analysis_result,
                workflow=workflow,
                llm=llm,
                model_name=model_key,
                changed_files_info=changed_files_info,
            ).run()

            if not fault_localizer["fault_localization_data"]:
                print(f"[MAIN] Fault localization found no data for {sha_fail}")
                continue

            if isinstance(fault_localizer, dict):
                fault_localizer.setdefault("sha_fail", sha_fail)
                fault_loc_index[sha_fail] = fault_localizer
                _save_results_index_ordered(fault_loc_path, fault_loc_index, dataset)
            else:
                print(f"[WARN] fault_localizer not dict for {sha_fail}; skipping.")
                continue

            if not fault_localizer.get("fault_localization_data"):
                print(f"[MAIN] No suspicious files for {sha_fail}, skipping patch generation...")
                continue

        except Exception as e:
            print(f"[ERROR] Fault localization failed for {sha_fail}: {e}")
            continue

        # 4) PATCH GENERATION
        try:
            patch_generator = PatchGeneration(
                bug_report=fault_localizer,
                repo_path=repo_path,
                task_id=task_id,
                error_details=log_analysis_result,
                workflow_path=workflow_path,
                workflow=workflow,
                llm=llm,
                model_name=model_key,
            ).run()

            if not isinstance(patch_generator, dict):
                print(f"[MAIN] Patch generator returned non-dict for {sha_fail}, skipping...")
                continue

            if not patch_generator.get("diff"):
                print(f"[MAIN] No patch generated for {sha_fail}")
                continue

            patch_generator.setdefault("sha_fail", sha_fail)
            patches_index[sha_fail] = patch_generator
            _save_results_index_ordered(patches_path, patches_index, dataset)

        except Exception as e:
            print(f"[ERROR] Patch generation failed for {sha_fail}: {e}")
            continue

    # Return ordered patches
    ordered_patches = []
    sha_order = [dp.get("sha_fail") for dp in dataset if dp.get("sha_fail")]
    seen = set()
    for sha in sha_order:
        if sha in patches_index and sha not in seen:
            ordered_patches.append(patches_index[sha])
            seen.add(sha)
    for sha, rec in patches_index.items():
        if sha not in seen:
            ordered_patches.append(rec)

    return ordered_patches


if __name__ == "__main__":
    # Correct locations (repo root)
    config_path = os.path.join(REPO_ROOT, "config.yaml")
    dataset_path = os.path.join(REPO_ROOT, "dataset", "lca_dataset.parquet")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset parquet not found at: {dataset_path}")

    config = OmegaConf.load(config_path)

    # Debug
    dataset_df = pd.read_parquet(dataset_path)
    print("[DEBUG] REPO_ROOT:", REPO_ROOT)
    print("[DEBUG] config_path:", config_path)
    print("[DEBUG] dataset_path:", dataset_path)
    print("[DEBUG] columns:", list(dataset_df.columns))
    if "id" in dataset_df.columns:
        print("[DEBUG] id dtype:", dataset_df["id"].dtype)
        print("[DEBUG] first 10 ids:", dataset_df["id"].head(10).tolist())
    else:
        print("[ERROR] No 'id' column in dataset.")

    dataset = dataset_df.to_dict(orient="records")

    model_key = "gpt-5-mini"  # or "gpt4o"
    llm = get_llm(model_key)

    results = process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm")

    global_output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    global_index = _load_results_index(global_output_file, key_field="sha_fail")

    for item in results:
        if isinstance(item, dict) and "sha_fail" in item:
            global_index[item["sha_fail"]] = item

    _save_results_index_ordered(global_output_file, global_index, dataset)
    print(f"[MAIN] Global results saved in {global_output_file}")
