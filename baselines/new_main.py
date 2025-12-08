#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
from omegaconf import OmegaConf
from dotenv import load_dotenv

from utilities.fetch_failed_commit_changed_files import collect_changed_files_for_fail_and_parent
from utilities.llm_provider import get_llm
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()


# --------------------- Helper functions ---------------------
def sanitize_dict(d):
    """Recursively sanitize dictionary to handle JSON-unfriendly characters."""
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    elif isinstance(d, str):
        # Replace problematic backticks with single quotes (or other replacements)
        return d.replace('`', "'")
    else:
        return d


def _load_results_index_safe(path: str, key_field: str = "sha_fail") -> dict:
    """Load JSON file safely and return dict indexed by key_field."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return {}
        index = {}
        for item in data:
            if isinstance(item, dict) and key_field in item:
                # Sanitize to avoid JSON issues
                item = sanitize_dict(item)
                index[item[key_field]] = item
        return index
    except Exception as e:
        print(f"[WARN] Failed to load {path}, starting fresh: {e}")
        return {}


def _save_results_index_ordered(path: str, index: dict, dataset: list, key_field: str = "sha_fail") -> None:
    """Save index dict to JSON ordered according to dataset order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = set()
    ordered = []

    # First, order according to dataset
    for dp in dataset:
        sha = dp.get(key_field)
        if sha and sha in index:
            ordered.append(index[sha])
            seen.add(sha)

    # Then add any extra items not in dataset
    for sha, rec in index.items():
        if sha not in seen:
            ordered.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4, ensure_ascii=False)


# --------------------- Main processing ---------------------
def process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm"):
    """Process dataset incrementally and update JSONs safely."""
    # Folders and paths
    changed_files_dir = config.changed_files_folder
    os.makedirs(changed_files_dir, exist_ok=True)

    result_dir = os.path.join(config.project_result_dir, f"{model_key}_{log_analyzer_type}")
    os.makedirs(result_dir, exist_ok=True)

    log_details_path = os.path.join(result_dir, "log_details.json")
    fault_loc_path = os.path.join(result_dir, "fault_localization.json")
    patches_path = os.path.join(result_dir, "generated_patches.json")

    # Load existing data safely
    log_details_index = _load_results_index_safe(log_details_path)
    fault_loc_index = _load_results_index_safe(fault_loc_path)
    patches_index = _load_results_index_safe(patches_path)

    subset = dataset[550:]  # Adjust subset as needed

    for datapoint in subset:
        sha_fail = datapoint["sha_fail"]
        repo_name = datapoint["repo_name"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)
        benchmark_owner = config.benchmark_owner
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]
        task_id = datapoint["id"]

        print(f"\n[MAIN] Processing failed commit: {sha_fail}")

        # Ensure repo is at the failed commit
        ensure_repo_at_commit(repo_url, repo_path, sha_fail)

        # --------------------- CI LOG ANALYSIS ---------------------
        try:
            if log_analyzer_type == "llm":
                log_analysis_result = CILogAnalyzerLLM(
                    repo_path, logs, sha_fail, workflow, workflow_path, llm=llm, model_name=model_key
                ).run()
            else:
                log_analysis_result = CILogAnalyzerBM25(
                    repo_path, logs, sha_fail, workflow, workflow_path, llm=llm, model_name=model_key
                ).run()

            if isinstance(log_analysis_result, dict):
                log_analysis_result = sanitize_dict(log_analysis_result)
                log_analysis_result["sha_fail"] = sha_fail
                log_details_index[sha_fail] = log_analysis_result
                _save_results_index_ordered(log_details_path, log_details_index, dataset)

        except Exception as e:
            print(f"[ERROR] CI log analysis failed for {sha_fail}: {e}")
            continue

        # --------------------- CHANGED FILES ---------------------
        try:
            changed_files_info = collect_changed_files_for_fail_and_parent(
                owner=benchmark_owner,
                repo=repo_name,
                repo_path=repo_path,
                sha_fail=sha_fail,
                workflow_rel_path=workflow_path,
                workflow_yaml_from_dataset=workflow,
            )

            changed_files_info = sanitize_dict(changed_files_info)
            changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
            with open(changed_files_path, "w", encoding="utf-8") as f:
                json.dump(changed_files_info, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"[ERROR] Failed collecting changed files for {sha_fail}: {e}")

        # --------------------- FAULT LOCALIZATION ---------------------
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

            if isinstance(fault_localizer, dict):
                fault_localizer = sanitize_dict(fault_localizer)
                fault_localizer["sha_fail"] = sha_fail
                fault_loc_index[sha_fail] = fault_localizer
                _save_results_index_ordered(fault_loc_path, fault_loc_index, dataset)

            if not fault_localizer.get("fault_localization_data"):
                print(f"[MAIN] No suspicious files found for {sha_fail}, skipping patch generation")
                continue

        except Exception as e:
            print(f"[ERROR] Fault localization failed for {sha_fail}: {e}")
            continue

        # --------------------- PATCH GENERATION ---------------------
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

            if isinstance(patch_generator, dict) and patch_generator.get("diff"):
                patch_generator = sanitize_dict(patch_generator)
                patch_generator["sha_fail"] = sha_fail
                patches_index[sha_fail] = patch_generator
                _save_results_index_ordered(patches_path, patches_index, dataset)

        except Exception as e:
            print(f"[ERROR] Patch generation failed for {sha_fail}: {e}")
            continue

    # Return final ordered list of patches
    ordered_patches = [patches_index[dp["sha_fail"]] for dp in dataset if dp["sha_fail"] in patches_index]
    return ordered_patches


# --------------------- MAIN ---------------------
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")

    model_key = "deepseek-coder"
    llm = get_llm(model_key)

    # Load dataset
    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    # Process dataset
    results = process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="bm25")

    # Maintain global generated_patches.json safely
    global_output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    global_index = _load_results_index_safe(global_output_file)
    for item in results:
        global_index[item["sha_fail"]] = item
    _save_results_index_ordered(global_output_file, global_index, dataset)

    print(f"[MAIN] Global results saved in {global_output_file}")
