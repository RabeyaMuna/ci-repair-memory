#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
import subprocess
from omegaconf import OmegaConf
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datasets import load_dataset  # (unused right now but kept)

from utilities.fetch_failed_commit_changed_files import (
    collect_changed_files_for_fail_and_parent,
)
from utilities.llm_provider import get_llm, get_tracked_llm
from utilities.token_tracker import TokenTracker
from utilities.fl_evaluator import evaluate_fl
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()


def _load_json_list(path):
    """Load a JSON list from disk, returning [] if the file does not exist."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _merge_by_key(existing: list, new_item: dict, key: str) -> list:
    """
    Return a new list where any entry whose ``key`` matches ``new_item[key]``
    is replaced by ``new_item``; if no match exists, ``new_item`` is appended.
    """
    result = [e for e in existing if e.get(key) != new_item.get(key)]
    result.append(new_item)
    return result


def process_entire_dataset(
    dataset,
    config,
    llm,
    model_key,
    log_analyzer_type="llm",
    tracker: TokenTracker = None,
):
    result_dir = os.path.join(
        config.project_result_dir, f"{model_key}_{log_analyzer_type}"
    )
    os.makedirs(result_dir, exist_ok=True)
    token_report_path = os.path.join(result_dir, "token_report.json")

    # ------------------------------------------------------------------
    # Always load whatever already exists on disk so every run accumulates
    # on top of prior work.  New issues are appended; re-processed issues
    # replace the old entry (keyed by sha_fail / task_id).
    # Token counts, API calls, cost, and tool-call stats are accumulated
    # from the saved token_report.json so the final report covers ALL runs.
    # ------------------------------------------------------------------
    error_details      = _load_json_list(os.path.join(result_dir, "log_details.json"))
    fault_localization = _load_json_list(os.path.join(result_dir, "fault_localization.json"))
    generated_patches  = _load_json_list(os.path.join(result_dir, "generated_patches.json"))

    if tracker:
        tracker.load_prior_report(token_report_path)

    print(
        f"loaded {len(error_details)} log entries, "
        f"{len(fault_localization)} FL entries, "
        f"{len(generated_patches)} patches from prior run."
    )

    # Folder to store one JSON per sha_fail with changed file info
    changed_files_dir = config.changed_files_folder

    subset = dataset[227:]

    for datapoint in subset:
        task_id = datapoint["id"]
        repo_name = datapoint["repo_name"]
        repo_owner = datapoint["repo_owner"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)
        head_branch = datapoint["head_branch"]
        sha_fail = datapoint["sha_fail"]
        benchmark_owner = config.benchmark_owner
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]
        sha_success = datapoint["sha_success"]

        print(f"\n[MAIN] Proceeding with failed commit: {sha_fail}")

        if tracker:
            tracker.start_task(task_id)

        try:
            # Make sure repo is cloned and at sha_fail
            ensure_repo_at_commit(repo_url, repo_path, sha_fail)

            # ------------------------------------------------------------------
            # 1) CI LOG ANALYSIS
            # ------------------------------------------------------------------
            try:
                if log_analyzer_type == "llm":
                    _llm_log = get_tracked_llm(model_key, tracker, "CILogAnalyzerLLM") if tracker else llm
                    log_analysis_result = CILogAnalyzerLLM(
                        repo_path,
                        logs,
                        sha_fail,
                        workflow,
                        workflow_path,
                        llm=_llm_log,
                        model_name=model_key,
                        task_id=task_id,
                    ).run()
                else:
                    _llm_log = get_tracked_llm(model_key, tracker, "CILogAnalyzerBM25") if tracker else llm
                    log_analysis_result = CILogAnalyzerBM25(
                        repo_path,
                        logs,
                        sha_fail,
                        workflow,
                        workflow_path,
                        llm=_llm_log,
                        model_name=model_key,
                        task_id=task_id,
                    ).run()

                error_details = _merge_by_key(error_details, log_analysis_result, "sha_fail")

                with open(os.path.join(result_dir, "log_details.json"), "w") as f:
                    json.dump(error_details, f, indent=4)

            except Exception as e:
                print(f"[ERROR] Failed processing {sha_fail} during error extraction: {e}")
                continue

            try:
                changed_files_info = collect_changed_files_for_fail_and_parent(
                    owner=benchmark_owner,
                    repo=repo_name,
                    repo_path=repo_path,
                    sha_fail=sha_fail,
                    workflow_rel_path=workflow_path,
                    workflow_yaml_from_dataset=workflow
                )

                # Save to per-commit JSON file in baselines/changed_files/{sha_fail}.json
                changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
                with open(changed_files_path, "w", encoding="utf-8") as f:
                    json.dump(changed_files_info, f, indent=4)

                print(f"[MAIN] Saved changed files info to {changed_files_path}")

            except Exception as e:
                print(f"[ERROR] Failed to collect/save changed files for {sha_fail}: {e}")
                continue

            # ------------------------------------------------------------------
            # 3) FAULT LOCALIZATION (later we can inject changed_files_info)
            # ------------------------------------------------------------------
            try:
                _llm_fl = get_tracked_llm(model_key, tracker, "FaultLocalization") if tracker else llm
                fault_localizer = FaultLocalization(
                    sha_fail=sha_fail,
                    repo_path=repo_path,
                    error_logs=log_analysis_result,
                    workflow=workflow,
                    llm=_llm_fl,
                    model_name=model_key,
                    changed_files_info=changed_files_info,
                ).run()

                fault_localization = _merge_by_key(fault_localization, fault_localizer, "sha_fail")

                with open(os.path.join(result_dir, "fault_localization.json"), "w") as f:
                    json.dump(fault_localization, f, indent=4)

                if not fault_localizer.get("fault_localization_data"):
                    print(f"[MAIN] No suspicious files found for {sha_fail}, skipping...")
                    continue

            except Exception as e:
                print(f"[ERROR] Failed processing {sha_fail} during fault localization: {e}")
                continue

            # ------------------------------------------------------------------
            # 4) PATCH GENERATION
            # ------------------------------------------------------------------
            try:
                _llm_pg = get_tracked_llm(model_key, tracker, "PatchGeneration") if tracker else llm
                patch_generator = PatchGeneration(
                    bug_report=fault_localizer,
                    repo_path=repo_path,
                    task_id=task_id,
                    error_details=log_analysis_result,
                    workflow_path=workflow_path,
                    workflow=workflow,
                    llm=_llm_pg,
                    model_name=model_key,
                    tracker=tracker,
                ).run()

                if not patch_generator.get("diff"):
                    print(f"[MAIN] No patch generated for {sha_fail}")
                    continue

                generated_patches = _merge_by_key(generated_patches, patch_generator, "sha_fail")

                with open(os.path.join(result_dir, "generated_patches.json"), "w") as f:
                    json.dump(generated_patches, f, indent=4)

            except Exception as e:
                print(f"[ERROR] Failed processing {sha_fail} during patch generation: {e}")
                continue

        finally:
            # Always close the task window and flush reports so no data is
            # lost if the run crashes or is interrupted mid-dataset.
            if tracker:
                tracker.end_task(task_id)
                tracker.save_json(token_report_path)
                tracker.save_step_trace(
                    os.path.join(result_dir, "step_trace.json")
                )

    return generated_patches


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_key = "gpt-5-mini"  # or "gpt4o", "deepseek-chat", etc.
    log_analyzer_type = "llm"  # "llm" or "bm25"
    llm = get_llm(model_key)

    # ------------------------------------------------------------------
    # Token / cost tracker — shared across the entire run
    # ------------------------------------------------------------------
    tracker = TokenTracker(model_name=model_key, log_analyzer_type=log_analyzer_type)

    hf_token = os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")

    dataset_path = hf_hub_download(
        repo_id="ci-benchmark-user/ci-repair-bench",
        filename="ci_repair_dataset.parquet",
        repo_type="dataset",
        token=hf_token,
    )

    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    # Set resume_from to the number of items already processed (0 = fresh run).

    try:
        results = process_entire_dataset(
            dataset, config, llm, model_key,
            log_analyzer_type=log_analyzer_type,
            tracker=tracker
        )
    finally:
        # ------------------------------------------------------------------
        # Always persist tracking reports even on crash / keyboard interrupt.
        # ------------------------------------------------------------------
        tracker.print_summary()
        result_dir = os.path.join(
            config.project_result_dir, f"{model_key}_{log_analyzer_type}"
        )
        os.makedirs(result_dir, exist_ok=True)

        # token costs + tool-call stats (compact — no prompt/response bodies)
        tracker.save_json(os.path.join(result_dir, "token_report.json"))

        # full step trace with every prompt sent and response received
        tracker.save_step_trace(os.path.join(result_dir, "step_trace.json"))

        # FL evaluation: predicted files vs files actually patched (from diff headers)
        fl_path     = os.path.join(result_dir, "fault_localization.json")
        patch_path  = os.path.join(result_dir, "generated_patches.json")
        if os.path.exists(fl_path) and os.path.exists(patch_path):
            try:
                evaluate_fl(
                    fault_localization_path=fl_path,
                    generated_patches_path=patch_path,
                    output_path=os.path.join(result_dir, "fl_evaluation.json"),
                )
            except Exception as fl_err:
                print(f"[MAIN] FL evaluation failed: {fl_err}")

    output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[MAIN] Results saved in {output_file}")
