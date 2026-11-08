#!/r
# usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
import subprocess
import argparse
from omegaconf import OmegaConf
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from datasets import load_dataset  # (unused right now but kept)

from utilities.fetch_failed_commit_changed_files import (
    collect_changed_files_for_fail_and_parent,
)
from utilities.llm_provider import get_llm
from utilities.llm_tracker import tracker
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()


def process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm", num_instances=None, base_dir=None):
    error_details = []
    fault_localization = []
    generated_patches = []

    # Folder to store one JSON per sha_fail with changed file info
    changed_files_dir = config.changed_files_folder
    if base_dir and not os.path.isabs(changed_files_dir):
        changed_files_dir = os.path.join(base_dir, changed_files_dir)
    os.makedirs(changed_files_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Make result_dir absolute if it's relative
    project_result_dir = config.project_result_dir
    if base_dir and not os.path.isabs(project_result_dir):
        project_result_dir = os.path.join(base_dir, project_result_dir)
    result_dir = os.path.join(
        project_result_dir, f"{model_key}_{log_analyzer_type}"
    )
    os.makedirs(result_dir, exist_ok=True)

    # Load existing results to check what's already been processed
    existing_log_details_path = os.path.join(result_dir, "log_details.json")
    existing_fl_path = os.path.join(result_dir, "fault_localization.json")
    existing_patches_path = os.path.join(result_dir, "generated_patches.json")

    # Track which instances completed each step
    log_shas = set()
    fl_shas = set()
    patch_shas = set()

    if os.path.exists(existing_log_details_path):
        with open(existing_log_details_path, "r") as f:
            error_details = json.load(f)
            log_shas.update(item.get("sha_fail") for item in error_details if item.get("sha_fail"))

    if os.path.exists(existing_fl_path):
        with open(existing_fl_path, "r") as f:
            fault_localization = json.load(f)
            fl_shas.update(item.get("sha_fail") for item in fault_localization if item.get("sha_fail"))

    if os.path.exists(existing_patches_path):
        with open(existing_patches_path, "r") as f:
            generated_patches = json.load(f)
            patch_shas.update(item.get("sha_fail") for item in generated_patches if item.get("sha_fail"))

    # Only skip if instance completed ALL three steps
    processed_sha_fails = log_shas & fl_shas & patch_shas

    # Determine subset based on num_instances parameter
    if num_instances is not None:
        subset = dataset[0:num_instances]
        print(f"[MAIN] Processing {num_instances} instances out of {len(dataset)}")
    else:
        subset = dataset
        print(f"[MAIN] Processing all {len(dataset)} instances")

    print(f"[MAIN] Found {len(processed_sha_fails)} already processed instances, will skip those")
    # target_ids = {241, 243, 281, 323}
    # subset = [dp for dp in dataset if dp.get("id") in target_ids]

    for datapoint in subset:
        task_id = datapoint["id"]
        repo_name = datapoint["repo_name"]
        repo_owner = datapoint["repo_owner"]
        sha_fail = datapoint["sha_fail"]

        # Check if this instance was already processed
        if sha_fail in processed_sha_fails:
            print(f"\n[MAIN] Skipping already processed commit: {sha_fail} (task_id: {task_id})")
            continue

        # Make paths absolute if they're relative
        baseline_repo_folder = config.baseline_repo_folder
        if base_dir and not os.path.isabs(baseline_repo_folder):
            baseline_repo_folder = os.path.join(base_dir, baseline_repo_folder)
        repo_path = os.path.join(baseline_repo_folder, repo_name)

        head_branch = datapoint["head_branch"]
        benchmark_owner = config.benchmark_owner
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]
        sha_success = datapoint["sha_success"]

        print(f"\n[MAIN] Proceeding with failed commit: {sha_fail} (task_id: {task_id})")

        # Make sure repo is cloned and at sha_fail
        ensure_repo_at_commit(repo_url, repo_path, sha_fail)

        # ------------------------------------------------------------------
        # 1) CI LOG ANALYSIS (skip if already done)
        # ------------------------------------------------------------------
        if sha_fail in log_shas:
            # Load existing log analysis
            log_analysis_result = next((item for item in error_details if item.get("sha_fail") == sha_fail), None)
            print(f"[MAIN] Using existing log analysis for {sha_fail}")
        else:
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

                # Only append if not already in list (prevent duplicates)
                if not any(item.get("sha_fail") == sha_fail for item in error_details):
                    error_details.append(log_analysis_result)

                with open(os.path.join(result_dir, "log_details.json"), "w") as f:
                    json.dump(error_details, f, indent=4)

                # Skip further processing if log analysis failed
                if "error" in log_analysis_result and log_analysis_result.get("error"):
                    print(f"[MAIN] Skipping {sha_fail} due to log analysis error: {log_analysis_result.get('error')[:100]}...")
                    continue

            except Exception as e:
                print(f"[ERROR] Failed processing {sha_fail} during error extraction: {e}")
                continue

        # ------------------------------------------------------------------
        # 2) CHANGED FILES (skip if already exists)
        # ------------------------------------------------------------------
        changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
        if os.path.exists(changed_files_path):
            with open(changed_files_path, "r", encoding="utf-8") as f:
                changed_files_info = json.load(f)
            print(f"[MAIN] Using existing changed files for {sha_fail}")
        else:
            try:
                changed_files_info = collect_changed_files_for_fail_and_parent(
                    owner=benchmark_owner,
                    repo=repo_name,
                    repo_path=repo_path,
                    sha_fail=sha_fail,
                    workflow_rel_path=workflow_path,
                    workflow_yaml_from_dataset=workflow
                )

                with open(changed_files_path, "w", encoding="utf-8") as f:
                    json.dump(changed_files_info, f, indent=4)

                print(f"[MAIN] Saved changed files info to {changed_files_path}")

            except Exception as e:
                print(f"[ERROR] Failed to collect/save changed files for {sha_fail}: {e}")
                continue

        # ------------------------------------------------------------------
        # 3) FAULT LOCALIZATION (skip if already done)
        # ------------------------------------------------------------------
        if sha_fail in fl_shas:
            # Load existing FL
            fault_localizer = next((item for item in fault_localization if item.get("sha_fail") == sha_fail), None)
            print(f"[MAIN] Using existing fault localization for {sha_fail}")
        else:
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

                # Only append if not already in list (prevent duplicates)
                if not any(item.get("sha_fail") == sha_fail for item in fault_localization):
                    fault_localization.append(fault_localizer)

                with open(os.path.join(result_dir, "fault_localization.json"), "w") as f:
                    json.dump(fault_localization, f, indent=4)

                if not fault_localizer.get("fault_localization_data"):
                    print(f"[MAIN] No suspicious files found for {sha_fail}, skipping...")
                    continue

            except Exception as e:
                print(f"[ERROR] Failed processing {sha_fail} during fault localization: {e}")
                continue

        # Check if FL has data before proceeding to patch
        if not fault_localizer or not fault_localizer.get("fault_localization_data"):
            print(f"[MAIN] No suspicious files found for {sha_fail}, skipping...")
            continue

        # # ------------------------------------------------------------------
        # # 4) PATCH GENERATION
        # # ------------------------------------------------------------------
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

            if not patch_generator.get("diff"):
                print(f"[MAIN] No patch generated for {sha_fail}")
                continue

            # Only append if not already in list (prevent duplicates)
            if not any(item.get("sha_fail") == sha_fail for item in generated_patches):
                generated_patches.append(patch_generator)

            with open(os.path.join(result_dir, "generated_patches.json"), "w") as f:
                json.dump(generated_patches, f, indent=4)

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during patch generation: {e}")
            continue

    return generated_patches


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run CI Repair Baseline Pipeline')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-mini',
        help='Model to use (e.g., gpt-5-mini, deepseek-v4-flash, gpt4o)'
    )
    parser.add_argument(
        '--analyzer',
        choices=['llm', 'bm25'],
        default='llm',
        help='Log analyzer type (default: llm)'
    )
    parser.add_argument(
        '--num-instances',
        type=int,
        default=None,
        help='Number of instances to process (default: all)'
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   CI Repair Baseline Pipeline                                ║
║                                                                              ║
║  Model:    {args.model:<60} ║
║  Analyzer: {args.analyzer:<60} ║
║  Instances: {str(args.num_instances or 'all'):<58} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_key = args.model
    llm = get_llm(model_key)

    hf_token = os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")

    print("Loading dataset...")
    dataset_path = hf_hub_download(
        repo_id="ci-benchmark-user/ci-repair-bench",
        filename="ci_repair_dataset.parquet",
        repo_type="dataset",
        token=hf_token,
    )

    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")
    print(f"Loaded {len(dataset)} instances from dataset\n")

    log_analyzer_type = args.analyzer

    results = process_entire_dataset(
        dataset, config, llm, model_key, log_analyzer_type=log_analyzer_type, num_instances=args.num_instances, base_dir=base_dir
    )

    output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[MAIN] Results saved in {output_file}")

    # Save LLM usage tracking report
    print("\n" + "=" * 80)
    tracker.print_summary()

    result_dir = os.path.join(
        config.project_result_dir, f"{model_key}_{log_analyzer_type}"
    )
    os.makedirs(result_dir, exist_ok=True)

    usage_report_path = os.path.join(result_dir, "llm_usage_report.json")
    tracker.save_report(usage_report_path, include_all_calls=True)
    print(f"[MAIN] LLM usage tracking saved to {usage_report_path}")
