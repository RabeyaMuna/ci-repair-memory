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
from utilities.llm_provider import filesystem_safe_model_key, get_default_model_key, get_llm, get_tracked_llm
from utilities.token_tracker import TokenTracker
from utilities.fl_evaluator import evaluate_fl
from utilities.memory_plugin import MemoryPlugin
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()


def _make_run_suffix(config) -> str:
    """
    Compute the result-dir suffix based on memory mode and ablation level.
    baseline        → _baseline
    memory L1       → _memory_L1
    memory L1+L2    → _memory_L1L2
    memory L1+L2+L3 → _memory   (backward-compat with already-existing runs)
    """
    if not bool(config.get("memory_enabled", False)):
        return "_baseline"
    ablation = str(config.get("memory_ablation_levels", "L1+L2+L3"))
    level_tag = ablation.replace("+", "")  # "L1" | "L1L2" | "L1L2L3"
    return "_memory" if level_tag == "L1L2L3" else f"_memory_{level_tag}"


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
    model_dir_key = filesystem_safe_model_key(model_key)
    result_dir = os.path.join(
        config.project_result_dir, f"{model_dir_key}_{log_analyzer_type}{_make_run_suffix(config)}"
    )
    os.makedirs(result_dir, exist_ok=True)
    token_report_path = os.path.join(result_dir, "token_report.json")
    memory_plugin = MemoryPlugin(config=config, result_dir=result_dir, llm=llm)

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
    existing_generated_patch_shas = {
        str(item.get("sha_fail") or "")
        for item in generated_patches
        if str(item.get("sha_fail") or "")
    }

    if tracker:
        tracker.load_prior_report(token_report_path)

    print(
        f"loaded {len(error_details)} log entries, "
        f"{len(fault_localization)} FL entries, "
        f"{len(generated_patches)} patches from prior run."
    )

    # Folder to store one JSON per sha_fail with changed file info
    changed_files_dir = config.changed_files_folder

    subset = dataset[0:]

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

        if bool(config.get("resume_skip_generated_patches", False)) and sha_fail in existing_generated_patch_shas:
            print(f"[MAIN] Skipping {sha_fail}: patch already generated in existing results.")
            continue

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

                if log_analysis_result.get("error"):
                    print(f"[MAIN] Log analysis failed for {sha_fail}: {log_analysis_result['error']} — skipping FL/patch")
                    continue

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
                memory_context = {"enabled": False, "matches": []}
                if memory_plugin.is_enabled():
                    memory_query = memory_plugin.build_query(
                        task_id=task_id,
                        sha_fail=sha_fail,
                        repo_name=repo_name,
                        workflow_path=workflow_path,
                        workflow=workflow,
                        log_analysis_result=log_analysis_result,
                        changed_files_info=changed_files_info,
                    )
                    memory_context = memory_plugin.retrieve(memory_query)

                _llm_fl = get_tracked_llm(model_key, tracker, "FaultLocalization") if tracker else llm
                fault_localizer = FaultLocalization(
                    sha_fail=sha_fail,
                    repo_path=repo_path,
                    error_logs=log_analysis_result,
                    workflow=workflow,
                    llm=_llm_fl,
                    model_name=model_key,
                    changed_files_info=changed_files_info,
                    memory_plugin=memory_plugin,
                    memory_context=memory_context,
                ).run()

                # Embed full memory context so each FL entry is self-contained
                # for ablation analysis (no need to join memory_retrieval_log.jsonl).
                fault_localizer["memory_summary"] = {
                    "ablation_levels": str(config.get("memory_ablation_levels", "L1+L2+L3")),
                    "memory_enabled": memory_context.get("enabled", False),
                    "level_scores": memory_context.get("level_scores", {"L1": 0.0, "L2": 0.0, "L3": 0.0}),
                    "weighted_similarity": memory_context.get("weighted_similarity", 0.0),
                    "selected_memory_levels": memory_context.get("selected_memory_levels", []),
                    "memory_injected": bool(memory_context.get("matches")),
                    "suppressed_reason": memory_context.get("reason"),
                    "candidate_files": memory_context.get("candidate_files", []),
                    "high_level_hints": memory_context.get("high_level_hints", []),
                    "l1_matches": memory_context.get("l1_matches", []),
                    "l2_matches": memory_context.get("l2_matches", []),
                    "l3_matches": memory_context.get("l3_matches", []),
                }

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
                existing_generated_patch_shas.add(sha_fail)

                with open(os.path.join(result_dir, "generated_patches.json"), "w") as f:
                    json.dump(generated_patches, f, indent=4)

                # Incremental FL evaluation — updates fl_evaluation.json after every
                # issue so scores are visible without waiting for the full run to finish.
                try:
                    evaluate_fl(
                        fault_localization_path=os.path.join(result_dir, "fault_localization.json"),
                        generated_patches_path=os.path.join(result_dir, "generated_patches.json"),
                        output_path=os.path.join(result_dir, "fl_evaluation.json"),
                    )
                except Exception as _fl_err:
                    print(f"[MAIN] Incremental FL evaluation failed: {_fl_err}")

                if bool(config.get("memory_writeback_enabled", False)):
                    memory_plugin.save_memory_entry(
                        task_id=task_id,
                        sha_fail=sha_fail,
                        repo_name=repo_name,
                        repo_owner=repo_owner,
                        workflow_path=workflow_path,
                        workflow=workflow,
                        log_analysis_result=log_analysis_result,
                        changed_files_info=changed_files_info,
                        fault_localizer=fault_localizer,
                        patch_generator=patch_generator,
                    )

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

    model_key = get_default_model_key()  # or "gpt4o", "deepseek-chat", etc.
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
            config.project_result_dir,
            f"{filesystem_safe_model_key(model_key)}_{log_analyzer_type}{_make_run_suffix(config)}",
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

    output_file = os.path.join(
        config.project_result_dir,
        f"generated_patches_{'memory' if config.get('memory_enabled', False) else 'baseline'}.json",
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[MAIN] Results saved in {output_file}")
