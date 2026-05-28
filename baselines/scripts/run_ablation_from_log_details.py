#!/usr/bin/env python3
"""
run_ablation_from_log_details.py

Reuses an existing log_details.json (CI log analysis output from a prior full
run) and re-runs only Fault Localization + Patch Generation with a specific
memory ablation level.  Skips the expensive CI log analysis step entirely.

The changed_files/ directory is also reused from the prior run — those files
are shared across all ablation configs.

Usage
-----
# L1 only
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \\
  --ablation-levels L1

# L1+L2
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \\
  --ablation-levels L1+L2

# L1+L2+L3
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \\
  --ablation-levels L1+L2+L3

# With a specific model and issue limit
baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py \\
  --ablation-levels L1 \\
  --source-log-details baselines/results/MiniMax-M2.5_llm_baseline/log_details.json \\
  --model-key MiniMax-M2.5 \\
  --max-issues 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from main import _make_run_suffix, _load_json_list, _merge_by_key  # noqa: E402
from utilities.llm_provider import (  # noqa: E402
    filesystem_safe_model_key,
    get_default_model_key,
    get_llm,
    get_tracked_llm,
)
from utilities.token_tracker import TokenTracker  # noqa: E402
from utilities.fl_evaluator import evaluate_fl  # noqa: E402
from utilities.memory_plugin import MemoryPlugin  # noqa: E402
from utilities.ensure_repo import ensure_repo_at_commit  # noqa: E402
from utilities.load_config import load_config  # noqa: E402
from ci_repair.fault_localization import FaultLocalization  # noqa: E402
from ci_repair.patch_generation import PatchGeneration  # noqa: E402

load_dotenv()

DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_SOURCE_LOG_DETAILS = (
    PROJECT_ROOT / "baselines" / "results" / "MiniMax-M2.5" / "log_details.json"
)


def _load_dataset(local_dataset: Path, config) -> List[Dict[str, Any]]:
    if local_dataset.exists():
        df = pd.read_parquet(local_dataset)
        return df.to_dict(orient="records")
    hf_token = os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")
    dataset_path = hf_hub_download(
        repo_id="ci-benchmark-user/ci-repair-bench",
        filename="ci_repair_dataset.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(dataset_path)
    return df.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run FL + Patch Generation for a memory ablation level, "
            "reusing an existing log_details.json from a prior full run."
        )
    )
    parser.add_argument(
        "--ablation-levels",
        choices=("L1", "L1+L2", "L1+L2+L3"),
        required=True,
        help="Memory ablation: which levels to activate.",
    )
    parser.add_argument(
        "--source-log-details",
        type=Path,
        default=DEFAULT_SOURCE_LOG_DETAILS,
        help=(
            "Path to the log_details.json to reuse. Defaults to the repo-relative "
            "MiniMax-M2.5 baseline log_details.json."
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-key", default=get_default_model_key())
    parser.add_argument(
        "--max-issues",
        type=int,
        default=0,
        help="Stop after this many issues (0 = all). Useful for quick tests.",
    )
    args = parser.parse_args()

    # --- Config -----------------------------------------------------------------
    config = load_config(args.config)
    config.memory_enabled = True
    config.memory_writeback_enabled = False
    config.memory_ablation_levels = args.ablation_levels

    model_key = args.model_key
    llm = get_llm(model_key)
    tracker = TokenTracker(model_name=model_key, log_analyzer_type="llm")

    result_dir = (
        Path(str(config.project_result_dir))
        / f"{filesystem_safe_model_key(model_key)}_llm{_make_run_suffix(config)}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    token_report_path = str(result_dir / "token_report.json")
    tracker.load_prior_report(token_report_path)

    print(f"[ABLATION] levels={args.ablation_levels}  result_dir={result_dir}")

    # --- Load source log_details ------------------------------------------------
    if not args.source_log_details.exists():
        raise SystemExit(f"Source log_details not found: {args.source_log_details}")

    source_entries: List[Dict[str, Any]] = _load_json_list(str(args.source_log_details))
    # Filter out entries that had errors in the original run
    log_details_by_sha: Dict[str, Dict[str, Any]] = {
        str(entry.get("sha_fail") or ""): entry
        for entry in source_entries
        if entry.get("sha_fail") and not entry.get("error")
    }
    print(f"[ABLATION] Loaded {len(log_details_by_sha)} valid log entries from {args.source_log_details}")

    # Copy log_details.json into the ablation result dir so it's self-contained
    dest_log_details = result_dir / "log_details.json"
    with open(str(dest_log_details), "w", encoding="utf-8") as fh:
        json.dump(list(log_details_by_sha.values()), fh, indent=4)

    # --- Load prior progress (resumability) -------------------------------------
    fault_localization: List[Dict[str, Any]] = _load_json_list(str(result_dir / "fault_localization.json"))
    generated_patches: List[Dict[str, Any]] = _load_json_list(str(result_dir / "generated_patches.json"))

    done_shas = {str(p.get("sha_fail") or "") for p in generated_patches if p.get("sha_fail")}
    print(
        f"[ABLATION] Already done: {len(done_shas)}  "
        f"Remaining: {len(log_details_by_sha) - len(done_shas)}"
    )

    # --- Load dataset for metadata ----------------------------------------------
    dataset = _load_dataset(args.dataset, config)
    dataset_by_sha: Dict[str, Dict[str, Any]] = {
        str(row.get("sha_fail") or ""): row for row in dataset if row.get("sha_fail")
    }

    memory_plugin = MemoryPlugin(config=config, result_dir=str(result_dir), llm=llm)
    changed_files_dir = config.changed_files_folder

    processed = 0
    for sha_fail, log_analysis_result in log_details_by_sha.items():
        if sha_fail in done_shas:
            print(f"[ABLATION] Skipping {sha_fail} (already done)")
            continue

        if args.max_issues > 0 and processed >= args.max_issues:
            print(f"[ABLATION] Reached --max-issues={args.max_issues}, stopping.")
            break

        datapoint = dataset_by_sha.get(sha_fail)
        if not datapoint:
            print(f"[ABLATION] No dataset row for {sha_fail}, skipping")
            continue

        task_id = str(datapoint["id"])
        repo_name = datapoint["repo_name"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)
        benchmark_owner = config.benchmark_owner
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]

        print(f"\n[ABLATION] {sha_fail}  task={task_id}  repo={repo_name}")
        tracker.start_task(task_id)

        try:
            ensure_repo_at_commit(repo_url, repo_path, sha_fail)

            # Load changed files — already computed during the full run
            changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
            if not os.path.exists(changed_files_path):
                print(f"[ABLATION] No changed_files file for {sha_fail}, skipping")
                continue

            with open(changed_files_path, "r", encoding="utf-8") as fh:
                changed_files_info = json.load(fh)

            # --- Fault Localization ----------------------------------------------
            try:
                memory_context: Dict[str, Any] = {"enabled": False, "matches": []}
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

                _llm_fl = get_tracked_llm(model_key, tracker, "FaultLocalization")
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
                    "ablation_levels": args.ablation_levels,
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
                with open(str(result_dir / "fault_localization.json"), "w") as fh:
                    json.dump(fault_localization, fh, indent=4)

                if not fault_localizer.get("fault_localization_data"):
                    print(f"[ABLATION] No suspicious files for {sha_fail}, skipping patch gen")
                    continue

            except Exception as exc:
                print(f"[ERROR] FL failed for {sha_fail}: {exc}")
                continue

            # --- Patch Generation ------------------------------------------------
            try:
                _llm_pg = get_tracked_llm(model_key, tracker, "PatchGeneration")
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
                    print(f"[ABLATION] No patch generated for {sha_fail}")
                    continue

                generated_patches = _merge_by_key(generated_patches, patch_generator, "sha_fail")
                with open(str(result_dir / "generated_patches.json"), "w") as fh:
                    json.dump(generated_patches, fh, indent=4)

                # Incremental FL evaluation — updates fl_evaluation.json after every
                # issue so scores are visible without waiting for the full run to finish.
                try:
                    evaluate_fl(
                        fault_localization_path=str(result_dir / "fault_localization.json"),
                        generated_patches_path=str(result_dir / "generated_patches.json"),
                        output_path=str(result_dir / "fl_evaluation.json"),
                    )
                except Exception as _fl_err:
                    print(f"[ABLATION] Incremental FL evaluation failed: {_fl_err}")

                processed += 1

            except Exception as exc:
                print(f"[ERROR] Patch gen failed for {sha_fail}: {exc}")
                continue

        finally:
            tracker.end_task(task_id)
            tracker.save_json(token_report_path)
            tracker.save_step_trace(str(result_dir / "step_trace.json"))

    # --- FL evaluation ----------------------------------------------------------
    fl_path = result_dir / "fault_localization.json"
    patch_path = result_dir / "generated_patches.json"
    if fl_path.exists() and patch_path.exists():
        try:
            evaluate_fl(
                fault_localization_path=str(fl_path),
                generated_patches_path=str(patch_path),
                output_path=str(result_dir / "fl_evaluation.json"),
            )
        except Exception as fl_err:
            print(f"[ABLATION] FL evaluation failed: {fl_err}")

    tracker.print_summary()
    print(f"\n[ABLATION] Finished. Results in {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
