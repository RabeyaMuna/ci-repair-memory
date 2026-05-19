#!/usr/bin/env python3
"""
backfill_memory_summary.py

Injects a `memory_summary` field into every entry of an existing
fault_localization.json by re-running the memory retrieval step.

Memory retrieval is pure lexical scoring — no LLM calls, no API cost.
The script reads log_details.json and changed_files/{sha_fail}.json
(already on disk from the original run) and calls MemoryPlugin.retrieve()
for each issue, then writes the result back into fault_localization.json.

Usage
-----
# Backfill the full L1+L2+L3 run
baselines/.venv/bin/python baselines/scripts/backfill_memory_summary.py \\
  --result-dir baselines/results/MiniMax-M2.5_llm_memory \\
  --ablation-levels L1+L2+L3

# Dry-run (print what would be added, do not write)
baselines/.venv/bin/python baselines/scripts/backfill_memory_summary.py \\
  --result-dir baselines/results/MiniMax-M2.5_llm_memory \\
  --ablation-levels L1+L2+L3 \\
  --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from utilities.memory_plugin import MemoryPlugin  # noqa: E402

load_dotenv()

DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill memory_summary into an existing fault_localization.json (no LLM calls)."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to the result directory containing fault_localization.json and log_details.json.",
    )
    parser.add_argument(
        "--ablation-levels",
        choices=("L1", "L1+L2", "L1+L2+L3"),
        default="L1+L2+L3",
        help="Which memory levels were active during this run.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing memory_summary entries (use after fixing the scoring formula).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without writing to disk.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    fl_path = result_dir / "fault_localization.json"
    log_details_path = result_dir / "log_details.json"

    if not fl_path.exists():
        raise SystemExit(f"fault_localization.json not found in {result_dir}")
    if not log_details_path.exists():
        raise SystemExit(f"log_details.json not found in {result_dir}")

    # --- Config -----------------------------------------------------------------
    config = OmegaConf.load(str(args.config))
    config.memory_enabled = True
    config.memory_writeback_enabled = False
    config.memory_ablation_levels = args.ablation_levels

    # --- Memory plugin (no LLM needed — retrieval is pure lexical) --------------
    memory_plugin = MemoryPlugin(config=config, result_dir=str(result_dir), llm=None)
    if not memory_plugin.failure_memory:
        raise SystemExit(
            "Memory bank is empty — make sure failure_memory.json / repo_memory.json / "
            "cross_memory.json exist in config.project_result_dir."
        )

    print(
        f"[BACKFILL] Memory bank: "
        f"L1={len(memory_plugin.failure_memory)}  "
        f"L2={len(memory_plugin.repo_memory)}  "
        f"L3={len(memory_plugin.cross_memory)}"
    )

    # --- Load existing data -----------------------------------------------------
    fl_records: List[Dict[str, Any]] = _load_json_list(str(fl_path))
    log_details: List[Dict[str, Any]] = _load_json_list(str(log_details_path))

    log_by_sha: Dict[str, Dict[str, Any]] = {
        str(e.get("sha_fail") or ""): e for e in log_details if e.get("sha_fail")
    }

    # Load dataset for repo_name and workflow_path (not stored in log_details)
    dataset_by_sha: Dict[str, Dict[str, Any]] = {}
    dataset_path = args.dataset
    if dataset_path.exists():
        df = pd.read_parquet(str(dataset_path))
        for row in df.to_dict(orient="records"):
            sha = str(row.get("sha_fail") or "")
            if sha:
                dataset_by_sha[sha] = row
        print(f"[BACKFILL] Loaded {len(dataset_by_sha)} dataset rows for repo metadata")
    else:
        hf_token = os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")
        hf_path = hf_hub_download(
            repo_id="ci-benchmark-user/ci-repair-bench",
            filename="ci_repair_dataset.parquet",
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(hf_path)
        for row in df.to_dict(orient="records"):
            sha = str(row.get("sha_fail") or "")
            if sha:
                dataset_by_sha[sha] = row
        print(f"[BACKFILL] Loaded {len(dataset_by_sha)} dataset rows from HuggingFace")

    changed_files_dir = str(config.changed_files_folder)

    already_filled = sum(1 for r in fl_records if "memory_summary" in r)
    print(
        f"[BACKFILL] {len(fl_records)} FL entries  |  "
        f"{already_filled} already have memory_summary  |  "
        f"{len(fl_records) - already_filled} to backfill"
    )

    updated = 0
    skipped_no_log = 0
    skipped_no_changed_files = 0

    for record in fl_records:
        sha_fail = str(record.get("sha_fail") or "")
        task_id = str(record.get("id") or "")

        if "memory_summary" in record and not args.force:
            continue  # already backfilled

        log_analysis_result = log_by_sha.get(sha_fail)
        if not log_analysis_result:
            print(f"[BACKFILL] No log_details entry for {sha_fail}, skipping")
            skipped_no_log += 1
            continue

        changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
        if not os.path.exists(changed_files_path):
            print(f"[BACKFILL] No changed_files for {sha_fail}, skipping")
            skipped_no_changed_files += 1
            changed_files_info = None
        else:
            with open(changed_files_path, encoding="utf-8") as f:
                changed_files_info = json.load(f)

        # Re-run retrieval (pure lexical — no LLM)
        datapoint = dataset_by_sha.get(sha_fail, {})
        repo_name = str(datapoint.get("repo_name") or "")
        workflow_path = str(datapoint.get("workflow_path") or "")
        workflow = datapoint.get("workflow") or ""

        try:
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
        except Exception as exc:
            print(f"[BACKFILL] Retrieval failed for {sha_fail}: {exc}")
            memory_context = {"enabled": False, "matches": [], "level_scores": {"L1": 0.0, "L2": 0.0, "L3": 0.0}, "weighted_similarity": 0.0, "selected_memory_levels": [], "candidate_files": [], "high_level_hints": [], "l1_matches": [], "l2_matches": [], "l3_matches": []}

        memory_summary = {
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

        if args.dry_run:
            print(
                f"  [{sha_fail[:12]}] L1={memory_summary['level_scores']['L1']:.3f} "
                f"L2={memory_summary['level_scores']['L2']:.3f} "
                f"L3={memory_summary['level_scores']['L3']:.3f} "
                f"weighted={memory_summary['weighted_similarity']:.3f} "
                f"injected={memory_summary['memory_injected']} "
                f"suppressed={memory_summary['suppressed_reason']}"
            )
        else:
            record["memory_summary"] = memory_summary

        updated += 1

    print(
        f"\n[BACKFILL] Done: {updated} entries processed  |  "
        f"skipped (no log): {skipped_no_log}  |  "
        f"skipped (no changed_files): {skipped_no_changed_files}"
    )

    if args.dry_run:
        print("[BACKFILL] Dry-run mode — nothing written.")
        return 0

    with open(str(fl_path), "w", encoding="utf-8") as f:
        json.dump(fl_records, f, indent=4)

    print(f"[BACKFILL] Saved updated fault_localization.json → {fl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
