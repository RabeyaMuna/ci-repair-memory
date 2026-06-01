#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from main import process_entire_dataset, _make_run_suffix  # noqa: E402
from utilities.load_config import load_config  # noqa: E402
from utilities.llm_provider import filesystem_safe_model_key, get_default_model_key, get_llm  # noqa: E402
from utilities.token_tracker import TokenTracker  # noqa: E402
from utilities.fl_evaluator import evaluate_fl  # noqa: E402


DEFAULT_SPLIT = PROJECT_ROOT / "baselines" / "results" / "trs" / "eval_issues.json"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
def _load_dataset(local_dataset: Path, config) -> List[Dict[str, Any]]:
    if local_dataset.exists():
        df = pd.read_parquet(local_dataset)
        return df.to_dict(orient="records")

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")
    dataset_path = hf_hub_download(
        repo_id="ci-benchmark-user/ci-repair-bench",
        filename="ci_repair_dataset.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(dataset_path)
    return df.to_dict(orient="records")


def _filter_eval_rows(
    split_rows: List[Dict[str, Any]],
    repos: List[str],
    max_per_repo: int,
) -> List[Dict[str, Any]]:
    if not repos:
        repos = sorted({str(row.get("repo_name") or "") for row in split_rows if str(row.get("repo_name") or "")})

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        repo_name = str(row.get("repo_name") or "")
        if repo_name in repos:
            grouped[repo_name].append(row)

    selected: List[Dict[str, Any]] = []
    for repo in repos:
        rows = grouped.get(repo, [])
        if max_per_repo > 0:
            rows = rows[:max_per_repo]
        selected.extend(rows)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline pipeline only on the remaining eval issues for selected repos.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repos", nargs="*", default=[])
    parser.add_argument("--max-per-repo", type=int, default=0, help="0 means all remaining eval issues for each repo.")
    parser.add_argument("--memory-mode", choices=("baseline", "memory"), default="baseline")
    parser.add_argument(
        "--ablation-levels",
        choices=("L1", "L1+L2", "L1+L2+L3"),
        default="L1+L2+L3",
        help="Memory ablation: which levels to activate. Only applies with --memory-mode memory.",
    )
    parser.add_argument("--model-key", default=get_default_model_key())
    args = parser.parse_args()

    # load_config() normalises all paths to absolute (resolves relative paths
    # from the project root), so repo_path and other dirs are unambiguous when
    # passed to subprocesses that run in a different cwd.
    config = load_config(args.config)
    config.memory_enabled = args.memory_mode == "memory"
    config.memory_writeback_enabled = False
    config.resume_skip_generated_patches = True
    if args.memory_mode == "memory":
        config.memory_ablation_levels = args.ablation_levels

    split_rows = json.loads(args.split_file.read_text(encoding="utf-8"))
    eval_rows = _filter_eval_rows(split_rows, args.repos, args.max_per_repo)
    eval_sha = {str(row["sha_fail"]) for row in eval_rows}

    dataset = _load_dataset(args.dataset, config)
    subset = [row for row in dataset if str(row.get("sha_fail") or "") in eval_sha]
    subset.sort(key=lambda row: str(row.get("id") or ""))

    if not subset:
        raise SystemExit("No eval issues matched the requested repos/split.")

    model_key = args.model_key
    llm = get_llm(model_key)
    tracker = TokenTracker(model_name=model_key)

    result = process_entire_dataset(
        subset,
        config,
        llm,
        model_key,
        tracker=tracker,
    )

    result_dir = Path(str(config.project_result_dir)) / f"{filesystem_safe_model_key(model_key)}{_make_run_suffix(config)}"
    result_dir.mkdir(parents=True, exist_ok=True)

    tracker.save_json(str(result_dir / "token_report.json"))
    tracker.save_step_trace(str(result_dir / "step_trace.json"))

    manifest = {
        "memory_mode": args.memory_mode,
        "ablation_levels": args.ablation_levels if args.memory_mode == "memory" else None,
        "repos": args.repos,
        "max_per_repo": args.max_per_repo,
        "tasks_run": len(subset),
        "task_ids": [str(row.get("id") or "") for row in subset],
        "sha_fails": [str(row.get("sha_fail") or "") for row in subset],
    }
    (result_dir / "eval_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fl_path = result_dir / "fault_localization.json"
    patch_path = result_dir / "generated_patches.json"
    if fl_path.exists() and patch_path.exists():
        evaluate_fl(
            fault_localization_path=str(fl_path),
            generated_patches_path=str(patch_path),
            output_path=str(result_dir / "fl_evaluation.json"),
        )

    print(
        json.dumps(
            {
                "result_dir": str(result_dir),
                "model_key": model_key,
                "memory_mode": args.memory_mode,
                "ablation_levels": args.ablation_levels if args.memory_mode == "memory" else None,
                "tasks_run": len(subset),
                "repos": args.repos,
                "returned_patches": len(result) if isinstance(result, list) else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
