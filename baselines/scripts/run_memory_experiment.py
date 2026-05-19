#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "baselines" / "scripts"
DEFAULT_OUTPUT = PROJECT_ROOT / "baselines" / "results" / "memory_experiment"


def _safe_model_dir_key(model_key: str) -> str:
    return str(model_key or "").strip().replace("/", "__")


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the full memory-vs-baseline experiment workflow.")
    parser.add_argument("--per-repo", type=int, default=5)
    parser.add_argument("--max-per-repo", type=int, default=0, help="0 means run all remaining eval issues.")
    parser.add_argument("--model-key", default=os.getenv("MEMCI_LLM_MODEL", "gpt-5-mini"))
    parser.add_argument("--log-analyzer-type", choices=("llm", "bm25"), default="llm")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python-bin", default=str(PROJECT_ROOT / "baselines" / ".venv" / "bin" / "python"))
    args = parser.parse_args()

    python_bin = args.python_bin
    cwd = PROJECT_ROOT
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Prepare split + seed memory artifacts
    _run(
        [
            python_bin,
            str(SCRIPTS_DIR / "prepare_memory_seed_split.py"),
            "--per-repo",
            str(args.per_repo),
        ],
        cwd=cwd,
    )

    # 2) Analyze only the seed issues with the CI log analyzer
    _run(
        [
            python_bin,
            str(SCRIPTS_DIR / "analyze_memory_seed_issues.py"),
            "--model-key",
            args.model_key,
        ],
        cwd=cwd,
    )

    # 3) Build the memory bank from analyzed seed issues
    _run(
        [
            python_bin,
            str(SCRIPTS_DIR / "build_memory_bank_from_seed_analysis.py"),
        ],
        cwd=cwd,
    )

    # 4) Publish the prepared memory bank into baselines/results/
    _run(
        [
            python_bin,
            str(SCRIPTS_DIR / "publish_memory_bank.py"),
        ],
        cwd=cwd,
    )

    # 5) Run baseline on remaining issues
    _run(
        [
            python_bin,
            str(SCRIPTS_DIR / "run_repo_eval_subset.py"),
            "--memory-mode",
            "baseline",
            "--max-per-repo",
            str(args.max_per_repo),
            "--model-key",
            args.model_key,
            "--log-analyzer-type",
            args.log_analyzer_type,
        ],
        cwd=cwd,
    )

    # 6) Run memory mode on the same remaining issues
    _run(
        [
            python_bin,
            str(SCRIPTS_DIR / "run_repo_eval_subset.py"),
            "--memory-mode",
            "memory",
            "--max-per-repo",
            str(args.max_per_repo),
            "--model-key",
            args.model_key,
            "--log-analyzer-type",
            args.log_analyzer_type,
        ],
        cwd=cwd,
    )

    # 7) Compare the two result directories
    model_dir_key = _safe_model_dir_key(args.model_key)
    baseline_dir = PROJECT_ROOT / "baselines" / "results" / f"{model_dir_key}_{args.log_analyzer_type}_baseline"
    memory_dir = PROJECT_ROOT / "baselines" / "results" / f"{model_dir_key}_{args.log_analyzer_type}_memory"

    compare_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "compare_memory_runs.py"),
        str(baseline_dir),
        str(memory_dir),
    ]
    print(f"[RUN] {' '.join(compare_cmd)}")
    compare_proc = subprocess.run(
        compare_cmd,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )

    comparison = json.loads(compare_proc.stdout)
    split_summary = _load_json(PROJECT_ROOT / "baselines" / "results" / "memory_seed_preparation" / "summary.json")

    report = {
        "experiment": {
            "per_repo_seed_count": args.per_repo,
            "max_per_repo_eval": args.max_per_repo,
            "model_key": args.model_key,
            "log_analyzer_type": args.log_analyzer_type,
        },
        "split_summary": split_summary,
        "comparison": comparison,
        "paths": {
            "baseline_result_dir": str(baseline_dir),
            "memory_result_dir": str(memory_dir),
        },
    }

    report_path = args.output_dir / "experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"report_path": str(report_path), "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
