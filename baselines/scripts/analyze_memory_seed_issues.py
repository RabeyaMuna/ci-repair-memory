#!/usr/bin/env python3
"""
Phase 2 of the memory bank pipeline:
  Run CILogAnalyzerLLM on each seed issue → save structured error context.

Inputs:
  --seed-file    trs_memory_seed_issues.json  (from the enriched TRS memory split)
  --dataset      lca_dataset.parquet
  --config       config.yaml

Outputs (in --output-dir):
  seed_log_details.json          — one record per seed with full LLM analysis
  seed_log_analysis_manifest.json — run summary / failure list

Resumable: if seed_log_details.json already exists, already-analyzed sha_fail
values are skipped so a partial run can be continued.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from utilities.ensure_repo import ensure_repo_at_commit
from utilities.llm_provider import get_default_model_key, get_llm


DEFAULT_SEED_FILE = PROJECT_ROOT / "baselines" / "results" / "trs" / "trs_memory_seed_issues.json"
DEFAULT_DATASET   = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_CONFIG    = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT    = PROJECT_ROOT / "baselines" / "results" / "trs"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_existing(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _analysis_has_error(entry: Dict[str, Any]) -> bool:
    analysis = entry.get("analysis")
    return not isinstance(analysis, dict) or bool(analysis.get("error"))


def _safe_scalar(value: Any, default: Any = "") -> Any:
    if value is None:
        return default
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _safe_logs(value: Any) -> List[Dict[str, Any]]:
    value = _safe_scalar(value, default=[])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, dict) else []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze selected memory seed issues with CILogAnalyzerLLM."
    )
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--dataset",   type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config",    type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir",type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-key", default=get_default_model_key())
    args = parser.parse_args()

    config     = OmegaConf.load(str(args.config))
    seed_rows  = json.loads(args.seed_file.read_text(encoding="utf-8"))
    df         = pd.read_parquet(args.dataset)
    dataset_rows = df.to_dict(orient="records")
    row_by_sha = {str(row.get("sha_fail") or ""): row for row in dataset_rows}

    model_key  = args.model_key
    llm        = get_llm(model_key)

    out_path   = args.output_dir / "seed_log_details.json"
    analyzed: List[Dict[str, Any]] = _load_existing(out_path)
    analyzed_by_sha = {
        str(entry.get("sha_fail") or ""): entry
        for entry in analyzed
        if str(entry.get("sha_fail") or "")
    }
    done_shas = {
        sha_fail
        for sha_fail, entry in analyzed_by_sha.items()
        if not _analysis_has_error(entry)
    }
    failures:  List[Dict[str, Any]] = []

    benchmark_owner    = str(config.get("benchmark_owner", ""))
    repo_cloned_folder = str(config.get("baseline_repo_folder", ""))

    for seed in seed_rows:
        sha_fail  = str(seed.get("sha_fail") or "")
        task_id   = str(seed.get("id") or "")

        if sha_fail in done_shas:
            print(f"[skip] {sha_fail} already analyzed.")
            continue

        dataset_row = row_by_sha.get(sha_fail)
        if not dataset_row:
            failures.append({"sha_fail": sha_fail, "id": task_id, "error": "dataset_row_not_found"})
            continue

        repo_name = str(dataset_row.get("repo_name") or "")
        repo_path = str(Path(repo_cloned_folder) / repo_name)
        repo_url  = f"https://github.com/{benchmark_owner}/{repo_name}.git"

        # Clone / checkout repo at the failing commit
        try:
            ensure_repo_at_commit(repo_url, repo_path, sha_fail)
        except Exception as clone_err:
            failures.append({"sha_fail": sha_fail, "id": task_id, "error": f"clone_failed: {clone_err}"})
            continue

        ci_logs = _safe_logs(dataset_row.get("logs"))
        if not ci_logs:
            result = {
                "error": "missing_logs",
                "message": "Dataset row does not contain CI logs for this sha_fail.",
                "sha_fail": sha_fail,
                "id": str(_safe_scalar(dataset_row.get("id"), default=task_id) or task_id),
            }
        else:
            analyzer = CILogAnalyzerLLM(
            repo_path    = repo_path,
            ci_log       = ci_logs,
            sha_fail     = sha_fail,
            workflow     = str(_safe_scalar(dataset_row.get("workflow"), default="") or ""),
            workflow_path= str(_safe_scalar(dataset_row.get("workflow_path"), default="") or ""),
            llm          = llm,
            model_name   = model_key,
            task_id      = str(_safe_scalar(dataset_row.get("id"), default=task_id) or task_id),
        )

        try:
            if ci_logs:
                result = analyzer.run()
            entry = {
                "id":            str(dataset_row.get("id") or task_id),
                "sha_fail":      sha_fail,
                "repo_name":     repo_name,
                "repo_owner":    str(_safe_scalar(dataset_row.get("repo_owner"), default="") or ""),
                "workflow_name": str(_safe_scalar(dataset_row.get("workflow_name"), default="") or ""),
                "workflow_path": str(_safe_scalar(dataset_row.get("workflow_path"), default="") or ""),
                "workflow":      str(_safe_scalar(dataset_row.get("workflow"), default="") or ""),
                "analysis":      result,
            }
            analyzed_by_sha[sha_fail] = entry
            if not _analysis_has_error(entry):
                done_shas.add(sha_fail)
            # Flush after each issue so a crash doesn't lose work
            analyzed = list(analyzed_by_sha.values())
            _write_json(out_path, analyzed)
            print(f"[done] {sha_fail} ({repo_name})")
        except Exception as exc:
            failures.append({"sha_fail": sha_fail, "id": task_id, "error": str(exc)})

    manifest = {
        "model_key":  model_key,
        "seed_file":  str(args.seed_file),
        "dataset":    str(args.dataset),
        "requested":  len(seed_rows),
        "analyzed":   len([entry for entry in analyzed_by_sha.values() if not _analysis_has_error(entry)]),
        "failed":     len(failures),
        "failures":   failures,
        "output":     str(out_path),
    }
    _write_json(args.output_dir / "seed_log_analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
