#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"parse_error": True, "raw": line})
    return rows


def _index_by_sha(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("sha_fail") or ""): row for row in rows if str(row.get("sha_fail") or "")}


def _task_steps(step_trace: Any, task_id: str, sha_fail: str) -> Any:
    if isinstance(step_trace, dict):
        for key in (task_id, sha_fail):
            if key and key in step_trace:
                return step_trace[key]
        return step_trace
    if isinstance(step_trace, list):
        return [
            row for row in step_trace
            if str(row.get("task_id") or row.get("id") or "") == task_id
            or str(row.get("sha_fail") or "") == sha_fail
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect per-task memory trajectories for one ablation run.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--shared-log-details", type=Path, required=True)
    parser.add_argument("--retrieval-log", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ablation-levels", required=True)
    args = parser.parse_args()

    eval_rows = _load_json(args.eval_file, [])
    log_details = _index_by_sha(_load_json(args.shared_log_details, []))
    fl_rows = _index_by_sha(_load_json(args.result_dir / "fault_localization.json", []))
    patch_rows = _index_by_sha(_load_json(args.result_dir / "generated_patches.json", []))
    retrieval_rows = _load_jsonl(args.retrieval_log)
    step_trace = _load_json(args.result_dir / "step_trace.json", [])
    token_report = _load_json(args.result_dir / "token_report.json", {})
    fl_eval = _load_json(args.result_dir / "fl_evaluation.json", {})

    retrieval_by_sha: Dict[str, List[Dict[str, Any]]] = {}
    for row in retrieval_rows:
        query = row.get("query", {}) if isinstance(row, dict) else {}
        sha_fail = str(query.get("sha_fail") or "")
        if sha_fail:
            retrieval_by_sha.setdefault(sha_fail, []).append(row)

    trajectories = []
    for eval_row in eval_rows:
        sha_fail = str(eval_row.get("sha_fail") or "")
        task_id = str(eval_row.get("id") or "")
        trajectories.append(
            {
                "id": task_id,
                "sha_fail": sha_fail,
                "repo_name": eval_row.get("repo_name"),
                "ablation_levels": args.ablation_levels,
                "input_issue": eval_row,
                "log_analysis": log_details.get(sha_fail),
                "memory_retrievals": retrieval_by_sha.get(sha_fail, []),
                "fault_localization": fl_rows.get(sha_fail),
                "generated_patch": patch_rows.get(sha_fail),
                "step_trace": _task_steps(step_trace, task_id, sha_fail),
            }
        )

    summary = {
        "ablation_levels": args.ablation_levels,
        "result_dir": str(args.result_dir),
        "shared_log_details": str(args.shared_log_details),
        "retrieval_log": str(args.retrieval_log),
        "tasks": len(trajectories),
        "with_memory_retrievals": sum(1 for row in trajectories if row["memory_retrievals"]),
        "with_fault_localization": sum(1 for row in trajectories if row["fault_localization"]),
        "with_generated_patch": sum(1 for row in trajectories if row["generated_patch"]),
        "token_report": token_report,
        "fl_evaluation": fl_eval,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectories.json").write_text(
        json.dumps(trajectories, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
