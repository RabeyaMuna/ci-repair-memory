# utilities/fl_evaluator.py
"""
Fault-Localization evaluation for RQ1.

Ground truth
------------
The files that actually needed to be changed are extracted from the diff
headers in ``generated_patches.json``:

    diff --git a/path/to/file.py b/path/to/file.py

These are the files the pipeline (or a reference solution) ultimately had to
patch in order to fix the CI failure.  ``changed_files/`` is intentionally
NOT used here — those JSON files record every file touched by the failing
commit, not the files that contain the bug.

Predicted
---------
The files predicted by the FaultLocalization agent, taken from
``fault_localization.json``.

Metrics (per task and aggregate)
---------------------------------
  exact_match   : predicted set == patched-files set
  hit_at_1      : top-1 predicted file is in the patched set
  hit_at_3      : any of top-3 predicted files is in the patched set
  hit_at_5      : any of top-5 predicted files is in the patched set
  precision     : |predicted ∩ patched| / |predicted|
  recall        : |predicted ∩ patched| / |patched|
  f1            : harmonic mean of precision and recall

Usage
-----
    from utilities.fl_evaluator import evaluate_fl

    report = evaluate_fl(
        fault_localization_path="results/gpt-5-mini_llm/fault_localization.json",
        generated_patches_path="results/gpt-5-mini_llm/generated_patches.json",
    )
    # saved automatically next to fault_localization.json as fl_evaluation.json
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalise(path: str) -> str:
    return path.strip().lstrip("/").replace("\\", "/")


def _extract_files_from_diff(diff_text: str) -> List[str]:
    """
    Parse every ``diff --git a/... b/...`` header in a unified diff and return
    the list of target file paths (the ``b/`` side, which is the file after
    the patch is applied).
    """
    files = []
    for match in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text, re.MULTILINE):
        fp = match.group(1).strip()
        if fp:
            files.append(fp)
    return files


def _precision_recall_f1(
    predicted: List[str], ground_truth: List[str]
) -> Dict[str, float]:
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(_normalise(p) for p in predicted)
    gt_set   = set(_normalise(g) for g in ground_truth)
    tp = len(pred_set & gt_set)

    precision = tp / len(pred_set)
    recall    = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def _hit_at_k(predicted: List[str], ground_truth: List[str], k: int) -> bool:
    gt_set = set(_normalise(g) for g in ground_truth)
    for p in predicted[:k]:
        if _normalise(p) in gt_set:
            return True
    return False


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def evaluate_fl(
    fault_localization_path: str,
    generated_patches_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    Parameters
    ----------
    fault_localization_path : path to fault_localization.json
    generated_patches_path  : path to generated_patches.json (same result dir)
    output_path             : where to write fl_evaluation.json
                              (defaults to same dir as fault_localization_path)

    Returns
    -------
    Full evaluation report dict (also written to output_path).
    """
    fl_records      = _load_json(fault_localization_path)  # list
    patch_records   = _load_json(generated_patches_path)   # list

    # Build a lookup: sha_fail -> list of patched files (from diff headers)
    patched_files_by_sha: Dict[str, List[str]] = {}
    for rec in patch_records:
        sha  = rec.get("sha_fail", "")
        diff = rec.get("diff", "")
        if sha and diff:
            patched_files_by_sha[sha] = _extract_files_from_diff(diff)

    per_task: List[dict] = []
    skipped:  List[str]  = []

    for record in fl_records:
        sha_fail = record.get("sha_fail", "")
        task_id  = str(record.get("id", ""))

        # Predicted files — ordered as returned by FL
        fl_data         = record.get("fault_localization_data") or []
        predicted_files = [e["file_path"] for e in fl_data if e.get("file_path")]

        # Ground-truth files — from the generated patch diff headers
        ground_truth_files = patched_files_by_sha.get(sha_fail)
        if ground_truth_files is None:
            # No patch was generated for this sha_fail
            skipped.append(sha_fail)
            continue
        if not ground_truth_files:
            skipped.append(sha_fail)
            continue

        metrics = _precision_recall_f1(predicted_files, ground_truth_files)

        per_task.append({
            "task_id":             task_id,
            "sha_fail":            sha_fail,
            "predicted_files":     predicted_files,
            "ground_truth_files":  ground_truth_files,   # files actually patched
            "exact_match": (
                set(_normalise(p) for p in predicted_files)
                == set(_normalise(g) for g in ground_truth_files)
            ),
            "hit_at_1": _hit_at_k(predicted_files, ground_truth_files, 1),
            "hit_at_3": _hit_at_k(predicted_files, ground_truth_files, 3),
            "hit_at_5": _hit_at_k(predicted_files, ground_truth_files, 5),
            "num_predicted":     len(predicted_files),
            "num_ground_truth":  len(ground_truth_files),
            **metrics,
        })

    # ---------- aggregate ----------
    n = len(per_task)
    if n:
        aggregate = {
            "tasks_evaluated":                n,
            "tasks_skipped_no_patch":         len(skipped),
            "exact_match_rate":  round(sum(t["exact_match"] for t in per_task) / n, 4),
            "hit_at_1_rate":     round(sum(t["hit_at_1"]    for t in per_task) / n, 4),
            "hit_at_3_rate":     round(sum(t["hit_at_3"]    for t in per_task) / n, 4),
            "hit_at_5_rate":     round(sum(t["hit_at_5"]    for t in per_task) / n, 4),
            "mean_precision":    round(sum(t["precision"]   for t in per_task) / n, 4),
            "mean_recall":       round(sum(t["recall"]      for t in per_task) / n, 4),
            "mean_f1":           round(sum(t["f1"]          for t in per_task) / n, 4),
        }
    else:
        aggregate = {
            "tasks_evaluated":        0,
            "tasks_skipped_no_patch": len(skipped),
        }

    report = {
        "aggregate":          aggregate,
        "per_task":           per_task,
        "skipped_sha_fails":  skipped,
    }

    # ---------- save ----------
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(fault_localization_path), "fl_evaluation.json"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[FL Evaluator] Saved → {output_path}")

    return report
