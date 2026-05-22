#!/usr/bin/env python3
"""
ablation_comparison.py

Compares Baseline vs L1 vs L1+L2 vs L1+L2+L3 ablation results and prints
a clean table ready for a research presentation / PPT.

Usage
-----
baselines/.venv/bin/python baselines/scripts/ablation_comparison.py

Outputs
-------
  - Console: formatted comparison table
  - ablation_comparison.json  — raw numbers for further analysis
  - ablation_comparison.csv   — import directly into Excel / Google Sheets for PPT
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "baselines" / "results"

# Add project root to path so utilities can be imported
sys.path.insert(0, str(PROJECT_ROOT / "baselines"))
from utilities.fl_evaluator import evaluate_fl  # noqa: E402

def _total_issues() -> int:
    """Count total benchmark issues from the baseline log_details (source of truth)."""
    log_path = RESULTS_DIR / "MiniMax-M2.5_llm_baseline" / "log_details.json"
    if not log_path.exists():
        return 87
    data = _load(log_path) or []
    return len([r for r in data if r.get("sha_fail") and not r.get("error")])

RUNS = [
    {
        "label":       "Baseline",
        "short":       "BASE",
        "result_dir":  RESULTS_DIR / "MiniMax-M2.5_llm_baseline",
        "has_memory":  False,
    },
    {
        "label":       "Memory L1",
        "short":       "L1",
        "result_dir":  RESULTS_DIR / "MiniMax-M2.5_llm_memory_L1",
        "has_memory":  True,
    },
    {
        "label":       "Memory L1+L2",
        "short":       "L1+L2",
        "result_dir":  RESULTS_DIR / "MiniMax-M2.5_llm_memory_L1L2",
        "has_memory":  True,
    },
    {
        "label":       "Memory L1+L2+L3",
        "short":       "L1+L2+L3",
        "result_dir":  RESULTS_DIR / "MiniMax-M2.5_llm_memory",
        "has_memory":  True,
    },
]


def _load(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalise(path: str) -> str:
    return path.strip().lstrip("/").replace("\\", "/")


def _extract_files_from_diff(diff_text: str) -> List[str]:
    files = []
    for match in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text, re.MULTILINE):
        fp = match.group(1).strip()
        if fp:
            files.append(fp)
    return files


def _precision_recall_f1(predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(_normalise(p) for p in predicted)
    gt_set = set(_normalise(g) for g in ground_truth)
    tp = len(pred_set & gt_set)

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _hit_at_k(predicted: List[str], ground_truth: List[str], k: int) -> bool:
    gt_set = set(_normalise(g) for g in ground_truth)
    for p in predicted[:k]:
        if _normalise(p) in gt_set:
            return True
    return False


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _delta(v: float, base: float) -> str:
    d = (v - base) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"


def summarize_run(run: Dict) -> Dict[str, Any]:
    d          = Path(run["result_dir"])
    patches    = _load(d / "generated_patches.json") or []
    fl_records = _load(d / "fault_localization.json") or []

    # Regenerate fl_evaluation.json with changed_files augmentation
    fl_eval_path = d / "fl_evaluation.json"
    if (d / "fault_localization.json").exists() and (d / "generated_patches.json").exists():
        try:
            evaluate_fl(
                fault_localization_path=str(d / "fault_localization.json"),
                generated_patches_path=str(d / "generated_patches.json"),
                output_path=str(fl_eval_path),
            )
        except Exception as e:
            print(f"  [WARN] Could not evaluate {run['label']}: {e}")

    fl_eval = _load(fl_eval_path)
    agg = (fl_eval or {}).get("aggregate", {})

    # ── Memory-specific metrics (from memory_summary embedded in FL entries) ──
    mem_injected   = 0
    sim_scores     = []
    l1_scores      = []
    l2_scores      = []
    l3_scores      = []
    retrieval_tasks = 0
    retrieval_hit1 = 0
    retrieval_hit3 = 0
    retrieval_hit5 = 0
    retrieval_precision = 0.0
    retrieval_recall = 0.0
    retrieval_f1 = 0.0

    patched_files_by_sha: Dict[str, List[str]] = {}
    for rec in patches:
        sha = rec.get("sha_fail", "")
        diff = rec.get("diff", "")
        if sha and diff:
            patched_files_by_sha[sha] = _extract_files_from_diff(diff)

    if run["has_memory"]:
        for rec in fl_records:
            ms = rec.get("memory_summary") or {}
            if ms.get("memory_injected"):
                mem_injected += 1
            ws = ms.get("weighted_similarity", 0.0)
            if ws and ws > 0:
                sim_scores.append(ws)
            ls = ms.get("level_scores") or {}
            if ls.get("L1", 0) > 0: l1_scores.append(ls["L1"])
            if ls.get("L2", 0) > 0: l2_scores.append(ls["L2"])
            if ls.get("L3", 0) > 0: l3_scores.append(ls["L3"])

            ground_truth_files = patched_files_by_sha.get(rec.get("sha_fail", ""))
            if not ground_truth_files:
                continue
            predicted_files = ms.get("candidate_files") or []
            metrics = _precision_recall_f1(predicted_files, ground_truth_files)
            retrieval_tasks += 1
            retrieval_hit1 += int(_hit_at_k(predicted_files, ground_truth_files, 1))
            retrieval_hit3 += int(_hit_at_k(predicted_files, ground_truth_files, 3))
            retrieval_hit5 += int(_hit_at_k(predicted_files, ground_truth_files, 5))
            retrieval_precision += metrics["precision"]
            retrieval_recall += metrics["recall"]
            retrieval_f1 += metrics["f1"]

    total_fl = len(fl_records)
    retrieval_agg = {
        "tasks_evaluated": retrieval_tasks,
        "hit_at_1": round(retrieval_hit1 / retrieval_tasks, 4) if retrieval_tasks else 0.0,
        "hit_at_3": round(retrieval_hit3 / retrieval_tasks, 4) if retrieval_tasks else 0.0,
        "hit_at_5": round(retrieval_hit5 / retrieval_tasks, 4) if retrieval_tasks else 0.0,
        "precision": round(retrieval_precision / retrieval_tasks, 4) if retrieval_tasks else 0.0,
        "recall": round(retrieval_recall / retrieval_tasks, 4) if retrieval_tasks else 0.0,
        "f1": round(retrieval_f1 / retrieval_tasks, 4) if retrieval_tasks else 0.0,
    }

    return {
        "label":              run["label"],
        "short":              run["short"],
        "has_memory":         run["has_memory"],
        # Coverage
        "total_issues":       _total_issues(),
        "fl_completed":       total_fl,
        "patches_generated":  sum(1 for p in patches if (p.get("diff") or "").strip()),
        # FL accuracy (vs ground truth)
        "tasks_evaluated":    agg.get("tasks_evaluated", 0),
        "exact_match":        agg.get("exact_match_rate", 0.0),
        "hit_at_1":           agg.get("hit_at_1_rate", 0.0),
        "hit_at_3":           agg.get("hit_at_3_rate", 0.0),
        "hit_at_5":           agg.get("hit_at_5_rate", 0.0),
        "precision":          agg.get("mean_precision", 0.0),
        "recall":             agg.get("mean_recall", 0.0),
        "f1":                 agg.get("mean_f1", 0.0),
        # Memory injection
        "memory_injected":    mem_injected,
        "injection_rate":     mem_injected / total_fl if total_fl > 0 else 0.0,
        "avg_weighted_sim":   sum(sim_scores) / len(sim_scores) if sim_scores else 0.0,
        "avg_l1_score":       sum(l1_scores)  / len(l1_scores)  if l1_scores  else 0.0,
        "avg_l2_score":       sum(l2_scores)  / len(l2_scores)  if l2_scores  else 0.0,
        "avg_l3_score":       sum(l3_scores)  / len(l3_scores)  if l3_scores  else 0.0,
        # Memory retrieval quality (candidate files vs ground-truth patched files)
        "memory_tasks_evaluated": retrieval_agg["tasks_evaluated"],
        "memory_hit_at_1": retrieval_agg["hit_at_1"],
        "memory_hit_at_3": retrieval_agg["hit_at_3"],
        "memory_hit_at_5": retrieval_agg["hit_at_5"],
        "memory_precision": retrieval_agg["precision"],
        "memory_recall": retrieval_agg["recall"],
        "memory_f1": retrieval_agg["f1"],
    }


def print_table(rows: List[Dict], base: Dict) -> None:
    W = 17
    sep = "-" * (W * 5 + 4)

    def row_line(cells):
        return "  ".join(str(c).ljust(W) for c in cells)

    print("\n" + "=" * 80)
    print("  ABLATION STUDY — FAULT LOCALIZATION RESULTS")
    print("=" * 80)

    # ── Table 1: FL Accuracy ──────────────────────────────────────────────────
    print("\n[ Table 1 ]  Fault Localization Accuracy  (vs ground truth patched files)\n")
    print(row_line(["Condition", "Exact Match", "Hit@1", "Hit@3", "Precision", "Recall", "F1"]))
    print("-" * (W * 7 + 12))
    for r in rows:
        delta_em  = f" ({_delta(r['exact_match'],  base['exact_match'])})"  if r['has_memory'] else ""
        delta_h1  = f" ({_delta(r['hit_at_1'],     base['hit_at_1'])})"     if r['has_memory'] else ""
        delta_h3  = f" ({_delta(r['hit_at_3'],     base['hit_at_3'])})"     if r['has_memory'] else ""
        delta_pr  = f" ({_delta(r['precision'],     base['precision'])})"   if r['has_memory'] else ""
        delta_re  = f" ({_delta(r['recall'],        base['recall'])})"      if r['has_memory'] else ""
        delta_f1  = f" ({_delta(r['f1'],            base['f1'])})"          if r['has_memory'] else ""
        print(row_line([
            r['label'],
            _pct(r['exact_match'])  + delta_em,
            _pct(r['hit_at_1'])     + delta_h1,
            _pct(r['hit_at_3'])     + delta_h3,
            _pct(r['precision'])    + delta_pr,
            _pct(r['recall'])       + delta_re,
            _pct(r['f1'])           + delta_f1,
        ]))
    print(f"\n  Evaluated on {rows[0]['tasks_evaluated']} tasks (issues with generated patches)")

    # ── Table 2: Memory Retrieval Quality ────────────────────────────────────
    mem_rows = [r for r in rows if r['has_memory']]
    if mem_rows:
        print("\n[ Table 2 ]  Memory Retrieval Quality  (candidate files vs ground truth patched files)\n")
        print(row_line(["Condition", "Tasks Eval", "Hit@1", "Hit@3", "Precision", "Recall", "F1"]))
        print("-" * (W * 7 + 12))
        for r in mem_rows:
            print(row_line([
                r['label'],
                str(r['memory_tasks_evaluated']),
                _pct(r['memory_hit_at_1']),
                _pct(r['memory_hit_at_3']),
                _pct(r['memory_precision']),
                _pct(r['memory_recall']),
                _pct(r['memory_f1']),
            ]))

    # ── Table 3: Memory Gating Diagnostics ───────────────────────────────────
    if mem_rows:
        print("\n[ Table 3 ]  Memory Gating Diagnostics\n")
        print(row_line(["Condition", "Injected", "Inject Rate", "Avg Sim", "Avg L1", "Avg L2", "Avg L3"]))
        print("-" * (W * 7 + 12))
        for r in mem_rows:
            print(row_line([
                r['label'],
                f"{r['memory_injected']}/{r['fl_completed']}",
                _pct(r['injection_rate']),
                f"{r['avg_weighted_sim']:.3f}",
                f"{r['avg_l1_score']:.3f}" if r['avg_l1_score'] else "—",
                f"{r['avg_l2_score']:.3f}" if r['avg_l2_score'] else "—",
                f"{r['avg_l3_score']:.3f}" if r['avg_l3_score'] else "—",
            ]))

    # ── Table 4: Coverage ─────────────────────────────────────────────────────
    print("\n[ Table 4 ]  Run Coverage\n")
    print(row_line(["Condition", "FL Done", "Patches", "Issues Total"]))
    print("-" * (W * 4 + 6))
    total = rows[0]["total_issues"]
    for r in rows:
        print(row_line([
            r['label'],
            f"{r['fl_completed']}/{total}",
            f"{r['patches_generated']}/{total}",
            str(total),
        ]))

    print("\n  NOTE: Runs still in progress will show partial numbers.")
    print("        Re-run this script after all runs complete for final results.\n")
    print("=" * 80 + "\n")


def save_csv(rows: List[Dict], out_path: Path) -> None:
    fields = [
        "label", "tasks_evaluated",
        "exact_match", "hit_at_1", "hit_at_3", "hit_at_5",
        "precision", "recall", "f1",
        "memory_tasks_evaluated", "memory_hit_at_1", "memory_hit_at_3", "memory_hit_at_5",
        "memory_precision", "memory_recall", "memory_f1",
        "patches_generated", "memory_injected", "injection_rate",
        "avg_weighted_sim", "avg_l1_score", "avg_l2_score", "avg_l3_score",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved → {out_path}")


def main() -> int:
    rows = [summarize_run(run) for run in RUNS]
    base = next(r for r in rows if not r["has_memory"])

    print_table(rows, base)

    out_dir = RESULTS_DIR
    json_out = out_dir / "ablation_comparison.json"
    csv_out  = out_dir / "ablation_comparison.csv"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  JSON saved → {json_out}")
    save_csv(rows, csv_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
