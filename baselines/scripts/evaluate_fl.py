#!/usr/bin/env python3
"""
evaluate_fl.py  —  Detailed Fault-Localization evaluation for one result directory.

Usage
-----
    # default: deepseek-chat_bm25
    python evaluate_fl.py

    # any other run
    python evaluate_fl.py results/gpt-4o-mini_llm
    python evaluate_fl.py results/deepseek-coder_llm

What it shows
-------------
For every task (sha_fail):
  - FL-predicted files + fault line ranges + issue type
  - Ground-truth files (from generated-patch diff headers + patched line hunks)
  - Per-file verdict: HIT / MISS / EXTRA
  - Whether the predicted line range overlaps the actually-patched lines

Aggregate
  - exact_match_rate, hit@1, hit@3, hit@5
  - mean precision / recall / F1
  - line-overlap accuracy (among file-level hits)

Output
------
  Printed to stdout  (human-readable table)
  Saved as  <result_dir>/fl_detailed_eval.json
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(path: str) -> str:
    """Strip absolute prefixes so paths are repo-relative and comparable."""
    p = path.strip().replace("\\", "/")
    # Remove absolute repo-clone prefix if present
    for marker in ["/repo_cloned/", "/baselines/repo_cloned/"]:
        if marker in p:
            p = p.split(marker, 1)[-1]
            # Remove repo-name prefix (first component)
            parts = p.split("/", 1)
            p = parts[1] if len(parts) > 1 else parts[0]
    return p.lstrip("/")


def extract_patched_files(diff_text: str) -> List[str]:
    """Return file paths from 'diff --git a/... b/...' headers (b/ side)."""
    return [
        m.group(1).strip()
        for m in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text, re.MULTILINE)
        if m.group(1).strip()
    ]


def extract_patched_hunks(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Return a dict:  file_path -> list of (start_line, end_line) tuples
    representing the NEW-file line ranges that were actually changed.

    Parses unified-diff hunk headers:
        @@ -old_start,old_count +new_start,new_count @@
    """
    result: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None

    for line in diff_text.splitlines():
        # file header
        m_file = re.match(r"^diff --git a/.+ b/(.+)$", line)
        if m_file:
            current_file = m_file.group(1).strip()
            result.setdefault(current_file, [])
            continue

        # hunk header  @@ -a,b +c,d @@
        if current_file is not None:
            m_hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if m_hunk:
                new_start = int(m_hunk.group(1))
                new_count = int(m_hunk.group(2)) if m_hunk.group(2) is not None else 1
                new_end   = new_start + max(new_count - 1, 0)
                result[current_file].append((new_start, new_end))

    return result


def ranges_overlap(
    r1: Optional[List[int]], r2_list: List[Tuple[int, int]]
) -> bool:
    """True if FL line range [r1[0], r1[1]] overlaps any patched hunk."""
    if not r1 or len(r1) < 2 or not r2_list:
        return False
    a, b = r1[0], r1[1]
    for (c, d) in r2_list:
        if a <= d and c <= b:      # overlap condition
            return True
    return False


def prf(predicted: List[str], ground_truth: List[str]) -> Dict[str, float]:
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p_set = set(norm(x) for x in predicted)
    g_set = set(norm(x) for x in ground_truth)
    tp = len(p_set & g_set)
    precision = tp / len(p_set)
    recall    = tp / len(g_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4)}


def hit_at_k(predicted: List[str], ground_truth: List[str], k: int) -> bool:
    g_set = set(norm(x) for x in ground_truth)
    return any(norm(p) in g_set for p in predicted[:k])


# ──────────────────────────────────────────────────────────────────────────────
# main evaluation
# ──────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 90
THIN = "-" * 90


def evaluate(result_dir: str) -> None:
    fl_path    = os.path.join(result_dir, "fault_localization.json")
    patch_path = os.path.join(result_dir, "generated_patches.json")

    if not os.path.exists(fl_path):
        print(f"[ERROR] Not found: {fl_path}")
        sys.exit(1)
    if not os.path.exists(patch_path):
        print(f"[ERROR] Not found: {patch_path}")
        sys.exit(1)

    fl_records    = load_json(fl_path)
    patch_records = load_json(patch_path)

    # Index patches by sha_fail
    patch_by_sha: Dict[str, dict] = {}
    for rec in patch_records:
        sha = rec.get("sha_fail", "")
        if sha:
            patch_by_sha[sha] = rec

    per_task_results = []
    skipped          = []

    print(f"\n{SEP}")
    print(f"  FL EVALUATION  —  {result_dir}")
    print(SEP)

    for fl_rec in fl_records:
        sha_fail = fl_rec.get("sha_fail", "")
        task_id  = str(fl_rec.get("id", "?"))
        fl_data  = fl_rec.get("fault_localization_data") or []

        # Predicted files (ordered list) + their fault details
        predicted_files: List[str] = []
        predicted_details: List[dict] = []
        for entry in fl_data:
            raw_fp = entry.get("file_path", "")
            fp     = norm(raw_fp)
            faults = entry.get("faults") or []
            predicted_files.append(fp)
            predicted_details.append({
                "file": fp,
                "faults": [
                    {
                        "line_range":               f.get("line_range"),
                        "issue_type":               f.get("issue_type", ""),
                        "fault_localization_level": f.get("fault_localization_level", ""),
                        "reason":                   f.get("reason", "")[:200],
                    }
                    for f in faults
                ],
            })

        # Ground-truth: patched files + hunk line ranges
        if sha_fail not in patch_by_sha:
            skipped.append(sha_fail)
            continue

        patch_rec  = patch_by_sha[sha_fail]
        diff_text  = patch_rec.get("diff", "")
        gt_files   = extract_patched_files(diff_text)        # repo-relative
        patch_hunks = extract_patched_hunks(diff_text)       # file -> [(start,end)]

        if not gt_files:
            skipped.append(sha_fail)
            continue

        gt_norm = set(norm(g) for g in gt_files)
        pred_norm = [norm(p) for p in predicted_files]

        # Per-file classification
        hits   = [p for p in pred_norm if p in gt_norm]
        extras = [p for p in pred_norm if p not in gt_norm]
        missed = [g for g in gt_files  if norm(g) not in set(pred_norm)]

        # Line-overlap check for each HIT file
        line_overlap_results = []
        for pd in predicted_details:
            fp_n = pd["file"]
            if fp_n not in gt_norm:
                continue
            # find matching patch hunk key (same norm path)
            hunk_key = next(
                (k for k in patch_hunks if norm(k) == fp_n), None
            )
            hunks = patch_hunks.get(hunk_key, []) if hunk_key else []
            for fault in pd["faults"]:
                lr = fault.get("line_range")
                overlaps = ranges_overlap(lr, hunks)
                line_overlap_results.append({
                    "file":            fp_n,
                    "predicted_range": lr,
                    "patched_hunks":   hunks,
                    "line_overlap":    overlaps,
                })

        metrics = prf(predicted_files, gt_files)

        task_result = {
            "task_id":            task_id,
            "sha_fail":           sha_fail,
            "predicted_files":    pred_norm,
            "ground_truth_files": [norm(g) for g in gt_files],
            "hits":               hits,
            "extra_predictions":  extras,
            "missed_files":       [norm(g) for g in missed],
            "exact_match":        set(pred_norm) == gt_norm,
            "hit_at_1":           hit_at_k(pred_norm, gt_files, 1),
            "hit_at_3":           hit_at_k(pred_norm, gt_files, 3),
            "hit_at_5":           hit_at_k(pred_norm, gt_files, 5),
            "num_predicted":      len(pred_norm),
            "num_ground_truth":   len(gt_files),
            "line_overlap_detail":line_overlap_results,
            **metrics,
        }
        per_task_results.append(task_result)

        # ── Print per-task block ──────────────────────────────────────────
        print(f"\n  Task {task_id}  |  sha: {sha_fail}")
        print(THIN)

        print(f"  FL PREDICTED  ({len(pred_norm)} file(s))")
        for pd in predicted_details:
            verdict = "HIT  ✓" if pd["file"] in gt_norm else "EXTRA ✗"
            print(f"    [{verdict}]  {pd['file']}")
            for fault in pd["faults"]:
                lr   = fault["line_range"]
                lr_s = f"lines {lr[0]}-{lr[1]}" if lr else "lines ?"
                print(f"             issue_type : {fault['issue_type']}")
                print(f"             level      : {fault['fault_localization_level']}")
                print(f"             location   : {lr_s}")
                print(f"             reason     : {fault['reason'][:120]}...")

        print(f"\n  GROUND TRUTH  ({len(gt_files)} file(s) actually patched)")
        for gf in gt_files:
            gf_n  = norm(gf)
            hunks = patch_hunks.get(gf, patch_hunks.get(gf_n, []))
            hunks_s = ", ".join(f"lines {s}-{e}" for s, e in hunks) if hunks else "?"
            found = "✓" if gf_n in set(pred_norm) else "✗ missed"
            print(f"    [{found}]  {gf_n}   patched at: {hunks_s}")

        if missed:
            print(f"\n  MISSED FILES (in patch but not predicted by FL):")
            for mf in missed:
                print(f"    ✗  {norm(mf)}")

        if line_overlap_results:
            print(f"\n  LINE OVERLAP (among hit files):")
            for lo in line_overlap_results:
                overlap_s = "YES ✓" if lo["line_overlap"] else "NO  ✗"
                predicted_s = (
                    f"lines {lo['predicted_range'][0]}-{lo['predicted_range'][1]}"
                    if lo["predicted_range"] else "unknown"
                )
                patched_s = (
                    ", ".join(f"lines {s}-{e}" for s, e in lo["patched_hunks"])
                    if lo["patched_hunks"] else "unknown"
                )
                print(f"    {lo['file']}")
                print(f"      predicted : {predicted_s}")
                print(f"      patched   : {patched_s}")
                print(f"      overlap   : {overlap_s}")

        print(f"\n  METRICS  precision={metrics['precision']}  "
              f"recall={metrics['recall']}  f1={metrics['f1']}  "
              f"hit@1={task_result['hit_at_1']}  "
              f"exact_match={task_result['exact_match']}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(per_task_results)
    if n == 0:
        print("\n[WARN] No tasks could be evaluated.")
        return

    exact_match_n        = sum(t["exact_match"]  for t in per_task_results)
    hit1_n               = sum(t["hit_at_1"]     for t in per_task_results)
    hit3_n               = sum(t["hit_at_3"]     for t in per_task_results)
    hit5_n               = sum(t["hit_at_5"]     for t in per_task_results)
    mean_precision       = sum(t["precision"]    for t in per_task_results) / n
    mean_recall          = sum(t["recall"]       for t in per_task_results) / n
    mean_f1              = sum(t["f1"]           for t in per_task_results) / n

    # Line-overlap accuracy (across all hit files)
    all_lo = [lo for t in per_task_results for lo in t["line_overlap_detail"]]
    lo_acc = (
        round(sum(lo["line_overlap"] for lo in all_lo) / len(all_lo), 4)
        if all_lo else None
    )

    aggregate = {
        "tasks_evaluated":        n,
        "tasks_skipped_no_patch": len(skipped),
        "exact_match_rate":  round(exact_match_n / n, 4),
        "hit_at_1_rate":     round(hit1_n        / n, 4),
        "hit_at_3_rate":     round(hit3_n        / n, 4),
        "hit_at_5_rate":     round(hit5_n        / n, 4),
        "mean_precision":    round(mean_precision,    4),
        "mean_recall":       round(mean_recall,       4),
        "mean_f1":           round(mean_f1,           4),
        "line_overlap_accuracy": lo_acc,
    }

    print(f"\n{SEP}")
    print("  AGGREGATE RESULTS")
    print(SEP)
    print(f"  Tasks evaluated             : {n}")
    print(f"  Tasks skipped (no patch)    : {len(skipped)}")
    print(THIN)
    print(f"  Exact match rate            : {aggregate['exact_match_rate']}")
    print(f"  Hit @ 1                     : {aggregate['hit_at_1_rate']}")
    print(f"  Hit @ 3                     : {aggregate['hit_at_3_rate']}")
    print(f"  Hit @ 5                     : {aggregate['hit_at_5_rate']}")
    print(THIN)
    print(f"  Mean precision              : {aggregate['mean_precision']}")
    print(f"  Mean recall                 : {aggregate['mean_recall']}")
    print(f"  Mean F1                     : {aggregate['mean_f1']}")
    if lo_acc is not None:
        print(f"  Line-overlap accuracy       : {lo_acc}")
    print(SEP)

    # ── Save ──────────────────────────────────────────────────────────────────
    report = {
        "aggregate":         aggregate,
        "per_task":          per_task_results,
        "skipped_sha_fails": skipped,
    }
    out_path = os.path.join(result_dir, "fl_detailed_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved → {out_path}\n")


# ──────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # scripts/ lives inside baselines/ — go one level up to reach results/
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        result_dir = arg if os.path.isabs(arg) else os.path.join(base, arg)
    else:
        result_dir = os.path.join(base, "results", "deepseek-chat_bm25")

    if not os.path.isdir(result_dir):
        print(f"[ERROR] Directory not found: {result_dir}")
        sys.exit(1)

    evaluate(result_dir)
