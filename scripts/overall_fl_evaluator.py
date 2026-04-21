from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------
# IO HELPERS
# -----------------------------
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("/")


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


# -----------------------------
# DIFF PARSING
# -----------------------------
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff_to_ground_truth(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    gt: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None

    for line in (diff_text or "").splitlines():
        file_match = _DIFF_HEADER_RE.match(line)
        if file_match:
            current_file = normalize_path(file_match.group(2))
            gt.setdefault(current_file, [])
            continue

        hunk_match = _HUNK_RE.match(line)
        if hunk_match and current_file:
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1

            start = new_start
            end = new_start if new_count == 0 else new_start + new_count - 1

            gt[current_file].append((start, end))

    return gt


# -----------------------------
# DATASET BUILDING
# -----------------------------
def build_dataset_records(parquet_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)

    if "diff" not in df.columns:
        raise ValueError("Parquet must contain 'diff' column.")

    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for _, row in df.iterrows():
        issue_id = safe_str(row.get("id"))
        sha_fail = safe_str(row.get("sha_fail"))
        diff_text = safe_str(row.get("diff"))

        key = (issue_id, sha_fail)
        if key in seen:
            continue
        seen.add(key)

        gt = parse_diff_to_ground_truth(diff_text)

        records.append(
            {
                "id": issue_id,
                "sha_fail": sha_fail,
                "ground_truth_files_ranges": gt,
            }
        )

    records.sort(key=lambda x: (x["id"]))
    return records


# -----------------------------
# PREDICTIONS
# -----------------------------
def build_prediction_index(fl_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}

    for rec in fl_records:
        issue_id = safe_str(rec.get("id"))
        sha_fail = safe_str(rec.get("sha_fail"))

        if issue_id:
            index[f"id::{issue_id}"] = rec
        if sha_fail:
            index[f"sha::{sha_fail}"] = rec

    return index


def get_matching_prediction(issue_id: str, sha_fail: str, index: Dict) -> Optional[Dict]:
    if issue_id and f"id::{issue_id}" in index:
        return index[f"id::{issue_id}"]
    if sha_fail and f"sha::{sha_fail}" in index:
        return index[f"sha::{sha_fail}"]
    return None


def extract_predicted_files_and_ranges(fl_record: Optional[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    if not fl_record:
        return {}

    result: Dict[str, List[Tuple[int, int]]] = {}

    for item in fl_record.get("fault_localization_data", []) or []:
        file_path = normalize_path(item.get("file_path", ""))
        if not file_path:
            continue

        ranges = []
        for fault in item.get("faults", []) or []:
            lr = fault.get("line_range")
            if isinstance(lr, list) and len(lr) == 2:
                try:
                    start, end = int(lr[0]), int(lr[1])
                    if start > end:
                        start, end = end, start
                    ranges.append((start, end))
                except Exception:
                    pass

        result[file_path] = ranges

    return result


# -----------------------------
# METRICS
# -----------------------------
def ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def has_location_match(pred_ranges, gt_ranges) -> bool:
    return any(ranges_overlap(p, g) for p in pred_ranges for g in gt_ranges)


def f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def evaluate_one_issue(gt: Dict, pred: Dict) -> Dict[str, Any]:
    gt_files = set(gt.keys())
    pred_files = set(pred.keys())

    matched = gt_files & pred_files
    false_pos = pred_files - gt_files

    # File metrics
    file_recall = len(matched) / len(gt_files) if gt_files else 0
    file_precision = len(matched) / len(pred_files) if pred_files else 0

    # Location metrics
    correct_loc = []
    for f in matched:
        if has_location_match(pred[f], gt[f]):
            correct_loc.append(f)

    fl_recall = len(correct_loc) / len(gt_files) if gt_files else 0
    fl_precision = len(correct_loc) / len(pred_files) if pred_files else 0

    return {
        "file_recall": round(file_recall, 4),
        "file_precision": round(file_precision, 4),
        "file_f1": round(f1(file_precision, file_recall), 4),

        "fl_recall": round(fl_recall, 4),
        "fl_precision": round(fl_precision, 4),
        "fl_f1": round(f1(fl_precision, fl_recall), 4),

        "num_gt_files": len(gt_files),
        "num_pred_files": len(pred_files),
        "num_matched": len(matched),
        "num_false_positive": len(false_pos),
        "num_correct_localized": len(correct_loc),
    }


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def evaluate_all_issues(fl_path: Path, parquet_path: Path, out_json: Path, out_csv: Path):
    fl_records = load_json(fl_path)
    pred_index = build_prediction_index(fl_records)
    dataset = build_dataset_records(parquet_path)

    results = []

    totals = {
        "file_recall": 0,
        "file_precision": 0,
        "file_f1": 0,
        "fl_recall": 0,
        "fl_precision": 0,
        "fl_f1": 0,
    }

    for rec in dataset:
        issue_id = rec["id"]
        sha = rec["sha_fail"]

        gt = rec["ground_truth_files_ranges"]
        pred_rec = get_matching_prediction(issue_id, sha, pred_index)
        pred = extract_predicted_files_and_ranges(pred_rec)

        eval_res = evaluate_one_issue(gt, pred)

        for k in totals:
            totals[k] += eval_res[k]

        results.append({
            "id": issue_id,
            "sha_fail": sha,
            **eval_res
        })

    n = len(results) or 1

    aggregate = {k: round((v / n) * 100, 2) for k, v in totals.items()}

    report = {
        "aggregate": aggregate,
        "per_issue": results,
    }

    save_json(out_json, report)
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print(json.dumps(aggregate, indent=2))


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fault-localization", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)

    args = parser.parse_args()

    evaluate_all_issues(
        Path(args.fault_localization),
        Path(args.parquet),
        Path(args.output_json),
        Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
    
    
    python /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/scripts/overall_fl_evaluator.py\
  --fault-localization /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/gpt-5-mini_llm/fault_localization.json \
  --parquet /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet \
  --output-json /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/gpt-5-mini_llm/fl_per_issue_report.json \
  --output-csv /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/gpt-5-mini_llm/fl_per_issue_report.csv