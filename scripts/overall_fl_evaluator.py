from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


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
    if value is None:
        return ""
    return str(value)


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff_to_ground_truth(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Extract ground-truth changed files and changed line ranges from unified diff.

    Returns:
        {
            "path/to/file.py": [(start, end), ...]
        }

    Uses the target-side hunk coordinates (+new_start, new_count).
    """
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
            new_count = int(hunk_match.group(4)) if hunk_match.group(4) is not None else 1

            if new_count == 0:
                start = new_start
                end = new_start
            else:
                start = new_start
                end = new_start + new_count - 1

            gt[current_file].append((start, end))

    return gt


def build_dataset_records(parquet_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)

    if "diff" not in df.columns:
        raise ValueError("The parquet file must contain a 'diff' column.")

    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for _, row in df.iterrows():
        issue_id = safe_str(row["id"]) if "id" in df.columns else ""
        sha_fail = safe_str(row["sha_fail"]) if "sha_fail" in df.columns else ""
        diff_text = safe_str(row["diff"])

        key = (issue_id, sha_fail)
        if key in seen:
            continue
        seen.add(key)

        gt_files_ranges = parse_diff_to_ground_truth(diff_text)

        records.append({
            "id": issue_id,
            "sha_fail": sha_fail,
            "ground_truth_files_ranges": gt_files_ranges,
            "ground_truth_files": sorted(gt_files_ranges.keys()),
        })

    def sort_key(rec: Dict[str, Any]):
        try:
            return (0, int(rec["id"]))
        except Exception:
            return (1, rec["id"])

    records.sort(key=sort_key)
    return records


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


def get_matching_prediction(
    issue_id: str,
    sha_fail: str,
    pred_index: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if issue_id and f"id::{issue_id}" in pred_index:
        return pred_index[f"id::{issue_id}"]
    if sha_fail and f"sha::{sha_fail}" in pred_index:
        return pred_index[f"sha::{sha_fail}"]
    return None


def extract_predicted_files_and_ranges(
    fl_record: Optional[Dict[str, Any]]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Returns:
        {
            "path/to/file.py": [(start, end), ...]
        }
    """
    if not fl_record:
        return {}

    result: Dict[str, List[Tuple[int, int]]] = {}

    for item in fl_record.get("fault_localization_data", []) or []:
        file_path = normalize_path(item.get("file_path", ""))
        if not file_path:
            continue

        ranges: List[Tuple[int, int]] = []
        for fault in item.get("faults", []) or []:
            lr = fault.get("line_range")
            if isinstance(lr, list) and len(lr) == 2:
                try:
                    start = int(lr[0])
                    end = int(lr[1])
                    if start > end:
                        start, end = end, start
                    ranges.append((start, end))
                except Exception:
                    pass

        result[file_path] = ranges

    return result


def ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def has_location_match(
    pred_ranges: List[Tuple[int, int]],
    gt_ranges: List[Tuple[int, int]]
) -> bool:
    for pr in pred_ranges:
        for gr in gt_ranges:
            if ranges_overlap(pr, gr):
                return True
    return False


def evaluate_one_issue(
    gt_files_ranges: Dict[str, List[Tuple[int, int]]],
    pred_files_ranges: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, Any]:
    """
    File Match:
        proportional file coverage = matched GT files / total GT files

    FL Accuracy:
        proportional file+location coverage = correctly localized GT files / total GT files
    """
    gt_files = set(gt_files_ranges.keys())
    pred_files = set(pred_files_ranges.keys())

    matched_files = sorted(gt_files & pred_files)

    # Proportional file match
    file_match = (
        len(matched_files) / len(gt_files)
        if len(gt_files) > 0 else 0.0
    )

    correctly_localized_files: List[str] = []

    for gt_file in gt_files:
        if gt_file not in pred_files_ranges:
            continue

        pred_ranges = pred_files_ranges.get(gt_file, [])
        gt_ranges = gt_files_ranges.get(gt_file, [])

        if has_location_match(pred_ranges, gt_ranges):
            correctly_localized_files.append(gt_file)

    fl_accuracy = (
        len(correctly_localized_files) / len(gt_files)
        if len(gt_files) > 0 else 0.0
    )

    per_file_details = []
    for gt_file in sorted(gt_files):
        per_file_details.append({
            "file_path": gt_file,
            "ground_truth_ranges": gt_files_ranges.get(gt_file, []),
            "predicted_ranges": pred_files_ranges.get(gt_file, []),
            "file_matched": gt_file in pred_files,
            "location_matched": gt_file in correctly_localized_files,
        })

    return {
        "ground_truth_files": sorted(gt_files),
        "predicted_files": sorted(pred_files),
        "matched_files": matched_files,
        "correctly_localized_files": sorted(correctly_localized_files),
        "file_match": round(file_match, 4),
        "fl_accuracy": round(fl_accuracy, 4),
        "num_ground_truth_files": len(gt_files),
        "num_predicted_files": len(pred_files),
        "num_matched_files": len(matched_files),
        "num_correctly_localized_files": len(correctly_localized_files),
        "per_file_details": per_file_details,
    }


def evaluate_all_issues(
    fault_localization_path: Path,
    parquet_path: Path,
    output_json: Path,
    output_csv: Path,
) -> Dict[str, Any]:
    fl_records = load_json(fault_localization_path)
    if not isinstance(fl_records, list):
        raise ValueError("fault_localization.json must be a list of issue records.")

    pred_index = build_prediction_index(fl_records)
    dataset_records = build_dataset_records(parquet_path)

    per_issue_results: List[Dict[str, Any]] = []
    missing_predictions: List[Dict[str, str]] = []

    total_issues = 0
    total_file_match = 0.0
    total_fl_accuracy = 0.0

    for gt_record in dataset_records:
        total_issues += 1

        issue_id = safe_str(gt_record.get("id"))
        sha_fail = safe_str(gt_record.get("sha_fail"))
        gt_files_ranges = gt_record["ground_truth_files_ranges"]

        pred_record = get_matching_prediction(issue_id, sha_fail, pred_index)

        if pred_record is None:
            missing_predictions.append({"id": issue_id, "sha_fail": sha_fail})
            pred_files_ranges = {}
        else:
            pred_files_ranges = extract_predicted_files_and_ranges(pred_record)

        issue_eval = evaluate_one_issue(gt_files_ranges, pred_files_ranges)

        total_file_match += issue_eval["file_match"]
        total_fl_accuracy += issue_eval["fl_accuracy"]

        per_issue_results.append({
            "id": issue_id,
            "sha_fail": sha_fail,
            **issue_eval,
        })

    denom = total_issues if total_issues > 0 else 1

    aggregate = {
        "issues_evaluated": total_issues,
        "missing_prediction_count": len(missing_predictions),
        "overall_file_match": round((total_file_match / denom) * 100, 2),
        "overall_fl_accuracy": round((total_fl_accuracy / denom) * 100, 2),
    }

    report = {
        "aggregate": aggregate,
        "per_issue": per_issue_results,
        "missing_predictions": missing_predictions,
        "config": {
            "fault_localization_path": str(fault_localization_path),
            "parquet_path": str(parquet_path),
            "file_match_definition": "matched_ground_truth_files / total_ground_truth_files",
            "fl_accuracy_definition": "correctly_localized_ground_truth_files / total_ground_truth_files",
            "missing_predictions_counted_as_zero": True,
        },
    }

    save_json(output_json, report)

    csv_rows = []
    for item in per_issue_results:
        csv_rows.append({
            "id": item["id"],
            "sha_fail": item["sha_fail"],
            "file_match": item["file_match"],
            "fl_accuracy": item["fl_accuracy"],
            "num_ground_truth_files": item["num_ground_truth_files"],
            "num_predicted_files": item["num_predicted_files"],
            "num_matched_files": item["num_matched_files"],
            "num_correctly_localized_files": item["num_correctly_localized_files"],
            "ground_truth_files": " | ".join(item["ground_truth_files"]),
            "predicted_files": " | ".join(item["predicted_files"]),
            "matched_files": " | ".join(item["matched_files"]),
            "correctly_localized_files": " | ".join(item["correctly_localized_files"]),
        })

    pd.DataFrame(csv_rows).to_csv(output_csv, index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate per-issue and overall File Match and FL Accuracy using ground-truth diff."
    )
    parser.add_argument(
        "--fault-localization",
        required=True,
        help="Path to fault_localization.json",
    )
    parser.add_argument(
        "--parquet",
        required=True,
        help="Path to lca_dataset.parquet",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV report",
    )

    args = parser.parse_args()

    report = evaluate_all_issues(
        fault_localization_path=Path(args.fault_localization),
        parquet_path=Path(args.parquet),
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
    )

    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
    
    
#     python /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/scripts/overall_fl_evaluator.py\
#   --fault-localization /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/deepseek-coder_llm/fault_localization.json \
#   --parquet /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet \
#   --output-json /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/deepseek-coder_llm/fl_per_issue_report.json \
#   --output-csv /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/results/deepseek-coder_llm/fl_per_issue_report.csv