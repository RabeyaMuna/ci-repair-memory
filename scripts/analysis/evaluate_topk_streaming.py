#!/usr/bin/env python3
"""Evaluate file localization from a large, pretty-printed preds.json.

The prediction file is memory-mapped, so even very large diff strings are never
decoded into Python objects. The order of unique ``diff --git`` file headers is
used as the prediction order, as in evaluate_file_localization.py.
"""

import argparse
import json
import mmap
import re
from pathlib import Path

import pyarrow.parquet as pq


ENTRY = re.compile(rb'(?m)^  "([^"\r\n]+)": \{\r?$')
DIFF_HEADER = b"diff --git a/"
K_VALUES = (1, 5, 10, 15)


def entry_matches(data: mmap.mmap):
    """Jump between top-level lines instead of regex-scanning multi-GB diffs."""
    pos = 0
    while True:
        marker = data.find(b"\n  \"", pos)
        if marker < 0:
            return
        pos = marker + 4
        match = ENTRY.match(data, marker + 1)
        if match:
            yield match


def predicted_files(data: mmap.mmap, start: int, end: int) -> list[str]:
    """Read the first 15 unique file paths in diff-header order."""
    files = []
    seen = set()
    pos = start
    while len(files) < max(K_VALUES):
        header = data.find(DIFF_HEADER, pos, end)
        if header < 0:
            break
        pos = header + len(DIFF_HEADER)
        # A header must start a decoded diff line (or the diff string itself).
        if not (data[max(start, header - 2):header] == b"\\n" or
                data[max(start, header - 9):header] == b'"diff": "'):
            continue
        line_end = data.find(b"\\n", pos, min(end, pos + 16384))
        if line_end < 0:
            continue
        second_path = data.find(b" b/", pos, line_end)
        if second_path < 0:
            continue
        raw_path = data[second_path + 3:line_end]
        try:
            path = json.loads(b'"' + raw_path + b'"')
        except (ValueError, UnicodeDecodeError):
            continue
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, default=Path("results/preds.json"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/lca_dataset.parquet"))
    parser.add_argument("--eval-ids", type=Path, default=Path("results/eval_issue_ids.json"),
                        help="Issue IDs to evaluate; omit with --all-predictions")
    parser.add_argument("--all-predictions", action="store_true",
                        help="Evaluate only IDs present in preds.json")
    parser.add_argument("--output", type=Path,
                        default=Path("results/topk_streaming_metrics.json"))
    args = parser.parse_args()

    table = pq.read_table(args.dataset, columns=["id", "changed_files"])
    ground_truth = {
        str(issue_id): set(files)
        for issue_id, files in zip(table.column("id").to_pylist(),
                                   table.column("changed_files").to_pylist())
        if files
    }
    eval_ids = None if args.all_predictions else {
        str(issue_id) for issue_id in json.loads(args.eval_ids.read_text())
    }
    hits = {k: 0 for k in K_VALUES}
    predictions_found = set()
    missing_ground_truth = set()
    predicted_count = 0
    with args.preds.open("rb") as pred_file, mmap.mmap(
        pred_file.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        previous = None
        for match in entry_matches(data):
            if previous is not None:
                issue_id = previous.group(1).decode("utf-8")
                if eval_ids is None or issue_id in eval_ids:
                    predicted_count += 1
                    if issue_id in ground_truth:
                        predictions_found.add(issue_id)
                        predicted = predicted_files(data, previous.end(), match.start())
                        truth = ground_truth[issue_id]
                        for k in K_VALUES:
                            if truth.intersection(predicted[:k]):
                                hits[k] += 1
                    else:
                        missing_ground_truth.add(issue_id)
            previous = match
        if previous is not None:
            issue_id = previous.group(1).decode("utf-8")
            if eval_ids is None or issue_id in eval_ids:
                predicted_count += 1
                if issue_id in ground_truth:
                    predictions_found.add(issue_id)
                    predicted = predicted_files(data, previous.end(), len(data))
                    truth = ground_truth[issue_id]
                    for k in K_VALUES:
                        if truth.intersection(predicted[:k]):
                            hits[k] += 1
                else:
                    missing_ground_truth.add(issue_id)

    if eval_ids is None:
        total = len(predictions_found)
        scope = "predictions with ground truth"
    else:
        total = len(eval_ids & ground_truth.keys())
        scope = "evaluation IDs with ground truth"
    result = {
        "scope": scope,
        "total_issues": total,
        "predictions_in_scope": predicted_count,
        "issues_with_predictions_and_ground_truth": len(predictions_found),
        "issues_without_predictions": total - len(predictions_found),
        "prediction_ids_without_ground_truth": sorted(missing_ground_truth),
        "top_k": {
            f"top_{k}": {"hits": hits[k], "total": total,
                         "accuracy_percent": round(100 * hits[k] / total, 2) if total else 0}
            for k in K_VALUES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
