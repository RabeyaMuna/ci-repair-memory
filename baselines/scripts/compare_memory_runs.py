#!/usr/bin/env python3

import json
import os
import sys
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def count_non_empty_patches(path: str) -> int:
    records = load_json(path)
    return sum(1 for item in records if (item.get("diff") or "").strip())


def summarize(result_dir: str) -> Dict[str, Any]:
    fl_eval_path = os.path.join(result_dir, "fl_evaluation.json")
    patch_path = os.path.join(result_dir, "generated_patches.json")
    retrieval_log = os.path.join(result_dir, "memory_retrieval_log.jsonl")

    fl_eval = load_json(fl_eval_path)
    agg = fl_eval.get("aggregate", {})
    retrieval_count = 0
    if os.path.exists(retrieval_log):
        with open(retrieval_log, "r", encoding="utf-8") as handle:
            retrieval_count = sum(1 for line in handle if line.strip())

    return {
        "result_dir": result_dir,
        "tasks_evaluated": agg.get("tasks_evaluated", 0),
        "exact_match_rate": agg.get("exact_match_rate", 0.0),
        "hit_at_1_rate": agg.get("hit_at_1_rate", 0.0),
        "hit_at_3_rate": agg.get("hit_at_3_rate", 0.0),
        "mean_precision": agg.get("mean_precision", 0.0),
        "mean_recall": agg.get("mean_recall", 0.0),
        "mean_f1": agg.get("mean_f1", 0.0),
        "generated_patches": count_non_empty_patches(patch_path),
        "memory_queries_logged": retrieval_count,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python compare_memory_runs.py <baseline_result_dir> <memory_result_dir>")
        return 1

    baseline = summarize(sys.argv[1])
    memory = summarize(sys.argv[2])

    diff = {
        key: round(memory[key] - baseline[key], 4)
        for key in (
            "exact_match_rate",
            "hit_at_1_rate",
            "hit_at_3_rate",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "generated_patches",
        )
    }

    print(json.dumps({"baseline": baseline, "memory": memory, "delta": diff}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
