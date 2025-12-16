#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import ast
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "dataset").exists() and (p / "results").exists():
            return p
    return Path(__file__).resolve().parents[1]


def parse_maybe_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def extract_labels(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    x = parse_maybe_list(x)

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        labels = []
        for item in x:
            labels.extend(extract_labels(item))
        return labels

    return [str(x)]


def _load_success_ids(success_path: Path) -> set[str]:
    success_ids: set[str] = set()
    with open(success_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" in obj:
                success_ids.add(str(obj["id"]))
    return success_ids


def _load_stream_conclusions(stream_results_path: Path | None) -> dict[str, str]:
    if not stream_results_path:
        return {}
    stream_results_path = Path(stream_results_path)
    if not stream_results_path.exists():
        return {}

    id_to_conc: dict[str, str] = {}
    with open(stream_results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get("id")
            conc = obj.get("conclusion")
            if rid is None:
                continue
            id_to_conc[str(rid)] = str(conc or "").lower()
    return id_to_conc


def _ids_from_list(rows: list | None) -> set[str]:
    out = set()
    if not rows:
        return out
    for r in rows:
        if isinstance(r, dict) and "id" in r:
            out.add(str(r["id"]))
    return out


def compute_overall_outcomes(
    *,
    dataset_ids: set[str],
    success_ids: set[str],
    jobs_results: list | None = None,
    jobs_ids_invalid: list | None = None,
    jobs_ids_await: list | None = None,
    stream_results_path: str | Path | None = None,
) -> dict:
    """
    NEW behavior:
    - Only count pass/fail/invalid/error/waiting over "attempted/pushed" ids.
    - Anything in dataset but never attempted is "not_pushed" and NOT counted as failed.
    """
    total_dataset = len(dataset_ids)

    results_ids = _ids_from_list(jobs_results)
    invalid_ids = _ids_from_list(jobs_ids_invalid)
    waiting_ids = _ids_from_list(jobs_ids_await)

    id_to_conc = _load_stream_conclusions(Path(stream_results_path) if stream_results_path else None)
    stream_ids = set(id_to_conc.keys())

    attempted_ids = (results_ids | invalid_ids | waiting_ids | stream_ids) & dataset_ids
    not_pushed_ids = dataset_ids - attempted_ids

    # Passed strictly from success file, but only among attempted
    passed_ids = (success_ids & attempted_ids)

    # Error IDs: stream says error, plus invalid list (your invalid list is essentially error-like)
    stream_error_ids = {rid for rid, c in id_to_conc.items() if c == "error"}
    error_ids = ((stream_error_ids | invalid_ids) & attempted_ids)

    # Restrict invalid/waiting to attempted
    invalid_ids &= attempted_ids
    waiting_ids &= attempted_ids

    # Precedence within attempted
    final_status: dict[str, str] = {}
    for rid in attempted_ids:
        if rid in waiting_ids:
            final_status[rid] = "waiting"
        elif rid in invalid_ids:
            final_status[rid] = "invalid"
        elif rid in error_ids:
            final_status[rid] = "error"
        elif rid in passed_ids:
            final_status[rid] = "passed"
        else:
            final_status[rid] = "failed"

    attempted = len(attempted_ids)
    passed = sum(1 for s in final_status.values() if s == "passed")
    failed = sum(1 for s in final_status.values() if s == "failed")
    invalid = sum(1 for s in final_status.values() if s == "invalid")
    error = sum(1 for s in final_status.values() if s == "error")
    waiting = sum(1 for s in final_status.values() if s == "waiting")
    not_pushed = len(not_pushed_ids)

    accuracy_attempted = round((passed / attempted) * 100, 2) if attempted else 0.0
    coverage_percent = round((attempted / total_dataset) * 100, 2) if total_dataset else 0.0

    return {
        "total_dataset": total_dataset,
        "attempted": attempted,
        "not_pushed": not_pushed,
        "passed": passed,
        "failed": failed,
        "invalid": invalid,
        "error": error,
        "waiting": waiting,
        "accuracy_percent_attempted": accuracy_attempted,
        "coverage_percent": coverage_percent,
    }


def run_error_type_accuracy_evaluation(
    dataset_path: str | Path | None = None,
    success_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    jobs_results: list | None = None,
    jobs_ids_invalid: list | None = None,
    jobs_ids_await: list | None = None,
    stream_results_path: str | Path | None = None,
) -> dict:
    repo_root = _find_repo_root(Path(__file__).resolve())

    dataset_path = Path(dataset_path) if dataset_path else (repo_root / "dataset" / "lca_dataset.parquet")
    success_path = Path(success_path) if success_path else (repo_root / "results" / "jobs_success_diff.jsonl")
    output_dir = Path(output_dir) if output_dir else (repo_root / "evaluation_plot")
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_plot_path = output_dir / "error_type_accuracy_lollipop.png"
    accuracy_table_path = output_dir / "error_type_accuracy_table.png"

    df = pd.read_parquet(dataset_path)
    if "id" not in df.columns or "error_type" not in df.columns:
        raise KeyError(f"Dataset must include columns ['id','error_type']. Found: {list(df.columns)}")

    df["id"] = df["id"].astype(str)
    df["error_labels"] = df["error_type"].apply(extract_labels)

    dataset_ids = set(df["id"].tolist())
    success_ids = _load_success_ids(success_path)

    overall = compute_overall_outcomes(
        dataset_ids=dataset_ids,
        success_ids=success_ids,
        jobs_results=jobs_results,
        jobs_ids_invalid=jobs_ids_invalid,
        jobs_ids_await=jobs_ids_await,
        stream_results_path=stream_results_path,
    )

    print("\n=== Overall outcome stats (attempted-only; dataset-grounded) ===")
    print(f"Repo root: {repo_root}")
    print(f"Dataset rows: {overall['total_dataset']}")
    print(f"Attempted/pushed: {overall['attempted']}  (coverage: {overall['coverage_percent']}%)")
    print(f"Not pushed: {overall['not_pushed']}  <-- NOT counted as failed")
    print(f"Passed:  {overall['passed']}")
    print(f"Failed:  {overall['failed']}")
    print(f"Invalid: {overall['invalid']}")
    print(f"Error:   {overall['error']}")
    print(f"Waiting: {overall['waiting']}")
    print(f"Accuracy (passed/attempted*100): {overall['accuracy_percent_attempted']}%")

    # -------- Per-error-type accuracy (still uses dataset + success_ids) ----------
    # This stays as before, because you want to know "what can be solved" by error type
    # based on dataset labels and success list.

    total_counter = defaultdict(int)
    success_counter = defaultdict(int)

    for row in df.itertuples(index=False):
        rid = str(getattr(row, "id"))
        labels = getattr(row, "error_labels")
        for label in labels:
            total_counter[label] += 1
            if rid in success_ids:
                success_counter[label] += 1

    total_error_labels = sum(total_counter.values())
    unique_error_types = len(total_counter)

    rows = []
    for err_type in sorted(total_counter.keys()):
        total = total_counter[err_type]
        solved = success_counter[err_type]
        accuracy = round((solved / total) * 100, 2) if total > 0 else 0.0
        share = round((total / total_error_labels) * 100, 2) if total_error_labels > 0 else 0.0
        rows.append([err_type, total, solved, accuracy, share])

    acc_df = pd.DataFrame(
        rows,
        columns=[
            "error_type",
            "total_cases",
            "solved_cases",
            "accuracy_percent",
            "share_of_all_error_labels_percent",
        ],
    )

    print("\n=== Per-Error-Type Accuracy ===\n")
    print(acc_df.to_string(index=False))

    sorted_df = acc_df.sort_values("accuracy_percent")
    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_df) * 0.45)))
    y_pos = np.arange(len(sorted_df))

    ax.hlines(y=y_pos, xmin=0, xmax=sorted_df["accuracy_percent"])
    ax.scatter(sorted_df["accuracy_percent"], y_pos, s=60)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_df["error_type"])
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Error-Type Repair Accuracy")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for x_val, y_idx, solved, total, acc in zip(
        sorted_df["accuracy_percent"],
        y_pos,
        sorted_df["solved_cases"],
        sorted_df["total_cases"],
        sorted_df["accuracy_percent"],
    ):
        ax.text(x_val, y_idx, f"  {solved}/{total}  ({acc}%)", va="center", ha="left", fontsize=8)

    fig.text(
        0.01,
        0.98,
        f"Dataset rows: {overall['total_dataset']}\n"
        f"Attempted: {overall['attempted']}  Not pushed: {overall['not_pushed']}\n"
        f"Passed: {overall['passed']}  Failed: {overall['failed']}  Waiting: {overall['waiting']}\n"
        f"Accuracy(passed/attempted): {overall['accuracy_percent_attempted']}%\n"
        f"Total error labels: {total_error_labels}\n"
        f"Unique error types: {unique_error_types}",
        ha="left",
        va="top",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(accuracy_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved accuracy lollipop plot → {accuracy_plot_path}")

    fig, ax = plt.subplots(figsize=(10, max(3, len(acc_df) * 0.35)))
    ax.axis("off")
    table = ax.table(
        cellText=acc_df.values,
        colLabels=acc_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.scale(1, 1.3)
    plt.title("Error Type Repair Accuracy (Counts & Shares)")
    plt.savefig(accuracy_table_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy table image → {accuracy_table_path}")

    return {
        "repo_root": str(repo_root),
        "dataset_path": str(dataset_path),
        "success_path": str(success_path),
        "output_dir": str(output_dir),
        "overall": overall,
        "total_error_labels": total_error_labels,
        "unique_error_types": unique_error_types,
        "accuracy_plot_path": str(accuracy_plot_path),
        "accuracy_table_path": str(accuracy_table_path),
    }


if __name__ == "__main__":
    run_error_type_accuracy_evaluation()
