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


def _load_jsonl_rows(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dict rows. Safe against bad lines."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def _ids_from_list(rows: list | None) -> set[str]:
    """
    Extract ids from common formats.
    Supports keys: id, job_id, run_id (extend if needed).
    """
    out: set[str] = set()
    if not rows:
        return out

    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in ("id", "job_id", "run_id"):
            if k in r and r[k] is not None:
                out.add(str(r[k]))
                break
    return out


def _load_success_ids(success_path: Path) -> set[str]:
    success_ids: set[str] = set()
    if not Path(success_path).exists():
        return success_ids
    with open(success_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and "id" in obj and obj["id"] is not None:
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
            if not isinstance(obj, dict):
                continue
            rid = obj.get("id")
            conc = obj.get("conclusion")
            if rid is None:
                continue
            id_to_conc[str(rid)] = str(conc or "").lower()
    return id_to_conc


def compute_overall_outcomes(
    *,
    dataset_ids: set[str],
    success_ids: set[str],
    jobs_results: list | None = None,
    jobs_ids_invalid: list | None = None,
    jobs_ids_await: list | None = None,
    jobs_ids_failure: list | None = None,
    jobs_ids_error: list | None = None,
    stream_results_path: str | Path | None = None,
) -> dict:
    """
    Fixed behavior:
    - Auto counts "attempted" using any evidence of an attempt:
      results/invalid/waiting/stream OR success/failure/error files.
    - "not_pushed" = dataset - attempted.
    - passed from success_ids, failed from failure_ids (plus fallback).
    """
    total_dataset = len(dataset_ids)

    results_ids = _ids_from_list(jobs_results)
    invalid_ids = _ids_from_list(jobs_ids_invalid)
    waiting_ids = _ids_from_list(jobs_ids_await)
    failure_ids = _ids_from_list(jobs_ids_failure)
    error_file_ids = _ids_from_list(jobs_ids_error)

    id_to_conc = _load_stream_conclusions(Path(stream_results_path) if stream_results_path else None)
    stream_ids = set(id_to_conc.keys())

    # IMPORTANT: success/failure/error also imply an attempt
    attempted_ids = (
        results_ids
        | invalid_ids
        | waiting_ids
        | stream_ids
        | set(success_ids)
        | failure_ids
        | error_file_ids
    ) & dataset_ids

    not_pushed_ids = dataset_ids - attempted_ids

    passed_ids = success_ids & attempted_ids

    stream_error_ids = {rid for rid, c in id_to_conc.items() if c == "error"}
    error_ids = (stream_error_ids | invalid_ids | error_file_ids) & attempted_ids

    invalid_ids &= attempted_ids
    waiting_ids &= attempted_ids
    failure_ids &= attempted_ids

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
        elif rid in failure_ids:
            final_status[rid] = "failed"
        else:
            # attempted but not categorized -> treat as failed
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

    results_dir = repo_root / "results"

    # Auto-load fallbacks if caller didn't pass lists
    if jobs_results is None:
        jobs_results = _load_jsonl_rows(results_dir / "jobs_results_diff.jsonl")
    if jobs_ids_invalid is None:
        jobs_ids_invalid = _load_jsonl_rows(results_dir / "jobs_invalid_diff.jsonl")
    if jobs_ids_await is None:
        jobs_ids_await = _load_jsonl_rows(results_dir / "jobs_awaiting_diff.jsonl")

    # Always load these as evidence of attempt
    jobs_ids_failure = _load_jsonl_rows(results_dir / "jobs_failure_diff.jsonl")
    jobs_ids_error = _load_jsonl_rows(results_dir / "jobs_error_diff.jsonl")

    accuracy_plot_path = output_dir / "error_type_accuracy_lollipop.png"
    accuracy_table_path = output_dir / "error_type_accuracy_table.png"
    overall_path = output_dir / "overall.json"

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
        jobs_ids_failure=jobs_ids_failure,
        jobs_ids_error=jobs_ids_error,
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

    # Write overall JSON
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSaved overall JSON → {overall_path}")

    # -------- Per-error-type accuracy (dataset labels + success list) ----------
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

    # ===================== STYLISH LOLLIPOP PLOT =====================
    sorted_by = "accuracy"  # or "volume"

    plot_df = acc_df.copy()
    if sorted_by == "volume":
        plot_df = plot_df.sort_values(["total_cases", "accuracy_percent"], ascending=[False, True])
    else:
        plot_df = plot_df.sort_values(["accuracy_percent", "total_cases"], ascending=[True, False])

    cmap = plt.get_cmap("tab20")

    def _stable_color(label: str):
        return cmap(abs(hash(label)) % 20)

    colors = plot_df["error_type"].map(_stable_color).to_list()

    fig_h = max(4.8, 0.48 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    y = np.arange(len(plot_df))
    x = plot_df["accuracy_percent"].to_numpy()

    ax.hlines(y=y, xmin=0, xmax=x, linewidth=2.2, alpha=0.35, color=colors)
    ax.scatter(x, y, s=110, c=colors, edgecolors="white", linewidths=1.2, zorder=3)

    # "Count bubble" at origin to show volume per type
    for yi, total in zip(y, plot_df["total_cases"].to_numpy()):
        ax.scatter([0], [yi], s=40 + 2.0 * min(total, 60), c=["#DDDDDD"], edgecolors="white",
                   linewidths=0.8, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["error_type"], fontsize=10)

    ax.set_xlim(0, max(100, float(np.nanmax(x)) + 5))
    ax.set_xlabel("Repair Accuracy (%)", fontsize=12)
    ax.set_title("Per-Error-Type Repair Accuracy", fontsize=15, pad=12)

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    median_acc = float(np.nanmedian(x)) if len(x) else 0.0
    ax.axvline(median_acc, linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(median_acc, len(y) - 0.2, f"median {median_acc:.1f}%",
            ha="center", va="bottom", fontsize=9, alpha=0.9)

    for xi, yi, solved, total, acc in zip(
        plot_df["accuracy_percent"].to_numpy(),
        y,
        plot_df["solved_cases"].to_numpy(),
        plot_df["total_cases"].to_numpy(),
        plot_df["accuracy_percent"].to_numpy(),
    ):
        ax.text(
            xi + 0.8,
            yi,
            f"{solved}/{total}  ({acc:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
            alpha=0.95,
        )

    # Callouts: best/worst 3
    k = min(3, len(plot_df))
    if k > 0:
        best_idx = plot_df["accuracy_percent"].nlargest(k).index
        worst_idx = plot_df["accuracy_percent"].nsmallest(k).index
        idx_to_y = {idx: int(pos) for pos, idx in enumerate(plot_df.index)}

        for idx in best_idx:
            yi = idx_to_y[idx]
            ax.text(ax.get_xlim()[1], yi, "  ★", va="center", ha="right", fontsize=12, alpha=0.9)
        for idx in worst_idx:
            yi = idx_to_y[idx]
            ax.text(-1, yi, "✖  ", va="center", ha="left", fontsize=11, alpha=0.9)

    summary = (
        f"Dataset rows: {overall['total_dataset']}\n"
        f"Attempted: {overall['attempted']} (coverage {overall['coverage_percent']}%)\n"
        f"Passed: {overall['passed']}  Failed: {overall['failed']}  Waiting: {overall['waiting']}\n"
        f"Accuracy (passed/attempted): {overall['accuracy_percent_attempted']}%\n"
        f"Total error labels: {total_error_labels}  |  Unique error types: {unique_error_types}\n"
        f"Sorted by: {sorted_by}"
    )
    ax.text(
        0.01,
        0.02,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )

    plt.tight_layout()
    plt.savefig(accuracy_plot_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"\nSaved stylish accuracy plot → {accuracy_plot_path}")
    # =================================================================

    # -------- Table image ----------
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
        "overall_path": str(overall_path),
        "total_error_labels": total_error_labels,
        "unique_error_types": unique_error_types,
        "accuracy_plot_path": str(accuracy_plot_path),
        "accuracy_table_path": str(accuracy_table_path),
    }


if __name__ == "__main__":
    run_error_type_accuracy_evaluation()
