#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate paper figures for per-error-type repair accuracy.

Outputs (default): <repo_root>/attachments/
  - error_type_accuracy_paper.pdf  (VECTOR: best for Overleaf/ICML)
  - error_type_accuracy_paper.png  (fallback)
  - overall.json

Run:
  python3 evaluation_plot/new_plot.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prettify_error_type(name: str) -> str:
    """Shorten category names for the paper figure (display only)."""
    s = str(name).strip()

    mapping = {
        "Documentation or Docstring Error": "Docstring",
        "Package Installation Error": "Package Install",
        "Configuration Error": "Config",
        "Environment Error": "Environment",
        "Type Checking Error": "Type Checking",
        "Assertion Error": "Assertion",
        "Syntax Error": "Syntax",
        "Runtime Error": "Runtime",
        "Dependency Issues": "Dependencies",
        "Test Failure": "Tests",
        "Code Linting": "Linting",
        "Code Formatting": "Formatting",
    }

    if s in mapping:
        s = mapping[s]

    if s.endswith(" Error"):
        s = s[:-6]

    # hard cap (prevents very long labels from breaking the figure)
    max_len = 18
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"

    return s

# ------------------------- ICML-friendly style (bigger + bold for x/y/% text) -------------------------

def set_icml_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "font.weight": "bold",

        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",

        "xtick.labelsize": 13,
        "ytick.labelsize": 13,

        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,

        # Embed TrueType fonts in PDF so text stays crisp/selectable
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ------------------------- Repo root -------------------------

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "dataset").exists() and (p / "results").exists():
            return p
    return Path(__file__).resolve().parents[1]


# ------------------------- Parsing -------------------------

def parse_maybe_list(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except Exception:
                return x
    return x


def extract_labels(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    x = parse_maybe_list(x)

    if isinstance(x, np.ndarray):
        x = x.tolist()

    if isinstance(x, Iterable) and not isinstance(x, (str, bytes, dict)):
        out: list[str] = []
        for item in x:
            out.extend(extract_labels(item))
        return out

    return [str(x)]


# ------------------------- JSONL loaders -------------------------

def load_jsonl_rows(path: Path) -> list[dict]:
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
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def ids_from_rows(rows: list[dict] | None) -> set[str]:
    out: set[str] = set()
    if not rows:
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in ("id", "job_id", "run_id"):
            v = r.get(k)
            if v is not None:
                out.add(str(v))
                break
    return out


def load_success_ids(success_path: Path) -> set[str]:
    success_path = Path(success_path)
    if not success_path.exists():
        return set()
    out: set[str] = set()
    with open(success_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("id") is not None:
                out.add(str(obj["id"]))
    return out


def load_stream_conclusions(stream_results_path: Path | None) -> dict[str, str]:
    if stream_results_path is None:
        return {}
    stream_results_path = Path(stream_results_path)
    if not stream_results_path.exists():
        return {}

    out: dict[str, str] = {}
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
            out[str(rid)] = str(conc or "").lower()
    return out


# ------------------------- Overall outcomes (ONLY total/passed/accuracy) -------------------------

def compute_overall_outcomes(
    *,
    dataset_ids: set[str],
    success_ids: set[str],
    jobs_results: list[dict] | None,
    jobs_invalid: list[dict] | None,
    jobs_awaiting: list[dict] | None,
    jobs_failure: list[dict] | None,
    jobs_error: list[dict] | None,
    stream_results_path: Path | None,
) -> dict[str, Any]:
    """
    Returns ONLY:
      - total_instances: number of dataset items
      - passed: number of successes that belong to the dataset and were attempted
      - accuracy_percent_total: passed / total_instances * 100

    Notes:
    - We intersect with dataset_ids to avoid counting stray ids.
    - We use "attempt evidence" to avoid counting successes that were never run/pushed.
    """

    total_instances = len(dataset_ids)

    # evidence of an attempt (dataset-grounded)
    results_ids = ids_from_rows(jobs_results)
    invalid_ids = ids_from_rows(jobs_invalid)
    waiting_ids = ids_from_rows(jobs_awaiting)
    failure_ids = ids_from_rows(jobs_failure)
    error_file_ids = ids_from_rows(jobs_error)

    id_to_conc = load_stream_conclusions(stream_results_path)
    stream_ids = set(id_to_conc.keys())

    attempted_ids = (
        results_ids
        | invalid_ids
        | waiting_ids
        | stream_ids
        | set(success_ids)
        | failure_ids
        | error_file_ids
    ) & dataset_ids

    passed_ids = (set(success_ids) & attempted_ids)
    passed = len(passed_ids)

    accuracy_percent_total = round((passed / total_instances) * 100, 2) if total_instances else 0.0

    return {
        "total_instances": total_instances,
        "passed": passed,
        "accuracy_percent_total": accuracy_percent_total,
    }

# ------------------------- Color helpers -------------------------

def _lighten_rgba(rgba, factor: float = 0.60):
    r, g, b, a = rgba
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b, a)


def _darken_rgba(rgba, factor: float = 0.28):
    r, g, b, a = rgba
    r = r * (1 - factor)
    g = g * (1 - factor)
    b = b * (1 - factor)
    return (r, g, b, a)


def _cat_idx(label: str, n: int) -> int:
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


# ------------------------- Plot -------------------------

def save_paper_plot(
    acc_df: pd.DataFrame,
    overall: dict[str, Any],
    out_pdf: Path,
    out_png: Path,
) -> None:
    set_icml_style()

    plot_df = acc_df.copy()
    plot_df["unsolved_cases"] = plot_df["total_cases"] - plot_df["solved_cases"]

    # Sort by HIGHEST solved -> LOWEST solved
    plot_df = plot_df.sort_values(
        ["solved_cases", "accuracy_percent", "total_cases", "error_type"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    labels = [prettify_error_type(x) for x in plot_df["error_type"].tolist()]

    solved = plot_df["solved_cases"].to_numpy(dtype=float)
    unsolved = plot_df["unsolved_cases"].to_numpy(dtype=float)
    total = plot_df["total_cases"].to_numpy(dtype=float)
    acc = plot_df["accuracy_percent"].to_numpy(dtype=float)

    base_cmap = plt.get_cmap("tab20")
    base_colors = [base_cmap(_cat_idx(lab, base_cmap.N)) for lab in labels]
    solved_colors = [_darken_rgba(c, factor=0.28) for c in base_colors]
    unsolved_colors = [_lighten_rgba(c, factor=0.60) for c in base_colors]

    n = len(plot_df)

    # Figure size tuned for ICML figure*
    fig_w = 12.5
    fig_h = max(5.0, 0.32 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Keep margins standard; everything will be INSIDE axes now
    fig.subplots_adjust(left=0.34, right=0.96, bottom=0.12, top=0.86)

    y = np.arange(n)
    bar_h = 0.72

    ax.barh(y, unsolved, height=bar_h, color=unsolved_colors, edgecolor="white", linewidth=0.7)
    ax.barh(y, solved, height=bar_h, left=unsolved, color=solved_colors, edgecolor="white", linewidth=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")

    ax.set_xlabel("Number of cases", fontsize=15, fontweight="bold")
    # ax.set_title("System Repair Accuracy per Error Type (Most Solved → Least Solved)", fontsize=16, fontweight="bold", pad=4)

    for t in ax.get_xticklabels():
        t.set_fontweight("bold")

    ax.grid(axis="x", linestyle="--", alpha=0.20)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.invert_yaxis()

    # Accuracy axis + dots on top axis
    ax2 = ax.twiny()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Repair Accuracy per Error Category (%)", fontsize=15, fontweight="bold")
    ax2.scatter(acc, y, s=36, c="black", zorder=5)
    for t in ax2.get_xticklabels():
        t.set_fontweight("bold")

    # median_acc = float(np.nanmedian(acc)) if len(acc) else 0.0
    # ax2.axvline(median_acc, linestyle=":", linewidth=1.6, alpha=0.95, color="black")
    # ax2.text(
    #     median_acc, 1.012,
    #     f"median {median_acc:.1f}%",
    #     transform=ax2.get_xaxis_transform(),
    #     ha="center", va="bottom",
    #     fontsize=13, fontweight="bold",
    #     color="black",
    # )

    # -------------------- Put Solved/Total(Accuracy) beside each bar (INSIDE axes) --------------------
    max_total = float(np.nanmax(total)) if len(total) else 1.0

    # Create a small right gutter inside the axes for labels + bottom-right box
    # (keeps labels and the box from sitting on the bar ends)
    ax.set_xlim(0, max_total * 1.22)

    x_right = ax.get_xlim()[1]
    label_pad = 0.02 * x_right  # small padding in data units

    # Position label near end of total bar, but clamp so it never goes out of xlim
    for yi, s, u, t, a in zip(y, solved, unsolved, total, acc):
        bar_end = float(u + s)  # end of stacked bar
        label_pad = 0.014 * x_right          # smaller padding -> closer to bar
        right_margin = 0.03 * x_right        # keep text away from border

        for yi, s, u, t, a in zip(y, solved, unsolved, total, acc):
            bar_end = float(u + s)
            txt = f"{int(s)}/{int(t)} ({a:.1f}%)"

            # First try: just after the bar
            x0 = bar_end + label_pad

            # If it would overflow, move it left so it fits
            # Approximate width in data units: use character count
            approx_char_w = 0.0065 * x_right   # tweakable constant
            approx_text_w = approx_char_w * len(txt)

            # Clamp: ensure x0 + text_width <= x_right - right_margin
            x_safe = min(x0, (x_right - right_margin) - approx_text_w)

            # Never go negative
            x_safe = max(0.0, x_safe)

            ax.text(
                x_safe, yi, txt,
                va="center", ha="left",
                fontsize=12, fontweight="bold",
                color="black",
                clip_on=True,
            )


    # -------------------- Overall box: bottom-right INSIDE axes (no overlap) --------------------
    overall_box = (
        f"Total instances: {overall['total_instances']}\n"
        f"Passed: {overall['passed']}\n"
        f"Accuracy: {overall['accuracy_percent_total']:.2f}%"
    )

    ax.text(
        0.985, 0.02,
        overall_box,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=12.5, fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="black",
            linewidth=1.1,
            alpha=0.98,
        ),
        zorder=10,
        clip_on=False,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_png, dpi=450, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

# ------------------------- Main evaluation -------------------------

def run(dataset_path: Path, results_dir: Path, output_dir: Path, stream_results_path: Path | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    success_path = results_dir / "jobs_success_diff.jsonl"
    jobs_results_path = results_dir / "jobs_results_diff.jsonl"
    jobs_invalid_path = results_dir / "jobs_invalid_diff.jsonl"
    jobs_awaiting_path = results_dir / "jobs_awaiting_diff.jsonl"
    jobs_failure_path = results_dir / "jobs_failure_diff.jsonl"
    jobs_error_path = results_dir / "jobs_error_diff.jsonl"

    plot_pdf = output_dir / "error_type_accuracy_paper.pdf"
    plot_png = output_dir / "error_type_accuracy_paper.png"
    overall_json = output_dir / "overall.json"

    df = pd.read_parquet(dataset_path)
    if "id" not in df.columns or "error_type" not in df.columns:
        raise KeyError(f"Dataset must include columns ['id','error_type']. Found: {list(df.columns)}")

    df["id"] = df["id"].astype(str)
    df["error_labels"] = df["error_type"].apply(extract_labels)

    dataset_ids = set(df["id"].tolist())
    success_ids = load_success_ids(success_path)

    jobs_results = load_jsonl_rows(jobs_results_path)
    jobs_invalid = load_jsonl_rows(jobs_invalid_path)
    jobs_awaiting = load_jsonl_rows(jobs_awaiting_path)
    jobs_failure = load_jsonl_rows(jobs_failure_path)
    jobs_error = load_jsonl_rows(jobs_error_path)

    overall = compute_overall_outcomes(
        dataset_ids=dataset_ids,
        success_ids=success_ids,
        jobs_results=jobs_results,
        jobs_invalid=jobs_invalid,
        jobs_awaiting=jobs_awaiting,
        jobs_failure=jobs_failure,
        jobs_error=jobs_error,
        stream_results_path=stream_results_path,
    )

    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    # Per-error-type counters
    total_counter: dict[str, int] = defaultdict(int)
    success_counter: dict[str, int] = defaultdict(int)

    for row in df.itertuples(index=False):
        rid = str(getattr(row, "id"))
        labels = getattr(row, "error_labels")
        for label in labels:
            total_counter[label] += 1
            if rid in success_ids:
                success_counter[label] += 1

    total_error_labels = sum(total_counter.values())

    rows = []
    for err_type in sorted(total_counter.keys()):
        total = total_counter[err_type]
        solved = success_counter[err_type]
        accuracy = round((solved / total) * 100, 2) if total else 0.0
        share = round((total / total_error_labels) * 100, 2) if total_error_labels else 0.0
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

    save_paper_plot(acc_df, overall, plot_pdf, plot_png)

    return {
        "plot_pdf": str(plot_pdf),
        "plot_png": str(plot_png),
        "overall_json": str(overall_json),
        "overall": overall,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--results", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--stream", type=str, default=None)
    return p


def main() -> None:
    repo_root = find_repo_root(Path(__file__).resolve())
    args = build_argparser().parse_args()

    dataset_path = Path(args.dataset) if args.dataset else (repo_root / "dataset" / "lca_dataset.parquet")
    results_dir = Path(args.results) if args.results else (repo_root / "results")
    output_dir = Path(args.output) if args.output else (repo_root / "attachments")
    stream_path = Path(args.stream) if args.stream else None

    info = run(dataset_path, results_dir, output_dir, stream_path)

    print("\nSaved outputs:")
    print(f"  Plot (PDF): {info['plot_pdf']}")
    print(f"  Plot (PNG): {info['plot_png']}")
    print(f"  Overall:    {info['overall_json']}")
    print(f"  Overall values: {info['overall']}")


if __name__ == "__main__":
    main()
