#!/usr/bin/env python3
import json
import ast
from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Paths ----------
DATASET_PATH = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet")
SUCCESS_PATH = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/jobs_success_diff.jsonl")

OUTPUT_DIR = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/evaluation_plot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCURACY_PLOT_PATH = OUTPUT_DIR / "error_type_accuracy_lollipop.png"
ACCURACY_TABLE_PATH = OUTPUT_DIR / "error_type_accuracy_table.png"

# -------- Load dataset ----------
df = pd.read_parquet(DATASET_PATH)
df["id"] = df["id"].astype(str)

# ---- Helpers to normalize / flatten labels ----
def parse_maybe_list(x):
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v
        except Exception:
            return x
    return x


def extract_labels(x):
    """Flatten any shape into a list of plain string labels."""
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


df["error_labels"] = df["error_type"].apply(extract_labels)

total_datapoints = len(df)

# -------- Load success IDs ----------
success_ids = set()
with open(SUCCESS_PATH, "r") as f:
    for line in f:
        obj = json.loads(line)
        success_ids.add(str(obj["id"]))

# -------- Count totals & successes per error type ----------
total_counter = defaultdict(int)
success_counter = defaultdict(int)

for row in df.itertuples():
    rid = str(row.id)
    labels = row.error_labels
    for label in labels:
        total_counter[label] += 1
        if rid in success_ids:
            success_counter[label] += 1

total_error_labels = sum(total_counter.values())
unique_error_types = len(total_counter)

# -------- Build accuracy table ----------
rows = []
for err_type in sorted(total_counter.keys()):
    total = total_counter[err_type]
    solved = success_counter[err_type]
    accuracy = round((solved / total) * 100, 2) if total > 0 else 0.0
    share = round((total / total_error_labels) * 100, 2) if total_error_labels > 0 else 0.0

    rows.append([
        err_type,
        total,
        solved,
        accuracy,
        share,
    ])

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

print("\n=== Global stats ===")
print(f"Total datapoints (jobs): {total_datapoints}")
print(f"Total error labels (all types counted): {total_error_labels}")
print(f"Unique error types: {unique_error_types}")

print("\n=== Per-Error-Type Accuracy ===\n")
print(acc_df.to_string(index=False))

# -------- Lollipop (dot) chart for accuracy ----------
sorted_df = acc_df.sort_values("accuracy_percent")

fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_df) * 0.45)))

y_pos = np.arange(len(sorted_df))

# lines from 0 to point (lollipop sticks)
ax.hlines(y=y_pos, xmin=0, xmax=sorted_df["accuracy_percent"])

# dots at accuracies
ax.scatter(sorted_df["accuracy_percent"], y_pos, s=60)

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_df["error_type"])
ax.set_xlabel("Accuracy (%)")
ax.set_title("Per-Error-Type Repair Accuracy")

ax.grid(axis="x", linestyle="--", alpha=0.4)

# annotate each point with "solved/total  (acc%)"
for x_val, y_idx, solved, total, acc in zip(
    sorted_df["accuracy_percent"],
    y_pos,
    sorted_df["solved_cases"],
    sorted_df["total_cases"],
    sorted_df["accuracy_percent"],
):
    ax.text(
        x_val,
        y_idx,
        f"  {solved}/{total}  ({acc}%)",
        va="center",
        ha="left",
        fontsize=8,
    )

# summary text at top-left of the figure
fig.text(
    0.01,
    0.98,
    f"Total datapoints: {total_datapoints}\n"
    f"Total error labels: {total_error_labels}\n"
    f"Unique error types: {unique_error_types}",
    ha="left",
    va="top",
    fontsize=9,
)

plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(ACCURACY_PLOT_PATH, dpi=200)
plt.close()

print(f"\nSaved accuracy lollipop plot → {ACCURACY_PLOT_PATH}")

# -------- Save accuracy table as image ----------
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
plt.savefig(ACCURACY_TABLE_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved accuracy table image → {ACCURACY_TABLE_PATH}")
