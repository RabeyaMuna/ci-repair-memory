#!/usr/bin/env python3
import ast
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- PATHS ----------
DATASET_PATH = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet"
)
OUTPUT_DIR = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/data_analysis/dataset_plot"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BAR_PLOT_PATH = OUTPUT_DIR / "error_type_distribution.png"
TABLE_IMAGE_PATH = OUTPUT_DIR / "error_type_table.png"

# ---------- LOAD DATA ----------
df = pd.read_parquet(DATASET_PATH)

if "error_type" not in df.columns:
    raise SystemExit("error_type column is missing from the dataset.")


# ---------- HELPER: FLATTEN ANY SHAPE INTO PLAIN STRING LABELS ----------
def extract_labels(x):
    """Return a flat list of plain string labels, no nested [] / arrays left."""
    # None / NaN -> no labels
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # if it's a string that *might* represent a list, try to parse it
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            # if parsed is still a string, fall through below
            if not isinstance(parsed, str):
                return extract_labels(parsed)
        except Exception:
            return [x]
        return [x]

    # NumPy array -> list
    if isinstance(x, np.ndarray):
        x = x.tolist()

    # Generic iterable (list, tuple, set, pd.Series, etc.) but NOT string/bytes
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        labels = []
        for item in x:
            labels.extend(extract_labels(item))
        return labels

    # Fallback: single scalar -> string
    return [str(x)]


# ---------- FLATTEN ALL ERROR TYPES ----------
all_labels = []
for cell in df["error_type"]:
    all_labels.extend(extract_labels(cell))

# Now all_labels is like:
# ["Environment Error", "Runtime Error", "Syntax Error", "Environment Error", "Assertion Error", ...]

# ---------- COUNT & PERCENTAGES ----------
counter = Counter(all_labels)
total_error_labels = sum(counter.values())

rows = []
for error_type, count in counter.most_common():  # sorted by count desc
    pct = round(count / total_error_labels * 100, 2)
    rows.append([error_type, count, pct])

stats_df = pd.DataFrame(rows, columns=["error_type", "count", "percentage"])

print("Error type statistics (overall):")
print(stats_df.to_string(index=False))

# ---------- 1. HORIZONTAL BAR PLOT ----------
sorted_df = stats_df.sort_values("count", ascending=True)

plt.figure(figsize=(8, max(4, len(sorted_df) * 0.4)))
plt.barh(sorted_df["error_type"], sorted_df["percentage"])
plt.xlabel("Percentage of all error labels (%)")
plt.title("Error Type Distribution Across Dataset")
plt.tight_layout()
plt.savefig(BAR_PLOT_PATH, dpi=200)
plt.close()

print(f"\nSaved bar plot to: {BAR_PLOT_PATH}")

# ---------- 2. TABLE AS IMAGE ----------
fig, ax = plt.subplots(figsize=(8, max(3, len(stats_df) * 0.35)))
ax.axis("off")

table = ax.table(
    cellText=stats_df.values,
    colLabels=stats_df.columns,
    cellLoc="center",
    loc="center",
)
table.scale(1, 1.3)
plt.title("Error Types: Count and Percentage of All Error Labels")
plt.savefig(TABLE_IMAGE_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved table image to: {TABLE_IMAGE_PATH}")
