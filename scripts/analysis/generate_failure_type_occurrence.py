#!/usr/bin/env python3
"""Generate publication-ready failure-type occurrence statistics."""

import argparse
import ast
import csv
from collections import Counter
from pathlib import Path

import pandas as pd


def normalize_types(value):
    """Return a failure-type cell as a clean list of labels."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return [value]
        else:
            return [value]
    if not isinstance(value, (list, tuple, set)):
        return [str(value).strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def count_occurrences(values):
    """Count each failure type at most once per dataset instance."""
    counts = Counter()
    for value in values:
        counts.update(set(normalize_types(value)))
    return counts


def latex_escape(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def write_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["failure_type", "occurrences"])
        writer.writerows(rows)


def write_latex(rows, num_instances, output_path):
    total_occurrences = sum(count for _, count in rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \caption{Failure-type occurrences across "
        + f"{num_instances:,} instances. Categories are not mutually exclusive.}}",
        r"  \label{tab:failure-type-occurrence}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    CI failure type & \# Occurrences \\",
        r"    \midrule",
    ]
    lines.extend(
        f"    {latex_escape(failure_type)} & {count:,} \\\\"
        for failure_type, count in rows
    )
    lines.extend(
        [
            r"    \midrule",
            f"    \\textbf{{Total occurrences}} & \\textbf{{{total_occurrences:,}}} \\\\",
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(rows, output_path):
    import matplotlib.pyplot as plt

    labels = [failure_type for failure_type, _ in reversed(rows)]
    counts = [count for _, count in reversed(rows)]
    fig_height = max(2.8, 0.28 * len(rows))
    fig, axis = plt.subplots(figsize=(5.2, fig_height))
    axis.barh(labels, counts, color="#4C78A8")
    axis.set_xlabel("Number of instances")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create a compact LaTeX table of failure-type occurrences."
    )
    parser.add_argument("--dataset", default="dataset/lca_dataset.parquet")
    parser.add_argument(
        "--column",
        default="error_type",
        help="Parquet column containing failure labels (default: error_type)",
    )
    parser.add_argument("--output-dir", default="results/paper")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a horizontal PDF bar chart",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_parquet(dataset_path, columns=[args.column])
    counts = count_occurrences(dataframe[args.column])
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    csv_path = output_dir / "failure_type_occurrence.csv"
    latex_path = output_dir / "failure_type_occurrence.tex"
    write_csv(rows, csv_path)
    write_latex(rows, len(dataframe), latex_path)

    print(f"Instances: {len(dataframe):,}")
    print(f"Unique failure types: {len(rows):,}")
    print(f"Total occurrences: {sum(count for _, count in rows):,}")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")

    if args.plot:
        plot_path = output_dir / "failure_type_occurrence.pdf"
        write_plot(rows, plot_path)
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
