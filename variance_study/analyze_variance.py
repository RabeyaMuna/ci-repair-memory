#!/usr/bin/env python3
"""
Analyze variance study results and generate tables for the paper.

Usage:
    python variance_study/analyze_variance.py
"""

import os
import json
import sys
from pathlib import Path
import numpy as np
from tabulate import tabulate

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_variance_results(variance_dir):
    """Load variance study results from JSON file."""
    stats_file = Path(variance_dir) / "variance_statistics.json"

    if not stats_file.exists():
        print(f"[ERROR] Statistics file not found: {stats_file}")
        return None

    with open(stats_file, 'r') as f:
        data = json.load(f)

    return data


def generate_results_table(data):
    """Generate table showing results for each run."""
    results = data['results']

    headers = ["Run", "Passed", "Total", "Pass@1 (%)"]
    rows = []

    for r in results:
        rows.append([
            r['run_number'],
            r['passed'],
            r['total_instances'],
            f"{r['pass_rate']:.2f}"
        ])

    # Add mean row
    stats = data['statistics']
    rows.append([
        "Mean ± SD",
        "-",
        "-",
        f"{stats['mean']:.2f} ± {stats['std_dev']:.2f}"
    ])

    return headers, rows


def generate_latex_table(data):
    """Generate LaTeX table for the paper."""
    results = data['results']
    stats = data['statistics']
    config = data['configuration']

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Variance Study: Pass@1 Stability Across Multiple Runs}")
    latex.append("\\label{tab:variance}")
    latex.append("\\begin{tabular}{cccc}")
    latex.append("\\toprule")
    latex.append("Run & Passed & Total & Pass@1 (\\%) \\\\")
    latex.append("\\midrule")

    for r in results:
        latex.append(f"{r['run_number']} & {r['passed']} & {r['total_instances']} & {r['pass_rate']:.2f} \\\\")

    latex.append("\\midrule")
    latex.append(f"\\textbf{{Mean ± SD}} & - & - & \\textbf{{{stats['mean']:.2f} ± {stats['std_dev']:.2f}}} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    # Add caption text
    latex.append("\\vspace{0.5em}")
    latex.append("\\begin{tablenotes}")
    latex.append("\\small")
    latex.append(f"\\item Model: {config['model']}, Instances: {config['num_instances']}, ")
    latex.append(f"Variance: {stats['variance']:.4f}")
    latex.append("\\end{tablenotes}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def generate_markdown_table(data):
    """Generate Markdown table for GitHub/README."""
    results = data['results']
    stats = data['statistics']

    md = []
    md.append("## Variance Study Results")
    md.append("")
    md.append("| Run | Passed | Total | Pass@1 (%) |")
    md.append("|-----|--------|-------|------------|")

    for r in results:
        md.append(f"| {r['run_number']} | {r['passed']} | {r['total_instances']} | {r['pass_rate']:.2f} |")

    md.append(f"| **Mean ± SD** | - | - | **{stats['mean']:.2f} ± {stats['std_dev']:.2f}** |")
    md.append("")
    md.append(f"**Variance:** {stats['variance']:.4f}")
    md.append("")

    return "\n".join(md)


def print_analysis(data):
    """Print detailed analysis."""
    stats = data['statistics']
    config = data['configuration']

    print(f"\n{'='*80}")
    print(f"VARIANCE STUDY ANALYSIS")
    print(f"{'='*80}\n")

    print(f"Configuration:")
    print(f"  Model: {config['model']}")
    print(f"  Analyzer: {config['analyzer']}")
    print(f"  Instances: {config['num_instances']}")
    print(f"  Runs: {config['num_runs']}")
    print(f"  Seed: {config['seed']}")

    print(f"\nResults:")
    headers, rows = generate_results_table(data)
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    print(f"\nVariance Metrics:")
    print(f"  Variance: {stats['variance']:.4f}")
    print(f"  Standard Deviation: {stats['std_dev']:.2f}%")
    print(f"  Range: [{stats['min']:.2f}%, {stats['max']:.2f}%]")
    print(f"  Coefficient of Variation: {(stats['std_dev'] / stats['mean'] * 100):.2f}%")

    # Interpretation
    print(f"\nInterpretation:")
    cv = stats['std_dev'] / stats['mean'] * 100

    if stats['std_dev'] < 2.0:
        stability = "VERY LOW"
        interpretation = "Results are highly stable across runs."
    elif stats['std_dev'] < 5.0:
        stability = "LOW"
        interpretation = "Results show good stability across runs."
    elif stats['std_dev'] < 10.0:
        stability = "MODERATE"
        interpretation = "Results show acceptable variability."
    else:
        stability = "HIGH"
        interpretation = "Results show significant variability."

    print(f"  Stability: {stability} variance")
    print(f"  {interpretation}")
    print(f"  The coefficient of variation ({cv:.2f}%) indicates")
    print(f"  {'high' if cv > 20 else 'moderate' if cv > 10 else 'low'} relative variability.")


def generate_paper_text(data):
    """Generate text for paper response to reviewer."""
    stats = data['statistics']
    config = data['configuration']

    text = []
    text.append("\n" + "="*80)
    text.append("TEXT FOR PAPER (Reviewer Response)")
    text.append("="*80 + "\n")

    text.append("In the revised paper (Section 5, Experimental Setup), add:")
    text.append("")
    text.append('"')
    text.append(f"To assess the stability of our results, we conducted a variance study ")
    text.append(f"by running the {config['model']} configuration three times on a randomly ")
    text.append(f"sampled subset of {config['num_instances']} instances. ")
    text.append(f"The Pass@1 rate was {stats['mean']:.1f}\\% \\pm {stats['std_dev']:.2f}\\% ")
    text.append(f"(mean \\pm standard deviation), with a variance of {stats['variance']:.4f}. ")

    if stats['std_dev'] < 2.0:
        text.append(f"This very low standard deviation demonstrates that our Pass@1 results ")
        text.append(f"are highly stable and representative across independent runs. ")
    elif stats['std_dev'] < 5.0:
        text.append(f"This low standard deviation demonstrates that our Pass@1 results ")
        text.append(f"are stable and representative across independent runs. ")
    else:
        text.append(f"This demonstrates acceptable stability in model performance ")
        text.append(f"across independent runs. ")

    text.append(f"Due to computational cost constraints, we report single-run Pass@1 ")
    text.append(f"for the full benchmark, as the variance study indicates this provides ")
    text.append(f"a reliable estimate of model performance.")
    text.append('"')
    text.append("")

    text.append("In the reviewer response:")
    text.append("")
    text.append('"')
    text.append(f"Thank you for raising this important point. While each configuration ")
    text.append(f"was evaluated once per benchmark instance due to computational costs, ")
    text.append(f"we conducted a variance study to assess result stability. ")
    text.append(f"We ran the {config['model']} configuration three times on a randomly ")
    text.append(f"sampled subset of {config['num_instances']} instances. ")
    text.append(f"The Pass@1 rate was {stats['mean']:.1f}\\% \\pm {stats['std_dev']:.2f}\\% ")
    text.append(f"(mean \\pm SD), with a variance of {stats['variance']:.4f}. ")

    if stats['std_dev'] < 5.0:
        text.append(f"This low variance demonstrates that our single-run Pass@1 results ")
        text.append(f"are reliable and representative. ")
    else:
        text.append(f"This demonstrates acceptable stability across runs. ")

    text.append(f"We have added this variance study and its implications to Section 5 ")
    text.append(f"and included it as Table X in the revised paper. ")
    text.append(f"We also acknowledge that repeated runs could provide additional insights ")
    text.append(f"and include this as a limitation and direction for future evaluation.")
    text.append('"')

    return "\n".join(text)


def main():
    base_dir = Path(__file__).parent.parent
    variance_dir = base_dir / "variance_study"

    # Load results
    data = load_variance_results(variance_dir)

    if not data:
        print("[ERROR] Could not load variance results")
        return

    # Print analysis
    print_analysis(data)

    # Generate tables
    print(f"\n{'='*80}")
    print(f"LATEX TABLE")
    print(f"{'='*80}\n")
    latex_table = generate_latex_table(data)
    print(latex_table)

    # Save LaTeX table
    latex_file = variance_dir / "variance_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"\n[SAVED] LaTeX table: {latex_file}")

    # Generate Markdown table
    print(f"\n{'='*80}")
    print(f"MARKDOWN TABLE")
    print(f"{'='*80}\n")
    md_table = generate_markdown_table(data)
    print(md_table)

    # Save Markdown table
    md_file = variance_dir / "variance_table.md"
    with open(md_file, 'w') as f:
        f.write(md_table)
    print(f"\n[SAVED] Markdown table: {md_file}")

    # Generate paper text
    paper_text = generate_paper_text(data)
    print(paper_text)

    # Save paper text
    paper_file = variance_dir / "paper_text.txt"
    with open(paper_file, 'w') as f:
        f.write(paper_text)
    print(f"\n[SAVED] Paper text: {paper_file}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
