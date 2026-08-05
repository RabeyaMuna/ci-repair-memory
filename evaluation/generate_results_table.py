#!/usr/bin/env python3
"""
Generate publication-ready results tables in multiple formats (Markdown, LaTeX, CSV).
"""

import json
import argparse
from pathlib import Path


def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)


def generate_markdown_table(results):
    """Generate Markdown table."""
    agg = results['aggregate']

    table = """
# Evaluation Results Table

## Overall Metrics

| Metric | Value |
|--------|-------|
| Total Instances | {:,} |
| Instances with Predictions | {} ({:.2f}%) |
| Success Rate | {:.2f}% |

## Accuracy Metrics

| Metric | Count | Rate (%) | Rate (with predictions) (%) |
|--------|-------|----------|----------------------------|
| **Exact Match** | {} | {:.2f}% | {:.2f}% |
| **Top-1** | {} | {:.2f}% | {:.2f}% |
| **Top-2** | {} | {:.2f}% | {:.2f}% |
| **Top-3** | {} | {:.2f}% | {:.2f}% |
| **Top-4** | {} | {:.2f}% | {:.2f}% |
| **Top-5** | {} | {:.2f}% | {:.2f}% |
| **Top-10** | {} | {:.2f}% | {:.2f}% |
| **Top-15** | {} | {:.2f}% | {:.2f}% |

## Precision

| Metric | Value |
|--------|-------|
| Mean Precision (Overall) | {:.4f} |
| Mean Precision (With Predictions) | {:.4f} |
""".format(
        agg['total_instances'],
        agg['with_predictions'], agg['success_rate'],
        agg['success_rate'],
        agg['exact_match'], agg['exact_match_rate'],
        agg['exact_match'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_1'], agg['top_1_rate'],
        agg['top_1'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_2'], agg['top_2_rate'],
        agg['top_2'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_3'], agg['top_3_rate'],
        agg['top_3'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_4'], agg['top_4_rate'],
        agg['top_4'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_5'], agg['top_5_rate'],
        agg['top_5'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_10'], agg['top_10_rate'],
        agg['top_10'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['top_15'], agg['top_15_rate'],
        agg['top_15'] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0,
        agg['mean_precision'],
        agg['total_precision'] / agg['with_predictions'] if agg['with_predictions'] > 0 else 0,
    )

    return table


def generate_latex_table(results):
    """Generate LaTeX table."""
    agg = results['aggregate']

    with_pred_rate = lambda k: (agg[k] / agg['with_predictions'] * 100 if agg['with_predictions'] > 0 else 0)

    table = r"""
\begin{table}[htbp]
\centering
\caption{Evaluation Results: Top-K Accuracy and Precision}
\label{tab:eval_results}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Count} & \textbf{Overall Rate (\%)} & \textbf{Rate (w/ Preds) (\%)} \\
\midrule
Exact Match     & """ + f"{agg['exact_match']:<4}" + r""" & """ + f"{agg['exact_match_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('exact_match'):<6.2f}" + r""" \\
Top-1           & """ + f"{agg['top_1']:<4}" + r""" & """ + f"{agg['top_1_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('top_1'):<6.2f}" + r""" \\
Top-2           & """ + f"{agg['top_2']:<4}" + r""" & """ + f"{agg['top_2_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('top_2'):<6.2f}" + r""" \\
Top-3           & """ + f"{agg['top_3']:<4}" + r""" & """ + f"{agg['top_3_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('top_3'):<6.2f}" + r""" \\
Top-5           & """ + f"{agg['top_5']:<4}" + r""" & """ + f"{agg['top_5_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('top_5'):<6.2f}" + r""" \\
Top-10          & """ + f"{agg['top_10']:<4}" + r""" & """ + f"{agg['top_10_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('top_10'):<6.2f}" + r""" \\
Top-15          & """ + f"{agg['top_15']:<4}" + r""" & """ + f"{agg['top_15_rate']:<6.2f}" + r""" & """ + f"{with_pred_rate('top_15'):<6.2f}" + r""" \\
\midrule
\multicolumn{4}{l}{\textit{Precision Metrics}} \\
Mean Precision (Overall)         & \multicolumn{3}{r}{""" + f"{agg['mean_precision']:.4f}" + r"""} \\
Mean Precision (With Predictions) & \multicolumn{3}{r}{""" + f"{with_pred_rate('total_precision') / 100:.4f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}

% Coverage Statistics
Total Instances: """ + f"{agg['total_instances']:,}" + r"""
Instances with Predictions: """ + f"{agg['with_predictions']}" + r""" (""" + f"{agg['success_rate']:.2f}" + r"""\%)
"""

    return table


def generate_compact_table(results):
    """Generate compact comparison table."""
    agg = results['aggregate']
    n_pred = agg['with_predictions']

    compact = """
COMPACT RESULTS TABLE
=====================

Metric         | Overall | With Predictions
--------       | ------- | ----------------
Coverage       | {:.2f}% | 100.00%
Exact Match    | {:.2f}% | {:.2f}%
Top-1          | {:.2f}% | {:.2f}%
Top-3          | {:.2f}% | {:.2f}%
Top-5          | {:.2f}% | {:.2f}%
Top-10         | {:.2f}% | {:.2f}%
Top-15         | {:.2f}% | {:.2f}%
Precision      | {:.4f} | {:.4f}
""".format(
        agg['success_rate'],
        agg['exact_match_rate'], agg['exact_match'] / n_pred * 100 if n_pred > 0 else 0,
        agg['top_1_rate'], agg['top_1'] / n_pred * 100 if n_pred > 0 else 0,
        agg['top_3_rate'], agg['top_3'] / n_pred * 100 if n_pred > 0 else 0,
        agg['top_5_rate'], agg['top_5'] / n_pred * 100 if n_pred > 0 else 0,
        agg['top_10_rate'], agg['top_10'] / n_pred * 100 if n_pred > 0 else 0,
        agg['top_15_rate'], agg['top_15'] / n_pred * 100 if n_pred > 0 else 0,
        agg['mean_precision'], agg['total_precision'] / n_pred if n_pred > 0 else 0,
    )

    return compact


def main():
    parser = argparse.ArgumentParser(description='Generate results tables in multiple formats')
    parser.add_argument('--results', default='results/evaluation_results.json',
                        help='Path to evaluation results JSON')
    parser.add_argument('--format', choices=['all', 'markdown', 'latex', 'compact'],
                        default='all', help='Output format')
    parser.add_argument('--output-dir', default='results/tables',
                        help='Output directory for tables')

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    if args.format in ['all', 'markdown']:
        md_table = generate_markdown_table(results)
        md_path = output_dir / 'results_table.md'
        with open(md_path, 'w') as f:
            f.write(md_table)
        print(f"Markdown table saved to: {md_path}")

    if args.format in ['all', 'latex']:
        latex_table = generate_latex_table(results)
        latex_path = output_dir / 'results_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")

    if args.format in ['all', 'compact']:
        compact_table = generate_compact_table(results)
        compact_path = output_dir / 'results_table_compact.txt'
        with open(compact_path, 'w') as f:
            f.write(compact_table)
        print(f"Compact table saved to: {compact_path}")
        print("\n" + compact_table)

    print("\n✓ Table generation complete!")


if __name__ == "__main__":
    main()
