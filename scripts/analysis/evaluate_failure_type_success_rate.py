#!/usr/bin/env python3
"""
Failure Type Success Rate Analysis for ICLR Paper

Shows: For each failure type, what percentage of instances were successfully solved?
This answers: "When the system encounters failure type X, how likely is it to solve it?"
"""

import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set up paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "paper"
OUTPUT_DIR.mkdir(exist_ok=True)

# Input files
FAILURE_CLASSIFICATIONS = RESULTS_DIR / "failure_classifications_12types.json"
ALL_RESULTS = RESULTS_DIR / "jobs_results_diff.jsonl"

def load_benchmark_results():
    """Load all benchmark results."""
    successful_ids = set()
    failed_ids = set()

    with open(ALL_RESULTS, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                issue_id = result['id']
                conclusion = result['conclusion']

                if conclusion == 'success':
                    successful_ids.add(issue_id)
                else:
                    failed_ids.add(issue_id)

    return successful_ids, failed_ids

def load_failure_classifications():
    """Load failure type classifications."""
    with open(FAILURE_CLASSIFICATIONS, 'r') as f:
        data = json.load(f)

    classifications = {}
    for item in data['classifications']:
        issue_id = item['issue_id']
        failure_types = item['failure_type']
        if isinstance(failure_types, str):
            failure_types = [failure_types]
        classifications[issue_id] = failure_types

    return classifications, data['taxonomy']

def analyze_success_rates():
    """Calculate success rate for each failure type."""

    print("=" * 90)
    print("FAILURE TYPE SUCCESS RATE ANALYSIS (For ICLR Paper)")
    print("=" * 90)
    print()

    # Load data
    successful_ids, failed_ids = load_benchmark_results()
    all_tested_ids = successful_ids | failed_ids
    classifications, taxonomy = load_failure_classifications()

    print(f"Total instances tested: {len(all_tested_ids)}")
    print(f"Total successful: {len(successful_ids)} ({len(successful_ids)/len(all_tested_ids)*100:.1f}%)")
    print()

    # Count per failure type
    failure_stats = defaultdict(lambda: {
        'total_occurrences': 0,
        'solved': 0,
        'failed': 0,
        'success_rate': 0.0,
        'solved_ids': [],
        'failed_ids': []
    })

    for issue_id, failure_types in classifications.items():
        if issue_id not in all_tested_ids:
            continue

        is_successful = issue_id in successful_ids

        for failure_type in failure_types:
            failure_stats[failure_type]['total_occurrences'] += 1

            if is_successful:
                failure_stats[failure_type]['solved'] += 1
                failure_stats[failure_type]['solved_ids'].append(issue_id)
            else:
                failure_stats[failure_type]['failed'] += 1
                failure_stats[failure_type]['failed_ids'].append(issue_id)

    # Calculate success rates
    for failure_type in failure_stats:
        total = failure_stats[failure_type]['total_occurrences']
        solved = failure_stats[failure_type]['solved']
        if total > 0:
            failure_stats[failure_type]['success_rate'] = (solved / total) * 100

    # Sort by success rate (highest to lowest)
    sorted_by_rate = sorted(
        failure_stats.items(),
        key=lambda x: x[1]['success_rate'],
        reverse=True
    )

    # Print results
    print("=" * 90)
    print("SUCCESS RATE BY FAILURE TYPE (Sorted by Success Rate)")
    print("=" * 90)
    print()
    print(f"{'Failure Type':<30} {'#Solved':<10} {'#Total':<10} {'Success Rate':<15}")
    print("-" * 90)

    for failure_type, stats in sorted_by_rate:
        print(f"{failure_type:<30} {stats['solved']:<10} {stats['total_occurrences']:<10} "
              f"{stats['success_rate']:>6.1f}%")

    print("=" * 90)
    print()

    # Key insights
    print("KEY INSIGHTS:")
    print()
    top_3 = sorted_by_rate[:3]
    print(f"Top 3 Success Rates:")
    for i, (ft, stats) in enumerate(top_3, 1):
        print(f"  {i}. {ft}: {stats['success_rate']:.1f}% ({stats['solved']}/{stats['total_occurrences']})")
    print()

    bottom_3 = [x for x in sorted_by_rate if x[1]['success_rate'] > 0][-3:]
    print(f"Bottom 3 (among solvable types):")
    for i, (ft, stats) in enumerate(bottom_3, 1):
        print(f"  {i}. {ft}: {stats['success_rate']:.1f}% ({stats['solved']}/{stats['total_occurrences']})")
    print()

    unsolvable = [(ft, stats) for ft, stats in sorted_by_rate if stats['success_rate'] == 0]
    if unsolvable:
        print(f"Unsolvable ({len(unsolvable)} types):")
        for ft, stats in unsolvable:
            print(f"  - {ft}: 0% (0/{stats['total_occurrences']})")
    print()

    # Save results
    output_data = {
        'summary': {
            'total_instances_tested': len(all_tested_ids),
            'total_successful': len(successful_ids),
            'overall_success_rate': len(successful_ids)/len(all_tested_ids)*100
        },
        'failure_type_success_rates': [
            {
                'rank': i+1,
                'failure_type': ft,
                'solved': stats['solved'],
                'total_occurrences': stats['total_occurrences'],
                'failed': stats['failed'],
                'success_rate': stats['success_rate'],
                'solved_ids': stats['solved_ids'],
                'failed_ids': stats['failed_ids']
            }
            for i, (ft, stats) in enumerate(sorted_by_rate)
        ]
    }

    output_file = OUTPUT_DIR / "failure_type_success_rates.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed results saved to: {output_file}")
    print()

    # Generate visualizations
    generate_visualizations(sorted_by_rate)

    # Generate LaTeX table
    generate_latex_table(sorted_by_rate)

    return output_data

def generate_visualizations(sorted_types):
    """Generate publication-quality visualizations."""

    # Figure 1: Success Rate Bar Chart (sorted by rate, highest at top)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Reverse order so highest rate appears at top of chart
    failure_types = [ft for ft, _ in reversed(sorted_types)]
    success_rates = [stats['success_rate'] for _, stats in reversed(sorted_types)]

    # Color by rate: green (>30%), yellow (10-30%), red (<10%)
    colors = []
    for rate in success_rates:
        if rate >= 30:
            colors.append('#2ecc71')  # Green
        elif rate >= 10:
            colors.append('#f39c12')  # Orange
        elif rate > 0:
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#95a5a6')  # Gray for 0%

    bars = ax.barh(range(len(failure_types)), success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if rate > 0:
            ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(failure_types)))
    ax.set_yticklabels(failure_types)
    ax.set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failure Type', fontsize=12, fontweight='bold')
    ax.set_title('CI Repair Success Rate by Failure Type\n(Percentage of instances successfully solved)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(success_rates) * 1.15 if success_rates else 100)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='High (≥30%)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Medium (10-30%)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Low (<10%)'),
        Patch(facecolor='#95a5a6', alpha=0.8, label='None (0%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "success_rate_by_failure_type.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Success vs Total Occurrences (Bubble chart)
    fig, ax = plt.subplots(figsize=(12, 8))

    total_counts = [stats['total_occurrences'] for _, stats in sorted_types]
    solved_counts = [stats['solved'] for _, stats in sorted_types]
    success_rates = [stats['success_rate'] for _, stats in sorted_types]

    # Bubble size based on success rate
    sizes = [rate * 10 if rate > 0 else 50 for rate in success_rates]

    scatter = ax.scatter(total_counts, solved_counts, s=sizes, c=success_rates,
                        cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5,
                        vmin=0, vmax=max(success_rates) if success_rates else 100)

    # Add labels for notable points
    for i, (ft, stats) in enumerate(sorted_types):
        if stats['success_rate'] > 20 or stats['solved'] > 5:
            ax.annotate(ft, (total_counts[i], solved_counts[i]),
                       fontsize=9, ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Total Occurrences (Pushed Instances)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Successfully Solved', fontsize=12, fontweight='bold')
    ax.set_title('Failure Type: Occurrences vs Solved\n(Bubble size = Success Rate)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "occurrence_vs_solved_bubble.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Visualizations saved:")
    print(f"  - {OUTPUT_DIR}/success_rate_by_failure_type.png")
    print(f"  - {OUTPUT_DIR}/occurrence_vs_solved_bubble.png")
    print()

def generate_latex_table(sorted_types):
    """Generate LaTeX table for ICLR paper."""

    latex_output = []
    latex_output.append("\\begin{table}[t]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{CI Repair Success Rate by Failure Type}")
    latex_output.append("\\label{tab:failure_type_success_rate}")
    latex_output.append("\\begin{tabular}{llccc}")
    latex_output.append("\\toprule")
    latex_output.append("\\textbf{Rank} & \\textbf{Failure Type} & \\textbf{\\#Solved} & \\textbf{\\#Total} & \\textbf{Success Rate} \\\\")
    latex_output.append("\\midrule")

    for rank, (failure_type, stats) in enumerate(sorted_types, 1):
        # Add visual separator for unsolvable types
        if rank > 1 and stats['success_rate'] == 0 and sorted_types[rank-2][1]['success_rate'] > 0:
            latex_output.append("\\midrule")

        latex_output.append(
            f"{rank} & {failure_type.replace('_', ' ')} & "
            f"{stats['solved']} & {stats['total_occurrences']} & "
            f"{stats['success_rate']:.1f}\\% \\\\"
        )

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")

    latex_file = OUTPUT_DIR / "failure_type_success_rate_table.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_output))

    print(f"✓ LaTeX table saved to: {latex_file}")
    print()

if __name__ == "__main__":
    try:
        results = analyze_success_rates()
        print("=" * 90)
        print("ANALYSIS COMPLETE - Ready for ICLR submission")
        print("=" * 90)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
