#!/usr/bin/env python3
"""
Analyze CI Repair Performance by Failure Type

This script evaluates:
1. Which failure types have at least one successful repair
2. Success rate per failure type
3. Distribution of successes across failure types
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend

# Set up paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "paper"
OUTPUT_DIR.mkdir(exist_ok=True)

# Input files
FAILURE_CLASSIFICATIONS = RESULTS_DIR / "failure_classifications_12types.json"
SUCCESS_RESULTS = RESULTS_DIR / "jobs_success_diff.jsonl"
ALL_RESULTS = RESULTS_DIR / "jobs_results_diff.jsonl"

def load_failure_classifications():
    """Load failure type classifications for each issue."""
    with open(FAILURE_CLASSIFICATIONS, 'r') as f:
        data = json.load(f)

    # Create mapping: issue_id -> [failure_types]
    classifications = {}
    for item in data['classifications']:
        issue_id = item['issue_id']
        failure_types = item['failure_type']
        # Ensure failure_types is a list
        if isinstance(failure_types, str):
            failure_types = [failure_types]
        classifications[issue_id] = failure_types

    return classifications, data['taxonomy']

def load_benchmark_results():
    """Load benchmark results."""
    successful_ids = set()
    failed_ids = set()

    # Load all results to categorize
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

def analyze_failure_type_performance():
    """Main analysis function."""

    print("=" * 80)
    print("CI REPAIR PERFORMANCE BY FAILURE TYPE")
    print("=" * 80)
    print()

    # Load data
    classifications, taxonomy = load_failure_classifications()
    successful_ids, failed_ids = load_benchmark_results()

    all_tested_ids = successful_ids | failed_ids

    print(f"Total instances tested: {len(all_tested_ids)}")
    print(f"Total successful: {len(successful_ids)}")
    print(f"Total failed: {len(failed_ids)}")
    print(f"Overall success rate: {len(successful_ids)/len(all_tested_ids)*100:.1f}%")
    print()

    # Track statistics per failure type
    failure_stats = defaultdict(lambda: {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'success_rate': 0.0,
        'can_solve': False,
        'successful_ids': [],
        'failed_ids': []
    })

    # Count occurrences and successes
    for issue_id, failure_types in classifications.items():
        if issue_id not in all_tested_ids:
            continue  # Skip if not tested in this benchmark run

        is_successful = issue_id in successful_ids

        for failure_type in failure_types:
            failure_stats[failure_type]['total'] += 1

            if is_successful:
                failure_stats[failure_type]['successful'] += 1
                failure_stats[failure_type]['successful_ids'].append(issue_id)
                failure_stats[failure_type]['can_solve'] = True
            else:
                failure_stats[failure_type]['failed'] += 1
                failure_stats[failure_type]['failed_ids'].append(issue_id)

    # Calculate success rates
    for failure_type in failure_stats:
        total = failure_stats[failure_type]['total']
        successful = failure_stats[failure_type]['successful']
        if total > 0:
            failure_stats[failure_type]['success_rate'] = (successful / total) * 100

    # Sort by success count (descending), then by total occurrences
    sorted_types = sorted(
        failure_stats.items(),
        key=lambda x: (x[1]['successful'], x[1]['total']),
        reverse=True
    )

    # Print results
    print("=" * 80)
    print("PERFORMANCE BY FAILURE TYPE")
    print("=" * 80)
    print()
    print(f"{'Failure Type':<30} {'#Occur.':<10} {'#Success':<10} {'#Fail':<10} {'Rate':<10} {'Solvable':<10}")
    print("-" * 90)

    solvable_types = []
    unsolvable_types = []

    for failure_type, stats in sorted_types:
        solvable = "✓ Yes" if stats['can_solve'] else "✗ No"
        if stats['can_solve']:
            solvable_types.append(failure_type)
        else:
            unsolvable_types.append(failure_type)

        print(f"{failure_type:<30} {stats['total']:<10} {stats['successful']:<10} "
              f"{stats['failed']:<10} {stats['success_rate']:<9.1f}% {solvable:<10}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total unique failure types: {len(failure_stats)}")
    print(f"Solvable failure types (≥1 success): {len(solvable_types)}")
    print(f"Unsolvable failure types (0 success): {len(unsolvable_types)}")
    print(f"Solvability rate: {len(solvable_types)/len(failure_stats)*100:.1f}%")
    print()

    print("Solvable failure types:")
    for ft in solvable_types:
        print(f"  ✓ {ft}")
    print()

    if unsolvable_types:
        print("Unsolvable failure types:")
        for ft in unsolvable_types:
            print(f"  ✗ {ft}")
        print()

    # Save detailed results
    output_data = {
        'summary': {
            'total_instances_tested': len(all_tested_ids),
            'total_successful': len(successful_ids),
            'total_failed': len(failed_ids),
            'overall_success_rate': len(successful_ids)/len(all_tested_ids)*100,
            'total_failure_types': len(failure_stats),
            'solvable_failure_types': len(solvable_types),
            'unsolvable_failure_types': len(unsolvable_types),
            'solvability_rate': len(solvable_types)/len(failure_stats)*100
        },
        'failure_type_performance': {
            failure_type: {
                'total_occurrences': stats['total'],
                'successful': stats['successful'],
                'failed': stats['failed'],
                'success_rate': stats['success_rate'],
                'can_solve': stats['can_solve'],
                'successful_ids': stats['successful_ids'],
                'failed_ids': stats['failed_ids']
            }
            for failure_type, stats in sorted_types
        },
        'solvable_types': solvable_types,
        'unsolvable_types': unsolvable_types,
        'taxonomy': taxonomy
    }

    output_file = OUTPUT_DIR / "failure_type_performance_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed results saved to: {output_file}")
    print()

    # Generate visualizations
    generate_visualizations(failure_stats, sorted_types, output_data['summary'])

    # Generate LaTeX table
    generate_latex_table(sorted_types)

    return output_data

def generate_visualizations(failure_stats, sorted_types, summary):
    """Generate visualizations for the paper."""

    # Figure 1: Success vs Failure by Failure Type (Horizontal Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 8))

    failure_types = [ft for ft, _ in sorted_types]
    successful_counts = [stats['successful'] for _, stats in sorted_types]
    failed_counts = [stats['failed'] for _, stats in sorted_types]

    y_pos = range(len(failure_types))

    # Create horizontal stacked bar chart
    ax.barh(y_pos, successful_counts, label='Successful', color='#2ecc71', alpha=0.8)
    ax.barh(y_pos, failed_counts, left=successful_counts, label='Failed', color='#e74c3c', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(failure_types)
    ax.set_xlabel('Number of Instances', fontsize=12)
    ax.set_ylabel('Failure Type', fontsize=12)
    ax.set_title('CI Repair Performance by Failure Type', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_type_performance_stacked.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Success Rate by Failure Type
    fig, ax = plt.subplots(figsize=(12, 8))

    success_rates = [stats['success_rate'] for _, stats in sorted_types]
    colors = ['#2ecc71' if rate > 0 else '#e74c3c' for rate in success_rates]

    bars = ax.barh(y_pos, success_rates, color=colors, alpha=0.7)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        if rate > 0:
            ax.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(failure_types)
    ax.set_xlabel('Success Rate (%)', fontsize=12)
    ax.set_ylabel('Failure Type', fontsize=12)
    ax.set_title('Success Rate by Failure Type', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(success_rates) * 1.15 if success_rates else 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_type_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Solvability Pie Chart
    fig, ax = plt.subplots(figsize=(8, 8))

    solvable_count = summary['solvable_failure_types']
    unsolvable_count = summary['unsolvable_failure_types']

    sizes = [solvable_count, unsolvable_count]
    labels = [f'Solvable\n({solvable_count} types)', f'Unsolvable\n({unsolvable_count} types)']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, explode=explode, textprops={'fontsize': 12})
    ax.set_title('Failure Type Solvability\n(At least one successful repair)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_type_solvability.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Occurrence vs Success Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    total_counts = [stats['total'] for _, stats in sorted_types]
    successful_counts = [stats['successful'] for _, stats in sorted_types]

    # Color by solvability
    colors = ['#2ecc71' if stats['can_solve'] else '#e74c3c'
              for _, stats in sorted_types]

    scatter = ax.scatter(total_counts, successful_counts, s=200, c=colors,
                        alpha=0.6, edgecolors='black', linewidth=1)

    # Add labels
    for i, (ft, _) in enumerate(sorted_types):
        # Only label points with successes or high occurrence
        if successful_counts[i] > 0 or total_counts[i] > 10:
            ax.annotate(ft, (total_counts[i], successful_counts[i]),
                       fontsize=8, ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Total Occurrences', fontsize=12)
    ax.set_ylabel('Successful Repairs', fontsize=12)
    ax.set_title('Failure Type Occurrence vs Successful Repairs',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.6, label='Solvable (≥1 success)'),
        Patch(facecolor='#e74c3c', alpha=0.6, label='Unsolvable (0 success)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "failure_type_occurrence_vs_success.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Visualizations saved:")
    print(f"  - {OUTPUT_DIR}/failure_type_performance_stacked.png")
    print(f"  - {OUTPUT_DIR}/failure_type_success_rate.png")
    print(f"  - {OUTPUT_DIR}/failure_type_solvability.png")
    print(f"  - {OUTPUT_DIR}/failure_type_occurrence_vs_success.png")
    print()

def generate_latex_table(sorted_types):
    """Generate LaTeX table for the paper."""

    latex_output = []
    latex_output.append("\\begin{table}[h]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{CI Repair Performance by Failure Type}")
    latex_output.append("\\label{tab:failure_type_performance}")
    latex_output.append("\\begin{tabular}{lcccc}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Failure Type} & \\textbf{\\#Occur.} & \\textbf{\\#Success} & \\textbf{Success Rate} & \\textbf{Solvable} \\\\")
    latex_output.append("\\hline")

    for failure_type, stats in sorted_types:
        solvable = "\\checkmark" if stats['can_solve'] else "\\texttimes"
        latex_output.append(
            f"{failure_type.replace('_', ' ')} & "
            f"{stats['total']} & "
            f"{stats['successful']} & "
            f"{stats['success_rate']:.1f}\\% & "
            f"{solvable} \\\\"
        )

    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")

    latex_file = OUTPUT_DIR / "failure_type_performance_table.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_output))

    print(f"✓ LaTeX table saved to: {latex_file}")
    print()

if __name__ == "__main__":
    try:
        results = analyze_failure_type_performance()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
