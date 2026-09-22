#!/usr/bin/env python3
"""
Simple Analysis: Which Failure Types Were Successfully Solved?

Counts only SUCCESSFUL instances and shows which failure types were solved most.
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
SUCCESS_RESULTS = RESULTS_DIR / "jobs_success_diff.jsonl"

def load_successful_instances():
    """Load only successful instances."""
    successful_ids = set()

    with open(SUCCESS_RESULTS, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                successful_ids.add(result['id'])

    return successful_ids

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

def analyze_solved_failure_types():
    """Main analysis - count successful instances per failure type."""

    print("=" * 80)
    print("FAILURE TYPES SOLVED (Based on Successful Instances)")
    print("=" * 80)
    print()

    # Load data
    successful_ids = load_successful_instances()
    classifications, taxonomy = load_failure_classifications()

    print(f"Total successful instances: {len(successful_ids)}")
    print()

    # Count successful instances per failure type
    failure_type_solved = defaultdict(lambda: {
        'count': 0,
        'instance_ids': []
    })

    for issue_id in successful_ids:
        if issue_id in classifications:
            failure_types = classifications[issue_id]
            for failure_type in failure_types:
                failure_type_solved[failure_type]['count'] += 1
                failure_type_solved[failure_type]['instance_ids'].append(issue_id)

    # Sort by count (highest to lowest)
    sorted_types = sorted(
        failure_type_solved.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    # Print results
    print("=" * 80)
    print(f"{'Failure Type':<30} {'# Solved':<15} {'% of Successes':<15}")
    print("-" * 80)

    for failure_type, data in sorted_types:
        count = data['count']
        percentage = (count / len(successful_ids)) * 100
        print(f"{failure_type:<30} {count:<15} {percentage:>6.1f}%")

    print("=" * 80)
    print()

    # Summary
    print("SUMMARY:")
    print(f"✓ {len(sorted_types)} failure types were successfully solved")
    print(f"✓ Top 3 most solved: {', '.join([ft for ft, _ in sorted_types[:3]])}")

    # Find failure types not solved
    all_failure_types = set(taxonomy.keys())
    solved_types = set(failure_type_solved.keys())
    unsolved_types = all_failure_types - solved_types

    if unsolved_types:
        print(f"✗ {len(unsolved_types)} failure types with 0 successful repairs:")
        for ft in sorted(unsolved_types):
            print(f"  - {ft}")
    print()

    # Save detailed results
    output_data = {
        'total_successful_instances': len(successful_ids),
        'total_failure_types_solved': len(sorted_types),
        'failure_types_solved': {
            failure_type: {
                'solved_count': data['count'],
                'percentage_of_successes': (data['count'] / len(successful_ids)) * 100,
                'instance_ids': data['instance_ids']
            }
            for failure_type, data in sorted_types
        },
        'unsolved_failure_types': sorted(list(unsolved_types)),
        'ranking': [
            {
                'rank': i+1,
                'failure_type': ft,
                'solved_count': data['count']
            }
            for i, (ft, data) in enumerate(sorted_types)
        ]
    }

    output_file = OUTPUT_DIR / "solved_failure_types.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Detailed results saved to: {output_file}")
    print()

    # Generate visualizations
    generate_visualizations(sorted_types, len(successful_ids), unsolved_types)

    # Generate simple table
    generate_simple_table(sorted_types, len(successful_ids))

    return output_data

def generate_visualizations(sorted_types, total_success, unsolved_types):
    """Generate simple visualizations."""

    # Figure 1: Bar chart of solved count (highest to lowest)
    fig, ax = plt.subplots(figsize=(12, 8))

    failure_types = [ft for ft, _ in sorted_types]
    solved_counts = [data['count'] for _, data in sorted_types]

    colors = ['#2ecc71' for _ in sorted_types]

    bars = ax.barh(range(len(failure_types)), solved_counts, color=colors, alpha=0.8)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, solved_counts)):
        ax.text(count + 0.3, i, str(count), va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(failure_types)))
    ax.set_yticklabels(failure_types)
    ax.set_xlabel('Number of Successfully Solved Instances', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failure Type', fontsize=12, fontweight='bold')
    ax.set_title(f'Failure Types Successfully Solved (Total: {total_success} successful instances)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(solved_counts) * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "solved_failure_types_ranked.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Top 5 failure types with percentages
    fig, ax = plt.subplots(figsize=(10, 6))

    top_5 = sorted_types[:5]
    top_types = [ft for ft, _ in top_5]
    top_counts = [data['count'] for _, data in top_5]
    top_percentages = [(count / total_success) * 100 for count in top_counts]

    bars = ax.bar(range(len(top_types)), top_counts, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, count, pct) in enumerate(zip(bars, top_counts, top_percentages)):
        ax.text(i, count + 0.5, f'{count}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(top_types)))
    ax.set_xticklabels(top_types, rotation=15, ha='right')
    ax.set_ylabel('Number of Solved Instances', fontsize=12, fontweight='bold')
    ax.set_title('Top 5 Most Solved Failure Types', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(top_counts) * 1.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_5_solved_failure_types.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Visualizations saved:")
    print(f"  - {OUTPUT_DIR}/solved_failure_types_ranked.png")
    print(f"  - {OUTPUT_DIR}/top_5_solved_failure_types.png")
    print()

def generate_simple_table(sorted_types, total_success):
    """Generate simple LaTeX table."""

    latex_output = []
    latex_output.append("\\begin{table}[h]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Failure Types Successfully Solved (Ranked by Count)}")
    latex_output.append("\\label{tab:solved_failure_types}")
    latex_output.append("\\begin{tabular}{clcc}")
    latex_output.append("\\hline")
    latex_output.append("\\textbf{Rank} & \\textbf{Failure Type} & \\textbf{\\# Solved} & \\textbf{\\% of Successes} \\\\")
    latex_output.append("\\hline")

    for rank, (failure_type, data) in enumerate(sorted_types, 1):
        count = data['count']
        percentage = (count / total_success) * 100
        latex_output.append(
            f"{rank} & {failure_type.replace('_', ' ')} & {count} & {percentage:.1f}\\% \\\\"
        )

    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")

    latex_file = OUTPUT_DIR / "solved_failure_types_table.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_output))

    print(f"✓ LaTeX table saved to: {latex_file}")
    print()

if __name__ == "__main__":
    try:
        results = analyze_solved_failure_types()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
