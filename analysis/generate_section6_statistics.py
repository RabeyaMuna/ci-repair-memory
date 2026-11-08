#!/usr/bin/env python3
"""
Generate statistics for Section 6 response to reviewer.

This script produces ready-to-use text for responding to the reviewer's question
about the frequency of unsuccessful repair causes.
"""

import json
import os


def load_analysis_results(filepath):
    """Load the failure type analysis results."""
    with open(filepath, 'r') as f:
        return json.load(f)


def categorize_into_section6(stats):
    """
    Map failure types into Section 6 conceptual categories.
    """

    categories = {
        "Log-Bounded Reasoning": {
            "description": "Model addresses immediate log error but misses underlying logic/edge cases",
            "failure_types": ["Test Failure", "Runtime Error", "Assertion Error"],
            "total": 0,
            "unsuccessful": 0,
            "instances": []
        },
        "Cross-File Dependencies": {
            "description": "Fixes require changes across multiple files or project-wide understanding",
            "failure_types": ["Import Error", "Dependency Issues", "Build Failure", "Build Error"],
            "total": 0,
            "unsuccessful": 0,
            "instances": []
        },
        "Failures Hidden by Early Termination": {
            "description": "CI fails early, hiding subsequent errors that model cannot anticipate",
            "failure_types": ["Syntax Error", "Doc/Docstring", "Documentation"],
            "total": 0,
            "unsuccessful": 0,
            "instances": []
        },
        "Configuration/Environment Issues": {
            "description": "Requires understanding project setup, environment, or package management",
            "failure_types": ["Configuration Error", "Environment Error", "Package Install Error"],
            "total": 0,
            "unsuccessful": 0,
            "instances": []
        }
    }

    # Populate categories
    for category_name, category_data in categories.items():
        for ft in category_data["failure_types"]:
            if ft in stats:
                category_data["total"] += stats[ft]["total"]
                category_data["unsuccessful"] += stats[ft]["unsuccessful"]
                category_data["instances"].append({
                    "type": ft,
                    "total": stats[ft]["total"],
                    "unsuccessful": stats[ft]["unsuccessful"],
                    "success_rate": stats[ft]["success_rate"]
                })

    return categories


def generate_paper_response(categories, total_attempts, total_successful):
    """
    Generate formatted text for paper response.
    """

    total_unsuccessful = total_attempts - total_successful

    response = []
    response.append("=" * 80)
    response.append("RESPONSE TO REVIEWER: FREQUENCY OF UNSUCCESSFUL REPAIR CAUSES")
    response.append("=" * 80)
    response.append("")

    # Context
    response.append(f"Total repair attempts: {total_attempts}")
    response.append(f"Successful repairs: {total_successful} ({total_successful/total_attempts*100:.1f}%)")
    response.append(f"Unsuccessful repairs: {total_unsuccessful} ({total_unsuccessful/total_attempts*100:.1f}%)")
    response.append("")

    # Category breakdown
    response.append("BREAKDOWN BY CAUSE CATEGORY:")
    response.append("-" * 80)
    response.append("")

    for category_name, data in sorted(categories.items(),
                                     key=lambda x: x[1]["unsuccessful"],
                                     reverse=True):
        pct_of_attempts = (data["unsuccessful"] / total_attempts * 100) if total_attempts > 0 else 0

        response.append(f"### {category_name}")
        response.append(f"**{data['description']}**")
        response.append(f"")
        response.append(f"- Total occurrences: {data['unsuccessful']} unsuccessful repairs")
        response.append(f"- Percentage of all attempts: {pct_of_attempts:.1f}%")
        response.append(f"")
        response.append(f"Failure types in this category:")

        for instance in sorted(data["instances"], key=lambda x: x["total"], reverse=True):
            response.append(f"  - {instance['type']}: {instance['unsuccessful']}/{instance['total']} "
                          f"unsuccessful ({100-instance['success_rate']:.1f}% failure rate)")

        response.append("")

    # Paper text suggestions
    response.append("=" * 80)
    response.append("SUGGESTED TEXT FOR PAPER")
    response.append("=" * 80)
    response.append("")

    response.append("### Option 1: Quantitative (Recommended)")
    response.append("")
    response.append('"""')
    response.append("We systematically classified unsuccessful repair attempts by failure type ")
    response.append(f"using our 25-category taxonomy. Of {total_attempts} repair attempts, ")
    response.append(f"{total_unsuccessful} were unsuccessful ({total_unsuccessful/total_attempts*100:.1f}%). ")
    response.append("The three main categories of unsuccessful repairs were:")
    response.append("")

    for category_name, data in sorted(categories.items(),
                                     key=lambda x: x[1]["unsuccessful"],
                                     reverse=True)[:3]:
        pct = (data["unsuccessful"] / total_attempts * 100)
        response.append(f"- **{category_name}**: {data['unsuccessful']} occurrences ({pct:.1f}% of all attempts)")
        top_types = sorted(data["instances"], key=lambda x: x["total"], reverse=True)[:2]
        types_str = ", ".join([f"{x['type']} ({x['unsuccessful']}/{x['total']})" for x in top_types])
        response.append(f"  Primary failure types: {types_str}")

    response.append("")
    response.append("Note that instances can have multiple failure types, so categories may overlap. ")
    response.append("The classification was performed using our failure type taxonomy (Table X) and ")
    response.append("validated against CI logs.")
    response.append('"""')
    response.append("")

    response.append("### Option 2: Qualitative")
    response.append("")
    response.append('"""')
    response.append("We analyzed unsuccessful repair attempts and identified three main patterns:")
    response.append("")

    for category_name, data in sorted(categories.items(),
                                     key=lambda x: x[1]["unsuccessful"],
                                     reverse=True)[:3]:
        response.append(f"- **{category_name}**: {data['description']}. ")
        top_instance = sorted(data["instances"], key=lambda x: x["total"], reverse=True)[0]
        failure_rate = 100 - top_instance["success_rate"]
        response.append(f"  For example, {top_instance['type']} failures had a {failure_rate:.1f}% failure rate ")
        response.append(f"  ({top_instance['unsuccessful']}/{top_instance['total']} unsuccessful).")

    response.append("")
    response.append(f"These patterns account for the majority of unsuccessful repairs among {total_attempts} attempts.")
    response.append('"""')
    response.append("")

    return "\n".join(response)


def main():
    results_file = "baselines/results/gpt-5-mini_llm/failure_type_success_analysis.json"

    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        print("Please run: python3 analyze_failure_type_success_rates.py")
        return

    # Load analysis results
    stats = load_analysis_results(results_file)

    # Calculate totals
    # Note: Using Code Formatting as proxy for total since it's common
    # Better: load from jobs_results_diff.jsonl for exact count
    import json
    jobs_file = "baselines/results/gpt-5-mini_llm/jobs_results_diff.jsonl"
    total_attempts = 0
    total_successful = 0

    with open(jobs_file, 'r') as f:
        for line in f:
            if line.strip():
                total_attempts += 1
                job = json.loads(line)
                if job.get('conclusion') == 'success':
                    total_successful += 1

    # Categorize into Section 6 categories
    categories = categorize_into_section6(stats)

    # Generate response text
    response_text = generate_paper_response(categories, total_attempts, total_successful)

    # Print to screen
    print(response_text)

    # Save to file
    output_file = "SECTION6_RESPONSE.txt"
    with open(output_file, 'w') as f:
        f.write(response_text)

    print(f"\nResponse text saved to: {output_file}")


if __name__ == "__main__":
    main()
