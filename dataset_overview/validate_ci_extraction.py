#!/usr/bin/env python3
"""
Validation Script for LLM-Extracted CI Contexts

Purpose:
- Manually validate a stratified sample of structured CI extractions
- Report extraction agreement/accuracy
- Identify systematic extraction errors

Usage:
    python validate_ci_extraction.py --sample-size 50 --stratify-by error_category
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
STRUCTURED_CI_PATH = PROJECT_ROOT / "log_details.json"
OUTPUT_DIR = PROJECT_ROOT / "dataset_overview"


def load_data():
    """Load dataset and structured CI contexts."""
    df = pd.read_parquet(DATASET_PATH)

    with open(STRUCTURED_CI_PATH) as f:
        ci_contexts = json.load(f)

    # Index by ID
    ci_dict = {str(ctx['id']): ctx for ctx in ci_contexts}

    return df, ci_dict


def stratified_sample(
    ci_contexts: List[Dict[str, Any]],
    n: int,
    stratify_by: str = "error_category"
) -> List[Dict[str, Any]]:
    """
    Sample instances stratified by error category.

    Ensures validation covers diverse failure types.
    """

    # Group by primary error category
    by_category = {}
    for ctx in ci_contexts:
        error_types = ctx.get("error_types", [])
        if error_types:
            category = error_types[0].get("category", "Unknown")
        else:
            category = "Unknown"

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(ctx)

    print(f"\n📊 Distribution by {stratify_by}:")
    for cat, items in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {cat}: {len(items)}")

    # Sample proportionally
    total = len(ci_contexts)
    sample = []

    for category, items in by_category.items():
        proportion = len(items) / total
        n_from_category = max(1, int(n * proportion))

        if len(items) <= n_from_category:
            sample.extend(items)
        else:
            sample.extend(random.sample(items, n_from_category))

    # If we're under target, sample more from largest categories
    if len(sample) < n:
        remaining = n - len(sample)
        largest_categories = sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True)

        for category, items in largest_categories:
            available = [item for item in items if item not in sample]
            if not available:
                continue

            to_add = min(remaining, len(available))
            sample.extend(random.sample(available, to_add))
            remaining -= to_add

            if remaining == 0:
                break

    return sample[:n]


def generate_validation_sheet(sample: List[Dict[str, Any]], output_path: Path):
    """
    Generate validation sheet for manual annotation.

    Format:
    - ID
    - Error context (LLM-extracted)
    - Failure signals (LLM-extracted)
    - Tools (LLM-extracted)
    - Affected files (LLM-extracted)
    - Raw logs snippet (for comparison)
    - Validation fields: [Correct/Incorrect/Partial] + Notes
    """

    validation_data = []

    for ctx in sample:
        instance_id = ctx.get('id')

        # LLM-extracted fields
        error_context = ctx.get("error_context", [])
        failure_signals = ctx.get("failure_signals", [])
        error_types = ctx.get("error_types", [])
        relevant_files = ctx.get("relevant_files", [])
        failed_jobs = ctx.get("failed_job", [])

        # Extract structured info
        tools = list({rf.get("failed_tool") for rf in relevant_files if rf.get("failed_tool")})
        affected_files = [rf.get("file") for rf in relevant_files if rf.get("file")]
        commands = [fj.get("command") for fj in failed_jobs if fj.get("command")]

        validation_data.append({
            "id": instance_id,
            "error_context": " ".join(error_context),
            "failure_signals": failure_signals,
            "error_categories": [et.get("category") for et in error_types],
            "error_subcategories": [et.get("subcategory") for et in error_types],
            "tools": tools,
            "commands": commands,
            "affected_files": affected_files,
            # Validation columns
            "context_correct": "",  # [Correct/Incorrect/Partial]
            "signals_correct": "",
            "categories_correct": "",
            "tools_correct": "",
            "files_correct": "",
            "overall_quality": "",  # [Excellent/Good/Fair/Poor]
            "notes": "",
        })

    # Save as JSON for easier manual editing
    with open(output_path, 'w') as f:
        json.dump(validation_data, f, indent=2)

    print(f"\n✅ Validation sheet saved to: {output_path}")
    print(f"   Total instances: {len(validation_data)}")
    print("\n📝 Manual validation steps:")
    print("   1. Open the validation JSON file")
    print("   2. For each instance, compare LLM extraction with raw logs")
    print("   3. Fill in validation fields:")
    print("      - context_correct: Correct/Incorrect/Partial")
    print("      - signals_correct: Correct/Incorrect/Partial")
    print("      - categories_correct: Correct/Incorrect/Partial")
    print("      - tools_correct: Correct/Incorrect/Partial")
    print("      - files_correct: Correct/Incorrect/Partial")
    print("      - overall_quality: Excellent/Good/Fair/Poor")
    print("      - notes: Any systematic errors or issues")
    print("   4. Run: python validate_ci_extraction.py --analyze validation_annotated.json")


def analyze_validation_results(annotated_path: Path):
    """
    Analyze completed validation annotations.

    Reports:
    - Per-field accuracy
    - Overall quality distribution
    - Common error patterns
    """

    with open(annotated_path) as f:
        annotations = json.load(f)

    # Count annotations
    fields = ["context_correct", "signals_correct", "categories_correct", "tools_correct", "files_correct"]

    print("\n" + "="*70)
    print("VALIDATION ANALYSIS RESULTS")
    print("="*70)

    print(f"\nTotal validated instances: {len(annotations)}")

    # Per-field accuracy
    print("\n📊 Per-Field Accuracy:")
    for field in fields:
        counts = Counter([ann.get(field, "").strip() for ann in annotations if ann.get(field)])

        total = sum(counts.values())
        if total == 0:
            print(f"  {field}: No annotations")
            continue

        correct = counts.get("Correct", 0)
        partial = counts.get("Partial", 0)
        incorrect = counts.get("Incorrect", 0)

        accuracy = (correct + 0.5 * partial) / total if total > 0 else 0

        print(f"  {field}:")
        print(f"    Correct: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"    Partial: {partial}/{total} ({100*partial/total:.1f}%)")
        print(f"    Incorrect: {incorrect}/{total} ({100*incorrect/total:.1f}%)")
        print(f"    Weighted accuracy: {100*accuracy:.1f}%")

    # Overall quality
    print("\n📈 Overall Quality Distribution:")
    quality_counts = Counter([ann.get("overall_quality", "").strip() for ann in annotations if ann.get("overall_quality")])
    total_quality = sum(quality_counts.values())

    for quality in ["Excellent", "Good", "Fair", "Poor"]:
        count = quality_counts.get(quality, 0)
        if total_quality > 0:
            print(f"  {quality}: {count}/{total_quality} ({100*count/total_quality:.1f}%)")

    # Error patterns
    print("\n🔍 Common Issues (from notes):")
    notes = [ann.get("notes", "").strip() for ann in annotations if ann.get("notes", "").strip()]

    if notes:
        for i, note in enumerate(notes[:10], 1):
            print(f"  {i}. {note}")
    else:
        print("  No notes provided")

    # Save summary
    summary = {
        "total_validated": len(annotations),
        "per_field_accuracy": {
            field: {
                "correct": Counter([ann.get(field) for ann in annotations]).get("Correct", 0),
                "partial": Counter([ann.get(field) for ann in annotations]).get("Partial", 0),
                "incorrect": Counter([ann.get(field) for ann in annotations]).get("Incorrect", 0),
            }
            for field in fields
        },
        "overall_quality": dict(quality_counts),
        "common_issues": notes[:10],
    }

    summary_path = OUTPUT_DIR / "validation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate LLM-extracted CI contexts")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of instances to validate (default: 50)")
    parser.add_argument("--stratify-by", default="error_category",
                        help="Stratification strategy (default: error_category)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path to annotated validation file for analysis")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    random.seed(args.seed)

    if args.analyze:
        # Analyze completed annotations
        analyze_validation_results(Path(args.analyze))
        return

    # Generate validation sheet
    print("="*70)
    print("CI CONTEXT EXTRACTION VALIDATION")
    print("="*70)

    print("\nLoading data...")
    df, ci_contexts_dict = load_data()

    ci_contexts_list = list(ci_contexts_dict.values())
    print(f"Loaded {len(ci_contexts_list)} structured CI contexts")

    print(f"\nGenerating stratified sample (n={args.sample_size})...")
    sample = stratified_sample(ci_contexts_list, args.sample_size, args.stratify_by)

    print(f"Sampled {len(sample)} instances")

    output_path = OUTPUT_DIR / "validation_sheet.json"
    generate_validation_sheet(sample, output_path)

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
