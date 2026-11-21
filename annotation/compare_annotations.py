#!/usr/bin/env python3
"""
Compare two independent annotations and calculate Cohen's Kappa.
Usage: python3 compare_annotations.py annotator1.csv annotator2.csv
       python3 compare_annotations.py annotator1.json annotator2.json
"""

import json
import sys
import csv
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict

def load_annotations(filepath):
    """Load annotations from JSON or CSV file."""
    if filepath.endswith('.csv'):
        return load_csv_annotations(filepath)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)

def load_csv_annotations(filepath):
    """Load annotations from CSV file."""
    annotations = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            instance_id = row['ID']

            # Parse failure types (comma-separated)
            types_str = row.get('Your_Failure_Types', '') or row.get('System_Generated_Failure_Types', '')
            if types_str:
                types = [t.strip() for t in types_str.split(',') if t.strip()]
            else:
                types = []

            annotations[instance_id] = types

    return annotations

def calculate_kappa(ann1, ann2):
    """Calculate Cohen's Kappa for multi-label classification."""

    # Get all failure types
    all_types = set()
    for annotations in [ann1, ann2]:
        for types in annotations.values():
            all_types.update(types)

    all_types = sorted(all_types)

    # Find common instances
    common_ids = set(ann1.keys()) & set(ann2.keys())

    if not common_ids:
        print("Error: No common instances found!")
        return None

    # Build binary labels
    labels1 = []
    labels2 = []

    for instance_id in sorted(common_ids):
        types1 = set(ann1[instance_id])
        types2 = set(ann2[instance_id])

        for failure_type in all_types:
            labels1.append(1 if failure_type in types1 else 0)
            labels2.append(1 if failure_type in types2 else 0)

    # Calculate kappa
    kappa = cohen_kappa_score(labels1, labels2)
    agreements = sum(1 for l1, l2 in zip(labels1, labels2) if l1 == l2)

    return {
        'kappa': kappa,
        'total_instances': len(common_ids),
        'total_decisions': len(labels1),
        'agreements': agreements,
        'agreement_pct': 100 * agreements / len(labels1),
        'all_types': all_types
    }

def find_disagreements(ann1, ann2):
    """Find instances where annotators disagreed."""
    common_ids = set(ann1.keys()) & set(ann2.keys())

    disagreements = []
    perfect_matches = 0

    for instance_id in sorted(common_ids):
        types1 = set(ann1[instance_id])
        types2 = set(ann2[instance_id])

        if types1 != types2:
            only_ann1 = types1 - types2
            only_ann2 = types2 - types1
            both = types1 & types2

            disagreements.append({
                'id': instance_id,
                'annotator1': sorted(types1),
                'annotator2': sorted(types2),
                'both_agree': sorted(both),
                'only_annotator1': sorted(only_ann1),
                'only_annotator2': sorted(only_ann2)
            })
        else:
            perfect_matches += 1

    return disagreements, perfect_matches

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_annotations.py annotator1.json annotator2.json")
        print("\nExample:")
        print("  python3 compare_annotations.py my_annotations.json colleague_annotations.json")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    print("="*80)
    print(" INTER-ANNOTATOR AGREEMENT ANALYSIS")
    print("="*80)

    # Load annotations
    print(f"\nLoading annotations...")
    print(f"  Annotator 1: {file1}")
    print(f"  Annotator 2: {file2}")

    ann1 = load_annotations(file1)
    ann2 = load_annotations(file2)

    print(f"\n  Annotator 1: {len(ann1)} instances")
    print(f"  Annotator 2: {len(ann2)} instances")

    # Calculate kappa
    print("\n" + "="*80)
    print(" COHEN'S KAPPA")
    print("="*80)

    result = calculate_kappa(ann1, ann2)

    if result:
        print(f"\n  Kappa Score: {result['kappa']:.4f}")

        # Interpretation
        kappa = result['kappa']
        if kappa < 0:
            interp = "Poor (less than chance)"
        elif kappa < 0.20:
            interp = "Slight"
        elif kappa < 0.40:
            interp = "Fair"
        elif kappa < 0.60:
            interp = "Moderate"
        elif kappa < 0.80:
            interp = "Substantial"
        else:
            interp = "Almost Perfect"

        print(f"  Interpretation: {interp}")

        print(f"\n  Total instances: {result['total_instances']}")
        print(f"  Total binary decisions: {result['total_decisions']}")
        print(f"  Agreements: {result['agreements']} ({result['agreement_pct']:.2f}%)")

    # Find disagreements
    print("\n" + "="*80)
    print(" INSTANCE-LEVEL ANALYSIS")
    print("="*80)

    disagreements, perfect_matches = find_disagreements(ann1, ann2)

    common_ids = set(ann1.keys()) & set(ann2.keys())
    print(f"\n  Perfect matches: {perfect_matches} ({100*perfect_matches/len(common_ids):.2f}%)")
    print(f"  Has differences: {len(disagreements)} ({100*len(disagreements)/len(common_ids):.2f}%)")

    # Show sample disagreements
    if disagreements:
        print("\n" + "="*80)
        print(" SAMPLE DISAGREEMENTS (first 10)")
        print("="*80)

        for d in disagreements[:10]:
            print(f"\n  Instance {d['id']}:")
            if d['both_agree']:
                print(f"    Both: {', '.join(d['both_agree'])}")
            if d['only_annotator1']:
                print(f"    Only Ann1: {', '.join(d['only_annotator1'])}")
            if d['only_annotator2']:
                print(f"    Only Ann2: {', '.join(d['only_annotator2'])}")

    # Save detailed results
    output_file = 'kappa_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'kappa_score': result['kappa'] if result else None,
            'agreement_statistics': result if result else {},
            'disagreements': disagreements,
            'perfect_matches': perfect_matches
        }, f, indent=2)

    print("\n" + "="*80)
    print(f"✓ Detailed results saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
