#!/usr/bin/env python3
"""
Compute Cohen's Kappa between two failure type classification files.

Usage:
  python analysis/compute_kappa.py \
    --file1 results/failure_classifications_12types.json \
    --file2 results/failure_classifications_gpt_4o_mini.json
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


TAXONOMY = [
    "Code Formatting",
    "Linting",
    "Syntax Error",
    "Runtime Error",
    "Test Failure",
    "Assertion Error",
    "Type Checking",
    "Dependency Issues",
    "Package Install Error",
    "Configuration Error",
    "Environment Error",
    "Doc/Docstring"
]


def load_classifications(file_path: str):
    """Load classifications from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    classifications = {}
    for entry in data['classifications']:
        instance_id = str(entry['issue_id'])
        failure_types = entry.get('failure_type', [])
        classifications[instance_id] = failure_types

    model = data.get('model', 'unknown')
    return classifications, model


def compute_kappa(class1, class2, model1_name, model2_name):
    """Compute Cohen's Kappa and agreement statistics."""

    common_ids = sorted(set(class1.keys()) & set(class2.keys()))

    print(f"\nDataset Overlap:")
    print(f"  {model1_name}: {len(class1)} instances")
    print(f"  {model2_name}: {len(class2)} instances")
    print(f"  Common: {len(common_ids)} instances")
    print(f"  {model1_name} only: {len(set(class1.keys()) - set(class2.keys()))}")
    print(f"  {model2_name} only: {len(set(class2.keys()) - set(class1.keys()))}")

    # Primary failure type agreement
    primary1 = []
    primary2 = []

    for iid in common_ids:
        types1 = class1[iid]
        types2 = class2[iid]

        p1 = types1[0] if types1 else "Unknown"
        p2 = types2[0] if types2 else "Unknown"

        primary1.append(p1)
        primary2.append(p2)

    kappa_primary = cohen_kappa_score(primary1, primary2)
    agreements = sum(1 for a, b in zip(primary1, primary2) if a == b)
    agreement_rate = agreements / len(common_ids)

    # Per-type binary Kappa
    kappa_per_type = {}
    for ftype in TAXONOMY:
        binary1 = [1 if ftype in class1[iid] else 0 for iid in common_ids]
        binary2 = [1 if ftype in class2[iid] else 0 for iid in common_ids]

        if len(set(binary1)) > 1 and len(set(binary2)) > 1:
            try:
                kappa = cohen_kappa_score(binary1, binary2)
                kappa_per_type[ftype] = {
                    "kappa": round(kappa, 3),
                    f"{model1_name}_count": sum(binary1),
                    f"{model2_name}_count": sum(binary2)
                }
            except:
                pass

    avg_kappa = np.mean([v['kappa'] for v in kappa_per_type.values()]) if kappa_per_type else None

    # Confusion matrix
    unique_types = sorted(set(primary1) | set(primary2))
    cm = confusion_matrix(primary1, primary2, labels=unique_types)

    cm_dict = {}
    for i, true_label in enumerate(unique_types):
        cm_dict[true_label] = {}
        for j, pred_label in enumerate(unique_types):
            if cm[i][j] > 0:
                cm_dict[true_label][pred_label] = int(cm[i][j])

    # Instance details
    instance_details = []
    for iid in common_ids:
        types1 = class1[iid]
        types2 = class2[iid]

        p1 = types1[0] if types1 else "Unknown"
        p2 = types2[0] if types2 else "Unknown"

        instance_details.append({
            "id": iid,
            f"{model1_name}_primary": p1,
            f"{model2_name}_primary": p2,
            f"{model1_name}_all": types1,
            f"{model2_name}_all": types2,
            "primary_agrees": p1 == p2
        })

    return {
        "total_instances": len(common_ids),
        "primary_failure_type": {
            "kappa": round(kappa_primary, 3),
            "agreement_rate": round(agreement_rate, 3),
            "num_agreements": agreements,
            "confusion_matrix": cm_dict
        },
        "per_type_binary": {
            "average_kappa": round(avg_kappa, 3) if avg_kappa else None,
            "individual_types": kappa_per_type
        },
        "instance_details": instance_details
    }


def interpret_kappa(kappa):
    """Interpret Kappa according to Landis & Koch (1977)."""
    if kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compute Cohen\'s Kappa between two classification files')
    parser.add_argument('--file1', type=str, required=True,
                        help='First classification file (Annotator 1)')
    parser.add_argument('--file2', type=str, required=True,
                        help='Second classification file (Annotator 2)')
    parser.add_argument('--output', type=str, default='analysis/cohen_kappa_results.json',
                        help='Output file for results')
    args = parser.parse_args()

    print("="*80)
    print("COHEN'S KAPPA INTER-ANNOTATOR AGREEMENT")
    print("="*80)

    # Load classifications
    print(f"\nLoading {args.file1}...")
    class1, model1 = load_classifications(args.file1)
    print(f"  Model: {model1}")

    print(f"\nLoading {args.file2}...")
    class2, model2 = load_classifications(args.file2)
    print(f"  Model: {model2}")

    # Compute Kappa
    results = compute_kappa(class1, class2, model1, model2)

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal common instances: {results['total_instances']}")
    print("\nPRIMARY FAILURE TYPE AGREEMENT:")
    print(f"  Cohen's Kappa:  κ = {results['primary_failure_type']['kappa']}")
    print(f"  Raw Agreement:  {results['primary_failure_type']['agreement_rate']:.1%}")
    print(f"  Num Agreements: {results['primary_failure_type']['num_agreements']}/{results['total_instances']}")

    kappa_val = results['primary_failure_type']['kappa']
    interpretation = interpret_kappa(kappa_val)
    print(f"  Interpretation: {interpretation} agreement (Landis & Koch, 1977)")

    if results['per_type_binary']['average_kappa']:
        print(f"\nAVERAGE KAPPA (PER-TYPE BINARY): {results['per_type_binary']['average_kappa']}")

    # Show disagreements
    disagreements = [inst for inst in results['instance_details'] if not inst['primary_agrees']]
    agreements_list = [inst for inst in results['instance_details'] if inst['primary_agrees']]

    print(f"\nAGREEMENT BREAKDOWN:")
    print(f"  Agreements:     {len(agreements_list)} instances")
    print(f"  Disagreements:  {len(disagreements)} instances")

    if disagreements:
        print(f"\nSAMPLE DISAGREEMENTS (first 10):")
        for inst in disagreements[:10]:
            print(f"  ID {inst['id']}:")
            print(f"    {model1}: {inst[f'{model1}_primary']}")
            print(f"    {model2}: {inst[f'{model2}_primary']}")
        if len(disagreements) > 10:
            print(f"  ... and {len(disagreements) - 10} more")

    # Per-type statistics
    print(f"\nPER-TYPE KAPPA (sorted by Kappa):")
    sorted_types = sorted(
        results['per_type_binary']['individual_types'].items(),
        key=lambda x: x[1].get('kappa', -1),
        reverse=True
    )
    for ftype, stats in sorted_types[:10]:
        m1_count = stats.get(f'{model1}_count', 0)
        m2_count = stats.get(f'{model2}_count', 0)
        print(f"  {ftype:25s} κ={stats['kappa']:.3f}  ({model1}:{m1_count:3d}, {model2}:{m2_count:3d})")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_data = {
        "metadata": {
            "annotator1": model1,
            "annotator1_file": args.file1,
            "annotator2": model2,
            "annotator2_file": args.file2
        },
        "kappa_analysis": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("\n" + "="*80)
    print("FOR PAPER (Section 3.2, Step 5):")
    print("="*80)
    print(f"\n  Two independent annotators classified {results['total_instances']} instances,")
    print(f"  achieving Cohen's Kappa of κ = {kappa_val}, indicating")
    print(f"  {interpretation.lower()} agreement (Landis & Koch, 1977).")
    print()


if __name__ == "__main__":
    main()
