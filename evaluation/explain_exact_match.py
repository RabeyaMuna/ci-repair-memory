#!/usr/bin/env python3
"""
Explain why Exact Match is low while Top-K can be high.
"""

def example_1():
    """Example showing why Top-1 can be high but Exact Match low."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Why Top-1 (50%) > Exact Match (6.25%)")
    print("="*80)

    examples = [
        {
            "id": "104",
            "predicted": ["logger.py", "utils.py", "config.py", "main.py", "test.py", "helper.py", "parser.py", "validator.py", "formatter.py", "handler.py"],
            "gt": ["logger.py"],
        },
        {
            "id": "107",
            "predicted": ["main.py", "wrong.py"],
            "gt": ["main.py"],
        },
        {
            "id": "117",
            "predicted": ["config.py", "wrong1.py", "wrong2.py", "wrong3.py", "wrong4.py"],
            "gt": ["config.py"],
        },
        {
            "id": "108",
            "predicted": ["logger.py", "config.py"],
            "gt": ["logger.py"],
        },
    ]

    print("\nLet's look at 4 instances where GT file is in Position 1:")
    print()

    for ex in examples:
        print(f"Instance {ex['id']}:")
        print(f"  Predicted (top 3): {ex['predicted'][:3]}")
        print(f"  Ground Truth: {ex['gt']}")
        print(f"  Position 1: {ex['predicted'][0]}")

        # Check metrics
        top1 = ex['predicted'][0] in ex['gt']
        exact = set(ex['predicted']) == set(ex['gt'])

        print(f"  ✓ Top-1 Hit: {top1} (position 1 is in GT)")
        print(f"  {'✓' if exact else '✗'} Exact Match: {exact} (sets {'equal' if exact else 'NOT equal'})")

        if not exact:
            extra = set(ex['predicted']) - set(ex['gt'])
            print(f"    Problem: {len(extra)} extra files predicted: {list(extra)[:3]}...")

        print()

    print("RESULT:")
    print("  - 4/4 instances have Top-1 hit (100%)")
    print("  - 0/4 instances have Exact Match (0%)")
    print("  - Why? Because they predicted extra files beyond GT!")
    print()


def example_2():
    """Show the relationship between metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Relationship Between Metrics")
    print("="*80)

    print("\nMetric Strictness (from least to most strict):")
    print()
    print("1. Top-K Hit (LEAST STRICT)")
    print("   - Only needs ONE GT file in top K positions")
    print("   - Example: Predicted 100 files, 1 is GT at position 1 → Top-1 Hit ✓")
    print()
    print("2. Precision (MEDIUM STRICT)")
    print("   - Measures what % of predictions are correct")
    print("   - Example: Predicted 10 files, 5 are GT → 50% precision")
    print()
    print("3. Exact Match (MOST STRICT)")
    print("   - Requires predicted SET == GT SET exactly")
    print("   - Example: Predicted {A, B, C}, GT {A} → No match (extra B, C)")
    print("   - Example: Predicted {A}, GT {A, B} → No match (missing B)")
    print("   - Example: Predicted {A, B}, GT {A, B} → Exact match ✓")
    print()


def show_your_data():
    """Show actual distribution from your data."""
    print("\n" + "="*80)
    print("YOUR ACTUAL DATA ANALYSIS")
    print("="*80)

    print("\nYou have 16 instances with predictions:")
    print()

    # Categorize instances
    categories = {
        "exact_match": ["108"],  # Only 1 instance
        "top1_but_not_exact": ["104", "105", "107", "114", "117", "118", "126"],  # 7 instances
        "top3_but_not_top1": ["110", "113", "115", "116", "122"],  # 5 instances
        "never_hit": ["103", "106"],  # 2 instances
    }

    print("1. EXACT MATCH (1 instance = 6.25%)")
    print("   - Instance 108: Predicted exactly right")
    print()

    print("2. TOP-1 HIT but NOT EXACT (7 instances)")
    print("   - These have GT in position 1")
    print("   - But predicted extra files → no exact match")
    print("   - Examples: 104, 105, 107, 114, 117, 118, 126")
    print("   - This is why Top-1 (50%) > Exact Match (6.25%)!")
    print()

    print("3. TOP-3 HIT but NOT TOP-1 (5 instances)")
    print("   - GT file is in position 2 or 3")
    print("   - Examples: 110, 113, 115, 116, 122")
    print()

    print("4. NEVER HIT (2 instances)")
    print("   - No GT file in any position")
    print("   - Examples: 103, 106")
    print()

    print("BREAKDOWN:")
    print("  Total instances: 16")
    print("  - Exact Match: 1 (6.25%)")
    print("  - Top-1 Hit: 8 (50%) = 1 exact + 7 with extras")
    print("  - Top-3 Hit: 13 (81.25%) = 8 top-1 + 5 more")
    print()
    print("This shows: Top-K can be high even when Exact Match is low!")
    print()


def concrete_example():
    """Most concrete example."""
    print("\n" + "="*80)
    print("CONCRETE EXAMPLE: Instance 104")
    print("="*80)

    predicted = [
        "logger.py",      # Position 1 ← GT file!
        "utils.py",       # Position 2
        "config.py",      # Position 3
        "main.py",        # Position 4
        "test.py",        # Position 5
        "helper.py",      # Position 6
        "parser.py",      # Position 7
        "validator.py",   # Position 8
        "formatter.py",   # Position 9
        "handler.py",     # Position 10
    ]

    gt = ["logger.py"]

    print(f"\nPredicted {len(predicted)} files:")
    for i, f in enumerate(predicted, 1):
        marker = "← GT FILE!" if f in gt else ""
        print(f"  {i:2d}. {f:20s} {marker}")

    print(f"\nGround Truth: {gt}")

    print("\nMetric Calculations:")
    print("-" * 80)

    # Top-1
    print("\n✓ Top-1 Hit: YES")
    print(f"  - Position 1 is '{predicted[0]}'")
    print(f"  - Is it in GT? YES")
    print(f"  - Contributes to 50% Top-1 hit rate")

    # Exact Match
    pred_set = set(predicted)
    gt_set = set(gt)
    print("\n✗ Exact Match: NO")
    print(f"  - Predicted set has {len(pred_set)} files")
    print(f"  - GT set has {len(gt_set)} file")
    print(f"  - Are they equal? NO")
    print(f"  - Problem: Predicted {len(pred_set - gt_set)} extra files")
    print(f"  - Does NOT contribute to exact match rate")

    # Precision
    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set)
    print(f"\nPrecision: {precision:.2%}")
    print(f"  - Correct predictions: {len(intersection)}")
    print(f"  - Total predictions: {len(pred_set)}")
    print(f"  - Only 10% of predictions were correct!")

    print("\n" + "="*80)
    print("CONCLUSION: High Top-1 (✓) but Low Exact Match (✗) and Low Precision (10%)")
    print("="*80)


def main():
    print("\n" + "="*80)
    print("WHY IS EXACT MATCH LOW WHEN TOP-K IS HIGH?")
    print("="*80)

    example_1()
    example_2()
    show_your_data()
    concrete_example()

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
1. EXACT MATCH IS THE STRICTEST METRIC
   - Requires predicted set == GT set (no extra, no missing files)
   - Your result: 6.25% (1/16) - Very low!

2. TOP-K IS MORE LENIENT
   - Only needs ONE GT file in top K
   - Your result: 50% Top-1, 81.25% Top-3 - Much higher!

3. WHY THE DIFFERENCE?
   - 7 instances have GT in position 1 BUT predicted extra files
   - These count for Top-1 (50%) but NOT for Exact Match (6.25%)

4. THIS IS NORMAL AND EXPECTED!
   - Exact Match should be ≤ Top-K
   - Your ordering: Exact Match (6.25%) < Top-1 (50%) < Top-3 (81.25%) ✓ Correct!

5. YOUR PROBLEM:
   - You're over-predicting (too many files)
   - Example: Instance 104 predicted 10 files, only 1 is GT
   - Example: Instance 105 predicted 85 files, only 1 is GT
   - This gives good Top-K but terrible Exact Match and Precision
""")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
