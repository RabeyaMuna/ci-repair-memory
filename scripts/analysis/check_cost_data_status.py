#!/usr/bin/env python3
"""
check_cost_data_status.py

Quick status checker to see which experimental runs have token_report.json available.
Helps you understand what data you have before running the cost analysis.

Usage:
    python scripts/analysis/check_cost_data_status.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "baselines" / "results"

CONFIGS = [
    {
        "name": "Baseline (No Memory)",
        "dir": "MiniMax-M2.5_llm_baseline",
        "memory": "none"
    },
    {
        "name": "Backward (L1 only)",
        "dir": "MiniMax-M2.5_llm_memory_L1",
        "memory": "L1"
    },
    {
        "name": "Forward (L1+L2)",
        "dir": "MiniMax-M2.5_llm_memory_L1L2",
        "memory": "L1+L2"
    },
    {
        "name": "Bidirectional (L1+L2+L3)",
        "dir": "MiniMax-M2.5_llm_memory",
        "memory": "L1+L2+L3"
    }
]


def check_file_status(config_dir: Path, filename: str) -> tuple[bool, str]:
    """Check if a file exists and return status info."""
    filepath = config_dir / filename

    if not filepath.exists():
        return False, "missing"

    if filepath.stat().st_size == 0:
        return False, "empty"

    if filename.endswith('.json'):
        try:
            with open(filepath) as f:
                data = json.load(f)
                if filename == 'token_report.json':
                    num_tasks = len(data.get('per_task', []))
                    total_cost = data.get('llm', {}).get('overall', {}).get('cost_usd', 0)
                    return True, f"✓ ({num_tasks} tasks, ${total_cost:.4f})"
                elif filename == 'fault_localization.json':
                    return True, f"✓ ({len(data)} entries)"
                elif filename == 'generated_patches.json':
                    return True, f"✓ ({len(data)} patches)"
                else:
                    return True, "✓"
        except Exception as e:
            return False, f"error: {str(e)[:30]}"

    return True, "✓"


def main():
    print("\n" + "=" * 90)
    print("  COST DATA STATUS CHECK")
    print("=" * 90)
    print()

    if not RESULTS_DIR.exists():
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        print(f"\nExpected directory structure:")
        print(f"  {RESULTS_DIR}/")
        for config in CONFIGS:
            print(f"    └── {config['dir']}/")
            print(f"        ├── token_report.json")
            print(f"        ├── fault_localization.json")
            print(f"        └── generated_patches.json")
        print()
        return 1

    print(f"Results directory: {RESULTS_DIR}\n")

    all_complete = True
    complete_count = 0

    for config in CONFIGS:
        config_dir = RESULTS_DIR / config['dir']
        print(f"[ {config['name']} ]")
        print(f"  Directory: {config['dir']}")

        if not config_dir.exists():
            print(f"  Status: ❌ Directory not found")
            print(f"  Action: Run the experiment for {config['memory']} configuration")
            all_complete = False
            print()
            continue

        # Check key files
        files_to_check = [
            ('token_report.json', 'Token usage & cost data'),
            ('fault_localization.json', 'Fault localization results'),
            ('generated_patches.json', 'Generated patches'),
            ('fl_evaluation.json', 'FL evaluation metrics'),
        ]

        config_complete = True
        for filename, description in files_to_check:
            exists, status = check_file_status(config_dir, filename)

            if filename == 'token_report.json':
                if not exists:
                    config_complete = False
                    all_complete = False
                symbol = "✓" if exists else "❌"
                print(f"  {symbol} {filename:<30} {description:<30} {status}")
            else:
                symbol = "✓" if exists else "·"
                print(f"  {symbol} {filename:<30} {description:<30} {status}")

        if config_complete:
            complete_count += 1
            print(f"  Status: ✅ READY for cost analysis")
        else:
            print(f"  Status: ⚠️  Missing token_report.json - run experiment first")

        print()

    # Summary
    print("=" * 90)
    print(f"Summary: {complete_count}/{len(CONFIGS)} configurations have cost data available")
    print("=" * 90)
    print()

    if all_complete:
        print("✅ All configurations complete!")
        print("\nNext steps:")
        print("  1. Run cost analysis:")
        print("     python scripts/analysis/calculate_memory_ablation_costs.py")
        print()
        print("  2. Generate LaTeX table:")
        print("     python scripts/analysis/calculate_memory_ablation_costs.py --latex results/table.tex")
        print()
    elif complete_count > 0:
        print(f"⚠️  Partial data available ({complete_count}/{len(CONFIGS)} configs)")
        print("\nYou can:")
        print("  1. Run cost analysis on available data:")
        print("     python scripts/analysis/calculate_memory_ablation_costs.py")
        print()
        print("  2. Or complete missing experiments:")
        for config in CONFIGS:
            config_dir = RESULTS_DIR / config['dir']
            if not (config_dir / 'token_report.json').exists():
                if config['memory'] == 'none':
                    print(f"     baselines/.venv/bin/python baselines/main.py --model-key MiniMax-M2.5")
                else:
                    print(f"     baselines/.venv/bin/python baselines/scripts/run_ablation_from_log_details.py --ablation-levels {config['memory']}")
        print()
    else:
        print("❌ No cost data found")
        print("\nTo generate cost data:")
        print("  1. Run baseline experiment:")
        print("     cd baselines")
        print("     .venv/bin/python main.py --model-key MiniMax-M2.5")
        print()
        print("  2. Then run memory ablations:")
        print("     .venv/bin/python scripts/run_ablation_from_log_details.py --ablation-levels L1")
        print("     .venv/bin/python scripts/run_ablation_from_log_details.py --ablation-levels L1+L2")
        print("     .venv/bin/python scripts/run_ablation_from_log_details.py --ablation-levels L1+L2+L3")
        print()
        print("  OR use estimates:")
        print("     python scripts/analysis/estimate_memory_costs.py --instances 410")
        print()

    return 0 if all_complete else 1


if __name__ == "__main__":
    exit(main())
