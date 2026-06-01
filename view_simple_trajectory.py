#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Trajectory Viewer

View trajectory.json files.

Usage:
    python view_simple_trajectory.py results/trajectories/gpt-4o-mini/baseline/123/trajectory.json
    python view_simple_trajectory.py --model gpt-4o-mini --task-id 123
    python view_simple_trajectory.py --model gpt-4o-mini --task-id 123 --compare
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines"))
from utilities.load_config import load_config


def view_trajectory(filepath: str):
    """View a single trajectory"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        traj = json.load(f)

    print("\n" + "=" * 80)
    print(f"TRAJECTORY: {traj['task_id']}")
    print("=" * 80)

    print(f"\nRepo: {traj['repo_name']}")
    print(f"SHA: {traj['sha_fail']}")
    print(f"Model: {traj['model']}")
    print(f"Condition: {traj['condition']}" + (f" ({traj['level']})" if traj.get('level') else ""))
    print(f"Duration: {traj.get('duration_sec', 0):.2f}s")
    print(f"Patch Generated: {'Yes ✓' if traj.get('patch_generated') else 'No ✗'}")

    tokens = traj.get('total_tokens', {})
    if tokens:
        print(f"Tokens: {tokens.get('input', 0):,} in / {tokens.get('output', 0):,} out")

    cost = traj.get('cost_usd', 0)
    if cost > 0:
        print(f"Cost: ${cost:.4f}")

    print("\n" + "-" * 80)
    print("STEPS")
    print("-" * 80)

    for step in traj.get('steps', []):
        success = "✓" if step['success'] else "✗"
        print(f"\n{step['step']}. [{success}] {step['action'].upper()} ({step.get('duration', 0):.2f}s)")

        # Show output
        output = step.get('output', {})
        for key, value in output.items():
            if isinstance(value, list) and len(value) <= 3:
                print(f"   • {key}: {value}")
            elif isinstance(value, list):
                print(f"   • {key}: {len(value)} items")
            else:
                print(f"   • {key}: {value}")

        # Show error
        if step.get('error'):
            print(f"   ✗ {step['error']}")

    print("\n" + "=" * 80 + "\n")


def find_trajectory(result_dir: str, model: str, condition: str, level: str, task_id: str) -> str:
    """Find trajectory file path"""
    model_safe = model.replace("/", "-").replace(":", "-")

    if condition == "baseline":
        path = os.path.join(result_dir, "trajectories", model_safe, "baseline", task_id, "trajectory.json")
    else:
        path = os.path.join(result_dir, "trajectories", model_safe, "memory", level, task_id, "trajectory.json")

    return path


def compare_trajectories(result_dir: str, model: str, task_id: str):
    """Compare trajectory across conditions"""
    print("\n" + "=" * 80)
    print(f"COMPARISON: {task_id}")
    print("=" * 80)

    conditions = [
        ("baseline", None),
        ("memory", "L1"),
        ("memory", "L1L2"),
        ("memory", "L1L2L3")
    ]

    trajs = {}

    for condition, level in conditions:
        path = find_trajectory(result_dir, model, condition, level or "", task_id)

        if os.path.exists(path):
            with open(path, 'r') as f:
                traj = json.load(f)

            cond_name = condition if condition == "baseline" else f"{condition}_{level}"
            trajs[cond_name] = traj

    if not trajs:
        print(f"No trajectories found for {task_id}")
        return

    # Summary table
    print(f"\n{'Condition':<15} {'Steps':<8} {'Duration':<12} {'Patch':<8} {'Tokens'}")
    print("-" * 80)

    for name, traj in trajs.items():
        steps = len(traj.get('steps', []))
        duration = f"{traj.get('duration_sec', 0):.1f}s"
        patch = "Yes" if traj.get('patch_generated') else "No"
        tokens = traj.get('total_tokens', {})
        token_str = f"{tokens.get('input', 0):,}/{tokens.get('output', 0):,}"

        print(f"{name:<15} {steps:<8} {duration:<12} {patch:<8} {token_str}")

    # FL predictions
    print("\n" + "-" * 80)
    print("FL PREDICTIONS")
    print("-" * 80)

    for name, traj in trajs.items():
        fl = traj.get('fl_predictions', [])
        print(f"\n{name}: {len(fl)} files")
        for f in fl[:5]:
            print(f"  • {f}")
        if len(fl) > 5:
            print(f"  ... {len(fl) - 5} more")

    # Memory comparison
    if any('memory' in n for n in trajs.keys()):
        print("\n" + "-" * 80)
        print("MEMORY RETRIEVAL")
        print("-" * 80)

        for name, traj in trajs.items():
            if 'memory' not in name:
                continue

            for step in traj.get('steps', []):
                if step['action'] == 'memory_retrieval':
                    output = step.get('output', {})
                    print(f"\n{name}:")
                    print(f"  Injected: {output.get('memory_injected')}")
                    print(f"  Similarity: {output.get('weighted_similarity', 0):.3f}")
                    if output.get('selected_levels'):
                        print(f"  Levels: {', '.join(output['selected_levels'])}")
                    break

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="View simple trajectory files")

    parser.add_argument("file", nargs="?", help="Path to trajectory.json")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--task-id", help="Task ID")
    parser.add_argument("--condition", choices=["baseline", "memory"], help="Condition")
    parser.add_argument("--level", choices=["L1", "L1L2", "L1L2L3"], help="Memory level")
    parser.add_argument("--compare", action="store_true", help="Compare across conditions")
    parser.add_argument("--result-dir", help="Results directory")

    args = parser.parse_args()

    if args.file:
        # Direct file path
        view_trajectory(args.file)
    elif args.model and args.task_id:
        # Find by model/task_id
        config = load_config()
        result_dir = args.result_dir or config.project_result_dir

        if args.compare:
            compare_trajectories(result_dir, args.model, args.task_id)
        else:
            if not args.condition:
                parser.error("--condition required (unless using --compare)")

            if args.condition == "memory" and not args.level:
                parser.error("--level required for memory condition")

            path = find_trajectory(result_dir, args.model, args.condition, args.level or "", args.task_id)
            view_trajectory(path)
    else:
        parser.error("Provide file path OR --model and --task-id")


if __name__ == "__main__":
    main()
