#!/usr/bin/env python3
"""
Sample instances from existing results file.
"""

import json
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sample instances from existing results")
    parser.add_argument("--run1-results", required=True, help="Path to Run 1 results file")
    parser.add_argument("--num-instances", type=int, default=100, help="Number to sample")
    parser.add_argument("--output", required=True, help="Output file for sampled IDs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Read results
    results = []
    with open(args.run1_results, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                results.append(result)
            except:
                continue

    if not results:
        print(f"[ERROR] No results found in {args.run1_results}")
        return 1

    print(f"[INFO] Found {len(results)} instances in Run 1")

    # Sample
    random.seed(args.seed)
    sampled = random.sample(results, min(args.num_instances, len(results)))

    # Extract IDs
    sampled_ids = []
    for r in sampled:
        instance_id = r.get('id') or r.get('datapoint_id') or r.get('issue_id') or r.get('task_id')
        if instance_id:
            sampled_ids.append(str(instance_id))

    print(f"[INFO] Sampled {len(sampled_ids)} instances")
    print(f"[INFO] First 10: {sampled_ids[:10]}")

    # Save IDs (comma-separated for command line use)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(",".join(sampled_ids))

    # Also save as JSON
    json_output = output_path.with_suffix('.json')
    with open(json_output, 'w') as f:
        json.dump({
            "num_instances": len(sampled_ids),
            "seed": args.seed,
            "instance_ids": sampled_ids,
            "source": args.run1_results
        }, f, indent=2)

    print(f"[SAVED] Sampled IDs: {output_path}")
    print(f"[SAVED] JSON metadata: {json_output}")

    # Also copy Run 1 results for these instances
    run1_sampled = output_path.parent / "run_1_results.jsonl"
    with open(run1_sampled, 'w') as f:
        for r in sampled:
            f.write(json.dumps(r) + '\n')

    print(f"[SAVED] Run 1 results (sampled): {run1_sampled}")

    return 0


if __name__ == "__main__":
    exit(main())
