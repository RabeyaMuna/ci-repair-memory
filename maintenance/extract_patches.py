#!/usr/bin/env python3
"""
Extract patches from dataset and create generated_patches.json
Validates diffs to prevent corruption
"""
import pandas as pd
import json
import argparse
from pathlib import Path


def extract_patches(dataset_path: str, output_path: str = "maintenance/reports/generated_patches.json"):
    """Extract patches from dataset to JSON format"""

    print(f"[INFO] Loading dataset: {dataset_path}")

    # Load dataset
    if dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path}")

    print(f"[INFO] Loaded {len(df)} instances")

    # Extract patches
    patches = []
    skipped = 0

    for _, row in df.iterrows():
        # Handle both formats
        if 'repo' in row:
            repo = row['repo']
        elif 'repo_owner' in row and 'repo_name' in row:
            repo = f"{row['repo_owner']}/{row['repo_name']}"
        else:
            print(f"[WARN] Instance {row['id']} missing repo info, skipping")
            skipped += 1
            continue

        # Handle both diff and diff_fix
        diff = row.get('diff') or row.get('diff_fix', '')

        if not diff:
            print(f"[WARN] Instance {row['id']} has no diff, skipping")
            skipped += 1
            continue

        # Validate and clean diff - prevent corruption
        # Ensure diff is string (not bytes or corrupted)
        if isinstance(diff, bytes):
            try:
                diff = diff.decode('utf-8')
            except:
                print(f"[WARN] Instance {row['id']} has corrupted diff (cannot decode), skipping")
                skipped += 1
                continue

        diff = str(diff)

        # Basic validation: check if not empty
        if not diff.strip():
            print(f"[WARN] Instance {row['id']} has empty diff, skipping")
            skipped += 1
            continue

        # Check for common diff markers
        has_diff_markers = any(marker in diff for marker in ['diff --git', '@@', '---', '+++', 'index '])
        if not has_diff_markers:
            print(f"[WARN] Instance {row['id']} diff may be corrupted (no diff markers)")

        patches.append({
            "id": str(row['id']),
            "repo": repo,
            "sha_fail": str(row['sha_fail']),
            "diff": diff,  # Preserved exactly as-is, no escaping
            "workflow_path": str(row.get('workflow_path', '')),
            "workflow": str(row.get('workflow', ''))
        })

    # Save to JSON (ensure_ascii=False preserves exact text without corruption)
    output_file = Path(output_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(patches, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Created {output_file} with {len(patches)} patches")
    if skipped > 0:
        print(f"[INFO] Skipped {skipped} instances (missing/corrupted diffs)")
    print(f"[INFO] All diffs preserved without corruption")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--output", default="maintenance/reports/generated_patches.json", help="Output JSON file")

    args = parser.parse_args()
    extract_patches(args.dataset, args.output)
