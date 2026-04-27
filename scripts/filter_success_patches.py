#!/usr/bin/env python3
"""
Filter generated_patches.json to only those whose IDs correspond to successful jobs.

It will OVERWRITE:
  results/generated_patches.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_PATH = REPO_ROOT / "results" / "jobs_success_diff.jsonl"
DEFAULT_PATCHES_PATH = REPO_ROOT / "results" / "generated_patches.json"

def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a file that may be:
      - a JSON array: [ {...}, {...}, ... ]
      - JSONL: one JSON object per line
    and return a list of dicts.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try normal JSON first
    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            # Fall back to JSONL parsing
            pass

    # Fallback: assume JSONL (one object per non-empty line)
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-path",
        type=Path,
        default=DEFAULT_JOBS_PATH,
        help=f"Path to jobs JSONL/JSON file (default: {DEFAULT_JOBS_PATH})",
    )
    parser.add_argument(
        "--patches-path",
        type=Path,
        default=DEFAULT_PATCHES_PATH,
        help=f"Path to generated patches JSON file (default: {DEFAULT_PATCHES_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_path = args.jobs_path
    patches_path = args.patches_path

    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs file not found: {jobs_path}")

    if not patches_path.exists():
        raise FileNotFoundError(f"Patches file not found: {patches_path}")

    # 1. Load job metadata
    jobs = load_json_or_jsonl(jobs_path)
    print(f"Loaded {len(jobs)} job records from {jobs_path}")

    # 2. Collect IDs with conclusion == "success"
    success_ids = {
        job["id"]
        for job in jobs
        if isinstance(job, dict) and job.get("conclusion") == "success"
    }
    print(f"Found {len(success_ids)} IDs with conclusion == 'success'.")

    if not success_ids:
        print("No successful IDs found; nothing to filter.")
        return

    # 3. Load all generated patches
    patches = load_json_or_jsonl(patches_path)
    print(f"Loaded {len(patches)} patch records from {patches_path}")

    # 4. Filter patches to only those whose id is in success_ids
    filtered_patches = [
        p for p in patches if isinstance(p, dict) and p.get("id") in success_ids
    ]
    print(f"Keeping {len(filtered_patches)} patch records with successful IDs.")

    # 5. OVERWRITE generated_patches.json with filtered patches (JSON array)
    patches_path.write_text(
        json.dumps(filtered_patches, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Overwrote {patches_path} with filtered patches.")


if __name__ == "__main__":
    main()
