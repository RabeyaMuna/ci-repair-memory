#!/usr/bin/env python3
"""
Filter generated_patches.json to only those whose IDs correspond to successful jobs.

It will OVERWRITE:
  /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/generated_patches.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any

JOBS_PATH = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/jobs_success_diff.jsonl"
)
PATCHES_PATH = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/generated_patches.json"
)

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


def main() -> None:
    if not JOBS_PATH.exists():
        raise FileNotFoundError(f"Jobs file not found: {JOBS_PATH}")

    if not PATCHES_PATH.exists():
        raise FileNotFoundError(f"Patches file not found: {PATCHES_PATH}")

    # 1. Load job metadata
    jobs = load_json_or_jsonl(JOBS_PATH)
    print(f"Loaded {len(jobs)} job records from {JOBS_PATH}")

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
    patches = load_json_or_jsonl(PATCHES_PATH)
    print(f"Loaded {len(patches)} patch records from {PATCHES_PATH}")

    # 4. Filter patches to only those whose id is in success_ids
    filtered_patches = [
        p for p in patches if isinstance(p, dict) and p.get("id") in success_ids
    ]
    print(f"Keeping {len(filtered_patches)} patch records with successful IDs.")

    # 5. OVERWRITE generated_patches.json with filtered patches (JSON array)
    PATCHES_PATH.write_text(
        json.dumps(filtered_patches, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Overwrote {PATCHES_PATH} with filtered patches.")


if __name__ == "__main__":
    main()
