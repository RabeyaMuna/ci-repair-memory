#!/usr/bin/env python3
"""
Deduplicate Results Files

Removes duplicate entries from JSONL files, keeping only the newest
(last occurrence) of each ID.

Usage:
    python deduplicate_results.py results/jobs_ids_diff.jsonl
    python deduplicate_results.py results/jobs_results_diff.jsonl --backup
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter


def deduplicate_jsonl(input_file: str, backup: bool = True):
    """
    Deduplicate a JSONL file by ID, keeping the last occurrence.

    Args:
        input_file: Path to JSONL file
        backup: Create .backup file before overwriting
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        return

    # Read all entries
    print(f"📂 Reading {input_file}...")
    entries = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"⚠️  Skipping invalid JSON line")

    if not entries:
        print("⚠️  File is empty")
        return

    print(f"   Total entries: {len(entries)}")

    # Count duplicates
    id_counts = Counter(str(e.get("id")) for e in entries)
    duplicates = {id: count for id, count in id_counts.items() if count > 1}

    if not duplicates:
        print("✓ No duplicates found")
        return

    # Keep only last occurrence of each ID
    seen_ids = {}
    for entry in entries:
        entry_id = str(entry.get("id"))
        seen_ids[entry_id] = entry

    deduplicated = list(seen_ids.values())

    print(f"   Unique IDs: {len(deduplicated)}")
    print(f"   Duplicates removed: {len(entries) - len(deduplicated)}")

    # Create backup
    if backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".backup")
        print(f"\n💾 Creating backup: {backup_path}")
        input_path.rename(backup_path)

    # Write deduplicated entries
    print(f"💾 Writing deduplicated file...")
    with open(input_path, "w") as f:
        for entry in deduplicated:
            json.dump(entry, f)
            f.write("\n")

    print(f"\n✅ Done!")
    print(f"   File: {input_file}")
    print(f"   Entries: {len(entries)} → {len(deduplicated)}")

    # Show duplicated IDs
    if len(duplicates) <= 20:
        print(f"\n📋 Duplicated IDs (kept newest):")
        for id, count in sorted(duplicates.items()):
            print(f"   ID {id}: {count} occurrences")
    else:
        print(f"\n📋 {len(duplicates)} IDs had duplicates (kept newest)")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate JSONL results files"
    )
    parser.add_argument(
        "file",
        help="Path to JSONL file to deduplicate"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup file"
    )

    args = parser.parse_args()

    deduplicate_jsonl(args.file, backup=not args.no_backup)


if __name__ == "__main__":
    main()
