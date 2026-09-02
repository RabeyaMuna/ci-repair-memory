#!/usr/bin/env python3
"""Clean stale, duplicate, and incomplete GPT-5-mini LLM result records."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "baselines" / "results" / "gpt-5-mini_llm"
DATASET_PATH = ROOT / "dataset" / "ci_repair_dataset.parquet"

STALE_IDS = {
    96, 147, 212, 224, 225, 228, 229, 230, 241, 248, 251, 252, 253,
    254, 256, 260, 264, 266, 285, 286, 289, 299, 300, 301, 302, 303,
    313, 314, 318, 319, 321, 369, 509, 551, 575, 576,
}


def normalized_id(record):
    try:
        return int(record.get("id"))
    except (TypeError, ValueError):
        return None


def load_records(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def deduplicate(records):
    """Keep the last record for a SHA (or ID when SHA is unavailable)."""
    kept = []
    position = {}
    duplicates = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        sha = str(record.get("sha_fail") or "").strip()
        rid = normalized_id(record)
        key = ("sha", sha) if sha else (("id", rid) if rid is not None else None)
        if key is not None and key in position:
            kept[position[key]] = record
            duplicates += 1
        else:
            if key is not None:
                position[key] = len(kept)
            kept.append(record)
    return kept, duplicates


def main():
    dataset = pd.read_parquet(DATASET_PATH, columns=["id", "sha_fail"])
    numeric_ids = pd.to_numeric(dataset["id"], errors="coerce")
    stale_shas = set(
        dataset.loc[numeric_ids.isin(STALE_IDS), "sha_fail"].dropna().astype(str)
    )

    fl_path = RESULT_DIR / "fault_localization.json"
    fault_records = load_records(fl_path)
    empty_locations_removed = 0
    for record in fault_records:
        if not isinstance(record, dict):
            continue
        locations = record.get("fault_localization_data") or []
        nonempty_locations = []
        for location in locations:
            if isinstance(location, dict) and location.get("faults"):
                nonempty_locations.append(location)
            else:
                empty_locations_removed += 1
        record["fault_localization_data"] = nonempty_locations
    incomplete_shas = {
        str(record.get("sha_fail") or "")
        for record in fault_records
        if isinstance(record, dict)
        and not record.get("fault_localization_data")
        and record.get("sha_fail")
    }

    summary = {}
    for path in sorted(RESULT_DIR.glob("*.json")):
        records = fault_records if path == fl_path else load_records(path)
        before = len(records)
        filtered = []
        removed_stale = 0
        removed_incomplete = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            sha = str(record.get("sha_fail") or "")
            if normalized_id(record) in STALE_IDS or sha in stale_shas:
                removed_stale += 1
                continue
            if sha in incomplete_shas:
                removed_incomplete += 1
                continue
            filtered.append(record)

        filtered, duplicate_count = deduplicate(filtered)
        path.write_text(
            json.dumps(filtered, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary[path.name] = {
            "before": before,
            "removed_requested": removed_stale,
            "removed_incomplete_error": removed_incomplete,
            "removed_duplicates": duplicate_count,
            "after": len(filtered),
        }
        if path == fl_path:
            summary[path.name]["removed_empty_locations"] = empty_locations_removed

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
