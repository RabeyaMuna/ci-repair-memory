#!/usr/bin/env python3
"""Synchronize canonical failure classifications into the parquet dataset."""

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_and_validate(classifications_path, dataframe):
    with classifications_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    classifications = payload.get("classifications", [])
    dataset_ids = [str(value) for value in dataframe["id"]]
    classification_ids = [str(item.get("issue_id")) for item in classifications]

    duplicate_dataset_ids = sorted(
        key for key, count in Counter(dataset_ids).items() if count > 1
    )
    duplicate_classification_ids = sorted(
        key for key, count in Counter(classification_ids).items() if count > 1
    )
    missing_ids = sorted(set(dataset_ids) - set(classification_ids))
    extra_ids = sorted(set(classification_ids) - set(dataset_ids))

    errors = []
    if duplicate_dataset_ids:
        errors.append(f"duplicate dataset IDs: {duplicate_dataset_ids}")
    if duplicate_classification_ids:
        errors.append(
            f"duplicate classification IDs: {duplicate_classification_ids}"
        )
    if missing_ids:
        errors.append(f"dataset IDs missing classifications: {missing_ids}")
    if extra_ids:
        errors.append(f"classification IDs absent from dataset: {extra_ids}")

    taxonomy = set(payload.get("taxonomy", {}).keys())
    for item in classifications:
        issue_id = str(item.get("issue_id"))
        failure_types = item.get("failure_type") or []
        subtypes = item.get("sub_type") or []
        details = item.get("detail") or []
        if not failure_types:
            errors.append(f"issue {issue_id} has no failure type")
        if len(failure_types) != len(subtypes):
            errors.append(f"issue {issue_id} has mismatched type/subtype lengths")
        if details and len(failure_types) != len(details):
            errors.append(f"issue {issue_id} has mismatched type/detail lengths")
        unknown_types = sorted(set(failure_types) - taxonomy) if taxonomy else []
        if unknown_types:
            errors.append(f"issue {issue_id} has non-taxonomy types: {unknown_types}")

    if errors:
        preview = "\n  - ".join(errors[:20])
        suffix = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Classification validation failed:\n  - {preview}{suffix}")

    return payload, {
        str(item["issue_id"]): item for item in classifications
    }


def synchronize(dataframe, classification_map):
    updated = dataframe.copy()

    def field(issue_id, name):
        return list(classification_map[str(issue_id)].get(name) or [])

    # error_type is retained as the canonical paper-facing field. The more
    # explicit failure_* fields store the same labels plus subtype/detail data.
    updated["error_type"] = updated["id"].map(
        lambda issue_id: field(issue_id, "failure_type")
    )
    updated["failure_types"] = updated["id"].map(
        lambda issue_id: field(issue_id, "failure_type")
    )
    updated["failure_subtypes"] = updated["id"].map(
        lambda issue_id: field(issue_id, "sub_type")
    )
    updated["failure_details"] = updated["id"].map(
        lambda issue_id: field(issue_id, "detail")
    )
    updated["num_failure_types"] = updated["failure_types"].map(len)
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Strictly synchronize classification JSON into a parquet dataset."
    )
    parser.add_argument(
        "--classifications",
        default="results/failure_classifications_12types.json",
    )
    parser.add_argument("--dataset", default="dataset/lca_dataset.parquet")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before replacing the dataset",
    )
    args = parser.parse_args()

    classifications_path = Path(args.classifications)
    dataset_path = Path(args.dataset)
    dataframe = pd.read_parquet(dataset_path)
    payload, classification_map = load_and_validate(classifications_path, dataframe)
    updated = synchronize(dataframe, classification_map)

    backup_path = None
    if not args.no_backup:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = dataset_path.with_name(
            f"{dataset_path.stem}.before-classification-sync-{timestamp}{dataset_path.suffix}"
        )
        shutil.copy2(dataset_path, backup_path)

    temporary_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    updated.to_parquet(temporary_path, index=False)
    # Verify that the serialized artifact remains readable before replacement.
    verified = pd.read_parquet(temporary_path)
    if len(verified) != len(updated):
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Serialized dataset row count changed; original not replaced")
    os.replace(temporary_path, dataset_path)

    counts = Counter(
        failure_type
        for types in updated["failure_types"]
        for failure_type in set(types)
    )
    print(f"Updated dataset: {dataset_path}")
    if backup_path:
        print(f"Backup: {backup_path}")
    print(f"Instances synchronized: {len(updated):,}")
    print(f"Taxonomy size: {len(payload.get('taxonomy', {})):,}")
    print(f"Failure-type occurrences: {sum(counts.values()):,}")


if __name__ == "__main__":
    main()
