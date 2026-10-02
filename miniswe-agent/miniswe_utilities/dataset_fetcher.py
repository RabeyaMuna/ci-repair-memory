#!/usr/bin/env python3
"""
dataset_fetcher.py - Reusable HuggingFace dataset fetcher
==========================================================

Fetch CI-REPAIR-BENCH dataset from HuggingFace with optional repo filtering.
Can be imported as a module or used as a CLI tool.

Module Usage:
    from utilities.dataset_fetcher import fetch_dataset, filter_by_repos

    # Fetch all issues
    issues = fetch_dataset(split="train")

    # Fetch specific repos
    issues = fetch_dataset(split="train", repos="flower,agno")

    # Save to file
    from utilities.dataset_fetcher import save_issues_to_jsonl
    save_issues_to_jsonl(issues, "data/issues.jsonl")

CLI Usage:
    # Fetch all issues
    python utilities/dataset_fetcher.py --split train --out data/issues.jsonl

    # Fetch specific repos
    python utilities/dataset_fetcher.py --split train --repos "flower,agno" --out data/issues.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HF_DATASET = "ci-benchmark-user/ci-repair-bench"

# Required fields for cibench to work
REQUIRED_FIELDS = [
    "id",
    "sha_fail",
    "repo_owner",
    "repo_name",
    "workflow_path",
    "workflow_name",
    "workflow",
    "logs",
]


def matches_repo_filter(record: dict[str, Any], repos: str | list[str] | None) -> bool:
    """
    Check if a record matches the repo filter.

    Supports multiple formats:
    - Short name: "flower" matches repo_name="flower"
    - Full path: "adap/flower" matches "adap/flower"
    - Partial match: "flow" matches "flower"

    Args:
        record: Issue record with repo_name, repo_owner, or repo field
        repos: Comma-separated string or list of repo filters

    Returns:
        True if record matches any filter (or no filter provided)
    """
    if not repos:
        return True

    # Parse repo filters
    if isinstance(repos, str):
        repo_filters = [
            repo.strip().lower() for repo in repos.split(",") if repo.strip()
        ]
    else:
        repo_filters = [str(repo).strip().lower() for repo in repos if repo]

    if not repo_filters:
        return True

    # Extract repo identifiers from record
    repo_name = str(record.get("repo_name") or "").lower()
    repo_owner = str(record.get("repo_owner") or "").lower()
    repo_full = str(record.get("repo") or "").lower()

    # Build all possible repo representations
    repo_representations = [
        repo_name,  # "flower"
        repo_full,  # "adap/flower"
        f"{repo_owner}/{repo_name}",  # "adap/flower"
    ]

    # Match against any filter
    for repo_filter in repo_filters:
        # Check if filter is a full path (contains /)
        if "/" in repo_filter:
            # Full path: exact match or substring
            if any(repo_filter in rep for rep in repo_representations):
                return True
        else:
            # Short name: match against repo_name only
            if repo_filter in repo_name or repo_name in repo_filter:
                return True

    return False


def fetch_dataset(
    split: str = "train",
    repos: str | list[str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Fetch dataset from HuggingFace with optional repo filtering.

    Args:
        split: Dataset split to fetch ('train', 'test', or 'all')
        repos: Optional repo filter (comma-separated string or list)
        verbose: Print progress messages

    Returns:
        List of issue records
    """
    try:
        from datasets import load_dataset, get_dataset_split_names
    except ImportError:
        print(
            "ERROR: 'datasets' package not installed. Run: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"Loading '{HF_DATASET}' split='{split}' from HuggingFace...")

    issues = []

    if split == "all":
        # Fetch all splits
        splits = get_dataset_split_names(HF_DATASET)
        if verbose:
            print(f"Fetching all splits: {splits}")

        for split_name in splits:
            ds = load_dataset(HF_DATASET, split=split_name)
            split_issues = []
            for row in ds:
                record = dict(row)
                if matches_repo_filter(record, repos):
                    split_issues.append(record)
            issues.extend(split_issues)
            if verbose:
                print(f"  {split_name}: {len(split_issues)}/{len(ds)} issues")
    else:
        # Fetch single split
        ds = load_dataset(HF_DATASET, split=split)
        total = len(ds)
        if verbose:
            print(f"Downloaded {total} instances")
            if repos:
                print(f"Repo filter: {repos}")

        for row in ds:
            record = dict(row)
            if matches_repo_filter(record, repos):
                issues.append(record)

    if verbose:
        print(f"Loaded {len(issues)} issues (after filtering)")

    if repos and len(issues) == 0:
        print(
            f"ERROR: Repo filter matched no issues: {repos}",
            file=sys.stderr,
        )

    return issues


def filter_by_repos(
    issues: list[dict[str, Any]],
    repos: str | list[str],
) -> list[dict[str, Any]]:
    """
    Filter issues by repo names.

    Args:
        issues: List of issue records
        repos: Comma-separated string or list of repo filters

    Returns:
        Filtered list of issues
    """
    return [issue for issue in issues if matches_repo_filter(issue, repos)]


def save_issues_to_jsonl(
    issues: list[dict[str, Any]],
    output_path: str | Path,
    validate_fields: bool = True,
) -> None:
    """
    Save issues to JSONL file.

    Args:
        issues: List of issue records
        output_path: Path to output JSONL file
        validate_fields: Whether to validate required fields
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate first issue if requested
    if validate_fields and issues:
        missing = [f for f in REQUIRED_FIELDS if f not in issues[0]]
        if missing:
            print(
                f"WARNING: First instance is missing fields: {missing}. "
                "cibench may skip or fail those instances.",
                file=sys.stderr,
            )

    with output_path.open("w", encoding="utf-8") as f:
        for issue in issues:
            f.write(json.dumps(issue, ensure_ascii=False) + "\n")

    print(f"Saved {len(issues)} issues -> {output_path}")


def main() -> None:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Download CI-REPAIR-BENCH from HuggingFace with optional repo filtering."
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download (default: train). Use 'train', 'test', or 'all'.",
    )
    parser.add_argument(
        "--out",
        default="data/ci_dataset.jsonl",
        help="Output JSONL file path (default: data/ci_dataset.jsonl).",
    )
    parser.add_argument(
        "--repos",
        default=None,
        help="Optional comma-separated repo filter. Matches repo_name, owner/repo, or repo.",
    )
    args = parser.parse_args()

    # Fetch dataset
    issues = fetch_dataset(
        split=args.split,
        repos=args.repos,
        verbose=True,
    )

    # Save to file
    save_issues_to_jsonl(
        issues,
        args.out,
        validate_fields=True,
    )

    print()
    print("Run the benchmark with:")
    print(
        f"  mini-swe-agent cibench --dataset {args.out} --output results/baseline --no-memory-enabled"
    )
    print(f"  mini-swe-agent cibench --dataset {args.out} --output results/l1_l2_l3 \\")
    print("      --memory-enabled --memory-root data/trs --memory-ablation L1+L2+L3")


if __name__ == "__main__":
    main()
