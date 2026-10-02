"""
chronological_splitter.py
=========================
Chronological split utility for preventing temporal data leakage.

Split dataset into memory (earliest 30%) and eval (latest 70%) sets
BEFORE decomposition, so only memory data gets decomposed.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from rich.console import Console

console = Console()


def split_chronologically(
    dataset: List[Dict[str, Any]],
    memory_ratio: float = 0.3,
    group_by_repo: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset chronologically (earliest -> memory, latest -> eval).

    Args:
        dataset: List of issues with commit_date field
        memory_ratio: Ratio for memory set (default 0.3 = 30%)
        group_by_repo: If True, split per repository (recommended)

    Returns:
        (memory_set, eval_set) tuple
    """
    if not group_by_repo:
        # Global chronological split
        sorted_issues = sorted(dataset, key=lambda x: x.get("commit_date", ""))
        cutoff = int(len(sorted_issues) * memory_ratio)
        return sorted_issues[:cutoff], sorted_issues[cutoff:]

    # Per-repository chronological split (by repo_name ONLY, ignoring owner)
    repos = defaultdict(list)
    for issue in dataset:
        # Use ONLY repo_name, ignore repo_owner to group forks together
        repo_name = issue.get('repo_name', 'unknown')
        repos[repo_name].append(issue)

    memory_set = []
    eval_set = []

    for repo_name, issues in repos.items():
        # Issues are already sorted by commit_date

        # Single-issue repos: skip memory, all go to eval
        if len(issues) == 1:
            eval_set.extend(issues)
            continue

        cutoff = max(1, int(len(issues) * memory_ratio))  # At least 1 in memory

        # Ensure we don't put ALL in memory (leave at least 1 for eval)
        if cutoff >= len(issues):
            cutoff = len(issues) - 1

        memory_set.extend(issues[:cutoff])
        eval_set.extend(issues[cutoff:])

    return memory_set, eval_set


def create_split_from_huggingface(
    dataset_name: str = "ci-benchmark-user/ci-repair-bench",
    memory_ratio: float = 0.3,
    output_dir: str = "data",
    repo_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load dataset from HuggingFace and create chronological split.

    Args:
        dataset_name: HuggingFace dataset name
        memory_ratio: Ratio for memory set (default 0.3)
        output_dir: Output directory for saving files
        repo_filter: Optional list of repos to filter (e.g., ["agno", "flower"])
        verbose: Show progress messages

    Returns:
        Dict with 'memory', 'eval', and 'metadata'
    """
    if verbose:
        console.print(f"[cyan]Loading dataset: {dataset_name}[/cyan]")

    # Load dataset
    ds = load_dataset(dataset_name)
    dataset = [dict(item) for item in ds["train"]]

    if verbose:
        console.print(f"[green]Loaded {len(dataset)} issues[/green]")

    # Filter by repos if specified
    if repo_filter:
        repo_filter_lower = [r.lower() for r in repo_filter]
        if verbose:
            console.print(
                f"[yellow]Filtering by repos: {', '.join(repo_filter)}[/yellow]"
            )

        filtered_dataset = []
        for issue in dataset:
            repo_name = issue.get("repo_name", "").lower()
            repo_owner = issue.get("repo_owner", "").lower()
            repo_full = f"{repo_owner}/{repo_name}"

            if any(f in repo_name or f in repo_full for f in repo_filter_lower):
                filtered_dataset.append(issue)

        if verbose:
            console.print(f"[green]Filtered to {len(filtered_dataset)} issues[/green]")
        dataset = filtered_dataset

    # Create split
    memory_set, eval_set = split_chronologically(
        dataset,
        memory_ratio=memory_ratio,
        group_by_repo=True,
    )

    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save memory set
    memory_path = output_path / "memory_set.jsonl"
    with open(memory_path, "w") as f:
        for issue in memory_set:
            f.write(json.dumps(issue) + "\n")

    # Save eval set
    eval_path = output_path / "eval_set.jsonl"
    with open(eval_path, "w") as f:
        for issue in eval_set:
            f.write(json.dumps(issue) + "\n")

    # Save IDs
    memory_ids = [str(issue["id"]) for issue in memory_set]
    eval_ids = [str(issue["id"]) for issue in eval_set]

    with open(output_path / "memory_issue_ids.json", "w") as f:
        json.dump(memory_ids, f, indent=2)

    with open(output_path / "eval_issue_ids.json", "w") as f:
        json.dump(eval_ids, f, indent=2)

    # Create metadata
    memory_dates = [
        issue["commit_date"] for issue in memory_set if issue.get("commit_date")
    ]
    eval_dates = [
        issue["commit_date"] for issue in eval_set if issue.get("commit_date")
    ]

    metadata = {
        "total_issues": len(dataset),
        "memory_size": len(memory_set),
        "eval_size": len(eval_set),
        "memory_ratio": memory_ratio,
        "selection_strategy": "chronological",
        "split_by_repo": True,
        "temporal_leakage_prevented": True,
    }

    if memory_dates:
        metadata["memory_date_range"] = {
            "earliest": min(memory_dates),
            "latest": max(memory_dates),
        }

    if eval_dates:
        metadata["eval_date_range"] = {
            "earliest": min(eval_dates),
            "latest": max(eval_dates),
        }

    with open(output_path / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        console.print(f"\n[green]OK Saved {len(memory_set)} memory issues[/green]")
        console.print(f"[green]OK Saved {len(eval_set)} eval issues[/green]")
        console.print(f"[green]OK Saved to {output_path}[/green]")

    return {
        "memory": memory_set,
        "eval": eval_set,
        "metadata": metadata,
    }


def load_split(
    data_dir: str = "data",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load existing memory and eval splits from disk.

    Args:
        data_dir: Directory containing split files

    Returns:
        (memory_set, eval_set) tuple
    """
    data_path = Path(data_dir)

    memory_set = []
    eval_set = []

    # Load memory
    memory_path = data_path / "memory_set.jsonl"
    if memory_path.exists():
        with open(memory_path, "r") as f:
            for line in f:
                if line.strip():
                    memory_set.append(json.loads(line))

    # Load eval
    eval_path = data_path / "eval_set.jsonl"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            for line in f:
                if line.strip():
                    eval_set.append(json.loads(line))

    return memory_set, eval_set
