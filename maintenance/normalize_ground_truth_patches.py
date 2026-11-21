#!/usr/bin/env python3
"""Rebase curated ground-truth file changes onto each failed commit.

This utility is deliberately limited to ground-truth evaluation. It regenerates
the complete repository transition as a direct ``sha_fail -> sha_success`` diff.
Stored hunk direction, path selection, and blob IDs are not trusted because some
dataset records are incomplete, intermediate, or reversed. Using this process
for model predictions would leak the known successful state.
"""

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


class NormalizationError(RuntimeError):
    """Raised when a ground-truth patch cannot be normalized safely."""


def run_git(repo: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    """Run Git and retain stdout/stderr for an actionable failure message."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise NormalizationError(
            f"git {' '.join(args)} failed ({result.returncode}):\n{result.stderr.strip()}"
        )
    return result


def patch_blocks(diff: str) -> list[str]:
    """Return complete `diff --git` records without rewriting their contents."""
    starts = []
    offset = 0
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            starts.append(offset)
        offset += len(line)
    if not starts:
        raise NormalizationError("Patch contains no `diff --git` records")
    starts.append(len(diff))
    return [diff[start:end] for start, end in zip(starts, starts[1:])]


def header_paths(block: str) -> tuple[str, str]:
    """Parse old/new paths from a Git diff header, including quoted paths."""
    first_line = block.splitlines()[0]
    fields = shlex.split(first_line)
    if len(fields) != 4 or fields[:2] != ["diff", "--git"]:
        raise NormalizationError(f"Unsupported diff header: {first_line}")
    old_path, new_path = fields[2], fields[3]
    if not old_path.startswith("a/") or not new_path.startswith("b/"):
        raise NormalizationError(f"Unsupported diff paths: {first_line}")
    return old_path[2:], new_path[2:]


def curated_paths(diff: str) -> list[str]:
    """Extract curated paths without trusting stored hunk direction/content."""
    paths = []
    for block in patch_blocks(diff):
        old_path, new_path = header_paths(block)
        for path in (old_path, new_path):
            if path not in paths:
                paths.append(path)
    return paths


def validate_on_failed_commit(
    repo: Path,
    sha_fail: str,
    sha_success: str,
    diff: str,
) -> None:
    """Apply on sha_fail and prove the complete tree becomes sha_success."""
    with tempfile.TemporaryDirectory(prefix="ci-ground-truth-") as directory:
        worktree = Path(directory) / "worktree"
        run_git(repo, "worktree", "add", "--detach", str(worktree), sha_fail)
        try:
            run_git(worktree, "apply", "--check", "--index", "-", input_text=diff)
            run_git(worktree, "apply", "--index", "-", input_text=diff)
            actual_tree = run_git(worktree, "write-tree").stdout.strip()
            expected_tree = run_git(
                worktree, "rev-parse", f"{sha_success}^{{tree}}"
            ).stdout.strip()
            if actual_tree != expected_tree:
                raise NormalizationError(
                    f"Applied tree {actual_tree} does not match sha_success {expected_tree}"
                )
        finally:
            subprocess.run(
                ["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
                text=True,
            )


def load_patch_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            record = dict(value) if isinstance(value, dict) else {"diff": str(value)}
            record.setdefault("id", key)
            records.append(record)
        return records
    raise NormalizationError(f"Unsupported patch JSON structure: {type(data).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize curated ground-truth patches onto sha_fail"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--patches", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists() and not run_git(repo, "rev-parse", "--git-dir").stdout.strip():
        raise NormalizationError(f"Not a Git repository: {repo}")

    dataset = pd.read_parquet(args.dataset)
    dataset_by_id = {str(row["id"]): row for _, row in dataset.iterrows()}
    patches_by_id = {
        str(record.get("id")): record for record in load_patch_records(args.patches)
    }

    normalized = []
    for instance_id in args.ids:
        instance_id = str(instance_id)
        if instance_id not in dataset_by_id:
            raise NormalizationError(f"ID {instance_id} is absent from the dataset")
        if instance_id not in patches_by_id:
            raise NormalizationError(f"ID {instance_id} is absent from the patch file")

        row = dataset_by_id[instance_id]
        record = dict(patches_by_id[instance_id])
        original_diff = record.get("diff") or record.get("patch") or ""
        sha_fail = str(row["sha_fail"])
        sha_success = str(row["sha_success"])

        run_git(repo, "cat-file", "-e", f"{sha_fail}^{{commit}}")
        run_git(repo, "cat-file", "-e", f"{sha_success}^{{commit}}")
        corrected = run_git(
            repo,
            "diff",
            "--binary",
            "--full-index",
            sha_fail,
            sha_success,
        ).stdout
        paths = [
            path
            for path in run_git(
                repo, "diff", "--name-only", sha_fail, sha_success
            ).stdout.splitlines()
            if path
        ]
        if corrected.strip():
            validate_on_failed_commit(repo, sha_fail, sha_success, corrected)
        elif run_git(repo, "rev-parse", f"{sha_fail}^{{tree}}").stdout.strip() != run_git(
            repo, "rev-parse", f"{sha_success}^{{tree}}"
        ).stdout.strip():
            raise NormalizationError(
                f"ID {instance_id} has different endpoint trees but an empty patch"
            )

        record["sha_fail"] = sha_fail
        record["diff"] = corrected
        record.pop("patch", None)
        record["ground_truth_normalized"] = True
        record["ground_truth_sha_success"] = sha_success
        record["ground_truth_curated_file_count"] = len(paths)
        normalized.append(record)
        print(
            f"[OK] ID {instance_id}: {len(paths)} repository paths, "
            f"{len(corrected.splitlines())} normalized diff lines"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(normalized, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    os.replace(temporary, args.output)
    print(f"[OK] Wrote {len(normalized)} normalized patches to {args.output}")


if __name__ == "__main__":
    main()
