#!/usr/bin/env python3
"""
Git checkout utilities for CI repair benchmark.

Handles repository cloning and commit checkout with robust fetch strategies
for commits from deleted PR branches, shallow clones, etc.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def prepare_repo_checkout(
    issue: dict[str, Any],
    checkout: Path,
    refresh: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Prepare a repository checkout at the failing commit.

    Handles:
    - Shallow clones (auto-unshallow)
    - Commits from deleted PR branches
    - Multiple fetch strategies

    Args:
        issue: Issue dictionary with repo info and sha_fail
        checkout: Path where to checkout the repository
        refresh: If True, remove existing checkout and re-clone
        dry_run: If True, skip actual git operations

    Returns:
        The SHA of the checkout commit (for later diff comparison)

    Raises:
        ValueError: If issue missing required fields
        subprocess.CalledProcessError: If git operations fail
    """
    sha = str(issue.get("sha_fail") or "").strip()
    slug = _get_repo_slug(issue)

    if not slug:
        raise ValueError(f"Issue {_get_issue_id(issue)} has no repo owner/name")
    if not sha:
        raise ValueError(f"Issue {_get_issue_id(issue)} has no sha_fail")

    # Remove old checkout if refresh requested
    if checkout.exists() and refresh:
        shutil.rmtree(checkout)

    if dry_run and not (checkout / ".git").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        return sha  # Return the SHA even in dry run

    # Clone directly from GitHub if not already cloned
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", f"https://github.com/{slug}.git", str(checkout)],
            check=True,
        )

    # Try to checkout the commit - if it fails, fetch it first
    result = subprocess.run(
        ["git", "checkout", "--force", sha],
        cwd=checkout,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Commit not in default branch - fetch it
        print(f"[git] Commit {sha[:8]} not in default branch, fetching...")
        _fetch_and_checkout_commit(checkout, sha)

    # Clean untracked files and directories
    subprocess.run(["git", "clean", "-fdx"], cwd=checkout, check=True)

    return sha


def _fetch_and_checkout_commit(checkout: Path, sha: str) -> None:
    """
    Fetch a commit that's not in the default branch and check it out.

    Tries multiple strategies:
    1. Unshallow if repository is shallow
    2. Direct commit fetch
    3. Fetch all PR refs
    4. Fetch all branches

    Args:
        checkout: Path to the git repository
        sha: Commit SHA to fetch and checkout

    Raises:
        subprocess.CalledProcessError: If all strategies fail
    """
    # First, unshallow the repository if it's shallow
    is_shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=checkout,
        capture_output=True,
        text=True,
    ).stdout.strip() == "true"

    if is_shallow:
        print(f"[git] Repository is shallow, unshallowing...")
        subprocess.run(
            ["git", "fetch", "--unshallow"],
            cwd=checkout,
            capture_output=True,
            text=True,
        )

    # Try multiple fetch strategies
    fetch_strategies = [
        (["git", "fetch", "origin", sha], "direct commit fetch"),
        (["git", "fetch", "origin", "+refs/pull/*/head:refs/remotes/origin/pr/*"], "all PR refs"),
        (["git", "fetch", "--all"], "all branches"),
    ]

    checkout_succeeded = False
    for fetch_cmd, description in fetch_strategies:
        fetch_result = subprocess.run(
            fetch_cmd,
            cwd=checkout,
            capture_output=True,
            text=True,
        )
        if fetch_result.returncode == 0:
            # Try checkout after successful fetch
            checkout_result = subprocess.run(
                ["git", "checkout", "--force", sha],
                cwd=checkout,
                capture_output=True,
                text=True,
            )
            if checkout_result.returncode == 0:
                print(f"[git] ✓ Successfully fetched via {description}")
                checkout_succeeded = True
                break

    if not checkout_succeeded:
        # Final attempt - if all strategies failed, this will raise
        print(f"[git] ✗ All fetch strategies failed, attempting final checkout...")
        subprocess.run(
            ["git", "checkout", "--force", sha],
            cwd=checkout,
            check=True,  # Will raise if still fails
        )


def _get_repo_slug(issue: dict[str, Any]) -> str:
    """Get repository slug (owner/name) from issue."""
    owner = str(issue.get("repo_owner") or "").strip()
    name = str(issue.get("repo_name") or "").strip()
    repo = str(issue.get("repo") or "").strip()

    if owner and name:
        return f"{owner}/{name}"
    return repo


def _get_issue_id(issue: dict[str, Any]) -> str:
    """Get issue ID from issue dictionary."""
    return str(issue.get("instance_id") or issue.get("id") or issue.get("sha_fail"))
