#!/usr/bin/env python3
"""
Setup permanent benchmark branches for each issue.

Creates persistent branches in forked repos (RabeyaMuna/*) pointing to sha_fail.
Branch naming: benchmark_{owner}_{repo}_issue_{id}

This allows:
- Persistent CI testing without re-creating branches
- Ground truth validation over time
- CI workflow health monitoring
"""

import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration
DATASET_FILE = Path(__file__).parent.parent / "dataset" / "lca_dataset.parquet"
OUTPUT_FILE = Path(__file__).parent / "results" / "branches" / "benchmark_branches.json"
BENCHMARK_OWNER = os.environ.get("BENCHMARK_OWNER", "RabeyaMuna")
GITHUB_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

# Settings
TEMP_REPOS_DIR = Path(__file__).parent.parent / "temp_repos"


def run_git_command(cmd: List[str], cwd: str = None) -> Tuple[bool, str]:
    """Run a git command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def clone_or_get_repo(owner: str, repo: str) -> Optional[str]:
    """Clone repository if needed, return path."""
    repo_path = TEMP_REPOS_DIR / BENCHMARK_OWNER / repo

    if repo_path.exists():
        print(f"    Using existing repo: {repo_path}")
        # Fetch latest
        run_git_command(["git", "fetch", "--all"], cwd=str(repo_path))
        return str(repo_path)

    print(f"    Cloning {BENCHMARK_OWNER}/{repo}...")
    repo_path.parent.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://github.com/{BENCHMARK_OWNER}/{repo}.git"
    success, output = run_git_command(["git", "clone", clone_url, str(repo_path)])

    if not success:
        print(f"    ❌ Failed to clone: {output}")
        return None

    return str(repo_path)


def branch_exists_remote(repo_path: str, branch_name: str) -> bool:
    """Check if branch exists on remote."""
    success, output = run_git_command(
        ["git", "ls-remote", "--heads", "origin", branch_name],
        cwd=repo_path
    )
    return success and branch_name in output


def create_benchmark_branch(
    issue_id: str,
    owner: str,
    repo: str,
    sha_fail: str,
    force: bool = False
) -> Optional[Dict]:
    """
    Create a permanent benchmark branch for an issue.

    Returns branch info dict if successful, None otherwise.
    """
    # Generate branch name
    # Format: benchmark_{original_owner}_{repo}_issue_{id}
    branch_name = f"benchmark_{owner}_{repo}_issue_{issue_id}".replace("/", "_")

    print(f"\n[{issue_id}] {owner}/{repo}")
    print(f"  Branch: {branch_name}")
    print(f"  Target: {sha_fail[:8]}")

    # Clone/get repo
    repo_path = clone_or_get_repo(owner, repo)
    if not repo_path:
        return None

    # Check if branch already exists
    if not force and branch_exists_remote(repo_path, branch_name):
        print(f"  ✓ Branch already exists (use --force to recreate)")
        return {
            "id": issue_id,
            "owner": owner,
            "repo": repo,
            "branch_name": branch_name,
            "sha_fail": sha_fail,
            "status": "existing",
            "created_at": None,
            "fork_repo": f"{BENCHMARK_OWNER}/{repo}"
        }

    # Fetch the commit if not available locally
    success, _ = run_git_command(["git", "fetch", "origin", sha_fail], cwd=repo_path)
    if not success:
        print(f"  ⚠️  Fetching commit {sha_fail[:8]}...")

    # Create branch locally
    print(f"  Creating branch at {sha_fail[:8]}...")

    # Delete local branch if exists
    run_git_command(["git", "branch", "-D", branch_name], cwd=repo_path)

    success, output = run_git_command(
        ["git", "branch", branch_name, sha_fail],
        cwd=repo_path
    )

    if not success:
        print(f"  ❌ Failed to create branch: {output}")
        return None

    # Push branch to remote
    print(f"  Pushing to {BENCHMARK_OWNER}/{repo}...")

    # Force push if recreating
    push_cmd = ["git", "push", "origin", branch_name]
    if force:
        push_cmd.insert(2, "--force")

    success, output = run_git_command(push_cmd, cwd=repo_path)

    if not success:
        print(f"  ❌ Failed to push: {output}")
        # Clean up local branch
        run_git_command(["git", "branch", "-D", branch_name], cwd=repo_path)
        return None

    print(f"  ✓ Branch created successfully")

    return {
        "id": issue_id,
        "owner": owner,
        "repo": repo,
        "branch_name": branch_name,
        "sha_fail": sha_fail,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "fork_repo": f"{BENCHMARK_OWNER}/{repo}",
        "url": f"https://github.com/{BENCHMARK_OWNER}/{repo}/tree/{branch_name}"
    }


def main():
    """Main function to setup benchmark branches."""
    if not GITHUB_TOKEN:
        print("[ERROR] No GitHub token found. Set GH_TOKEN or GITHUB_TOKEN in .env")
        return

    print(f"✓ GitHub token loaded")
    print(f"✓ Benchmark owner: {BENCHMARK_OWNER}")

    # Load dataset
    print(f"\nLoading dataset from: {DATASET_FILE}")
    df = pd.read_parquet(DATASET_FILE)
    print(f"Found {len(df)} issues in dataset")

    # Load existing branches if any
    existing_branches = {}
    if OUTPUT_FILE.exists():
        print(f"\n✓ Found existing branches file")
        with open(OUTPUT_FILE, "r") as f:
            existing_data = json.load(f)
            existing_branches = {b["id"]: b for b in existing_data}
        print(f"  {len(existing_branches)} branches already tracked")

    # Ask for confirmation
    print(f"\n{'='*80}")
    print("This will create permanent branches in your forked repositories.")
    print(f"Repository: {BENCHMARK_OWNER}/*")
    print(f"Total issues: {len(df)}")
    print(f"Already created: {len(existing_branches)}")
    print(f"New branches: {len(df) - len(existing_branches)}")
    print(f"{'='*80}")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Process each issue
    results = []
    created_count = 0
    failed_count = 0
    skipped_count = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        issue_id = str(row['id'])
        owner = str(row['repo_owner'])
        repo = str(row['repo_name'])
        sha_fail = str(row['sha_fail'])

        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(df)}")

        try:
            # Skip if already exists
            if issue_id in existing_branches:
                print(f"  ⏭️  Skipping - branch already created")
                results.append(existing_branches[issue_id])
                skipped_count += 1
                continue

            branch_info = create_benchmark_branch(issue_id, owner, repo, sha_fail)

            if branch_info:
                results.append(branch_info)
                if branch_info["status"] == "created":
                    created_count += 1
                else:
                    skipped_count += 1
            else:
                failed_count += 1
                # Still track failed attempts
                results.append({
                    "id": issue_id,
                    "owner": owner,
                    "repo": repo,
                    "status": "failed",
                    "error": "Failed to create branch"
                })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1

        # Save intermediate results every 10 issues
        if i % 10 == 0:
            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  💾 Saved checkpoint ({len(results)} branches)")

    # Save final results
    print(f"\n{'='*80}")
    print(f"Saving results to: {OUTPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Total issues: {len(df)}")
    print(f"  Newly created: {created_count}")
    print(f"  Already existed: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"\n✓ Benchmark branches setup complete!")
    print(f"  Results saved to: {OUTPUT_FILE}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
