#!/usr/bin/env python3
"""
Check for missing repositories
Identifies which repos are needed but not yet cloned
"""
import json
import subprocess
from pathlib import Path
from collections import Counter
import argparse


def check_commit_in_repos(sha_fail: str, repo_dir: Path) -> str:
    """
    Find which repo contains a commit

    Returns:
        repo name if found, None otherwise
    """
    for repo_path in repo_dir.iterdir():
        if not repo_path.is_dir() or repo_path.name.startswith('.'):
            continue

        try:
            result = subprocess.run(
                ["git", "cat-file", "-t", sha_fail],
                cwd=repo_path,
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return repo_path.name
        except:
            continue

    return None


def main():
    parser = argparse.ArgumentParser(description="Check for missing repositories")
    parser.add_argument("--model", default="gpt-5-mini_llm", help="Model name")
    parser.add_argument("--limit", type=int, default=150, help="Number of entries to check")

    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    log_details_path = base_dir / "baselines" / "results" / args.model / "log_details.json"
    repo_dir = base_dir / "baselines" / "repo_cloned"

    print("🔍 Checking for Missing Repositories")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Checking first {args.limit} entries")
    print(f"Repo directory: {repo_dir}")
    print()

    # Load log_details
    with open(log_details_path) as f:
        log_details = json.load(f)

    print(f"[LOAD] Loaded {len(log_details)} entries from log_details.json")
    print()

    # Check each commit
    found_repos = []
    missing_commits = []

    print("Checking commits...")
    for i, entry in enumerate(log_details[:args.limit]):
        sha_fail = entry.get("sha_fail", "")
        entry_id = entry.get("id", str(i))

        if not sha_fail:
            continue

        if (i + 1) % 10 == 0:
            print(f"  Checked {i+1}/{args.limit}...")

        repo_name = check_commit_in_repos(sha_fail, repo_dir)

        if repo_name:
            found_repos.append(repo_name)
        else:
            missing_commits.append((entry_id, sha_fail))

    print()
    print("=" * 60)
    print("[STATS] Results:")
    print()

    # Summary
    print(f"[OK] Found: {len(found_repos)}/{args.limit}")
    print(f"[ERROR] Missing: {len(missing_commits)}/{args.limit}")
    print()

    # Show which repos were used
    if found_repos:
        repo_counts = Counter(found_repos)
        print(f"Repositories used ({len(repo_counts)} unique):")
        for repo, count in repo_counts.most_common(10):
            print(f"  {repo}: {count} commits")
        if len(repo_counts) > 10:
            print(f"  ... and {len(repo_counts) - 10} more")
        print()

    # Show missing commits
    if missing_commits:
        print(f"[WARN]  Missing commits ({len(missing_commits)}):")
        for entry_id, sha in missing_commits[:10]:
            print(f"  ID={entry_id}, SHA={sha[:7]}")
        if len(missing_commits) > 10:
            print(f"  ... and {len(missing_commits) - 10} more")
        print()
        print("These commits are either:")
        print("  1. In repos that aren't cloned yet")
        print("  2. Not fetched yet (need to run git fetch --all)")
        print()
        print("Try running:")
        print("  cd ../repo_cloned && ../BRaIn/fetch_missing_commits.sh")
    else:
        print("[OK] All commits found! Ready to index.")
        print()
        print("Next steps:")
        print(f"  python index_repositories.py --model {args.model} --limit {args.limit}")

    print("=" * 60)


if __name__ == "__main__":
    main()
