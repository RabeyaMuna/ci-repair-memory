#!/usr/bin/env python3
"""
Cancel all running GitHub workflow jobs from the benchmark.
"""
import os
import json
import subprocess
import sys
from pathlib import Path

def get_repo_dirs(base_dir):
    """Find all repository directories."""
    repo_dir = Path(base_dir) / "repo"
    if not repo_dir.exists():
        print(f"[ERROR] Repo directory not found: {repo_dir}")
        return []

    return [d for d in repo_dir.iterdir() if d.is_dir() and (d / ".git").exists()]

def cancel_workflows_for_repo(repo_path):
    """Cancel all running workflows for a specific repository."""
    repo_name = repo_path.name
    print(f"\n[{repo_name}] Checking for running workflows...")

    try:
        # Get remote URL to extract owner/repo
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            print(f"  [SKIP] No remote found")
            return

        remote_url = result.stdout.strip()
        # Extract owner/repo from URL (e.g., git@github.com:owner/repo.git or https://github.com/owner/repo)
        if "github.com" not in remote_url:
            print(f"  [SKIP] Not a GitHub repo")
            return

        # Parse owner/repo
        if remote_url.startswith("git@"):
            # git@github.com:owner/repo.git
            parts = remote_url.split(":")[-1].replace(".git", "").split("/")
        else:
            # https://github.com/owner/repo or https://github.com/owner/repo.git
            parts = remote_url.split("github.com/")[-1].replace(".git", "").split("/")

        if len(parts) < 2:
            print(f"  [SKIP] Could not parse repo: {remote_url}")
            return

        owner, repo = parts[0], parts[1]

        # List running workflows using gh CLI
        list_result = subprocess.run(
            ["gh", "run", "list", "--repo", f"{owner}/{repo}", "--status", "in_progress", "--json", "databaseId,status,name"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if list_result.returncode != 0:
            print(f"  [ERROR] Failed to list workflows: {list_result.stderr}")
            return

        runs = json.loads(list_result.stdout)

        if not runs:
            print(f"  [OK] No running workflows")
            return

        print(f"  [FOUND] {len(runs)} running workflow(s)")

        # Cancel each workflow
        cancelled = 0
        for run in runs:
            run_id = run["databaseId"]
            name = run["name"]

            cancel_result = subprocess.run(
                ["gh", "run", "cancel", str(run_id), "--repo", f"{owner}/{repo}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if cancel_result.returncode == 0:
                print(f"  [CANCELLED] Workflow '{name}' (ID: {run_id})")
                cancelled += 1
            else:
                print(f"  [FAILED] Could not cancel '{name}' (ID: {run_id}): {cancel_result.stderr.strip()}")

        print(f"  [SUMMARY] Cancelled {cancelled}/{len(runs)} workflows")

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Command timed out")
    except Exception as e:
        print(f"  [ERROR] {e}")

def main():
    # Get base directory from config
    config_path = Path(__file__).parent.parent / "config.yaml"

    if config_path.exists():
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        base_dir = config.get("base_dir", os.getcwd())
    else:
        base_dir = os.getcwd()

    print(f"Base directory: {base_dir}")
    print("=" * 70)

    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] GitHub CLI (gh) not found. Install it with:")
        print("  brew install gh  # macOS")
        print("  # or visit https://cli.github.com/")
        sys.exit(1)

    # Get all repo directories
    repo_dirs = get_repo_dirs(base_dir)

    if not repo_dirs:
        print("[ERROR] No repository directories found")
        sys.exit(1)

    print(f"Found {len(repo_dirs)} repositories\n")

    # Cancel workflows for each repo
    for repo_dir in sorted(repo_dirs):
        cancel_workflows_for_repo(repo_dir)

    print("\n" + "=" * 70)
    print("[DONE] Finished cancelling workflows")

if __name__ == "__main__":
    main()
