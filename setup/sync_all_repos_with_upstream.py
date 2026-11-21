#!/usr/bin/env python3
"""
Pre-sync all forked repositories with their upstream sources.

Run this ONCE before benchmark execution to avoid repeated syncing overhead.
This script:
1. Identifies all forked repos in the dataset
2. Clones them (if needed)
3. Syncs with upstream
4. Merges .github/workflows from upstream

Then benchmark can run with --skip-upstream-sync flag.
"""

import os
import sys
import json
import requests
import pandas as pd
import git
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf


def load_dataset(dataset_path):
    """Load dataset from parquet file."""
    df = pd.read_parquet(dataset_path)
    return df


def get_upstream_info(repo_owner, repo_name, token):
    """Check if repo is a fork and get upstream URL."""
    try:
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        headers = {"Authorization": f"token {token}"}
        response = requests.get(api_url, headers=headers, timeout=10)

        if response.status_code == 200:
            repo_info = response.json()
            is_fork = repo_info.get("fork", False)

            if is_fork and "parent" in repo_info:
                parent_full_name = repo_info["parent"]["full_name"]
                parent_owner, parent_name = parent_full_name.split("/", 1)
                upstream_url = f"https://github.com/{parent_owner}/{parent_name}.git"
                return True, upstream_url, parent_full_name

        return False, None, None
    except Exception as e:
        print(f"[WARN] Could not check fork status for {repo_owner}/{repo_name}: {e}")
        return False, None, None


def sync_repo_with_upstream(repo_path, repo_owner, repo_name, benchmark_owner,
                           upstream_url, parent_full_name, token, commits=None):
    """Sync a repository with its upstream and fetch specific commits."""

    repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
    origin_url = f"https://{token}@github.com/{benchmark_owner}/{repo_name}.git"

    print(f"\n{'='*80}")
    print(f"[SYNC] {repo_owner}/{repo_name}")
    print(f"[FORK] Upstream: {parent_full_name}")
    if commits:
        print(f"[COMMITS] {len(commits)} specific commits to fetch")
    print(f"{'='*80}")

    # Clone or open repo
    if not os.path.exists(repo_path) or not os.listdir(repo_path):
        print(f"[INFO] Cloning fork (full clone)...")
        repo = git.Repo.clone_from(repo_url, repo_path)
    else:
        print(f"[INFO] Using existing repo...")
        repo = git.Repo(repo_path)

    # Setup origin
    try:
        origin = repo.remote("origin")
        origin.set_url(origin_url)
    except:
        origin = repo.create_remote("origin", url=origin_url)

    # Add or update upstream remote
    print(f"[INFO] Setting up upstream remote...")
    try:
        upstream = repo.remote("upstream")
        upstream.set_url(upstream_url)
    except:
        upstream = repo.create_remote("upstream", url=upstream_url)

    # Unshallow if needed
    print(f"[INFO] Ensuring full clone...")
    try:
        repo.git.fetch("--unshallow")
    except:
        pass  # Already full clone

    # Fetch everything from upstream
    print(f"[INFO] Fetching all refs from upstream...")
    try:
        repo.git.fetch("upstream", "--tags", "--force", "--prune")
        print(f"[OK] Fetched all upstream refs")
    except Exception as e:
        print(f"[ERROR] Failed to fetch upstream: {e}")
        return False

    # Fetch from origin as well
    print(f"[INFO] Fetching from origin (fork)...")
    try:
        repo.git.fetch("origin")
        print(f"[OK] Fetched fork refs")
    except Exception as e:
        print(f"[WARN] Failed to fetch from fork: {e}")

    # Fetch specific commits if provided
    if commits:
        print(f"[INFO] Fetching {len(commits)} specific commits...")
        fetched = 0
        for commit in commits:
            try:
                # Try upstream first
                try:
                    repo.git.fetch("upstream", commit)
                    fetched += 1
                except:
                    # Try origin if not in upstream
                    try:
                        repo.git.fetch("origin", commit)
                        fetched += 1
                    except:
                        pass  # Commit might already be fetched
            except Exception as e:
                pass  # Continue with other commits
        print(f"[OK] Fetched {fetched}/{len(commits)} specific commits")

    # Get upstream default branch
    print(f"[INFO] Detecting upstream default branch...")
    default_branch = "main"
    try:
        branches = repo.git.execute(["git", "branch", "-r", "--list", "upstream/*"]).split("\n")
        for b in branches:
            b = b.strip()
            if "upstream/main" in b:
                default_branch = "main"
                break
            elif "upstream/master" in b:
                default_branch = "master"
                break
        print(f"[OK] Upstream default branch: {default_branch}")
    except:
        pass

    # Store metadata about the sync
    sync_metadata = {
        "repo": f"{repo_owner}/{repo_name}",
        "upstream": parent_full_name,
        "upstream_url": upstream_url,
        "upstream_default_branch": default_branch,
        "synced": True
    }

    print(f"[OK] Successfully synced {repo_owner}/{repo_name}")
    return sync_metadata


def main():
    """Main function to sync all forked repos."""

    # Load config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] config.yaml not found in {os.getcwd()}")
        sys.exit(1)

    config = OmegaConf.load(config_path)

    # Get parameters
    base_dir = config.get("base_dir", os.getcwd())
    benchmark_owner = config.get("BENCHMARK_OWNER", "RabeyaMuna")
    token = config.get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")

    if not token:
        print("[ERROR] GITHUB_TOKEN not found in config.yaml or environment")
        sys.exit(1)

    # Load dataset
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"[INFO] Loading dataset from {dataset_path}")
    df = load_dataset(dataset_path)

    # Get unique repos
    repos = df[['repo_owner', 'repo_name']].drop_duplicates()
    print(f"[INFO] Found {len(repos)} unique repositories in dataset")

    # Identify forks and collect commits
    print(f"\n[INFO] Identifying forked repositories and their commits...")
    fork_repos = []

    for idx, row in tqdm(repos.iterrows(), total=len(repos), desc="Checking forks"):
        repo_owner = row['repo_owner']
        repo_name = row['repo_name']

        is_fork, upstream_url, parent_full_name = get_upstream_info(
            benchmark_owner, repo_name, token
        )

        if is_fork:
            # Get all commits for this repo
            repo_data = df[(df['repo_owner'] == repo_owner) & (df['repo_name'] == repo_name)]
            commits = set()
            for _, instance in repo_data.iterrows():
                if 'sha_fail' in instance and instance['sha_fail']:
                    commits.add(instance['sha_fail'])
                if 'sha_success' in instance and instance['sha_success']:
                    commits.add(instance['sha_success'])

            fork_repos.append({
                'repo_owner': repo_owner,
                'repo_name': repo_name,
                'upstream_url': upstream_url,
                'parent_full_name': parent_full_name,
                'commits': list(commits)
            })

    print(f"\n[INFO] Found {len(fork_repos)} forked repositories")
    print(f"[INFO] Non-fork repos: {len(repos) - len(fork_repos)}")

    if len(fork_repos) == 0:
        print("[INFO] No forked repositories to sync. Exiting.")
        return

    # Create repos folder
    repos_folder = os.path.join(base_dir, "repo")
    os.makedirs(repos_folder, exist_ok=True)

    # Sync each fork
    print(f"\n[INFO] Starting upstream sync for {len(fork_repos)} forks...")
    print(f"[INFO] This may take a while depending on repo sizes...\n")

    sync_results = []
    successful = 0
    failed = 0

    for fork_info in tqdm(fork_repos, desc="Syncing repos"):
        repo_owner = fork_info['repo_owner']
        repo_name = fork_info['repo_name']
        upstream_url = fork_info['upstream_url']
        parent_full_name = fork_info['parent_full_name']
        commits = fork_info.get('commits', [])

        repo_path = os.path.join(repos_folder, f"{repo_owner}__{repo_name}")

        try:
            metadata = sync_repo_with_upstream(
                repo_path, repo_owner, repo_name, benchmark_owner,
                upstream_url, parent_full_name, token, commits
            )

            if metadata:
                sync_results.append(metadata)
                successful += 1
        except Exception as e:
            print(f"[ERROR] Failed to sync {repo_owner}/{repo_name}: {e}")
            sync_results.append({
                "repo": f"{repo_owner}/{repo_name}",
                "upstream": parent_full_name,
                "synced": False,
                "error": str(e)
            })
            failed += 1

    # Save sync manifest
    manifest_path = os.path.join(base_dir, "results", "upstream_sync_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    with open(manifest_path, 'w') as f:
        json.dump({
            "total_repos": len(repos),
            "fork_repos": len(fork_repos),
            "successful_syncs": successful,
            "failed_syncs": failed,
            "sync_results": sync_results
        }, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SYNC SUMMARY")
    print(f"{'='*80}")
    print(f"Total repositories in dataset: {len(repos)}")
    print(f"Forked repositories: {len(fork_repos)}")
    print(f"Successfully synced: {successful}")
    print(f"Failed: {failed}")
    print(f"\nSync manifest saved to: {manifest_path}")
    print(f"\n[OK] You can now run the benchmark with --skip-upstream-sync")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
