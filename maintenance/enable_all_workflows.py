#!/usr/bin/env python3
"""Enable GitHub Actions at repository scope for explicitly named repositories."""

import argparse
import os
from pathlib import Path

import requests
from omegaconf import OmegaConf


def enable_repository_actions(benchmark_owner, repo_name, token):
    """Enable and verify repository-level GitHub Actions permissions."""
    url = (
        f"https://api.github.com/repos/{benchmark_owner}/{repo_name}"
        "/actions/permissions"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.put(
        url,
        headers=headers,
        json={"enabled": True, "allowed_actions": "all"},
        timeout=20,
    )
    if response.status_code not in (200, 204):
        print(
            f"[ERROR] {benchmark_owner}/{repo_name}: could not enable "
            f"repository Actions (HTTP {response.status_code})"
        )
        return False

    verify_response = requests.get(url, headers=headers, timeout=20)
    if not verify_response.ok:
        print(
            f"[ERROR] {benchmark_owner}/{repo_name}: could not verify "
            f"repository Actions (HTTP {verify_response.status_code})"
        )
        return False

    if verify_response.json().get("enabled") is not True:
        print(f"[ERROR] {benchmark_owner}/{repo_name}: Actions remains disabled")
        return False

    print(f"[OK] {benchmark_owner}/{repo_name}: repository Actions enabled")
    return True


def main():
    project_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.load(project_root / "config.yaml")

    parser = argparse.ArgumentParser(
        description=(
            "Enable repository-level GitHub Actions without reading benchmark "
            "job or awaiting-ID files"
        )
    )
    parser.add_argument(
        "repo_names",
        nargs="+",
        metavar="REPO",
        help="Repository name under the benchmark owner (one or more)",
    )
    parser.add_argument(
        "--benchmark-owner",
        default=config.get("benchmark_owner", "Muna4029"),
        help="GitHub owner containing the benchmark repositories",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or config.get("GITHUB_TOKEN")
    if not token:
        parser.error("GITHUB_TOKEN is required in the environment or config.yaml")

    failed = []
    for repo_name in dict.fromkeys(args.repo_names):
        try:
            if not enable_repository_actions(
                args.benchmark_owner, repo_name, token
            ):
                failed.append(repo_name)
        except requests.RequestException as error:
            print(f"[ERROR] {args.benchmark_owner}/{repo_name}: {error}")
            failed.append(repo_name)

    if failed:
        print(f"[SUMMARY] Failed repositories: {', '.join(failed)}")
        return 1

    print(f"[SUMMARY] Enabled repository Actions for {len(set(args.repo_names))} repo(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
