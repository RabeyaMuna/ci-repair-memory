#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import requests
from dotenv import load_dotenv

# ======== CONFIGURATION ========

current_dir = os.getcwd()
config_path = os.path.join(current_dir, "config.yaml")
config = OmegaConf.load(config_path)

USERNAME = config.get("username_gh")  # your GitHub username

DATASET_PATH = Path(
    os.path.join(
        config.get("base_dir"),
        "dataset",
        "lca_dataset.parquet",
    )
)

# Optional: where to also save the list of unique repos as owner/repo
REPO_LIST_FILE = Path(
    os.path.join(
        config.get("base_dir"),
        "setup_github",
        "unique_repo_urls.txt",
    )
)

POLL_SECONDS = 30
POLL_INTERVAL = 3
REQUEST_DELAY = 2
# ===============================

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise EnvironmentError(
        "GITHUB_TOKEN not found in environment. Put it in a .env file or export it."
    )

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    # You can add API version if you want:
    # "X-GitHub-Api-Version": "2022-11-28",
}
API_BASE = "https://api.github.com"


def repo_exists_in_user(username: str, repo_name: str) -> bool:
    url = f"{API_BASE}/repos/{username}/{repo_name}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 200:
        data = resp.json()
        return bool(data.get("fork", False))
    elif resp.status_code == 404:
        return False
    else:
        print(
            f"[WARN] Unexpected status while checking {username}/{repo_name}: "
            f"{resp.status_code} {resp.text[:120]}"
        )
        return False


def fork_repository(owner: str, repo: str) -> bool:
    url = f"{API_BASE}/repos/{owner}/{repo}/forks"
    resp = requests.post(url, headers=HEADERS)
    if resp.status_code in (201, 202):
        print(f"[INFO] Fork request accepted: {owner}/{repo}")
        return True
    if resp.status_code == 422:
        print(f"[INFO] GitHub reported 422 for {owner}/{repo} (possibly already forked).")
        return True
    if resp.status_code == 404:
        print(f"[ERROR] Not found or no access: {owner}/{repo}")
        return False
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        reset = resp.headers.get("X-RateLimit-Reset")
        print("[ERROR] Rate limited by GitHub API.")
        if reset:
            try:
                reset_epoch = int(reset)
                wait = max(0, reset_epoch - int(time.time())) + 5
                print(f"[INFO] Sleeping until reset (~{wait}s).")
                time.sleep(wait)
            except Exception:
                print("[WARN] Could not parse rate limit reset header. Sleeping 60s.")
                time.sleep(60)
        else:
            print("[WARN] No rate limit reset header. Sleeping 60s.")
            time.sleep(60)
        return fork_repository(owner, repo)

    print(
        f"[ERROR] Failed to fork {owner}/{repo}: "
        f"{resp.status_code} {resp.text[:200]}"
    )
    return False


def poll_until_exists(username: str, repo_name: str, timeout_s: int, interval_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if repo_exists_in_user(username, repo_name):
            return True
        time.sleep(interval_s)
    return False


def load_unique_repos_from_dataset(dataset_path: Path) -> list[str]:
    """
    Load the parquet dataset and return a list of unique 'owner/repo' strings
    using the 'repo_owner' and 'repo_name' columns.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"[INFO] Loading dataset from {dataset_path} ...")
    df = pd.read_parquet(dataset_path)

    for col in ("repo_owner", "repo_name"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataset.")

    sub = df[["repo_owner", "repo_name"]].dropna()
    sub["repo_owner"] = sub["repo_owner"].astype(str).str.strip()
    sub["repo_name"] = sub["repo_name"].astype(str).str.strip()
    sub = sub[(sub["repo_owner"] != "") & (sub["repo_name"] != "")]
    sub = sub.drop_duplicates()

    repos: list[str] = [
        f"{row.repo_owner}/{row.repo_name}"
        for row in sub.itertuples(index=False)
    ]

    print(f"[INFO] Found {len(repos)} unique owner/repo pairs in dataset.")
    return repos


def forked_repo_list(repos: list[str], path: Path) -> None:
    """
    Optionally write the unique repo list to a text file as owner/repo per line.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in repos:
                f.write(r + "\n")
        print(f"[INFO] Saved unique repo list to: {path}")
    except Exception as e:
        print(f"[WARN] Could not write repo list file ({path}): {e}")


# ================= NEW: enable Actions + workflows ================= #

def enable_actions_for_repo(owner: str, repo: str) -> None:
    """
    Ensure GitHub Actions is enabled for this repo, and enable all workflows.
    Safe to call repeatedly; if already enabled, it just prints and returns.
    """
    # 1) Check / set Actions permissions
    perm_url = f"{API_BASE}/repos/{owner}/{repo}/actions/permissions"
    resp = requests.get(perm_url, headers=HEADERS)

    if resp.status_code == 404:
        print(f"[WARN] Cannot manage Actions for {owner}/{repo} (404).")
        return
    if resp.status_code == 403:
        print(f"[WARN] No permission to manage Actions for {owner}/{repo} (403).")
        return
    if resp.status_code != 200:
        print(
            f"[WARN] Unexpected status getting Actions permissions for "
            f"{owner}/{repo}: {resp.status_code} {resp.text[:120]}"
        )
        return

    data = resp.json()
    already_enabled = data.get("enabled", False)

    if not already_enabled:
        print(f"[INFO] Enabling Actions for {owner}/{repo}...")
        payload = {
            "enabled": True,
            "allowed_actions": "all",   # or "local_only"/"selected" if you want
        }
        set_resp = requests.put(perm_url, headers=HEADERS, json=payload)
        if set_resp.status_code not in (200, 204):
            print(
                f"[WARN] Failed to enable Actions for {owner}/{repo}: "
                f"{set_resp.status_code} {set_resp.text[:120]}"
            )
        else:
            print(f"[OK] Actions enabled for {owner}/{repo}")
    else:
        print(f"[INFO] Actions already enabled for {owner}/{repo}")

    # 2) Enable all workflows (so schedules, etc., start running)
    workflows_url = f"{API_BASE}/repos/{owner}/{repo}/actions/workflows"
    wf_resp = requests.get(workflows_url, headers=HEADERS)
    if wf_resp.status_code != 200:
        print(
            f"[WARN] Could not list workflows for {owner}/{repo}: "
            f"{wf_resp.status_code} {wf_resp.text[:120]}"
        )
        return

    workflows = wf_resp.json().get("workflows", [])
    if not workflows:
        print(f"[INFO] No workflows found in {owner}/{repo}")
        return

    for wf in workflows:
        wf_id = wf.get("id")
        wf_name = wf.get("name")
        if wf_id is None:
            continue
        enable_url = f"{API_BASE}/repos/{owner}/{repo}/actions/workflows/{wf_id}/enable"
        en_resp = requests.put(enable_url, headers=HEADERS)
        if en_resp.status_code == 204:
            print(f"  [OK] Enabled workflow: {wf_name}")
        elif en_resp.status_code == 202:
            print(f"  [OK] Workflow {wf_name} accepted for enabling (202).")
        elif en_resp.status_code == 409:
            # sometimes returned if workflow is already active
            print(f"  [INFO] Workflow already active or cannot be enabled: {wf_name}")
        else:
            print(
                f"  [WARN] Could not enable workflow {wf_name} "
                f"({wf_id}) for {owner}/{repo}: "
                f"{en_resp.status_code} {en_resp.text[:120]}"
            )

# ================================================================== #


def main():
    repos = load_unique_repos_from_dataset(DATASET_PATH)
    forked_repo_list(repos, REPO_LIST_FILE)

    print(f"[INFO] Starting forking process for {len(repos)} repos...")

    for owner_repo in repos:
        owner, repo = owner_repo.split("/", 1)
        target_repo_name = repo

        if repo_exists_in_user(USERNAME, target_repo_name):
            print(
                f"[SKIP] Already forked: {USERNAME}/{target_repo_name} "
                f"(source: {owner_repo})"
            )
            # NEW: ensure Actions & workflows are enabled on existing fork
            enable_actions_for_repo(USERNAME, target_repo_name)
            time.sleep(REQUEST_DELAY)
            continue

        print(f"[ACTION] Forking: {owner_repo} -> {USERNAME}/{target_repo_name}")
        ok = fork_repository(owner, repo)
        if not ok:
            print(f"[FAIL] Could not initiate fork for {owner_repo}")
            time.sleep(REQUEST_DELAY)
            continue

        if poll_until_exists(
            USERNAME,
            target_repo_name,
            timeout_s=POLL_SECONDS,
            interval_s=POLL_INTERVAL,
        ):
            print(f"[OK] Fork available: https://github.com/{USERNAME}/{target_repo_name}")
            # NEW: enable Actions + workflows for the new fork
            enable_actions_for_repo(USERNAME, target_repo_name)
        else:
            print(
                f"[PENDING] Fork not visible yet: https://github.com/{USERNAME}/{target_repo_name} "
                f"(may appear later on GitHub)."
            )

        time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    main()
