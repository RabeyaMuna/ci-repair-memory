#!/usr/bin/env python3

"""
Add a collaborator with write (push) permission to all repositories
owned by the authenticated user (here: Muna4029).
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

# ====== CONFIG ======
OWNER = ""          # your GitHub username
COLLABORATOR = "" # collaborator's GitHub username
PERMISSION = "push"         # write permission (only)

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    print("ERROR: Please set the GITHUB_TOKEN environment variable first.")
    sys.exit(1)

session = requests.Session()
session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
})


def list_owned_repos() -> list:
    """
    List all repositories owned by the authenticated user.
    Uses /user/repos with affiliation=owner so it includes private repos too.
    """
    repos = []
    page = 1

    while True:
        url = "https://api.github.com/user/repos"
        params = {
            "per_page": 100,
            "page": page,
            "affiliation": "owner",
        }
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        # make sure we only take repos actually owned by OWNER
        for r in data:
            if r.get("owner", {}).get("login") == OWNER:
                repos.append(r)

        page += 1

    return repos


def add_collaborator(owner: str, repo: str, collaborator: str, permission: str) -> None:
    """
    Add a collaborator to a single repo with the given permission.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/collaborators/{collaborator}"
    payload = {"permission": permission}
    resp = session.put(url, json=payload)

    if resp.status_code in (201, 204):
        print(f"[OK]  {collaborator} added to {owner}/{repo} with '{permission}' permission.")
    elif resp.status_code == 404:
        print(f"[SKIP] {owner}/{repo}: not found or you lack admin rights.")
    elif resp.status_code == 403:
        print(f"[FAIL] {owner}/{repo}: permission denied or rate-limited: {resp.text}")
    else:
        print(f"[FAIL] {owner}/{repo}: {resp.status_code} {resp.text}")


def main():
    print(f"Listing repos owned by {OWNER}...")
    repos = list_owned_repos()
    print(f"Found {len(repos)} repos owned by {OWNER}.\n")

    for r in repos:
        repo_name = r["name"]
        add_collaborator(OWNER, repo_name, COLLABORATOR, PERMISSION)
        # tiny sleep to be gentle with the API
        time.sleep(0.2)


if __name__ == "__main__":
    main()
