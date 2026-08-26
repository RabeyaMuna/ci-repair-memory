#!/usr/bin/env python3
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    Timeout,
    RequestException,
)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------
# Paths / Config
# -----------------------------
INPUT_JSONL = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/jobs_failure_diff.jsonl")
OUTPUT_JSON = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/failed_job_logs.json")

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------------
# GitHub auth / session
# -----------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise SystemExit("GITHUB_TOKEN not found in environment variables.")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "CI-REPAIR-BENCH-fetch-failed-logs",
}

session = requests.Session()
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# -----------------------------
# Helpers
# -----------------------------
def github_api_get(url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            resp = session.get(url, headers=HEADERS, params=params, timeout=(10, 60))

            # Rate limit handling
            if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers:
                remaining = int(resp.headers.get("X-RateLimit-Remaining", "1"))
                if remaining == 0:
                    reset_time = int(resp.headers.get("X-RateLimit-Reset", time.time()))
                    wait = max(reset_time - int(time.time()) + 5, 5)
                    logging.warning(f"Rate limit reached. Sleeping {wait}s...")
                    time.sleep(wait)
                    continue

            # Retry on server errors
            if resp.status_code in (500, 502, 503, 504):
                wait = min(60, 2 ** attempt)
                logging.warning(f"Server error {resp.status_code} for {url}, retrying in {wait}s...")
                time.sleep(wait)
                continue

            if resp.ok:
                return resp.json()

            logging.error(f"GitHub API error {resp.status_code} for {url}: {resp.text[:200]}")
            return None

        except (ChunkedEncodingError, ConnectionError, Timeout, RequestException) as e:
            wait = min(60, 2 ** attempt)
            logging.warning(f"Connection error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logging.error(f"Failed to fetch after {max_retries} retries: {url}")
    return None


def parse_owner_repo_run_from_url(url: str) -> Tuple[str, str, int]:
    """
    https://github.com/<owner>/<repo>/actions/runs/<run_id>
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] != "actions" or parts[3] != "runs":
        raise ValueError(f"Cannot parse owner/repo/run_id from URL: {url}")

    owner = parts[0]
    repo = parts[1]
    run_id = int(parts[4])
    return owner, repo, run_id


def get_failed_jobs_and_logs(owner: str, repo: str, run_id: int) -> List[Dict[str, str]]:
    """
    Returns list of { "step_name": <job_name>, "log": <log_text> } for jobs with conclusion == "failure".
    NOTE: This keeps your current .text behavior (even though logs are often zipped).
    """
    jobs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/jobs"
    jobs_data = github_api_get(jobs_url)
    failed_jobs: List[Dict[str, str]] = []

    if not jobs_data or "jobs" not in jobs_data:
        logging.warning(f"No jobs data for run {run_id} in {owner}/{repo}")
        return failed_jobs

    for job in jobs_data["jobs"]:
        if job.get("conclusion") != "failure":
            continue

        job_name = job.get("name", f"job_{job.get('id')}")
        job_id = job["id"]
        log_url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"

        try:
            log_resp = session.get(log_url, headers=HEADERS, timeout=(10, 120))
        except (Timeout, RequestException) as e:
            logging.warning(f"Error fetching job log from {log_url}: {e}")
            continue

        # Handle rate limit for logs
        if log_resp.status_code == 403 and "X-RateLimit-Reset" in log_resp.headers:
            reset_time = int(log_resp.headers.get("X-RateLimit-Reset", time.time()))
            wait = max(reset_time - int(time.time()) + 5, 5)
            logging.warning(f"Rate limit hit on job logs. Waiting {wait}s...")
            time.sleep(wait)
            try:
                log_resp = session.get(log_url, headers=HEADERS, timeout=(10, 120))
            except (Timeout, RequestException) as e:
                logging.warning(f"Error fetching job log from {log_url} after wait: {e}")
                continue

        if not log_resp.ok:
            logging.warning(f"Failed to fetch job log {log_url}: status {log_resp.status_code}")
            continue

        log_text = log_resp.text  # keeping your original behavior

        failed_jobs.append({"step_name": job_name, "log": log_text})

    return failed_jobs


# -----------------------------
# Main logic
# -----------------------------
def run():
    if not INPUT_JSONL.exists():
        raise SystemExit(f"Input JSONL not found: {INPUT_JSONL}")

    # Group entries by id, keep the "most recent" by run_id (higher run_id usually means newer run)
    latest_by_id: Dict[str, Dict[str, Any]] = {}

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line: {line[:200]}")
                continue

            entry_id = entry.get("id")
            url = entry.get("url")
            if entry_id is None or not url:
                continue

            entry_id = str(entry_id)

            # "use id and commits": keep only lines that have a commit (or sha_fail)
            # (Your file uses "commit" as the failing SHA in the snippet)
            if not entry.get("commit") and not entry.get("sha_fail"):
                continue

            try:
                owner, repo, run_id = parse_owner_repo_run_from_url(url)
            except ValueError as e:
                logging.warning(str(e))
                continue

            prev = latest_by_id.get(entry_id)
            if prev is None:
                latest_by_id[entry_id] = {"entry": entry, "owner": owner, "repo": repo, "run_id": run_id}
            else:
                if run_id > prev["run_id"]:
                    latest_by_id[entry_id] = {"entry": entry, "owner": owner, "repo": repo, "run_id": run_id}

    results: List[Dict[str, Any]] = []

    # Now fetch logs for the latest failing run per id
    for entry_id, packed in sorted(latest_by_id.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        owner = packed["owner"]
        repo = packed["repo"]
        run_id = packed["run_id"]

        logging.info(f"Processing latest run for id={entry_id}: {owner}/{repo} run_id={run_id}")

        # Confirm run is failed
        run_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}"
        run_data = github_api_get(run_url)
        if not run_data:
            logging.warning(f"Could not fetch run data for id={entry_id}, run_id={run_id}")
            continue

        if run_data.get("conclusion") != "failure":
            logging.info(f"Latest run for id={entry_id} is not failure (conclusion={run_data.get('conclusion')}), skipping.")
            continue

        failed_jobs = get_failed_jobs_and_logs(owner, repo, run_id)
        if not failed_jobs:
            logging.info(f"No failed jobs found for id={entry_id}, run_id={run_id}")
            continue

        # Output exactly like your example:
        results.append({
            "id": int(entry_id) if entry_id.isdigit() else entry_id,
            "failed_jobs": failed_jobs,
        })

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)

    logging.info(f"Done. Wrote {len(results)} entries to {OUTPUT_JSON}")


if __name__ == "__main__":
    run()
