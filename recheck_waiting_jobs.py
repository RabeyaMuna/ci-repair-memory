#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recheck waiting jobs - Root version for easy access
"""

import os
import json
import shutil
import requests
from omegaconf import OmegaConf
from benchmark.benchmark_utils import save_jsonl
from benchmark.benchmark_functions import get_results
from benchmark import CIFixBenchmark

# -----------------------------
# Helpers
# -----------------------------
def read_jsonl(path):
    """Safely read a JSONL file, skipping malformed lines."""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping invalid JSON at line {i} in {path}: {e}")
    return data


def save_overwrite(path, records):
    """Overwrite JSONL at path with records (no merge)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_jsonl(path, records)


def _headers_from_env_or_creds(creds):
    """Build GitHub API headers from GH_TOKEN/GITHUB_TOKEN or creds dict."""
    token = (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or (isinstance(creds, dict) and (creds.get("token") or creds.get("github_token")))
        or None
    )
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _extract_owner_repo_run_id(url: str):
    """
    Parse .../github.com/<owner>/<repo>/actions/runs/<run_id> WITHOUT regex.
    Returns (owner, repo, run_id) or (None, None, None).
    """
    if not url:
        return None, None, None
    parts = url.strip().strip("/").split("/")
    # Expect: https: // github.com / <owner> / <repo> / actions / runs / <run_id>
    # Find the "github.com" segment first (schema-independent).
    try:
        idx = parts.index("github.com")
    except ValueError:
        # If full URL includes scheme, e.g., https://github.com/...
        # split("//") then split("/") again to normalize:
        if "//" in url:
            host_and_path = url.split("//", 1)[1]
            parts = host_and_path.strip("/").split("/")
            try:
                idx = parts.index("github.com")
            except ValueError:
                # fall back: find "actions" then read around it
                if "actions" in parts and "runs" in parts:
                    a = parts.index("actions")
                    if a >= 2 and a + 2 < len(parts):
                        return parts[a - 2], parts[a - 1], parts[a + 2]
                return None, None, None
        else:
            if "actions" in parts and "runs" in parts:
                a = parts.index("actions")
                if a >= 2 and a + 2 < len(parts):
                    return parts[a - 2], parts[a - 1], parts[a + 2]
            return None, None, None

    if idx + 5 >= len(parts):
        return None, None, None
    owner = parts[idx + 1]
    repo = parts[idx + 2]
    # parts[idx+3] == 'actions', parts[idx+4] == 'runs'
    run_id = parts[idx + 5]
    return owner, repo, run_id


def infer_conclusion_from_api(record, creds):
    """
    If record is 'waiting/queued/in_progress/empty', fetch the run's jobs and
    compute a fast-fail-aware conclusion. Returns (conclusion, metadata) tuple.

    Metadata includes cancellation reasons and job details for debugging.
    """
    concl = (record.get("conclusion") or "").lower()
    if concl not in ("", "waiting", "queued", "in_progress", None):
        return concl, {}

    owner, repo, run_id = _extract_owner_repo_run_id(record.get("url") or "")
    if not (owner and repo and run_id):
        return concl or "waiting", {}

    headers = _headers_from_env_or_creds(creds)
    base = f"https://api.github.com/repos/{owner}/{repo}/actions"

    try:
        # Get all jobs in the run
        resp = requests.get(f"{base}/runs/{run_id}/jobs?per_page=100", headers=headers, timeout=20)
        resp.raise_for_status()
        jobs = resp.json().get("jobs", []) or []

        statuses = [str(j.get("status") or "").lower() for j in jobs]
        conclusions = [str(j.get("conclusion") or "").lower() for j in jobs]

        # Collect cancellation metadata
        cancelled_jobs = [
            {
                "job_name": j.get("name"),
                "job_id": j.get("id"),
                "conclusion": j.get("conclusion"),
                "status": j.get("status"),
                "started_at": j.get("started_at"),
                "completed_at": j.get("completed_at")
            }
            for j in jobs
            if str(j.get("conclusion") or "").lower() == "cancelled"
        ]

        # Fast-fail rules
        if any(s == "completed" and c == "failure" for s, c in zip(statuses, conclusions)):
            return "failure", {}

        # Check for cancellations (treat as separate category now)
        if ("cancelled" in conclusions):
            # Pure cancellation or mixed with completed jobs
            has_completed = "completed" in statuses
            metadata = {
                "cancelled_jobs": cancelled_jobs,
                "total_jobs": len(jobs),
                "had_completed_jobs": has_completed,
                "run_id": run_id
            }
            return "cancelled", metadata

        if any(c == "failure" for c in conclusions):
            return "failure", {}

        # If still undecided, check run-level fields
        run = requests.get(f"{base}/runs/{run_id}", headers=headers, timeout=20).json()
        run_concl = (run.get("conclusion") or "").lower()
        run_stat = (run.get("status") or "").lower()

        if run_concl == "cancelled":
            # Run-level cancellation
            metadata = {
                "run_level_cancellation": True,
                "run_id": run_id,
                "total_jobs": len(jobs),
                "event": run.get("event"),
                "triggering_actor": run.get("triggering_actor", {}).get("login") if run.get("triggering_actor") else None
            }
            return "cancelled", metadata

        if run_concl in ("failure", "success", "timed_out", "timeout"):
            return run_concl or (run_stat or "waiting"), {}

        return "waiting", {}

    except requests.RequestException as e:
        print(f"[infer_conclusion_from_api] API error for run {run_id}: {e}")
        return concl or "waiting", {}


# -----------------------------
# Failure-only fast-fail normalization (no success inference)
# -----------------------------
def normalize_failure_only(records):
    """
    Normalize ONLY failures (leave success and cancellations separate).

    Embedded jobs (r['jobs'] / r['job_list']):
      1) ANY job status=='completed' AND conclusion=='failure'  -> run 'failure'
      2) ANY job 'cancelled' -> keep as 'cancelled' (tracked separately)
      3) ANY job conclusion=='failure'                           -> run 'failure'
      4) Else unchanged

    Flat per-job rows (group by run_id/workflow_run_id/runId): same rules per group.

    Fallback (plain run-level rows, no jobs/run_id):
      - 'cancelled' stays as 'cancelled' (not converted to failure)
      - 'timed_out' treated as failure when status is 'completed' or missing.
    """
    # Case A: embedded job arrays
    if any(isinstance(r, dict) and (r.get("jobs") or r.get("job_list")) for r in records):
        out = []
        for r in records:
            if not isinstance(r, dict):
                out.append(r); continue
            jobs = r.get("jobs") or r.get("job_list") or []
            if not jobs:
                out.append(r); continue

            statuses = [((j or {}).get("status") or "").lower() for j in jobs]
            conclusions = [((j or {}).get("conclusion") or "").lower() for j in jobs]

            # Check for actual failures first
            if any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions)):
                rr = dict(r); rr["conclusion"] = "failure"; out.append(rr); continue
            if any(co == "failure" for co in conclusions):
                rr = dict(r); rr["conclusion"] = "failure"; out.append(rr); continue

            # Keep cancellations separate (don't convert to failure)
            if "cancelled" in conclusions:
                # Already has cancelled conclusion, keep it
                out.append(r); continue

            out.append(r)
        return out

    # Case B: flat per-job rows grouped by run id
    run_key = None
    for k in ("run_id", "workflow_run_id", "runId"):
        if any(isinstance(r, dict) and k in r for r in records):
            run_key = k
            break

    if run_key:
        groups = {}
        for r in records:
            if not isinstance(r, dict):
                continue
            groups.setdefault(r.get(run_key), []).append(r)

        failing_runs = set()
        cancelled_runs = set()

        for rid, rows in groups.items():
            if rid is None:
                continue
            statuses = [((row.get("status") or "").lower()) for row in rows]
            conclusions = [((row.get("conclusion") or "").lower()) for row in rows]

            # Check for actual failures
            if any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions)):
                failing_runs.add(rid); continue
            if any(co == "failure" for co in conclusions):
                failing_runs.add(rid); continue

            # Track cancellations separately
            if "cancelled" in conclusions:
                cancelled_runs.add(rid)

        out = []
        for r in records:
            if isinstance(r, dict):
                rid = r.get(run_key)
                if rid in failing_runs:
                    rr = dict(r); rr["conclusion"] = "failure"; out.append(rr)
                elif rid in cancelled_runs and (r.get("conclusion") or "").lower() != "cancelled":
                    # Mark as cancelled if in a cancelled run
                    rr = dict(r); rr["conclusion"] = "cancelled"; out.append(rr)
                else:
                    out.append(r)
            else:
                out.append(r)
        return out

    # Case C: fallback run-level rows
    out = []
    for r in records:
        if not isinstance(r, dict):
            out.append(r); continue
        concl = (r.get("conclusion") or "").lower()
        status = (r.get("status") or "").lower()
        if concl == "failure":
            rr = dict(r); rr["conclusion"] = "failure"; out.append(rr)
        elif concl in ("timed_out", "timeout") and (status == "completed" or not status):
            # Timeouts are treated as failures
            rr = dict(r); rr["conclusion"] = "failure"; out.append(rr)
        elif concl == "cancelled":
            # Keep cancellations separate
            out.append(r)
        else:
            out.append(r)
    return out


# -----------------------------
# Config / Paths
current_dir = os.getcwd()
CONFIG_PATH = os.path.join(current_dir, "config.yaml")
config = OmegaConf.load(CONFIG_PATH)
base_dir = config.get("base_dir")

results_dir       = os.path.join(base_dir, "results")
jobs_pushed_file  = os.path.join(results_dir, "jobs_ids_diff.jsonl")      # SOURCE OF TRUTH (input)

# dedicated outputs
awaiting_file     = os.path.join(results_dir, "jobs_awaiting_diff.jsonl")
success_file      = os.path.join(results_dir, "jobs_success_diff.jsonl")
failure_file      = os.path.join(results_dir, "jobs_failure_diff.jsonl")
cancelled_file    = os.path.join(results_dir, "jobs_cancelled_diff.jsonl")  # NEW: Track cancellations separately
errors_file       = os.path.join(results_dir, "jobs_error_diff.jsonl")

# combined (success + failure) for analysis
results_file      = os.path.join(results_dir, "jobs_results_diff.jsonl")

config_path       = os.path.join(base_dir, "config.yaml")

# -----------------------------
# Initialize benchmark
# -----------------------------
model_name = "diff"
bench = CIFixBenchmark(model_name, config_path)

print("\n===============================")
print(" Rechecking pushed CI jobs (ONLY jobs_ids_diff.jsonl)...")
print("===============================\n")

# -----------------------------
# Load pushed jobs
# -----------------------------
pushed_jobs = read_jsonl(jobs_pushed_file)
if not pushed_jobs:
    print(f"No pushed jobs found in {jobs_pushed_file}.")
    raise SystemExit(1)

print(f"Found {len(pushed_jobs)} pushed job(s) to check.\n")

# -----------------------------
# Poll all jobs
# -----------------------------
checked = []
total = len(pushed_jobs)
for i, job in enumerate(pushed_jobs, 1):
    if not isinstance(job, dict) or "id" not in job:
        continue

    repo = job.get("repo_name", "unknown")
    job_id = job.get("id", "?")
    existing_conclusion = job.get("conclusion", "").lower()
    existing_url = job.get("url", "")

    # Skip re-checking if already has final conclusion (success/failure)
    if existing_conclusion in ("success", "failure", "cancelled", "timeout") and existing_url:
        print(f"[{i}/{total}] {repo} (ID {job_id}) already {existing_conclusion} - skipping")
        checked.append(job)
        continue

    print(f"[{i}/{total}] Checking {repo} (ID {job_id})...", end=" ", flush=True)

    try:
        job_url, conclusion = get_results(job, bench.config, bench.credentials)
        job["url"] = job_url
        job["conclusion"] = conclusion
        print(f"→ {conclusion}")
    except Exception as e:
        print(f"→ ERROR: {e}")
        # Preserve existing conclusion if we have one
        if not job.get("conclusion"):
            job["conclusion"] = "waiting"
    checked.append(job)

# -----------------------------
# Enrich undecided rows via API (no regex)
# -----------------------------
enriched = []
for row in checked:
    c = (row.get("conclusion") or "").lower()
    if c in ("", "waiting", "queued", "in_progress", "invalid", "notfound"):
        inferred, metadata = infer_conclusion_from_api(row, bench.credentials)
        if inferred:
            row["conclusion"] = inferred
            # Store cancellation metadata for debugging/rerun decisions
            if metadata:
                row["cancellation_metadata"] = metadata
    enriched.append(row)

# -----------------------------
# Normalize failures (fast-fail)
# -----------------------------
normalized = normalize_failure_only(enriched)

# -----------------------------
# Split into dedicated outputs
# -----------------------------
success, failure, cancelled, errors, waiting = [], [], [], [], []
for job in normalized:
    repo   = job.get("repo_name", "unknown")
    branch = job.get("branch_name", "unknown")
    issue_id = job.get("id", "unknown")
    concl  = (job.get("conclusion") or "").lower()

    if concl in ("waiting", "queued", "in_progress", ""):
        waiting.append(job)
        print(f"[{repo} | {branch}] still waiting...")
    elif concl == "success":
        success.append(job)
        print(f"[{repo} | {branch}] → SUCCESS")
    elif concl == "cancelled":
        # Track cancellations separately for analysis and potential rerun
        cancelled.append(job)
        metadata = job.get("cancellation_metadata", {})
        cancelled_count = len(metadata.get("cancelled_jobs", []))
        total_jobs = metadata.get("total_jobs", "?")
        print(f"[{repo} | {branch}] → CANCELLED (ID {issue_id}, {cancelled_count}/{total_jobs} jobs)")
    elif concl in ("failure", "timed_out", "timeout"):
        # Pure failures and timeouts (not cancellations)
        failure.append(job)
        print(f"[{repo} | {branch}] → FAILURE (from {concl})")
    elif concl == "error":
        errors.append(job)
        print(f"[{repo} | {branch}] → ERROR")
    else:
        waiting.append(job)
        print(f"[{repo} | {branch}] unknown conclusion '{concl}', keeping as waiting")

# -----------------------------
# Write dedicated outputs
# -----------------------------
save_overwrite(success_file, success)
save_overwrite(failure_file, failure)
save_overwrite(cancelled_file, cancelled)
save_overwrite(errors_file,  errors)
save_overwrite(awaiting_file, waiting)

print("\n[Write-out]")
print(f"  success:   {len(success)}   → {success_file}")
print(f"  failure:   {len(failure)}   → {failure_file}")
print(f"  cancelled: {len(cancelled)} → {cancelled_file}")
print(f"  error:     {len(errors)}    → {errors_file}")
print(f"  waiting:   {len(waiting)}   → {awaiting_file}")

# Calculate accuracy (cancelled jobs are excluded from success/failure metrics)
total_completed = len(success) + len(failure) + len(errors)
if total_completed > 0:
    accuracy = len(success) / total_completed
else:
    accuracy = 0.0
# -----------------------------
# Build combined results (success + failure) for analysis
# -----------------------------
combined_results = success + failure
save_overwrite(results_file, combined_results)

print(f"\nCombined results (success+failure) written → {results_file}")
print(f"[Sanity] Combined row count: {len(combined_results)}")

print(f"\n  Accuracy (success / completed_non_cancelled): {accuracy:.4f} ({accuracy*100:.2f}%)")

# -----------------------------
# Cancellation Summary
# -----------------------------
if cancelled:
    print("\n" + "="*80)
    print(" CANCELLATION SUMMARY")
    print("="*80)
    print(f"\nTotal cancelled: {len(cancelled)}")
    print("\nCancelled issues (rerun candidates):")

    for job in cancelled:
        issue_id = job.get("id", "?")
        repo = job.get("repo_name", "unknown")
        url = job.get("url", "N/A")
        metadata = job.get("cancellation_metadata", {})

        print(f"\n  Issue ID: {issue_id}")
        print(f"    Repo: {repo}")
        print(f"    URL: {url}")

        if metadata:
            if "cancelled_jobs" in metadata:
                cancelled_job_names = [j["job_name"] for j in metadata["cancelled_jobs"]]
                print(f"    Cancelled jobs: {', '.join(cancelled_job_names)}")
                print(f"    Total jobs in run: {metadata.get('total_jobs', '?')}")
                print(f"    Had completed jobs: {metadata.get('had_completed_jobs', False)}")

            if "run_level_cancellation" in metadata:
                print(f"    Cancellation type: Run-level")
                print(f"    Trigger event: {metadata.get('event', 'unknown')}")
                if metadata.get("triggering_actor"):
                    print(f"    Triggered by: {metadata.get('triggering_actor')}")

    print(f"\n  → All cancelled issues saved to: {cancelled_file}")
    print("  → Consider rerunning these issues if cancellations were unintentional")
    print("="*80)

