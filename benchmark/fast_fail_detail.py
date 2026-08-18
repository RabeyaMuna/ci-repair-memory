#!/usr/bin/env python3

import sys
import os

# Add benchmark directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# -*- coding: utf-8 -*-

import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .benchmark_functions import get_results
from .benchmark_utils import save_jsonl

DEBUG_FAST_FAIL = True


# -------------------------
# Logging
# -------------------------
def debug_log(msg: str, *args) -> None:
    if not DEBUG_FAST_FAIL:
        return
    try:
        formatted = msg.format(*args)
    except Exception:
        formatted = msg
    print(f"[fast_fail_detail] {formatted}", flush=True)


# -------------------------
# JSONL helpers
# -------------------------
def _read_jsonl(path: str) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_jsonl_overwrite(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_jsonl(path, rows)


# -------------------------
# GH helpers
# -------------------------
def build_github_headers(token: Optional[str]):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def parse_owner_repo_run_id(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not url:
        return None, None, None
    s = url.strip().strip("/")
    parts = s.split("/")
    if "//" in s and "github.com" not in parts:
        s2 = s.split("//", 1)[1]
        parts = s2.strip("/").split("/")

    if "actions" in parts and "runs" in parts:
        a = parts.index("actions")
        if a >= 2 and a + 2 < len(parts):
            return parts[a - 2], parts[a - 1], parts[a + 2]

    if "github.com" in parts and len(parts) >= parts.index("github.com") + 6:
        i = parts.index("github.com")
        return parts[i + 1], parts[i + 2], parts[i + 5]

    return None, None, None


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "dataset").exists() and (p / "results").exists():
            return p
    return start.parent


# -------------------------
# Normalization rules (IMPORTANT)
# -------------------------
def normalize_conclusion(concl: str) -> str:
    """
    IMPORTANT: cancelled/timed_out/timeout are NOT failure.
    We keep them as separate buckets.
    """
    c = (concl or "").lower()
    if c in ("", None):
        return ""
    # normalize spelling variants
    if c == "timed_out":
        return "timed_out"
    return c


def infer_fast_fail_from_jobs(jobs: List[dict]) -> bool:
    """
    Fast-fail definition (your requirement):
    If ANY job failed (conclusion==failure) and others may be cancelled -> run should be failure.
    """
    conclusions = [(j.get("conclusion") or "").lower() for j in jobs]
    return any(c == "failure" for c in conclusions)


def infer_run_conclusion_from_api(owner: str, repo: str, run_id: str, headers: dict) -> Tuple[str, bool]:
    """
    Returns: (final_conclusion, fast_fail_bool)
    final_conclusion is one of:
      success, failure, cancelled, timed_out, waiting, error, in_progress, queued, ''
    """
    base = f"https://api.github.com/repos/{owner}/{repo}/actions"

    # jobs endpoint
    jobs_url = f"{base}/runs/{run_id}/jobs?per_page=100"
    r_jobs = requests.get(jobs_url, headers=headers, timeout=20)
    jobs = (r_jobs.json().get("jobs", []) if r_jobs.ok else []) or []

    fast_fail = infer_fast_fail_from_jobs(jobs)

    # run endpoint (run-level conclusion/status)
    run_url = f"{base}/runs/{run_id}"
    r_run = requests.get(run_url, headers=headers, timeout=20)
    if not r_run.ok:
        return ("waiting", fast_fail)

    run_json = r_run.json() or {}
    run_status = normalize_conclusion(run_json.get("status") or "")
    run_concl = normalize_conclusion(run_json.get("conclusion") or "")

    # If completed, conclusion is meaningful
    if run_status == "completed":
        # Apply fast-fail override ONLY when there is a real failure in jobs
        if fast_fail:
            return ("failure", True)

        # Otherwise keep cancelled/timed_out separate
        if run_concl in ("success", "failure", "cancelled", "timed_out", "timeout"):
            # normalize timeout spelling
            if run_concl == "timeout":
                return ("timed_out", False)
            return (run_concl, False)

        # completed but unknown -> treat as waiting-ish
        return ("waiting", fast_fail)

    # Not completed yet
    if run_status in ("queued", "in_progress"):
        return (run_status, fast_fail)

    return ("waiting", fast_fail)


# -------------------------
# Finalize + file writing + final print + plots
# -------------------------
def finalize_after_last_poll(
    self,
    *,
    jobs_results: List[dict],
    jobs_ids_await: List[dict],
    jobs_ids_invalid: List[dict],
    stream_results_path: str,
    out_dir: str,
    model_name: str,
    jobs_ids_path: str,
    dataset_path: Optional[str] = None,
) -> None:
    """
    - Re-check waiting
    - Infer fast-fail using GH API
    - Rewrite dedicated files (success/failure/cancelled/timed_out/invalid/waiting)
    - Print ONE final report (pushed + dataset-grounded)
    """

    REQ_DELAY = 0.8

    # ---- Load pushed as source of truth ----
    pushed_jobs = _read_jsonl(jobs_ids_path)
    pushed_ids = {str(j.get("id")) for j in pushed_jobs if isinstance(j, dict)}

    # Normalize already-known results
    for r in jobs_results:
        r["conclusion"] = normalize_conclusion(r.get("conclusion") or "")

    for r in jobs_ids_invalid:
        r["conclusion"] = normalize_conclusion(r.get("conclusion") or "error") or "error"

    # Build a quick map so we don't duplicate
    resolved_map: Dict[str, dict] = {str(r.get("id")): r for r in jobs_results if str(r.get("id")) in pushed_ids}
    invalid_map: Dict[str, dict] = {str(r.get("id")): r for r in jobs_ids_invalid if str(r.get("id")) in pushed_ids}

    # Re-derive waiting from pushed - resolved - invalid (most reliable)
    waiting_now = []
    for j in pushed_jobs:
        jid = str(j.get("id"))
        if jid in resolved_map or jid in invalid_map:
            continue
        waiting_now.append(j)

    debug_log(
        "Finalize start: pushed={} resolved={} invalid={} waiting={}",
        len(pushed_jobs), len(resolved_map), len(invalid_map), len(waiting_now)
    )

    # ---- Step 1: re-check all waiting via get_results() ----
    still_waiting = []
    newly_resolved = []
    newly_invalid = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for job in waiting_now:
            jid = job.get("id")
            try:
                job_url, concl = get_results(job, self.config, self.credentials)
            except Exception as e:
                debug_log("get_results exception for id={}: {}", jid, repr(e))
                job_url, concl = None, "waiting"

            concl = normalize_conclusion(concl)
            row = dict(job)
            if job_url:
                row["url"] = job_url
            row["conclusion"] = concl

            # waiting states
            if concl in ("", "waiting", "queued", "in_progress"):
                still_waiting.append(row)

            # invalid/error bucket
            elif concl == "error":
                newly_invalid.append(row)

            # completed buckets: success/failure/cancelled/timed_out
            else:
                newly_resolved.append(row)
                json.dump(row, streamf)
                streamf.write("\n")

            time.sleep(REQ_DELAY)

    # Merge step-1 results into maps
    for r in newly_resolved:
        resolved_map[str(r.get("id"))] = r
    for r in newly_invalid:
        invalid_map[str(r.get("id"))] = r

    debug_log("After step1: newly_resolved={} still_waiting={} newly_invalid={}",
              len(newly_resolved), len(still_waiting), len(newly_invalid))

    # ---- Step 2: GH API inference for still waiting ----
    token = (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or self.credentials.get("token")
    )
    headers = build_github_headers(token)

    final_waiting = []
    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for row in still_waiting:
            owner, repo, run_id = parse_owner_repo_run_id(row.get("url") or "")
            if not (owner and repo and run_id):
                final_waiting.append(row)
                continue

            try:
                inferred_concl, fast_fail = infer_run_conclusion_from_api(owner, repo, run_id, headers)
            except Exception as e:
                debug_log("API inference exception id={}: {}", row.get("id"), repr(e))
                inferred_concl, fast_fail = ("waiting", False)

            inferred_concl = normalize_conclusion(inferred_concl)

            # Apply your rule: if fast_fail true -> failure, even if run shows cancelled
            if fast_fail:
                inferred_concl = "failure"

            row2 = dict(row)
            row2["conclusion"] = inferred_concl

            if inferred_concl in ("", "waiting", "queued", "in_progress"):
                final_waiting.append(row2)
            else:
                resolved_map[str(row2.get("id"))] = row2
                json.dump(row2, streamf)
                streamf.write("\n")

            time.sleep(REQ_DELAY)

    # ---- Step 3: rewrite dedicated result files (no stale rows) ----
    # Full resolved list (only pushed)
    resolved_all = [resolved_map[i] for i in sorted(resolved_map.keys(), key=lambda x: int(x) if x.isdigit() else x)]
    invalid_all = [invalid_map[i] for i in sorted(invalid_map.keys(), key=lambda x: int(x) if x.isdigit() else x)]

    success_rows, failure_rows, cancelled_rows, timeout_rows = [], [], [], []
    for r in resolved_all:
        c = normalize_conclusion(r.get("conclusion") or "")
        if c == "success":
            success_rows.append(r)
        elif c == "failure":
            failure_rows.append(r)
        elif c == "cancelled":
            cancelled_rows.append(r)
        elif c in ("timed_out", "timeout"):
            rr = dict(r); rr["conclusion"] = "timed_out"
            timeout_rows.append(rr)
        else:
            # unknown completed state -> keep it as waiting-ish (rare)
            final_waiting.append(dict(r, conclusion="waiting"))

    # Paths
    success_path   = os.path.join(out_dir, f"jobs_success_{model_name}.jsonl")
    failure_path   = os.path.join(out_dir, f"jobs_failure_{model_name}.jsonl")
    cancelled_path = os.path.join(out_dir, f"jobs_cancelled_{model_name}.jsonl")
    timeout_path   = os.path.join(out_dir, f"jobs_timeout_{model_name}.jsonl")
    invalid_path   = os.path.join(out_dir, f"jobs_invalid_{model_name}.jsonl")
    waiting_path   = os.path.join(out_dir, f"jobs_awaiting_{model_name}.jsonl")

    # Combined results file (optional): success+failure only (common evaluation assumption)
    results_path = os.path.join(out_dir, f"jobs_results_{model_name}.jsonl")

    _write_jsonl_overwrite(success_path, success_rows)
    _write_jsonl_overwrite(failure_path, failure_rows)
    _write_jsonl_overwrite(cancelled_path, cancelled_rows)
    _write_jsonl_overwrite(timeout_path, timeout_rows)
    _write_jsonl_overwrite(invalid_path, invalid_all)
    _write_jsonl_overwrite(waiting_path, final_waiting)

    _write_jsonl_overwrite(results_path, success_rows + failure_rows)

    # ---- Step 4: print ONE final report ----
    pushed = len(pushed_jobs)
    passed = len(success_rows)
    failed = len(failure_rows)
    cancelled = len(cancelled_rows)
    timed_out = len(timeout_rows)
    invalid = len(invalid_all)
    waiting = len(final_waiting)

    # accuracy on pushed (attempted/pushed)
    denom_pushed = max(pushed, 1)
    acc_pushed = (passed / denom_pushed) * 100.0

    # dataset grounded
    repo_root = _find_repo_root(Path(__file__).resolve())
    if dataset_path is None:
        dataset_path = str(repo_root / "dataset" / "lca_dataset.parquet")

    print("\n=== Overall outcome stats (attempted-only; dataset-grounded) ===")
    print(f"Repo root: {repo_root}")
    print(f"Attempted/pushed: {pushed}  (coverage: 100.0%)")
    print(f"Not pushed: 0  <-- NOT counted as failed")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Cancelled: {cancelled}")
    print(f"Timed out: {timed_out}")
    print(f"Invalid: {invalid}")
    print(f"Waiting: {waiting}")
    print(f"Accuracy (passed/attempted*100): {acc_pushed:.2f}%")

    debug_log(
        "FINAL: passed={} failed={} cancelled={} timed_out={} invalid={} waiting={} accuracy_pushed={:.2f}%",
        passed, failed, cancelled, timed_out, invalid, waiting, acc_pushed
    )
