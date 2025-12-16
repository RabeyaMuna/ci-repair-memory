#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import requests
from pathlib import Path
from  benchmark_functions import get_results
from evaluation_plot.error_type_base import run_error_type_accuracy_evaluation


DEBUG_FAST_FAIL = True

def debug_log(msg: str, *args) -> None:
    if not DEBUG_FAST_FAIL:
        return
    try:
        formatted = msg.format(*args)
    except Exception:
        formatted = msg
    print(f"[fast_fail_detail] {formatted}", flush=True)


def normalize_run_level_conclusion(concl: str) -> str:
    c = (concl or "").lower()
    if c in ("cancelled", "timed_out", "timeout"):
        return "failure"
    return c or ""


def parse_owner_repo_run_id(url: str):
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


def build_github_headers(token: str | None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "dataset").exists() and (p / "results").exists():
            return p
    return start.parent


def finalize_after_last_poll(
    self,
    *,
    jobs_results: list,
    jobs_ids_await: list,
    jobs_ids_invalid: list,
    stream_results_path: str,
) -> None:
    REQ_DELAY = 0.8

    debug_log(
        "Starting finalize_after_last_poll: jobs_results={}, jobs_ids_await={}, jobs_ids_invalid={}",
        len(jobs_results),
        len(jobs_ids_await),
        len(jobs_ids_invalid),
    )
    debug_log("Streaming results path: {}", stream_results_path)

    # ---------- 1) Re-check all still-waiting once via get_results() ----------
    resolved_now = []
    still_waiting = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for job in list(jobs_ids_await):
            job_id = job.get("id")
            debug_log("Re-checking job (step1): id={}, raw_job={}", job_id, job)

            try:
                job_url, conclusion = get_results(job, self.config, self.credentials)
                debug_log("get_results -> job_id={}, url={}, conclusion={}", job_id, job_url, conclusion)
            except Exception as e:
                debug_log("get_results EXCEPTION for job_id={}: {} (marking as waiting)", job_id, repr(e))
                job_url, conclusion = None, "waiting"

            conclusion = normalize_run_level_conclusion(conclusion)
            debug_log("Normalized conclusion for job_id={}: {}", job_id, conclusion)

            if conclusion in ("waiting", "queued", "in_progress", ""):
                row = dict(job)
                if job_url:
                    row["url"] = job_url
                row["conclusion"] = "waiting"
                still_waiting.append(row)

            elif conclusion == "error":
                rr = dict(job)
                rr["url"] = job_url
                rr["conclusion"] = "error"
                jobs_ids_invalid.append(rr)

            else:
                rr = dict(job)
                rr["url"] = job_url
                rr["conclusion"] = conclusion  # success or failure
                resolved_now.append(rr)
                json.dump(rr, streamf); streamf.write("\n")

            time.sleep(REQ_DELAY)

    if resolved_now:
        jobs_results.extend(resolved_now)
        debug_log("Step1 complete: newly resolved count={}, jobs_results now={}", len(resolved_now), len(jobs_results))

    debug_log("Jobs still waiting after step1: {}", len(still_waiting))

    # ---------- 2) For still-waiting rows, use GH API to infer fast-fail ----------
    token = (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or self.credentials.get("token")
    )
    headers = build_github_headers(token)

    inferred_resolved = []
    undecided_after_api = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for row in still_waiting:
            job_id = row.get("id")
            owner, repo, run_id = parse_owner_repo_run_id(row.get("url"))

            if not (owner and repo and run_id):
                undecided_after_api.append(row)
                continue

            base = f"https://api.github.com/repos/{owner}/{repo}/actions"
            try:
                jobs_url = f"{base}/runs/{run_id}/jobs?per_page=100"
                r_jobs = requests.get(jobs_url, headers=headers, timeout=20)

                jobs = (r_jobs.json().get("jobs", []) if r_jobs.ok else []) or []
                statuses = [str(j.get("status") or "").lower() for j in jobs]
                conclusions = [str(j.get("conclusion") or "").lower() for j in jobs]

                fast_fail = (
                    any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions))
                    or ("cancelled" in conclusions and "completed" in statuses)
                    or any(co == "failure" for co in conclusions)
                )

                if fast_fail:
                    rr = dict(row); rr["conclusion"] = "failure"
                    inferred_resolved.append(rr)
                    json.dump(rr, streamf); streamf.write("\n")
                    time.sleep(REQ_DELAY)
                    continue

                run_url = f"{base}/runs/{run_id}"
                r_run = requests.get(run_url, headers=headers, timeout=20)
                rj = r_run.json() if r_run.ok else {}
                run_concl = normalize_run_level_conclusion(rj.get("conclusion"))

                if run_concl in ("failure", "success"):
                    rr = dict(row); rr["conclusion"] = run_concl
                    inferred_resolved.append(rr)
                    json.dump(rr, streamf); streamf.write("\n")
                else:
                    undecided_after_api.append(row)

            except requests.RequestException:
                undecided_after_api.append(row)

            time.sleep(REQ_DELAY)

    if inferred_resolved:
        jobs_results.extend(inferred_resolved)

    # ---------- 3) Replace caller's waiting list with final undecided rows ----------
    jobs_ids_await[:] = undecided_after_api

    debug_log(
        "End finalize_after_last_poll: jobs_results={}, jobs_ids_await={}, jobs_ids_invalid={}",
        len(jobs_results),
        len(jobs_ids_await),
        len(jobs_ids_invalid),
    )

    # ---------- 4) FINAL EVALUATION (both overall + per-error-type plots) ----------
    try:
        repo_root = _find_repo_root(Path(__file__).resolve())
        stats = run_error_type_accuracy_evaluation(
            dataset_path=repo_root / "dataset" / "lca_dataset.parquet",
            success_path=repo_root / "results" / "jobs_success_diff.jsonl",
            output_dir=repo_root / "evaluation_plot",
            jobs_ids_invalid=jobs_ids_invalid,
            jobs_ids_await=jobs_ids_await,
            stream_results_path=stream_results_path,
        )
        overall = stats.get("overall", {})
        debug_log(
            "FINAL: passed={} failed={} invalid={} error={} waiting={} accuracy={}%",
            overall.get("passed"),
            overall.get("failed"),
            overall.get("invalid"),
            overall.get("error"),
            overall.get("waiting"),
            overall.get("accuracy_percent"),
        )
    except Exception as e:
        debug_log("Final evaluation FAILED: {}", repr(e))
