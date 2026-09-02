#!/usr/bin/env python3
"""Compute observed CI durations from timestamps embedded in benchmark logs."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TIMESTAMP = re.compile(
    r"(?m)^(?:ï»¿|\ufeff)?"
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"
)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value if isinstance(value, list) else []


def log_bounds(text: str):
    matches = TIMESTAMP.findall(str(text or ""))
    if not matches:
        return None
    timestamps = [parse_timestamp(item) for item in matches]
    return min(timestamps), max(timestamps)


def summarize(values: pd.Series):
    values = values.dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "mean_seconds": round(float(values.mean()), 3),
        "median_seconds": round(float(values.median()), 3),
        "p25_seconds": round(float(values.quantile(0.25)), 3),
        "p75_seconds": round(float(values.quantile(0.75)), 3),
        "p90_seconds": round(float(values.quantile(0.90)), 3),
        "minimum_seconds": round(float(values.min()), 3),
        "maximum_seconds": round(float(values.max()), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="dataset/ci_repair_dataset.parquet", type=Path
    )
    parser.add_argument(
        "--output-dir", default="dataset_overview/ci_duration", type=Path
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset)
    rows = []
    for _, instance in frame.iterrows():
        bounds = []
        for entry in as_list(instance.get("logs")):
            if isinstance(entry, dict):
                value = log_bounds(entry.get("log", ""))
                if value:
                    bounds.append(value)

        total_jobs = int(instance.get("total_jobs") or 0)
        logged_jobs = len(bounds)
        if bounds:
            observed_start = min(start for start, _ in bounds)
            observed_end = max(end for _, end in bounds)
            wall_seconds = (observed_end - observed_start).total_seconds()
            runner_seconds = sum((end - start).total_seconds() for start, end in bounds)
        else:
            observed_start = observed_end = None
            wall_seconds = runner_seconds = None

        rows.append(
            {
                "id": instance.get("id"),
                "repo_name": instance.get("repo_name"),
                "sha_fail": instance.get("sha_fail"),
                "total_jobs": total_jobs,
                "logged_jobs": logged_jobs,
                "complete_job_log_coverage": total_jobs > 0 and logged_jobs == total_jobs,
                "observed_start": observed_start.isoformat() if observed_start else None,
                "observed_end": observed_end.isoformat() if observed_end else None,
                "observed_wall_seconds": wall_seconds,
                "observed_runner_seconds": runner_seconds,
            }
        )

    durations = pd.DataFrame(rows)
    complete = durations[durations["complete_job_log_coverage"]]
    partial = durations[~durations["complete_job_log_coverage"]]
    summary = {
        "definition": (
            "Observed wall time is max(log timestamp) minus min(log timestamp) "
            "across the job logs retained for one benchmark instance."
        ),
        "caveat": (
            "This is an exact workflow-span proxy only when logs for every job "
            "are retained; partial-log durations are lower-bound observations."
        ),
        "instances_total": len(durations),
        "instances_with_timestamps": int(durations["observed_wall_seconds"].notna().sum()),
        "instances_with_complete_job_log_coverage": len(complete),
        "instances_with_partial_job_log_coverage": len(partial),
        "complete_coverage_wall_time": summarize(complete["observed_wall_seconds"]),
        "all_observed_wall_time": summarize(durations["observed_wall_seconds"]),
        "partial_coverage_wall_time_lower_bound": summarize(partial["observed_wall_seconds"]),
        "complete_coverage_runner_time": summarize(complete["observed_runner_seconds"]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations.to_csv(args.output_dir / "per_instance_ci_durations.csv", index=False)
    (args.output_dir / "ci_duration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
