#!/usr/bin/env python3
"""
repo_recurrence_stats.py

Compute repo-level recurrence and transfer precision statistics to justify
which repositories are suitable for memory-based CI repair experiments.

Definitions
-----------
Recurring pair
  Two issues in the same repository form a recurring pair if their combined
  multi-signal similarity is at least the configured threshold:

    score = 0.20 * error_similarity
          + 0.15 * tool_similarity
          + 0.30 * file_similarity
          + 0.35 * text_similarity

Recurring issue
  An issue is counted as recurring if it appears in at least one recurring pair.

Precision
  Among recurring pairs where both issues have ground-truth patched files,
  precision is the fraction whose patched-file basenames overlap. This estimates
  whether retrieved recurrence is actionable for file-level transfer, not merely
  superficially similar in text.

File-pattern/reason precision
  A stricter actionable precision. Among recurring pairs with ground-truth
  patched files on both sides, count a pair as precise only if:
    1. patched-file basenames overlap, and
    2. failure pattern matches strongly OR failure reason text is strongly similar

Outputs
-------
  - baselines/results/repo_recurrence_stats.json
  - baselines/results/repo_recurrence_stats.csv
  - baselines/results/repo_recurrence_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "baselines") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "baselines"))

from prepare_memory_seed_split import (
    DEFAULT_DATASET,
    DEFAULT_PATCHES,
    TARGET_REPOS,
    _prepare_rows,
)
from utilities.memory_plugin import _cosine_similarity, _jaccard

DEFAULT_OUTPUT = PROJECT_ROOT / "baselines" / "results"


SIMILARITY_THRESHOLD = 0.55
SIGNAL_WEIGHTS = {
    "error": 0.20,
    "tool": 0.15,
    "file": 0.30,
    "text": 0.35,
}
PATTERN_SIMILARITY_THRESHOLD = 0.80
REASON_SIMILARITY_THRESHOLD = 0.55


def _norm_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _basename(path: str) -> str:
    return os.path.basename((path or "").strip().replace("\\", "/"))


def _gt_basenames(row: Dict[str, Any]) -> set[str]:
    return {
        _basename(str(path))
        for path in (row.get("ground_truth_files") or [])
        if _basename(str(path))
    }


def _all_file_basenames(row: Dict[str, Any]) -> set[str]:
    paths = list(row.get("ground_truth_files") or []) + list(row.get("changed_files") or [])
    return {_basename(str(path)) for path in paths if _basename(str(path))}


def _query_text(row: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("primary_error_type") or ""),
            str(row.get("issue_type") or ""),
            " ".join(str(x) for x in (row.get("failed_tool") or [])),
            str(row.get("logs_summary") or row.get("error_context_summary") or ""),
            " ".join(str(x) for x in (row.get("ground_truth_files") or [])),
        ]
    ).strip()


def _pair_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, float]:
    left_error = _norm_text(str(left.get("primary_error_type") or ""))
    right_error = _norm_text(str(right.get("primary_error_type") or ""))
    left_issue = _norm_text(str(left.get("issue_type") or ""))
    right_issue = _norm_text(str(right.get("issue_type") or ""))
    left_tools = [_norm_text(str(x)) for x in (left.get("failed_tool") or []) if _norm_text(str(x))]
    right_tools = [_norm_text(str(x)) for x in (right.get("failed_tool") or []) if _norm_text(str(x))]
    left_files = _all_file_basenames(left)
    right_files = _all_file_basenames(right)
    left_reason = _norm_text(str(left.get("logs_summary") or left.get("error_context_summary") or ""))
    right_reason = _norm_text(str(right.get("logs_summary") or right.get("error_context_summary") or ""))

    pattern_similarity = 1.0 if left_issue and left_issue == right_issue else _cosine_similarity(
        left_issue,
        right_issue,
    )
    error_similarity = 1.0 if left_error and left_error == right_error else _cosine_similarity(
        f"{left_error} {left_issue}".strip(),
        f"{right_error} {right_issue}".strip(),
    )
    tool_similarity = _jaccard(left_tools, right_tools)
    file_similarity = _jaccard(left_files, right_files)
    text_similarity = _cosine_similarity(_query_text(left), _query_text(right))
    reason_similarity = _cosine_similarity(left_reason, right_reason)
    score = (
        SIGNAL_WEIGHTS["error"] * error_similarity
        + SIGNAL_WEIGHTS["tool"] * tool_similarity
        + SIGNAL_WEIGHTS["file"] * file_similarity
        + SIGNAL_WEIGHTS["text"] * text_similarity
    )
    return {
        "error_similarity": round(error_similarity, 4),
        "pattern_similarity": round(pattern_similarity, 4),
        "tool_similarity": round(tool_similarity, 4),
        "file_similarity": round(file_similarity, 4),
        "text_similarity": round(text_similarity, 4),
        "reason_similarity": round(reason_similarity, 4),
        "score": round(score, 4),
    }


def _top_patterns(rows: Iterable[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    counter: Counter[Tuple[str, str]] = Counter(
        (
            _norm_text(str(row.get("primary_error_type") or "")) or "unknown",
            _norm_text(str(row.get("issue_type") or "")) or "unknown",
        )
        for row in rows
    )
    out: List[Dict[str, Any]] = []
    for (primary_error, issue_type), count in counter.most_common(limit):
        out.append(
            {
                "primary_error_type": primary_error,
                "issue_type": issue_type,
                "count": count,
            }
        )
    return out


def summarize_repo(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    recurring_issue_ids: set[str] = set()
    precise_issue_ids: set[str] = set()
    file_pattern_reason_precise_issue_ids: set[str] = set()
    recurring_pairs = 0
    precise_pairs = 0
    file_pattern_reason_precise_pairs = 0
    pairs_with_gt = 0
    signal_sums = defaultdict(float)
    pair_examples: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            sims = _pair_similarity(left, right)
            if sims["score"] < SIMILARITY_THRESHOLD:
                continue

            recurring_pairs += 1
            recurring_issue_ids.add(left["id"])
            recurring_issue_ids.add(right["id"])
            for key, value in sims.items():
                signal_sums[key] += value

            gt_overlap = sorted(_gt_basenames(left) & _gt_basenames(right))
            if _gt_basenames(left) and _gt_basenames(right):
                pairs_with_gt += 1
                if gt_overlap:
                    precise_pairs += 1
                    precise_issue_ids.add(left["id"])
                    precise_issue_ids.add(right["id"])
                    if (
                        sims["pattern_similarity"] >= PATTERN_SIMILARITY_THRESHOLD
                        or sims["reason_similarity"] >= REASON_SIMILARITY_THRESHOLD
                    ):
                        file_pattern_reason_precise_pairs += 1
                        file_pattern_reason_precise_issue_ids.add(left["id"])
                        file_pattern_reason_precise_issue_ids.add(right["id"])

            pair_rows.append(
                {
                    "repo": left["repo_name"],
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "left_error_type": left.get("primary_error_type", ""),
                    "right_error_type": right.get("primary_error_type", ""),
                    "left_issue_type": left.get("issue_type", ""),
                    "right_issue_type": right.get("issue_type", ""),
                    "left_failed_tools": "; ".join(left.get("failed_tool") or []),
                    "right_failed_tools": "; ".join(right.get("failed_tool") or []),
                    "left_ground_truth_files": "; ".join(left.get("ground_truth_files") or []),
                    "right_ground_truth_files": "; ".join(right.get("ground_truth_files") or []),
                    "ground_truth_overlap": "; ".join(gt_overlap),
                    "recurring_pair": True,
                    "file_overlap": bool(gt_overlap),
                    "pattern_match_strong": sims["pattern_similarity"] >= PATTERN_SIMILARITY_THRESHOLD,
                    "reason_match_strong": sims["reason_similarity"] >= REASON_SIMILARITY_THRESHOLD,
                    "precise_file_pattern_reason_pair": bool(
                        gt_overlap and (
                            sims["pattern_similarity"] >= PATTERN_SIMILARITY_THRESHOLD
                            or sims["reason_similarity"] >= REASON_SIMILARITY_THRESHOLD
                        )
                    ),
                    "score": sims["score"],
                    "error_similarity": sims["error_similarity"],
                    "pattern_similarity": sims["pattern_similarity"],
                    "tool_similarity": sims["tool_similarity"],
                    "file_similarity": sims["file_similarity"],
                    "text_similarity": sims["text_similarity"],
                    "reason_similarity": sims["reason_similarity"],
                    "left_text": _query_text(left),
                    "right_text": _query_text(right),
                }
            )

            if len(pair_examples) < 5:
                pair_examples.append(
                    {
                        "left_id": left["id"],
                        "right_id": right["id"],
                        "left_error": left.get("primary_error_type"),
                        "right_error": right.get("primary_error_type"),
                        "score": sims["score"],
                        "error_similarity": sims["error_similarity"],
                        "pattern_similarity": sims["pattern_similarity"],
                        "tool_similarity": sims["tool_similarity"],
                        "file_similarity": sims["file_similarity"],
                        "text_similarity": sims["text_similarity"],
                        "reason_similarity": sims["reason_similarity"],
                        "ground_truth_overlap": gt_overlap,
                    }
                )

    repo_name = rows[0]["repo_name"] if rows else ""
    recurring_rows = [row for row in rows if row["id"] in recurring_issue_ids]
    recurring_pct = len(recurring_issue_ids) / len(rows) if rows else 0.0
    pair_precision_pct = precise_pairs / pairs_with_gt if pairs_with_gt else 0.0
    file_pattern_reason_precision_pct = (
        file_pattern_reason_precise_pairs / pairs_with_gt if pairs_with_gt else 0.0
    )
    issue_precision_pct = (
        len(precise_issue_ids) / len(recurring_issue_ids) if recurring_issue_ids else 0.0
    )
    file_pattern_reason_issue_precision_pct = (
        len(file_pattern_reason_precise_issue_ids) / len(recurring_issue_ids)
        if recurring_issue_ids
        else 0.0
    )

    return {
        "repo": repo_name,
        "issues": len(rows),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "recurring_issues": len(recurring_issue_ids),
        "recurring_pct": round(recurring_pct, 4),
        "recurring_pairs": recurring_pairs,
        "pair_precision_eligible": pairs_with_gt,
        "precise_pairs": precise_pairs,
        "pair_precision_pct": round(pair_precision_pct, 4),
        "file_pattern_reason_precise_pairs": file_pattern_reason_precise_pairs,
        "file_pattern_reason_precision_pct": round(file_pattern_reason_precision_pct, 4),
        "precise_recurring_issues": len(precise_issue_ids),
        "issue_precision_pct": round(issue_precision_pct, 4),
        "file_pattern_reason_precise_issues": len(file_pattern_reason_precise_issue_ids),
        "file_pattern_reason_issue_precision_pct": round(file_pattern_reason_issue_precision_pct, 4),
        "avg_score": round(signal_sums["score"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "avg_error_similarity": round(signal_sums["error_similarity"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "avg_pattern_similarity": round(signal_sums["pattern_similarity"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "avg_tool_similarity": round(signal_sums["tool_similarity"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "avg_file_similarity": round(signal_sums["file_similarity"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "avg_text_similarity": round(signal_sums["text_similarity"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "avg_reason_similarity": round(signal_sums["reason_similarity"] / recurring_pairs, 4) if recurring_pairs else 0.0,
        "top_patterns": _top_patterns(recurring_rows),
        "example_pairs": pair_examples,
    }, pair_rows


def save_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "repo",
        "issues",
        "recurring_issues",
        "recurring_pct",
        "recurring_pairs",
        "pair_precision_eligible",
        "precise_pairs",
        "pair_precision_pct",
        "file_pattern_reason_precise_pairs",
        "file_pattern_reason_precision_pct",
        "precise_recurring_issues",
        "issue_precision_pct",
        "file_pattern_reason_precise_issues",
        "file_pattern_reason_issue_precision_pct",
        "avg_score",
        "avg_error_similarity",
        "avg_pattern_similarity",
        "avg_tool_similarity",
        "avg_file_similarity",
        "avg_text_similarity",
        "avg_reason_similarity",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_pairs_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    fields = [
        "repo",
        "left_id",
        "right_id",
        "left_error_type",
        "right_error_type",
        "left_issue_type",
        "right_issue_type",
        "left_failed_tools",
        "right_failed_tools",
        "left_ground_truth_files",
        "right_ground_truth_files",
        "ground_truth_overlap",
        "recurring_pair",
        "file_overlap",
        "pattern_match_strong",
        "reason_match_strong",
        "precise_file_pattern_reason_pair",
        "score",
        "error_similarity",
        "pattern_similarity",
        "tool_similarity",
        "file_similarity",
        "text_similarity",
        "reason_similarity",
        "left_text",
        "right_text",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute repo-level recurrence and transfer precision statistics."
    )
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    all_rows, _gt_rows = _prepare_rows(args.dataset, args.patches)

    repo_rows = []
    pair_rows: List[Dict[str, Any]] = []
    for repo in TARGET_REPOS:
        rows = [row for row in all_rows if row["repo_name"] == repo]
        repo_summary, repo_pairs = summarize_repo(rows)
        repo_rows.append(repo_summary)
        pair_rows.extend(repo_pairs)

    payload = {
        "definitions": {
            "similarity_score": "0.20*error_similarity + 0.15*tool_similarity + 0.30*file_similarity + 0.35*text_similarity",
            "recurring_pair": f"Same-repo issue pair with similarity score >= {SIMILARITY_THRESHOLD:.2f}.",
            "recurring_issue": "Issue that appears in at least one recurring pair.",
            "pair_precision_pct": "Among recurring pairs where both issues have ground-truth patched files, the fraction with overlapping patched-file basenames.",
            "file_pattern_reason_precision_pct": (
                "Among recurring pairs where both issues have ground-truth patched files, "
                "the fraction with patched-file basename overlap and either strong failure-pattern "
                f"similarity (>= {PATTERN_SIMILARITY_THRESHOLD:.2f}) or strong failure-reason "
                f"text similarity (>= {REASON_SIMILARITY_THRESHOLD:.2f})."
            ),
            "issue_precision_pct": "Among recurring issues, the fraction that participate in at least one precise recurring pair.",
        },
        "repos": repo_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.output_dir / "repo_recurrence_stats.json"
    csv_out = args.output_dir / "repo_recurrence_stats.csv"
    pairs_csv_out = args.output_dir / "repo_recurrence_pairs.csv"
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(repo_rows, csv_out)
    save_pairs_csv(pair_rows, pairs_csv_out)

    print("\nRepo recurrence stats\n")
    print(
        f"{'Repo':<10} {'Issues':>6} {'Recurring%':>12} {'File+Pat/Reason%':>18} {'AvgScore':>10} {'Pairs':>8}"
    )
    print("-" * 68)
    for row in repo_rows:
        print(
            f"{row['repo']:<10} "
            f"{row['issues']:>6} "
            f"{row['recurring_pct'] * 100:>11.1f}% "
            f"{row['file_pattern_reason_precision_pct'] * 100:>17.1f}% "
            f"{row['avg_score']:>10.3f} "
            f"{row['recurring_pairs']:>8}"
        )

    print(f"\nJSON saved -> {json_out}")
    print(f"CSV saved  -> {csv_out}")
    print(f"Pairs CSV saved -> {pairs_csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
