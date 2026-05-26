#!/usr/bin/env python3
"""
enrich_split_for_pipeline.py
────────────────────────────
Joins a split JSON file (output of rcss_split.py or similarity_group_split.py)
with the parquet dataset to produce pipeline-compatible issue files.

The split files only contain:  sha_fail, repo, workflow_name, workflow_path, ...
The pipeline scripts need:     sha_fail, repo_name, repo_owner, diff,
                                ground_truth_files, changed_files, id,
                                workflow_name, workflow_path, logs_summary, ...

Usage
-----
# Enrich memory issues (full fields — required for build_memory_bank.py)
python3 scripts/enrich_split_for_pipeline.py \\
    --split   results/rcss_split/memory_30pct.json \\
    --output  baselines/results/rcss/rcss_memory_seed_issues.json

# Enrich eval issues (only sha_fail + repo_name needed for run_repo_eval_subset.py)
python3 scripts/enrich_split_for_pipeline.py \\
    --split   results/rcss_split/eval_70pct.json \\
    --output  baselines/results/rcss/rcss_eval_issues.json

# SGS — memory
python3 scripts/enrich_split_for_pipeline.py \\
    --split   results/similarity_group_split/memory_issues.json \\
    --output  baselines/results/sgs/sgs_memory_seed_issues.json

# SGS — test
python3 scripts/enrich_split_for_pipeline.py \\
    --split   results/similarity_group_split/test_issues.json \\
    --output  baselines/results/sgs/sgs_eval_issues.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"


# ── helpers ───────────────────────────────────────────────────────────────────

def _jsonable(value: Any) -> Any:
    """Recursively coerce numpy / non-JSON-serialisable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return [_jsonable(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _extract_files_from_diff(diff_text: str) -> List[str]:
    """Return list of changed file paths from a unified diff."""
    return [
        m.group(1).strip().lstrip("/").replace("\\", "/")
        for m in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text or "", re.MULTILINE)
        if m.group(1).strip()
    ]


def _clip(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summarize_logs(logs: Any, basenames: set) -> str:
    """Compact log summary — pull error/failure lines and file-name hits."""
    text = json.dumps(_jsonable(logs), ensure_ascii=False)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    interesting: List[str] = []
    for line in lines:
        lower = line.lower()
        if any(tok in lower for tok in ("error", "failed", "traceback", "exception", "warning:", "assert")):
            interesting.append(line)
        elif basenames and any(n.lower() in lower for n in basenames):
            interesting.append(line)
        if len(interesting) >= 8:
            break
    if not interesting:
        interesting = lines[-6:]
    return _clip(" | ".join(interesting))


# ── repo summary table ────────────────────────────────────────────────────────

def _print_repo_table(enriched: List[Dict], split_path: Path) -> None:
    """Print a per-repo breakdown table after enrichment."""
    from collections import defaultdict, Counter

    # Determine role label from filename
    fname = split_path.stem.lower()
    if "memory" in fname:
        role_label = "MEMORY"
    elif "eval" in fname or "test" in fname:
        role_label = "EVAL"
    else:
        role_label = "SPLIT"

    # Group by repo_name
    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for iss in enriched:
        repo = str(iss.get("repo_name") or iss.get("repo") or "unknown")
        by_repo[repo].append(iss)

    # Sort by issue count descending
    rows = sorted(by_repo.items(), key=lambda x: -len(x[1]))

    # Column widths
    W_REPO  = 40
    W_N     = 5
    W_IDS   = 40
    W_ET    = 35

    sep  = "-" * (W_REPO + W_N + W_IDS + W_ET + 10)
    hdr  = (
        f"{'Repo':<{W_REPO}}  {'#':>{W_N}}  "
        f"{'Issue IDs':<{W_IDS}}  {'Error Types':<{W_ET}}"
    )

    print()
    print("=" * len(sep))
    print(f"  {role_label} ISSUES — PER-REPO BREAKDOWN  ({len(enriched)} total)")
    print("=" * len(sep))
    print(hdr)
    print(sep)

    for repo, issues in rows:
        # Issue IDs — sorted numerically where possible
        ids = sorted(
            [str(iss.get("id") or "") for iss in issues if iss.get("id")],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        ids_str = ", ".join(ids)
        if len(ids_str) > W_IDS:
            ids_str = ids_str[:W_IDS - 3] + "..."

        # Error types — aggregate counts
        et_counter: Counter = Counter()
        for iss in issues:
            for et in (iss.get("overall_error_types") or iss.get("error_types") or []):
                et_counter[str(et).strip()] += 1
        et_parts = [
            f"{et}({n})" if n > 1 else et
            for et, n in et_counter.most_common(3)
        ]
        et_str = ", ".join(et_parts)
        if len(et_str) > W_ET:
            et_str = et_str[:W_ET - 3] + "..."

        print(
            f"{repo:<{W_REPO}}  {len(issues):>{W_N}}  "
            f"{ids_str:<{W_IDS}}  {et_str:<{W_ET}}"
        )

    print(sep)
    print(f"{'TOTAL':<{W_REPO}}  {len(enriched):>{W_N}}")
    print("=" * len(sep))
    print()


# ── core ──────────────────────────────────────────────────────────────────────

def enrich(split_path: Path, dataset_path: Path, output_path: Path) -> None:
    # Load split
    split_rows: List[Dict] = json.loads(split_path.read_text(encoding="utf-8"))
    sha_set = {str(r.get("sha_fail") or "") for r in split_rows}
    print(f"[enrich] Split issues  : {len(split_rows)}")
    print(f"[enrich] Unique sha_fails: {len(sha_set)}")

    # Load parquet
    df = pd.read_parquet(str(dataset_path))
    df["sha_fail"] = df["sha_fail"].astype(str)
    pq = df[df["sha_fail"].isin(sha_set)].copy()
    pq_by_sha: Dict[str, Any] = {row["sha_fail"]: row for row in pq.to_dict(orient="records")}
    print(f"[enrich] Parquet matches : {len(pq_by_sha)} / {len(sha_set)}")

    enriched: List[Dict] = []
    missing: List[str] = []

    for split_row in split_rows:
        sha = str(split_row.get("sha_fail") or "")
        pq_row = pq_by_sha.get(sha)

        if pq_row is None:
            print(f"  [WARN] sha not found in parquet: {sha[:16]}...")
            missing.append(sha)
            # Keep split row as-is (pipeline will skip for L1/L2/L3 if no diff)
            enriched.append(_jsonable(split_row))
            continue

        # Derive ground_truth_files from diff
        diff_text   = str(pq_row.get("diff") or "")
        gt_files    = _extract_files_from_diff(diff_text)

        cf_raw = pq_row.get("changed_files")
        if cf_raw is None:
            cf_raw = []
        elif not isinstance(cf_raw, list):
            try:
                cf_raw = list(cf_raw)  # numpy array etc.
            except Exception:
                cf_raw = [cf_raw]
        changed_files = [str(f).strip().lstrip("/").replace("\\", "/") for f in cf_raw if f]

        basenames = {Path(f).name for f in (gt_files or changed_files) if f}

        # Logs summary
        logs_raw = pq_row.get("logs")
        if logs_raw is None:
            logs_raw = []
        logs_summary = _summarize_logs(logs_raw, basenames)

        # Error types (from parquet 'error_type' column)
        et_raw = pq_row.get("error_type")
        if et_raw is None:
            error_types = []
        elif isinstance(et_raw, list):
            error_types = [str(x) for x in et_raw]
        else:
            try:
                import numpy as np
                if isinstance(et_raw, np.ndarray):
                    error_types = [str(x) for x in et_raw.tolist()]
                else:
                    error_types = [str(et_raw)]
            except Exception:
                error_types = [str(et_raw)]

        merged = {
            # Core identifiers
            "id":                str(pq_row.get("id") or split_row.get("id") or ""),
            "sha_fail":          sha,
            "sha_success":       str(pq_row.get("sha_success") or ""),
            # Repo fields — both formats for compatibility
            "repo_owner":        str(pq_row.get("repo_owner") or ""),
            "repo_name":         str(pq_row.get("repo_name")  or ""),
            "repo":              f"{pq_row.get('repo_owner', '')}/{pq_row.get('repo_name', '')}",
            # Workflow
            "workflow_name":     str(pq_row.get("workflow_name")     or split_row.get("workflow_name") or ""),
            "workflow_filename":  str(pq_row.get("workflow_filename") or ""),
            "workflow_path":     str(pq_row.get("workflow_path")     or split_row.get("workflow_path") or ""),
            "workflow":          str(pq_row.get("workflow")          or ""),
            # Logs
            "logs_summary":      logs_summary,
            # Error / type info
            "error_types":       _jsonable(error_types),
            "primary_error_type": error_types[0] if error_types else "",
            # Files
            "changed_files":     _jsonable(changed_files),
            "ground_truth_files": gt_files,
            # Ground truth diff
            "diff":              diff_text,
            # Pass-through split metadata
            "fix_type":          split_row.get("fix_type") or "",
            "overall_error_types": _jsonable(split_row.get("overall_error_types") or error_types),
            "split_role":        split_row.get("split_role") or split_row.get("role") or "",
        }

        # Preserve any extra split-specific fields (recurrence stats, group info, etc.)
        for k, v in split_row.items():
            if k not in merged:
                merged[k] = _jsonable(v)

        enriched.append(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[enrich] Written → {output_path}")
    print(f"  Total enriched : {len(enriched)}")
    print(f"  Missing in parquet: {len(missing)}")
    if missing:
        print("  Missing SHAs:", missing[:5])

    _print_repo_table(enriched, split_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a split JSON with parquet fields for use with pipeline scripts."
    )
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="Path to split JSON (memory_30pct.json, eval_70pct.json, memory_issues.json, test_issues.json, ...)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for enriched JSON",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to lca_dataset.parquet  (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    enrich(
        split_path=args.split,
        dataset_path=args.dataset,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
