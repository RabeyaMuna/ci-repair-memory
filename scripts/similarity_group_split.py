#!/usr/bin/env python3
"""
Similarity-Group Split
======================

Purpose
-------
For each repository, find groups of mutually similar issues (cosine sim ≥ θ).
Within each similarity group:
    oldest 70%  →  MEMORY  (training / prior experience)
    newest 30%  →  TEST    (evaluation)

Issues with NO similar peer within their repo are EXCLUDED from both sets.

Why this design?
    The evaluation question is: "does having similar prior experience help
    solve similar future issues?"
    To answer that cleanly, every test issue MUST have at least one similar
    memory issue. Including dissimilar issues would contaminate the evaluation
    with cases where memory cannot possibly help.

Similarity formula (weighted cosine, 2 signals):
    sim(i,j) = 0.3125 × cos(error_doc_i, error_doc_j)
             + 0.6875 × cos(fix_doc_i, fix_doc_j)

error_doc priority order (WHY > WHAT > HOW > WHERE):
    1. overall_failure_reasons   ← highest: semantic narrative of WHY it failed
    2. per-file reason           ← per-file specific narrative
    3. overall_error_types + issue_type + error_subtype
    4. failed_tools + failed_command
    5. file paths                ← lowest: supporting context only

Embedding model: sentence-transformers/all-MiniLM-L6-v2
    (same as O-CRD §3.1, validated for semantic similarity on code/issue text)

Inputs:
    error_details.json        — per-issue CI failure signals
    lca_dataset.parquet       — ground-truth diffs (sha_fail join key)

Outputs (written to OUTPUT_DIR):
    similarity_groups.json         — per-repo group details
    similarity_group_split.csv     — per-issue table (group, similarity, role)
    memory_issues.json             — 70% of each group → memory
    test_issues.json               — 30% of each group → test
    excluded_issues.json           — issues with no similar peer (not used)
    split_summary.txt              — human-readable overview
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

ERROR_DETAILS_PATH = (
    "/Users/rabeyakhatunmuna/Documents/mem-ci-repair-agent/results/error_details.json"
)
PARQUET_PATH = (
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet"
)
OUTPUT_DIR = (
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/similarity_group_split"
)

# ── Parameters ─────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.50   # default similarity threshold for recurrence links
MIN_REPO_SIZE        = 3      # repos with fewer issues skipped entirely
MIN_GROUP_SIZE       = 2      # groups smaller than this have no test split
TEST_FRACTION        = 0.30   # 30% of each group → test

# ── Similarity weights ─────────────────────────────────────────────────────────

W_ERROR = 0.3125
W_FIX   = 0.6875

# ── Embedding ──────────────────────────────────────────────────────────────────

_model:   Any = None
_backend: str = "none"
_cache:   Dict[str, np.ndarray] = {}


def _load_model() -> None:
    global _model, _backend
    try:
        from sentence_transformers import SentenceTransformer
        _model   = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = "sentence_transformers"
        print("[SGS] Model: sentence-transformers/all-MiniLM-L6-v2  (O-CRD §3.1)")
        return
    except Exception as e:
        print(f"[SGS] sentence-transformers unavailable ({e}); trying fastembed…")
    try:
        from fastembed import TextEmbedding
        _model   = TextEmbedding("BAAI/bge-base-en-v1.5")
        _backend = "fastembed"
        print("[SGS] Model: fastembed/BAAI/bge-base-en-v1.5")
        return
    except Exception as e:
        raise RuntimeError(f"No embedding model available.\n{e}")


def _embed(text: str) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    if text in _cache:
        return _cache[text]
    try:
        if _backend == "sentence_transformers":
            vec = _model.encode(text, normalize_embeddings=True)
            vec = np.array(vec, dtype=np.float32)
        else:
            vec = np.array(next(_model.embed([text])), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        _cache[text] = vec
        return vec
    except Exception:
        return None


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


# ── Fix strategy ───────────────────────────────────────────────────────────────

_FIX_PATTERNS = [
    (r"\+\s*(from\s+\S+\s+import|import\s+\S+)",                    "import_fix"),
    (r"(requirements.*\.txt|pyproject\.toml|setup\.cfg|setup\.py)", "dependency_update"),
    (r"(mypy|pyright|pytype)",                                       "type_check_fix"),
    (r"(ruff|black|flake8|isort|pylint)",                            "format_fix"),
    (r"\+.*:\s*(int|str|float|bool|List|Dict|Optional|Union|Any)\b", "type_annotation_fix"),
    (r"def test_|assert |pytest\.",                                  "test_fix"),
    (r"(\.github/workflows/|on:\s|runs-on:)",                        "config_fix"),
    (r"pip install|poetry add|conda install",                        "dependency_install_fix"),
]


def _classify_fix(diff: str) -> str:
    if not diff:
        return "unknown_fix"
    for pat, label in _FIX_PATTERNS:
        if re.search(pat, diff, re.IGNORECASE):
            return label
    added   = sum(1 for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.split("\n") if l.startswith("-") and not l.startswith("---"))
    return "minor_fix" if added + removed <= 4 else "structural_fix"


def _fix_doc(failure_reasons: List[str], diff: str) -> str:
    ft    = _classify_fix(diff)
    files = re.findall(r"diff --git a/(\S+)", diff or "")
    exts  = list({Path(f).suffix.lower() for f in files if Path(f).suffix})
    complaint = ""
    for r in (failure_reasons or []):
        for s in re.split(r"[.!?]", r):
            if len(s.strip()) > 20:
                complaint = s.strip()[:250]
                break
        if complaint:
            break
    parts = [ft]
    if complaint:
        parts.append(f"resolves: {complaint}")
    if exts:
        parts.append(f"in: {' '.join(exts)}")
    return " | ".join(parts)


# ── Document builders ──────────────────────────────────────────────────────────

def _error_doc(issue: Dict) -> str:
    """
    Priority: WHY (failure reason) > WHAT (category) > HOW (tool/cmd) > WHERE (file)
    Aligned with O-CRD §3.1 — failure narrative is the primary signal.
    """
    parts: List[str] = []

    # 1. Overall failure narrative — WHY (highest priority)
    for r in (issue.get("overall_failure_reasons") or []):
        parts.append(r[:500])

    # 2. Per-file reason — specific narrative per file
    for ef in (issue.get("effected_files") or []):
        parts.append(str(ef.get("reason") or "")[:300])

    # 3. Error category labels — WHAT
    parts.extend(issue.get("overall_error_types") or [])
    for ef in (issue.get("effected_files") or []):
        parts.append(str(ef.get("issue_type")    or ""))
        parts.append(str(ef.get("error_type")    or ""))
        parts.append(str(ef.get("error_subtype") or ""))

    # 4. Tool and command — HOW
    for ef in (issue.get("effected_files") or []):
        for t in (ef.get("failed_tools") or []):
            parts.append(str(t).lower())
        parts.append((ef.get("failed_command") or "")[:150])

    # 5. File paths — WHERE (lowest priority)
    for ef in (issue.get("effected_files") or []):
        f = str(ef.get("file") or "").strip()
        if f:
            parts.append(f)

    return " | ".join(x for x in parts if x.strip())


# ── Similarity ─────────────────────────────────────────────────────────────────

def _sim(a: Dict, b: Dict) -> float:
    return (
        W_ERROR    * _cos(a.get("_ve"), b.get("_ve"))
      + W_FIX      * _cos(a.get("_vf"), b.get("_vf"))
    )


# ── Union-Find ─────────────────────────────────────────────────────────────────

class _UF:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._p[px] = py

    def groups(self) -> Dict[int, List[int]]:
        d: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self._p)):
            d[self.find(i)].append(i)
        return dict(d)


# ── Data loading and embedding ─────────────────────────────────────────────────

def _load_and_embed() -> List[Dict]:
    with open(ERROR_DETAILS_PATH) as f:
        ed_list = json.load(f)
    ed_map = {d["sha_fail"]: d for d in ed_list}

    df         = pd.read_parquet(PARQUET_PATH)
    diff_map   = dict(zip(df["sha_fail"], df["diff"].fillna("")))
    order_map  = {sha: idx for idx, sha in enumerate(df["sha_fail"])}

    issues: List[Dict] = []
    for _, row in df.iterrows():
        sha  = row["sha_fail"]
        ed   = ed_map.get(sha, {})
        issue: Dict = {
            **{"sha_fail":      sha,
               "repo":          f"{row['repo_owner']}/{row['repo_name']}",
               "workflow_name": row.get("workflow_name", ""),
               "workflow_path": row.get("workflow_path", "")},
            **ed,
        }
        issue["sha_fail"] = sha
        if not issue.get("overall_error_types"):
            issue["overall_error_types"]     = [str(row.get("error_type", ""))]
            issue["overall_failure_reasons"] = []
            issue["effected_files"]          = []
            issue["failed_jobs"]             = []
        issues.append(issue)

    total = len(issues)
    print(f"[SGS] Embedding {total} issues…")
    for idx, iss in enumerate(issues):
        diff = diff_map.get(iss["sha_fail"], "")
        iss["_ve"]         = _embed(_error_doc(iss))
        iss["_vf"]         = _embed(_fix_doc(iss.get("overall_failure_reasons") or [], diff))
        iss["_fix_type"]   = _classify_fix(diff)
        iss["_order_idx"]  = order_map.get(iss["sha_fail"], idx)
        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"[SGS]   {idx+1}/{total}")

    return issues


# ── Within-repo similarity grouping ───────────────────────────────────────────

def _group_repo(repo_issues: List[Dict]) -> List[List[Dict]]:
    """
    Single-linkage clustering within one repo.
    Two issues joined if sim ≥ SIMILARITY_THRESHOLD.
    Returns list of groups (each group is a list of issue dicts).
    """
    n  = len(repo_issues)
    uf = _UF(n)

    # Collect all pairs above threshold
    pairs: List[Tuple[float, int, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            s = _sim(repo_issues[a], repo_issues[b])
            if s >= SIMILARITY_THRESHOLD:
                pairs.append((s, a, b))

    # Merge highest-similarity pairs first
    for _, a, b in sorted(pairs, reverse=True):
        uf.union(a, b)

    return [[repo_issues[i] for i in members] for members in uf.groups().values()]


# ── Per-group split (chronological 70/30) ─────────────────────────────────────

def _split_group(group: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Sort group chronologically (parquet order_idx as proxy).
    Oldest 70% → memory.  Newest 30% → test.
    """
    ordered     = sorted(group, key=lambda x: x["_order_idx"])
    split_point = max(1, int(len(ordered) * (1 - TEST_FRACTION)))  # 70% memory
    return ordered[:split_point], ordered[split_point:]


# ── Average intra-group similarity ────────────────────────────────────────────

def _avg_sim(group: List[Dict]) -> float:
    n = len(group)
    if n < 2:
        return 0.0
    sims = [
        _sim(group[a], group[b])
        for a in range(n)
        for b in range(a + 1, n)
    ]
    return float(np.mean(sims)) if sims else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _load_model()
    issues = _load_and_embed()

    # Group by repo
    repo_idx: Dict[str, List[int]] = defaultdict(list)
    for i, iss in enumerate(issues):
        repo_idx[iss["repo"]].append(i)

    memory_issues:   List[Dict] = []
    test_issues:     List[Dict] = []
    excluded_issues: List[Dict] = []

    # Per-issue table rows
    table_rows: List[Dict] = []

    # Per-repo group report
    group_report: Dict[str, Any] = {}

    print("\n[SGS] Grouping and splitting within each repo…")

    for repo, idxs in sorted(repo_idx.items()):
        repo_issues = [issues[i] for i in idxs]
        n_repo      = len(repo_issues)

        # Skip small repos
        if n_repo < MIN_REPO_SIZE:
            for iss in repo_issues:
                excluded_issues.append(iss)
                table_rows.append(_row(iss, repo, n_repo, group_id="—",
                                       group_size=1, avg_sim=0.0,
                                       role="excluded (small repo)"))
            continue

        # Find similarity groups
        groups = _group_repo(repo_issues)

        repo_groups_report = []
        group_counter      = 0

        for group in groups:
            g_size  = len(group)
            g_sim   = _avg_sim(group)
            group_counter += 1
            gid     = f"{repo}#G{group_counter}"

            if g_size < MIN_GROUP_SIZE:
                # Singleton — no similar peer → excluded
                for iss in group:
                    excluded_issues.append(iss)
                    table_rows.append(_row(iss, repo, n_repo, gid,
                                           g_size, g_sim,
                                           "excluded (no similar peer)"))
                repo_groups_report.append({
                    "group_id":       gid,
                    "size":           g_size,
                    "avg_similarity": round(g_sim, 4),
                    "decision":       "excluded (singleton)",
                    "memory_ids":     [],
                    "test_ids":       [],
                })
                continue

            # Valid group — split chronologically
            mem, tst = _split_group(group)

            for iss in mem:
                memory_issues.append(iss)
                table_rows.append(_row(iss, repo, n_repo, gid,
                                       g_size, g_sim, "memory"))
            for iss in tst:
                test_issues.append(iss)
                table_rows.append(_row(iss, repo, n_repo, gid,
                                       g_size, g_sim, "test"))

            repo_groups_report.append({
                "group_id":       gid,
                "size":           g_size,
                "avg_similarity": round(g_sim, 4),
                "decision":       f"{len(mem)} memory / {len(tst)} test",
                "fix_types":      list({x["_fix_type"] for x in group}),
                "error_types":    list({
                    et for x in group
                    for et in (x.get("overall_error_types") or [])
                }),
                "memory_ids": [x.get("id", x["sha_fail"][:8]) for x in mem],
                "test_ids":   [x.get("id", x["sha_fail"][:8]) for x in tst],
            })

        group_report[repo] = {
            "total_issues":   n_repo,
            "n_groups":       len(groups),
            "n_valid_groups": sum(1 for g in repo_groups_report if "/" in g["decision"]),
            "groups":         repo_groups_report,
        }

    # ── Sort table by repo (issue count desc), then group, then role ───────────
    table_rows.sort(key=lambda r: (
        -r["repo_total_issues"],
        r["repo"],
        r["group_id"],
        0 if r["role"] == "memory" else (1 if r["role"] == "test" else 2),
    ))

    # ── Print table ────────────────────────────────────────────────────────────
    _print_table(table_rows)

    # ── Print repo summary ─────────────────────────────────────────────────────
    _print_repo_summary(group_report)

    # ── Write outputs ──────────────────────────────────────────────────────────
    _write_csv(table_rows)
    _write_json(memory_issues,   "memory",     "memory_issues.json")
    _write_json(test_issues,     "test",       "test_issues.json")
    _write_json(excluded_issues, "excluded",   "excluded_issues.json")

    with open(f"{OUTPUT_DIR}/similarity_groups.json", "w") as f:
        json.dump(group_report, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    from collections import Counter
    lines = [
        "",
        "=" * 65,
        "Similarity-Group Split  —  Summary",
        "=" * 65,
        f"Total issues              : {len(issues)}",
        f"Repos analysed (≥{MIN_REPO_SIZE} issues)  : "
            f"{sum(1 for r, v in group_report.items() if v['total_issues'] >= MIN_REPO_SIZE)}",
        f"Repos skipped (< {MIN_REPO_SIZE})      : "
            f"{sum(1 for i in issues if i['repo'] not in group_report)}",
        "",
        f"Issues in similar groups  : {len(memory_issues) + len(test_issues)}",
        f"  → Memory (70%)          : {len(memory_issues)}",
        f"  → Test   (30%)          : {len(test_issues)}",
        f"Issues excluded (no peer) : {len(excluded_issues)}",
        "",
        f"Similarity threshold θ    : {SIMILARITY_THRESHOLD}  (recurrence link cutoff)",
        f"Weights  error={W_ERROR}  fix={W_FIX}",
        "",
        "Test set fix-type distribution:",
    ]
    for ft, cnt in Counter(x["_fix_type"] for x in test_issues).most_common():
        lines.append(f"  {ft:<30} {cnt}")
    lines += ["", "Memory set fix-type distribution:"]
    for ft, cnt in Counter(x["_fix_type"] for x in memory_issues).most_common():
        lines.append(f"  {ft:<30} {cnt}")

    summary = "\n".join(lines)
    print(summary)
    with open(f"{OUTPUT_DIR}/split_summary.txt", "w") as f:
        f.write(summary)

    print(f"\n[SGS] Outputs → {OUTPUT_DIR}/")
    print(f"      similarity_group_split.csv  — {len(table_rows)} rows")
    print(f"      similarity_groups.json")
    print(f"      memory_issues.json          — {len(memory_issues)} issues")
    print(f"      test_issues.json            — {len(test_issues)} issues")
    print(f"      excluded_issues.json        — {len(excluded_issues)} issues")
    print(f"      split_summary.txt")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row(iss: Dict, repo: str, n_repo: int, group_id: str,
         group_size: int, avg_sim: float, role: str) -> Dict:
    return {
        "repo":             repo,
        "repo_total_issues": n_repo,
        "issue_id":         iss.get("id", iss.get("sha_fail", "")[:8]),
        "sha_fail":         iss.get("sha_fail", ""),
        "error_type":       ", ".join(iss.get("overall_error_types") or [])[:35],
        "fix_type":         iss.get("_fix_type", ""),
        "group_id":         group_id,
        "group_size":       group_size,
        "avg_group_sim":    f"{avg_sim:.4f}" if avg_sim > 0 else "—",
        "role":             role,
    }


def _print_table(rows: List[Dict], n: int = 50) -> None:
    print("\n" + "=" * 130)
    print(f"PER-ISSUE SIMILARITY-GROUP SPLIT TABLE  (first {n} rows — full table in similarity_group_split.csv)")
    print("=" * 130)
    hdr = (f"{'Repo':<38} {'N':>3}  {'ID':>5}  {'Error Type':<28}  "
           f"{'Group':>7}  {'Sz':>3}  {'AvgSim':>7}  {'Fix Type':<22}  Role")
    print(hdr)
    print("-" * 130)
    prev_repo = None
    shown     = 0
    for r in rows:
        if shown >= n:
            break
        sep = "  " if r["repo"] == prev_repo else "──"
        print(
            f"{r['repo']:<38} {r['repo_total_issues']:>3}  "
            f"{str(r['issue_id']):>5}  {r['error_type']:<28}  "
            f"{str(r['group_id']).split('#')[-1]:>7}  {r['group_size']:>3}  "
            f"{r['avg_group_sim']:>7}  {r['fix_type']:<22}  {r['role']}"
        )
        prev_repo = r["repo"]
        shown += 1


def _print_repo_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 105)
    print("PER-REPO SUMMARY  (repos with ≥ 3 issues, sorted by valid groups desc)")
    print("=" * 105)
    hdr = (f"{'Repo':<42} {'Issues':>6}  {'Groups':>6}  {'Valid':>5}  "
           f"{'Memory':>6}  {'Test':>5}  {'Excluded':>8}")
    print(hdr)
    print("-" * 105)
    rows = sorted(report.items(),
                  key=lambda kv: (-kv[1]["n_valid_groups"], -kv[1]["total_issues"]))
    for repo, v in rows:
        mem_n = sum(len(g["memory_ids"]) for g in v["groups"])
        tst_n = sum(len(g["test_ids"])   for g in v["groups"])
        exc_n = v["total_issues"] - mem_n - tst_n
        print(
            f"{repo:<42} {v['total_issues']:>6}  {v['n_groups']:>6}  "
            f"{v['n_valid_groups']:>5}  {mem_n:>6}  {tst_n:>5}  {exc_n:>8}"
        )


def _write_csv(rows: List[Dict]) -> None:
    path = f"{OUTPUT_DIR}/similarity_group_split.csv"
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def _write_json(issues: List[Dict], role: str, filename: str) -> None:
    out = []
    for iss in issues:
        out.append({
            "id":                  iss.get("id", ""),
            "sha_fail":            iss.get("sha_fail", ""),
            "repo":                iss.get("repo", ""),
            "workflow_name":       iss.get("workflow_name", ""),
            "workflow_path":       iss.get("workflow_path", ""),
            "overall_error_types": iss.get("overall_error_types", []),
            "fix_type":            iss.get("_fix_type", ""),
            "role":                role,
        })
    with open(f"{OUTPUT_DIR}/{filename}", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    run()
