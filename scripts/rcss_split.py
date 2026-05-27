#!/usr/bin/env python3
"""
RCSS: Recurrence-Coherence Stratified Split
============================================

Checks similarity and recurrence of CI failures within each repository,
produces two analysis tables, then splits issues into memory (30%) and
evaluation (70%) sets.

Algorithm aligned with O-CRD §3.1 (Li et al., 2026):
  - Same embedding model: All-MiniLM-L6-v2 (Sentence-Transformers)
  - Same cosine similarity within same repository
  - Extended: two-signal weighted similarity formula
  - Extended: explicit recurrence count/percentage scoring per issue

Inputs:
    error_details.json        — per-issue CI failure signals
    lca_dataset.parquet       — ground-truth diffs (sha_fail join key)

Outputs:
    recurrence_per_issue.csv  — per-issue recurrence table
    recurrence_per_repo.csv   — per-repo summary table
    memory_30pct.json         — top 30% recurring issues → L1/L2/L3 memory
    eval_70pct.json           — remaining issues → evaluation benchmark
    split_summary.txt         — human-readable overview
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

from dataset_source import get_ci_repair_dataset_path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERROR_DETAILS_PATH = os.getenv("ERROR_DETAILS_PATH", str(PROJECT_ROOT / "results" / "error_details.json"))
PARQUET_PATH = get_ci_repair_dataset_path(PROJECT_ROOT)
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "rcss_split")

# ── Algorithm parameters ───────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.50   # default similarity threshold for recurrence links
MIN_REPO_SIZE        = 3      # repos with fewer issues are excluded from split
MEMORY_FRACTION      = 0.30   # top 30% recurring issues → memory

# ── Similarity weights ─────────────────────────────────────────────────────────

W_ERROR = 0.3125
W_FIX   = 0.6875

# ── Embedding model ────────────────────────────────────────────────────────────

_model: Any   = None
_backend: str = "none"
_cache: Dict[str, np.ndarray] = {}


def _load_model() -> None:
    global _model, _backend
    try:
        from sentence_transformers import SentenceTransformer
        _model   = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = "sentence_transformers"
        print("[RCSS] Embedding model: sentence-transformers/all-MiniLM-L6-v2  (same as O-CRD §3.1)")
        return
    except Exception as e:
        print(f"[RCSS] sentence-transformers unavailable ({e}); trying fastembed…")

    try:
        from fastembed import TextEmbedding
        _model   = TextEmbedding("BAAI/bge-base-en-v1.5")
        _backend = "fastembed"
        print("[RCSS] Embedding model: fastembed/BAAI/bge-base-en-v1.5")
        return
    except Exception as e:
        raise RuntimeError(f"No embedding model available. Install sentence-transformers.\n{e}")


def embed(text: str) -> Optional[np.ndarray]:
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


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


# ── Fix strategy extraction ────────────────────────────────────────────────────

_FIX_PATTERNS: List[Tuple[str, str]] = [
    (r"\+\s*(from\s+\S+\s+import|import\s+\S+)",                    "import_fix"),
    (r"(requirements.*\.txt|pyproject\.toml|setup\.cfg|setup\.py)", "dependency_update"),
    (r"(mypy|pyright|pytype)",                                       "type_check_fix"),
    (r"(ruff|black|flake8|isort|pylint)",                            "format_fix"),
    (r"\+.*:\s*(int|str|float|bool|List|Dict|Optional|Union|Any)\b", "type_annotation_fix"),
    (r"def test_|assert |pytest\.",                                  "test_fix"),
    (r"(\.github/workflows/|on:\s|runs-on:)",                        "config_fix"),
    (r"pip install|poetry add|conda install",                        "dependency_install_fix"),
]


def classify_fix(diff: str) -> str:
    if not diff:
        return "unknown_fix"
    for pattern, label in _FIX_PATTERNS:
        if re.search(pattern, diff, re.IGNORECASE):
            return label
    added   = sum(1 for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.split("\n") if l.startswith("-") and not l.startswith("---"))
    return "minor_fix" if added + removed <= 4 else "structural_fix"


def _changed_file_exts(diff: str) -> List[str]:
    files = re.findall(r"diff --git a/(\S+)", diff or "")
    return list({Path(f).suffix.lower() for f in files if Path(f).suffix})


def build_fix_doc(failure_reasons: List[str], diff: str) -> str:
    """
    Derives a semantic fix-strategy descriptor from failure reasoning + diff.
    Two issues requiring the same class of fix will embed close together even
    if the surface error messages differ — this is the key signal for recurrence.
    """
    fix_type  = classify_fix(diff)
    file_exts = _changed_file_exts(diff)

    complaint = ""
    for reason in (failure_reasons or []):
        for sentence in re.split(r"[.!?]", reason):
            sentence = sentence.strip()
            if len(sentence) > 20:
                complaint = sentence[:250]
                break
        if complaint:
            break

    parts = [fix_type]
    if complaint:
        parts.append(f"resolves: {complaint}")
    if file_exts:
        parts.append(f"in: {' '.join(file_exts)}")
    return " | ".join(parts)


# ── Document builders ──────────────────────────────────────────────────────────

def build_error_doc(issue: Dict) -> str:
    """
    What broke — failure-reason-first document, aligned with O-CRD §3.1.

    O-CRD encodes 'issue summaries' (the failure narrative) as the primary signal.
    The goal of memory retrieval is to find issues with the SAME failure pattern
    requiring the SAME fix — determined by WHY it failed, not WHERE.

    Priority order (position in string controls embedding weight):

      1. overall_failure_reasons  WHY it failed: richest semantic narrative (HIGHEST)
      2. per-file reason          per-file specific narrative of WHY
      3. overall_error_types      WHAT category (Code Styling, Type Checking...)
      4. issue_type + error_type + error_subtype   sub-category labels
      5. failed_tools + failed_command             HOW it broke (ruff, mypy, pytest...)
      6. file paths               WHERE (supporting context only — LOWEST priority)

    Why failure reason > file path:
      Same file + different reason = different problem, different fix  (bad retrieval if file=priority)
      Different file + same reason = same problem, same fix            (correct retrieval)
    """
    parts: List[str] = []

    # ── 1. Overall failure narrative (WHY — highest priority) ─────────────────
    for reason in (issue.get("overall_failure_reasons") or []):
        parts.append(reason[:500])

    # ── 2. Per-file reason (detailed per-file narrative of WHY) ───────────────
    for ef in (issue.get("effected_files") or []):
        parts.append(str(ef.get("reason") or "")[:300])

    # ── 3. Error category labels (WHAT type of failure) ───────────────────────
    parts.extend(issue.get("overall_error_types") or [])
    for ef in (issue.get("effected_files") or []):
        parts.append(str(ef.get("issue_type")    or ""))
        parts.append(str(ef.get("error_type")    or ""))
        parts.append(str(ef.get("error_subtype") or ""))

    # ── 4. Tools and commands (HOW it broke) ──────────────────────────────────
    for ef in (issue.get("effected_files") or []):
        for t in (ef.get("failed_tools") or []):
            parts.append(str(t).lower())
        parts.append((ef.get("failed_command") or "")[:150])

    # ── 5. File paths (WHERE — supporting context, lowest priority) ───────────
    for ef in (issue.get("effected_files") or []):
        f = str(ef.get("file") or "").strip()
        if f:
            parts.append(f)

    return " | ".join(x for x in parts if x.strip())


# ── Similarity ─────────────────────────────────────────────────────────────────

def compute_sim(a: Dict, b: Dict) -> float:
    """
    Weighted cosine similarity across failure-context and repair-pattern signals.
    Fix signal has the larger weight because two issues are truly
    'the same problem' only if they require the same class of fix.
    """
    return (
        W_ERROR    * cosine(a.get("_error_vec"),    b.get("_error_vec"))
      + W_FIX      * cosine(a.get("_fix_vec"),      b.get("_fix_vec"))
    )


# ── Data loading & embedding ───────────────────────────────────────────────────

def load_and_embed(error_details_path: str, parquet_path: str) -> List[Dict]:
    """
    Load error_details.json and lca_dataset.parquet, join on sha_fail,
    build semantic documents, embed.

    Issues present in parquet but missing from error_details are included
    using parquet's error_type column as a fallback signal.
    """
    # Load error_details
    with open(error_details_path) as f:
        ed_list: List[Dict] = json.load(f)
    ed_map: Dict[str, Dict] = {d["sha_fail"]: d for d in ed_list}

    # Load parquet
    df = pd.read_parquet(parquet_path)
    diff_map:  Dict[str, str] = dict(zip(df["sha_fail"], df["diff"].fillna("")))
    order_map: Dict[str, int] = {sha: idx for idx, sha in enumerate(df["sha_fail"])}

    # Build repo map from parquet (primary source for repo names)
    parquet_rows: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        sha = row["sha_fail"]
        parquet_rows[sha] = {
            "sha_fail":       sha,
            "repo":           f"{row['repo_owner']}/{row['repo_name']}",
            "repo_owner":     row["repo_owner"],
            "repo_name":      row["repo_name"],
            "workflow_name":  row.get("workflow_name", ""),
            "workflow_path":  row.get("workflow_path", ""),
            "error_type_raw": str(row.get("error_type", "")),
        }

    print(f"[RCSS] Parquet: {len(parquet_rows)} issues across "
          f"{df['repo_owner'].nunique() + df['repo_name'].nunique()//2} repos")
    print(f"[RCSS] error_details: {len(ed_map)} issues")

    enriched: List[Dict] = []
    all_shas = list(parquet_rows.keys())
    total    = len(all_shas)
    print(f"[RCSS] Embedding {total} issues…")

    for idx, sha in enumerate(all_shas):
        p    = parquet_rows[sha]
        ed   = ed_map.get(sha, {})
        diff = diff_map.get(sha, "")

        # Merge parquet and error_details (error_details wins on overlap)
        issue: Dict = {**p, **ed}
        issue["sha_fail"] = sha  # ensure sha is always from parquet

        # Fallback for issues not in error_details
        if not issue.get("overall_error_types"):
            et = p["error_type_raw"]
            issue["overall_error_types"]  = [et] if et else []
            issue["overall_failure_reasons"] = []
            issue["effected_files"] = []
            issue["failed_jobs"]    = []

        error_doc = build_error_doc(issue)
        fix_doc   = build_fix_doc(
            issue.get("overall_failure_reasons") or [], diff
        )

        issue["_error_doc"] = error_doc
        issue["_fix_doc"]   = fix_doc
        issue["_error_vec"] = embed(error_doc)
        issue["_fix_vec"]   = embed(fix_doc)
        issue["_fix_type"]  = classify_fix(diff)
        issue["_order_idx"] = order_map.get(sha, idx)
        issue["diff"]       = diff

        enriched.append(issue)

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"[RCSS]   {idx + 1}/{total} embedded")

    return enriched


# ── Per-repo recurrence scoring ────────────────────────────────────────────────

def score_recurrence(issues: List[Dict]) -> List[Dict]:
    """
    For each issue in a qualifying repo (size ≥ MIN_REPO_SIZE):
        recurrence_count = how many other issues in the same repo have sim ≥ θ
        recurrence_pct   = recurrence_count / (n-1) × 100
        avg_similarity   = mean sim of those similar peers (0.0 if none)

    Issues in small repos get recurrence_count = -1 (not analysed).
    """
    # Group by repo
    repo_idx: Dict[str, List[int]] = defaultdict(list)
    for i, iss in enumerate(issues):
        repo_idx[iss["repo"]].append(i)

    for repo, idxs in repo_idx.items():
        n = len(idxs)

        if n < MIN_REPO_SIZE:
            for i in idxs:
                issues[i]["_rec_count"] = -1
                issues[i]["_rec_pct"]   = 0.0
                issues[i]["_avg_sim"]   = 0.0
            continue

        # Build full pairwise sim matrix for this repo
        sims = np.zeros((n, n), dtype=np.float32)
        for a in range(n):
            for b in range(a + 1, n):
                s = compute_sim(issues[idxs[a]], issues[idxs[b]])
                sims[a, b] = s
                sims[b, a] = s

        for local_i, global_i in enumerate(idxs):
            row = sims[local_i]
            peers_above = float(np.sum(row >= SIMILARITY_THRESHOLD))
            sim_above   = row[row >= SIMILARITY_THRESHOLD]

            issues[global_i]["_rec_count"] = int(peers_above)
            issues[global_i]["_rec_pct"]   = (peers_above / (n - 1)) * 100.0
            issues[global_i]["_avg_sim"]   = float(np.mean(sim_above)) if len(sim_above) else 0.0

    return issues


# ── Table builders ─────────────────────────────────────────────────────────────

def build_per_issue_table(issues: List[Dict]) -> List[Dict]:
    """
    Per-issue recurrence detail — sorted by:
      1. repo total_issues DESC
      2. recurrence_count DESC
      3. avg_similarity DESC
    """
    # Count repo sizes
    repo_sizes: Dict[str, int] = defaultdict(int)
    for iss in issues:
        repo_sizes[iss["repo"]] += 1

    rows = []
    for iss in issues:
        repo = iss["repo"]
        rc   = iss.get("_rec_count", -1)
        rows.append({
            "repo":             repo,
            "total_issues":     repo_sizes[repo],
            "issue_id":         iss.get("id", iss.get("sha_fail", "")[:8]),
            "sha_fail":         iss.get("sha_fail", ""),
            "error_type":       ", ".join(iss.get("overall_error_types") or [])[:40],
            "fix_type":         iss.get("_fix_type", ""),
            "recurrence_count": rc if rc >= 0 else "n/a",
            "recurrence_pct":   f"{iss.get('_rec_pct', 0.0):.1f}%" if rc >= 0 else "n/a",
            "avg_similarity":   f"{iss.get('_avg_sim', 0.0):.4f}" if rc >= 0 else "n/a",
            "qualified":        "yes" if rc > 0 else ("no_recurrence" if rc == 0 else "small_repo"),
        })

    rows.sort(key=lambda r: (
        -r["total_issues"],
        -(r["recurrence_count"] if isinstance(r["recurrence_count"], int) else -1),
        -(float(r["avg_similarity"]) if r["avg_similarity"] not in ("n/a", "0.0000") else 0.0),
    ))
    return rows


def build_per_repo_table(issues: List[Dict]) -> List[Dict]:
    """
    Per-repo summary — overall recurrence count, percentage, avg similarity.
    Sorted by recurrence_pct DESC.
    """
    repo_data: Dict[str, Dict] = defaultdict(lambda: {
        "total": 0,
        "with_recurrence": 0,
        "sims": [],
        "qualified_for_split": False,
    })

    for iss in issues:
        repo = iss["repo"]
        rc   = iss.get("_rec_count", -1)
        repo_data[repo]["total"] += 1

        if rc < 0:
            continue  # small repo
        repo_data[repo]["qualified_for_split"] = True
        if rc > 0:
            repo_data[repo]["with_recurrence"] += 1
            repo_data[repo]["sims"].append(iss.get("_avg_sim", 0.0))

    rows = []
    for repo, d in repo_data.items():
        total = d["total"]
        wr    = d["with_recurrence"]
        rec_pct = (wr / total * 100.0) if total > 0 else 0.0
        avg_sim = float(np.mean(d["sims"])) if d["sims"] else 0.0
        rows.append({
            "repo":                    repo,
            "total_issues":            total,
            "issues_with_recurrence":  wr,
            "recurrence_pct":          f"{rec_pct:.1f}%",
            "avg_similarity":          f"{avg_sim:.4f}",
            "qualified_for_split":     "yes" if d["qualified_for_split"] else f"no (< {MIN_REPO_SIZE})",
        })

    rows.sort(key=lambda r: -float(r["recurrence_pct"].rstrip("%")))
    return rows


# ── Split ──────────────────────────────────────────────────────────────────────

def split_issues(issues: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Qualifying pool: recurrence_count ≥ 1 AND in repo with ≥ MIN_REPO_SIZE issues.
    Sort qualifying pool by recurrence_count DESC, avg_similarity DESC.
    Top MEMORY_FRACTION → memory.
    Rest + non-qualifying → evaluation.
    """
    qualifying = [iss for iss in issues if iss.get("_rec_count", -1) > 0]
    non_qualifying = [iss for iss in issues if iss.get("_rec_count", -1) <= 0]

    # Sort: most recurring + most similar first
    qualifying.sort(key=lambda x: (-x["_rec_count"], -x["_avg_sim"]))

    split_point = max(1, int(len(qualifying) * MEMORY_FRACTION))
    memory_issues = qualifying[:split_point]
    eval_issues   = qualifying[split_point:] + non_qualifying

    return memory_issues, eval_issues


def _to_output(issues: List[Dict], role: str) -> List[Dict]:
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
            "recurrence_count":    iss.get("_rec_count", -1),
            "recurrence_pct":      round(iss.get("_rec_pct", 0.0), 2),
            "avg_similarity":      round(iss.get("_avg_sim", 0.0), 4),
            "split_role":          role,
        })
    return out


# ── Pretty print tables ────────────────────────────────────────────────────────

def _print_per_issue_table(rows: List[Dict], n: int = 30) -> None:
    print("\n" + "=" * 120)
    print("PER-ISSUE RECURRENCE TABLE  (top {:d} rows shown — full table in recurrence_per_issue.csv)".format(n))
    print("=" * 120)
    header = (
        f"{'Repo':<38} {'Issues':>6}  {'ID':>6}  {'Error Type':<28}  "
        f"{'Recur.':>6}  {'Recur.%':>8}  {'Avg Sim':>8}  {'Status'}"
    )
    print(header)
    print("-" * 120)
    for r in rows[:n]:
        rc = r["recurrence_count"]
        print(
            f"{r['repo']:<38} {r['total_issues']:>6}  {str(r['issue_id']):>6}  "
            f"{r['error_type']:<28}  {str(rc):>6}  {r['recurrence_pct']:>8}  "
            f"{r['avg_similarity']:>8}  {r['qualified']}"
        )


def _print_per_repo_table(rows: List[Dict]) -> None:
    print("\n" + "=" * 110)
    print("PER-REPO RECURRENCE SUMMARY TABLE  (sorted by recurrence % desc)")
    print("=" * 110)
    header = (
        f"{'Repo':<42} {'Issues':>6}  {'W/Recur':>7}  {'Recur.%':>8}  "
        f"{'Avg Sim':>8}  {'Split?'}"
    )
    print(header)
    print("-" * 110)
    for r in rows:
        print(
            f"{r['repo']:<42} {r['total_issues']:>6}  "
            f"{r['issues_with_recurrence']:>7}  {r['recurrence_pct']:>8}  "
            f"{r['avg_similarity']:>8}  {r['qualified_for_split']}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1: load & embed
    _load_model()
    issues = load_and_embed(ERROR_DETAILS_PATH, PARQUET_PATH)

    # Phase 2+3: pairwise similarity + recurrence scoring
    print("\n[RCSS] Computing pairwise cosine similarities within each repo…")
    issues = score_recurrence(issues)

    # Phase 4: build and display tables
    per_issue_rows = build_per_issue_table(issues)
    per_repo_rows  = build_per_repo_table(issues)

    _print_per_issue_table(per_issue_rows, n=40)
    _print_per_repo_table(per_repo_rows)

    # Phase 5: split
    memory_issues, eval_issues = split_issues(issues)

    # ── Write CSV tables ───────────────────────────────────────────────────────
    per_issue_csv = f"{OUTPUT_DIR}/recurrence_per_issue.csv"
    with open(per_issue_csv, "w", newline="") as f:
        if per_issue_rows:
            writer = csv.DictWriter(f, fieldnames=per_issue_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_issue_rows)

    per_repo_csv = f"{OUTPUT_DIR}/recurrence_per_repo.csv"
    with open(per_repo_csv, "w", newline="") as f:
        if per_repo_rows:
            writer = csv.DictWriter(f, fieldnames=per_repo_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_repo_rows)

    # ── Write split JSON ───────────────────────────────────────────────────────
    memory_output = _to_output(memory_issues, "memory")
    eval_output   = _to_output(eval_issues,   "evaluation")

    with open(f"{OUTPUT_DIR}/memory_30pct.json", "w") as f:
        json.dump(memory_output, f, indent=2)

    with open(f"{OUTPUT_DIR}/eval_70pct.json", "w") as f:
        json.dump(eval_output, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    total      = len(issues)
    qualifying = [x for x in issues if x.get("_rec_count", -1) > 0]
    no_recur   = [x for x in issues if x.get("_rec_count", 0) == 0]
    small_repo = [x for x in issues if x.get("_rec_count", -1) < 0]

    from collections import Counter
    mem_fix_dist  = Counter(x["_fix_type"] for x in memory_issues)
    eval_fix_dist = Counter(x["_fix_type"] for x in eval_issues)

    lines = [
        "",
        "=" * 70,
        "RCSS Split Summary",
        "=" * 70,
        f"Total issues                    : {total}",
        f"Repos analysed (size ≥ {MIN_REPO_SIZE})       : "
            f"{sum(1 for r in per_repo_rows if r['qualified_for_split']=='yes')}",
        f"Repos skipped  (size < {MIN_REPO_SIZE})       : "
            f"{sum(1 for r in per_repo_rows if r['qualified_for_split']!='yes')}",
        "",
        f"Issues with recurrence (≥1 similar peer)  : {len(qualifying)}",
        f"Issues with no recurrence (0 similar peer): {len(no_recur)}",
        f"Issues in small repos (excluded)          : {len(small_repo)}",
        "",
        f"Similarity threshold θ          : {SIMILARITY_THRESHOLD}",
        f"Weights  error={W_ERROR}  fix={W_FIX}",
        "",
        f"── Split ─────────────────────────────────────────────────",
        f"Memory set (top {int(MEMORY_FRACTION*100)}% by recurrence+sim): {len(memory_output)}",
        f"Evaluation set                  : {len(eval_output)}",
        f"Memory/Eval ratio               : {len(memory_output)/max(len(eval_output),1):.2f}",
        "",
        "Fix type distribution in memory set:",
    ]
    for ft, cnt in mem_fix_dist.most_common():
        lines.append(f"  {ft:<30} {cnt}")
    lines += ["", "Fix type distribution in eval set:"]
    for ft, cnt in eval_fix_dist.most_common():
        lines.append(f"  {ft:<30} {cnt}")

    summary = "\n".join(lines)
    print(summary)
    with open(f"{OUTPUT_DIR}/split_summary.txt", "w") as f:
        f.write(summary)

    print(f"\n[RCSS] Outputs written to: {OUTPUT_DIR}/")
    print(f"       recurrence_per_issue.csv  — {len(per_issue_rows)} rows")
    print(f"       recurrence_per_repo.csv   — {len(per_repo_rows)} rows")
    print(f"       memory_30pct.json         — {len(memory_output)} issues")
    print(f"       eval_70pct.json           — {len(eval_output)} issues")
    print(f"       split_summary.txt")


if __name__ == "__main__":
    run()
