#!/usr/bin/env python3
"""
analyze_repo_similarity.py
══════════════════════════

STEP 0 — run this FIRST before any split.

Computes within-repo pairwise cosine similarity for all issues and prints:
  1. Per-issue similarity table  — which issues are similar to how many others
  2. Per-repo summary table      — total issues, similar pairs, groups, eligible %
  3. Dataset-level stats         — threshold, distribution, qualifying repos

Nothing is split or written to memory — this is analysis only.
Results are saved to:
  results/similarity_analysis/
      per_issue_similarity.csv    — one row per issue
      per_repo_summary.csv        — one row per repo
      similarity_pairs.csv        — every qualifying pair with its score
      analysis_summary.txt        — human-readable report

After reviewing the tables, run:
  venv/bin/python3 scripts/temporal_recurrence_split.py   ← do the primary TRS split

Similarity formula (2-signal weighted cosine):
    sim(i,j) = 0.3125 × cos(error_doc) + 0.6875 × cos(fix_doc)

error_doc priority: WHY > WHAT > HOW > WHERE
    overall_failure_reasons → per-file reason → error_types → tools/cmds → file paths

Embedding: sentence-transformers/all-MiniLM-L6-v2  (O-CRD §3.1)
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
OUTPUT_DIR = str(PROJECT_ROOT / "results" / "similarity_analysis")

# ── Parameters ─────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.60   # p80 of within-repo cosine distribution with specific signals
                              # genuine matches score 0.75+, so threshold at 0.60 safely
                              # captures all real recurrences while excluding unrelated pairs
MIN_REPO_SIZE        = 3      # repos with fewer issues are skipped

W_ERROR = 0.40   # error signal: specific codes + tools + subtypes
W_FIX   = 0.60   # fix signal: category + error codes + actual changed lines

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
        print("[SIM] Model: sentence-transformers/all-MiniLM-L6-v2  (O-CRD §3.1)")
        return
    except Exception as e:
        print(f"[SIM] sentence-transformers unavailable ({e}); trying fastembed…")
    try:
        from fastembed import TextEmbedding
        _model   = TextEmbedding("BAAI/bge-base-en-v1.5")
        _backend = "fastembed"
        return
    except Exception as e:
        raise RuntimeError(f"No embedding model available: {e}")


def _embed(text: str) -> Optional[np.ndarray]:
    if not text or not text.strip():
        return None
    if text in _cache:
        return _cache[text]
    try:
        if _backend == "sentence_transformers":
            vec = _model.encode(text, normalize_embeddings=True)
        else:
            vec = np.array(next(_model.embed([text])), dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        vec = np.array(vec, dtype=np.float32)
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


def _fix_doc(failure_reasons: List[str], diff: str, error_types: List[str] = None) -> str:
    """
    Specific fix-signal document.  Uses error codes + actual changed lines + file names.
    Avoids the broken complaint extraction that picked up job-name prefixes
    ('style-check / style-check (3') from period-splits on version strings like '(3.9):'.
    """
    ft = _classify_fix(diff)

    # Error codes from failure reasons (F821, E501, B006 …)
    all_text = " ".join(failure_reasons or [])
    codes = list(dict.fromkeys(re.findall(r"\b[A-Z]\d{3,4}\b", all_text)))[:4]

    # File basenames changed by this fix
    changed_files = re.findall(r"diff --git a/(\S+)", diff or "")
    filenames = [Path(f).name for f in changed_files[:4]]

    # First 2 meaningful added lines (the actual fix content)
    added_lines = [
        l[1:].strip()
        for l in (diff or "").splitlines()
        if l.startswith("+") and not l.startswith("+++") and len(l.strip()) > 5
    ]
    meaningful = [
        l for l in added_lines
        if len(l) > 15
        and not l.strip().startswith("#")
        and not l.strip().startswith("//")
    ][:2]

    parts: List[str] = [ft]
    if error_types:
        parts.append(" ".join(error_types[:2]))
    if codes:
        parts.append("codes: " + " ".join(codes))
    if filenames:
        parts.append("files: " + " ".join(filenames))
    if meaningful:
        fix_content = " | ".join(l[:120] for l in meaningful)
        parts.append("fix: " + fix_content)
    return " | ".join(parts)


# ── Document builders ──────────────────────────────────────────────────────────

def _error_doc(issue: Dict) -> str:
    """
    Specific error-signal document using codes + tools + subtypes (not long narrative text).
    Long failure reasons create false similarity via shared words like 'failed', 'exit code 1'.
    """
    parts: List[str] = []

    # 1. Overall error types (high-level category)
    parts.extend(issue.get("overall_error_types") or [])

    # 2. Per-file specific subtypes
    for ef in (issue.get("effected_files") or []):
        for key in ("error_subtype", "issue_type", "error_type"):
            val = str(ef.get(key) or "").strip()
            if val:
                parts.append(val)

    # 3. Tools and exact commands
    for ef in (issue.get("effected_files") or []):
        for t in (ef.get("failed_tools") or []):
            parts.append(str(t).lower())
        cmd = str(ef.get("failed_command") or "").strip()
        if cmd:
            parts.append(cmd[:80])

    # 4. Error codes (most discriminating signal): F821, E501, ImportError, …
    all_text = " ".join(issue.get("overall_failure_reasons") or [])
    codes = list(dict.fromkeys(re.findall(
        r"\b([A-Z]\d{3,4}|ImportError|ModuleNotFoundError|TypeError|AttributeError"
        r"|NameError|SyntaxError|AssertionError|KeyError|ValueError|RuntimeError"
        r"|FileNotFoundError|PermissionError|ConnectionError)\b",
        all_text,
    )))[:6]
    parts.extend(codes)

    # 5. File basenames (not full paths — basenames recur across commits)
    for ef in (issue.get("effected_files") or []):
        f = str(ef.get("file") or "").strip()
        if f:
            parts.append(Path(f).name)

    return " | ".join(x for x in parts if x.strip())


def _sim(a: Dict, b: Dict) -> float:
    return (
        W_ERROR    * _cos(a.get("_ve"), b.get("_ve"))
      + W_FIX      * _cos(a.get("_vf"), b.get("_vf"))
    )


# ── Union-Find for grouping ────────────────────────────────────────────────────

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


# ── Core analysis ──────────────────────────────────────────────────────────────

def _analyse_repo(
    repo: str,
    issues: List[Dict],
    pq_by_sha: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict], List[Tuple]]:
    """
    Returns:
        issue_rows  — per-issue stats (similar_count, avg_sim, max_sim, group_id)
        pair_rows   — qualifying pairs (id_a, id_b, score)
        all_scores  — raw similarity floats (for distribution stats)
    """
    n = len(issues)

    # ── embed ──
    for iss in issues:
        sha  = str(iss.get("sha_fail") or "")
        pq   = pq_by_sha.get(sha, {})
        diff = str(pq.get("diff") or "")
        f_reasons = iss.get("overall_failure_reasons") or []
        iss["_ve"] = _embed(_error_doc(iss))
        iss["_vf"] = _embed(_fix_doc(
            f_reasons,
            diff,
            iss.get("overall_error_types") or [],
        ))

    # ── pairwise similarity ──
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    pair_rows: List[Tuple] = []
    all_scores: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            s = _sim(issues[i], issues[j])
            sim_matrix[i, j] = sim_matrix[j, i] = s
            all_scores.append(s)
            if s >= SIMILARITY_THRESHOLD:
                pair_rows.append((
                    issues[i].get("id"), issues[j].get("id"),
                    issues[i].get("sha_fail"), issues[j].get("sha_fail"),
                    round(s, 4),
                    issues[i].get("overall_error_types", []),
                    issues[j].get("overall_error_types", []),
                ))

    # ── Union-Find grouping ──
    uf = _UF(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= SIMILARITY_THRESHOLD:
                uf.union(i, j)

    group_map: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        group_map[uf.find(i)].append(i)

    # assign group labels (G1, G2, ...) sorted by group size desc
    sorted_groups = sorted(group_map.values(), key=len, reverse=True)
    idx_to_group: Dict[int, str] = {}
    idx_to_group_size: Dict[int, int] = {}
    for g_idx, members in enumerate(sorted_groups, 1):
        label = f"G{g_idx}"
        for m in members:
            idx_to_group[m] = label
            idx_to_group_size[m] = len(members)

    # ── per-issue stats ──
    issue_rows: List[Dict] = []
    for i, iss in enumerate(issues):
        sims_above = [sim_matrix[i, j] for j in range(n) if j != i and sim_matrix[i, j] >= SIMILARITY_THRESHOLD]
        row = {
            "repo":            repo,
            "id":              iss.get("id"),
            "sha_fail":        (iss.get("sha_fail") or "")[:16] + "...",
            "error_types":     ", ".join((iss.get("overall_error_types") or [])[:2]),
            "group":           idx_to_group[i],
            "group_size":      idx_to_group_size[i],
            "similar_count":   len(sims_above),
            "has_similar":     len(sims_above) > 0,
            "avg_sim":         round(float(np.mean(sims_above)), 4) if sims_above else 0.0,
            "max_sim":         round(float(np.max(sims_above)), 4) if sims_above else 0.0,
        }
        issue_rows.append(row)

    return issue_rows, pair_rows, all_scores


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_per_repo_table(repo_summary: List[Dict]) -> None:
    H = (
        f"{'Repo':<42} {'Issues':>6} {'SimilarIssues':>13} {'%Eligible':>9} "
        f"{'Groups':>6} {'Pairs':>6} {'AvgSim':>7} {'MaxSim':>7}"
    )
    sep = "─" * len(H)
    print("\n" + "═" * len(H))
    print("PER-REPO SIMILARITY SUMMARY  (repos ≥ 3 issues, sorted by %eligible ↓)")
    print("═" * len(H))
    print(H)
    print(sep)
    for r in repo_summary:
        print(
            f"{r['repo']:<42} {r['total_issues']:>6} {r['similar_issues']:>13} "
            f"{r['eligible_pct']:>8.1f}% {r['groups']:>6} {r['pairs']:>6} "
            f"{r['avg_sim']:>7.4f} {r['max_sim']:>7.4f}"
        )
    print(sep)


def _print_per_issue_table(rows: List[Dict], limit: int = 60) -> None:
    H = (
        f"{'Repo':<38} {'ID':>5}  {'ErrorType':<28} {'Group':>5}  "
        f"{'GrpSz':>5} {'SimilarN':>8} {'AvgSim':>7} {'MaxSim':>7}"
    )
    sep = "─" * len(H)
    print("\n" + "═" * len(H))
    print(f"PER-ISSUE SIMILARITY TABLE  (first {limit} rows — full table in per_issue_similarity.csv)")
    print("═" * len(H))
    print(H)
    print(sep)
    for r in rows[:limit]:
        print(
            f"{r['repo']:<38} {str(r['id']):>5}  {r['error_types'][:28]:<28} "
            f"{r['group']:>5}  {r['group_size']:>5} {r['similar_count']:>8} "
            f"{r['avg_sim']:>7.4f} {r['max_sim']:>7.4f}"
        )
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    print("[SIM] Loading error_details.json …")
    issues_raw: List[Dict] = json.loads(Path(ERROR_DETAILS_PATH).read_text(encoding="utf-8"))
    print(f"[SIM]   {len(issues_raw)} issues loaded")

    print("[SIM] Loading parquet …")
    df = pd.read_parquet(PARQUET_PATH)
    df["sha_fail"] = df["sha_fail"].astype(str)
    pq_by_sha: Dict[str, Any] = {r["sha_fail"]: r for r in df.to_dict(orient="records")}

    # ── load model ──
    _load_model()

    # ── group by repo ──
    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for iss in issues_raw:
        repo = str(iss.get("repo") or "")
        by_repo[repo].append(iss)

    all_issue_rows:  List[Dict]  = []
    all_pair_rows:   List[Tuple] = []
    all_scores:      List[float] = []
    repo_summary:    List[Dict]  = []

    repos_qualified = 0
    repos_skipped   = 0

    print(f"\n[SIM] Analysing {len(by_repo)} repos …")

    for repo, issues in sorted(by_repo.items(), key=lambda x: -len(x[1])):
        if len(issues) < MIN_REPO_SIZE:
            repos_skipped += 1
            continue

        repos_qualified += 1
        issue_rows, pair_rows, scores = _analyse_repo(repo, issues, pq_by_sha)

        all_issue_rows.extend(issue_rows)
        all_pair_rows.extend(pair_rows)
        all_scores.extend(scores)

        similar_issues  = sum(1 for r in issue_rows if r["has_similar"])
        unique_groups   = len({r["group"] for r in issue_rows if r["group_size"] >= 2})
        sims_above      = [s for _, _, _, _, s, _, _ in pair_rows]

        repo_summary.append({
            "repo":            repo,
            "total_issues":    len(issues),
            "similar_issues":  similar_issues,
            "eligible_pct":    round(100 * similar_issues / len(issues), 1),
            "groups":          unique_groups,
            "pairs":           len(pair_rows),
            "avg_sim":         round(float(np.mean(sims_above)), 4) if sims_above else 0.0,
            "max_sim":         round(float(np.max(sims_above)), 4) if sims_above else 0.0,
        })

    # sort repo_summary: eligible_pct desc, then total_issues desc
    repo_summary.sort(key=lambda r: (-r["eligible_pct"], -r["total_issues"]))

    # ── distribution stats ──
    arr = np.array(all_scores, dtype=np.float32)
    p10, p25, p50, p75, p90 = (
        float(np.percentile(arr, p)) for p in (10, 25, 50, 75, 90)
    ) if len(arr) else (0,) * 5

    # ── print tables ──
    all_issue_rows_sorted = sorted(
        all_issue_rows,
        key=lambda r: (-r["similar_count"], -r["avg_sim"])
    )
    _print_per_issue_table(all_issue_rows_sorted)
    _print_per_repo_table(repo_summary)

    # ── summary stats ──
    total_issues     = len(all_issue_rows)
    similar_issues   = sum(1 for r in all_issue_rows if r["has_similar"])
    qualifying_pairs = len(all_pair_rows)

    print(f"""
╔══════════════════════════════════════════════════════╗
║           SIMILARITY ANALYSIS — SUMMARY              ║
╠══════════════════════════════════════════════════════╣
║  Threshold θ            : {SIMILARITY_THRESHOLD:.2f}  (recurrence link cutoff) ║
║  Weights  error={W_ERROR}  fix={W_FIX}                                  ║
╠══════════════════════════════════════════════════════╣
║  Total issues           : {total_issues:>5}                          ║
║  Issues with similar ≥1 : {similar_issues:>5}  ({100*similar_issues/total_issues:.1f}% eligible for split) ║
║  Issues without similar : {total_issues-similar_issues:>5}  (would be excluded)        ║
║  Qualifying pairs (≥θ)  : {qualifying_pairs:>5}                          ║
╠══════════════════════════════════════════════════════╣
║  Repos analysed (≥3 iss): {repos_qualified:>5}                          ║
║  Repos skipped (<3 iss) : {repos_skipped:>5}                          ║
╠══════════════════════════════════════════════════════╣
║  Within-repo sim distribution (all pairs):           ║
║    p10={p10:.3f}  p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}  ║
╚══════════════════════════════════════════════════════╝""")

    print(f"\n[SIM] → If θ=0.50 looks wrong, change SIMILARITY_THRESHOLD in temporal_recurrence_split.py")
    print(f"[SIM] → Then run: venv/bin/python3 scripts/temporal_recurrence_split.py\n")

    # ── save CSV outputs ──
    # per_issue_similarity.csv
    issue_csv = out / "per_issue_similarity.csv"
    with issue_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_issue_rows_sorted[0].keys()))
        w.writeheader()
        w.writerows(all_issue_rows_sorted)

    # per_repo_summary.csv
    repo_csv = out / "per_repo_summary.csv"
    with repo_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(repo_summary[0].keys()))
        w.writeheader()
        w.writerows(repo_summary)

    # similarity_pairs.csv
    pairs_csv = out / "similarity_pairs.csv"
    with pairs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id_a", "id_b", "sha_a", "sha_b", "sim_score", "error_types_a", "error_types_b"])
        for row in sorted(all_pair_rows, key=lambda x: -x[4]):
            writer.writerow([
                row[0], row[1], row[2], row[3], row[4],
                "|".join(row[5]) if isinstance(row[5], list) else row[5],
                "|".join(row[6]) if isinstance(row[6], list) else row[6],
            ])

    # analysis_summary.txt
    summary_txt = out / "analysis_summary.txt"
    summary_txt.write_text(
        f"Similarity Analysis Summary\n"
        f"===========================\n"
        f"Threshold θ          : {SIMILARITY_THRESHOLD}\n"
        f"Weights (err/fix)    : {W_ERROR}/{W_FIX}\n\n"
        f"Total issues         : {total_issues}\n"
        f"Similar issues (≥θ)  : {similar_issues}  ({100*similar_issues/total_issues:.1f}%)\n"
        f"No-peer issues       : {total_issues - similar_issues}\n"
        f"Qualifying pairs     : {qualifying_pairs}\n\n"
        f"Repos analysed (≥3)  : {repos_qualified}\n"
        f"Repos skipped (<3)   : {repos_skipped}\n\n"
        f"Similarity distribution:\n"
        f"  p10={p10:.4f}  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f}  p90={p90:.4f}\n\n"
        f"Next step:\n"
        f"  venv/bin/python3 scripts/temporal_recurrence_split.py\n",
        encoding="utf-8",
    )

    print(f"[SIM] Outputs → {out}/")
    print(f"       per_issue_similarity.csv   ({len(all_issue_rows_sorted)} rows)")
    print(f"       per_repo_summary.csv       ({len(repo_summary)} rows)")
    print(f"       similarity_pairs.csv       ({len(all_pair_rows)} qualifying pairs)")
    print(f"       analysis_summary.txt\n")


if __name__ == "__main__":
    main()
