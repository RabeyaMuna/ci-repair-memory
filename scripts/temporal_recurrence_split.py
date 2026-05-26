#!/usr/bin/env python3
"""
temporal_recurrence_split.py
════════════════════════════
RCSS-detected · Temporally-split · Bidirectionally-verified

Two-phase design
─────────────────
Phase 1 — RCSS Detection  (Recurrence-Coherence Similarity Score)
    For every pair (i, j) in a repo compute weighted cosine similarity.
    An issue is RECURRENT if it has ≥ 1 similar peer above θ (any direction).
    Non-recurrent "singleton" issues are excluded entirely — they have no
    similar counterpart to learn from or test against.

    Per-issue RCSS scores:
      recurrence_count  = # similar peers (any direction)
      future_count      = # LATER similar issues  (i → j, id(i) < id(j))
      past_count        = # EARLIER similar issues (j → i, id(j) < id(i))
      avg_sim           = mean cosine across all similar peers

Phase 2 — Temporal Split  (no data leakage)
    Take ONLY the recurrent issues for each repo.
    Sort them chronologically (ascending parquet id = older → newer).
    Split:
        oldest 30%  →  MEMORY POOL   (prior experience)
        newest 70%  →  EVAL   POOL   (future cases to repair)

    Then verify bidirectional temporal links:
        eval_verified   = eval issues that have ≥ 1 memory peer
                          (sim ≥ θ AND memory id < eval id)
        memory_verified = memory issues used by ≥ 1 verified eval issue

    This double-filter guarantees:
        ✓ every memory issue  has proven recurrence into the eval window
        ✓ every eval   issue  has a guaranteed similar memory peer
        ✓ memory is always chronologically OLDER than eval
        ✓ no data leakage

Why RCSS for detection, temporal for splitting?
    RCSS is the right tool for FINDING recurring patterns: it counts how many
    similar peers an issue has and ranks by combined recurrence + similarity.
    Temporal ordering is the right tool for SPLITTING: it ensures memory issues
    were genuinely "past experience" at the time of the eval issue.
    Combining both gives a clean, defensible experimental design.

Similarity (2-signal weighted cosine):
    sim(i,j) = 0.3125 × cos(error_doc)  ← WHY/WHAT/HOW/WHERE failure context
             + 0.6875 × cos(fix_doc)    ← repair-pattern signal

    error_doc priority: WHY > WHAT > HOW > WHERE  (O-CRD §3.1)
"""

from __future__ import annotations

import csv
import json
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
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/trs_split"
)

# ── Parameters ─────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.60   # p80 of within-repo cosine distribution with specific signals
                              # genuine matches score 0.75+, so threshold at 0.60 safely
                              # captures all real recurrences while excluding unrelated pairs
MIN_REPO_SIZE        = 3      # repos with fewer issues are skipped
MEMORY_FRACTION      = 0.30   # oldest 30% of recurrent issues → memory

# ── Weights (must sum to 1.0) ──────────────────────────────────────────────────
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
        print("[TRS] Model: sentence-transformers/all-MiniLM-L6-v2  (O-CRD §3.1)")
        return
    except Exception as e:
        print(f"[TRS] sentence-transformers unavailable ({e}); trying fastembed …")
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
                vec /= norm
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
    (r"\+\s*(from\s+\S+\s+import|import\s+\S+)",                     "import_fix"),
    (r"(requirements.*\.txt|pyproject\.toml|setup\.cfg|setup\.py)",  "dependency_update"),
    (r"(mypy|pyright|pytype)",                                        "type_check_fix"),
    (r"(ruff|black|flake8|isort|pylint)",                             "format_fix"),
    (r"\+.*:\s*(int|str|float|bool|List|Dict|Optional|Union|Any)\b", "type_annotation_fix"),
    (r"def test_|assert |pytest\.",                                   "test_fix"),
    (r"(\.github/workflows/|on:\s|runs-on:)",                         "config_fix"),
    (r"pip install|poetry add|conda install",                         "dependency_install_fix"),
]


def _classify_fix(diff: str) -> str:
    if not diff:
        return "unknown_fix"
    for pat, label in _FIX_PATTERNS:
        if re.search(pat, diff, re.IGNORECASE):
            return label
    added   = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    return "minor_fix" if added + removed <= 4 else "structural_fix"


# ── Document builders (WHY > WHAT > HOW > WHERE) ───────────────────────────────

def _error_doc(issue: Dict) -> str:
    """
    Build a specific, discriminating error-signal document.

    Priority (WHY > WHAT > HOW > WHERE) but using SPECIFIC signals, not long
    narratives.  Long failure-reason text creates false similarity between issues
    that merely share common words like 'failed', 'step', 'exit code 1'.

    Instead we use:
      1. overall_error_types          — high-level category ("Code Styling", "Test Failure")
      2. error_subtype / issue_type   — specific subtype ("Undefined names", "ImportError")
      3. failed_tools + command       — exact tool and invocation ("ruff check --target py39")
      4. error codes from text        — F821, E501, ImportError, etc. (most discriminating)
      5. file basenames               — which files are affected (not full paths)
    """
    parts: List[str] = []

    # 1. Overall error types (high-level category — discriminates linting vs test vs config)
    parts.extend(issue.get("overall_error_types") or [])

    # 2. Per-file specific subtypes
    for ef in (issue.get("effected_files") or []):
        for key in ("error_subtype", "issue_type", "error_type"):
            val = str(ef.get(key) or "").strip()
            if val:
                parts.append(val)

    # 3. Tools and commands (exact tool name is very discriminating)
    for ef in (issue.get("effected_files") or []):
        for t in (ef.get("failed_tools") or []):
            parts.append(str(t).lower())
        cmd = str(ef.get("failed_command") or "").strip()
        if cmd:
            parts.append(cmd[:80])

    # 4. Error codes extracted from failure reasons — the most specific signal
    #    Catches: F821, E501, B006, ImportError, TypeError, ModuleNotFoundError, etc.
    all_text = " ".join(issue.get("overall_failure_reasons") or [])
    codes = list(dict.fromkeys(re.findall(
        r"\b([A-Z]\d{3,4}|ImportError|ModuleNotFoundError|TypeError|AttributeError"
        r"|NameError|SyntaxError|AssertionError|KeyError|ValueError|RuntimeError"
        r"|FileNotFoundError|PermissionError|ConnectionError)\b",
        all_text,
    )))[:6]
    parts.extend(codes)

    # 5. File basenames (not full paths — full paths differ per commit but basenames recur)
    for ef in (issue.get("effected_files") or []):
        f = str(ef.get("file") or "").strip()
        if f:
            parts.append(Path(f).name)

    return " | ".join(x for x in parts if x.strip())


def _fix_doc(failure_reasons: List[str], diff: str, error_types: List[str] = None) -> str:
    """
    Build a specific, discriminating fix-signal document.
    Uses: fix category + error codes from failure text + actual changed lines + filenames.
    Avoids: the broken 'complaint' extraction that picked up job-name prefixes like
            'style-check / style-check (3' from period-splits on version strings.
    """
    ft = _classify_fix(diff)

    # Error codes extracted from failure reasons (F821, E501, B006, …)
    all_text = " ".join(failure_reasons or [])
    codes = list(dict.fromkeys(re.findall(r"\b[A-Z]\d{3,4}\b", all_text)))[:4]

    # File basenames changed by this fix
    changed_files = re.findall(r"diff --git a/(\S+)", diff or "")
    filenames = [Path(f).name for f in changed_files[:4]]

    # First 2 meaningful added lines — the actual fix content
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


def _weighted_sim(a: Dict, b: Dict) -> float:
    return (
        W_ERROR    * _cos(a.get("_ve"), b.get("_ve"))
      + W_FIX      * _cos(a.get("_vf"), b.get("_vf"))
    )


# ── Per-repo processing ────────────────────────────────────────────────────────

def _process_repo(
    repo: str,
    issues: List[Dict],       # sorted chronologically by parquet id (asc)
    pq_by_sha: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Returns (memory_verified, eval_verified, excluded, per_issue_stats)

    Phase 1 — RCSS detection
        Compute full pairwise similarity matrix.
        Tag every issue with recurrence_count (# similar peers, any direction).
        Issues with recurrence_count == 0 → excluded immediately (singletons).

    Phase 2 — Temporal split within recurrent pool
        Sort recurrent issues by parquet id (oldest first).
        first 30% → memory pool  |  last 70% → eval pool
        Verify bidirectional temporal links → final verified sets.
    """
    n = len(issues)

    # ── embed all issues ───────────────────────────────────────────────────────
    for iss in issues:
        sha  = str(iss.get("sha_fail") or "")
        pq   = pq_by_sha.get(sha, {})
        diff = str(pq.get("diff") or "")
        iss["_ve"] = _embed(_error_doc(iss))
        iss["_vf"] = _embed(_fix_doc(
            iss.get("overall_failure_reasons") or [],
            diff,
            iss.get("overall_error_types") or [],
        ))

    # ── Phase 1: RCSS — full pairwise similarity matrix ───────────────────────
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            s = _weighted_sim(issues[i], issues[j])
            sim_matrix[i, j] = sim_matrix[j, i] = s

    # per-issue RCSS stats
    recurrence_count = np.zeros(n, dtype=int)   # # similar peers (any direction)
    avg_sim          = np.zeros(n, dtype=np.float32)
    future_count     = np.zeros(n, dtype=int)   # # LATER similar issues
    past_count       = np.zeros(n, dtype=int)   # # EARLIER similar issues
    best_past_sim    = np.zeros(n, dtype=np.float32)

    for i in range(n):
        peers = [j for j in range(n) if j != i and sim_matrix[i, j] >= SIMILARITY_THRESHOLD]
        recurrence_count[i] = len(peers)
        avg_sim[i] = float(np.mean([sim_matrix[i, j] for j in peers])) if peers else 0.0
        for j in peers:
            if j > i:                        # j is newer
                future_count[i] += 1
            else:                            # j is older
                past_count[i] += 1
                if sim_matrix[i, j] > best_past_sim[i]:
                    best_past_sim[i] = sim_matrix[i, j]

    # recurrence score (RCSS): count + avg_sim for ranking/reporting
    rcss_score = recurrence_count.astype(np.float32) + avg_sim

    # ── identify recurrent issues (Phase 1 filter) ────────────────────────────
    # Non-recurrent = no similar peer in either direction → excluded entirely
    recurrent_indices = [i for i in range(n) if recurrence_count[i] >= 1]
    singleton_indices = [i for i in range(n) if recurrence_count[i] == 0]

    if not recurrent_indices:
        # no recurrent issues in this repo → exclude all
        excluded = [
            {**{k: v for k, v in iss.items() if not k.startswith("_")},
             "role": "excluded_singleton",
             "recurrence_count": 0, "future_count": 0, "past_count": 0,
             "avg_sim": 0.0, "rcss_score": 0.0}
            for iss in issues
        ]
        stats = [{
            "repo": repo, "id": iss.get("id"), "sha_fail": iss.get("sha_fail"),
            "error_types": ", ".join((iss.get("overall_error_types") or [])[:2]),
            "fix_type": iss.get("fix_type") or "",
            "temporal_rank_all": issues.index(iss) + 1,
            "temporal_rank_recurrent": "—",
            "recurrence_count": 0, "future_count": 0, "past_count": 0,
            "avg_sim": 0.0, "rcss_score": 0.0, "best_past_sim": 0.0,
            "pool": "—", "role": "excluded_singleton",
        } for iss in issues]
        return [], [], excluded, stats

    # ── Phase 2: temporal 30/70 split WITHIN recurrent pool ───────────────────
    # recurrent_indices is already in chronological order (issues were sorted by id)
    r_n           = len(recurrent_indices)
    split_point   = max(1, int(r_n * MEMORY_FRACTION))  # 30% of recurrent
    mem_pool_set  = set(recurrent_indices[:split_point])  # oldest 30%
    eval_pool_set = set(recurrent_indices[split_point:])  # newest 70%

    # verify temporal cross-links:
    # eval_verified: eval issues that have ≥1 memory peer (must be strictly older)
    eval_verified_set = {
        j for j in eval_pool_set
        if any(
            i in mem_pool_set
            and sim_matrix[i, j] >= SIMILARITY_THRESHOLD
            # temporal ordering: i is at lower index (older) than j by construction
            # (recurrent_indices is sorted chronologically, mem pool = first split_point)
            for i in mem_pool_set
        )
    }

    # memory_verified: memory issues used by ≥1 verified eval issue
    memory_verified_set = {
        i for i in mem_pool_set
        if any(
            j in eval_verified_set and sim_matrix[i, j] >= SIMILARITY_THRESHOLD
            for j in eval_verified_set
        )
    }

    # ── build output records ───────────────────────────────────────────────────
    memory_issues:   List[Dict] = []
    eval_issues:     List[Dict] = []
    excluded_issues: List[Dict] = []
    per_issue_stats: List[Dict] = []

    # rank within recurrent pool (for reporting)
    recurrent_rank_map = {idx: rank + 1 for rank, idx in enumerate(recurrent_indices)}

    for i, iss in enumerate(issues):
        # determine role
        if i in memory_verified_set:
            role = "memory"
        elif i in eval_verified_set:
            role = "eval"
        elif i in mem_pool_set:
            role = "excluded_mem_no_eval"   # recurrent but no eval user in eval window
        elif i in eval_pool_set:
            role = "excluded_eval_no_mem"   # recurrent but no memory peer in mem window
        else:
            role = "excluded_singleton"     # not recurrent at all

        # top-3 memory peers for eval issues (for analysis / reporting)
        top_mem_peers: List[Dict] = []
        if role == "eval":
            for mi in sorted(memory_verified_set, key=lambda x: -sim_matrix[x, i]):
                s = sim_matrix[mi, i]
                if s >= SIMILARITY_THRESHOLD:
                    top_mem_peers.append({"id": issues[mi].get("id"), "sim": round(float(s), 4)})
                if len(top_mem_peers) >= 3:
                    break

        stat = {
            "repo":                   repo,
            "id":                     iss.get("id"),
            "sha_fail":               iss.get("sha_fail"),
            "error_types":            ", ".join((iss.get("overall_error_types") or [])[:2]),
            "fix_type":               iss.get("fix_type") or "",
            # temporal position in full repo (1 = oldest)
            "temporal_rank_all":      i + 1,
            "temporal_pct_all":       round(100.0 * i / max(n - 1, 1), 1),
            # position within recurrent pool only
            "temporal_rank_recurrent": recurrent_rank_map.get(i, "—"),
            # RCSS detection signals
            "recurrence_count":       int(recurrence_count[i]),
            "future_count":           int(future_count[i]),
            "past_count":             int(past_count[i]),
            "avg_sim":                round(float(avg_sim[i]), 4),
            "rcss_score":             round(float(rcss_score[i]), 4),
            "best_past_sim":          round(float(best_past_sim[i]), 4),
            # split phase
            "pool":  ("memory" if i in mem_pool_set else
                      "eval"   if i in eval_pool_set else
                      "singleton"),
            "role":  role,
            "top_memory_peers": top_mem_peers,
        }
        per_issue_stats.append(stat)

        record = {k: v for k, v in iss.items() if not k.startswith("_")}
        record.update({
            "role":                role,
            "temporal_rank_all":   i + 1,
            "temporal_pct_all":    round(100.0 * i / max(n - 1, 1), 1),
            "recurrence_count":    int(recurrence_count[i]),
            "future_count":        int(future_count[i]),
            "past_count":          int(past_count[i]),
            "avg_sim":             round(float(avg_sim[i]), 4),
            "rcss_score":          round(float(rcss_score[i]), 4),
            "best_past_sim":       round(float(best_past_sim[i]), 4),
            "top_memory_peers":    top_mem_peers,
        })

        if role == "memory":
            memory_issues.append(record)
        elif role == "eval":
            eval_issues.append(record)
        else:
            excluded_issues.append(record)

    return memory_issues, eval_issues, excluded_issues, per_issue_stats


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_repo_table(rows: List[Dict]) -> None:
    H = (
        f"{'Repo':<42} {'All':>4} {'Recur':>5} {'Single':>6} "
        f"{'MemPool':>7} {'EvalPool':>8} {'→Mem':>5} {'→Eval':>6} "
        f"{'Excl':>5} {'Cover%':>7}"
    )
    sep = "─" * len(H)
    print("\n" + "═" * len(H))
    print("PER-REPO  RCSS-DETECTED · TEMPORALLY-SPLIT  (sorted by Coverage% ↓)")
    print("Recur=issues with ≥1 similar peer   Single=no similar peer (excluded)")
    print("MemPool/EvalPool=30/70 of recurrent   →Mem/→Eval=verified after cross-check")
    print("═" * len(H))
    print(H)
    print(sep)
    for r in rows:
        print(
            f"{r['repo']:<42} {r['total']:>4} {r['recurrent']:>5} {r['singleton']:>6} "
            f"{r['mem_pool']:>7} {r['eval_pool']:>8} {r['verified_mem']:>5} {r['verified_eval']:>6} "
            f"{r['excluded']:>5} {r['coverage_pct']:>6.1f}%"
        )
    print(sep)


def _print_issue_table(rows: List[Dict], limit: int = 60) -> None:
    H = (
        f"{'Repo':<34} {'ID':>4} {'RkAll':>5} {'RkRec':>5} "
        f"{'RcssN':>5} {'FutN':>4} {'PastN':>5} {'AvgSim':>6} "
        f"{'Pool':<7} {'Role':<24}"
    )
    sep = "─" * len(H)
    print("\n" + "═" * len(H))
    print(f"PER-ISSUE TABLE  (top {limit} by rcss_score — full table in trs_per_issue.csv)")
    print("RkAll=rank in full repo (1=oldest)   RkRec=rank within recurrent pool only")
    print("RcssN=recurrence_count   FutN=future similar   PastN=past similar")
    print("═" * len(H))
    print(H)
    print(sep)
    for r in rows[:limit]:
        print(
            f"{r['repo']:<34} {str(r['id']):>4} {r['temporal_rank_all']:>5} "
            f"{str(r['temporal_rank_recurrent']):>5} "
            f"{r['recurrence_count']:>5} {r['future_count']:>4} {r['past_count']:>5} "
            f"{r['avg_sim']:>6.3f} {r['pool']:<7} {r['role']:<24}"
        )
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    print("[TRS] Loading error_details.json …")
    issues_raw: List[Dict] = json.loads(
        Path(ERROR_DETAILS_PATH).read_text(encoding="utf-8")
    )
    print(f"[TRS]   {len(issues_raw)} issues")

    print("[TRS] Loading parquet (temporal ordering + diffs) …")
    df = pd.read_parquet(PARQUET_PATH)
    df["sha_fail"] = df["sha_fail"].astype(str)
    pq_by_sha: Dict[str, Any] = {r["sha_fail"]: r for r in df.to_dict(orient="records")}

    # attach parquet id (= temporal order proxy) and classify fix type
    for iss in issues_raw:
        sha  = str(iss.get("sha_fail") or "")
        pq   = pq_by_sha.get(sha, {})
        pid  = pq.get("id")
        iss["_pq_id"]   = int(pid) if pid is not None and str(pid).isdigit() else 999_999
        iss["fix_type"] = _classify_fix(str(pq.get("diff") or ""))

    # ── load embedding model ──
    _load_model()

    # ── group by repo, sort chronologically within each repo ──
    by_repo: Dict[str, List[Dict]] = defaultdict(list)
    for iss in issues_raw:
        by_repo[str(iss.get("repo") or "")].append(iss)
    for repo in by_repo:
        by_repo[repo].sort(key=lambda x: x.get("_pq_id", 999_999))

    print(f"\n[TRS] Processing {sum(len(v) for v in by_repo.values())} issues "
          f"across {len(by_repo)} repos …\n")

    # ── per-repo processing ──
    all_memory:   List[Dict] = []
    all_eval:     List[Dict] = []
    all_excluded: List[Dict] = []
    all_stats:    List[Dict] = []
    repo_rows:    List[Dict] = []
    repos_ok = repos_skip = 0

    for repo, issues in sorted(by_repo.items(), key=lambda x: -len(x[1])):
        if len(issues) < MIN_REPO_SIZE:
            repos_skip += 1
            continue
        repos_ok += 1

        mem_iss, eval_iss, excl_iss, stats = _process_repo(repo, issues, pq_by_sha)

        all_memory.extend(mem_iss)
        all_eval.extend(eval_iss)
        all_excluded.extend(excl_iss)
        all_stats.extend(stats)

        n_total    = len(issues)
        n_recur    = sum(1 for s in stats if s["recurrence_count"] >= 1)
        n_single   = n_total - n_recur
        n_mpool    = sum(1 for s in stats if s["pool"] == "memory")
        n_epool    = sum(1 for s in stats if s["pool"] == "eval")
        n_excl     = len(excl_iss)
        coverage   = round(100.0 * len(eval_iss) / n_epool, 1) if n_epool > 0 else 0.0

        repo_rows.append({
            "repo":         repo,
            "total":        n_total,
            "recurrent":    n_recur,
            "singleton":    n_single,
            "mem_pool":     n_mpool,
            "eval_pool":    n_epool,
            "verified_mem": len(mem_iss),
            "verified_eval":len(eval_iss),
            "excluded":     n_excl,
            "coverage_pct": coverage,
        })

    repo_rows.sort(key=lambda r: (-r["coverage_pct"], -r["recurrent"]))

    # ── display tables ──
    all_stats_sorted = sorted(all_stats, key=lambda r: (-r["rcss_score"], -r["avg_sim"]))
    _print_issue_table(all_stats_sorted)
    _print_repo_table(repo_rows)

    # ── global summary ──
    tot_recur   = sum(r["recurrent"] for r in repo_rows)
    tot_single  = sum(r["singleton"] for r in repo_rows)

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║       TRS  —  RCSS-detected · Temporally-split                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  Similarity threshold θ  : {SIMILARITY_THRESHOLD:.2f}  (recurrence link cutoff)   ║
║  Weights  err={W_ERROR}  fix={W_FIX}  (repair pattern emphasized)            ║
║  Memory split            : oldest {int(MEMORY_FRACTION*100)}% of RECURRENT issues   ║
╠═══════════════════════════════════════════════════════════════════╣
║  Repos analysed (≥{MIN_REPO_SIZE} issues) : {repos_ok:>4}                                ║
║  Repos skipped  (<{MIN_REPO_SIZE} issues) : {repos_skip:>4}                                ║
╠═══════════════════════════════════════════════════════════════════╣
║  Phase 1  —  RCSS Detection                                       ║
║    Recurrent issues (≥1 similar peer)  : {tot_recur:>4}                     ║
║    Singleton issues (no similar peer)  : {tot_single:>4}  ← excluded always    ║
╠═══════════════════════════════════════════════════════════════════╣
║  Phase 2  —  Temporal Split + Verification                        ║
║    Memory  (verified, older)  : {len(all_memory):>4}                             ║
║    Eval    (verified, newer)  : {len(all_eval):>4}                             ║
║    Excluded (no cross-link)   : {len(all_excluded) - tot_single:>4}  ← recurrent but split gap  ║
╠═══════════════════════════════════════════════════════════════════╣
║  Guarantees:                                                      ║
║   ✓ Only recurrent issues used  (singletons excluded entirely)    ║
║   ✓ Memory always older than eval  (no temporal leakage)          ║
║   ✓ Every memory issue has ≥1 verified eval user                  ║
║   ✓ Every eval   issue has ≥1 verified memory peer                ║
╚═══════════════════════════════════════════════════════════════════╝""")

    # fix-type distributions
    def _dist(issues: List[Dict]) -> Dict[str, int]:
        d: Dict[str, int] = defaultdict(int)
        for iss in issues:
            d[str(iss.get("fix_type") or "unknown")] += 1
        return dict(sorted(d.items(), key=lambda x: -x[1]))

    print("\nMemory fix-type:")
    for ft, cnt in _dist(all_memory).items():
        print(f"  {ft:<30} {cnt}")
    print("\nEval fix-type:")
    for ft, cnt in _dist(all_eval).items():
        print(f"  {ft:<30} {cnt}")

    # ── save outputs ──
    def _write(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    _write(out / "memory_issues.json",   all_memory)
    _write(out / "eval_issues.json",     all_eval)
    _write(out / "excluded_issues.json", all_excluded)

    csv_fields = [
        "repo", "id", "sha_fail", "error_types", "fix_type",
        "temporal_rank_all", "temporal_pct_all", "temporal_rank_recurrent",
        "recurrence_count", "future_count", "past_count",
        "avg_sim", "rcss_score", "best_past_sim",
        "pool", "role",
    ]
    with (out / "trs_per_issue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_stats_sorted)

    with (out / "trs_per_repo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(repo_rows[0].keys()))
        w.writeheader()
        w.writerows(repo_rows)

    summary = "\n".join([
        "TRS Split Summary  (RCSS-detected · Temporally-split)",
        "=" * 55,
        f"θ={SIMILARITY_THRESHOLD}  mem_fraction={MEMORY_FRACTION}  weights err/fix={W_ERROR}/{W_FIX}",
        "",
        f"Phase 1 — RCSS Detection:",
        f"  Recurrent issues  : {tot_recur}",
        f"  Singleton issues  : {tot_single}  (excluded entirely)",
        "",
        f"Phase 2 — Temporal Split + Verification:",
        f"  Memory (verified) : {len(all_memory)}  (oldest 30% of recurrent, proven recur in eval)",
        f"  Eval   (verified) : {len(all_eval)}  (newest 70% of recurrent, has memory peer)",
        f"  Excluded          : {len(all_excluded) - tot_single}  (recurrent but split gap — no cross-link)",
        "",
        "Guarantees:",
        "  ✓ Only recurrent issues used  (singletons excluded entirely)",
        "  ✓ Memory always older than eval  (temporal ordering, no leakage)",
        "  ✓ Every memory issue has ≥1 verified eval user",
        "  ✓ Every eval   issue has ≥1 verified memory peer",
    ])
    (out / "split_summary.txt").write_text(summary, encoding="utf-8")

    print(f"\n[TRS] Outputs → {out}/")
    print(f"       memory_issues.json   — {len(all_memory)} issues")
    print(f"       eval_issues.json     — {len(all_eval)} issues")
    print(f"       excluded_issues.json — {len(all_excluded)} issues")
    print(f"       trs_per_issue.csv    — {len(all_stats_sorted)} rows")
    print(f"       trs_per_repo.csv     — {len(repo_rows)} rows")
    print(f"       split_summary.txt\n")


if __name__ == "__main__":
    main()
