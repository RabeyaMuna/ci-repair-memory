#!/usr/bin/env python3
"""
SACS: Similarity-Aware Chronological Split
===========================================
Step 1  — Check recurrence and similarity across all issues.
Step 2  — Split ONLY issues that have genuine recurrence patterns.
Step 3  — Oldest 70% of each recurring cluster → memory (previous experience).
          Newest 30% → test (future problems to solve).

Issues with no recurrence (cluster size < MIN_GROUP_SIZE) go entirely to
memory — they cannot be meaningfully evaluated but can still serve as
background context.

Similarity formula (fully cosine-based):
    sim(A, B) = 0.3125 × cosine(error_doc_A, error_doc_B)
              + 0.6875 × cosine(fix_doc_A, fix_doc_B)

Inputs:
    error_details.json        — per-issue failure signals (effected files,
                                 failure reasons, error types, failed cmds)
    lca_dataset.parquet       — ground-truth diffs (sha_fail is join key)

Outputs (written to OUTPUT_DIR):
    recurrence_report.json    — per-repo cluster analysis
    memory_split.json         — issues qualified as memory
    test_split.json           — issues for evaluation
    split_summary.txt         — human-readable overview
"""

from __future__ import annotations

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
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/results/sacs_split"
)

# ── Thresholds ─────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD   = 0.60   # minimum sim to place two issues in same cluster
MEMORY_QUALIFY_THRESH  = 0.65   # minimum sim to retrieve a memory at inference time
MIN_GROUP_SIZE         = 3      # clusters smaller than this → all memory, no test split

# ── Similarity weights ─────────────────────────────────────────────────────────

W_ERROR = 0.3125   # what broke (error type + failure reasoning + failed cmds)
W_FIX   = 0.6875   # what fixed it (derived fix strategy from diff analysis)

# ── Embedding model ────────────────────────────────────────────────────────────

_model: Any = None
_backend: str = "none"
_cache: Dict[str, np.ndarray] = {}


def _load_model() -> None:
    global _model, _backend
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _backend = "sentence_transformers"
        print("[SACS] Loaded: sentence-transformers/all-MiniLM-L6-v2")
        return
    except Exception as e:
        print(f"[SACS] sentence-transformers unavailable ({e}); trying fastembed…")

    try:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-base-en-v1.5")
        _backend = "fastembed"
        print("[SACS] Loaded: fastembed/BAAI/bge-base-en-v1.5")
        return
    except Exception as e:
        raise RuntimeError(
            f"No embedding model available. Install sentence-transformers or fastembed.\n{e}"
        )


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

# Order matters — more specific patterns first
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
    """Return a semantic fix-type label by inspecting the diff."""
    if not diff:
        return "unknown_fix"
    for pattern, label in _FIX_PATTERNS:
        if re.search(pattern, diff, re.IGNORECASE):
            return label
    added   = sum(1 for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.split("\n") if l.startswith("-") and not l.startswith("---"))
    if added + removed <= 4:
        return "minor_fix"
    return "structural_fix"


def _changed_file_exts(diff: str) -> List[str]:
    files = re.findall(r"diff --git a/(\S+)", diff or "")
    return list({Path(f).suffix.lower() for f in files if Path(f).suffix})


def build_fix_doc(failure_reasons: List[str], diff: str) -> str:
    """
    Derive a semantic fix-strategy descriptor:
        fix_type | complaint | affected file types
    This is NOT the raw diff — it is a derived natural language representation
    so that two issues requiring the same class of fix score high regardless
    of which specific file or line was changed.
    """
    fix_type   = classify_fix(diff)
    file_exts  = _changed_file_exts(diff)

    # Extract core complaint: first meaningful sentence from failure reason
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
    error_doc = overall_error_types
              + overall_failure_reasons (truncated)
              + per-file: issue_type + error_subtype + failed_tools + failed_command
    """
    parts: List[str] = []
    parts.extend(issue.get("overall_error_types") or [])
    for reason in (issue.get("overall_failure_reasons") or []):
        parts.append(reason[:350])
    for ef in (issue.get("effected_files") or []):
        parts.append(str(ef.get("issue_type") or ""))
        parts.append(str(ef.get("error_subtype") or ""))
        for t in (ef.get("failed_tools") or []):
            parts.append(str(t).lower())
        cmd = ef.get("failed_command") or ""
        parts.append(cmd[:120])
    return " | ".join(x for x in parts if x.strip())


# ── Similarity ─────────────────────────────────────────────────────────────────

def compute_similarity(a: Dict, b: Dict) -> float:
    """
    Weighted cosine similarity across failure-context and repair-pattern signals.
    Both vectors are pre-computed L2-normalised embeddings so that
    np.dot == cosine similarity.
    """
    error_sim    = cosine(a.get("error_vec"),    b.get("error_vec"))
    fix_sim      = cosine(a.get("fix_vec"),      b.get("fix_vec"))
    return W_ERROR * error_sim + W_FIX * fix_sim


# ── Union-Find (for agglomerative clustering) ──────────────────────────────────

class UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank   = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1

    def clusters(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


def cluster_repo_issues(
    repo_issue_indices: List[int],
    issues: List[Dict],
) -> List[List[int]]:
    """
    Single-linkage agglomerative clustering for one repo's issues.
    Two issues are merged into the same cluster if sim ≥ SIMILARITY_THRESHOLD.
    """
    n = len(repo_issue_indices)
    if n < 2:
        return [[repo_issue_indices[i]] for i in range(n)]

    uf = UnionFind(n)

    # Compute all pairwise similarities and collect pairs above threshold
    pairs: List[Tuple[float, int, int]] = []
    for a in range(n):
        for b in range(a + 1, n):
            s = compute_similarity(
                issues[repo_issue_indices[a]],
                issues[repo_issue_indices[b]],
            )
            if s >= SIMILARITY_THRESHOLD:
                pairs.append((s, a, b))

    # Merge highest-similarity pairs first (greedy single-linkage)
    for _, a, b in sorted(pairs, reverse=True):
        uf.union(a, b)

    return [
        [repo_issue_indices[local_i] for local_i in local_cluster]
        for local_cluster in uf.clusters().values()
    ]


# ── Chronological split ────────────────────────────────────────────────────────

def chronological_split(
    cluster: List[int],
    issues: List[Dict],
) -> Tuple[List[int], List[int]]:
    """
    Sort cluster by temporal order (parquet row index as proxy),
    then split 70 % memory / 30 % test.
    The oldest issues become memory (previous experience).
    The newest issues become test (future problems to solve).
    """
    ordered = sorted(cluster, key=lambda i: issues[i]["order_idx"])
    split_point = max(1, int(len(ordered) * 0.70))
    return ordered[:split_point], ordered[split_point:]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_and_embed(error_details_path: str, parquet_path: str) -> List[Dict]:
    """
    Load both data sources, join on sha_fail, build semantic documents, embed.
    """
    with open(error_details_path) as f:
        raw_issues: List[Dict] = json.load(f)

    df = pd.read_parquet(parquet_path)
    diff_map: Dict[str, str]  = dict(zip(df["sha_fail"], df["diff"].fillna("")))
    # Use parquet row index as chronological proxy
    order_map: Dict[str, int] = {sha: idx for idx, sha in enumerate(df["sha_fail"])}

    enriched: List[Dict] = []
    total = len(raw_issues)
    print(f"[SACS] Embedding {total} issues (this may take ~1-2 min)…")

    for idx, issue in enumerate(raw_issues):
        sha   = issue.get("sha_fail", "")
        diff  = diff_map.get(sha, "")

        error_doc = build_error_doc(issue)
        fix_doc   = build_fix_doc(
            issue.get("overall_failure_reasons") or [], diff
        )

        enriched.append({
            **issue,
            "diff":      diff,
            "fix_type":  classify_fix(diff),
            "error_doc": error_doc,
            "fix_doc":   fix_doc,
            "error_vec": embed(error_doc),
            "fix_vec":   embed(fix_doc),
            "order_idx": order_map.get(sha, idx),  # chronological proxy
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"[SACS]   {idx + 1}/{total} embedded")

    return enriched


def run_sacs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load & embed ───────────────────────────────────────────────────
    _load_model()
    issues = load_and_embed(ERROR_DETAILS_PATH, PARQUET_PATH)

    # ── Step 2: Recurrence clustering per repo ─────────────────────────────────
    print("\n[SACS] Clustering issues by similarity within each repo…")

    repo_index: Dict[str, List[int]] = defaultdict(list)
    for i, issue in enumerate(issues):
        repo_index[issue["repo"]].append(i)

    recurrence_report: Dict[str, Any] = {}
    memory_ids: List[str] = []
    test_ids:   List[str] = []

    n_recurring  = 0
    n_too_small  = 0
    n_singleton  = 0

    for repo, indices in sorted(repo_index.items()):
        clusters = cluster_repo_issues(indices, issues)

        repo_entry: Dict[str, Any] = {
            "total_issues": len(indices),
            "n_clusters":   len(clusters),
            "clusters":     [],
        }

        for cluster in clusters:
            size = len(cluster)
            fix_types    = list({issues[i]["fix_type"] for i in cluster})
            error_types  = list({
                et
                for i in cluster
                for et in (issues[i].get("overall_error_types") or [])
            })

            # Intra-cluster average similarity
            sims: List[float] = []
            for a in range(size):
                for b in range(a + 1, size):
                    sims.append(compute_similarity(issues[cluster[a]], issues[cluster[b]]))
            avg_sim = float(np.mean(sims)) if sims else 0.0

            if size < MIN_GROUP_SIZE:
                # Too small to split — all go to memory
                for i in cluster:
                    memory_ids.append(issues[i]["id"])
                n_too_small += size
                repo_entry["clusters"].append({
                    "size":                 size,
                    "decision":             f"memory_only (size < {MIN_GROUP_SIZE})",
                    "avg_intra_similarity": round(avg_sim, 4),
                    "fix_types":            fix_types,
                    "error_types":          error_types,
                    "memory_ids":           [issues[i]["id"] for i in cluster],
                    "test_ids":             [],
                })
                continue

            # ── Qualifying cluster: chronological split ────────────────────────
            mem_idx, tst_idx = chronological_split(cluster, issues)
            for i in mem_idx:
                memory_ids.append(issues[i]["id"])
            for i in tst_idx:
                test_ids.append(issues[i]["id"])
            n_recurring += size

            repo_entry["clusters"].append({
                "size":                 size,
                "decision":             f"{len(mem_idx)} memory / {len(tst_idx)} test",
                "avg_intra_similarity": round(avg_sim, 4),
                "fix_types":            fix_types,
                "error_types":          error_types,
                "memory_ids":           [issues[i]["id"] for i in mem_idx],
                "test_ids":             [issues[i]["id"] for i in tst_idx],
            })

        recurrence_report[repo] = repo_entry

    # ── Step 3: Singletons that had no similar pair at all ─────────────────────
    # (already handled above as "too small", but count them separately)
    n_singleton = sum(
        1 for repo, report in recurrence_report.items()
        for c in report["clusters"]
        if c["size"] == 1
    )

    # ── Step 4: Build output dicts ─────────────────────────────────────────────
    id_to_issue = {issue["id"]: issue for issue in issues}

    def _build_output(ids: List[str], role: str) -> List[Dict]:
        out = []
        for iid in ids:
            iss = id_to_issue.get(iid)
            if not iss:
                continue
            out.append({
                "id":                  iss["id"],
                "sha_fail":            iss["sha_fail"],
                "repo":                iss["repo"],
                "workflow_name":       iss.get("workflow_name"),
                "workflow_path":       iss.get("workflow_path"),
                "overall_error_types": iss.get("overall_error_types"),
                "fix_type":            iss["fix_type"],
                "split_role":          role,
            })
        return out

    memory_output = _build_output(memory_ids, "memory")
    test_output   = _build_output(test_ids,   "test")

    # ── Step 5: Write outputs ──────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/recurrence_report.json", "w") as f:
        json.dump(recurrence_report, f, indent=2)

    with open(f"{OUTPUT_DIR}/memory_split.json", "w") as f:
        json.dump(memory_output, f, indent=2)

    with open(f"{OUTPUT_DIR}/test_split.json", "w") as f:
        json.dump(test_output, f, indent=2)

    # ── Step 6: Human-readable summary ────────────────────────────────────────
    lines = [
        "=" * 60,
        "SACS Split Summary",
        "=" * 60,
        f"Total issues processed   : {len(issues)}",
        f"Issues in recurring clusters (≥ {MIN_GROUP_SIZE}) : {n_recurring}",
        f"Issues in small clusters (< {MIN_GROUP_SIZE})  : {n_too_small}",
        f"  of which true singletons : {n_singleton}",
        "",
        f"Memory set size  : {len(memory_output)}",
        f"Test set size    : {len(test_output)}",
        f"Memory/Test ratio: {len(memory_output) / max(len(test_output), 1):.2f}",
        "",
        f"Similarity threshold (cluster) : {SIMILARITY_THRESHOLD}",
        f"Similarity threshold (retrieve): {MEMORY_QUALIFY_THRESH}",
        f"Weights  error={W_ERROR}  fix={W_FIX}",
        "",
        "Top repos by issue count:",
    ]

    top_repos = sorted(
        recurrence_report.items(),
        key=lambda kv: -kv[1]["total_issues"],
    )[:15]
    for repo, report in top_repos:
        n_split = sum(
            1 for c in report["clusters"]
            if c["size"] >= MIN_GROUP_SIZE
        )
        lines.append(
            f"  {repo:<40} total={report['total_issues']:3d}  "
            f"clusters={report['n_clusters']:2d}  split_clusters={n_split}"
        )

    lines += [
        "",
        "Fix type distribution in test set:",
    ]
    from collections import Counter
    fix_counts = Counter(r["fix_type"] for r in test_output)
    for ft, cnt in fix_counts.most_common():
        lines.append(f"  {ft:<30} {cnt}")

    summary = "\n".join(lines)
    print("\n" + summary)

    with open(f"{OUTPUT_DIR}/split_summary.txt", "w") as f:
        f.write(summary)

    print(f"\n[SACS] All outputs written to: {OUTPUT_DIR}/")
    print(f"[SACS] Files: recurrence_report.json | memory_split.json | test_split.json | split_summary.txt")


if __name__ == "__main__":
    run_sacs()
