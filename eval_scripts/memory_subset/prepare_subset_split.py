#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCHES = PROJECT_ROOT / "generated_patches_list" / "generated_patches_success_only.json"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "baselines" / "results" / "trs"
DEFAULT_REPOS = ("agno", "camel", "flower")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _row_identity(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("id") or ""), str(row.get("sha_fail") or row.get("failure_sha") or ""))


def _identity_set(rows: Iterable[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    return {_row_identity(row) for row in rows}


def _load_memory_identities(memory_bank_dir: Path | None) -> Set[Tuple[str, str]]:
    if not memory_bank_dir:
        return set()
    identities: Set[Tuple[str, str]] = set()
    for filename in ("failure_memory.json", "repo_memory.json", "cross_memory.json"):
        path = memory_bank_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else list(payload.values()) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            issue_id = str(row.get("id") or row.get("issue_id") or row.get("dataset_id") or "")
            sha = str(row.get("sha_fail") or row.get("failure_sha") or row.get("commit_sha") or "")
            identities.add((issue_id, sha))
    return identities


def _repo_from_patch(row: Dict[str, Any]) -> str:
    repo_name = str(row.get("repo_name") or "").strip()
    if repo_name:
        return repo_name
    text = " ".join(
        str(row.get(key) or "")
        for key in ("repo_url", "success_url", "compare_api_url", "diff_source")
    )
    match = re.search(r"github\.com/(?:repos/)?[^/\s]+/([^/\s?#)]+)", text)
    return match.group(1).replace(".git", "") if match else ""


def _jsonable(value: Any) -> Any:
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


def _merge_patch_dataset_row(patch: Dict[str, Any], dataset_row: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**_jsonable(dataset_row), **_jsonable(patch)}
    for key in ("repo_name", "repo_owner", "workflow_path", "workflow", "sha_success"):
        if not merged.get(key) and dataset_row.get(key) is not None:
            merged[key] = _jsonable(dataset_row.get(key))
    if not merged.get("repo_name"):
        merged["repo_name"] = _repo_from_patch(patch)
    return merged


def _sort_key(row: Dict[str, Any]) -> tuple[int, str]:
    raw = str(row.get("id") or "")
    return (int(raw), raw) if raw.isdigit() else (10**12, raw)


def _repo_from_dataset_row(row: Dict[str, Any]) -> str:
    repo_name = str(row.get("repo_name") or "").strip()
    if repo_name:
        return repo_name
    repo = str(row.get("repo") or row.get("repo_url") or "").strip()
    if "/" in repo:
        return repo.rstrip("/").split("/")[-1].replace(".git", "")
    return repo


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a TRS-shaped memory/eval split for selected repos. "
            "Memory issues come from generated_patches_success_only.json; eval issues come from the dataset and exclude memory issues."
        )
    )
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repos", nargs="*", default=list(DEFAULT_REPOS), help="Repos to include. Defaults to agno camel flower.")
    parser.add_argument("--extra-repos", nargs="*", default=[], help="Additional repos to include on top of --repos.")
    parser.add_argument(
        "--eval-per-repo",
        type=int,
        default=5,
        help="Number of non-memory dataset issues to select for eval per repo. Default: 5.",
    )
    parser.add_argument(
        "--eval-max-per-repo",
        type=int,
        default=None,
        help="Deprecated alias for --eval-per-repo.",
    )
    parser.add_argument(
        "--memory-per-repo",
        type=int,
        default=None,
        help="Optional override: use only this many success-patch issues per repo as memory. Default: all.",
    )
    parser.add_argument(
        "--fill-eval-from-dataset",
        action="store_true",
        help="Deprecated compatibility flag. Eval rows are always selected from dataset rows not used as memory.",
    )
    parser.add_argument(
        "--existing-memory-bank",
        type=Path,
        default=None,
        help="Memory bank directory used to exclude already-memorized issues when filling eval rows from the dataset.",
    )
    args = parser.parse_args()
    if args.eval_max_per_repo is not None:
        args.eval_per_repo = args.eval_max_per_repo

    patches = json.loads(args.patches.read_text(encoding="utf-8"))
    dataset_rows = pd.read_parquet(args.dataset).to_dict(orient="records")
    by_sha = {str(row.get("sha_fail") or ""): row for row in dataset_rows}
    by_id = {str(row.get("id") or ""): row for row in dataset_rows}
    dataset_by_repo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        repo_name = _repo_from_dataset_row(row).lower()
        if repo_name:
            dataset_by_repo[repo_name].append(_jsonable(row))
    existing_memory_identities = _load_memory_identities(args.existing_memory_bank)

    selected_repos = []
    seen_repos = set()
    for repo in [*args.repos, *args.extra_repos]:
        key = str(repo or "").strip().lower()
        if key and key not in seen_repos:
            seen_repos.add(key)
            selected_repos.append(key)

    repos = set(selected_repos)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    unmatched: List[Dict[str, Any]] = []

    for patch in patches:
        repo_name = _repo_from_patch(patch).lower()
        if repo_name not in repos:
            continue
        dataset_row = by_sha.get(str(patch.get("sha_fail") or "")) or by_id.get(str(patch.get("id") or ""))
        if not dataset_row:
            unmatched.append({"id": patch.get("id"), "sha_fail": patch.get("sha_fail"), "repo_name": repo_name})
            continue
        merged = _merge_patch_dataset_row(patch, dataset_row)
        merged["repo_name"] = str(merged.get("repo_name") or repo_name)
        grouped[repo_name].append(merged)

    all_success_rows: List[Dict[str, Any]] = []
    seed_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    per_repo = {}

    for repo in selected_repos:
        repo_key = repo.lower()
        rows = sorted(grouped.get(repo_key, []), key=_sort_key)
        seeds = rows
        if args.memory_per_repo is not None:
            seeds = rows[: max(args.memory_per_repo, 0)]

        blocked = _identity_set(seeds) | existing_memory_identities
        blocked_ids = {issue_id for issue_id, _ in blocked if issue_id}
        blocked_shas = {sha for _, sha in blocked if sha}
        remaining: List[Dict[str, Any]] = []
        for candidate in sorted(dataset_by_repo.get(repo_key, []), key=_sort_key):
            issue_id, sha = _row_identity(candidate)
            if (issue_id and issue_id in blocked_ids) or (sha and sha in blocked_shas):
                continue
            eval_row = dict(candidate)
            eval_row.setdefault("repo_name", repo_key)
            eval_row["eval_source"] = "dataset_non_memory"
            remaining.append(eval_row)
            blocked_ids.add(issue_id)
            blocked_shas.add(sha)
            if args.eval_per_repo > 0 and len(remaining) >= args.eval_per_repo:
                break

        all_success_rows.extend(rows)
        seed_rows.extend(seeds)
        eval_rows.extend(remaining)
        per_repo[repo_key] = {
            "total_success_issues": len(rows),
            "memory": len(seeds),
            "eval": len(remaining),
            "seed_ids": [str(row.get("id") or "") for row in seeds],
            "eval_ids": [str(row.get("id") or "") for row in remaining],
        }

    summary = {
        "repos": selected_repos,
        "memory_per_repo": args.memory_per_repo,
        "eval_per_repo": args.eval_per_repo,
        "existing_memory_bank": str(args.existing_memory_bank) if args.existing_memory_bank else None,
        "split_policy": (
            "explicit_memory_per_repo"
            if args.memory_per_repo is not None
            else "all_success_patch_issues_used_as_memory_eval_from_dataset_non_memory"
        ),
        "total_success_issues": len(all_success_rows),
        "memory_issues": len(seed_rows),
        "eval_issues": len(eval_rows),
        "per_repo": per_repo,
        "unmatched_success_patch_rows": unmatched,
        "paths": {
            "all": str(args.output_dir / "all_success_issues.json"),
            "memory": str(args.output_dir / "memory_issues.json"),
            "seed_alias": str(args.output_dir / "memory_seed_issues.json"),
            "eval": str(args.output_dir / "eval_issues.json"),
        },
    }

    _write_json(args.output_dir / "all_success_issues.json", all_success_rows)
    _write_json(args.output_dir / "memory_issues.json", seed_rows)
    _write_json(args.output_dir / "memory_seed_issues.json", seed_rows)
    _write_json(args.output_dir / "eval_issues.json", eval_rows)
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
