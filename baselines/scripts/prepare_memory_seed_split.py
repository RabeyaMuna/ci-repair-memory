#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCHES = PROJECT_ROOT / "generated_patches_list" / "generated_patches_success_only.json"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "baselines" / "results"

TARGET_REPOS = ("agno", "axolotl", "conan", "flower")


def _normalize_path(path: str) -> str:
    return (path or "").strip().lstrip("/").replace("\\", "/")


def _basename(path: str) -> str:
    return os.path.basename(_normalize_path(path))


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return [value]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return [_jsonable(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _clip(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _extract_files_from_diff(diff_text: str) -> List[str]:
    return [
        _normalize_path(m.group(1).strip())
        for m in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text or "", re.MULTILINE)
        if m.group(1).strip()
    ]


def _extract_file_diff(diff_text: str, file_path: str) -> str:
    target = _normalize_path(file_path)
    capture = False
    lines: List[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            if capture:
                break
            match = re.match(r"^diff --git a/.+ b/(.+)$", line)
            capture = bool(match and _normalize_path(match.group(1)) == target)
        if capture:
            lines.append(line)
    return "\n".join(lines)


def _extract_failed_commands(workflow: str) -> List[str]:
    commands: List[str] = []
    current_run: List[str] = []
    in_run = False
    run_indent = 0

    for raw in (workflow or "").splitlines():
        line = raw.rstrip("\n")
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if not stripped:
            continue

        if re.match(r"^-?\s*run:\s*\|?\s*$", line):
            in_run = True
            run_indent = indent
            current_run = []
            inline = line.split("run:", 1)[1].strip()
            if inline and inline != "|":
                commands.append(inline)
                in_run = False
            continue

        if in_run:
            if indent > run_indent:
                current_run.append(stripped)
                continue
            if current_run:
                cmd = " && ".join(current_run).strip()
                if cmd:
                    commands.append(cmd)
            in_run = False
            current_run = []

        if stripped.startswith("run:"):
            cmd = stripped.split("run:", 1)[1].strip()
            if cmd and cmd != "|":
                commands.append(cmd)

    if in_run and current_run:
        cmd = " && ".join(current_run).strip()
        if cmd:
            commands.append(cmd)

    deduped: List[str] = []
    for cmd in commands:
        cleaned = " ".join(cmd.split())
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def _infer_failed_tools(workflow: str, logs_blob: str) -> List[str]:
    text = f"{workflow}\n{logs_blob}".lower()
    tool_map = {
        "ruff": ("ruff",),
        "pytest": ("pytest",),
        "mypy": ("mypy",),
        "black": ("black",),
        "isort": ("isort",),
        "pre-commit": ("pre-commit",),
        "taplo": ("taplo",),
        "tox": ("tox",),
        "cmake": ("cmake",),
    }
    tools = [name for name, patterns in tool_map.items() if any(p in text for p in patterns)]
    return tools


def _summarize_logs(logs: Any, changed_files: List[str]) -> str:
    text = json.dumps(_jsonable(logs), ensure_ascii=False)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    interesting: List[str] = []
    needles = tuple({_basename(path) for path in changed_files if path})
    for line in lines:
        lower = line.lower()
        if any(tok in lower for tok in ("error", "failed", "traceback", "exception", "warning:", "assert")):
            interesting.append(line)
        elif needles and any(name.lower() in lower for name in needles):
            interesting.append(line)
        if len(interesting) >= 8:
            break

    if not interesting:
        interesting = lines[-6:]
    return _clip(" | ".join(interesting), 1200)


def _classify_issue_type(error_types: List[str], diff_text: str, workflow_path: str) -> str:
    lower_errors = " ".join(error_types).lower()
    lower_diff = diff_text.lower()
    lower_workflow = workflow_path.lower()

    if "format" in lower_errors or "lint" in lower_errors:
        return "formatting"
    if "syntax" in lower_errors:
        return "syntax"
    if "dependency" in lower_errors or "environment" in lower_errors:
        return "dependency_or_env"
    if "test" in lower_errors:
        return "test_failure"
    if lower_workflow.endswith(".yml") or lower_workflow.endswith(".yaml"):
        if ".github/workflows/" in lower_workflow:
            return "workflow_config"
    if any(tok in lower_diff for tok in ("import ", "from ", "pyproject.toml", "requirements.txt")):
        return "config_or_import"
    return "other"


def _classify_error_level(file_path: str) -> List[str]:
    path = _normalize_path(file_path).lower()
    levels: List[str] = []
    if ".github/workflows/" in path or path.endswith((".yml", ".yaml", ".toml", ".ini", ".cfg")):
        levels.append("config")
    if "/test" in path or path.startswith("test/") or "/tests/" in path:
        levels.append("test")
    if path.endswith((".py", ".pyi", ".tsx", ".ts", ".js")):
        levels.append("code")
    if not levels:
        levels.append("project")
    return levels


def _build_seed_candidates(rows: List[Dict[str, Any]], per_repo: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["repo_name"]].append(row)

    seed: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []

    for repo in TARGET_REPOS:
        repo_rows = sorted(
            grouped.get(repo, []),
            key=lambda item: (
                -len(item.get("error_types", [])),
                -len(item.get("ground_truth_files", [])),
                str(item.get("id", "")),
            ),
        )
        selected = _select_diverse(repo_rows, per_repo)
        selected_ids = {row["sha_fail"] for row in selected}
        seed.extend(selected)
        rest.extend([row for row in repo_rows if row["sha_fail"] not in selected_ids])
    return seed, rest


def _select_diverse(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_issue_types: set[str] = set()
    seen_files: set[str] = set()

    for row in rows:
        if len(selected) >= limit:
            break
        issue_type = row.get("issue_type", "")
        files = set(row.get("ground_truth_files", []))
        if issue_type not in seen_issue_types:
            selected.append(row)
            seen_issue_types.add(issue_type)
            seen_files.update(files)

    for row in rows:
        if len(selected) >= limit:
            break
        if row in selected:
            continue
        files = set(row.get("ground_truth_files", []))
        if not files or not files.issubset(seen_files):
            selected.append(row)
            seen_files.update(files)

    for row in rows:
        if len(selected) >= limit:
            break
        if row not in selected:
            selected.append(row)

    return selected[:limit]


def _build_l1(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in seed_rows:
        for file_path in row.get("ground_truth_files", []):
            failure_pattern = row["issue_type"]
            failure_reason = row["error_context_summary"]
            out.append(
                {
                    "issue_id": row["id"],
                    "sha_fail": row["sha_fail"],
                    "repo": row["repo_name"],
                    "repo_name": row["repo_name"],
                    "file": file_path,
                    "line_range": [],
                    "error_type": row["primary_error_type"],
                    "error_subtype": "",
                    "issue_type": row["issue_type"],
                    "failure_pattern": failure_pattern,
                    "failed_cmd": row["failed_cmd"],
                    "failed_tool": row["failed_tool"],
                    "dependent_files": [x for x in row["ground_truth_files"] if x != file_path],
                    "reason": failure_reason,
                    "failure_reason": failure_reason,
                    "fix_strategy": _clip(_extract_file_diff(row["diff"], file_path), 500),
                    "error_level": _classify_error_level(file_path),
                    "recommended_api": [],
                    "workflow_path": row["workflow_path"],
                    "workflow_name": row["workflow_name"],
                    "workflow_validation": {
                        "validation_type": row["workflow_name"],
                        "validation_commands": row["failed_cmd"],
                    },
                }
            )
    return out


def _build_l2(seed_rows: List[Dict[str, Any]], l1_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["repo_name"], row["primary_error_type"], row["issue_type"])].append(row)

    by_issue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in l1_rows:
        by_issue[str(row["issue_id"])].append(row)

    out: List[Dict[str, Any]] = []
    for (repo_name, error_type, issue_type), members in grouped.items():
        files: List[Dict[str, Any]] = []
        file_seen = set()
        changed_files: List[str] = []
        failure_reasons: List[str] = []
        failed_cmd: List[str] = []
        failed_tool: List[str] = []
        error_levels: List[str] = []
        fix_strategies: List[str] = []

        for member in members:
            for l1 in by_issue.get(str(member["id"]), []):
                fp = l1["file"]
                if fp not in file_seen:
                    file_seen.add(fp)
                    files.append(
                        {
                            "file": fp,
                            "line_range": l1.get("line_range", []),
                            "reason": l1.get("reason", ""),
                            "failure_reason": l1.get("failure_reason", ""),
                            "failure_pattern": l1.get("failure_pattern", ""),
                            "issue_type": l1.get("issue_type", ""),
                            "error_type": l1.get("error_type", ""),
                        }
                    )
                for dep in l1.get("dependent_files", []):
                    if dep not in changed_files:
                        changed_files.append(dep)
                if fp not in changed_files:
                    changed_files.append(fp)

            if member["error_context_summary"] not in failure_reasons:
                failure_reasons.append(member["error_context_summary"])
            for cmd in member["failed_cmd"]:
                if cmd not in failed_cmd:
                    failed_cmd.append(cmd)
            for tool in member["failed_tool"]:
                if tool not in failed_tool:
                    failed_tool.append(tool)
            for gf in member["ground_truth_files"]:
                if gf not in changed_files:
                    changed_files.append(gf)
                for lvl in _classify_error_level(gf):
                    if lvl not in error_levels:
                        error_levels.append(lvl)
            snippet = _clip(_extract_file_diff(member["diff"], member["ground_truth_files"][0]), 220) if member["ground_truth_files"] else ""
            if snippet and snippet not in fix_strategies:
                fix_strategies.append(snippet)

        out.append(
            {
                "repo": repo_name,
                "repo_name": repo_name,
                "error_type": error_type,
                "issue_type": issue_type,
                "failed_cmd": failed_cmd,
                "error_level": error_levels,
                "files": files,
                "changed_files": changed_files,
                "changed_file_entries": [{"file": x, "line_range": [], "fix_family": "", "issue_id": ""} for x in changed_files],
                "fix_strategy": fix_strategies[0] if fix_strategies else "",
                "fix_strategies": fix_strategies,
                "failure_reason": _clip(" | ".join(failure_reasons), 500),
                "failure_reasons": failure_reasons[:5],
                "failure_patterns": sorted(
                    {
                        str(item.get("failure_pattern") or "").strip()
                        for item in files
                        if str(item.get("failure_pattern") or "").strip()
                    }
                ),
                "failed_tool": failed_tool,
                "pattern_name": f"{error_type} {issue_type}".strip(),
                "issue_ids": [m["id"] for m in members],
                "seed_count": len(members),
            }
        )
    return out


def _build_l3(seed_rows: List[Dict[str, Any]], l2_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        grouped[(row["error_type"], row["issue_type"])].append(row)

    out: List[Dict[str, Any]] = []
    for (error_type, issue_type), members in grouped.items():
        repos: List[str] = []
        repo_names: List[str] = []
        failed_cmd: List[str] = []
        failed_tool: List[str] = []
        error_level: List[str] = []
        failure_reasons: List[str] = []
        fix_strategies: List[str] = []
        changed_files: List[str] = []
        evidence_issue_ids: List[str] = []

        for member in members:
            if member["repo"] not in repos:
                repos.append(member["repo"])
            if member["repo_name"] not in repo_names:
                repo_names.append(member["repo_name"])
            for cmd in member["failed_cmd"]:
                if cmd not in failed_cmd:
                    failed_cmd.append(cmd)
            for tool in member["failed_tool"]:
                if tool not in failed_tool:
                    failed_tool.append(tool)
            for lvl in member["error_level"]:
                if lvl not in error_level:
                    error_level.append(lvl)
            for reason in member["failure_reasons"]:
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
            for strat in member["fix_strategies"]:
                if strat not in fix_strategies:
                    fix_strategies.append(strat)
            for path in member["changed_files"]:
                if path not in changed_files:
                    changed_files.append(path)
            for iid in member["issue_ids"]:
                if iid not in evidence_issue_ids:
                    evidence_issue_ids.append(iid)

        reason_snippet = _clip(" | ".join(failure_reasons[:2]), 300) if failure_reasons else ""
        principle = (
            f"error_type={error_type}"
            + (f" issue_type={issue_type}" if issue_type else "")
            + (f": {reason_snippet}" if reason_snippet else "")
            + f" (seen in repos: {', '.join(repo_names) if repo_names else 'multiple'})"
        )
        out.append(
            {
                "error_type": error_type,
                "issue_type": issue_type,
                "failure_pattern": issue_type,
                "repos": repos,
                "repo_names": repo_names,
                "failed_tool": failed_tool,
                "failed_cmd": failed_cmd,
                "error_level": error_level,
                "principle": _clip(principle, 500),
                "failure_reasons": failure_reasons[:3],
                "failure_patterns": sorted(
                    {
                        str(member.get("issue_type") or "").strip()
                        for member in members
                        if str(member.get("issue_type") or "").strip()
                    }
                ),
                "fix_strategies": fix_strategies[:5],
                "fix_strategy": fix_strategies[0] if fix_strategies else "",
                "evidence_issue_ids": evidence_issue_ids,
                "recommended_tools": failed_tool[:4],
                "changed_files": changed_files,
                "changed_file_entries": [{"file": x, "line_range": [], "fix_family": "", "issue_id": ""} for x in changed_files],
            }
        )
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_rows(dataset_path: Path, patches_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    df = pd.read_parquet(dataset_path)
    patches = json.loads(patches_path.read_text(encoding="utf-8"))
    patch_by_sha = {str(row["sha_fail"]): row for row in patches}

    all_rows: List[Dict[str, Any]] = []
    prepared_with_gt: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        sha_fail = str(row.get("sha_fail") or "")
        patch = patch_by_sha.get(sha_fail)
        repo_name = str(row.get("repo_name") or "")
        if repo_name not in TARGET_REPOS:
            continue

        error_types = [str(x) for x in _safe_list(row.get("error_type")) if str(x).strip()]
        workflow = str(row.get("workflow") or "")
        changed_files = [_normalize_path(str(x)) for x in _safe_list(row.get("changed_files")) if str(x).strip()]
        diff_text = str((patch or {}).get("diff") or row.get("diff") or "")
        ground_truth_files = _extract_files_from_diff(diff_text) or changed_files
        logs = row.get("logs")
        logs_summary = _summarize_logs(logs, ground_truth_files)
        failed_cmd = _extract_failed_commands(workflow)
        failed_tool = _infer_failed_tools(workflow, json.dumps(_jsonable(logs), ensure_ascii=False))
        issue_type = _classify_issue_type(error_types, diff_text, str(row.get("workflow_path") or ""))

        prepared_row = {
            "id": str(row.get("id") or ""),
            "sha_fail": sha_fail,
            "sha_success": str(row.get("sha_success") or (patch or {}).get("sha_success") or ""),
            "repo_name": repo_name,
            "repo_owner": str(row.get("repo_owner") or (patch or {}).get("repo_owner") or ""),
            "workflow_name": str(row.get("workflow_name") or ""),
            "workflow_filename": str(row.get("workflow_filename") or ""),
            "workflow_path": str(row.get("workflow_path") or ""),
            "workflow": workflow,
            "logs_summary": logs_summary,
            "error_types": error_types,
            "primary_error_type": error_types[0] if error_types else "",
            "changed_files": changed_files,
            "ground_truth_files": ground_truth_files,
            "diff": diff_text,
            "failed_cmd": failed_cmd,
            "failed_tool": failed_tool,
            "issue_type": issue_type,
            "error_context_summary": logs_summary,
            "has_ground_truth_patch": bool(patch and diff_text.strip()),
        }
        all_rows.append(prepared_row)
        if prepared_row["has_ground_truth_patch"]:
            prepared_with_gt.append(
                dict(prepared_row)
            )

    return all_rows, prepared_with_gt


def main() -> int:
    parser = argparse.ArgumentParser(description="Select seed issues per repo and write seed/eval splits.")
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-repo", type=int, default=5)
    args = parser.parse_args()

    all_rows, gt_rows = _prepare_rows(args.dataset, args.patches)
    seed_rows, _unused_success_rest = _build_seed_candidates(gt_rows, args.per_repo)
    seed_sha = {row["sha_fail"] for row in seed_rows}
    eval_rows = [row for row in all_rows if row["sha_fail"] not in seed_sha]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(args.output_dir / "memory_seed_issues.json", seed_rows)
    _write_json(args.output_dir / "memory_eval_issues.json", eval_rows)

    summary = {
        "target_repos": list(TARGET_REPOS),
        "per_repo_requested": args.per_repo,
        "seed_counts": {
            repo: sum(1 for row in seed_rows if row["repo_name"] == repo)
            for repo in TARGET_REPOS
        },
        "eval_counts": {
            repo: sum(1 for row in eval_rows if row["repo_name"] == repo)
            for repo in TARGET_REPOS
        },
        "eligible_ground_truth_counts": {
            repo: sum(1 for row in gt_rows if row["repo_name"] == repo)
            for repo in TARGET_REPOS
        },
        "total_repo_issue_counts": {
            repo: sum(1 for row in all_rows if row["repo_name"] == repo)
            for repo in TARGET_REPOS
        },
        "output_dir": str(args.output_dir),
        "next_step": "Run analyze_memory_seed_issues.py then build_memory_bank_from_seed_analysis.py to build the memory bank.",
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
