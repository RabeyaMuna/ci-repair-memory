#!/usr/bin/env python3
"""
Phase 3 of the memory bank pipeline — LLM-based L1/L2/L3 extraction.

Reads:
  --seed-file      memory_seed_issues.json   (from prepare_memory_seed_split.py)
  --analysis-file  seed_log_details.json     (from analyze_memory_seed_issues.py)

Writes to --output-dir (default: baselines/results/):
  failure_memory.json   — L1: per-file failure records
  repo_memory.json      — L2: repo-level recurring patterns
  cross_memory.json     — L3: cross-repo generalizable principles
  memory_bank_summary.json

Pipeline:
  1. For each seed issue + log analysis → LLM prompt → L1 records (one per ground-truth file)
  2. Group L1 by (repo, error_type, failure_pattern/issue subtype) → LLM prompt → L2 patterns
  3. Group L2 by (error_type, failure_pattern/issue subtype) across repos → LLM prompt → L3 principles
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf


PROJECT_ROOT  = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from utilities.llm_provider import get_default_model_key, get_llm


DEFAULT_SEED_FILE     = PROJECT_ROOT / "baselines" / "results" / "memory_seed_issues.json"
DEFAULT_ANALYSIS_FILE = PROJECT_ROOT / "baselines" / "results" / "seed_log_details.json"
DEFAULT_OUTPUT        = PROJECT_ROOT / "baselines" / "results"
DEFAULT_CONFIG        = PROJECT_ROOT / "config.yaml"


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_path(path: str) -> str:
    return (path or "").strip().lstrip("/").replace("\\", "/")


def _clip(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _extract_files_from_diff(diff_text: str) -> List[str]:
    return [
        _normalize_path(m.group(1).strip())
        for m in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text or "", re.MULTILINE)
        if m.group(1).strip()
    ]


def _extract_file_diff(diff_text: str, file_path: str) -> str:
    target  = _normalize_path(file_path)
    capture = False
    lines: List[str] = []
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            if capture:
                break
            match   = re.match(r"^diff --git a/.+ b/(.+)$", line)
            capture = bool(match and _normalize_path(match.group(1)) == target)
        if capture:
            lines.append(line)
    return "\n".join(lines)


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON; return {} on failure."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: find the first { … } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


def _dedupe(items: List[str]) -> List[str]:
    seen: set = set()
    out:  List[str] = []
    for x in items:
        s = str(x or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x or "").strip()]
    text = str(value).strip()
    return [text] if text else []


def _canonicalize_issue_subtype(row: Dict[str, Any]) -> str:
    """Normalize noisy LLM subtype labels without losing the broad error type."""
    raw = " ".join(
        str(row.get(key) or "")
        for key in ("issue_subtype", "failure_pattern", "pattern_name", "issue_type", "root_cause_category")
    ).lower()
    path = _normalize_path(str(row.get("file") or ""))
    ext = Path(path).suffix.lower()

    if ext == ".rst" or "rst" in raw:
        if any(token in raw for token in ("heading", "title", "underline", "overline", "decoration", "adornment", "section")):
            return "rst-docs-heading-format"
        return "rst-docs-format"
    if ext in {".md", ".mdx"} or "markdown" in raw:
        return "markdown-docs-format"
    if "docstring" in raw:
        return "python-docstring-format"
    if any(token in raw for token in ("unused-import", "unused import", "f401")):
        return "unused-import"
    if any(token in raw for token in ("unsorted-import", "import-sort", "isort", "i001")):
        return "unsorted-import"
    if any(token in raw for token in ("undefined-name", "undefined name", "f821")):
        return "undefined-name"
    if any(token in raw for token in ("missing-optional-dependency", "optional-dependency", "missing optional")):
        return "missing-optional-dependency"
    if any(token in raw for token in ("dependency version", "version conflict", "incompatible-version", "incompatible version")):
        return "dependency-version-conflict"
    if any(token in raw for token in ("invalid type annotation", "type annotation", "type-check", "type checking", "mypy", "pyright")):
        return "type-contract"
    if any(token in raw for token in ("workflow", "path filter", "paths-filter", "trigger")):
        return "workflow-config"
    return str(
        row.get("issue_subtype")
        or row.get("failure_pattern")
        or row.get("pattern_name")
        or row.get("issue_type")
        or ""
    ).strip()


def _canonical_error_type(row: Dict[str, Any]) -> str:
    return str(row.get("error_type") or "").strip()


def _extract_workflow_run_commands(workflow_text: str) -> List[str]:
    commands: List[str] = []
    lines = (workflow_text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("run: |") or stripped.startswith("run: >"):
            base_indent = len(line) - len(line.lstrip())
            block: List[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_line.strip() and next_indent <= base_indent:
                    break
                text = next_line.strip()
                if text and not text.startswith("#"):
                    block.append(text)
                i += 1
            if block:
                commands.append("\n".join(block))
            continue
        if stripped.startswith("run: "):
            command = stripped[len("run: "):].strip()
            if command:
                commands.append(command)
        i += 1
    return _dedupe(commands)


def _resolve_referenced_script_snippets(
    *,
    repo_root: Path | None,
    run_commands: List[str],
    limit: int = 8,
) -> List[Dict[str, str]]:
    if repo_root is None or not repo_root.exists():
        return []

    snippets: List[Dict[str, str]] = []
    seen: set[str] = set()
    script_pattern = re.compile(r"(?:^|\s)(?:bash\s+|sh\s+)?(\./[A-Za-z0-9_./-]+\.(?:sh|bash)|\./[A-Za-z0-9_./-]+/test\.sh)")
    for command in run_commands:
        for match in script_pattern.finditer(command):
            rel = _normalize_path(match.group(1).lstrip("./"))
            if not rel or rel in seen:
                continue
            seen.add(rel)
            path = repo_root / rel
            if not path.exists() or not path.is_file():
                continue
            try:
                snippets.append({"path": rel, "content": _clip(path.read_text(encoding="utf-8", errors="replace"), 5000)})
            except Exception:
                continue
            if len(snippets) >= limit:
                return snippets
    return snippets


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _l1_record_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("sha_fail") or ""), _normalize_path(str(row.get("file") or "")))


def _subtype_key(row: Dict[str, Any]) -> str:
    return _canonicalize_issue_subtype(row)


def _l2_record_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("repo") or row.get("repo_name") or ""),
        _canonical_error_type(row),
        _subtype_key(row),
    )


def _l3_record_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (_canonical_error_type(row), _subtype_key(row))


def _structured_file_refs(value: Any) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    for item in value if isinstance(value, list) else ([] if value in (None, "") else [value]):
        if isinstance(item, dict):
            path = _normalize_path(str(item.get("file") or item.get("path") or ""))
            reason = str(item.get("reason") or item.get("modification_reason") or "").strip()
        else:
            path = _normalize_path(str(item or ""))
            reason = ""
        if path:
            refs.append({"file": path, "reason": reason})
    seen = set()
    out: List[Dict[str, str]] = []
    for ref in refs:
        key = (ref["file"], ref["reason"])
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return out


def _structured_issue_patterns(value: Any) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    for item in value if isinstance(value, list) else ([] if value in (None, "") else [value]):
        if isinstance(item, dict):
            issue = str(item.get("issue") or item.get("issue_subtype") or item.get("error_type") or "").strip()
            reason = str(item.get("reason") or item.get("why_it_appears") or "").strip()
            modification_hint = str(item.get("modification_hint") or item.get("fix_hint") or "").strip()
            files = [
                _normalize_path(str(x))
                for x in (item.get("files") or [])
                if str(x).strip()
            ]
            if issue or reason or files:
                patterns.append(
                    {
                        "issue": issue,
                        "reason": reason,
                        "modification_hint": modification_hint,
                        "files": _dedupe(files),
                    }
                )
        else:
            text = str(item or "").strip()
            if text:
                patterns.append({"issue": text, "reason": "", "modification_hint": "", "files": []})
    return patterns


def _dedupe_issue_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_issue: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pattern in patterns:
        issue = str(pattern.get("issue") or "").strip()
        reason = str(pattern.get("reason") or "").strip()
        key = (issue, reason)
        existing = by_issue.get(key)
        files = _dedupe([_normalize_path(str(x)) for x in (pattern.get("files") or []) if str(x).strip()])
        if existing is None:
            by_issue[key] = {
                "issue": issue,
                "reason": reason,
                "modification_hint": str(pattern.get("modification_hint") or "").strip(),
                "files": files,
            }
        else:
            existing["files"] = _dedupe(existing.get("files", []) + files)
            if not existing.get("modification_hint") and pattern.get("modification_hint"):
                existing["modification_hint"] = str(pattern.get("modification_hint") or "").strip()
    return [p for p in by_issue.values() if p.get("issue") or p.get("reason") or p.get("files")]


def _derive_dependent_issue_patterns(
    current_members: List[Dict[str, Any]],
    related_l1_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Promote same-issue sibling L1 groups to dependent_issue_patterns.

    This keeps broad related work, such as Flower RST docs formatting beside Python
    docstring formatting, out of per-file depended_files while preserving it for
    retrieval and organizer prompts.
    """
    if not current_members or not related_l1_rows:
        return []

    current_repo = str(current_members[0].get("repo") or current_members[0].get("repo_name") or "")
    current_key = (
        _canonical_error_type(current_members[0]),
        _subtype_key(current_members[0]),
    )
    current_issue_ids = {str(row.get("issue_id") or "") for row in current_members if str(row.get("issue_id") or "")}
    current_shas = {str(row.get("sha_fail") or "") for row in current_members if str(row.get("sha_fail") or "")}
    current_files = {_normalize_path(str(row.get("file") or "")) for row in current_members}

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in related_l1_rows:
        repo = str(row.get("repo") or row.get("repo_name") or "")
        if repo != current_repo:
            continue
        issue_id = str(row.get("issue_id") or "")
        sha = str(row.get("sha_fail") or "")
        if current_issue_ids and issue_id not in current_issue_ids:
            continue
        if not current_issue_ids and current_shas and sha not in current_shas:
            continue
        row_key = (_canonical_error_type(row), _subtype_key(row))
        row_file = _normalize_path(str(row.get("file") or ""))
        if row_key == current_key or row_file in current_files:
            continue
        grouped[row_key].append(row)

    patterns: List[Dict[str, Any]] = []
    for (error_type, subtype), rows in sorted(grouped.items()):
        files = _dedupe(_normalize_path(str(row.get("file") or "")) for row in rows if row.get("file"))
        tools = _dedupe(t for row in rows for t in (row.get("failed_tool") or []))
        cmds = _dedupe(c for row in rows for c in (row.get("failed_cmd") or []))
        tool_or_cmd_text = ", ".join(_dedupe(tools + cmds))
        failure_patterns = _dedupe(str(row.get("failure_pattern") or "").strip() for row in rows if row.get("failure_pattern"))
        sample_reasons = _dedupe(str(row.get("failure_reason") or "").strip() for row in rows if row.get("failure_reason"))[:3]
        reason_parts = [
            "This issue appeared in the same ground-truth repair context and may be needed to pass the same CI validation set."
        ]
        if failure_patterns:
            reason_parts.append(f"Failure patterns: {', '.join(failure_patterns[:8])}.")
        if sample_reasons:
            reason_parts.append(f"Representative reasons: {' | '.join(sample_reasons)}")
        issue_label = f"{error_type}::{subtype}" if error_type else subtype
        patterns.append(
            {
                "issue": issue_label,
                "reason": " ".join(reason_parts),
                "modification_hint": (
                    "Inspect these files if the current failure shares the same workflow commands "
                    f"or tools: {tool_or_cmd_text}."
                    if tool_or_cmd_text
                    else "Inspect these files if the current failure shares the same workflow validation context."
                ),
                "files": files,
            }
        )
    return patterns


def _remove_cross_issue_file_dependencies(
    modified_files: List[Dict[str, Any]],
    current_members: List[Dict[str, Any]],
    related_l1_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not modified_files or not related_l1_rows or not current_members:
        return modified_files

    current_key = (_canonical_error_type(current_members[0]), _subtype_key(current_members[0]))
    related_key_by_file = {
        _normalize_path(str(row.get("file") or "")): (_canonical_error_type(row), _subtype_key(row))
        for row in related_l1_rows
        if row.get("file")
    }

    cleaned: List[Dict[str, Any]] = []
    for item in modified_files:
        next_item = dict(item)
        kept_refs = []
        for ref in item.get("depended_files") or []:
            ref_file = _normalize_path(str(ref.get("file") or ""))
            if related_key_by_file.get(ref_file) and related_key_by_file[ref_file] != current_key:
                continue
            kept_refs.append(ref)
        next_item["depended_files"] = kept_refs
        cleaned.append(next_item)
    return cleaned


def _structured_modified_files(value: Any, fallback_members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else []
    by_file: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        path = _normalize_path(str(item.get("file") or item.get("path") or ""))
        if not path:
            continue
        by_file[path] = {
            "file": path,
            "failure_pattern": str(item.get("failure_pattern") or "").strip(),
            "failure_reason": str(item.get("failure_reason") or item.get("reason") or "").strip(),
            "modification_hint": str(item.get("modification_hint") or item.get("fix_hint") or "").strip(),
            "depended_files": _structured_file_refs(
                item.get("depended_files")
                or item.get("dependent_files")
                or []
            ),
        }

    for m in fallback_members:
        path = _normalize_path(str(m.get("file") or ""))
        if not path:
            continue
        fallback = {
            "file": path,
            "failure_pattern": str(m.get("failure_pattern") or ""),
            "failure_reason": str(m.get("failure_reason") or ""),
            "modification_hint": str(m.get("fix_strategy") or ""),
            "depended_files": _structured_file_refs(m.get("dependent_files", [])),
        }
        if path not in by_file:
            by_file[path] = fallback
        else:
            current = by_file[path]
            for key in ("failure_pattern", "failure_reason", "modification_hint"):
                if not current.get(key) and fallback.get(key):
                    current[key] = fallback[key]
            current["depended_files"] = _structured_file_refs(
                (current.get("depended_files") or []) + (fallback.get("depended_files") or [])
            )

    return list(by_file.values())


def _structured_validation_failures(
    value: Any,
    *,
    failed_cmds: List[str],
    failed_tools: List[str],
    modified_files: List[Dict[str, Any]],
    issue_subtype: str,
    failure_reason: str,
    fix_strategy: str,
) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        failed_cmd = str(item.get("failed_cmd") or item.get("command") or item.get("validation") or item.get("ci_validation") or item.get("check") or "").strip()
        failed_tool = str(item.get("failed_tool") or item.get("tool") or "").strip()
        problem = str(item.get("problem") or item.get("issue") or item.get("failure_reason") or "").strip()
        files = _structured_modified_files(item.get("files") or item.get("modified_files") or [], [])
        if failed_cmd or failed_tool or problem or files:
            out.append(
                {
                    "failed_cmd": failed_cmd,
                    "failed_tool": failed_tool,
                    "problem": problem,
                    "issue_subtype": str(item.get("issue_subtype") or issue_subtype).strip(),
                    "reason": str(item.get("reason") or failure_reason).strip(),
                    "modification_hint": str(item.get("modification_hint") or fix_strategy).strip(),
                    "modified_files": files,
                }
            )
    if out:
        return out

    commands = failed_cmds or failed_tools or ["unknown failed command"]
    return [
        {
            "failed_cmd": command if command in failed_cmds else "",
            "failed_tool": command if command in failed_tools and command not in failed_cmds else "",
            "problem": issue_subtype,
            "issue_subtype": issue_subtype,
            "reason": failure_reason,
            "modification_hint": fix_strategy,
            "modified_files": modified_files,
        }
        for command in commands
    ]


def _analysis_contexts_for_l1_group(
    l1_records: List[Dict[str, Any]],
    analysis_by_sha: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for sha_fail in _dedupe(str(row.get("sha_fail") or "") for row in l1_records):
        wrapper = analysis_by_sha.get(sha_fail, {})
        analysis = wrapper.get("analysis") if isinstance(wrapper.get("analysis"), dict) else wrapper
        if not isinstance(analysis, dict):
            continue
        contexts.append(
            {
                "sha_fail": sha_fail,
                "id": str(wrapper.get("id") or ""),
                "error_context": analysis.get("error_context", []),
                "error_types": analysis.get("error_types", []),
                "relevant_files": analysis.get("relevant_files", []),
                "failed_job": analysis.get("failed_job", []),
                "workflow_path": str(wrapper.get("workflow_path") or ""),
                "workflow": _clip(str(wrapper.get("workflow") or ""), 3000),
            }
        )
    return contexts


# ── Phase 1: L1 extraction ───────────────────────────────────────────────────

_L1_SCHEMA = """\
{
  "repo":            "repo name as given in FAILURE CONTEXT above",
  "file":            "path to the specific file involved in the failure, relative to repo root",
  "workflow_path":   "path to the CI workflow file (as given in FAILURE CONTEXT above)",
  "error_type":      "high-level category from log analysis, e.g. 'Code Formatting', 'Test Failure', 'Type Checking', 'Dependency Error'",
  "issue_subtype":   "specific actionable subtype, e.g. 'unused-import', 'unsorted-import', 'missing-optional-dependency', 'incompatible-version', 'type-stub-mismatch', 'assertion-mismatch'",
  "root_cause_category": "one of: code_style | import_sorting | unused_code | dependency_missing | dependency_version | type_contract | test_expectation | api_change | workflow_config | docs_format | other",
  "failure_pattern": "specific pattern name; keep this as specific as issue_subtype, not just the broad error_type",
  "ci_validation":   "workflow command/check likely validating this file, e.g. ruff, mypy, pytest, docs build, package install",
  "failure_reason":  "2-3 sentence explanation of why THIS FILE caused the CI failure, grounded in CI logs, workflow validation, and/or ground-truth diff",
  "fix_strategy":    "1-2 sentence description of what the ground-truth fix actually did to this file",
  "fix_pattern":     ["keyword1", "keyword2"],
  "failed_tool":     ["ruff", "pytest", "mypy", ...],
  "failed_cmd":      ["actual failed CI command(s), e.g. pre-commit run --all-files, pytest ..., mypy ..."],
  "issue_type":      "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
  "dependent_files": [
    {
      "file": "path/to/related_file.py",
      "reason": "how this file is linked to the current file/failure and what problem or modification may be needed"
    }
  ]
}"""


def _build_l1_prompt(
    repo_name:    str,
    sha_fail:     str,
    workflow_path: str,
    error_context: List[str],
    error_types:  List[Dict[str, Any]],
    relevant_files: List[Dict[str, Any]],
    failed_job:   List[Dict[str, Any]],
    file_path:    str,
    file_diff:    str,
    all_gt_files: List[str],
    workflow_text: str,
    workflow_run_commands: List[str],
    referenced_script_snippets: List[Dict[str, str]],
) -> str:
    ec_text   = "\n".join(f"  • {x}" for x in error_context) or "  (none)"
    et_text   = json.dumps(error_types,   indent=2, ensure_ascii=False)
    rf_text   = json.dumps(relevant_files, indent=2, ensure_ascii=False)
    fj_text   = json.dumps(failed_job,    indent=2, ensure_ascii=False)
    gt_text   = ", ".join(all_gt_files) or "(none)"
    diff_text = _clip(file_diff, 2000) or "(no diff available)"
    workflow_snippet = _clip(workflow_text, 3000) or "(workflow content unavailable)"
    run_commands_text = json.dumps(workflow_run_commands, indent=2, ensure_ascii=False)
    script_snippets_text = json.dumps(referenced_script_snippets, indent=2, ensure_ascii=False)

    return f"""\
You are a CI failure memory extraction agent.

Given a CI failure's structured log analysis and the ground-truth fix diff for one specific file,
extract a structured memory record that future fault-localization agents can retrieve and use.

════════════════════════════════════
FAILURE CONTEXT
════════════════════════════════════
repo          : {repo_name}
sha_fail      : {sha_fail}
workflow_path : {workflow_path}
all_gt_files  : {gt_text}

════════════════════════════════════
LOG ANALYSIS (from CILogAnalyzerLLM)
════════════════════════════════════
error_context:
{ec_text}

error_types:
{et_text}

relevant_files (log-identified):
{rf_text}

failed_job:
{fj_text}

════════════════════════════════════
CI WORKFLOW VALIDATION
════════════════════════════════════
The log may show only the first failing command. Use this workflow to infer which
validations/checks the changed file was meant to satisfy.

{workflow_snippet}

workflow_run_commands:
{run_commands_text}

referenced_script_snippets:
{script_snippets_text}

════════════════════════════════════
GROUND TRUTH FIX — file: {file_path}
════════════════════════════════════
{diff_text}

════════════════════════════════════
TASK
════════════════════════════════════
Extract a structured L1 memory record for file "{file_path}".
Use ONLY evidence from the log analysis, workflow validation, and diff above.
Do NOT speculate beyond what is shown.
The CI log can be incomplete and may only show the first failure. If the explicit log does not
mention this file, infer the likely issue from the workflow validation commands and the ground-truth
diff. For example:
- import sorting/unused imports/formatting changes usually map to ruff/pre-commit/style checks.
- typing/import contract changes usually map to mypy/pyright/type-check jobs.
- pyproject/dependency/version changes may map to install, import, test, or type-check failures.
- docs/rst/markdown changes may map to the exact docs validation command shown in workflow_run_commands
  or referenced_script_snippets, such as mdformat, docstrfmt, sphinx-build, rstcheck, markdownlint, or docs build.
Separate broad error_type from specific issue_subtype. For example, Code Formatting can be
unused-import, unsorted-import, trailing-whitespace, line-too-long, or docs-heading-format.
Analyze dependency causality: a dependency/version problem may surface as type checking,
test failure, or import failure. Capture that real subtype and root_cause_category.
If a dependency or version change causes additional failures, explain the cascade: which dependency
changed, which validation command exposed it, which files were affected, and what fix was applied.
Use the most specific failed_cmd from referenced_script_snippets when the workflow calls a helper script.
For example, if the workflow runs ./framework/dev/test.sh and that script runs mdformat/docstrfmt/mypy,
use the concrete inner command that validates this file in failed_cmd, while also keeping the outer script
command if useful.
For dependent_files, include only files with a concrete relationship to this file/failure,
and explain whether the related file likely also needs inspection or modification.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L1_SCHEMA}
""".strip()


def extract_l1_records(
    seed_rows:     List[Dict[str, Any]],
    analysis_rows: List[Dict[str, Any]],
    llm,
    output_path: Path | None = None,
    repo_base_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    analysis_by_sha = {str(r.get("sha_fail") or ""): r for r in analysis_rows}
    out: List[Dict[str, Any]] = _load_json_list(output_path) if output_path else []
    existing_keys = {_l1_record_key(row) for row in out}

    for seed in seed_rows:
        sha_fail      = str(seed.get("sha_fail") or "")
        task_id       = str(seed.get("id") or "")
        repo_name     = str(seed.get("repo_name") or "")
        workflow_path = str(seed.get("workflow_path") or "")
        workflow_text = str(seed.get("workflow") or "")
        workflow_run_commands = _extract_workflow_run_commands(workflow_text)
        repo_root = (repo_base_dir / repo_name) if repo_base_dir and repo_name else None
        referenced_script_snippets = _resolve_referenced_script_snippets(
            repo_root=repo_root,
            run_commands=workflow_run_commands,
        )
        diff_text     = str(seed.get("diff") or "")
        gt_files      = _extract_files_from_diff(diff_text)
        if not gt_files:
            gt_files = [_normalize_path(x) for x in (seed.get("changed_files") or [])]

        wrapper  = analysis_by_sha.get(sha_fail, {})
        analysis = wrapper.get("analysis") if isinstance(wrapper.get("analysis"), dict) else wrapper

        error_context  = [str(x) for x in (analysis.get("error_context") or []) if str(x).strip()]
        error_types    = analysis.get("error_types")   or []
        relevant_files = analysis.get("relevant_files") or []
        failed_job     = analysis.get("failed_job")    or []

        if not gt_files:
            print(f"[L1] {sha_fail}: no files found in diff, skipping.")
            continue

        for file_path in gt_files:
            key = (sha_fail, _normalize_path(file_path))
            if key in existing_keys:
                print(f"[L1] skip existing {sha_fail}/{file_path}")
                continue

            file_diff = _extract_file_diff(diff_text, file_path)
            prompt    = _build_l1_prompt(
                repo_name=repo_name, sha_fail=sha_fail, workflow_path=workflow_path,
                error_context=error_context, error_types=error_types,
                relevant_files=relevant_files, failed_job=failed_job,
                file_path=file_path, file_diff=file_diff, all_gt_files=gt_files,
                workflow_text=workflow_text,
                workflow_run_commands=workflow_run_commands,
                referenced_script_snippets=referenced_script_snippets,
            )

            try:
                raw    = llm.invoke(prompt).content
                record = _parse_llm_json(raw)
            except Exception as exc:
                print(f"[L1] LLM failed for {sha_fail}/{file_path}: {exc}")
                record = {}

            # Hardcoded structural fields are ground truth; use LLM values as fallback
            # when the seed data is missing a field (e.g. workflow_path can be empty).
            resolved_repo          = repo_name     or str(record.get("repo")          or "")
            resolved_file          = file_path     or str(record.get("file")          or "")
            resolved_workflow_path = workflow_path or str(record.get("workflow_path") or "")

            failure_reason = _clip(str(record.get("failure_reason") or ""), 500)
            raw_error_type = str(record.get("error_type") or "")
            raw_issue_subtype = str(record.get("issue_subtype") or record.get("failure_pattern") or "")
            raw_root_cause_category = str(record.get("root_cause_category") or "")
            ci_validation = str(record.get("ci_validation") or "")
            failed_cmds = _dedupe(
                _as_str_list(record.get("failed_cmd"))
                + ([ci_validation] if ci_validation else [])
            )
            failed_tools = _dedupe(
                _as_str_list(record.get("failed_tool"))
            )
            memory_record = {
                "issue_id":        task_id,
                "sha_fail":        sha_fail,
                "repo":            resolved_repo,
                "repo_name":       resolved_repo,
                "workflow_path":   resolved_workflow_path,
                "workflow_name":   str(seed.get("workflow_name") or ""),
                "file":            resolved_file,
                "line_number":     None,
                "error_type":      raw_error_type,
                "issue_subtype":   raw_issue_subtype,
                "root_cause_category": raw_root_cause_category,
                "failure_pattern": str(record.get("failure_pattern") or ""),
                "ci_validation":   ci_validation,
                "failure_reason":  failure_reason,
                "reason":          failure_reason,
                "fix_strategy":    _clip(str(record.get("fix_strategy") or ""), 500),
                "fix_pattern":     [str(x) for x in (record.get("fix_pattern") or [])],
                "failed_tool":     failed_tools,
                "failed_cmd":      failed_cmds,
                "issue_type":      str(record.get("issue_type") or ""),
                "dependent_files": _structured_file_refs(record.get("dependent_files", [])),
            }
            out.append(memory_record)
            existing_keys.add(_l1_record_key(memory_record))
            if output_path:
                _write_json(output_path, out)
            print(f"[L1] {sha_fail}/{resolved_file} → error_type={record.get('error_type', '?')}")

    return out


# ── Phase 2: L2 synthesis ────────────────────────────────────────────────────

_L2_SCHEMA = """\
{
  "repo": "repo name",
  "repo_name": "repo name",
  "workflow_path": "path to the CI workflow file if common/known",
  "issue": {
    "error_type": "broad category, e.g. Dependency Error",
    "issue_subtype": "specific actionable subtype, e.g. missing-optional-dependency",
    "root_cause_category": "dominant root-cause category",
    "failure_pattern": "specific repo pattern",
    "failure_reason": "repo-specific recurring root cause",
    "fix_strategy": "repo-specific common fix approach",
    "fix_pattern": ["keyword1", "keyword2"],
    "failed_tool": ["pytest", "mypy"],
    "failed_cmd": ["actual failed CI commands and workflow validation commands this issue must satisfy"],
    "ci_validations": ["workflow checks/commands that validate this repo issue pattern"]
  },
  "modified_files": [
    {
      "file": "path/to/file",
      "failure_pattern": "what pattern this file triggers",
      "failure_reason": "why this file is involved in this issue",
      "modification_hint": "what usually needs to be changed in this file",
      "depended_files": [
        {
          "file": "path/to/dependent_file.py",
          "reason": "how this dependent file is linked and whether it may need inspection/modification"
        }
      ]
    }
  ],
  "validation_failures": [
    {
      "failed_cmd": "actual failed CI command from log/workflow, e.g. pre-commit run --all-files, pytest ..., mypy ...",
      "failed_tool": "tool invoked by the failed command, e.g. ruff, mypy, pytest, pip",
      "problem": "specific problem this failed command exposed for this repo issue",
      "issue_subtype": "specific subtype under this failed command",
      "reason": "why this command fails for the issue",
      "modification_hint": "what kind of change is needed to pass this failed command",
      "modified_files": [
        {
          "file": "path/to/file",
          "failure_pattern": "file-specific failure pattern",
          "failure_reason": "why this file participates in this validation failure",
          "modification_hint": "what likely needs modification in this file",
          "depended_files": [
            {
              "file": "path/to/related_file.py",
              "reason": "why this related file is connected to the validation failure"
            }
          ]
        }
      ]
    }
  ],
  "dependent_issue_patterns": [
    {
      "issue": "secondary issue that can appear while fixing this repo pattern, e.g. type-checking after dependency version change",
      "reason": "why this secondary issue is linked to the primary issue in this repo",
      "modification_hint": "what may need inspection or modification if this secondary issue appears",
      "files": ["files commonly involved, if known"]
    }
  ]
}"""


def _build_l2_prompt(
    repo_name: str,
    error_type: str,
    subtype: str,
    l1_records: List[Dict[str, Any]],
    ci_contexts: List[Dict[str, Any]],
) -> str:
    l1_text = json.dumps(
        [{k: v for k, v in r.items() if k not in ("sha_fail", "issue_id", "workflow_name")}
         for r in l1_records],
        indent=2, ensure_ascii=False,
    )
    ci_text = json.dumps(ci_contexts, indent=2, ensure_ascii=False)
    return f"""\
You are a CI failure pattern analyst.

Given multiple L1 failure records from the SAME REPO with the same broad error type and issue subtype,
synthesize a REPO-LEVEL PATTERN memory entry that captures the recurring failure pattern.

════════════════════════════════════
CONTEXT
════════════════════════════════════
repo       : {repo_name}
error_type : {error_type}
subtype    : {subtype or "(none)"}

════════════════════════════════════
L1 FAILURE RECORDS
════════════════════════════════════
{_clip(l1_text, 4000)}

════════════════════════════════════
CI LOG ANALYSIS CONTEXT FOR THESE RECORDS
════════════════════════════════════
{_clip(ci_text, 4000)}

════════════════════════════════════
TASK
════════════════════════════════════
Synthesize a single L2 repo-level memory record from the L1 records above.
Organize the result around one issue:
1. issue: what exact repo issue pattern this is and how it is usually fixed.
2. modified_files: files involved in this issue that failed or needed modification.
3. validation_failures: for each actual failed CI command/check from the log/workflow, the problem it exposes and the files involved to fix/pass it.
4. modified_files[].depended_files: directly related files that may also need inspection/modification.
5. dependent_issue_patterns: secondary issue types from the same repair context that can arise while fixing this issue.
Identify the common specific issue subtype, root cause, dependency/file relationship, and fix approach.
This must be a dynamic repo-pattern analysis, not a fixed mapping:
- Use CI log evidence, workflow commands, helper-script commands, and the ground-truth file changes.
- If the log only shows the first failure, infer other validated problems from the workflow/script commands and diff.
- For dependency/version issues, explain what dependency changed, which validations broke because of it,
  what secondary issues appeared (type checking, import errors, formatting, tests, docs, workflow config),
  and which files were modified for each problem.
- Under each issue/problem, make clear which files had the issue, which files were modified, and why.
Do not merge distinct subtypes just because they share a broad error_type.
If a dependency/version issue causes type-checking, import, or test failures, preserve the dependency/version subtype.
Also identify dependent_issue_patterns: secondary issue types that appeared in the same repair context
or previously appeared because of fixing this issue in this repo. For example, dependency/version fixes
can trigger type-checking failures, import/module failures, formatting updates, test expectation changes,
or pyproject/workflow changes. Python formatting and RST/docs formatting can also appear together under
the same CI validation set. These are not necessarily direct file dependencies; they are repo-level
follow-up or sibling issue patterns.
Use modified_files for the files that were modified or likely need modification for this repo pattern.
For each modified file, include depended_files only when there is a real direct file-to-file dependency
or related-file reason. Do not put broad sibling issue files there. For example, changed .rst files from
the same Flower repair should be represented as a docs/RST dependent_issue_pattern for a Python docstring
issue, and as modified_files in their own docs/RST issue record, not as depended_files of the Python file.
Only include modified_files that appear in the L1 records.
Use the CI log context to distinguish distinct issue subtypes and causality. If the broad error type is
the same but the CI evidence shows different subtypes, preserve the specific subtype in issue.issue_subtype
and issue.failure_pattern. Capture dependency/version cascades, type-checking side effects, test failures,
formatting fallout, pyproject/workflow changes, and file relationships when supported by evidence.
Use the workflow validation commands together with the diff. If the CI log only exposes the first failure,
still identify other issue patterns that are supported by the workflow validations and ground-truth changes.
Set issue.ci_validations to the checks/commands that likely validate this repo pattern.
For validation_failures, organize by actual failed CI command/check, using failed_cmd and failed_tool.
Under each failed_cmd, list the specific problem and the modified_files needed to pass that command. This is important when one CI run
contains dependency, type-checking, formatting, and test-validation effects in the same fix.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L2_SCHEMA}
""".strip()


def build_l2_records(
    l1_rows: List[Dict[str, Any]],
    llm,
    output_path: Path | None = None,
    analysis_by_sha: Dict[str, Dict[str, Any]] | None = None,
    related_l1_rows: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in l1_rows:
        key = (str(row.get("repo") or ""), _canonical_error_type(row), _subtype_key(row))
        grouped[key].append(row)

    out: List[Dict[str, Any]] = _load_json_list(output_path) if output_path else []
    by_key = {_l2_record_key(row): row for row in out}
    for (repo, error_type, subtype), members in grouped.items():
        ci_contexts = _analysis_contexts_for_l1_group(members, analysis_by_sha or {})
        prompt = _build_l2_prompt(repo, error_type, subtype, members, ci_contexts)
        try:
            raw    = llm.invoke(prompt).content
            record = _parse_llm_json(raw)
        except Exception as exc:
            print(f"[L2] LLM failed for ({repo}, {error_type}, {subtype}): {exc}")
            record = {}

        issue_obj = record.get("issue") if isinstance(record.get("issue"), dict) else {}

        # Aggregate structural fields from L1 members
        # Tools/cmds: aggregate from L1 first, then supplement with anything the LLM inferred
        all_tools     = _dedupe(
            [t for m in members for t in (m.get("failed_tool") or [])]
            + _as_str_list(record.get("failed_tool"))
            + _as_str_list(issue_obj.get("failed_tool") if isinstance(issue_obj, dict) else None)
        )
        all_cmds      = _dedupe(
            [c for m in members for c in (m.get("failed_cmd") or [])]
            + _as_str_list(record.get("failed_cmd"))
            + _as_str_list(issue_obj.get("failed_cmd") if isinstance(issue_obj, dict) else None)
        )
        ci_validations = _dedupe(
            [str(m.get("ci_validation") or "") for m in members if m.get("ci_validation")]
            + [str(x) for x in ((issue_obj.get("ci_validations") if isinstance(issue_obj, dict) else None) or record.get("ci_validations") or [])]
        )
        all_cmds = _dedupe(all_cmds + ci_validations)
        issue_ids     = _dedupe(str(m.get("issue_id") or "") for m in members)
        changed_files = _dedupe(_normalize_path(m.get("file", "")) for m in members if m.get("file"))

        # workflow_path: collect from L1 members; supplement with LLM-returned value
        workflow_paths = _dedupe(
            [str(m.get("workflow_path") or "") for m in members if m.get("workflow_path")]
            + ([str(record.get("workflow_path"))] if record.get("workflow_path") else [])
        )

        # LLM-returned error_type and repo are treated as confirmation; grouping key is authoritative
        resolved_repo       = repo       or str(record.get("repo")       or "")
        resolved_error_type = error_type or str(record.get("error_type") or "")
        resolved_error_type = resolved_error_type or str(issue_obj.get("error_type") or "")
        resolved_issue_subtype = subtype or str(record.get("issue_subtype") or issue_obj.get("issue_subtype") or "")
        root_cause_category = str(record.get("root_cause_category") or issue_obj.get("root_cause_category") or "")

        modified_files = _structured_modified_files(
            record.get("modified_files") or record.get("files") or [],
            members,
        )
        modified_files = _remove_cross_issue_file_dependencies(
            modified_files,
            members,
            related_l1_rows or [],
        )

        resolved_failure_pattern = str(record.get("failure_pattern") or issue_obj.get("failure_pattern") or resolved_issue_subtype)
        pattern_name = f"{resolved_error_type} {resolved_failure_pattern}".strip() or resolved_error_type
        failure_reason = _clip(str(record.get("failure_reason") or issue_obj.get("failure_reason") or ""), 500)
        fix_strategy = _clip(str(record.get("fix_strategy") or issue_obj.get("fix_strategy") or ""), 500)
        fix_pattern = [str(x) for x in (record.get("fix_pattern") or issue_obj.get("fix_pattern") or [])]

        dependent_issue_patterns = _dedupe_issue_patterns(
            _structured_issue_patterns(record.get("dependent_issue_patterns", []))
            + _derive_dependent_issue_patterns(members, related_l1_rows or [])
        )

        memory_record = {
            "repo":            resolved_repo,
            "repo_name":       resolved_repo,
            "issue": {
                "error_type": resolved_error_type,
                "issue_subtype": resolved_issue_subtype,
                "root_cause_category": root_cause_category,
                "failure_pattern": resolved_failure_pattern,
                "failure_reason": failure_reason,
                "fix_strategy": fix_strategy,
                "fix_pattern": fix_pattern,
                "failed_tool": all_tools,
                "failed_cmd": all_cmds,
                "ci_validations": ci_validations,
            },
            "error_type":      resolved_error_type,
            "issue_subtype":   resolved_issue_subtype,
            "root_cause_category": root_cause_category,
            "workflow_path":   workflow_paths[0] if workflow_paths else "",
            "failure_pattern": resolved_failure_pattern,
            "failure_reason":  failure_reason,
            "fix_strategy":    fix_strategy,
            "fix_pattern":     fix_pattern,
            "dependent_issue_patterns": dependent_issue_patterns,
            "failed_tool":     all_tools,
            "failed_cmd":      all_cmds,
            "ci_validations":  ci_validations,
            "modified_files":  modified_files,
            "validation_failures": _structured_validation_failures(
                record.get("validation_failures", []),
                failed_cmds=all_cmds,
                failed_tools=all_tools,
                modified_files=modified_files,
                issue_subtype=resolved_issue_subtype,
                failure_reason=failure_reason,
                fix_strategy=fix_strategy,
            ),
            "changed_files":   changed_files,
            "pattern_name":    pattern_name,
            "issue_ids":       issue_ids,
            "seed_count":      len(members),
        }
        by_key[_l2_record_key(memory_record)] = memory_record
        out = list(by_key.values())
        if output_path:
            _write_json(output_path, out)
        print(f"[L2] ({resolved_repo}, {resolved_error_type}, {subtype}) → pattern={resolved_failure_pattern or '?'}")

    return out


# ── Phase 3: L3 synthesis ────────────────────────────────────────────────────

_L3_SCHEMA = """\
{
  "error_type":      "the error type shared by all L2 records in this group",
  "issue_subtype":   "specific actionable subtype shared by this group",
  "root_cause_category": "dominant root-cause category for the subtype",
  "failed_tool":     ["all distinct failed tools observed across the L2 records"],
  "principle":       "2-3 sentence generalizable insight about this class of CI failure, applicable across repos",
  "fix_strategy":    "universal fix direction that applies across repos",
  "secondary_issue_patterns": [
    {
      "issue": "secondary issue type often linked to this primary issue subtype",
      "reason": "why it tends to appear after or alongside the primary fix",
      "modification_hint": "what to inspect or modify when this secondary issue appears",
      "repos": ["repos where this pattern appeared"]
    }
  ],
  "failure_patterns":["all distinct pattern names observed across the L2 records"],
  "failure_reasons": ["2-3 representative and diverse root-cause descriptions from different repos"]
}"""


def _build_l3_prompt(error_type: str, subtype: str, l2_records: List[Dict[str, Any]]) -> str:
    l2_text = json.dumps(
        [{k: v for k, v in r.items() if k not in ("changed_files", "failed_cmd", "issue_ids", "seed_count")}
         for r in l2_records],
        indent=2, ensure_ascii=False,
    )
    repos = _dedupe(str(r.get("repo") or "") for r in l2_records)
    return f"""\
You are a CI failure principle analyst.

Given L2 repo-level patterns with the same broad error type and issue subtype from one or more repos,
extract a CROSS-REPO GENERALIZABLE PRINCIPLE for this class of CI failure.

════════════════════════════════════
CONTEXT
════════════════════════════════════
error_type : {error_type}
subtype    : {subtype or "(none)"}
repos      : {', '.join(repos)}

════════════════════════════════════
L2 REPO PATTERNS (from different repos)
════════════════════════════════════
{_clip(l2_text, 4000)}

════════════════════════════════════
TASK
════════════════════════════════════
Synthesize a single L3 cross-repo principle from the L2 patterns above.
Focus on what is UNIVERSAL for this specific issue_subtype, not just the broad error_type.
The principle and fix_strategy must be actionable for any repo encountering this error type.
Do not merge unrelated subtypes such as unused imports, unsorted imports, version conflicts,
missing optional dependencies, assertion expectation changes, or type contract mismatches.
Preserve secondary_issue_patterns from L2 when they are recurring or generally useful.
Example: dependency/version fixes can cascade into type checking, import errors, formatting,
test expectation updates, pyproject changes, or workflow install changes.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L3_SCHEMA}
""".strip()


def build_l3_records(l2_rows: List[Dict[str, Any]], llm, output_path: Path | None = None) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        grouped[(_canonical_error_type(row), _subtype_key(row))].append(row)

    out: List[Dict[str, Any]] = _load_json_list(output_path) if output_path else []
    by_key = {_l3_record_key(row): row for row in out}
    for (error_type, subtype), members in grouped.items():
        prompt = _build_l3_prompt(error_type, subtype, members)
        try:
            raw    = llm.invoke(prompt).content
            record = _parse_llm_json(raw)
        except Exception as exc:
            print(f"[L3] LLM failed for error_type={error_type}, subtype={subtype}: {exc}")
            record = {}

        repos      = _dedupe(str(m.get("repo") or "") for m in members)
        # Aggregate tools from L2 members; supplement with anything the LLM additionally inferred
        all_tools  = _dedupe(
            [t for m in members for t in (m.get("failed_tool") or [])]
            + [str(x) for x in (record.get("failed_tool") or [])]
        )
        issue_ids  = _dedupe(iid for m in members for iid in (m.get("issue_ids") or []))
        issue_type = subtype or (str(members[0].get("failure_pattern") or "") if members else "")

        # LLM-returned error_type is treated as confirmation; grouping key is authoritative
        resolved_error_type = error_type or str(record.get("error_type") or "")
        resolved_issue_subtype = issue_type or str(record.get("issue_subtype") or "")
        root_cause_category = str(record.get("root_cause_category") or "")

        # Build a semantic principle string from LLM output
        llm_principle = str(record.get("principle") or "").strip()
        llm_reason1   = (record.get("failure_reasons") or [""])[0] if record else ""
        principle = (
            f"error_type={resolved_error_type}: {_clip(llm_principle or llm_reason1, 400)}"
            f" (seen in repos: {', '.join(repos)})"
        ) if (llm_principle or llm_reason1) else (
            f"error_type={resolved_error_type} (seen in repos: {', '.join(repos)})"
        )

        fix_strategy = _clip(str(record.get("fix_strategy") or ""), 500)
        memory_record = {
            "error_type":         resolved_error_type,
            "issue_subtype":      resolved_issue_subtype,
            "root_cause_category": root_cause_category,
            "issue_type":         resolved_issue_subtype,
            "failure_pattern":    resolved_issue_subtype,
            "repos":              repos,
            "repo_names":         repos,
            "failed_tool":        all_tools,
            "principle":          _clip(principle, 600),
            "fix_strategy":       fix_strategy,
            "fix_strategies":     [fix_strategy] if fix_strategy else [],
            "secondary_issue_patterns": _structured_issue_patterns(record.get("secondary_issue_patterns", [])),
            "failure_patterns":   [str(x) for x in (record.get("failure_patterns") or [])],
            "failure_reasons":    [_clip(str(x), 300) for x in (record.get("failure_reasons") or [])[:3]],
            "evidence_issue_ids": issue_ids,
        }
        by_key[_l3_record_key(memory_record)] = memory_record
        out = list(by_key.values())
        if output_path:
            _write_json(output_path, out)
        print(f"[L3] error_type={resolved_error_type} subtype={issue_type} → repos={repos}")

    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-based L1/L2/L3 memory bank extraction from seed log analysis."
    )
    parser.add_argument("--seed-file",     type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    parser.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config",        type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-key",     default=get_default_model_key())
    args = parser.parse_args()

    seed_rows     = json.loads(args.seed_file.read_text(encoding="utf-8"))
    analysis_rows = json.loads(args.analysis_file.read_text(encoding="utf-8"))
    analysis_by_sha = {
        str(row.get("sha_fail") or ""): row
        for row in analysis_rows
        if isinstance(row, dict) and str(row.get("sha_fail") or "")
    }

    model_key = args.model_key
    llm       = get_llm(model_key)
    config = OmegaConf.load(args.config) if args.config.exists() else {}
    repo_base_dir = Path(str(config.get("baseline_repo_folder", ""))) if config and config.get("baseline_repo_folder") else None
    l1_path = args.output_dir / "failure_memory.json"
    l2_path = args.output_dir / "repo_memory.json"
    l3_path = args.output_dir / "cross_memory.json"

    print(f"\n=== Per-issue L1/L2/L3 extraction ({len(seed_rows)} seed issues) ===")
    for idx, seed in enumerate(seed_rows, start=1):
        sha_fail = str(seed.get("sha_fail") or "")
        task_id = str(seed.get("id") or "")
        print(f"\n=== Issue {idx}/{len(seed_rows)}: id={task_id} sha={sha_fail} ===")

        print("[Issue] L1 extraction")
        l1_rows = extract_l1_records([seed], analysis_rows, llm, output_path=l1_path, repo_base_dir=repo_base_dir)
        issue_l1_rows = [row for row in l1_rows if str(row.get("sha_fail") or "") == sha_fail]
        if not issue_l1_rows:
            print(f"[Issue] no L1 records for {sha_fail}; skipping L2/L3 update.")
            continue

        affected_l2_keys: List[Tuple[str, str, str]] = []
        seen_l2_keys = set()
        for row in issue_l1_rows:
            repo = str(row.get("repo") or row.get("repo_name") or "").strip()
            if not repo:
                continue
            key = (repo, _canonical_error_type(row), _subtype_key(row))
            if key not in seen_l2_keys:
                seen_l2_keys.add(key)
                affected_l2_keys.append(key)

        affected_l3_keys: List[Tuple[str, str]] = []
        seen_l3_keys = set()
        for repo, error_type, subtype in affected_l2_keys:
            all_l1_rows = _load_json_list(l1_path)
            group_l1_rows = [
                row for row in all_l1_rows
                if str(row.get("repo") or row.get("repo_name") or "") == repo
                and _canonical_error_type(row) == error_type
                and _subtype_key(row) == subtype
            ]
            if not group_l1_rows:
                continue

            print(f"[Issue] L2 update repo={repo} error_type={error_type} subtype={subtype} records={len(group_l1_rows)}")
            build_l2_records(
                group_l1_rows,
                llm,
                output_path=l2_path,
                analysis_by_sha=analysis_by_sha,
                related_l1_rows=issue_l1_rows,
            )
            if error_type:
                key = (error_type, subtype)
                if key not in seen_l3_keys:
                    seen_l3_keys.add(key)
                    affected_l3_keys.append(key)

        for error_type, subtype in affected_l3_keys:
            all_l2_rows = _load_json_list(l2_path)
            group_l2_rows = [
                row for row in all_l2_rows
                if _canonical_error_type(row) == error_type
                and _subtype_key(row) == subtype
            ]
            if not group_l2_rows:
                continue

            print(f"[Issue] L3 update error_type={error_type} subtype={subtype} records={len(group_l2_rows)}")
            build_l3_records(group_l2_rows, llm, output_path=l3_path)

    l1_rows = _load_json_list(l1_path)
    l2_rows = _load_json_list(l2_path)
    l3_rows = _load_json_list(l3_path)
    print(f"\nL1: {len(l1_rows)} records written.")
    print(f"L2: {len(l2_rows)} records written.")
    print(f"L3: {len(l3_rows)} records written.")

    summary = {
        "model_key":    model_key,
        "seed_file":    str(args.seed_file),
        "analysis_file":str(args.analysis_file),
        "output_dir":   str(args.output_dir),
        "memory_counts": {
            "L1_failure_memory": len(l1_rows),
            "L2_repo_memory":    len(l2_rows),
            "L3_cross_memory":   len(l3_rows),
        },
        "schema": {
            "L1": ["file", "error_type", "issue_subtype", "root_cause_category", "failure_pattern", "failure_reason", "fix_strategy",
                   "fix_pattern", "failed_tool", "issue_type", "dependent_files", "repo"],
            "L2": ["error_type", "issue_subtype", "root_cause_category", "failure_pattern", "failure_reason", "fix_strategy",
                   "dependent_issue_patterns",
                   "validation_failures[failed_cmd,failed_tool,problem,modified_files]",
                   "fix_pattern", "modified_files[file,failure_pattern,failure_reason,modification_hint,depended_files]", "failed_tool", "repo"],
            "L3": ["error_type", "issue_subtype", "root_cause_category", "failure_pattern", "principle", "fix_strategy",
                   "secondary_issue_patterns", "failure_patterns",
                   "failure_reasons", "failed_tool", "repos"],
        },
    }
    _write_json(args.output_dir / "memory_bank_summary.json", summary)
    print(f"\n=== Done ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
