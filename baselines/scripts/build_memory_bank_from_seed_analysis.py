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
  2. Group L1 by (repo, error_type) → LLM prompt → L2 patterns
  3. Group L2 by error_type across repos → LLM prompt → L3 principles
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


# ── Phase 1: L1 extraction ───────────────────────────────────────────────────

_L1_SCHEMA = """\
{
  "repo":            "repo name as given in FAILURE CONTEXT above",
  "file":            "path to the specific file involved in the failure, relative to repo root",
  "workflow_path":   "path to the CI workflow file (as given in FAILURE CONTEXT above)",
  "error_type":      "high-level category from log analysis, e.g. 'Code Formatting', 'Test Failure', 'Type Checking', 'Dependency Error'",
  "failure_pattern": "specific pattern name, e.g. 'unused-import', 'line-too-long', 'assertion-error', 'missing-package'",
  "failure_reason":  "2-3 sentence explanation of why THIS FILE caused the CI failure, grounded in log evidence",
  "fix_strategy":    "1-2 sentence description of what the ground-truth fix actually did to this file",
  "fix_pattern":     ["keyword1", "keyword2"],
  "failed_tool":     ["ruff", "pytest", "mypy", ...],
  "issue_type":      "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
  "dependent_files": ["other files involved in the same failure, if determinable from logs/diff"]
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
) -> str:
    ec_text   = "\n".join(f"  • {x}" for x in error_context) or "  (none)"
    et_text   = json.dumps(error_types,   indent=2, ensure_ascii=False)
    rf_text   = json.dumps(relevant_files, indent=2, ensure_ascii=False)
    fj_text   = json.dumps(failed_job,    indent=2, ensure_ascii=False)
    gt_text   = ", ".join(all_gt_files) or "(none)"
    diff_text = _clip(file_diff, 2000) or "(no diff available)"

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
GROUND TRUTH FIX — file: {file_path}
════════════════════════════════════
{diff_text}

════════════════════════════════════
TASK
════════════════════════════════════
Extract a structured L1 memory record for file "{file_path}".
Use ONLY evidence from the log analysis and the diff above.
Do NOT speculate beyond what is shown.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L1_SCHEMA}
""".strip()


def extract_l1_records(
    seed_rows:     List[Dict[str, Any]],
    analysis_rows: List[Dict[str, Any]],
    llm,
) -> List[Dict[str, Any]]:
    analysis_by_sha = {str(r.get("sha_fail") or ""): r for r in analysis_rows}
    out: List[Dict[str, Any]] = []

    for seed in seed_rows:
        sha_fail      = str(seed.get("sha_fail") or "")
        task_id       = str(seed.get("id") or "")
        repo_name     = str(seed.get("repo_name") or "")
        workflow_path = str(seed.get("workflow_path") or "")
        diff_text     = str(seed.get("diff") or "")
        gt_files      = [_normalize_path(x) for x in (seed.get("ground_truth_files") or [])]

        wrapper  = analysis_by_sha.get(sha_fail, {})
        analysis = wrapper.get("analysis") or {}

        error_context  = [str(x) for x in (analysis.get("error_context") or []) if str(x).strip()]
        error_types    = analysis.get("error_types")   or []
        relevant_files = analysis.get("relevant_files") or []
        failed_job     = analysis.get("failed_job")    or []

        if not gt_files:
            print(f"[L1] {sha_fail}: no ground-truth files, skipping.")
            continue

        for file_path in gt_files:
            file_diff = _extract_file_diff(diff_text, file_path)
            prompt    = _build_l1_prompt(
                repo_name=repo_name, sha_fail=sha_fail, workflow_path=workflow_path,
                error_context=error_context, error_types=error_types,
                relevant_files=relevant_files, failed_job=failed_job,
                file_path=file_path, file_diff=file_diff, all_gt_files=gt_files,
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
            out.append({
                "issue_id":        task_id,
                "sha_fail":        sha_fail,
                "repo":            resolved_repo,
                "repo_name":       resolved_repo,
                "workflow_path":   resolved_workflow_path,
                "workflow_name":   str(seed.get("workflow_name") or ""),
                "file":            resolved_file,
                "line_number":     None,
                "error_type":      str(record.get("error_type")      or ""),
                "failure_pattern": str(record.get("failure_pattern") or ""),
                "failure_reason":  failure_reason,
                "reason":          failure_reason,
                "fix_strategy":    _clip(str(record.get("fix_strategy") or ""), 500),
                "fix_pattern":     [str(x) for x in (record.get("fix_pattern") or [])],
                "failed_tool":     _dedupe([str(x) for x in (record.get("failed_tool") or [])]),
                "issue_type":      str(record.get("issue_type") or ""),
                "dependent_files": _dedupe([_normalize_path(str(x)) for x in (record.get("dependent_files") or [])]),
            })
            print(f"[L1] {sha_fail}/{resolved_file} → error_type={record.get('error_type', '?')}")

    return out


# ── Phase 2: L2 synthesis ────────────────────────────────────────────────────

_L2_SCHEMA = """\
{
  "error_type":      "the error type shared by all L1 records in this group",
  "failed_tool":     ["all distinct failed tools observed across the L1 records"],
    "repo":            "repo full name, e.g. 'owner/repo'",
    "workflow_path":   "path to the workflow file where the failure occurred, relative to .github/workflows",
  "failed_cmd":      ["all distinct failed commands observed for failures of this type and reason"],
  "failure_pattern": "synthesized pattern name capturing the common theme across all L1 records",
  "failure_reason":  "2-3 sentence synthesis of the recurring root cause seen across these cases",
  "fix_strategy":    "1-2 sentence description of the common fix approach",
  "fix_pattern":     ["keyword1", "keyword2"],
  "files": [
    {
      "file":            "path/to/file",
      "failure_pattern": "what pattern this file triggers",
      "failure_reason":  "why this file type is typically involved"
    }
  ]
}"""


def _build_l2_prompt(repo_name: str, error_type: str, l1_records: List[Dict[str, Any]]) -> str:
    l1_text = json.dumps(
        [{k: v for k, v in r.items() if k not in ("sha_fail", "issue_id", "workflow_name")}
         for r in l1_records],
        indent=2, ensure_ascii=False,
    )
    return f"""\
You are a CI failure pattern analyst.

Given multiple L1 failure records from the SAME REPO with related error types,
synthesize a REPO-LEVEL PATTERN memory entry that captures the recurring failure pattern.

════════════════════════════════════
CONTEXT
════════════════════════════════════
repo       : {repo_name}
error_type : {error_type}

════════════════════════════════════
L1 FAILURE RECORDS
════════════════════════════════════
{_clip(l1_text, 4000)}

════════════════════════════════════
TASK
════════════════════════════════════
Synthesize a single L2 repo-level memory record from the L1 records above.
Identify the common failure pattern, root cause, and fix approach.
Only include files that appear in the L1 records.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L2_SCHEMA}
""".strip()


def build_l2_records(l1_rows: List[Dict[str, Any]], llm) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in l1_rows:
        key = (str(row.get("repo") or ""), str(row.get("error_type") or ""))
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (repo, error_type), members in grouped.items():
        prompt = _build_l2_prompt(repo, error_type, members)
        try:
            raw    = llm.invoke(prompt).content
            record = _parse_llm_json(raw)
        except Exception as exc:
            print(f"[L2] LLM failed for ({repo}, {error_type}): {exc}")
            record = {}

        # Aggregate structural fields from L1 members
        # Tools/cmds: aggregate from L1 first, then supplement with anything the LLM inferred
        all_tools     = _dedupe(
            [t for m in members for t in (m.get("failed_tool") or [])]
            + [str(x) for x in (record.get("failed_tool") or [])]
        )
        all_cmds      = _dedupe(
            [c for m in members for c in (m.get("failed_cmd") or [])]
            + [str(x) for x in (record.get("failed_cmd") or [])]
        )
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

        # Use LLM-synthesized files if provided, else fall back to L1 files
        llm_files = record.get("files") or []
        if not llm_files:
            llm_files = [{"file": m["file"], "failure_pattern": m.get("failure_pattern", ""),
                          "failure_reason": m.get("failure_reason", "")} for m in members if m.get("file")]

        pattern_name = f"{resolved_error_type} {record.get('failure_pattern', '')}".strip() or resolved_error_type

        out.append({
            "repo":            resolved_repo,
            "repo_name":       resolved_repo,
            "error_type":      resolved_error_type,
            "workflow_path":   workflow_paths[0] if workflow_paths else "",
            "failure_pattern": str(record.get("failure_pattern") or ""),
            "failure_reason":  _clip(str(record.get("failure_reason") or ""), 500),
            "fix_strategy":    _clip(str(record.get("fix_strategy") or ""), 500),
            "fix_pattern":     [str(x) for x in (record.get("fix_pattern") or [])],
            "failed_tool":     all_tools,
            "failed_cmd":      all_cmds,
            "files":           llm_files,
            "changed_files":   changed_files,
            "pattern_name":    pattern_name,
            "issue_ids":       issue_ids,
            "seed_count":      len(members),
        })
        print(f"[L2] ({resolved_repo}, {resolved_error_type}) → pattern={record.get('failure_pattern', '?')}")

    return out


# ── Phase 3: L3 synthesis ────────────────────────────────────────────────────

_L3_SCHEMA = """\
{
  "error_type":      "the error type shared by all L2 records in this group",
  "failed_tool":     ["all distinct failed tools observed across the L2 records"],
  "principle":       "2-3 sentence generalizable insight about this class of CI failure, applicable across repos",
  "fix_strategy":    "universal fix direction that applies across repos",
  "failure_patterns":["all distinct pattern names observed across the L2 records"],
  "failure_reasons": ["2-3 representative and diverse root-cause descriptions from different repos"]
}"""


def _build_l3_prompt(error_type: str, l2_records: List[Dict[str, Any]]) -> str:
    l2_text = json.dumps(
        [{k: v for k, v in r.items() if k not in ("changed_files", "failed_cmd", "issue_ids", "seed_count")}
         for r in l2_records],
        indent=2, ensure_ascii=False,
    )
    repos = _dedupe(str(r.get("repo") or "") for r in l2_records)
    return f"""\
You are a CI failure principle analyst.

Given L2 repo-level patterns with related error types from MULTIPLE DIFFERENT REPOS,
extract a CROSS-REPO GENERALIZABLE PRINCIPLE for this class of CI failure.

════════════════════════════════════
CONTEXT
════════════════════════════════════
error_type : {error_type}
repos      : {', '.join(repos)}

════════════════════════════════════
L2 REPO PATTERNS (from different repos)
════════════════════════════════════
{_clip(l2_text, 4000)}

════════════════════════════════════
TASK
════════════════════════════════════
Synthesize a single L3 cross-repo principle from the L2 patterns above.
Focus on what is UNIVERSAL across repos, not repo-specific details.
The principle and fix_strategy must be actionable for any repo encountering this error type.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L3_SCHEMA}
""".strip()


def build_l3_records(l2_rows: List[Dict[str, Any]], llm) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in l2_rows:
        grouped[str(row.get("error_type") or "")].append(row)

    out: List[Dict[str, Any]] = []
    for error_type, members in grouped.items():
        prompt = _build_l3_prompt(error_type, members)
        try:
            raw    = llm.invoke(prompt).content
            record = _parse_llm_json(raw)
        except Exception as exc:
            print(f"[L3] LLM failed for error_type={error_type}: {exc}")
            record = {}

        repos      = _dedupe(str(m.get("repo") or "") for m in members)
        # Aggregate tools from L2 members; supplement with anything the LLM additionally inferred
        all_tools  = _dedupe(
            [t for m in members for t in (m.get("failed_tool") or [])]
            + [str(x) for x in (record.get("failed_tool") or [])]
        )
        issue_ids  = _dedupe(iid for m in members for iid in (m.get("issue_ids") or []))
        issue_type = str(members[0].get("failure_pattern") or "") if members else ""

        # LLM-returned error_type is treated as confirmation; grouping key is authoritative
        resolved_error_type = error_type or str(record.get("error_type") or "")

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
        out.append({
            "error_type":         resolved_error_type,
            "issue_type":         issue_type,
            "failure_pattern":    issue_type,
            "repos":              repos,
            "repo_names":         repos,
            "failed_tool":        all_tools,
            "principle":          _clip(principle, 600),
            "fix_strategy":       fix_strategy,
            "fix_strategies":     [fix_strategy] if fix_strategy else [],
            "failure_patterns":   [str(x) for x in (record.get("failure_patterns") or [])],
            "failure_reasons":    [_clip(str(x), 300) for x in (record.get("failure_reasons") or [])[:3]],
            "evidence_issue_ids": issue_ids,
        })
        print(f"[L3] error_type={resolved_error_type} → repos={repos}")

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

    model_key = args.model_key
    llm       = get_llm(model_key)

    print(f"\n=== Phase 1: L1 extraction ({len(seed_rows)} seed issues) ===")
    l1_rows = extract_l1_records(seed_rows, analysis_rows, llm)
    _write_json(args.output_dir / "failure_memory.json", l1_rows)
    print(f"L1: {len(l1_rows)} records written.")

    print(f"\n=== Phase 2: L2 synthesis ({len(l1_rows)} L1 records) ===")
    l2_rows = build_l2_records(l1_rows, llm)
    _write_json(args.output_dir / "repo_memory.json", l2_rows)
    print(f"L2: {len(l2_rows)} records written.")

    print(f"\n=== Phase 3: L3 synthesis ({len(l2_rows)} L2 records) ===")
    l3_rows = build_l3_records(l2_rows, llm)
    _write_json(args.output_dir / "cross_memory.json", l3_rows)
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
            "L1": ["file", "error_type", "failure_pattern", "failure_reason", "fix_strategy",
                   "fix_pattern", "failed_tool", "issue_type", "dependent_files", "repo"],
            "L2": ["error_type", "failure_pattern", "failure_reason", "fix_strategy",
                   "fix_pattern", "files[file,failure_pattern,failure_reason]", "failed_tool", "repo"],
            "L3": ["error_type", "failure_pattern", "principle", "fix_strategy", "failure_patterns",
                   "failure_reasons", "failed_tool", "repos"],
        },
    }
    _write_json(args.output_dir / "memory_bank_summary.json", summary)
    print(f"\n=== Done ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
