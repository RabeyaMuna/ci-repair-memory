#!/usr/bin/env python3
"""
Memory bank builder — reads existing CILogAnalyzerLLM output and extracts L1/L2/L3.

Design
──────
Everything is driven by LLM prompts — no regex grouping, no rule-based taxonomy tables.

L1  (per file, parallel within issue)
    One LLM call per changed file.
    Extracts: what failed, why, how fixed, which other files are dependent and WHY.
    dependent_files → [{file, reason, must_change}]  (structured, not just a list of paths)

L2  (per distinct CI failure — two-step, self-contained records)
    Step 2 — Identification (1 lightweight LLM call):
      Input : compact L1 summaries [{file, issue_type, failure_pattern, failure_reason, failed_tool}]
      Output: [{issue_type, what_failed, files: ["actual/path.py"]}]
      Every gt_file must appear in exactly one item's files list.
    Step 3 — Enrichment per failure (1 LLM call per identified failure):
      Input : what_failed + FULL L1 records for those files + compact OTHER failures summary
      Chain-of-thought: root cause → per-file contribution → dependency links → causal chain
      Output: complete L2 record with changed_files [{file, reason, failure_reason, fix_strategy}]
    3 distinct CI failures (e.g. pip + mypy + ruff) → 3 enriched L2 records.
    Each record contains:
      • changed_files  [{file, reason, failure_reason, fix_strategy}] — 4-field, grounded in L1
      • dependent_files [{file, reason}] — files from OTHER failures causally linked
      • dependent_issues — what caused this failure or what it caused

L3  (one per CI failure, cross-repo)
    ONE LLM call using only the L2 sub-issue records (no file names, no repo name).
    Produces a universal principle: root_cause_pattern, cascade_pattern, fix_strategy.
    Strictly no repo-specific details — useful for any codebase.

Writes immediately after each issue (resumable by sha_fail):
  baselines/results/trs/failure_memory.json     ← L1 records
  baselines/results/trs/repo_memory.json        ← L2 records (≥1 per sha_fail)
  baselines/results/trs/cross_memory.json       ← L3 records (one per sha_fail)
  baselines/results/trs/memory_bank_summary.json

Usage:
  baselines/.venv/bin/python baselines/scripts/build_memory_bank.py
  baselines/.venv/bin/python baselines/scripts/build_memory_bank.py --model-key MiniMax-M2.5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf


PROJECT_ROOT   = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from utilities.llm_provider import get_default_model_key, get_llm


DEFAULT_SEED_FILE     = PROJECT_ROOT / "baselines" / "results" / "trs" / "trs_memory_seed_issues.json"
DEFAULT_ANALYSIS_FILE = PROJECT_ROOT / "baselines" / "results" / "trs" / "seed_log_details.json"
DEFAULT_CONFIG        = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT        = PROJECT_ROOT / "baselines" / "results" / "trs"


# ── generic helpers ────────────────────────────────────────────────────────────

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _normalize_path(path: str) -> str:
    return (path or "").strip().lstrip("/").replace("\\", "/")


def _clip(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _dedupe(items) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        s = str(x or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _extract_file_diff(diff_text: str, file_path: str) -> str:
    """Extract the diff section for a single file from the full diff."""
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


def _strip_fence(raw: str) -> str:
    """Remove markdown code fences (```json ... ```) from LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Parse a JSON object from LLM output."""
    text = _strip_fence(raw)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return {}


def _parse_llm_json_array(raw: str) -> List[Dict[str, Any]]:
    """Parse a JSON array from LLM output."""
    text = _strip_fence(raw)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)]
    except json.JSONDecodeError:
        pass
    # Fallback: find the first [...] block
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [r for r in result if isinstance(r, dict)]
        except Exception:
            pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# L1 — Per-file extraction
# ══════════════════════════════════════════════════════════════════════════════

_L1_SCHEMA = """\
{
  "repo":            "repo name",
  "file":            "path to this specific file, relative to repo root",
  "workflow_path":   "path to the CI workflow file",
  "error_type":      "high-level failure category, e.g. 'Type Checking', 'Dependency Error', 'Code Formatting', 'Test Failure'",
  "failure_pattern": "specific short pattern name, e.g. 'nullable-kwargs-unpack', 'package-version-constraint', 'unused-import'",
  "failure_reason":  "2-3 sentences: WHY this specific file caused the CI failure — grounded in the diff and log evidence. Be precise about what line or construct triggered the error.",
  "fix_strategy":    "1-2 sentences: what the ground-truth diff actually did to fix this file specifically.",
  "fix_pattern":     ["keyword1", "keyword2"],
  "failed_tool":     ["tool names: ruff, pytest, mypy, pip, etc."],
  "issue_type":      "one of exactly: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
  "dependent_files": [
    {
      "file":        "path to a related file from all_gt_files that is coupled to this file's failure",
      "reason":      "precise reason why this file and the related file are coupled — e.g. 'imports from it', 'tests it', 'shares a version constraint', 'co-patched to fix the same root cause'",
      "must_change": true
    }
  ]
}"""


def _build_l1_prompt(
    repo_name: str,
    sha_fail: str,
    workflow_path: str,
    error_context: List[str],
    error_types: List[Dict],
    relevant_files: List[Dict],
    failed_job: List[Dict],
    file_path: str,
    file_diff: str,
    all_gt_files: List[str],
) -> str:
    ec_text   = "\n".join(f"  • {x}" for x in error_context) or "  (none)"
    et_text   = json.dumps(error_types,    indent=2, ensure_ascii=False)
    rf_text   = json.dumps(relevant_files, indent=2, ensure_ascii=False)
    fj_text   = json.dumps(failed_job,     indent=2, ensure_ascii=False)
    gt_text   = "\n".join(f"  [{i}] {f}" for i, f in enumerate(all_gt_files)) or "  (none)"
    diff_text = _clip(file_diff, 2000) or "(no diff available for this file)"

    return f"""\
You are a CI failure memory extraction agent.

Your task is to extract a structured L1 memory record for ONE specific file
from a CI failure. The record will be used to help future agents fix similar failures.

════════════════════════════════
FAILURE CONTEXT
════════════════════════════════
repo          : {repo_name}
sha_fail      : {sha_fail}
workflow_path : {workflow_path}

════════════════════════════════
ALL CHANGED FILES IN THIS FAILURE
════════════════════════════════
(These are all files that were changed in the ground-truth fix.
 Some may be related to this file's failure.)
{gt_text}

════════════════════════════════
CI LOG ANALYSIS
════════════════════════════════
error_context (what the CI log says failed):
{ec_text}

error_types (structured error classification):
{et_text}

relevant_files (files the CI log mentions):
{rf_text}

failed_job (job steps that failed):
{fj_text}

════════════════════════════════
GROUND TRUTH FIX DIFF — file: {file_path}
════════════════════════════════
{diff_text}

════════════════════════════════
TASK
════════════════════════════════
Extract an L1 memory record for file "{file_path}".

For `failure_reason`: explain precisely WHY this file caused the CI failure.
  Reference the specific line, construct, or pattern from the diff/log — not a generic statement.

For `fix_strategy`: describe what the diff actually did to fix this file, not what the ideal fix would be.

For `dependent_files`: list only files from "ALL CHANGED FILES" above that are genuinely coupled
  to THIS file's failure. For each, give a precise `reason` (e.g., "imports this module",
  "shares the version constraint that was changed", "test file that exercises the fixed function")
  and set `must_change: true` if both files MUST be changed together for the fix to work,
  or `false` if the relationship is informational only.
  Leave as empty array [] if no file is genuinely coupled.

Return ONLY valid JSON matching this schema exactly (no markdown fences, no extra keys):
{_L1_SCHEMA}
""".strip()


def _normalise_dependent_files(raw: Any, all_gt_files: List[str]) -> List[Dict[str, Any]]:
    """
    Normalise whatever the LLM returned for dependent_files into
    [{file, reason, must_change}], filtering to files that exist in all_gt_files.
    Handles both old list-of-strings format and new list-of-dicts format.
    """
    if not raw:
        return []
    gt_set = {_normalize_path(f) for f in all_gt_files}
    result: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw:
        if isinstance(item, str):
            fp = _normalize_path(item)
            if fp and fp not in seen and fp in gt_set:
                seen.add(fp)
                result.append({"file": fp, "reason": "", "must_change": True})
        elif isinstance(item, dict):
            fp = _normalize_path(str(item.get("file") or ""))
            if fp and fp not in seen and fp in gt_set:
                seen.add(fp)
                result.append({
                    "file":        fp,
                    "reason":      _clip(str(item.get("reason") or ""), 300),
                    "must_change": bool(item.get("must_change", True)),
                })
    return result


def extract_l1_for_issue(
    repo_name: str,
    sha_fail: str,
    workflow_path: str,
    error_context: List[str],
    error_types: List[Dict],
    relevant_files: List[Dict],
    failed_job: List[Dict],
    diff_text: str,
    gt_files: List[str],
    task_id: str,
    workflow_name: str,
    llm,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for file_path in gt_files:
        file_diff = _extract_file_diff(diff_text, file_path)
        prompt    = _build_l1_prompt(
            repo_name=repo_name,
            sha_fail=sha_fail,
            workflow_path=workflow_path,
            error_context=error_context,
            error_types=error_types,
            relevant_files=relevant_files,
            failed_job=failed_job,
            file_path=file_path,
            file_diff=file_diff,
            all_gt_files=gt_files,
        )
        try:
            raw    = llm.invoke(prompt).content
            record = _parse_llm_json(raw)
        except Exception as exc:
            print(f"  [L1] LLM error for {file_path}: {exc}")
            record = {}

        resolved_file  = file_path or _normalize_path(str(record.get("file") or ""))
        resolved_wp    = workflow_path or str(record.get("workflow_path") or "")
        failure_reason = _clip(str(record.get("failure_reason") or ""), 500)
        fix_strategy   = _clip(str(record.get("fix_strategy")   or ""), 500)
        dep_files      = _normalise_dependent_files(record.get("dependent_files"), gt_files)

        records.append({
            "issue_id":        task_id,
            "sha_fail":        sha_fail,
            "repo":            repo_name,
            "repo_name":       repo_name,
            "workflow_path":   resolved_wp,
            "workflow_name":   workflow_name,
            "file":            resolved_file,
            "line_number":     None,
            "error_type":      str(record.get("error_type")      or ""),
            "failure_pattern": str(record.get("failure_pattern") or ""),
            "failure_reason":  failure_reason,
            "reason":          failure_reason,   # alias for retrieval compat
            "fix_strategy":    fix_strategy,
            "fix_pattern":     [str(x) for x in (record.get("fix_pattern") or [])],
            "failed_tool":     _dedupe([str(x) for x in (record.get("failed_tool") or [])]),
            "issue_type":      str(record.get("issue_type") or "other"),
            "dependent_files": dep_files,        # [{file, reason, must_change}]
        })
        print(
            f"  [L1] {resolved_file}"
            f" → issue_type={record.get('issue_type', '?')}"
            f" deps={[d['file'] for d in dep_files]}"
        )
    return records


# ══════════════════════════════════════════════════════════════════════════════
# L2 — Sub-issue extraction (two-step: identification + per-failure enrichment)
# ══════════════════════════════════════════════════════════════════════════════
#
# Step 2 — Identification (1 lightweight LLM call):
#   Input : compact L1 summaries [{file, issue_type, failure_pattern, failure_reason, failed_tool}]
#   Output: [{issue_type, what_failed, files: ["actual/path.py"]}]
#   Validation: every gt_file in exactly one item's files list; fallback for unassigned.
#
# Step 3 — Enrichment per failure (1 LLM call per identified failure):
#   Input : what_failed + FULL L1 records for those files + compact OTHER failures summary
#   Prompt: chain-of-thought — root cause → per-file → dependency links → causal chain
#   Output: complete L2 record with changed_files [{file, reason, failure_reason, fix_strategy}]

_L2_ID_SCHEMA = """\
[
  {
    "issue_type":  "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
    "what_failed": "one sentence: which CI job/check failed and what made it fail",
    "files":       ["exact/file/path.py as given in L1 summaries"]
  }
]

Rules:
- Return a JSON ARRAY — one object per DISTINCT CI job/check that failed.
- Each file from the L1 summaries MUST appear in exactly one object's files list — no omissions.
- Use the EXACT file paths as given in the L1 summaries — do not alter them.
- If all files belong to the same CI failure → 1 object with all files listed.
- If files belong to different CI failures → one object per failure, each file in exactly one."""


def _build_l2_id_prompt(repo_name: str, sha_fail: str, l1_records: List[Dict]) -> str:
    """Step 2: Lightweight identification — which files belong to which CI failure."""
    summaries = [
        {
            "file":            r.get("file", ""),
            "issue_type":      r.get("issue_type", ""),
            "failure_pattern": r.get("failure_pattern", ""),
            "failure_reason":  _clip(str(r.get("failure_reason") or ""), 120),
            "failed_tool":     r.get("failed_tool", []),
        }
        for r in l1_records
    ]

    return f"""\
You are a CI failure analyst.

A CI run failed. The L1 summaries below describe what went wrong per file.
Your task: identify how many DISTINCT CI jobs/checks failed in this run,
and assign each file to the ONE CI failure it belongs to.

════════════════════════════════
REPO CONTEXT
════════════════════════════════
repo     : {repo_name}
sha_fail : {sha_fail}

════════════════════════════════
L1 FILE SUMMARIES (one per changed file)
════════════════════════════════
{json.dumps(summaries, indent=2, ensure_ascii=False)}

════════════════════════════════
TASK
════════════════════════════════
A "distinct CI failure" is a specific CI job or check that failed independently:
  • "pip install" failing is one CI failure
  • "mypy --strict" failing is a SEPARATE CI failure
  • "ruff check" failing is a SEPARATE CI failure
  • "pytest" failing is a SEPARATE CI failure

Look at each file's issue_type, failed_tool, and failure_reason.
Files that share the same CI check belong to the same CI failure.

Rules:
  • Every file must appear in exactly one object's "files" list — no omissions.
  • Use exact file paths as given above — do not change them.
  • If all files failed due to the same CI check → return 1 object.
  • If files failed in different CI checks → return one object per check.

Return ONLY a valid JSON array matching this schema (no markdown fences, no extra keys):
{_L2_ID_SCHEMA}""".strip()


_L2_ENRICH_SCHEMA = """\
{
  "issue_type":      "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
  "error_type":      "high-level category, e.g. 'Dependency Error', 'Type Checking', 'Code Formatting'",
  "failure_pattern": "short pattern name, e.g. 'pip-pep440-violation', 'mypy-strict-optional'",
  "failure_reason":  "2-3 sentences: root cause of THIS specific CI job/check failing — grounded in L1 evidence",
  "failed_tool":     ["tools that failed for this specific CI failure"],
  "failed_cmd":      ["exact CI commands that failed, e.g. 'pip install -e .[all]', 'mypy camel/ --strict'"],
  "fix_strategy":    "1-2 sentences: what the overall fix did to resolve THIS CI failure",
  "fix_pattern":     ["keyword1", "keyword2"],
  "changed_files": [
    {
      "file":           "exact/file/path.py",
      "reason":         "why this file was changed to fix THIS CI failure",
      "failure_reason": "what specifically in this file caused the CI failure — from L1 evidence",
      "fix_strategy":   "what was done to fix this specific file — from L1 evidence"
    }
  ],
  "dependent_files": [
    {
      "file":   "path/to/file from a DIFFERENT CI failure that is causally linked to this one",
      "reason": "why this file from another CI failure is linked"
    }
  ],
  "dependent_issues": [
    {
      "issue_type":  "the other CI failure's issue_type",
      "direction":   "caused_by OR causes",
      "reason":      "precise causal explanation of the link",
      "what_arose":  "what specific problem appeared in THIS failure because of the link",
      "what_to_fix": "what must be done in the linked failure to resolve the dependency"
    }
  ],
  "dependency_note": "one sentence summarising the causal chain, or empty string if standalone"
}"""


def _build_l2_enrich_prompt(
    repo_name: str,
    sha_fail: str,
    what_failed: str,
    issue_type: str,
    this_l1_records: List[Dict],
    other_failures_summary: List[Dict],
) -> str:
    """Step 3: Chain-of-thought enrichment for ONE identified CI failure."""
    # Full L1 records for this failure's files (strip internal/alias fields)
    this_l1_data = [
        {k: v for k, v in r.items()
         if k not in ("sha_fail", "issue_id", "workflow_name", "line_number", "reason")}
        for r in this_l1_records
    ]

    other_text = (
        json.dumps(other_failures_summary, indent=2, ensure_ascii=False)
        if other_failures_summary
        else "  (none — this is the only CI failure in this run)"
    )

    return f"""\
You are a CI failure analyst building a structured memory record for ONE CI failure.

════════════════════════════════
REPO CONTEXT
════════════════════════════════
repo        : {repo_name}
sha_fail    : {sha_fail}
issue_type  : {issue_type}
what_failed : {what_failed}

════════════════════════════════
L1 RECORDS FOR THIS CI FAILURE
(full details for files that belong to this specific CI failure)
════════════════════════════════
{_clip(json.dumps(this_l1_data, indent=2, ensure_ascii=False), 4000)}

════════════════════════════════
OTHER CI FAILURES IN THE SAME RUN
(compact summaries — for cross-failure dependency analysis only)
════════════════════════════════
{other_text}

════════════════════════════════
CHAIN-OF-THOUGHT TASK
════════════════════════════════
Answer these questions step by step, then produce the final JSON:

1. ROOT CAUSE: Why did this specific CI job/check fail?
   What is the fundamental root cause, grounded in the L1 evidence above?

2. PER-FILE CONTRIBUTION: For each file in the L1 records above:
   - What exactly in that file caused this CI failure?
   - What was done in the ground-truth fix to address it?

3. CROSS-FAILURE DEPENDENCIES: Looking at the other CI failures in this run:
   - Did any other CI failure CAUSE this one? (direction: "caused_by")
   - Did this failure CAUSE any other CI failure? (direction: "causes")
   - What files from those other failures are causally linked to this one?

4. CAUSAL CHAIN: Write a one-sentence summary of how this failure fits in the
   overall failure chain for this CI run (or "standalone" if no dependencies).

Now produce a complete L2 record for this CI failure:
  • changed_files: ALL files from the L1 records above, with:
      - reason         : why it was changed for THIS failure
      - failure_reason : what in this file caused the CI failure (from L1)
      - fix_strategy   : what the fix did in this file (from L1)
  • dependent_files: files from OTHER CI failures causally linked to THIS one
  • dependent_issues: causal links to other CI failures (from step 3 above)
  • dependency_note: one-sentence causal chain summary (from step 4 above)

Return ONLY valid JSON matching this schema exactly (no markdown fences, no extra keys):
{_L2_ENRICH_SCHEMA}
""".strip()


def _normalise_l2_changed_files(raw: Any) -> List[Dict[str, str]]:
    """Normalise changed_files into [{file, reason, failure_reason, fix_strategy}] (4 fields)."""
    if not raw or not isinstance(raw, list):
        return []
    result = []
    seen: set = set()
    for item in raw:
        if isinstance(item, str):
            fp = _normalize_path(item)
            if fp and fp not in seen:
                seen.add(fp)
                result.append({"file": fp, "reason": "", "failure_reason": "", "fix_strategy": ""})
        elif isinstance(item, dict):
            fp = _normalize_path(str(item.get("file") or ""))
            if fp and fp not in seen:
                seen.add(fp)
                result.append({
                    "file":           fp,
                    "reason":         _clip(str(item.get("reason")         or ""), 300),
                    "failure_reason": _clip(str(item.get("failure_reason") or ""), 300),
                    "fix_strategy":   _clip(str(item.get("fix_strategy")   or ""), 300),
                })
    return result


def _normalise_l2_file_list(raw: Any) -> List[Dict[str, str]]:
    """Normalise dependent_files into [{file, reason}] (2 fields)."""
    if not raw or not isinstance(raw, list):
        return []
    result = []
    seen: set = set()
    for item in raw:
        if isinstance(item, str):
            fp = _normalize_path(item)
            if fp and fp not in seen:
                seen.add(fp)
                result.append({"file": fp, "reason": ""})
        elif isinstance(item, dict):
            fp = _normalize_path(str(item.get("file") or ""))
            if fp and fp not in seen:
                seen.add(fp)
                result.append({
                    "file":   fp,
                    "reason": _clip(str(item.get("reason") or ""), 300),
                })
    return result


def extract_l2_sub_issues(
    repo_name: str,
    sha_fail: str,
    l1_records: List[Dict],
    llm,
) -> List[Dict[str, Any]]:
    """
    Two-step L2 extraction:

    Step 2 — Lightweight identification (1 LLM call):
      Receives compact L1 summaries and groups files by distinct CI failure.
      Returns [{issue_type, what_failed, files: [...]}].
      Validation: every gt_file in exactly one item; unassigned files → fallback group.

    Step 3 — Enrichment per failure (1 LLM call per failure):
      Chain-of-thought prompt with full L1 records for that failure's files.
      Returns complete L2 record with changed_files [{file, reason, failure_reason, fix_strategy}].

    3 distinct CI failures → 3 L2 records, each fully self-contained with actual file paths.
    """
    # Build file → L1 record lookup
    l1_by_file: Dict[str, Dict] = {}
    for r in l1_records:
        fp = _normalize_path(r.get("file", ""))
        if fp:
            l1_by_file[fp] = r
    all_gt_files = set(l1_by_file.keys())

    # ── Step 2: Identification ──────────────────────────────────────────────
    print(f"  [L2 Step2] identifying distinct CI failures for {len(l1_records)} L1 records...")
    id_prompt   = _build_l2_id_prompt(repo_name, sha_fail, l1_records)
    id_records: List[Dict] = []
    try:
        raw_id      = llm.invoke(id_prompt).content
        id_records  = _parse_llm_json_array(raw_id)
    except Exception as exc:
        print(f"  [L2 Step2] LLM error: {exc}")

    # Validate: every gt_file must appear in exactly one item
    assigned_files: set = set()
    valid_id: List[Dict] = []
    for item in id_records:
        if not isinstance(item, dict):
            continue
        files = [_normalize_path(f) for f in (item.get("files") or []) if f]
        known = [f for f in files if f in all_gt_files]
        if known:
            valid_id.append({
                "issue_type":  str(item.get("issue_type") or "other"),
                "what_failed": str(item.get("what_failed") or ""),
                "files":       known,
            })
            assigned_files.update(known)

    # Fallback: collect unassigned files into a catch-all group
    unassigned = all_gt_files - assigned_files
    if unassigned:
        print(f"  [L2 Step2] {len(unassigned)} unassigned file(s) → fallback group")
        valid_id.append({
            "issue_type":  "other",
            "what_failed": "unclassified files not assigned to any identified CI failure",
            "files":       sorted(unassigned),
        })

    if not valid_id:
        print("  [L2 Step2] no valid groups — single-failure fallback")
        valid_id = [{
            "issue_type":  "other",
            "what_failed": "single CI failure covering all changed files",
            "files":       list(all_gt_files),
        }]

    n_failures = len(valid_id)
    print(f"  [L2 Step2] identified {n_failures} distinct CI failure(s)")

    # ── Step 3: Enrichment per failure ──────────────────────────────────────
    task_ids = _dedupe(str(r.get("issue_id") or "") for r in l1_records)
    results:  List[Dict[str, Any]] = []

    for idx, failure in enumerate(valid_id):
        issue_type  = failure["issue_type"]
        what_failed = failure["what_failed"]
        this_files  = failure["files"]

        # Full L1 records for this failure's files
        this_l1 = [l1_by_file[fp] for fp in this_files if fp in l1_by_file]

        # Compact summaries of OTHER failures (context only — no full L1 data)
        other_summaries = [
            {
                "issue_type":  v["issue_type"],
                "what_failed": v["what_failed"],
                "files":       v["files"],
            }
            for j, v in enumerate(valid_id) if j != idx
        ]

        print(
            f"  [L2 Step3/{issue_type}] ({idx+1}/{n_failures})"
            f" enriching {len(this_files)} file(s)..."
        )

        enrich_prompt = _build_l2_enrich_prompt(
            repo_name=repo_name,
            sha_fail=sha_fail,
            what_failed=what_failed,
            issue_type=issue_type,
            this_l1_records=this_l1,
            other_failures_summary=other_summaries,
        )

        rec: Dict = {}
        try:
            raw_enrich = llm.invoke(enrich_prompt).content
            rec        = _parse_llm_json(raw_enrich)
        except Exception as exc:
            print(f"  [L2 Step3] LLM error for failure {idx}: {exc}")

        if not rec:
            # Fallback: build minimal record from L1 evidence
            print(f"  [L2 Step3] no record parsed — L1 fallback for failure {idx}")
            rec = {
                "issue_type":      issue_type,
                "error_type":      this_l1[0].get("error_type", "")      if this_l1 else "",
                "failure_pattern": this_l1[0].get("failure_pattern", "") if this_l1 else "",
                "failure_reason":  this_l1[0].get("failure_reason", "")  if this_l1 else "",
                "failed_tool":     _dedupe([t for r in this_l1 for t in (r.get("failed_tool") or [])]),
                "failed_cmd":      [],
                "fix_strategy":    this_l1[0].get("fix_strategy", "")    if this_l1 else "",
                "fix_pattern":     this_l1[0].get("fix_pattern", [])     if this_l1 else [],
                "changed_files": [
                    {
                        "file":           r.get("file", ""),
                        "reason":         r.get("failure_reason", ""),
                        "failure_reason": r.get("failure_reason", ""),
                        "fix_strategy":   r.get("fix_strategy", ""),
                    }
                    for r in this_l1 if r.get("file")
                ],
                "dependent_files":  [],
                "dependent_issues": [],
                "dependency_note":  "",
            }

        # Normalise: changed_files → 4-field; dependent_files → 2-field
        changed_files   = _normalise_l2_changed_files(rec.get("changed_files"))
        dependent_files = _normalise_l2_file_list(rec.get("dependent_files"))

        # Aggregate tools from L1 + LLM output
        all_tools = _dedupe(
            [t for r in this_l1 for t in (r.get("failed_tool") or [])]
            + [str(x) for x in (rec.get("failed_tool") or [])]
        )
        workflow_paths = _dedupe(
            [str(r.get("workflow_path") or "") for r in this_l1 if r.get("workflow_path")]
        )

        # Normalise dependent_issues
        dep_issues = [
            {
                "issue_type":  str(di.get("issue_type") or ""),
                "direction":   str(di.get("direction")  or ""),
                "reason":      _clip(str(di.get("reason")      or ""), 300),
                "what_arose":  _clip(str(di.get("what_arose")  or ""), 200),
                "what_to_fix": _clip(str(di.get("what_to_fix") or ""), 200),
            }
            for di in (rec.get("dependent_issues") or [])
            if isinstance(di, dict) and di.get("issue_type")
        ]

        print(
            f"  [L2/{issue_type}] ({idx+1}/{n_failures})"
            f" pattern={rec.get('failure_pattern', '?')}"
            f" changed={len(changed_files)} files"
            f" dep_issues={[d['issue_type'] for d in dep_issues]}"
        )

        results.append({
            # Identity
            "sha_fail":         sha_fail,
            "sub_issue_id":     f"{sha_fail}_{idx}",
            "sub_issue_index":  idx,
            "total_sub_issues": n_failures,
            "task_ids":         task_ids,
            # Repo context
            "repo":             repo_name,
            "repo_name":        repo_name,
            "workflow_path":    workflow_paths[0] if workflow_paths else "",
            # Classification
            "issue_type":       str(rec.get("issue_type") or issue_type),
            "error_type":       str(rec.get("error_type") or ""),
            # Failure details
            "failure_pattern":  str(rec.get("failure_pattern") or ""),
            "failure_reason":   _clip(str(rec.get("failure_reason") or ""), 500),
            "failed_tool":      all_tools,
            "failed_cmd":       _dedupe([str(x) for x in (rec.get("failed_cmd") or [])]),
            "fix_strategy":     _clip(str(rec.get("fix_strategy") or ""), 500),
            "fix_pattern":      [str(x) for x in (rec.get("fix_pattern") or [])],
            # Files — fully self-contained, actual paths, no indices
            "changed_files":    changed_files,    # [{file, reason, failure_reason, fix_strategy}]
            "dependent_files":  dependent_files,  # [{file, reason}] — from OTHER CI failures
            # Cross-failure dependency
            "dependent_issues": dep_issues,       # [{issue_type, direction, reason, what_arose, what_to_fix}]
            "dependency_note":  _clip(str(rec.get("dependency_note") or ""), 300),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# L3 — Universal cross-repo principle
# ══════════════════════════════════════════════════════════════════════════════

_L3_SCHEMA = """\
{
  "error_type":         "primary error type — the most upstream / root cause type",
  "sub_issue_types":    ["all distinct sub-issue types observed in this failure"],
  "root_cause_pattern": "1-2 sentences: what fundamentally triggers this class of CI failure — the root cause pattern, abstracted from any specific repo or file",
  "cascade_pattern":    "1-2 sentences: how failures propagate once the root cause occurs — what breaks, in what order, and why",
  "principle":          "2-3 sentences: universal insight applicable to ANY codebase encountering this failure pattern — no file names, no repo names, no sha values",
  "fix_strategy":       "ordered list of universal fix steps as a single string, e.g. '1. Fix root cause. 2. Update call sites. 3. Sweep type errors.'",
  "failure_patterns":   ["distinct failure pattern names observed across all sub-issues"],
  "failure_reasons":    ["1-2 representative abstracted root-cause descriptions — no file names, no repo-specific details"]
}"""


def _build_l3_prompt(l2_records: List[Dict]) -> str:
    # Strip all identity / repo-specific fields so the LLM can't reference them
    l2_stripped = []
    for r in l2_records:
        l2_stripped.append({
            "issue_type":       r.get("issue_type", ""),
            "error_type":       r.get("error_type", ""),
            "failure_pattern":  r.get("failure_pattern", ""),
            "failure_reason":   r.get("failure_reason", ""),
            "fix_strategy":     r.get("fix_strategy", ""),
            "fix_pattern":      r.get("fix_pattern", []),
            "failed_tool":      r.get("failed_tool", []),
            "failed_cmd":       r.get("failed_cmd", []),
            "dependent_issues": r.get("dependent_issues", []),
            "dependency_note":  r.get("dependency_note", ""),
            # Include per-file summaries with failure_reason + fix_strategy but NO filenames
            "file_summaries": [
                {
                    "failure_pattern": f.get("failure_pattern", ""),
                    "failure_reason":  _clip(str(f.get("failure_reason") or ""), 150),
                    "fix_strategy":    _clip(str(f.get("fix_strategy")   or ""), 120),
                }
                for f in (r.get("changed_files") or [])
            ],
        })

    sub_types = _dedupe(str(r.get("issue_type") or "") for r in l2_records if r.get("issue_type"))
    n = len(l2_records)

    return f"""\
You are a CI failure principle analyst.

A CI failure was decomposed into {n} sub-issue(s).
Your task is to extract a UNIVERSAL L3 principle from these sub-issues — a principle
that would help any developer fix this CLASS of CI failure in ANY codebase.

STRICT RULES:
  • Do NOT mention any repository name, file name, sha, or project-specific detail.
  • Write as if explaining to someone who has never seen this codebase.
  • The principle, root_cause_pattern, cascade_pattern and fix_strategy must be
    applicable to a Python project with similar CI tooling.

════════════════════════════════
SUB-ISSUE TYPES OBSERVED
════════════════════════════════
{', '.join(sub_types) or '(none)'}

════════════════════════════════
L2 SUB-ISSUE RECORDS (identity fields removed)
════════════════════════════════
{_clip(json.dumps(l2_stripped, indent=2, ensure_ascii=False), 4000)}

════════════════════════════════
TASK
════════════════════════════════
Extract ONE L3 universal principle.

root_cause_pattern : What class of root cause triggers this failure? (abstract, no specifics)
cascade_pattern    : How do downstream failures propagate from the root cause?
principle          : The key insight a developer needs to understand and prevent this failure.
fix_strategy       : Ordered, universal fix steps for this failure class.
failure_patterns   : Distinct pattern names observed (keep them generic, e.g. "nullable-kwargs-unpack").
failure_reasons    : 1-2 abstracted root-cause descriptions with no file/repo specifics.

Return ONLY valid JSON matching this schema exactly (no markdown fences, no extra keys):
{_L3_SCHEMA}
""".strip()


def extract_l3_for_issue(
    sha_fail: str,
    repo_name: str,
    l2_records: List[Dict],
    l1_records: List[Dict],
    llm,
) -> Dict[str, Any]:
    prompt = _build_l3_prompt(l2_records)
    try:
        raw    = llm.invoke(prompt).content
        record = _parse_llm_json(raw)
    except Exception as exc:
        print(f"  [L3] LLM error: {exc}")
        record = {}

    # Aggregate tools across all sub-issues
    all_tools = _dedupe(
        [t for r in l2_records for t in (r.get("failed_tool") or [])]
        + [t for r in l1_records  for t in (r.get("failed_tool") or [])]
        + [str(x) for x in (record.get("failed_tool") or [])]
    )

    sub_issue_types = _dedupe(
        [str(x) for x in (record.get("sub_issue_types") or [])]
        + [str(r.get("issue_type") or "") for r in l2_records]
    )

    # Build principle — must NOT include repo name or sha
    raw_principle = str(record.get("principle") or "").strip()
    primary_l2    = l2_records[0] if l2_records else {}
    primary_et    = str(record.get("error_type") or primary_l2.get("error_type") or "")
    principle     = (
        f"error_type={primary_et}: {_clip(raw_principle, 450)}"
        if raw_principle else
        f"error_type={primary_et}: (no principle extracted)"
    )

    fix_strategy = _clip(
        str(record.get("fix_strategy") or primary_l2.get("fix_strategy") or ""), 600
    )

    print(
        f"  [L3] error_type={primary_et}"
        f" sub_types={sub_issue_types}"
        f" n_sub_issues={len(l2_records)}"
    )

    return {
        # Identity (for traceability only — not used in retrieval scoring)
        "sha_fail":           sha_fail,
        "repo":               repo_name,
        "repos":              [repo_name],
        # Classification
        "error_type":         primary_et,
        "sub_issue_types":    sub_issue_types,
        "n_sub_issues":       len(l2_records),
        "failed_tool":        all_tools,
        # Universal knowledge
        "root_cause_pattern": _clip(str(record.get("root_cause_pattern") or ""), 400),
        "cascade_pattern":    _clip(str(record.get("cascade_pattern")    or ""), 400),
        "principle":          _clip(principle, 600),
        "fix_strategy":       fix_strategy,
        "fix_strategies":     [fix_strategy] if fix_strategy else [],
        "failure_patterns":   [str(x) for x in (record.get("failure_patterns") or [])],
        "failure_reasons":    [_clip(str(x), 300) for x in (record.get("failure_reasons") or [])[:3]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build L1/L2/L3 memory bank from seed_log_details.json. "
            "L1: per-file (structured dependent_files with reason+must_change). "
            "L2: two-step — LLM identification of CI failures + per-failure enrichment "
            "    with changed_files {file,reason,failure_reason,fix_strategy} and dependent_issues. "
            "L3: universal cross-repo principle (no repo-specific details). "
            "Resumable by sha_fail."
        )
    )
    parser.add_argument("--seed-file",     type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    parser.add_argument("--config",        type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-key",     default=get_default_model_key())
    args = parser.parse_args()

    seed_rows     = json.loads(args.seed_file.read_text(encoding="utf-8"))
    analysis_rows = json.loads(args.analysis_file.read_text(encoding="utf-8"))
    analysis_by_sha = {str(r.get("sha_fail") or ""): r for r in analysis_rows}

    llm = get_llm(args.model_key)

    out_l1 = args.output_dir / "failure_memory.json"
    out_l2 = args.output_dir / "repo_memory.json"
    out_l3 = args.output_dir / "cross_memory.json"

    l1_all: List[Dict] = _load_json(out_l1)
    l2_all: List[Dict] = _load_json(out_l2)
    l3_all: List[Dict] = _load_json(out_l3)
    # L3 is one record per sha_fail — use it as the completion marker
    done_shas = {str(r.get("sha_fail") or "") for r in l3_all}

    total   = len(seed_rows)
    skipped = 0
    done    = 0

    print(f"\n=== Building memory bank ({total} seed issues) ===")
    print(f"Already done : {len(done_shas)}")
    print(f"Remaining    : {total - len(done_shas)}")
    print(f"Design       : L1 dep_files={{file,reason,must_change}} | L2 two-step | L3 universal\n")

    for i, seed in enumerate(seed_rows, 1):
        sha_fail      = str(seed.get("sha_fail")      or "")
        task_id       = str(seed.get("id")            or "")
        repo_name     = str(seed.get("repo_name")     or "")
        workflow_path = str(seed.get("workflow_path") or "")
        workflow_name = str(seed.get("workflow_name") or "")
        diff_text     = str(seed.get("diff")          or "")
        gt_files      = [_normalize_path(x) for x in (seed.get("ground_truth_files") or [])]

        if sha_fail in done_shas:
            print(f"[{i}/{total}] skip {sha_fail[:12]}… ({repo_name}) — already done")
            skipped += 1
            continue

        wrapper  = analysis_by_sha.get(sha_fail, {})
        analysis = wrapper.get("analysis") or {}

        if not analysis:
            print(f"[{i}/{total}] skip {sha_fail[:12]}… ({repo_name}) — no log analysis")
            skipped += 1
            continue

        if not gt_files:
            print(f"[{i}/{total}] skip {sha_fail[:12]}… ({repo_name}) — no ground-truth files")
            skipped += 1
            continue

        error_context  = [str(x) for x in (analysis.get("error_context")   or []) if str(x).strip()]
        error_types    = analysis.get("error_types")    or []
        relevant_files = analysis.get("relevant_files") or []
        failed_job     = analysis.get("failed_job")     or []

        print(f"\n[{i}/{total}] {sha_fail[:12]}… ({repo_name}) — {len(gt_files)} files")

        # ── L1: per file ──────────────────────────────────────────────────────
        l1_records = extract_l1_for_issue(
            repo_name=repo_name, sha_fail=sha_fail, workflow_path=workflow_path,
            error_context=error_context, error_types=error_types,
            relevant_files=relevant_files, failed_job=failed_job,
            diff_text=diff_text, gt_files=gt_files,
            task_id=task_id, workflow_name=workflow_name, llm=llm,
        )
        l1_all.extend(l1_records)
        _write_json(out_l1, l1_all)

        if not l1_records:
            print("  [L1] no records extracted — skipping L2/L3")
            skipped += 1
            continue

        # ── L2: two-step sub-issue extraction ────────────────────────────────
        l2_sub_records = extract_l2_sub_issues(
            repo_name=repo_name, sha_fail=sha_fail, l1_records=l1_records, llm=llm,
        )
        l2_all.extend(l2_sub_records)
        _write_json(out_l2, l2_all)

        if not l2_sub_records:
            print("  [L2] no sub-issue records produced — skipping L3")
            skipped += 1
            continue

        # ── L3: universal principle ───────────────────────────────────────────
        l3_record = extract_l3_for_issue(
            sha_fail=sha_fail, repo_name=repo_name,
            l2_records=l2_sub_records, l1_records=l1_records, llm=llm,
        )
        l3_all.append(l3_record)
        _write_json(out_l3, l3_all)

        done += 1
        n_sub = len(l2_sub_records)
        print(
            f"  → L1({len(l1_records)} files)"
            f" + L2({n_sub} sub-issue{'s' if n_sub != 1 else ''})"
            f" + L3(1 universal)"
            f"   [totals: L1={len(l1_all)} L2={len(l2_all)} L3={len(l3_all)}]"
        )

    summary = {
        "model_key":     args.model_key,
        "seed_file":     str(args.seed_file),
        "analysis_file": str(args.analysis_file),
        "total_issues":  total,
        "processed":     done,
        "skipped":       skipped,
        "design": {
            "L1": "per-file; dependent_files={file,reason,must_change}",
            "L2": "two-step: identification + per-failure enrichment; changed_files={file,reason,failure_reason,fix_strategy}; dependent_issues={issue_type,direction,reason,what_arose,what_to_fix}",
            "L3": "universal principle; root_cause_pattern+cascade_pattern; no repo-specific details",
        },
        "memory_counts": {
            "L1_failure_memory": len(l1_all),
            "L2_repo_memory":    len(l2_all),
            "L3_cross_memory":   len(l3_all),
        },
    }
    _write_json(args.output_dir / "memory_bank_summary.json", summary)

    print(f"\n=== Done ===")
    print(f"Processed : {done}   Skipped : {skipped}")
    print(f"L1 : {len(l1_all)} records")
    print(f"L2 : {len(l2_all)} records (two-step sub-issues)")
    print(f"L3 : {len(l3_all)} records (universal)")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
