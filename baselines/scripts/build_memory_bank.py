#!/usr/bin/env python3
"""
Memory bank builder — reads existing CILogAnalyzerLLM output and extracts L1/L2/L3.

CILogAnalyzerLLM is already done (seed_log_details.json).
This script reads that file and for each issue runs:
  1. LLM L1 extraction  → per-file record (with dependent_files)
  2. LLM L2 extraction  → repo-level view of THIS issue
  3. LLM L3 extraction  → generalizable principle from THIS issue

Writes immediately after each issue (resumable by sha_fail):
  baselines/results/failure_memory.json   ← L1 records
  baselines/results/repo_memory.json      ← L2 records
  baselines/results/cross_memory.json     ← L3 records
  baselines/results/memory_bank_summary.json

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
from typing import Any, Dict, List

from omegaconf import OmegaConf


PROJECT_ROOT   = Path(__file__).resolve().parents[2]
BASELINES_ROOT = PROJECT_ROOT / "baselines"
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from utilities.llm_provider import get_default_model_key, get_llm


DEFAULT_SEED_FILE     = PROJECT_ROOT / "baselines" / "results" / "memory_seed_issues.json"
DEFAULT_ANALYSIS_FILE = PROJECT_ROOT / "baselines" / "results" / "seed_log_details.json"
DEFAULT_CONFIG        = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT        = PROJECT_ROOT / "baselines" / "results"


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


# ── L1 extraction (per file) ──────────────────────────────────────────────────

_L1_SCHEMA = """\
{
  "repo":            "repo name",
  "file":            "path to this specific file, relative to repo root",
  "workflow_path":   "path to the CI workflow file",
  "error_type":      "high-level category: 'Code Formatting', 'Test Failure', 'Type Checking', 'Dependency Error', etc.",
  "failure_pattern": "specific pattern name: 'unused-import', 'line-too-long', 'assertion-error', 'missing-package', etc.",
  "failure_reason":  "2-3 sentence explanation of WHY THIS FILE caused the CI failure, grounded in log evidence",
  "fix_strategy":    "1-2 sentence description of what the ground-truth fix actually did to this file",
  "fix_pattern":     ["keyword1", "keyword2"],
  "failed_tool":     ["ruff", "pytest", "mypy"],
  "issue_type":      "one of: formatting | test_failure | type_checking | dependency_or_env | workflow_config | import_or_module | other",
  "dependent_files": ["other files that changed together with this file in the same fix — files that are co-dependent or that this file imports/uses"]
}"""


def _build_l1_prompt(
    repo_name: str, sha_fail: str, workflow_path: str,
    error_context: List[str], error_types: List[Dict], relevant_files: List[Dict],
    failed_job: List[Dict], file_path: str, file_diff: str, all_gt_files: List[str],
) -> str:
    ec_text   = "\n".join(f"  • {x}" for x in error_context) or "  (none)"
    et_text   = json.dumps(error_types,    indent=2, ensure_ascii=False)
    rf_text   = json.dumps(relevant_files, indent=2, ensure_ascii=False)
    fj_text   = json.dumps(failed_job,     indent=2, ensure_ascii=False)
    gt_text   = ", ".join(all_gt_files) or "(none)"
    diff_text = _clip(file_diff, 2000) or "(no diff available)"

    return f"""\
You are a CI failure memory extraction agent.

Given a CI failure's structured log analysis and the ground-truth fix diff for ONE specific file,
extract a structured L1 memory record capturing what failed in this file and how it was fixed.

Pay special attention to `dependent_files`: list any other files from `all_gt_files` that
are closely related to this file's failure (imported by it, tests for it, or changed together).

════════════════════════════════
FAILURE CONTEXT
════════════════════════════════
repo          : {repo_name}
sha_fail      : {sha_fail}
workflow_path : {workflow_path}
all_gt_files  : {gt_text}

════════════════════════════════
LOG ANALYSIS
════════════════════════════════
error_context:
{ec_text}

error_types:
{et_text}

relevant_files (log-identified):
{rf_text}

failed_job:
{fj_text}

════════════════════════════════
GROUND TRUTH FIX — file: {file_path}
════════════════════════════════
{diff_text}

════════════════════════════════
TASK
════════════════════════════════
Extract an L1 memory record for file "{file_path}".
Use ONLY evidence from the log analysis and the diff above.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L1_SCHEMA}
""".strip()


def extract_l1_for_issue(
    repo_name: str, sha_fail: str, workflow_path: str,
    error_context: List[str], error_types: List[Dict], relevant_files: List[Dict],
    failed_job: List[Dict], diff_text: str, gt_files: List[str],
    task_id: str, workflow_name: str, llm,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
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
            print(f"  [L1] LLM failed for {file_path}: {exc}")
            record = {}

        resolved_file          = file_path or str(record.get("file")          or "")
        resolved_workflow_path = workflow_path or str(record.get("workflow_path") or "")
        failure_reason = _clip(str(record.get("failure_reason") or ""), 500)

        records.append({
            "issue_id":        task_id,
            "sha_fail":        sha_fail,
            "repo":            repo_name,
            "repo_name":       repo_name,
            "workflow_path":   resolved_workflow_path,
            "workflow_name":   workflow_name,
            "file":            resolved_file,
            "line_number":     None,
            "error_type":      str(record.get("error_type")      or ""),
            "failure_pattern": str(record.get("failure_pattern") or ""),
            "failure_reason":  failure_reason,
            "reason":          failure_reason,
            "fix_strategy":    _clip(str(record.get("fix_strategy") or ""), 500),
            "fix_pattern":     [str(x) for x in (record.get("fix_pattern") or [])],
            "failed_tool":     _dedupe([str(x) for x in (record.get("failed_tool") or [])]),
            "issue_type":      str(record.get("issue_type")      or ""),
            "dependent_files": _dedupe([_normalize_path(str(x)) for x in (record.get("dependent_files") or [])]),
        })
        print(f"  [L1] {resolved_file} → error_type={record.get('error_type', '?')}")
    return records


# ── L2 extraction (per issue — repo-level view) ───────────────────────────────

_L2_SCHEMA = """\
{
  "repo":            "repo name",
  "error_type":      "primary error type for this issue",
  "workflow_path":   "CI workflow file path",
  "failed_tool":     ["tools that failed: ruff, pytest, mypy, etc."],
  "failed_cmd":      ["exact commands that failed from the workflow"],
  "failure_pattern": "synthesized pattern name capturing the theme of this failure",
  "failure_reason":  "2-3 sentence explanation of the repo-level root cause — what went wrong across all files",
  "fix_strategy":    "1-2 sentence description of what the overall fix did at the repo level",
  "fix_pattern":     ["keyword1", "keyword2"],
  "files": [
    {
      "file":            "path/to/file",
      "failure_pattern": "what pattern this file triggers",
      "failure_reason":  "why this file is involved"
    }
  ]
}"""


def _build_l2_prompt(repo_name: str, sha_fail: str, l1_records: List[Dict]) -> str:
    l1_text = json.dumps(
        [{k: v for k, v in r.items() if k not in ("sha_fail", "issue_id", "workflow_name", "line_number")}
         for r in l1_records],
        indent=2, ensure_ascii=False,
    )
    return f"""\
You are a CI failure pattern analyst.

Given L1 failure records for ALL files changed in a SINGLE CI failure,
synthesize a REPO-LEVEL L2 memory record that captures the overall failure pattern
and how the files relate to each other.

════════════════════════════════
CONTEXT
════════════════════════════════
repo     : {repo_name}
sha_fail : {sha_fail}

════════════════════════════════
L1 RECORDS (all files in this failure)
════════════════════════════════
{_clip(l1_text, 4000)}

════════════════════════════════
TASK
════════════════════════════════
Synthesize ONE L2 repo-level record from the L1 records above.
Capture the overall failure pattern, the root cause across all files, and the fix approach.
Include all changed files with their specific patterns and reasons.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L2_SCHEMA}
""".strip()


def extract_l2_for_issue(
    repo_name: str, sha_fail: str, l1_records: List[Dict], llm,
) -> Dict[str, Any]:
    prompt = _build_l2_prompt(repo_name, sha_fail, l1_records)
    try:
        raw    = llm.invoke(prompt).content
        record = _parse_llm_json(raw)
    except Exception as exc:
        print(f"  [L2] LLM failed: {exc}")
        record = {}

    # Aggregate structural fields from L1 records; supplement with LLM output
    all_tools = _dedupe(
        [t for r in l1_records for t in (r.get("failed_tool") or [])]
        + [str(x) for x in (record.get("failed_tool") or [])]
    )
    workflow_paths = _dedupe(
        [str(r.get("workflow_path") or "") for r in l1_records if r.get("workflow_path")]
        + ([str(record.get("workflow_path"))] if record.get("workflow_path") else [])
    )
    llm_files = record.get("files") or [
        {"file": r["file"], "failure_pattern": r.get("failure_pattern", ""),
         "failure_reason": r.get("failure_reason", "")}
        for r in l1_records if r.get("file")
    ]
    changed_files = _dedupe(_normalize_path(r.get("file", "")) for r in l1_records if r.get("file"))
    issue_ids     = _dedupe(str(r.get("issue_id") or "") for r in l1_records)

    print(f"  [L2] repo={repo_name} pattern={record.get('failure_pattern', '?')}")
    return {
        "sha_fail":        sha_fail,
        "issue_ids":       issue_ids,
        "repo":            repo_name,
        "repo_name":       repo_name,
        "error_type":      str(record.get("error_type")      or ""),
        "workflow_path":   workflow_paths[0] if workflow_paths else "",
        "failed_tool":     all_tools,
        "failed_cmd":      _dedupe([str(x) for x in (record.get("failed_cmd") or [])]),
        "failure_pattern": str(record.get("failure_pattern") or ""),
        "failure_reason":  _clip(str(record.get("failure_reason") or ""), 500),
        "fix_strategy":    _clip(str(record.get("fix_strategy") or ""), 500),
        "fix_pattern":     [str(x) for x in (record.get("fix_pattern") or [])],
        "files":           llm_files,
        "changed_files":   changed_files,
    }


# ── L3 extraction (per issue — generalizable principle) ───────────────────────

_L3_SCHEMA = """\
{
  "error_type":       "primary error type",
  "failed_tool":      ["tools that failed"],
  "principle":        "2-3 sentence GENERALIZABLE insight about this class of CI failure — what would apply to ANY repo encountering this error type",
  "fix_strategy":     "universal fix direction applicable across repos",
  "failure_patterns": ["distinct pattern names observed in this failure"],
  "failure_reasons":  ["1-2 representative root-cause descriptions from this failure"]
}"""


def _build_l3_prompt(repo_name: str, sha_fail: str, l2_record: Dict, l1_records: List[Dict]) -> str:
    l2_text = json.dumps(
        {k: v for k, v in l2_record.items() if k not in ("sha_fail", "issue_ids", "changed_files")},
        indent=2, ensure_ascii=False,
    )
    patterns = _dedupe(str(r.get("failure_pattern") or "") for r in l1_records if r.get("failure_pattern"))
    return f"""\
You are a CI failure principle analyst.

Given the L2 repo-level record and L1 file-level records for a single CI failure,
extract a GENERALIZABLE L3 principle that would apply to ANY repo encountering this type of failure.

Focus on what is UNIVERSAL — not repo-specific details.

════════════════════════════════
CONTEXT
════════════════════════════════
repo              : {repo_name}
sha_fail          : {sha_fail}
file_patterns_seen: {', '.join(patterns) or '(none)'}

════════════════════════════════
L2 REPO-LEVEL RECORD
════════════════════════════════
{_clip(l2_text, 3000)}

════════════════════════════════
TASK
════════════════════════════════
Extract ONE L3 cross-repo principle from this failure.
The principle and fix_strategy must be actionable for any repo with this error type.

Return ONLY this JSON (no markdown fences, no extra keys):
{_L3_SCHEMA}
""".strip()


def extract_l3_for_issue(
    repo_name: str, sha_fail: str, l2_record: Dict, l1_records: List[Dict], llm,
) -> Dict[str, Any]:
    prompt = _build_l3_prompt(repo_name, sha_fail, l2_record, l1_records)
    try:
        raw    = llm.invoke(prompt).content
        record = _parse_llm_json(raw)
    except Exception as exc:
        print(f"  [L3] LLM failed: {exc}")
        record = {}

    all_tools = _dedupe(
        [t for r in l1_records for t in (r.get("failed_tool") or [])]
        + [str(x) for x in (record.get("failed_tool") or [])]
    )
    fix_strategy = _clip(str(record.get("fix_strategy") or l2_record.get("fix_strategy") or ""), 500)
    llm_principle = str(record.get("principle") or "").strip()
    principle = (
        f"error_type={record.get('error_type') or l2_record.get('error_type', '')}: "
        f"{_clip(llm_principle, 400)} (seen in repo: {repo_name})"
        if llm_principle else
        f"error_type={l2_record.get('error_type', '')} (seen in repo: {repo_name})"
    )

    print(f"  [L3] error_type={record.get('error_type', '?')}")
    return {
        "sha_fail":         sha_fail,
        "repo":             repo_name,
        "repos":            [repo_name],
        "error_type":       str(record.get("error_type") or l2_record.get("error_type") or ""),
        "issue_type":       str(l1_records[0].get("issue_type") or "") if l1_records else "",
        "failure_pattern":  str(l1_records[0].get("failure_pattern") or "") if l1_records else "",
        "failed_tool":      all_tools,
        "principle":        _clip(principle, 600),
        "fix_strategy":     fix_strategy,
        "fix_strategies":   [fix_strategy] if fix_strategy else [],
        "failure_patterns": [str(x) for x in (record.get("failure_patterns") or [])],
        "failure_reasons":  [_clip(str(x), 300) for x in (record.get("failure_reasons") or [])[:3]],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build L1/L2/L3 memory bank from existing seed_log_details.json — one issue at a time, resumable."
    )
    parser.add_argument("--seed-file",     type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS_FILE)
    parser.add_argument("--config",        type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-key",     default=get_default_model_key())
    args = parser.parse_args()

    # Load seed issues (for diff + ground_truth_files)
    seed_rows    = json.loads(args.seed_file.read_text(encoding="utf-8"))
    # Load existing CILogAnalyzerLLM output
    analysis_rows = json.loads(args.analysis_file.read_text(encoding="utf-8"))
    analysis_by_sha = {str(r.get("sha_fail") or ""): r for r in analysis_rows}

    llm = get_llm(args.model_key)

    out_l1 = args.output_dir / "failure_memory.json"
    out_l2 = args.output_dir / "repo_memory.json"
    out_l3 = args.output_dir / "cross_memory.json"

    # Load existing records — resumable by sha_fail
    l1_all: List[Dict] = _load_json(out_l1)
    l2_all: List[Dict] = _load_json(out_l2)
    l3_all: List[Dict] = _load_json(out_l3)
    done_shas = {str(r.get("sha_fail") or "") for r in l2_all}  # L2 is one per issue

    total   = len(seed_rows)
    skipped = 0
    done    = 0

    print(f"\n=== Building memory bank ({total} seed issues) ===")
    print(f"Already done: {len(done_shas)} | Remaining: {total - len(done_shas)}\n")

    for i, seed in enumerate(seed_rows, 1):
        sha_fail      = str(seed.get("sha_fail")      or "")
        task_id       = str(seed.get("id")            or "")
        repo_name     = str(seed.get("repo_name")     or "")
        workflow_path = str(seed.get("workflow_path") or "")
        workflow_name = str(seed.get("workflow_name") or "")
        diff_text     = str(seed.get("diff")          or "")
        gt_files      = [_normalize_path(x) for x in (seed.get("ground_truth_files") or [])]

        if sha_fail in done_shas:
            print(f"[{i}/{total}] skip {sha_fail[:12]}... ({repo_name}) — already done")
            skipped += 1
            continue

        # Get the already-computed CILogAnalyzerLLM output
        wrapper  = analysis_by_sha.get(sha_fail, {})
        analysis = wrapper.get("analysis") or {}

        if not analysis:
            print(f"[{i}/{total}] skip {sha_fail[:12]}... ({repo_name}) — no log analysis found in seed_log_details.json")
            skipped += 1
            continue

        if not gt_files:
            print(f"[{i}/{total}] skip {sha_fail[:12]}... ({repo_name}) — no ground-truth files")
            skipped += 1
            continue

        error_context  = [str(x) for x in (analysis.get("error_context")   or []) if str(x).strip()]
        error_types    = analysis.get("error_types")    or []
        relevant_files = analysis.get("relevant_files") or []
        failed_job     = analysis.get("failed_job")     or []

        print(f"\n[{i}/{total}] {sha_fail[:12]}... ({repo_name}) — {len(gt_files)} files")

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
            print(f"  [L1] no records extracted — skipping L2/L3")
            skipped += 1
            continue

        # ── L2: repo-level view of this issue ─────────────────────────────────
        l2_record = extract_l2_for_issue(
            repo_name=repo_name, sha_fail=sha_fail, l1_records=l1_records, llm=llm,
        )
        l2_all.append(l2_record)
        _write_json(out_l2, l2_all)

        # ── L3: generalizable principle from this issue ───────────────────────
        l3_record = extract_l3_for_issue(
            repo_name=repo_name, sha_fail=sha_fail,
            l2_record=l2_record, l1_records=l1_records, llm=llm,
        )
        l3_all.append(l3_record)
        _write_json(out_l3, l3_all)

        done += 1
        print(f"  → saved L1({len(l1_records)}) + L2(1) + L3(1)  [total: L1={len(l1_all)} L2={len(l2_all)} L3={len(l3_all)}]")

    summary = {
        "model_key":      args.model_key,
        "seed_file":      str(args.seed_file),
        "analysis_file":  str(args.analysis_file),
        "total_issues":   total,
        "processed":      done,
        "skipped":        skipped,
        "memory_counts": {
            "L1_failure_memory": len(l1_all),
            "L2_repo_memory":    len(l2_all),
            "L3_cross_memory":   len(l3_all),
        },
    }
    _write_json(args.output_dir / "memory_bank_summary.json", summary)

    print(f"\n=== Done ===")
    print(f"Processed: {done}  Skipped: {skipped}")
    print(f"L1: {len(l1_all)} records | L2: {len(l2_all)} records | L3: {len(l3_all)} records")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
