"""Shared CI failure and validation-sequence cache helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DETAILS_PATH = PROJECT_ROOT / "data" / "log_details.json"  # Standardized location
VALIDATION_CACHE_PATH = PROJECT_ROOT / "data" / "workflow_validation_cache.json"
EXCEPTION_DIR = PROJECT_ROOT / "exception"  # Error logs go here


def load_structured_ci_failure(
    sha_fail: str,
    issue_id: str = "",
    *,
    logs: Any = None,
    workflow: Any = None,
    workflow_path: str = "",
    llm: Any = None,
    model_name: str = "",
) -> Dict:
    """Load CI failure from cache, or generate it when inputs are available."""
    cached = find_cache_entry(LOG_DETAILS_PATH, sha_fail, issue_id)
    if cached:
        return cached

    if not (logs and workflow and llm):
        return {}

    try:
        from utilities.ci_log_analyzer import CILogAnalyzer

        analyzer = CILogAnalyzer(
            logs=logs,
            sha_fail=sha_fail,
            workflow=workflow,
            workflow_path=workflow_path,
            llm=llm,
            model_name=model_name,
            task_id=str(issue_id or sha_fail or "unknown"),
        )
        result = analyzer.run()
        if result:
            # Don't save error results to cache - log them to exception folder
            if "error" in result:
                _save_error_to_exception_folder(
                    issue_id=str(issue_id or sha_fail or "unknown"),
                    sha_fail=sha_fail,
                    error_data=result,
                )
                print(f"    ⚠️  CI log parsing failed for {issue_id}: {result.get('error')}")
                return {}  # Return empty dict, not error dict
            else:
                save_structured_ci_failure_cache(
                    issue_id=str(issue_id or sha_fail or "unknown"),
                    sha_fail=sha_fail,
                    structured_failure=result,
                )
        return result
    except Exception as e:
        print(f"    Warning: Could not generate structured CI failure: {e}")
        _save_error_to_exception_folder(
            issue_id=str(issue_id or sha_fail or "unknown"),
            sha_fail=sha_fail,
            error_data={"error": str(e), "traceback": __import__('traceback').format_exc()},
        )
        return {}


def load_validation_sequence(
    issue_id: str,
    sha_fail: str = "",
    *,
    workflow_content: str = "",
    workflow_path: str = "",
    repo_path: str = "",
    llm: Any = None,
    save: bool = False,
) -> Dict:
    """Load validation sequence from cache, or generate it from workflow content."""
    cached = find_cache_entry(VALIDATION_CACHE_PATH, sha_fail, issue_id)
    if cached:
        return validation_cache_result(cached, workflow_path)

    if not workflow_content:
        return empty_validation_result(workflow_path)

    try:
        from utilities.ci_workflow_aware_retrieval import (
            analyze_workflow_from_benchmark,
        )

        generated = analyze_workflow_from_benchmark(
            workflow_content=workflow_content,
            workflow_path=workflow_path,
            repo_path=repo_path,
            llm=llm,
            issue_id=issue_id,
            sha_fail=sha_fail,
        )
    except Exception as e:
        print(f"    Warning: Could not generate validation sequence: {e}")
        return empty_validation_result(workflow_path)

    # Check for errors in generated result
    if "error" in generated:
        _save_error_to_exception_folder(
            issue_id=issue_id,
            sha_fail=sha_fail,
            error_data=generated,
        )
        print(f"    ⚠️  Workflow validation failed for {issue_id}: {generated.get('error')}")
        return empty_validation_result(workflow_path)

    result = {
        "validation_sequence": generated.get("validation_sequence", []),
        "failure_info": generated.get("failure_info", ""),
        "workflow_path": generated.get("workflow_path", workflow_path),
    }
    if save:
        save_validation_sequence_cache(
            issue_id=issue_id,
            sha_fail=sha_fail,
            workflow_path=result["workflow_path"],
            validation_sequence=result["validation_sequence"],
            failure_info=result["failure_info"],
        )
    return result


def save_validation_sequence_cache(
    *,
    issue_id: str,
    sha_fail: str,
    workflow_path: str,
    validation_sequence: List[Dict],
    failure_info: str = "",
) -> None:
    """Save or replace one validation-sequence cache entry."""
    cache = [
        entry
        for entry in read_json_list(VALIDATION_CACHE_PATH)
        if not cache_entry_matches(entry, sha_fail, issue_id)
    ]
    cache.append(
        {
            "id": issue_id,
            "sha_fail": sha_fail,
            "workflow_path": workflow_path,
            "validation_sequence": validation_sequence,
            "failure_info": failure_info,
        }
    )
    VALIDATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def save_structured_ci_failure_cache(
    *,
    issue_id: str,
    sha_fail: str,
    structured_failure: Dict,
) -> None:
    """Save or replace one structured CI failure cache entry."""
    cache = [
        entry
        for entry in read_json_list(LOG_DETAILS_PATH)
        if not cache_entry_matches(entry, sha_fail, issue_id)
    ]
    entry = dict(structured_failure)
    entry.setdefault("id", issue_id)
    entry.setdefault("sha_fail", sha_fail)
    cache.append(entry)
    LOG_DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_DETAILS_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def find_cache_entry(path: Path, sha_fail: str, issue_id: str) -> Dict:
    """Return the first cache entry matching issue id or failing SHA."""
    for entry in read_json_list(path):
        if cache_entry_matches(entry, sha_fail, issue_id):
            return entry
    return {}


def read_json_list(path: Path) -> List[Dict]:
    """Read a JSON list cache. Missing or invalid caches behave like empty lists."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"    Warning: Could not read cache {path}: {e}")
        return []
    return data if isinstance(data, list) else []


def _save_error_to_exception_folder(
    *,
    issue_id: str,
    sha_fail: str,
    error_data: Dict,
) -> None:
    """Save error data to exception/ folder instead of cache."""
    EXCEPTION_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamped error file
    timestamp = int(time.time())
    error_file = EXCEPTION_DIR / f"ci_failure_{issue_id}_{sha_fail[:8]}_{timestamp}.json"

    with open(error_file, "w") as f:
        json.dump(
            {
                "issue_id": issue_id,
                "sha_fail": sha_fail,
                "timestamp": timestamp,
                **error_data,
            },
            f,
            indent=2,
        )
    print(f"    📝 Error logged to: {error_file.relative_to(PROJECT_ROOT)}")


def cache_entry_matches(entry: Dict, sha_fail: str, issue_id: str) -> bool:
    """Return whether a cache entry matches an issue id or failing SHA."""
    if not isinstance(entry, dict):
        return False
    entry_id = str(entry.get("id") or "")
    entry_sha = str(entry.get("sha_fail") or "")
    return bool(
        (sha_fail and entry_sha == str(sha_fail))
        or (issue_id and entry_id == str(issue_id))
    )


def validation_cache_result(entry: Dict, workflow_path: str = "") -> Dict:
    """Normalize a validation cache entry to the public return shape."""
    return {
        "validation_sequence": entry.get("validation_sequence", []),
        "failure_info": entry.get("failure_info") or entry.get("failure_summary", ""),
        "workflow_path": entry.get("workflow_path", workflow_path),
    }


def empty_validation_result(workflow_path: str = "") -> Dict:
    """Return an empty validation-cache result."""
    return {
        "validation_sequence": [],
        "failure_info": "",
        "workflow_path": workflow_path,
    }
