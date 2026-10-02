"""
Pre-validation and conflict detection for CI repair problems.

This module implements critical optimizations:
1. Pre-validate problems (skip if already fixed)
2. Detect conflicting solutions in merged problems
3. Verify file existence before running agent
"""
import logging
import os
import subprocess
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def pre_validate_problem(
    problem: Dict[str, Any],
    repo_path: str,
    timeout: int = 60
) -> Tuple[bool, Optional[str]]:
    """
    Run validation BEFORE agent to check if problem already fixed.

    This is critical for L2/L3 problems from previous experience - they may
    already be fixed in the current SHA.

    Args:
        problem: Problem dictionary
        repo_path: Path to repo
        timeout: Timeout in seconds for validation

    Returns:
        (should_skip, reason)
        - If should_skip=True, problem is already fixed, skip it
        - reason contains explanation
    """
    validation_cmd = problem.get('verification_cmd', '')

    if not validation_cmd:
        # No validation command, can't pre-check
        return (False, None)

    problem_id = problem.get('problem_id', 'unknown')
    logger.info(f"[Pre-Validation] Problem {problem_id}: Running '{validation_cmd}'")

    try:
        result = subprocess.run(
            validation_cmd,
            shell=True,
            cwd=repo_path,
            capture_output=True,
            timeout=timeout,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"[Pre-Validation] OK Problem {problem_id}: ALREADY FIXED (validation passed)")
            return (True, "validation passed - problem already fixed in current SHA")
        else:
            logger.info(f"[Pre-Validation] FAIL Problem {problem_id}: Still exists (exit code {result.returncode})")
            logger.debug(f"[Pre-Validation] Output: {result.stdout[:200]}")
            return (False, None)

    except subprocess.TimeoutExpired:
        logger.warning(f"[Pre-Validation] Problem {problem_id}: Timeout after {timeout}s - will try agent fix")
        return (False, None)
    except Exception as e:
        logger.warning(f"[Pre-Validation] Problem {problem_id}: Error {e} - will try agent fix")
        return (False, None)


def validate_files_exist(
    problem: Dict[str, Any],
    repo_path: str
) -> Tuple[bool, List[str]]:
    """
    Check if all referenced files actually exist in the repo.

    L2/L3 problems may reference files that were renamed/deleted.

    Args:
        problem: Problem dictionary
        repo_path: Path to repo

    Returns:
        (all_exist, missing_files)
    """
    files = problem.get('files', [])

    if not files:
        # No files specified, can't validate
        return (True, [])

    # Normalize to list
    if not isinstance(files, list):
        files = [files]

    missing = []
    for file_entry in files:
        # Handle dict format: {"path": "..."} or {"file": "..."}
        if isinstance(file_entry, dict):
            file_path = file_entry.get('path') or file_entry.get('file')
        else:
            file_path = file_entry

        if not file_path:
            continue

        # Check if file exists
        full_path = os.path.join(repo_path, file_path)
        if not os.path.exists(full_path):
            missing.append(file_path)

    if missing:
        problem_id = problem.get('problem_id', 'unknown')
        logger.warning(f"[File Check] Problem {problem_id}: Missing files: {missing}")

    return (len(missing) == 0, missing)


def detect_conflicting_solutions(problem: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Detect if a problem has conflicting solutions.

    Example of conflicting solutions:
    - Solution A: "Change code to fix type annotation"
    - Solution B: "Change config to add mypy plugin"

    These are mutually exclusive and confuse the agent.

    Args:
        problem: Problem dictionary

    Returns:
        (has_conflict, conflict_description)
    """
    # Check problem statement for multiple solutions
    problem_statement = str(problem.get('problem_statement', ''))
    fix_strategy = str(problem.get('fix_strategy', ''))

    # Look for indicators of multiple solutions
    indicators = [
        'multiple related issues',
        'alternative fix',
        '1.' and '2.',  # Numbered lists
        'option 1' and 'option 2',
        'either' and 'or',
    ]

    has_multiple = False
    for indicator in indicators:
        if isinstance(indicator, tuple):
            # Check for both parts
            if all(part.lower() in problem_statement.lower() for part in indicator):
                has_multiple = True
                break
        elif indicator.lower() in problem_statement.lower():
            has_multiple = True
            break

    if not has_multiple:
        return (False, None)

    # Now check if solutions conflict
    code_keywords = ['change code', 'modify file', 'edit', 'replace type', 'fix annotation']
    config_keywords = ['change config', 'modify pyproject', 'add plugin', 'update toml']

    text_to_check = (problem_statement + ' ' + fix_strategy).lower()

    has_code_solution = any(kw in text_to_check for kw in code_keywords)
    has_config_solution = any(kw in text_to_check for kw in config_keywords)

    if has_code_solution and has_config_solution:
        conflict_desc = "Problem has both code-change and config-change solutions (mutually exclusive)"
        logger.warning(f"[Conflict Detection] {conflict_desc}")
        return (True, conflict_desc)

    return (False, None)


def should_split_merged_problem(
    problem: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Determine if a merged problem should be split back into separate problems.

    Merged problems with conflicting solutions should NOT have been merged.

    Args:
        problem: Problem dictionary

    Returns:
        (should_split, reason)
    """
    problem_id = str(problem.get('problem_id', ''))

    # Check if this is a merged problem
    if not problem_id.startswith('merged_'):
        return (False, None)

    # Check for conflicts
    has_conflict, conflict_desc = detect_conflicting_solutions(problem)

    if has_conflict:
        reason = f"Merged problem has conflicting solutions: {conflict_desc}"
        logger.warning(f"[Split Detection] Problem {problem_id} should be split: {reason}")
        return (True, reason)

    return (False, None)


def filter_and_validate_problems(
    problems: List[Dict[str, Any]],
    repo_path: str,
    enable_pre_validation: bool = True,
    enable_file_check: bool = True,
    enable_conflict_detection: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Filter and validate all problems before running agent.

    This implements critical optimizations:
    1. Skip problems that are already fixed (pre-validation)
    2. Skip problems with missing files
    3. Warn about conflicting solutions

    Args:
        problems: List of problem dictionaries
        repo_path: Path to repo
        enable_pre_validation: Run validation before agent
        enable_file_check: Check file existence
        enable_conflict_detection: Detect conflicting solutions

    Returns:
        (valid_problems, skip_reasons)
        - valid_problems: Problems that should run
        - skip_reasons: Dict mapping problem_id -> [reasons for skip]
    """
    valid_problems = []
    skip_reasons = {}

    logger.info(f"[Problem Validation] Filtering {len(problems)} problems...")

    for problem in problems:
        problem_id = str(problem.get('problem_id', 'unknown'))
        reasons = []

        # Check 1: File existence
        if enable_file_check:
            all_exist, missing = validate_files_exist(problem, repo_path)
            if not all_exist:
                reasons.append(f"Missing files: {missing}")

        # Check 2: Pre-validation (already fixed?)
        if enable_pre_validation and not reasons:  # Only if passed file check
            should_skip, reason = pre_validate_problem(problem, repo_path)
            if should_skip:
                reasons.append(reason)

        # Check 3: Conflicting solutions (warning only, don't skip)
        if enable_conflict_detection:
            has_conflict, conflict_desc = detect_conflicting_solutions(problem)
            if has_conflict:
                logger.warning(f"[Problem Validation] Problem {problem_id}: {conflict_desc}")
                logger.warning(f"[Problem Validation] Agent may get confused - consider splitting this problem")
                # Don't skip, just warn

        # Decide
        if reasons:
            skip_reasons[problem_id] = reasons
            logger.info(f"[Problem Validation] ⊘ Skip problem {problem_id}: {'; '.join(reasons)}")
        else:
            valid_problems.append(problem)

    skipped_count = len(problems) - len(valid_problems)
    logger.info(f"[Problem Validation] Result: {len(valid_problems)} valid, {skipped_count} skipped")

    return (valid_problems, skip_reasons)
