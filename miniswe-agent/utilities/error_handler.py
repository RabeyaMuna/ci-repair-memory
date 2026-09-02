"""
Error handling utilities for saving errors to execption directory.

Provides consistent error tracking across the project.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


EXECPTION_DIR = Path("execption")


def save_error_to_execption(
    issue_id: str,
    error: Exception,
    error_type: str = "PROCESSING_ERROR",
    additional_context: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save error details to execption directory with issue ID as filename.

    Args:
        issue_id: Unique identifier for the issue/item that failed
        error: The exception that was raised
        error_type: Type of error (e.g., "DECOMPOSITION_ERROR", "LLM_ERROR")
        additional_context: Optional dict with extra context (sha_fail, repo, etc.)

    Returns:
        Path to the saved error file

    Example:
        try:
            process_issue(issue)
        except Exception as e:
            save_error_to_execption(
                issue_id="112",
                error=e,
                error_type="DECOMPOSITION_ERROR",
                additional_context={"sha_fail": issue.get("sha_fail")}
            )
    """
    # Create execption directory if it doesn't exist
    EXECPTION_DIR.mkdir(exist_ok=True)

    # Build error data
    error_data = {
        "error": error_type,
        "error_message": str(error),
        "error_trace": traceback.format_exc(),
        "error_type": type(error).__name__,
        "issue_id": issue_id,
    }

    # Add additional context if provided
    if additional_context:
        error_data.update(additional_context)

    # Save to file
    error_file = EXECPTION_DIR / f"{issue_id}.json"
    with open(error_file, "w") as f:
        json.dump(error_data, f, indent=2)

    return error_file


def load_error_from_execption(issue_id: str) -> Optional[Dict[str, Any]]:
    """
    Load error details from execption directory.

    Args:
        issue_id: Unique identifier for the issue

    Returns:
        Error data dict, or None if file doesn't exist
    """
    error_file = EXECPTION_DIR / f"{issue_id}.json"
    if not error_file.exists():
        return None

    with open(error_file, "r") as f:
        return json.load(f)


def has_error_in_execption(issue_id: str) -> bool:
    """
    Check if an error file exists for the given issue ID.

    Args:
        issue_id: Unique identifier for the issue

    Returns:
        True if error file exists, False otherwise
    """
    error_file = EXECPTION_DIR / f"{issue_id}.json"
    return error_file.exists()


def clear_error_from_execption(issue_id: str) -> bool:
    """
    Remove error file for the given issue ID.

    Args:
        issue_id: Unique identifier for the issue

    Returns:
        True if file was removed, False if it didn't exist
    """
    error_file = EXECPTION_DIR / f"{issue_id}.json"
    if error_file.exists():
        error_file.unlink()
        return True
    return False


def get_all_errors() -> Dict[str, Dict[str, Any]]:
    """
    Load all errors from execption directory.

    Returns:
        Dict mapping issue_id -> error_data
    """
    if not EXECPTION_DIR.exists():
        return {}

    errors = {}
    for error_file in EXECPTION_DIR.glob("*.json"):
        issue_id = error_file.stem
        try:
            with open(error_file, "r") as f:
                errors[issue_id] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {error_file}: {e}")

    return errors


def create_error_dict(
    error: Exception,
    error_type: str,
    issue_id: str,
    additional_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a standardized error dictionary without saving to file.

    Useful when you want to return error data but save it later.

    Args:
        error: The exception that was raised
        error_type: Type of error (e.g., "DECOMPOSITION_ERROR")
        issue_id: Unique identifier for the issue
        additional_context: Optional dict with extra context

    Returns:
        Error data dict
    """
    error_data = {
        "error": error_type,
        "error_message": str(error),
        "error_trace": traceback.format_exc(),
        "error_type": type(error).__name__,
        "issue_id": issue_id,
    }

    if additional_context:
        error_data.update(additional_context)

    return error_data
