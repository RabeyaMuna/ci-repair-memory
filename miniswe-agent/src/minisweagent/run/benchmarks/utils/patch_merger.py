"""Validate patches and compose repeated file changes with Git's three-way merge.

The final checkout is the preferred source of a patch. Legacy patch sequences
require a repository and an explicit base when HEAD is not the failed commit.
No hunk rewriting, heuristic filtering, or partial-success fallback is used.
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from minisweagent.run.benchmarks.utils.git_patch import (
    PatchValidationError,
    check_round_trip,
    diff_trees,
    patch_repository,
    run_git,
    validate_against_commit,
)

logger = logging.getLogger(__name__)


def parse_unified_diff(diff_text: str) -> List[Dict[str, str]]:
    """Split complete records without changing their contents; Git parses paths."""
    if not diff_text.strip():
        return []
    starts = [m.start() for m in re.finditer(r"^diff --git ", diff_text, re.MULTILINE)]
    if starts and diff_text[:starts[0]].strip():
        raise PatchValidationError("Unexpected content before the Git diff")
    if not starts:
        starts = [0]
    starts.append(len(diff_text))
    patches = []
    with tempfile.TemporaryDirectory(prefix="ci-patch-parse-") as directory:
        for start, end in zip(starts, starts[1:]):
            block = diff_text[start:end]
            stats = run_git(Path(directory), "apply", "--numstat", "-z", input=block.encode("utf-8")).stdout
            records = [record for record in stats.split(b"\0") if record]
            if len(records) != 1:
                raise PatchValidationError("Expected one complete file change per diff record")
            path = os.fsdecode(records[0].split(b"\t", 2)[2])
            hunk = re.search(r"^@@", block, re.MULTILINE)
            split = hunk.start() if hunk else len(block)
            patches.append({"file": path, "header": block[:split], "hunks": block[split:], "full_patch": block})
    return patches


def group_patches_by_file(patches: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped = {}
    for patch in patches:
        grouped.setdefault(patch["file"], []).append(patch)
    return grouped


def detect_duplicate_patches(diff_text: str) -> Dict[str, int]:
    return {path: len(patches) for path, patches in group_patches_by_file(parse_unified_diff(diff_text)).items()}


def filter_corrupted_patches(diff_text: str) -> str:
    """Compatibility entry point: reject malformed input, never discard changes."""
    parse_unified_diff(diff_text)
    return diff_text


def merge_duplicate_patches(diff_text: str, repo_path: Path = None, base_commit: str = "HEAD") -> str:
    """Compose ordered patches on the failed commit, rejecting unresolved conflicts.

    Git can combine sequential, cumulative and independent changes using their
    base blobs. Missing blobs or ambiguous conflicts are errors; callers must
    preserve the inputs and have the agent resolve the checkout before retrying.
    """
    if not diff_text.strip():
        return diff_text
    patches = parse_unified_diff(diff_text)
    groups = group_patches_by_file(patches)
    if all(len(group) == 1 for group in groups.values()):
        if repo_path is not None:
            validate_against_commit(diff_text, repo_path, base_commit)
        return diff_text
    if repo_path is None:
        raise PatchValidationError("Merging duplicate patches requires a repository at the failed commit")
    with patch_repository(repo_path, base_commit) as (isolated, base):
        for number, patch in enumerate(patches, 1):
            previous = run_git(isolated, "write-tree").stdout.strip().decode()
            raw = patch["full_patch"].encode("utf-8")
            applied = run_git(isolated, "apply", "--cached", "--3way", "--whitespace=nowarn", input=raw, check=False)
            if applied.returncode:
                # A conflicting three-way apply can leave unmerged index entries.
                run_git(isolated, "read-tree", previous)
                already_applied = run_git(
                    isolated, "apply", "--cached", "--reverse", "--check", "--whitespace=nowarn",
                    input=raw, check=False,
                )
                if already_applied.returncode:
                    error = applied.stderr.decode("utf-8", errors="replace").strip()
                    raise PatchValidationError(f"Patch {number} ({patch['file']}) could not be merged: {error}")
        target = run_git(isolated, "write-tree").stdout.strip().decode()
        merged = diff_trees(isolated, base, target)
        check_round_trip(isolated, base, merged, target)
        return merged


def validate_patch(diff_text: str, repo_path: Path = None, base_commit: str = "HEAD") -> Tuple[bool, str]:
    """Check syntax; with a repository, also apply against the specified commit."""
    if not diff_text.strip():
        return False, "Empty patch"
    try:
        duplicates = detect_duplicate_patches(diff_text)
        repeated = [path for path, count in duplicates.items() if count > 1]
        if repeated:
            return False, f"Duplicate patches for files: {', '.join(repeated)}"
        if repo_path is not None:
            validate_against_commit(diff_text, repo_path, base_commit)
    except (PatchValidationError, UnicodeError) as exc:
        return False, str(exc)
    return True, "OK"


def diagnose_patch_failure(
    diff_text: str, git_error: str, repo_path: Path = None
) -> Dict[str, Any]:
    """
    Diagnose why a patch failed to apply by analyzing the git error and patch content.

    Args:
        diff_text: The patch that failed
        git_error: Error message from git apply
        repo_path: Optional path to repository for file checks

    Returns:
        Dictionary with diagnosis information:
        - failure_type: "missing_file" | "context_mismatch" | "corrupt_patch" | "merge_conflict"
        - affected_files: List of files mentioned in error
        - diagnosis: Human-readable explanation
        - suggested_fix: What might resolve the issue
    """
    diagnosis = {
        "failure_type": "unknown",
        "affected_files": [],
        "diagnosis": "",
        "suggested_fix": "",
    }

    git_error_lower = git_error.lower()

    # Pattern 1: File doesn't exist in index
    if "does not exist in index" in git_error_lower:
        diagnosis["failure_type"] = "missing_file"
        # Extract file names from error
        import re
        file_matches = re.findall(r"error: ([^\s:]+): does not exist in index", git_error)
        diagnosis["affected_files"] = file_matches
        diagnosis["diagnosis"] = (
            f"Agent created new file(s) but patch doesn't include 'new file mode' header. "
            f"Files: {', '.join(file_matches)}"
        )
        diagnosis["suggested_fix"] = (
            "Use 'git add -N .' before generating diff to mark new files, "
            "or ensure patches include proper 'new file mode 100644' headers"
        )

    # Pattern 2: Patch doesn't apply (context mismatch)
    elif "patch does not apply" in git_error_lower or "patch failed:" in git_error_lower:
        diagnosis["failure_type"] = "context_mismatch"
        # Extract file and line info
        import re
        file_matches = re.findall(r"error: patch failed: ([^\s:]+):(\d+)", git_error)
        diagnosis["affected_files"] = [f[0] for f in file_matches]
        diagnosis["diagnosis"] = (
            f"Patch context lines don't match file content at target commit. "
            f"Files: {', '.join(diagnosis['affected_files'])}"
        )
        diagnosis["suggested_fix"] = (
            "Ensure agent reads current file state before editing. "
            "Agent may have stale view or file changed between analysis and edit. "
            "Consider generating patches with more context lines (git diff -U10)"
        )

    # Pattern 3: Corrupt patch / missing header
    elif "corrupt patch" in git_error_lower or "patch fragment without header" in git_error_lower:
        diagnosis["failure_type"] = "corrupt_patch"
        diagnosis["diagnosis"] = (
            "Patch structure is malformed - missing 'diff --git' header before hunks"
        )
        diagnosis["suggested_fix"] = (
            "Check patch merger logic for proper header/hunk concatenation. "
            "Regenerate the diff from the checkout and validate against the failed commit"
        )

        # Try to extract which file from the patch itself
        try:
            patches = parse_unified_diff(diff_text)
        except (PatchValidationError, UnicodeError):
            patches = []
        if patches:
            diagnosis["affected_files"] = [p["file"] for p in patches]

    # Pattern 4: Missing blob for 3-way merge
    elif "lacks the necessary blob" in git_error_lower:
        diagnosis["failure_type"] = "missing_blob"
        diagnosis["diagnosis"] = (
            "Repository missing git objects for 3-way merge. "
            "This happens with --shared clones that don't have full history"
        )
        diagnosis["suggested_fix"] = (
            "Don't use 'git clone --shared' for testbed, or fetch full objects: "
            "'git fetch origin <sha_fail>' before applying patch"
        )

    # Pattern 5: File doesn't match index
    elif "does not match index" in git_error_lower:
        diagnosis["failure_type"] = "index_mismatch"
        import re
        file_matches = re.findall(r"error: ([^\s:]+): does not match index", git_error)
        diagnosis["affected_files"] = file_matches
        diagnosis["diagnosis"] = (
            f"File content differs from what's in git index. "
            f"Files: {', '.join(file_matches)}"
        )
        diagnosis["suggested_fix"] = (
            "File may have unstaged changes or the patch expects a different base state. "
            "Ensure clean checkout at exact sha_fail before applying"
        )

    # If we have repo_path, check if files actually exist
    if repo_path and diagnosis["affected_files"]:
        existing = []
        missing = []
        for file_path in diagnosis["affected_files"]:
            full_path = repo_path / file_path
            if full_path.exists():
                existing.append(file_path)
            else:
                missing.append(file_path)

        if missing:
            diagnosis["diagnosis"] += f"\n  Missing files: {', '.join(missing)}"
        if existing:
            diagnosis["diagnosis"] += f"\n  Existing files: {', '.join(existing)}"

    return diagnosis
