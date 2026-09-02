# utilities/prompt_builder.py
"""
Reusable prompt building utilities for CI analysis.

This module provides prompt templates and builders for:
1. File selection from commits
2. Commit analysis
3. CI failure diagnosis

All prompts follow best practices:
- Output format at the top with examples
- Clear, concise rules
- Minimal tokens
"""

from typing import Dict, List


class FileSelectionPromptBuilder:
    """Build prompts for selecting CI-relevant files from commits."""

    @staticmethod
    def build_full_diff_prompt(
        commit_metadata: Dict,
        changed_files: List[str],
        commit_diff: str,
        structured_ci_failure: Dict,
        relevant_validations: List[Dict],
        max_diff_tokens: int = 8000,
    ) -> str:
        """
        Build prompt with full diff for small commits.

        Args:
            commit_metadata: Commit info (sha, message, etc.)
            changed_files: List of changed file paths
            commit_diff: Full git diff
            structured_ci_failure: CI failure info
            relevant_validations: List of validation commands
            max_diff_tokens: Max tokens to include from diff

        Returns:
            Prompt string ready for LLM
        """
        # Format validations concisely
        validations = "\n".join(
            [
                f"  - {v.get('cmd', v.get('validation_cmd', 'unknown'))}"
                for v in relevant_validations[:10]
            ]
        )

        # Extract failure info
        failure_job = structured_ci_failure.get("failed_job", "unknown")
        failure_step = structured_ci_failure.get("failed_step", "unknown")
        failure_file = structured_ci_failure.get("file", "not specified")
        failure_error = structured_ci_failure.get("error_message", "")[:200]

        failure_info = f"""- Job: {failure_job}
- Step: {failure_step}
- File: {failure_file}
- Error: {failure_error}"""

        # Truncate diff if needed
        diff_text = commit_diff[: max_diff_tokens * 4]  # ~4 chars per token
        truncated = " ... (truncated)" if len(commit_diff) > len(diff_text) else ""

        # Format file list
        files_display = ", ".join(changed_files[:20])
        if len(changed_files) > 20:
            files_display += f" ... +{len(changed_files) - 20} more"

        return f"""Select files whose changes affect CI validation.

OUTPUT (JSON only):
{{
  "selected_groups": [
    {{
      "files": ["file.py"],
      "failure_type": "type_check|lint|test|format|build|install",
      "issue_type": "specific_issue",
      "validation_cmd": "command",
      "reason": "why"
    }}
  ]
}}

EXAMPLE:
{{
  "selected_groups": [
    {{
      "files": ["src/auth.py", "src/users.py"],
      "failure_type": "type_check",
      "issue_type": "missing_return_annotation",
      "validation_cmd": "mypy src/",
      "reason": "Both missing return annotations"
    }},
    {{
      "files": ["pyproject.toml"],
      "failure_type": "build",
      "issue_type": "dependency_version",
      "validation_cmd": "pip install -e .",
      "reason": "Updated dependencies"
    }}
  ]
}}

---

COMMIT: {commit_metadata.get("sha", "unknown")[:8]}
MESSAGE: {commit_metadata.get("message", "")[:100]}

FILES ({len(changed_files)}): {files_display}

DIFF:
{diff_text}{truncated}

CI FAILURE:
{failure_info}

VALIDATIONS:
{validations}

---

RULES:
1. File mentioned in CI FAILURE → SELECT
2. File validated by any command above → SELECT
   - "mypy src/" checks all .py files under src/
   - "pytest tests/" checks all test files
   - "black --check ." checks all Python files
3. Config file read by validations → SELECT
   - pyproject.toml, setup.py, setup.cfg
   - .pylintrc, mypy.ini, pytest.ini
4. Dependency files → SELECT
   - requirements.txt, *.lock
5. Docs-only, comments-only → SKIP

GROUPING:
- Same validation_cmd + issue_type → same group
- Different validation or issue → different groups

Output JSON only (no markdown, no text outside JSON):"""

    @staticmethod
    def build_chunked_prompt(
        commit_metadata: Dict,
        chunk_files: List[str],
        chunk_diff: str,
        chunk_num: int,
        total_chunks: int,
        structured_ci_failure: Dict,
        relevant_validations: List[Dict],
    ) -> str:
        """
        Build prompt for analyzing a chunk of files.

        Args:
            commit_metadata: Commit info
            chunk_files: Files in this chunk
            chunk_diff: Diff for just these files
            chunk_num: Current chunk number (1-indexed)
            total_chunks: Total number of chunks
            structured_ci_failure: CI failure info
            relevant_validations: Validation commands

        Returns:
            Prompt string
        """
        validations = "\n".join(
            [
                f"  - {v.get('cmd', v.get('validation_cmd', 'unknown'))}"
                for v in relevant_validations[:8]
            ]
        )

        failure_summary = (
            f"{structured_ci_failure.get('failed_step', 'unknown')} "
            f"in {structured_ci_failure.get('file', 'unknown')}"
        )

        return f"""Select CI-relevant files from chunk {chunk_num}/{total_chunks}.

OUTPUT JSON:
{{
  "selected_groups": [
    {{"files": ["f.py"], "failure_type": "test", "issue_type": "issue", "validation_cmd": "cmd", "reason": "why"}}
  ]
}}

FILES IN CHUNK: {", ".join(chunk_files)}

DIFF:
{chunk_diff[:6000]}

CI FAILURE: {failure_summary}

VALIDATIONS:
{validations}

RULES:
1. File in CI failure → SELECT
2. File validated by commands → SELECT
3. Config (*.toml, *.ini, setup.*) → SELECT
4. Otherwise → SKIP

Output JSON only:"""

    @staticmethod
    def build_summary_prompt(
        commit_metadata: Dict,
        changed_files: List[str],
        changes_summary: str,
        structured_ci_failure: Dict,
        relevant_validations: List[Dict],
    ) -> str:
        """
        Build prompt with summary (for very large commits).

        Args:
            commit_metadata: Commit info
            changed_files: List of all changed files
            changes_summary: Pre-computed summary string
            structured_ci_failure: CI failure info
            relevant_validations: Validation commands

        Returns:
            Prompt string
        """
        validations = "\n".join(
            [
                f"  - {v.get('cmd', v.get('validation_cmd', 'unknown'))}"
                for v in relevant_validations[:8]
            ]
        )

        failure_job = structured_ci_failure.get("failed_job", "unknown")
        failure_step = structured_ci_failure.get("failed_step", "unknown")
        failure_file = structured_ci_failure.get("file", "not specified")
        failure_error = structured_ci_failure.get("error_message", "")[:150]

        failure_info = f"""- Job: {failure_job}
- Step: {failure_step}
- File: {failure_file}
- Error: {failure_error}"""

        return f"""Select CI-relevant files.

OUTPUT JSON:
{{
  "selected_groups": [
    {{"files": ["f.py"], "failure_type": "lint", "issue_type": "issue", "validation_cmd": "cmd", "reason": "why"}}
  ]
}}

COMMIT: {commit_metadata.get("sha", "unknown")[:8]} ({len(changed_files)} files)

CHANGES:
{changes_summary}

CI FAILURE:
{failure_info}

VALIDATIONS:
{validations}

RULES:
1. File in failure → SELECT
2. File in validation scope → SELECT
3. Config (*.toml, *.ini) → SELECT
4. Dependencies (*.txt, *.lock) → SELECT

Output JSON only:"""


class CommitAnalysisPromptBuilder:
    """Build prompts for analyzing commits to identify CI problems."""

    @staticmethod
    def build_minimal_analysis_prompt(
        commit_sha: str,
        commit_message: str,
        changed_files: List[str],
        ci_failure_summary: str,
        validations: List[str],
    ) -> str:
        """
        Build a minimal commit analysis prompt.

        Use this for simple cases or when you need to reduce token usage.

        Args:
            commit_sha: Commit SHA
            commit_message: Commit message
            changed_files: List of changed files
            ci_failure_summary: Brief CI failure description
            validations: List of validation commands

        Returns:
            Prompt string
        """
        files_str = ", ".join(changed_files[:10])
        if len(changed_files) > 10:
            files_str += f" ... +{len(changed_files) - 10} more"

        validations_str = "\n".join([f"  - {v}" for v in validations[:5]])

        return f"""Analyze commit for CI problems.

OUTPUT JSON:
{{
  "commit_sha": "{commit_sha}",
  "problems": [
    {{
      "files": ["f.py"],
      "failure_type": "test",
      "issue_type": "assertion",
      "problem": "description",
      "root_cause": "cause",
      "fixed": true/false,
      "validation_cmd": "cmd"
    }}
  ]
}}

COMMIT: {commit_sha[:8]} - {commit_message[:80]}
FILES: {files_str}
CI FAILURE: {ci_failure_summary}
VALIDATIONS:
{validations_str}

Analyze and output JSON only:"""


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to max length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_validations(validations: List[Dict], max_count: int = 10) -> str:
    """
    Format validation list for prompts.

    Args:
        validations: List of validation dicts
        max_count: Maximum validations to include

    Returns:
        Formatted string like:
            - mypy src/
            - pytest tests/
            ...
    """
    lines = []
    for v in validations[:max_count]:
        cmd = v.get("cmd", v.get("validation_cmd", v.get("command", "unknown")))
        lines.append(f"  - {cmd}")

    if len(validations) > max_count:
        lines.append(f"  ... and {len(validations) - max_count} more")

    return "\n".join(lines)


def extract_failure_info(structured_ci_failure: Dict) -> Dict[str, str]:
    """
    Extract key failure info from structured failure dict.

    Args:
        structured_ci_failure: CI failure data

    Returns:
        Dict with: job, step, file, line, error
    """
    return {
        "job": structured_ci_failure.get("failed_job", "unknown"),
        "step": structured_ci_failure.get("failed_step", "unknown"),
        "file": structured_ci_failure.get("file", "not specified"),
        "line": str(structured_ci_failure.get("line", "not specified")),
        "error": truncate_text(structured_ci_failure.get("error_message", ""), 200),
    }
