#!/usr/bin/env python3
"""
CI Workflow Patch Validator

Validates and filters workflow changes in patches:
1. Only validates if .github/workflows/* files are in the diff
2. Checks if changes are formatting-only (line-by-line)
3. Validates resulting YAML is not corrupted
4. Filters out invalid workflow changes, keeps rest of patch
"""

import re
import io
from typing import Tuple, List, Optional
from ruamel.yaml import YAML


def extract_workflow_files_from_diff(diff_text: str) -> List[str]:
    """
    Find all .github/workflows/*.y(a)ml files in the diff.

    Returns:
        List of workflow file paths found in diff
    """
    workflow_files = []
    pattern = r'diff --git a/(\.github/workflows/[^\s]+\.ya?ml) b/'

    for match in re.finditer(pattern, diff_text):
        workflow_path = match.group(1)
        workflow_files.append(workflow_path)

    return workflow_files


def extract_file_diff_section(diff_text: str, file_path: str) -> Optional[str]:
    """
    Extract the diff section for a specific file.

    Args:
        diff_text: Full git diff
        file_path: File to extract (e.g., '.github/workflows/tests.yml')

    Returns:
        Diff section for that file, or None if not found
    """
    # Escape special regex characters in the file path
    escaped_path = re.escape(file_path)

    # Pattern to match this file's diff section until next diff or end
    pattern = rf'diff --git a/{escaped_path} b/{escaped_path}.*?(?=^diff --git|\Z)'

    match = re.search(pattern, diff_text, re.MULTILINE | re.DOTALL)

    if match:
        return match.group(0)

    return None


def get_changed_lines_from_diff(diff_section: str) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Extract changed lines with line numbers from a diff section.

    Returns:
        (removed_lines, added_lines) as lists of (line_number, content) tuples
    """
    removed = []
    added = []

    current_removed_line = None
    current_added_line = None

    for line in diff_section.splitlines():
        # Parse hunk header: @@ -10,7 +10,7 @@
        if line.startswith('@@'):
            hunk_match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
            if hunk_match:
                current_removed_line = int(hunk_match.group(1))
                current_added_line = int(hunk_match.group(2))
            continue

        # Skip diff headers
        if line.startswith('diff --git') or line.startswith('index') or \
           line.startswith('---') or line.startswith('+++'):
            continue

        if current_removed_line is None or current_added_line is None:
            continue

        if line.startswith('-'):
            removed.append((current_removed_line, line[1:]))
            current_removed_line += 1
        elif line.startswith('+'):
            added.append((current_added_line, line[1:]))
            current_added_line += 1
        else:
            # Context line (no change)
            current_removed_line += 1
            current_added_line += 1

    return removed, added


def is_formatting_only_change(removed_lines: List[Tuple[int, str]],
                               added_lines: List[Tuple[int, str]]) -> Tuple[bool, str]:
    """
    Check if changes are formatting-only by comparing line content.

    Formatting-only means:
    - Same number of lines
    - Each line has same content when stripped (only whitespace differs)

    Args:
        removed_lines: List of (line_num, content) tuples
        added_lines: List of (line_num, content) tuples

    Returns:
        (is_formatting_only: bool, reason: str)
    """
    # Extract just the content (ignore line numbers for now)
    removed_content = [line for _, line in removed_lines]
    added_content = [line for _, line in added_lines]

    # Must have same number of changed lines
    if len(removed_content) != len(added_content):
        return False, f"Different number of lines: {len(removed_content)} removed vs {len(added_content)} added"

    # Compare each pair of lines
    for i, (removed, added) in enumerate(zip(removed_content, added_content)):
        removed_stripped = removed.strip()
        added_stripped = added.strip()

        # If stripped content differs, it's not formatting-only
        if removed_stripped != added_stripped:
            return False, f"Line {i+1} has content change: '{removed_stripped}' → '{added_stripped}'"

        # If stripped versions are same, the difference is only whitespace
        # This includes: trailing spaces, leading spaces, tabs vs spaces

    return True, "All changes are whitespace-only"


def apply_diff_to_content(original_content: str, diff_section: str) -> Optional[str]:
    """
    Apply a diff section to original content to get the modified content.

    Args:
        original_content: Original file content
        diff_section: Git diff section for this file

    Returns:
        Modified content, or None if application fails
    """
    removed_lines, added_lines = get_changed_lines_from_diff(diff_section)

    # Split original into lines
    original_lines = original_content.splitlines()

    # Build result by applying changes
    result_lines = original_lines.copy()

    # Group changes by line number
    # This is simplified - assumes non-overlapping hunks
    changes = {}
    for line_num, content in removed_lines:
        if line_num not in changes:
            changes[line_num] = {'removed': [], 'added': []}
        changes[line_num]['removed'].append(content)

    for line_num, content in added_lines:
        if line_num not in changes:
            changes[line_num] = {'removed': [], 'added': []}
        changes[line_num]['added'].append(content)

    # Apply changes (simplified - works for most cases)
    # For more complex diffs, we'd need proper patch application
    for line_num in sorted(changes.keys(), reverse=True):
        change = changes[line_num]
        idx = line_num - 1  # Convert to 0-based index

        if idx < len(result_lines):
            # Remove the old line
            if change['removed']:
                result_lines.pop(idx)

            # Insert new lines
            for added_line in reversed(change['added']):
                result_lines.insert(idx, added_line)

    return '\n'.join(result_lines)


def validate_yaml_syntax(content: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that content is valid YAML.

    Returns:
        (is_valid: bool, error_message: str or None)
    """
    try:
        yaml = YAML()
        yaml.load(io.StringIO(content))
        return True, None
    except Exception as e:
        return False, str(e)


def validate_workflow_change(
    diff_section: str,
    dataset_workflow_content: str,
    file_path: str
) -> dict:
    """
    Validate a workflow file change in a diff.

    Checks:
    1. Are changes formatting-only? (line-by-line comparison)
    2. Is resulting workflow valid YAML?

    Args:
        diff_section: Git diff section for the workflow file
        dataset_workflow_content: Original workflow from dataset
        file_path: Path to the workflow file

    Returns:
        {
            'valid': bool,
            'is_formatting_only': bool,
            'is_valid_yaml': bool,
            'reason': str,
            'modified_content': str or None
        }
    """
    result = {
        'valid': False,
        'is_formatting_only': False,
        'is_valid_yaml': False,
        'reason': '',
        'modified_content': None
    }

    # Step 1: Extract changed lines
    removed_lines, added_lines = get_changed_lines_from_diff(diff_section)

    if not removed_lines and not added_lines:
        # No actual changes (shouldn't happen, but handle it)
        result['valid'] = True
        result['is_formatting_only'] = True
        result['is_valid_yaml'] = True
        result['reason'] = "No changes detected"
        return result

    # Step 2: Check if formatting-only
    is_formatting, format_reason = is_formatting_only_change(removed_lines, added_lines)
    result['is_formatting_only'] = is_formatting

    if not is_formatting:
        result['reason'] = f"Not formatting-only: {format_reason}"
        return result

    # Step 3: Apply changes and validate YAML
    # Try to reconstruct the modified content
    try:
        modified_content = apply_diff_to_content(dataset_workflow_content, diff_section)
        result['modified_content'] = modified_content

        if modified_content:
            is_valid_yaml, yaml_error = validate_yaml_syntax(modified_content)
            result['is_valid_yaml'] = is_valid_yaml

            if not is_valid_yaml:
                result['reason'] = f"Resulting YAML is corrupted: {yaml_error}"
                return result

    except Exception as e:
        result['reason'] = f"Failed to apply diff: {e}"
        return result

    # All checks passed
    result['valid'] = True
    result['reason'] = "Formatting-only change, valid YAML"

    return result


def remove_file_from_diff(diff_text: str, file_path: str) -> str:
    """
    Remove a file's diff section from the full diff.

    Args:
        diff_text: Full git diff
        file_path: File to remove

    Returns:
        Filtered diff without this file
    """
    file_diff_section = extract_file_diff_section(diff_text, file_path)

    if not file_diff_section:
        return diff_text

    # Remove this section from the diff
    filtered_diff = diff_text.replace(file_diff_section, '')

    # Clean up any double newlines
    filtered_diff = re.sub(r'\n\n\n+', '\n\n', filtered_diff)

    return filtered_diff


def validate_and_filter_workflow_changes(
    diff_text: str,
    datapoint: dict
) -> Tuple[str, dict]:
    """
    Main validation function: validates workflow changes and filters invalid ones.

    IMPORTANT: Only validates the SPECIFIC workflow from the dataset.
    Other workflow files in the diff are left as-is.

    Workflow:
    1. Check if the DATASET workflow (datapoint['workflow_path']) is in the diff
    2. If yes, validate it
    3. If invalid, remove from diff
    4. Return filtered diff and validation report

    Args:
        diff_text: Full patch/diff text
        datapoint: Dataset entry with 'workflow' and 'workflow_path'

    Returns:
        (filtered_diff: str, validation_report: dict)
    """
    report = {
        'had_workflow_changes': False,
        'dataset_workflow_in_diff': False,
        'workflow_files_in_diff': [],
        'valid_workflows': [],
        'invalid_workflows': [],
        'filtered_diff_modified': False
    }

    # Get the SPECIFIC workflow from dataset
    dataset_workflow_path = datapoint.get('workflow_path', '')
    dataset_workflow_content = datapoint.get('workflow', '')

    if not dataset_workflow_path or not dataset_workflow_content:
        # No workflow in dataset → use diff as-is
        return diff_text, report

    # Step 1: Find ALL workflow files in diff (for reporting)
    workflow_files = extract_workflow_files_from_diff(diff_text)
    report['workflow_files_in_diff'] = workflow_files

    if not workflow_files:
        # No workflow changes → use diff as-is
        return diff_text, report

    report['had_workflow_changes'] = True

    # Step 2: Check if the DATASET workflow is in the diff
    if dataset_workflow_path not in workflow_files:
        # Diff contains OTHER workflows, but not the dataset one
        # → Leave diff as-is (don't validate other workflows)
        print(f"[VALIDATION] Diff contains workflows {workflow_files}, but not dataset workflow {dataset_workflow_path}")
        print(f"[VALIDATION] Skipping validation (only validate dataset workflow)")
        return diff_text, report

    # Dataset workflow IS in the diff → validate it
    report['dataset_workflow_in_diff'] = True
    filtered_diff = diff_text

    # Step 3: Validate ONLY the dataset workflow
    print(f"[VALIDATION] Validating dataset workflow: {dataset_workflow_path}")

    # Extract diff section for the dataset workflow
    diff_section = extract_file_diff_section(diff_text, dataset_workflow_path)

    if not diff_section:
        print(f"[WARN] Could not extract diff section for {dataset_workflow_path}")
        return diff_text, report

    # Validate this workflow change
    validation = validate_workflow_change(
        diff_section,
        dataset_workflow_content,
        dataset_workflow_path
    )

    if validation['valid']:
        print(f"  ✓ Valid: {validation['reason']}")
        report['valid_workflows'].append({
            'file': dataset_workflow_path,
            'reason': validation['reason']
        })
    else:
        print(f"  ✗ Invalid: {validation['reason']}")
        print(f"  → Removing {dataset_workflow_path} changes from patch")
        print(f"  → Dataset workflow will be used as-is")

        report['invalid_workflows'].append({
            'file': dataset_workflow_path,
            'reason': validation['reason'],
            'is_formatting_only': validation['is_formatting_only'],
            'is_valid_yaml': validation['is_valid_yaml']
        })

        # Remove the dataset workflow from the diff
        # Keep other files (including other workflows if any)
        filtered_diff = remove_file_from_diff(filtered_diff, dataset_workflow_path)
        report['filtered_diff_modified'] = True

    return filtered_diff, report


# Example usage
if __name__ == '__main__':
    # Test with example diff
    test_diff = """diff --git a/.github/workflows/tests.yml b/.github/workflows/tests.yml
index abc123..def456 100644
--- a/.github/workflows/tests.yml
+++ b/.github/workflows/tests.yml
@@ -6,7 +6,7 @@ jobs:
     steps:
-      - name: Run tests
+      - name: Run tests
         run: pytest
diff --git a/src/main.py b/src/main.py
index 111222..333444 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
 def main():
-    print("hello")
+    print("Hello, World!")
"""

    test_datapoint = {
        'workflow_path': '.github/workflows/tests.yml',
        'workflow': """name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        run: pytest
"""
    }

    filtered, report = validate_and_filter_workflow_changes(test_diff, test_datapoint)

    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    print(f"Had workflow changes: {report['had_workflow_changes']}")
    print(f"Files checked: {report['workflow_files_checked']}")
    print(f"Valid: {len(report['valid_workflows'])}")
    print(f"Invalid: {len(report['invalid_workflows'])}")
    print(f"Diff modified: {report['filtered_diff_modified']}")

    if report['invalid_workflows']:
        print("\nInvalid workflows:")
        for w in report['invalid_workflows']:
            print(f"  - {w['file']}: {w['reason']}")
