"""
problem_consolidator.py
=======================
Consolidate multiple memory problems into coherent repair tasks.

Key Insight: Problems affecting the same files should be fixed together,
not independently, to avoid patch conflicts.
"""

from collections import defaultdict
from typing import Any, Dict, List


def consolidate_problems_by_file(problems: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group problems by affected files.

    Args:
        problems: List of L2 memory problems

    Returns:
        Dictionary mapping file_path -> list of problems affecting that file
    """
    file_to_problems = defaultdict(list)

    for problem in problems:
        # Get affected files from problem
        files = problem.get('files', [])
        if not isinstance(files, list):
            files = [files] if files else []

        # If no files specified, treat as global problem
        if not files:
            file_to_problems['_global_'].append(problem)
            continue

        # Add problem to each file it affects
        for file_path in files:
            if file_path and isinstance(file_path, str):
                file_to_problems[file_path].append(problem)

    return dict(file_to_problems)


def merge_problems_for_file(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple problems affecting the same file into one coherent description.

    Args:
        problems: List of problems affecting the same file

    Returns:
        Merged problem with combined descriptions
    """
    if len(problems) == 1:
        return problems[0]

    # Collect all unique aspects
    all_issues = []
    all_root_causes = []
    all_fixes = []
    all_validation_cmds = set()

    for problem in problems:
        # Problem description
        issue = problem.get('problem', problem.get('description', ''))
        if issue and issue not in all_issues:
            all_issues.append(issue)

        # Root cause
        root_cause = problem.get('root_cause', '')
        if root_cause and root_cause not in all_root_causes:
            all_root_causes.append(root_cause)

        # Fix approach
        fix = problem.get('how_fixed', problem.get('fix', ''))
        if fix and fix not in all_fixes:
            all_fixes.append(fix)

        # Validation command
        val_cmd = problem.get('validation_cmd', '')
        if val_cmd:
            all_validation_cmds.add(val_cmd)

    # Merge into single problem
    merged = {
        'problem': '\n'.join(f"{i+1}. {issue}" for i, issue in enumerate(all_issues)),
        'root_cause': '\n'.join(f"{i+1}. {cause}" for i, cause in enumerate(all_root_causes)),
        'how_fixed': '\n'.join(f"{i+1}. {fix}" for i, fix in enumerate(all_fixes)),
        'validation_cmd': ', '.join(sorted(all_validation_cmds)),
        'files': problems[0].get('files', []),  # Same file(s)
        'merged_count': len(problems),
        'source_problems': [p.get('problem_id', i) for i, p in enumerate(problems)]
    }

    return merged


def consolidate_for_agent_task(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Consolidate problems into a single coherent repair task for the agent.

    Strategy:
    1. Group problems by file
    2. Merge problems within each file
    3. Prioritize high-frequency files
    4. Create structured task description

    Args:
        problems: List of L2 memory problems (from retrieval)

    Returns:
        Consolidated task structure ready for agent
    """
    if not problems:
        return {
            'mode': 'empty',
            'problems': [],
            'file_groups': {},
            'total_files': 0,
            'total_problems': 0
        }

    # Step 1: Group by file
    file_to_problems = consolidate_problems_by_file(problems)

    # Step 2: Merge within each file
    file_groups = {}
    for file_path, file_problems in file_to_problems.items():
        merged = merge_problems_for_file(file_problems)
        file_groups[file_path] = {
            'merged_problem': merged,
            'original_problems': file_problems,
            'problem_count': len(file_problems)
        }

    # Step 3: Prioritize (files with more problems first)
    sorted_files = sorted(
        file_groups.items(),
        key=lambda x: x[1]['problem_count'],
        reverse=True
    )

    # Step 4: Build structured task
    consolidated = {
        'mode': 'consolidated',
        'file_groups': dict(sorted_files),
        'total_files': len(file_groups),
        'total_problems': len(problems),
        'high_priority_files': [
            file_path for file_path, group in sorted_files
            if group['problem_count'] >= 2
        ]
    }

    return consolidated


def build_agent_task_from_consolidated(consolidated: Dict[str, Any], current_failure: str) -> str:
    """
    Build agent task prompt from consolidated problems.

    Args:
        consolidated: Output from consolidate_for_agent_task()
        current_failure: Current CI failure description

    Returns:
        Task string for agent
    """
    if consolidated['mode'] == 'empty':
        return f"Fix the following CI failure:\n{current_failure}"

    file_groups = consolidated['file_groups']
    high_priority = consolidated.get('high_priority_files', [])

    # Build task description
    task_parts = [
        "Fix the following CI failure using guidance from past successful repairs:",
        "",
        "CURRENT FAILURE:",
        current_failure,
        "",
        "PAST REPAIR PATTERNS (apply relevant fixes):",
        ""
    ]

    # High-priority files first (multiple problems)
    if high_priority:
        task_parts.append("HIGH PRIORITY FILES (multiple issues):")
        for file_path in high_priority:
            group = file_groups[file_path]
            merged = group['merged_problem']
            count = group['problem_count']

            task_parts.append(f"\n{file_path} ({count} related issues):")
            task_parts.append(f"  Issues: {merged.get('problem', 'N/A')}")
            task_parts.append(f"  Root causes: {merged.get('root_cause', 'N/A')}")
            task_parts.append(f"  Fixes: {merged.get('how_fixed', 'N/A')}")

        task_parts.append("")

    # Other files
    other_files = [f for f in file_groups.keys() if f not in high_priority]
    if other_files:
        task_parts.append("OTHER RELEVANT FILES:")
        for file_path in other_files:
            group = file_groups[file_path]
            merged = group['merged_problem']

            task_parts.append(f"\n{file_path}:")
            task_parts.append(f"  Issue: {merged.get('problem', 'N/A')}")
            task_parts.append(f"  Fix: {merged.get('how_fixed', 'N/A')}")

    task_parts.extend([
        "",
        "INSTRUCTIONS:",
        "1. Apply ALL relevant fixes to their respective files in ONE unified patch",
        "2. Generate a SINGLE coherent diff (not separate patches per file)",
        "3. Ensure fixes are compatible with each other",
        "4. Test that all validation commands pass"
    ])

    return "\n".join(task_parts)


# Example usage in ci_context.py or ci_memory_system.py:
"""
# Instead of:
for problem in l2_problems:
    task = f"Fix: {problem['description']}"
    agent.run(task)  # Multiple patches!

# Do this:
consolidated = consolidate_for_agent_task(l2_problems)
task = build_agent_task_from_consolidated(consolidated, current_failure)
patch = agent.run(task)  # ONE coherent patch!
"""
