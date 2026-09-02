"""
Unified Dependency Analyzer for All Three Decomposition Methods

This module provides a two-stage LLM approach:
1. Stage 1: Filter to caller-callee files from dependency graph
2. Stage 2: LLM summarizes the relevant files and their changes
3. Stage 3: LLM analyzes problem-level dependencies using the summary

This ensures all three methods (backward, forward, bidirectional) use the same
dependency analysis logic.
"""

from typing import Any, Dict, List
from pathlib import Path


def filter_relevant_files_from_graph(
    dependency_graph: Dict[str, Any],
    problems: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 1: Filter to only files that have caller-callee relationships.

    Args:
        dependency_graph: Output from build_dependency_graph()
        problems: List of problems being analyzed

    Returns:
        {
            "relevant_files": ["file1.py", "file2.py"],
            "edges": [{"from": "file1.py", "to": "file2.py", "type": "imports"}],
            "clusters": [[related files]],
            "problem_file_map": {1: ["file1.py"], 2: ["file2.py"]}
        }
    """
    nodes = dependency_graph.get("nodes", {})
    edges = dependency_graph.get("edges", [])
    clusters = dependency_graph.get("clusters", [])

    # Get all files affected by problems
    problem_files = set()
    problem_file_map = {}

    for problem in problems:
        problem_id = problem.get("problem_id", 0)
        affected = problem.get("affected_files", problem.get("files", []))
        problem_file_map[problem_id] = affected
        problem_files.update(affected)

    # Find all files that have caller-callee relationships with problem files
    relevant_files = set(problem_files)

    # Add files that are connected via edges
    for edge in edges:
        from_file = edge.get("from", "")
        to_file = edge.get("to", "")

        # If either end touches a problem file, include both
        if from_file in problem_files or to_file in problem_files:
            relevant_files.add(from_file)
            relevant_files.add(to_file)

    # Filter edges to only relevant files
    relevant_edges = [
        edge for edge in edges
        if edge.get("from") in relevant_files and edge.get("to") in relevant_files
    ]

    # Filter clusters to only relevant files
    relevant_clusters = [
        [f for f in cluster if f in relevant_files]
        for cluster in clusters
        if any(f in relevant_files for f in cluster)
    ]

    return {
        "relevant_files": sorted(relevant_files),
        "edges": relevant_edges,
        "clusters": relevant_clusters,
        "problem_file_map": problem_file_map,
        "total_files": len(relevant_files),
        "total_edges": len(relevant_edges),
    }


def get_file_changes_for_relevant_files(
    relevant_files: List[str],
    structured_diff: Dict[str, Any],
    repo_path: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Get file contents and changes for relevant files only.

    Args:
        relevant_files: List of files to fetch
        structured_diff: Parsed diff
        repo_path: Path to cloned repo (optional)

    Returns:
        {
            "file1.py": {
                "content": "full file content",
                "changes": [...],
                "summary": "what changed"
            }
        }
    """
    file_data = {}

    for file_info in structured_diff.get("files", []):
        file_path = file_info.get("path", "")

        if file_path not in relevant_files:
            continue  # Skip non-relevant files

        changes = file_info.get("changes", [])

        # Read file content if repo_path available
        content = ""
        if repo_path:
            try:
                full_path = Path(repo_path) / file_path
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8", errors="replace")[:200_000]
            except Exception as e:
                print(f"  Warning: Could not read {file_path}: {e}")

        # Summarize changes
        added_lines = sum(1 for c in changes if c.get("after") and not c.get("before"))
        removed_lines = sum(1 for c in changes if c.get("before") and not c.get("after"))
        modified_lines = sum(1 for c in changes if c.get("before") and c.get("after"))

        file_data[file_path] = {
            "content": content,
            "changes": changes,
            "summary": f"+{added_lines} -{removed_lines} ~{modified_lines} lines",
            "has_content": bool(content),
        }

    return file_data


def summarize_file_relationships_with_llm(
    relevant_graph: Dict[str, Any],
    file_data: Dict[str, Dict[str, Any]],
    llm: Any
) -> str:
    """
    Stage 2: Use LLM to summarize the relevant files and their relationships.

    This creates a concise summary that the dependency LLM can use.

    Args:
        relevant_graph: Filtered graph from filter_relevant_files_from_graph()
        file_data: File contents and changes from get_file_changes_for_relevant_files()
        llm: LLM instance

    Returns:
        Summary text describing file relationships and changes
    """
    # Build prompt for summarization
    prompt_parts = [
        "# File Relationship Summary Task",
        "",
        "Analyze the following files and their dependencies, then create a concise summary.",
        "",
        "## Files and Changes:",
        ""
    ]

    for file_path in relevant_graph["relevant_files"]:
        data = file_data.get(file_path, {})
        prompt_parts.append(f"### {file_path}")
        prompt_parts.append(f"Changes: {data.get('summary', 'unknown')}")

        # Include key imports/dependencies
        if data.get("has_content"):
            content = data["content"]
            # Extract first few imports
            import_lines = [line for line in content.split("\n")[:50] if "import " in line]
            if import_lines:
                prompt_parts.append("Key imports:")
                prompt_parts.extend(f"  - {line.strip()}" for line in import_lines[:5])

        prompt_parts.append("")

    prompt_parts.append("## File Dependencies:")
    prompt_parts.append("")

    for edge in relevant_graph["edges"]:
        from_file = edge["from"]
        to_file = edge["to"]
        edge_type = edge["type"]
        prompt_parts.append(f"- {from_file} --{edge_type}--> {to_file}")

    prompt_parts.append("")
    prompt_parts.append("## Dependency Clusters:")
    prompt_parts.append("")

    for i, cluster in enumerate(relevant_graph["clusters"], 1):
        prompt_parts.append(f"Cluster {i}: {', '.join(cluster)}")

    prompt_parts.extend([
        "",
        "## Task:",
        "Create a concise summary (max 500 words) explaining:",
        "1. What each file does",
        "2. How files depend on each other (caller-callee relationships)",
        "3. What changes were made and their potential impact",
        "4. Which files might affect which other files",
        "",
        "Focus on relationships that matter for understanding problem dependencies.",
        "",
        "Summary:"
    ])

    prompt = "\n".join(prompt_parts)

    # Call LLM for summary
    try:
        import time
        time.sleep(2)  # Rate limiting
        response = llm.invoke(prompt)

        if isinstance(response, dict):
            summary = response.get("content", str(response))
        else:
            summary = str(response)

        return summary.strip()

    except Exception as e:
        print(f"  Warning: LLM summarization failed: {e}")
        # Fallback: basic summary
        return f"Found {len(relevant_graph['relevant_files'])} related files with {len(relevant_graph['edges'])} dependencies."


def analyze_problem_dependencies_with_summary(
    problems: List[Dict[str, Any]],
    file_summary: str,
    relevant_graph: Dict[str, Any],
    llm: Any
) -> Dict[str, Any]:
    """
    Stage 3: Use file summary + graph to analyze problem-level dependencies.

    Args:
        problems: List of problems
        file_summary: Summary from Stage 2
        relevant_graph: Filtered graph
        llm: LLM instance

    Returns:
        {
            "dependencies": [{"from": 1, "to": 5, "reason": "..."}],
            "repair_sequence": [1, 5, 4, 2, 3]
        }
    """
    # Import the existing dependency analysis function
    try:
        from backward_decomposition.decompose_ci_failure import (
            _stage2_analyze_dependencies_and_sequence,
            build_full_dependency_prompt,
            _invoke_json,
            STRICT_JSON_RULES
        )
    except ImportError:
        print("  Warning: Could not import dependency analysis functions")
        return {"dependencies": [], "repair_sequence": [p.get("problem_id", i) for i, p in enumerate(problems, 1)]}

    # Build enhanced prompt with file summary
    problems_summary = []
    for idx, prob in enumerate(problems, 1):
        problems_summary.append({
            "problem_id": prob.get("problem_id", idx),
            "validation_order": prob.get("validation_order", "unknown"),
            "validation_cmd": prob.get("validation_cmd", "unknown"),
            "problem_type": prob.get("problem_type", "unknown"),
            "failure_type": prob.get("failure_type", "unknown"),
            "issue_type": prob.get("issue_type", "unknown"),
            "problem": prob.get("problem", ""),
            "root_cause": prob.get("root_cause", ""),
            "how_fixed": prob.get("how_fixed", ""),
            "why_fix_works": prob.get("why_fix_works", ""),
            "affected_files": prob.get("affected_files", []),
            "is_cascading": prob.get("is_cascading", False),
            "dependency_type": prob.get("dependency_type", ""),
            "cascade_explanation": prob.get("cascade_explanation", ""),
        })

    # Build graph info
    graph_info = {
        "validation_sequence": [],  # Will be filled from problems
        "file_relationships": relevant_graph.get("problem_file_map", {}),
        "file_summary": file_summary,  # NEW: Include the LLM summary
        "dependency_edges": relevant_graph.get("edges", []),
        "clusters": relevant_graph.get("clusters", []),
    }

    # Build prompt with file summary
    prompt = build_full_dependency_prompt(
        problems=problems_summary,
        graph_info=graph_info,
        strict_json_rules=STRICT_JSON_RULES,
    )

    # Add file summary section to prompt
    enhanced_prompt = f"""
{prompt}

## File Relationship Summary (from analysis):

{file_summary}

Use this summary to understand how files depend on each other, which helps identify problem dependencies.
"""

    try:
        import time
        time.sleep(3)  # Rate limiting
        response = _invoke_json(llm, enhanced_prompt)

        if isinstance(response, dict) and "dependencies" in response and "repair_sequence" in response:
            return response
        else:
            print(f"  Warning: Unexpected response format: {response}")
            return {
                "dependencies": [],
                "repair_sequence": [p.get("problem_id", i) for i, p in enumerate(problems, 1)]
            }

    except Exception as e:
        print(f"  Warning: Dependency analysis failed: {e}")
        return {
            "dependencies": [],
            "repair_sequence": [p.get("problem_id", i) for i, p in enumerate(problems, 1)]
        }


def unified_dependency_analysis(
    problems: List[Dict[str, Any]],
    dependency_graph: Dict[str, Any],
    structured_diff: Dict[str, Any],
    llm: Any,
    repo_path: str = None
) -> Dict[str, Any]:
    """
    Complete unified dependency analysis pipeline.

    This is the main entry point for all three decomposition methods.

    Args:
        problems: List of problems to analyze
        dependency_graph: Full dependency graph
        structured_diff: Parsed diff
        llm: LLM instance
        repo_path: Path to cloned repo (optional)

    Returns:
        {
            "dependencies": [...],
            "repair_sequence": [...],
            "file_summary": "...",
            "relevant_files": [...]
        }
    """
    print("  [Unified Dependency Analysis]")

    # Stage 1: Filter to relevant files
    print("    Stage 1: Filtering to caller-callee files...")
    relevant_graph = filter_relevant_files_from_graph(dependency_graph, problems)
    print(f"    → {relevant_graph['total_files']} relevant files, {relevant_graph['total_edges']} edges")

    # Stage 2: Get file data for relevant files
    print("    Stage 2: Fetching file contents and changes...")
    file_data = get_file_changes_for_relevant_files(
        relevant_graph["relevant_files"],
        structured_diff,
        repo_path
    )
    files_with_content = sum(1 for d in file_data.values() if d.get("has_content"))
    print(f"    → {len(file_data)} files fetched, {files_with_content} with content")

    # Stage 3: LLM summarizes file relationships
    print("    Stage 3: LLM summarizing file relationships...")
    file_summary = summarize_file_relationships_with_llm(
        relevant_graph,
        file_data,
        llm
    )
    print(f"    → Summary: {len(file_summary)} chars")

    # Stage 4: LLM analyzes problem dependencies
    print("    Stage 4: LLM analyzing problem dependencies...")
    dep_result = analyze_problem_dependencies_with_summary(
        problems,
        file_summary,
        relevant_graph,
        llm
    )
    print(f"    → {len(dep_result.get('dependencies', []))} dependencies found")

    # Return complete result
    return {
        **dep_result,
        "file_summary": file_summary,
        "relevant_files": relevant_graph["relevant_files"],
        "relevant_edges": relevant_graph["edges"],
        "relevant_clusters": relevant_graph["clusters"],
    }
