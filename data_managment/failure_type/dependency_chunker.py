#!/usr/bin/env python3
"""
Dependency-aware diff chunking.

Keeps related files (with import/call dependencies) in the same chunk
so the LLM can understand cascading failures better.
"""

import re
from typing import Dict, List, Set, Tuple


def extract_imports_from_diff(file_path: str, diff_content: str) -> Set[str]:
    """Extract imported modules from Python diff content."""
    imports = set()

    # Look for import statements in the diff
    for line in diff_content.split('\n'):
        if line.startswith('+') and ('import ' in line):
            # Match: import foo, from foo import bar, from .foo import bar
            if match := re.search(r'(?:from\s+[\.\w]+\s+)?import\s+([\w\s,]+)', line):
                imported = match.group(1)
                # Extract module names
                for module in imported.split(','):
                    module = module.strip().split(' as ')[0].strip()
                    if module and not module.startswith('_'):
                        imports.add(module)

    return imports


def build_dependency_graph(parsed_files: Dict[str, str]) -> Dict[str, Set[str]]:
    """Build dependency graph from parsed diff files.

    Returns: {file_path: set of files it depends on}
    """
    graph = {}
    file_modules = {}  # Map module names to file paths

    # First pass: map file paths to their module names
    for file_path in parsed_files.keys():
        if file_path.endswith('.py'):
            # Convert path to module name: src/foo/bar.py -> foo.bar
            module = file_path.replace('/', '.').replace('.py', '').strip('./')
            file_modules[module] = file_path
            # Also add the base module name
            if '.' in module:
                base = module.split('.')[-1]
                file_modules[base] = file_path

    # Second pass: find dependencies
    for file_path, diff_content in parsed_files.items():
        if not file_path.endswith('.py'):
            graph[file_path] = set()
            continue

        imports = extract_imports_from_diff(file_path, diff_content)
        dependencies = set()

        # Match imports to file paths
        for imp in imports:
            for module, dep_path in file_modules.items():
                if imp == module or imp.endswith('.' + module):
                    if dep_path != file_path:  # Don't self-depend
                        dependencies.add(dep_path)

        graph[file_path] = dependencies

    return graph


def cluster_by_dependencies(graph: Dict[str, Set[str]], parsed_files: Dict[str, str]) -> List[Set[str]]:
    """Cluster files by their dependencies using connected components."""

    visited = set()
    clusters = []

    def dfs(node: str, cluster: Set[str]):
        """Depth-first search to find connected component."""
        if node in visited:
            return
        visited.add(node)
        cluster.add(node)

        # Visit dependencies
        for dep in graph.get(node, set()):
            if dep in parsed_files:  # Only if file is in this diff
                dfs(dep, cluster)

        # Visit reverse dependencies (files that depend on this)
        for file_path, deps in graph.items():
            if node in deps and file_path not in visited:
                dfs(file_path, cluster)

    # Find all connected components
    for file_path in parsed_files.keys():
        if file_path not in visited:
            cluster = set()
            dfs(file_path, cluster)
            if cluster:
                clusters.append(cluster)

    return clusters


def chunk_diff_with_dependencies(diff: str, max_chars: int = 8000) -> List[Dict[str, any]]:
    """Chunk diff keeping dependent files together.

    Returns list of chunks, each with:
    - 'content': the diff content
    - 'files': list of files in this chunk
    - 'has_dependencies': whether files in chunk are related
    """

    # Parse diff into files
    parsed_files = {}
    current_file = None
    current_content = []

    for line in diff.split('\n'):
        if line.startswith('diff --git'):
            # Save previous file
            if current_file:
                parsed_files[current_file] = '\n'.join(current_content)

            # Extract new file path
            match = re.search(r'b/(.+?)(?:\s|$)', line)
            if match:
                current_file = match.group(1)
                current_content = [line]
        elif current_file:
            current_content.append(line)

    # Save last file
    if current_file:
        parsed_files[current_file] = '\n'.join(current_content)

    if not parsed_files:
        return [{'content': diff, 'files': [], 'has_dependencies': False}]

    # Build dependency graph
    dep_graph = build_dependency_graph(parsed_files)

    # Cluster by dependencies
    clusters = cluster_by_dependencies(dep_graph, parsed_files)

    # Create chunks from clusters
    chunks = []

    for cluster in clusters:
        cluster_files = sorted(cluster)
        cluster_content = '\n'.join(parsed_files[f] for f in cluster_files)

        # If cluster is too large, split it
        if len(cluster_content) <= max_chars:
            chunks.append({
                'content': cluster_content,
                'files': cluster_files,
                'has_dependencies': len(cluster_files) > 1
            })
        else:
            # Split large cluster by individual files
            for file_path in cluster_files:
                file_content = parsed_files[file_path]
                chunks.append({
                    'content': file_content,
                    'files': [file_path],
                    'has_dependencies': False
                })

    return chunks if chunks else [{'content': diff, 'files': [], 'has_dependencies': False}]
