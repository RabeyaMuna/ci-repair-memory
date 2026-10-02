"""Shared dependency graph evidence helpers for CI decomposition."""

from __future__ import annotations

from typing import Any

try:
    from scripts.dependency_detector import build_dependency_graph
except ImportError:
    from dependency_detector import build_dependency_graph

from utilities.deterministic_diff_parser import parse_diff_to_structured


def build_dependency_graph_for_diff(
    diff: str,
    *,
    repo_path: str | None = None,
) -> dict[str, Any]:
    """Parse a raw git diff and build dependency_detector's file graph."""
    structured_diff = parse_diff_to_structured(diff or "")
    return build_dependency_graph(structured_diff, repo_path=repo_path)


def build_dependency_graph_from_structured_diff(
    structured_diff: dict[str, Any],
    *,
    repo_path: str | None = None,
) -> dict[str, Any]:
    """Build dependency_detector's file graph from an existing structured diff."""
    return build_dependency_graph(structured_diff, repo_path=repo_path)


def dependency_graph_evidence(
    diff: str,
    *,
    repo_path: str | None = None,
    max_edges: int = 100,
    max_clusters: int = 50,
) -> dict[str, Any]:
    """Return prompt-ready dependency evidence for a raw git diff."""
    graph = build_dependency_graph_for_diff(diff, repo_path=repo_path)
    return summarize_dependency_graph(
        graph, max_edges=max_edges, max_clusters=max_clusters
    )


def summarize_dependency_graph(
    graph: dict[str, Any],
    *,
    max_edges: int = 100,
    max_clusters: int = 50,
) -> dict[str, Any]:
    """Bound dependency graph details for LLM context."""
    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    clusters = graph.get("clusters", []) if isinstance(graph, dict) else []

    return {
        "source": "dependency_detector.build_dependency_graph",
        "limits": (
            "Dependencies are inferred from changed files, changed hunks, and "
            "repository file contents when repo_path is available. Edges are "
            "heuristic and should be verified against the diff and CI commands."
        ),
        "available": bool(nodes),
        "files": list(nodes.keys()),
        "nodes": nodes,
        "edges": edges[:max_edges],
        "clusters": clusters[:max_clusters],
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_clusters": len(clusters),
        "truncated": len(edges) > max_edges or len(clusters) > max_clusters,
    }
