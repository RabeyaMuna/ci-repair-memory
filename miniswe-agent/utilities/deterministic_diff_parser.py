#!/usr/bin/env python3
"""
Deterministic Diff Parser - Parse git diff into structured data

Instead of sending raw diff to LLM, pre-process it into clean JSON structure.
This prevents JSON parsing errors from code/quotes in strings.

Usage:
    from deterministic_diff_parser import parse_diff_to_structured

    structured = parse_diff_to_structured(diff_text)
    # structured = {
    #   "files": [
    #     {
    #       "path": "path/to/file.py",
    #       "changes": [
    #         {"line": 42, "before": "old code", "after": "new code"}
    #       ]
    #     }
    #   ]
    # }
"""

import re
from typing import Any

MAX_CHANGE_TOKENS_PER_CHUNK = 70_000


def parse_diff_to_structured(diff: str) -> dict[str, Any]:
    """
    Parse git diff into structured format.

    Returns:
        {
          "files": [
            {
              "path": "path/to/file.py",
              "file_type": ".py",
              "changes": [
                {
                  "line": 42,
                  "before": "old code",
                  "after": "new code",
                  "change_type": "modified"  # or "added", "deleted"
                }
              ],
              "total_changes": 5
            }
          ],
          "total_files": 10,
          "total_changes": 150
        }
    """
    files = []
    current_file = None
    current_hunk_start = 0

    lines = diff.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # New file header
        if line.startswith("diff --git"):
            # Save previous file if exists
            if current_file:
                files.append(current_file)

            # Extract file path
            match = re.search(r"diff --git a/(.*?) b/", line)
            if match:
                file_path = match.group(1)
                file_type = _get_file_extension(file_path)
                current_file = {
                    "path": file_path,
                    "file_type": file_type,
                    "changes": [],
                    "total_changes": 0,
                }

        # Hunk header (@@ -X,Y +A,B @@)
        elif line.startswith("@@"):
            match = re.match(r"@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
            if match:
                current_hunk_start = int(match.group(2))  # New line number

        # Changed lines
        elif current_file:
            if line.startswith("-") and not line.startswith("---"):
                # Deletion
                before_code = line[1:]  # Remove '-' prefix

                # Check if next line is addition (modification)
                if (
                    i + 1 < len(lines)
                    and lines[i + 1].startswith("+")
                    and not lines[i + 1].startswith("+++")
                ):
                    after_code = lines[i + 1][1:]  # Remove '+' prefix
                    current_file["changes"].append(
                        {
                            "line": current_hunk_start,
                            "before": before_code,
                            "after": after_code,
                            "change_type": "modified",
                        }
                    )
                    current_file["total_changes"] += 1
                    i += 1  # Skip next line since we processed it
                else:
                    # Pure deletion
                    current_file["changes"].append(
                        {
                            "line": current_hunk_start,
                            "before": before_code,
                            "after": "",
                            "change_type": "deleted",
                        }
                    )
                    current_file["total_changes"] += 1

            elif line.startswith("+") and not line.startswith("+++"):
                # Addition (not paired with deletion above)
                after_code = line[1:]
                current_file["changes"].append(
                    {
                        "line": current_hunk_start,
                        "before": "",
                        "after": after_code,
                        "change_type": "added",
                    }
                )
                current_file["total_changes"] += 1
                current_hunk_start += 1

            elif not line.startswith(("-", "+", "@", "diff", "index", "---", "+++")):
                # Context line (unchanged)
                current_hunk_start += 1

        i += 1

    # Save last file
    if current_file:
        files.append(current_file)

    total_changes = sum(f["total_changes"] for f in files)

    return {"files": files, "total_files": len(files), "total_changes": total_changes}


def _get_file_extension(file_path: str) -> str:
    """Extract file extension."""
    if "." in file_path:
        return "." + file_path.rsplit(".", 1)[1]
    return ""


def chunk_structured_diff(
    structured: dict[str, Any],
    max_files_per_chunk: int = 10,
    dependency_graph: dict[str, Any] = None,
    model_name: str = None,
) -> list[dict[str, Any]]:
    """
    Split structured diff into chunks.

    NEW: Token-aware + Dependency-aware chunking!

    Strategy:
    1. If dependency_graph provided, use token + dependency-aware chunking
    2. Otherwise, fall back to simple file-count chunking

    Args:
        structured: Output from parse_diff_to_structured()
        max_files_per_chunk: Maximum files per chunk (legacy, for fallback)
        dependency_graph: Optional dependency graph from dependency_detector.build_dependency_graph()
        model_name: Model name for token limits (e.g., "minimax", "glm-5.2")

    Returns:
        List of chunks, each with:
        - files: List of file info dicts
        - total_files: Count
        - total_changes: Sum of changes
        - chunk_index: 1-indexed
        - dependency_cluster: List of file paths in this cluster (if using dependencies)
        - dependency_contexts: Caller -> callee structures
        - estimated_tokens: Estimated token count
    """
    files = structured["files"]

    # Use token + dependency-aware chunking if graph provided
    if dependency_graph and dependency_graph.get("clusters"):
        return _chunk_by_dependency_and_tokens(
            files, dependency_graph, model_name or "glm-5.2"
        )

    # Fall back to simple chunking
    return _chunk_by_file_count(files, max_files_per_chunk)


def _chunk_by_file_count(
    files: list[dict], max_files_per_chunk: int
) -> list[dict[str, Any]]:
    """Simple chunking by file count (original logic)."""
    chunks = []

    for i in range(0, len(files), max_files_per_chunk):
        chunk_files = files[i : i + max_files_per_chunk]
        chunk_total_changes = sum(f["total_changes"] for f in chunk_files)

        chunks.append(
            {
                "files": chunk_files,
                "total_files": len(chunk_files),
                "total_changes": chunk_total_changes,
                "chunk_index": len(chunks) + 1,
                "total_chunks": (len(files) + max_files_per_chunk - 1)
                // max_files_per_chunk,
            }
        )

    return chunks


def _chunk_by_dependency_and_tokens(
    files: list[dict], dependency_graph: dict[str, Any], model_name: str
) -> list[dict[str, Any]]:
    """
    Token-aware + Dependency-aware chunking.

    Keeps dependencies together when they fit in token limits.
    Splits intelligently when cluster is too large.

    Args:
        files: All changed files
        dependency_graph: Graph with clusters and edges
        model_name: Model name for token limits

    Returns:
        List of chunks with dependency contexts
    """
    from utilities.model_token_config import get_model_config

    # Get model limits
    model_config = get_model_config(model_name)
    max_input_tokens = model_config.get(
        "input_context_window",
        model_config.get("context_window", 100000),
    )

    # Use 50% of context for safety. The diff/change payload also has a hard
    # ceiling because very large structured outputs become unreliable even when
    # the model context window can fit them.
    target_chunk_tokens = int(max_input_tokens * 0.5)
    change_target_tokens = min(target_chunk_tokens, MAX_CHANGE_TOKENS_PER_CHUNK)

    print(
        f"[Chunking] Model: {model_name}, Target prompt size: {target_chunk_tokens} tokens, "
        f"Target change payload: {change_target_tokens} tokens"
    )

    chunks = []
    pending_independent_chunks: list[dict[str, Any]] = []
    file_path_to_info = {f["path"]: f for f in files}
    all_changed_files = set(file_path_to_info.keys())

    # Filter edges: BOTH caller AND callee must be modified
    edges = dependency_graph.get("edges", [])
    filtered_edges = [
        edge
        for edge in edges
        if edge.get("from") in all_changed_files and edge.get("to") in all_changed_files
    ]

    # Process each dependency cluster
    for cluster in dependency_graph.get("clusters", []):
        cluster_files = [
            file_path_to_info[path] for path in cluster if path in file_path_to_info
        ]

        if not cluster_files:
            continue

        # Build caller -> callee contexts
        dependency_contexts = _build_caller_callee_contexts_for_chunk(
            cluster, filtered_edges, file_path_to_info
        )

        # Build initial chunk
        chunk = {
            "files": cluster_files,
            "total_files": len(cluster_files),
            "total_changes": sum(
                f.get("total_changes", len(f.get("changes", []))) for f in cluster_files
            ),
            "dependency_cluster": cluster,
            "dependency_contexts": dependency_contexts,
        }

        # Estimate token size
        estimated_tokens = _estimate_chunk_tokens(chunk)
        chunk["estimated_tokens"] = estimated_tokens
        change_tokens = _estimate_chunk_change_tokens(chunk)
        chunk["change_tokens"] = change_tokens

        if (
            not dependency_contexts
            and estimated_tokens <= target_chunk_tokens
            and change_tokens <= change_target_tokens
        ):
            pending_independent_chunks.append(chunk)
            continue

        if pending_independent_chunks:
            chunks.extend(
                _pack_independent_chunks_by_tokens(
                    pending_independent_chunks,
                    target_chunk_tokens,
                    change_target_tokens,
                )
            )
            pending_independent_chunks = []

        if (
            estimated_tokens <= target_chunk_tokens
            and change_tokens <= change_target_tokens
        ):
            # Fits! Keep entire cluster together OK
            print(
                f"[Chunking] Cluster with {len(cluster_files)} files (~{estimated_tokens} tokens max, ~{change_tokens} change tokens) fits in one chunk"
            )
            print(
                f"[Chunking]   Classification: ~{len(cluster_files) * 50 + 8000} tokens, Deep analysis: ~{chunk.get('total_changes', 0) * 10 + 8500} tokens"
            )
            chunk["chunk_index"] = len(chunks) + 1
            chunks.append(chunk)
        else:
            if dependency_contexts:
                caller_group_chunks = _split_cluster_by_caller_groups(
                    cluster_files=cluster_files,
                    dependency_contexts=dependency_contexts,
                    file_path_to_info=file_path_to_info,
                )
                if len(caller_group_chunks) > 1:
                    bounded_caller_chunks = []
                    caller_chunks_fit = True
                    for sub_chunk in caller_group_chunks:
                        sub_chunk["estimated_tokens"] = _estimate_chunk_tokens(
                            sub_chunk
                        )
                        sub_chunk["change_tokens"] = _estimate_chunk_change_tokens(
                            sub_chunk
                        )
                        if (
                            sub_chunk["estimated_tokens"] > target_chunk_tokens
                            or sub_chunk["change_tokens"] > change_target_tokens
                        ):
                            caller_chunks_fit = False
                            break
                        bounded_caller_chunks.append(sub_chunk)

                    if caller_chunks_fit:
                        print(
                            f"[Chunking] Split dependency cluster with {len(cluster_files)} files "
                            f"into {len(bounded_caller_chunks)} caller-group chunks"
                        )
                        chunks.extend(bounded_caller_chunks)
                        continue

            # Too large! Need smart split
            print(
                f"[Chunking] WARNING: Cluster with {len(cluster_files)} files (~{estimated_tokens} tokens max, ~{change_tokens} change tokens) exceeds limits (prompt target {target_chunk_tokens}, change target {change_target_tokens}), splitting..."
            )
            print(
                "[Chunking]   Reason: Classification or deep analysis prompt too large"
            )
            if dependency_contexts:
                sub_chunks = _split_cluster_by_tokens(
                    cluster,
                    cluster_files,
                    dependency_contexts,
                    filtered_edges,
                    file_path_to_info,
                    change_target_tokens,
                )
            else:
                sub_chunks = _split_files_by_change_tokens(
                    cluster_files, change_target_tokens
                )

            for sub_chunk in sub_chunks:
                sub_chunk["chunk_index"] = len(chunks) + 1
                chunks.append(sub_chunk)

            print(f"[Chunking] Split into {len(sub_chunks)} sub-chunks")

    if pending_independent_chunks:
        chunks.extend(
            _pack_independent_chunks_by_tokens(
                pending_independent_chunks,
                target_chunk_tokens,
                change_target_tokens,
            )
        )

    # Add total_chunks to all
    total_chunks = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_index"] = index
        chunk["total_chunks"] = total_chunks

    return chunks


def _pack_independent_chunks_by_tokens(
    independent_chunks: list[dict[str, Any]],
    target_chunk_tokens: int,
    change_target_tokens: int,
) -> list[dict[str, Any]]:
    """Pack independent dependency clusters while staying under token budgets."""
    packed_chunks: list[dict[str, Any]] = []
    current_files: list[dict] = []
    current_clusters: list[str] = []

    def flush_current() -> None:
        if not current_files:
            return
        chunk = {
            "files": list(current_files),
            "total_files": len(current_files),
            "total_changes": sum(
                f.get("total_changes", len(f.get("changes", []))) for f in current_files
            ),
            "dependency_cluster": list(current_clusters),
            "dependency_contexts": [],
            "chunk_info": "Packed independent dependency clusters",
        }
        chunk["estimated_tokens"] = _estimate_chunk_tokens(chunk)
        chunk["change_tokens"] = _estimate_chunk_change_tokens(chunk)
        packed_chunks.append(chunk)

    for chunk in independent_chunks:
        candidate_files = current_files + list(chunk.get("files", []))
        candidate = {
            "files": candidate_files,
            "total_files": len(candidate_files),
            "total_changes": sum(
                f.get("total_changes", len(f.get("changes", [])))
                for f in candidate_files
            ),
            "dependency_contexts": [],
        }
        candidate_tokens = _estimate_chunk_tokens(candidate)
        candidate_change_tokens = _estimate_chunk_change_tokens(candidate)

        if current_files and (
            candidate_tokens > target_chunk_tokens
            or candidate_change_tokens > change_target_tokens
        ):
            flush_current()
            current_files = []
            current_clusters = []

        current_files.extend(chunk.get("files", []))
        current_clusters.extend(chunk.get("dependency_cluster", []))

    flush_current()

    if packed_chunks:
        print(
            f"[Chunking] Packed {len(independent_chunks)} independent cluster(s) "
            f"into {len(packed_chunks)} token-bounded chunk(s)"
        )

    return packed_chunks


def _split_files_by_change_tokens(
    files: list[dict[str, Any]], target_tokens: int
) -> list[dict[str, Any]]:
    """Split files into chunks by actual change payload token estimate."""
    chunks: list[dict[str, Any]] = []
    current_files: list[dict[str, Any]] = []
    current_tokens = 0

    def flush_current() -> None:
        if not current_files:
            return
        chunk = {
            "files": list(current_files),
            "total_files": len(current_files),
            "total_changes": sum(
                f.get("total_changes", len(f.get("changes", []))) for f in current_files
            ),
            "dependency_cluster": [
                f.get("path", "") for f in current_files if f.get("path")
            ],
            "dependency_contexts": [],
            "is_partial_cluster": True,
            "chunk_info": "Token split without dependency context",
        }
        chunk["estimated_tokens"] = _estimate_chunk_tokens(chunk)
        chunk["change_tokens"] = _estimate_chunk_change_tokens(chunk)
        chunks.append(chunk)

    for file_info in files:
        file_tokens = _estimate_file_tokens(file_info)

        # If single file exceeds target, split it into smaller chunks by hunks
        if file_tokens > target_tokens:
            print(f"[DEBUG] Large file detected: {file_info.get('path', 'unknown')} ({file_tokens} tokens > {target_tokens} target)")
            # Flush current batch first
            if current_files:
                flush_current()
                current_files = []
                current_tokens = 0

            # Split large file into multiple chunks by hunks
            print(f"[DEBUG] Splitting large file by hunks...")
            hunks = file_info.get("changes", [])
            if hunks:
                chunk_hunks = []
                chunk_tokens = 0

                for hunk in hunks:
                    hunk_tokens = len(str(hunk)) // 4  # Rough estimate

                    if chunk_hunks and chunk_tokens + hunk_tokens > target_tokens:
                        # Flush current hunk batch
                        file_chunk = dict(file_info)
                        file_chunk["changes"] = chunk_hunks
                        file_chunk["total_changes"] = len(chunk_hunks)
                        chunks.append({
                            "files": [file_chunk],
                            "total_files": 1,
                            "total_changes": len(chunk_hunks),
                            "dependency_cluster": [file_info.get("path", "")],
                            "dependency_contexts": [],
                            "is_partial_cluster": True,
                            "chunk_info": f"Large file split (part {len(chunks)+1})",
                            "estimated_tokens": chunk_tokens,
                        })
                        chunk_hunks = []
                        chunk_tokens = 0

                    chunk_hunks.append(hunk)
                    chunk_tokens += hunk_tokens

                # Flush last batch
                if chunk_hunks:
                    file_chunk = dict(file_info)
                    file_chunk["changes"] = chunk_hunks
                    file_chunk["total_changes"] = len(chunk_hunks)
                    chunks.append({
                        "files": [file_chunk],
                        "total_files": 1,
                        "total_changes": len(chunk_hunks),
                        "dependency_cluster": [file_info.get("path", "")],
                        "dependency_contexts": [],
                        "is_partial_cluster": True,
                        "chunk_info": f"Large file split (part {len(chunks)})",
                        "estimated_tokens": chunk_tokens,
                    })
            else:
                # No hunks to split, add as-is with warning
                print(f"  WARNING: File {file_info.get('path', 'unknown')} exceeds token limit ({file_tokens} > {target_tokens}) but has no hunks to split")
                current_files.append(file_info)
                current_tokens += file_tokens

            continue

        # Normal case: file fits in target
        if current_files and current_tokens + file_tokens > target_tokens:
            flush_current()
            current_files = []
            current_tokens = 0

        current_files.append(file_info)
        current_tokens += file_tokens

    flush_current()
    return chunks


def _split_cluster_by_caller_groups(
    cluster_files: list[dict],
    dependency_contexts: list[dict],
    file_path_to_info: dict[str, dict],
    max_files_per_chunk: int = 30,
) -> list[dict[str, Any]]:
    """
    Split a connected dependency component into bounded caller -> callee groups.

    Connected components can become too broad when many valid dependency edges
    share weak bridges. Classification only needs the direct caller context, so
    each chunk carries one caller and a bounded batch of direct callees.
    """
    cluster_paths = [f["path"] for f in cluster_files if f.get("path")]
    unassigned = set(cluster_paths)
    chunks: list[dict[str, Any]] = []

    priority_order = {
        "READS": 1,
        "TESTS": 1,
        "CONFIGURES": 2,
        "IMPORTS": 2,
        "RELATED_CHANGES": 3,
    }
    sorted_contexts = sorted(
        dependency_contexts,
        key=lambda ctx: (
            priority_order.get(str(ctx.get("dependency_type", "")).upper(), 99),
            -len(ctx.get("callees", [])),
        ),
    )

    for dep_context in sorted_contexts:
        caller = dep_context.get("caller", {})
        callees = dep_context.get("callees", [])
        dep_type = str(dep_context.get("dependency_type", "UNKNOWN")).upper()
        caller_file = caller.get("file")

        if not caller_file or caller_file not in file_path_to_info or not callees:
            continue

        direct_callees = [
            callee
            for callee in callees
            if callee.get("file") in unassigned
            and callee.get("file") in file_path_to_info
        ]
        caller_is_available = caller_file in unassigned

        if not direct_callees and not caller_is_available:
            continue

        batch_size = max(1, max_files_per_chunk - 1)
        callee_batches = [
            direct_callees[index : index + batch_size]
            for index in range(0, len(direct_callees), batch_size)
        ] or [[]]

        for batch_index, callee_batch in enumerate(callee_batches):
            chunk_paths: list[str] = []
            if caller_is_available and batch_index == 0:
                chunk_paths.append(caller_file)
            chunk_paths.extend(
                callee["file"]
                for callee in callee_batch
                if callee.get("file") in file_path_to_info
            )

            if not chunk_paths:
                continue

            chunk_files = [file_path_to_info[path] for path in chunk_paths]
            chunk_callees = [
                callee for callee in callee_batch if callee.get("file") in chunk_paths
            ]
            chunk = {
                "files": chunk_files,
                "total_files": len(chunk_files),
                "total_changes": sum(
                    f.get("total_changes", len(f.get("changes", [])))
                    for f in chunk_files
                ),
                "dependency_cluster": chunk_paths,
                "is_partial_cluster": True,
                "chunk_info": (
                    f"Caller-group chunk: {caller_file} -> "
                    f"{len(chunk_callees)} direct callees"
                ),
            }
            if chunk_callees:
                chunk["dependency_contexts"] = [
                    {
                        "dependency_type": dep_type,
                        "caller": caller,
                        "callees": chunk_callees,
                    }
                ]
            else:
                chunk["dependency_contexts"] = []

            chunks.append(chunk)
            unassigned.difference_update(chunk_paths)

    for path in cluster_paths:
        if path not in unassigned:
            continue
        file_info = file_path_to_info.get(path)
        if not file_info:
            continue
        chunks.append(
            {
                "files": [file_info],
                "total_files": 1,
                "total_changes": file_info.get(
                    "total_changes", len(file_info.get("changes", []))
                ),
                "dependency_cluster": [path],
                "dependency_contexts": [],
                "is_partial_cluster": True,
                "chunk_info": "Independent file from dependency cluster",
            }
        )
        unassigned.remove(path)

    return chunks


def _estimate_chunk_tokens(chunk: dict[str, Any]) -> int:
    """
    Estimate total tokens for a chunk for BOTH classification AND deep analysis.

    Returns the MAXIMUM of the two estimates to ensure BOTH prompts fit.

    Classification prompt includes:
    - CI context (~3000 tokens)
    - Validation sequence (~2000 tokens)
    - File summaries with 1 example change per file (~50 tokens/file)
    - Dependency context (~50 tokens per dependency)
    - Instructions (~3000 tokens)

    Deep analysis prompt includes:
    - CI context (~3000 tokens)
    - Validation context (~500 tokens)
    - ALL file changes (~10 tokens per change)
    - Dependency context (~50 tokens per dependency)
    - Instructions (~5000 tokens)
    """
    total_files = chunk.get("total_files", 0)
    total_changes = chunk.get("total_changes", 0)
    dep_contexts = chunk.get("dependency_contexts", [])

    # CLASSIFICATION prompt estimate
    classification_tokens = (
        3000  # CI context
        + 2000  # Validation sequence
        + (total_files * 50)  # File summaries (1 example per file)
        + (len(dep_contexts) * 50)  # Dependency contexts
        + 3000  # Instructions
    )

    # DEEP ANALYSIS prompt estimate
    deep_analysis_tokens = (
        3000  # CI context
        + 500  # Validation context
        + (total_changes * 10)  # ALL file changes
        + (len(dep_contexts) * 50)  # Dependency contexts
        + 5000  # Instructions (longer for atomic problem generation)
    )

    # Return MAXIMUM to ensure BOTH prompts fit
    max_tokens = max(classification_tokens, deep_analysis_tokens)

    return max_tokens


def _estimate_chunk_change_tokens(chunk: dict[str, Any]) -> int:
    """Estimate tokens for the actual before/after change payload in a chunk."""
    return sum(_estimate_file_tokens(file_info) for file_info in chunk.get("files", []))


def _estimate_file_tokens(file_info: dict[str, Any]) -> int:
    """Estimate tokens for a single file's actual change payload."""
    from utilities.diff_chunker import estimate_tokens

    file_tokens = estimate_tokens(str(file_info.get("path", ""))) + 10
    changes = file_info.get("changes", [])
    for change in changes:
        before_text = str(change.get("before", "") or "")
        after_text = str(change.get("after", "") or "")
        file_tokens += estimate_tokens(before_text)
        file_tokens += estimate_tokens(after_text)
        file_tokens += 8
    return file_tokens


def _split_cluster_by_tokens(
    cluster: list[str],
    cluster_files: list[dict],
    dependency_contexts: list[dict],
    filtered_edges: list[dict],
    file_path_to_info: dict[str, dict],
    target_tokens: int,
) -> list[dict[str, Any]]:
    """
    Split large dependency cluster while preserving relationships.

    Strategy:
    - Group by dependency type priority (READS > CONFIGURES > RELATED_CHANGES)
    - Keep caller + callees together when possible
    - Distribute callees across chunks if needed
    - Each sub-chunk gets its relevant dependency context
    """
    chunks = []

    # Priority order for dependencies
    priority_order = {
        "READS": 1,
        "TESTS": 1,
        "CONFIGURES": 2,
        "IMPORTS": 2,
        "RELATED_CHANGES": 3,
    }

    # Sort contexts by priority
    sorted_contexts = sorted(
        dependency_contexts,
        key=lambda ctx: priority_order.get(
            ctx.get("dependency_type", "RELATED_CHANGES"), 99
        ),
    )

    for dep_context in sorted_contexts:
        caller = dep_context.get("caller", {})
        callees = dep_context.get("callees", [])
        dep_type = dep_context.get("dependency_type", "UNKNOWN")

        caller_file = caller.get("file")
        if not caller_file or not callees:
            continue

        # Get file info
        caller_file_info = file_path_to_info.get(caller_file)
        if not caller_file_info:
            continue

        caller_tokens = _estimate_file_tokens(caller_file_info)

        # If caller itself exceeds limit, we MUST split it to preserve all data
        # Split caller file into smaller chunks by hunks
        if caller_tokens > target_tokens:
            print(f"[DEBUG] Caller file {caller_file} too large ({caller_tokens} > {target_tokens}), splitting by hunks")
            caller_hunks = caller_file_info.get("changes", [])

            if caller_hunks:
                # Split caller into multiple chunks by hunks
                chunk_hunks = []
                chunk_tokens = 5000  # Base overhead

                for hunk in caller_hunks:
                    hunk_tokens = len(str(hunk)) // 4  # Rough estimate

                    if chunk_hunks and chunk_tokens + hunk_tokens > target_tokens:
                        # Flush current chunk
                        caller_chunk = dict(caller_file_info)
                        caller_chunk["changes"] = chunk_hunks
                        caller_chunk["total_changes"] = len(chunk_hunks)
                        chunks.append({
                            "files": [caller_chunk],
                            "total_files": 1,
                            "total_changes": len(chunk_hunks),
                            "dependency_contexts": [{
                                "dependency_type": dep_type,
                                "caller": {"file": caller_file, "is_partial": True},
                                "callees": [],
                            }],
                            "estimated_tokens": chunk_tokens,
                            "is_partial_cluster": True,
                            "chunk_info": f"Large caller split (part {len(chunks)+1})",
                        })
                        chunk_hunks = []
                        chunk_tokens = 5000

                    chunk_hunks.append(hunk)
                    chunk_tokens += hunk_tokens

                # Flush last chunk
                if chunk_hunks:
                    caller_chunk = dict(caller_file_info)
                    caller_chunk["changes"] = chunk_hunks
                    caller_chunk["total_changes"] = len(chunk_hunks)
                    chunks.append({
                        "files": [caller_chunk],
                        "total_files": 1,
                        "total_changes": len(chunk_hunks),
                        "dependency_contexts": [{
                            "dependency_type": dep_type,
                            "caller": {"file": caller_file, "is_partial": True},
                            "callees": [],
                        }],
                        "estimated_tokens": chunk_tokens,
                        "is_partial_cluster": True,
                        "chunk_info": f"Large caller split (final part)",
                    })
                print(f"[DEBUG] Split large caller into {len([c for c in chunks if c.get('chunk_info', '').startswith('Large caller')])} chunks")

            # Skip to next dependency context (this one is fully processed)
            continue

        # Normal case: caller fits in limit
        # Start new chunk with caller
        current_chunk_files = [caller_file_info]
        current_chunk_callees = []
        current_tokens = caller_tokens + 5000  # Base overhead

        # Add callees until we hit token limit
        for callee in callees:
            callee_file = callee.get("file")
            callee_file_info = file_path_to_info.get(callee_file)

            if not callee_file_info:
                continue

            callee_tokens = _estimate_file_tokens(callee_file_info)

            if current_tokens + callee_tokens <= target_tokens:
                # Fits in current chunk
                current_chunk_files.append(callee_file_info)
                current_chunk_callees.append(callee)
                current_tokens += callee_tokens
            else:
                # Current chunk full, save it
                if current_chunk_callees:
                    chunks.append(
                        {
                            "files": current_chunk_files,
                            "total_files": len(current_chunk_files),
                            "total_changes": sum(
                                f.get("total_changes", len(f.get("changes", [])))
                                for f in current_chunk_files
                            ),
                            "dependency_contexts": [
                                {
                                    "dependency_type": dep_type,
                                    "caller": caller,
                                    "callees": current_chunk_callees,
                                }
                            ],
                            "estimated_tokens": current_tokens,
                            "is_partial_cluster": True,
                        }
                    )

                # Start new chunk with same caller
                current_chunk_files = [caller_file_info, callee_file_info]
                current_chunk_callees = [callee]
                current_tokens = (
                    _estimate_file_tokens(caller_file_info) + callee_tokens + 5000
                )

        # Add last chunk
        if current_chunk_callees:
            chunks.append(
                {
                    "files": current_chunk_files,
                    "total_files": len(current_chunk_files),
                    "total_changes": sum(
                        f.get("total_changes", len(f.get("changes", [])))
                        for f in current_chunk_files
                    ),
                    "dependency_contexts": [
                        {
                            "dependency_type": dep_type,
                            "caller": caller,
                            "callees": current_chunk_callees,
                        }
                    ],
                    "estimated_tokens": current_tokens,
                    "is_partial_cluster": True,
                }
            )

    # If no dependency-aware chunks were created, split by token limit
    if not chunks:
        print(f"[DEBUG] No dependency chunks created, falling back to file split (target: {target_tokens} tokens, {len(cluster_files)} files)")
        result = _split_files_by_change_tokens(cluster_files, target_tokens)
        print(f"[DEBUG] File split created {len(result)} chunks")
        return result

    print(f"[DEBUG] Dependency-aware split created {len(chunks)} chunks")
    return chunks


def _build_caller_callee_contexts_for_chunk(
    cluster: list[str], filtered_edges: list[dict], file_path_to_info: dict[str, dict]
) -> list[dict]:
    """
    Build caller -> callee contexts for a chunk from filtered edges.

    Args:
        cluster: List of file paths in this cluster
        filtered_edges: Edges where BOTH caller and callee are modified
        file_path_to_info: Map of file path -> file info with changes

    Returns:
        List of caller -> callee context dicts
    """
    cluster_set = set(cluster)
    contexts = []

    # Group edges by (caller, type)
    caller_groups = {}
    for edge in filtered_edges:
        caller = edge.get("from")
        callee = edge.get("to")
        dep_type = edge.get("type", "unknown")

        # Only include if both are in this cluster
        if caller in cluster_set and callee in cluster_set:
            key = (caller, dep_type)
            if key not in caller_groups:
                caller_groups[key] = {"caller": caller, "type": dep_type, "callees": []}
            if callee not in caller_groups[key]["callees"]:
                caller_groups[key]["callees"].append(callee)

    # Build structured contexts
    for (caller, dep_type), group in caller_groups.items():
        callees = group["callees"]
        caller_info = file_path_to_info.get(caller, {})

        # Build caller context
        caller_context = {
            "file": caller,
            "changes": _extract_changes_summary(caller_info),
            "role": _classify_file_role(caller),
        }

        # Build callee contexts
        callee_contexts = []
        for callee in callees:
            callee_info = file_path_to_info.get(callee, {})
            callee_context = {
                "file": callee,
                "changes": _extract_changes_summary(callee_info),
                "role": _classify_file_role(callee),
            }
            callee_contexts.append(callee_context)

        # Create structured context
        contexts.append(
            {
                "dependency_type": dep_type.upper(),
                "caller": caller_context,
                "callees": callee_contexts,
            }
        )

    return contexts


def _extract_changes_summary(file_info: dict) -> list[dict]:
    """Extract up to 3 sample changes from file info."""
    changes = file_info.get("changes", [])
    summary = []

    for change in changes[:3]:  # First 3 changes
        summary.append(
            {
                "line": change.get("line"),
                "before": (change.get("before", "") or "")[:200],  # Limit to 200 chars
                "after": (change.get("after", "") or "")[:200],
            }
        )

    return summary


def _classify_file_role(file_path: str) -> str:
    """Classify file role for dependency context."""
    if file_path.endswith("_test.py") or "/tests/" in file_path:
        return "test"
    elif file_path.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")):
        return "config"
    elif file_path.endswith((".rst", ".md")):
        return "docs"
    elif file_path.endswith(".py"):
        return "code"
    else:
        return "other"


def _chunk_by_dependency_clusters(
    files: list[dict], dependency_graph: dict[str, Any], max_files_per_chunk: int
) -> list[dict[str, Any]]:
    """
    Dependency-aware chunking - keep related files together.

    Strategy:
    1. Group files by dependency cluster
    2. Build caller -> callee contexts from graph edges
    3. If cluster > max_files, split it (but warn)
    4. Keep dependencies within same chunk when possible
    """
    from dependency_detector import explain_dependencies

    chunks = []
    file_path_to_info = {f["path"]: f for f in files}
    all_changed_files = set(file_path_to_info.keys())

    # Filter edges to only include modified files (BOTH caller and callee)
    edges = dependency_graph.get("edges", [])
    filtered_edges = [
        edge
        for edge in edges
        if edge.get("from") in all_changed_files and edge.get("to") in all_changed_files
    ]

    # Process each dependency cluster
    for cluster in dependency_graph.get("clusters", []):
        # Get file info for files in this cluster
        cluster_files = [
            file_path_to_info[path] for path in cluster if path in file_path_to_info
        ]

        if not cluster_files:
            continue

        # Build caller -> callee contexts for this cluster
        dependency_contexts = _build_caller_callee_contexts_for_chunk(
            cluster, filtered_edges, file_path_to_info
        )

        # If cluster fits in one chunk, keep it together
        if len(cluster_files) <= max_files_per_chunk:
            chunk_total_changes = sum(f["total_changes"] for f in cluster_files)

            chunk = {
                "files": cluster_files,
                "total_files": len(cluster_files),
                "total_changes": chunk_total_changes,
                "chunk_index": len(chunks) + 1,
                "dependency_cluster": cluster,
                "dependency_explanation": explain_dependencies(
                    cluster, dependency_graph
                ),
            }

            # Attach caller -> callee contexts if available
            if dependency_contexts:
                chunk["dependency_contexts"] = dependency_contexts

            chunks.append(chunk)
        else:
            # Cluster too large - split it (but keep track it's from same cluster)
            for i in range(0, len(cluster_files), max_files_per_chunk):
                sub_chunk_files = cluster_files[i : i + max_files_per_chunk]
                chunk_total_changes = sum(f["total_changes"] for f in sub_chunk_files)

                chunk = {
                    "files": sub_chunk_files,
                    "total_files": len(sub_chunk_files),
                    "total_changes": chunk_total_changes,
                    "chunk_index": len(chunks) + 1,
                    "dependency_cluster": cluster,
                    "dependency_explanation": f"Part of larger cluster: {explain_dependencies(cluster, dependency_graph)}",
                    "is_partial_cluster": True,
                }

                # Attach caller -> callee contexts if available
                if dependency_contexts:
                    chunk["dependency_contexts"] = dependency_contexts

                chunks.append(chunk)

    # Add total_chunks to all
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total_chunks

    return chunks


def format_structured_for_llm(
    chunk: dict[str, Any],
    max_changes_per_file: int | None = 3,
    max_chars_per_value: int | None = 80,
) -> str:
    """
    Format structured chunk for LLM prompt.

    Shows file path, change counts, and example changes so LLM can
    understand the type of changes (imports, types, formatting, etc.)

    Args:
        chunk: Structured diff chunk
        max_changes_per_file: Maximum changes to show per file, or None for all.
        max_chars_per_value: Maximum characters per before/after value, or None
            to preserve complete values.
    """
    lines = []

    for file_info in chunk["files"]:
        file_path = file_info["path"]
        changes = file_info["changes"]

        if not changes:
            continue

        # Count change types
        modified = sum(1 for c in changes if c["change_type"] == "modified")
        added = sum(1 for c in changes if c["change_type"] == "added")
        deleted = sum(1 for c in changes if c["change_type"] == "deleted")

        # File header with counts
        lines.append(
            f"{file_path}: {len(changes)} changes ({modified}M, {added}A, {deleted}D)"
        )

        # Classification can request every complete change so later reasoning is
        # not biased toward only the first config key or package operation.
        examples = (
            changes
            if max_changes_per_file is None
            else changes[:max_changes_per_file]
        )
        for change in examples:
            change_type = change.get("change_type", "")
            before = change.get("before", "") or ""
            after = change.get("after", "") or ""
            if max_chars_per_value is not None:
                before = before[:max_chars_per_value]
                after = after[:max_chars_per_value]

            if change_type == "modified" and before and after:
                lines.append(f"  - Modified: '{before}' -> '{after}'")
            elif change_type == "added" and after:
                lines.append(f"  - Added: '{after}'")
            elif change_type == "deleted" and before:
                lines.append(f"  - Deleted: '{before}'")

        lines.append("")  # Blank line between files

    return "\n".join(lines)


# Test function
def test_parser():
    """Test the diff parser."""
    sample_diff = """diff --git a/dev/pyproject.toml b/dev/pyproject.toml
index 2015a28c3fbe..d9d3b0c7703d 100644
--- a/dev/pyproject.toml
+++ b/dev/pyproject.toml
@@ -18,7 +18,7 @@ python = "^3.9"
 isort = "==5.13.2"
 black = { version = "==24.2.0" }
-#taplo = "==0.9.3"
+taplo = "==0.9.3"
 docformatter = "==1.7.5"
diff --git a/framework/py/flwr/common/exit/exit_code_test.py b/framework/py/flwr/common/exit/exit_code_test.py
index abc123..def456 100644
--- a/framework/py/flwr/common/exit/exit_code_test.py
+++ b/framework/py/flwr/common/exit/exit_code_test.py
@@ -10,6 +10,7 @@ import unittest

 def test_something():
+    # New comment
     pass
"""

    import json

    structured = parse_diff_to_structured(sample_diff)
    print("Structured diff:")
    print(json.dumps(structured, indent=2))

    print("\n" + "=" * 80)
    print("LLM-friendly format:")
    print("=" * 80)
    chunks = chunk_structured_diff(structured, max_files_per_chunk=5)
    for chunk in chunks:
        print(format_structured_for_llm(chunk))


if __name__ == "__main__":
    test_parser()
