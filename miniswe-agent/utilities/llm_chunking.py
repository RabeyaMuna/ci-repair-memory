"""
utilities/llm_chunking.py - Smart chunking and retry logic for LLM API calls

Provides intelligent chunk splitting based on model context windows and
automatic retry with fallback to smaller chunks.

Usage:
    from utilities import invoke_with_chunking_fallback

    result = invoke_with_chunking_fallback(
        llm=my_llm,
        items=my_large_list,
        process_fn=lambda chunk: my_processing_logic(chunk),
        estimate_tokens_fn=lambda item: len(str(item)) // 4,
    )
"""

import logging
from typing import Any, Callable, List, Optional, TypeVar

from utilities.diff_chunker import estimate_tokens
from utilities.llm_invoker import invoke_llm_with_retry

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def estimate_item_tokens(item: Any) -> int:
    """
    Default token estimation for any item.

    Args:
        item: Any object to estimate tokens for

    Returns:
        Estimated token count
    """
    if isinstance(item, str):
        return estimate_tokens(item)
    elif isinstance(item, dict):
        # For dicts, estimate all string values
        total = 0
        for key, value in item.items():
            total += estimate_tokens(str(key))
            if isinstance(value, str):
                total += estimate_tokens(value)
            elif isinstance(value, (list, dict)):
                total += estimate_item_tokens(value)
            else:
                total += len(str(value)) // 4
        return total
    elif isinstance(item, (list, tuple)):
        return sum(estimate_item_tokens(x) for x in item)
    else:
        return len(str(item)) // 4  # Rough estimate


def get_model_token_limits(llm: Any) -> dict[str, int]:
    """
    Extract token limits from LLM instance.

    Args:
        llm: LLM instance

    Returns:
        Dict with 'max_input_tokens' and 'safe_input_tokens'
    """
    # Try to get from model config
    model_name = getattr(llm, "memci_model_key", None) or getattr(
        llm, "model_name", None
    )

    # Import here to avoid circular dependency
    try:
        from utilities.model_token_config import get_model_config

        config = get_model_config(model_name)
        return {
            "max_input_tokens": config.get("max_input_tokens", 100000),
            "safe_input_tokens": config.get("max_input_tokens", 100000) // 2,
        }
    except Exception:
        # Fallback defaults
        return {
            "max_input_tokens": 100000,
            "safe_input_tokens": 50000,
        }


def split_items_by_tokens(
    items: List[T],
    target_max_tokens: int,
    estimate_fn: Optional[Callable[[T], int]] = None,
) -> List[List[T]]:
    """
    Split items into chunks based on token estimates using bin-packing algorithm.

    Args:
        items: List of items to split
        target_max_tokens: Target maximum tokens per chunk
        estimate_fn: Function to estimate tokens for each item (optional)

    Returns:
        List of item chunks, each under target_max_tokens
    """
    if not items:
        return []

    if estimate_fn is None:
        estimate_fn = estimate_item_tokens

    # Calculate token size for each item
    item_sizes = [(item, estimate_fn(item)) for item in items]

    # Sort by size (largest first) for better bin packing
    item_sizes.sort(key=lambda x: x[1], reverse=True)

    # Bin packing: Group items into chunks under target size
    chunks = []
    current_chunk = []
    current_tokens = 0

    for item, item_tokens in item_sizes:
        # If single item exceeds target, give it its own chunk
        if item_tokens > target_max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            chunks.append([item])
            LOGGER.warning(
                f"Single item exceeds target ({item_tokens:,} > {target_max_tokens:,} tokens)"
            )
            continue

        # If adding this item would exceed target, start new chunk
        if current_tokens + item_tokens > target_max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(item)
        current_tokens += item_tokens

    # Add remaining items
    if current_chunk:
        chunks.append(current_chunk)

    # If only one chunk and it's too large, fall back to simple half-split
    if len(chunks) == 1 and len(items) > 1:
        half = len(items) // 2
        chunks = [items[:half], items[half:]]
        LOGGER.info("Bin packing resulted in 1 chunk, using half-split instead")

    return chunks


def invoke_with_chunking_fallback(
    llm: Any,
    items: List[T],
    build_prompt_fn: Callable[[List[T]], str],
    process_response_fn: Callable[[Any], List[Any]],
    estimate_tokens_fn: Optional[Callable[[T], int]] = None,
    max_tokens: Optional[int] = None,
    initial_target_tokens: Optional[int] = None,
    max_recursion_depth: int = 5,
    combine_results_fn: Optional[Callable[[List[Any], List[Any]], List[Any]]] = None,
    verbose: bool = True,
    _depth: int = 0,
) -> List[Any]:
    """
    Invoke LLM with automatic chunking fallback on size errors.

    This is a generic wrapper that handles:
    1. Token estimation for items
    2. Smart splitting if input is too large
    3. Automatic retry with smaller chunks on failure
    4. Result aggregation

    Args:
        llm: LLM instance
        items: List of items to process
        build_prompt_fn: Function to build prompt from items chunk
        process_response_fn: Function to extract results from LLM response
        estimate_tokens_fn: Function to estimate tokens per item (optional)
        max_tokens: Max output tokens for LLM (optional)
        initial_target_tokens: Initial target input tokens (optional, auto-calculated if None)
        max_recursion_depth: Max depth for recursive splitting (default: 5)
        combine_results_fn: Function to combine results from multiple chunks (default: extend)
        verbose: Print progress messages (default: True)
        _depth: Internal recursion depth tracker

    Returns:
        Aggregated results from processing all chunks

    Example:
        ```python
        # Process a large list of files
        def build_prompt(files):
            return f"Analyze these files: {json.dumps(files)}"

        def process_response(response):
            return response if isinstance(response, list) else []

        results = invoke_with_chunking_fallback(
            llm=my_llm,
            items=file_list,
            build_prompt_fn=build_prompt,
            process_response_fn=process_response,
            estimate_tokens_fn=lambda f: len(str(f)) // 4,
        )
        ```
    """
    if _depth >= max_recursion_depth:
        LOGGER.error(f"Max recursion depth ({max_recursion_depth}) reached, giving up")
        return []

    if not items:
        return []

    indent = "  " * _depth

    # Get model limits
    limits = get_model_token_limits(llm)

    # Calculate target tokens for this chunk
    if initial_target_tokens is None:
        target_tokens = limits["safe_input_tokens"]
    else:
        target_tokens = initial_target_tokens

    # Reduce target on recursive calls to avoid repeated failures
    if _depth > 0:
        target_tokens = target_tokens // 2

    # Estimate total tokens for current items
    if estimate_tokens_fn is None:
        estimate_tokens_fn = estimate_item_tokens

    total_tokens = sum(estimate_tokens_fn(item) for item in items)

    if verbose:
        if _depth == 0:
            LOGGER.info(
                f"{indent}Processing {len(items)} items (~{total_tokens:,} tokens)"
            )
        else:
            LOGGER.info(
                f"{indent}Recursive depth {_depth}: {len(items)} items "
                f"(~{total_tokens:,} tokens, target: {target_tokens:,})"
            )

    # If items fit in target, process directly
    if total_tokens <= target_tokens or len(items) == 1:
        prompt = build_prompt_fn(items)

        try:
            if verbose and _depth == 0:
                LOGGER.info(f"{indent}Invoking LLM with {len(items)} items...")

            result = invoke_llm_with_retry(
                llm=llm,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,  # Use our own logging
            )

            # Handle SPLIT_REQUIRED signal
            if result == "SPLIT_REQUIRED":
                if len(items) == 1:
                    LOGGER.warning(
                        f"{indent}SPLIT_REQUIRED but only 1 item, cannot split further"
                    )
                    return []

                if verbose:
                    LOGGER.info(
                        f"{indent}LLM returned SPLIT_REQUIRED, splitting chunk..."
                    )

                # Split and retry
                sub_chunks = split_items_by_tokens(
                    items, target_tokens // 2, estimate_tokens_fn
                )

                return _process_chunks_recursively(
                    llm=llm,
                    chunks=sub_chunks,
                    build_prompt_fn=build_prompt_fn,
                    process_response_fn=process_response_fn,
                    estimate_tokens_fn=estimate_tokens_fn,
                    max_tokens=max_tokens,
                    initial_target_tokens=target_tokens // 2,
                    max_recursion_depth=max_recursion_depth,
                    combine_results_fn=combine_results_fn,
                    verbose=verbose,
                    _depth=_depth + 1,
                )

            # Process response
            processed = process_response_fn(result)

            if verbose and _depth == 0:
                result_count = len(processed) if isinstance(processed, list) else 1
                LOGGER.info(f"{indent}Success: {result_count} results")

            return processed if isinstance(processed, list) else [processed]

        except Exception as e:
            error_msg = str(e).lower()
            is_size_error = any(
                keyword in error_msg
                for keyword in [
                    "token",
                    "length",
                    "limit",
                    "too long",
                    "maximum",
                    "context",
                ]
            )

            if is_size_error and len(items) > 1:
                if verbose:
                    LOGGER.warning(f"{indent}Size error detected, splitting chunk...")

                # Split and retry
                sub_chunks = split_items_by_tokens(
                    items, target_tokens // 2, estimate_tokens_fn
                )

                return _process_chunks_recursively(
                    llm=llm,
                    chunks=sub_chunks,
                    build_prompt_fn=build_prompt_fn,
                    process_response_fn=process_response_fn,
                    estimate_tokens_fn=estimate_tokens_fn,
                    max_tokens=max_tokens,
                    initial_target_tokens=target_tokens // 2,
                    max_recursion_depth=max_recursion_depth,
                    combine_results_fn=combine_results_fn,
                    verbose=verbose,
                    _depth=_depth + 1,
                )
            else:
                LOGGER.error(f"{indent}Error processing chunk: {str(e)[:200]}")
                return []

    # Items don't fit, need to split proactively
    if verbose:
        LOGGER.info(
            f"{indent}Proactive split: {total_tokens:,} > {target_tokens:,} tokens"
        )

    sub_chunks = split_items_by_tokens(items, target_tokens, estimate_tokens_fn)

    return _process_chunks_recursively(
        llm=llm,
        chunks=sub_chunks,
        build_prompt_fn=build_prompt_fn,
        process_response_fn=process_response_fn,
        estimate_tokens_fn=estimate_tokens_fn,
        max_tokens=max_tokens,
        initial_target_tokens=target_tokens,
        max_recursion_depth=max_recursion_depth,
        combine_results_fn=combine_results_fn,
        verbose=verbose,
        _depth=_depth + 1,
    )


def _process_chunks_recursively(
    llm: Any,
    chunks: List[List[T]],
    build_prompt_fn: Callable[[List[T]], str],
    process_response_fn: Callable[[Any], List[Any]],
    estimate_tokens_fn: Optional[Callable[[T], int]],
    max_tokens: Optional[int],
    initial_target_tokens: Optional[int],
    max_recursion_depth: int,
    combine_results_fn: Optional[Callable[[List[Any], List[Any]], List[Any]]],
    verbose: bool,
    _depth: int,
) -> List[Any]:
    """Helper to process multiple chunks recursively."""
    if combine_results_fn is None:

        def combine_results_fn(a, b):
            return a + b  # Default: extend

    all_results = []

    for i, chunk in enumerate(chunks, 1):
        if verbose:
            LOGGER.info(f"{'  ' * _depth}Processing sub-chunk {i}/{len(chunks)}...")

        chunk_results = invoke_with_chunking_fallback(
            llm=llm,
            items=chunk,
            build_prompt_fn=build_prompt_fn,
            process_response_fn=process_response_fn,
            estimate_tokens_fn=estimate_tokens_fn,
            max_tokens=max_tokens,
            initial_target_tokens=initial_target_tokens,
            max_recursion_depth=max_recursion_depth,
            combine_results_fn=combine_results_fn,
            verbose=verbose,
            _depth=_depth,
        )

        all_results = combine_results_fn(all_results, chunk_results)

    return all_results
