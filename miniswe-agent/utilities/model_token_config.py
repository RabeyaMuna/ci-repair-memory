"""
Model-specific token limits and chunking configuration.

This module defines input/output token limits for different models to optimize
chunking and LLM analysis for larger inputs and outputs.

Usage:
    from utilities.model_token_config import get_model_config

    config = get_model_config("glm-5.2")
    max_input_tokens = config["input_chunk_tokens"]
    max_output_tokens = config["output_max_tokens"]
"""

from __future__ import annotations

from typing import TypedDict


class ModelConfig(TypedDict):
    """Model configuration for token limits and chunking."""

    # Input limits
    input_context_window: int  # Total context window
    input_chunk_tokens: int  # Recommended chunk size for processing
    input_chunk_chars: int  # Approximate characters per chunk

    # Output limits
    output_max_tokens: int  # Maximum output tokens
    output_safe_tokens: int  # Safe output limit (with buffer)

    # L1/L2/L3 analysis limits
    l1_chunk_size: int  # Number of L1 memories per chunk
    l1_max_total: int  # Maximum L1s to process (safety limit)
    l2_batch_size: int  # Number of L2 trajectories per batch
    l2_common_candidates: int  # L2 common problem candidates
    l2_consecutive_candidates: int  # L2 consecutive problem candidates
    decompose_max_files_per_chunk: int
    decompose_max_changes_per_chunk: int

    # Multi-stage flags
    requires_multi_stage: bool  # Whether output limit requires splitting
    supports_large_output: bool  # Can handle 100k+ token outputs


# Model configurations based on OpenRouter specs
# IMPORTANT: input_chunk_tokens + output_max_tokens + prompt_overhead MUST be <= input_context_window
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "minimax-m2.5": {
        "input_context_window": 196_608,  # Actual limit from litellm (was 245k, corrected)
        "input_chunk_tokens": 60_000,  # FIXED: Was 100k but API hard limit ~128k. 60k chunk + 3k prompt + 65k output = 128k total
        "input_chunk_chars": 240_000,  # ~4 chars per token (60k * 4)
        "output_max_tokens": 65_536,  # ACTUAL limit from litellm (was 16k - WRONG!)
        "output_safe_tokens": 60_000,  # Safe limit with buffer (~92% of max)
        "l1_chunk_size": 15,  # L1s per chunk (was 5)
        "l1_max_total": 60,  # Max total L1s to process (CONSERVATIVE: 4x increase from 15)
        "l2_batch_size": 20,  # Larger batch size (was 10)
        "l2_common_candidates": 120,  # Common problem candidates (CONSERVATIVE: 4x from 30)
        "l2_consecutive_candidates": 160,  # Consecutive candidates (CONSERVATIVE: 4x from 40)
        "decompose_max_files_per_chunk": 80,
        "decompose_max_changes_per_chunk": 400,
        "requires_multi_stage": False,  # Single-stage for most cases now! (was True)
        "supports_large_output": True,  # YES! 65k is large (was False)
    },
    "glm-5.2": {
        "input_context_window": 1_000_000,  # 1M tokens - HUGE!
        "input_chunk_tokens": 100_000,  # FIXED: Was 800k but Z.ai API has hard limit ~128k. 100k chunk + 3k prompt + 25k buffer = 128k
        "input_chunk_chars": 400_000,  # ~4 chars per token (100k tokens)
        "output_max_tokens": 131_072,  # Native Z.ai provider limit (128k)
        "output_safe_tokens": 120_000,  # Safe limit with buffer (~92% of max)
        "l1_chunk_size": 80,  # OPTIMIZED: 80 L1s per chunk (was 15)
        "l1_max_total": 400,  # OPTIMIZED: Max 400 L1s total (was 150)
        "l2_batch_size": 100,  # OPTIMIZED: 100 trajectories per batch (was 25)
        "l2_common_candidates": 1200,  # OPTIMIZED: 1200 common candidates (was 240)
        "l2_consecutive_candidates": 1600,  # OPTIMIZED: 1600 consecutive (was 320)
        "decompose_max_files_per_chunk": 600,  # OPTIMIZED: 600 files per chunk (was 150)
        "decompose_max_changes_per_chunk": 3200,  # OPTIMIZED: 3200 changes (was 800)
        "requires_multi_stage": False,  # Can do single-stage analysis
        "supports_large_output": True,  # Supports 100k+ outputs
    },
    "gpt-5-mini": {
        "input_context_window": 272_000,  # GPT-5-mini: 272k INPUT limit (not 400k total)
        "input_chunk_tokens": 100_000,  # FIXED: 100k chunk + 30k prompt overhead + buffer = ~140k <= 272k INPUT limit
        "input_chunk_chars": 400_000,  # ~4 chars per token
        "output_max_tokens": 128_000,  # GPT-5-mini max output: 128k
        "output_safe_tokens": 120_000,  # Safe limit with buffer (~94% of max)
        "l1_chunk_size": 40,  # L1s per chunk (increased due to larger context)
        "l1_max_total": 160,  # Max total L1s to process
        "l2_batch_size": 50,  # Larger batch size
        "l2_common_candidates": 300,  # Common problem candidates
        "l2_consecutive_candidates": 400,  # Consecutive candidates
        "decompose_max_files_per_chunk": 200,  # More files due to larger context
        "decompose_max_changes_per_chunk": 1000,  # More changes due to larger context
        "requires_multi_stage": False,  # Single-stage for most cases
        "supports_large_output": True,  # Supports large outputs (128k)
    },
    "gpt-5.4": {
        "input_context_window": 272_000,  # GPT-5.4-mini: 272k INPUT limit (not 400k total)
        "input_chunk_tokens": 100_000,  # FIXED: 100k chunk + 30k prompt overhead + buffer = ~140k <= 272k INPUT limit
        "input_chunk_chars": 400_000,  # ~4 chars per token
        "output_max_tokens": 128_000,  # GPT-5.4-mini max output: 128k
        "output_safe_tokens": 120_000,  # Safe limit with buffer (~94% of max)
        "l1_chunk_size": 40,  # L1s per chunk (increased due to larger context)
        "l1_max_total": 160,  # Max total L1s to process
        "l2_batch_size": 50,  # Larger batch size
        "l2_common_candidates": 300,  # Common problem candidates
        "l2_consecutive_candidates": 400,  # Consecutive candidates
        "decompose_max_files_per_chunk": 200,  # More files due to larger context
        "decompose_max_changes_per_chunk": 1000,  # More changes due to larger context
        "requires_multi_stage": False,  # Single-stage for most cases
        "supports_large_output": True,  # Supports large outputs (128k)
    },
    "deepseek-v4-flash": {
        "input_context_window": 1_000_000,  # DeepSeek-V4-Flash: 1M context window
        "input_chunk_tokens": 600_000,  # Use model capacity: 600k input + 384k output + overhead fits 1M
        "input_chunk_chars": 2_400_000,  # ~4 chars per token (600k * 4)
        "output_max_tokens": 384_000,  # DeepSeek-V4-Flash: 384K max output
        "output_safe_tokens": 350_000,  # Safe limit with buffer (~91% of max)
        # L1/L2 settings scale with model capacity - processing logic dynamically chunks
        "l1_chunk_size": 500,  # Proportional to context window
        "l1_max_total": 2000,  # Proportional to context window
        "l2_batch_size": 500,  # Proportional to context window
        "l2_common_candidates": 6000,  # Proportional to context window
        "l2_consecutive_candidates": 8000,  # Proportional to context window
        "decompose_max_files_per_chunk": 3000,  # Proportional to context window
        "decompose_max_changes_per_chunk": 15000,  # Proportional to context window
        "requires_multi_stage": False,  # Single-stage processing
        "supports_large_output": True,  # Supports massive outputs (384k)
    },
    # Fallback/default configuration
    "default": {
        "input_context_window": 128_000,
        "input_chunk_tokens": 70_000,
        "input_chunk_chars": 280_000,
        "output_max_tokens": 8_000,
        "output_safe_tokens": 7_000,
        "l1_chunk_size": 5,
        "l1_max_total": 15,
        "l2_batch_size": 10,
        "l2_common_candidates": 30,
        "l2_consecutive_candidates": 40,
        "decompose_max_files_per_chunk": 80,
        "decompose_max_changes_per_chunk": 400,
        "requires_multi_stage": True,
        "supports_large_output": False,
    },
}


def get_model_config(model_name: str | None) -> ModelConfig:
    """
    Get model-specific configuration for token limits and chunking.

    Args:
        model_name: Model name (e.g., "glm-5.2", "minimax-m2.5", "openrouter/z-ai/glm-5.2")

    Returns:
        ModelConfig dictionary with token limits and chunking parameters

    Examples:
        >>> config = get_model_config("glm-5.2")
        >>> config["input_chunk_tokens"]
        200000

        >>> config = get_model_config("minimax")
        >>> config["output_max_tokens"]
        16000
    """
    if not model_name:
        return MODEL_CONFIGS["default"]

    # Normalize model name
    normalized = str(model_name).lower().strip()

    # Remove provider prefixes
    for prefix in ["openrouter/", "z-ai/", "minimax/"]:
        normalized = normalized.replace(prefix, "")

    # Match to configuration
    if (
        "glm" in normalized
        or "gml" in normalized
        or "5.2" in normalized
        or "glm-5" in normalized
    ):
        return MODEL_CONFIGS["glm-5.2"]

    if "minimax" in normalized or "m2.5" in normalized or "2.5" in normalized:
        return MODEL_CONFIGS["minimax-m2.5"]

    # GPT-5.4 models (check specific version first)
    if "gpt-5.4" in normalized or "gpt5.4" in normalized or "2026-03-17" in normalized:
        return MODEL_CONFIGS["gpt-5.4"]

    # GPT-5-mini models
    if "gpt-5" in normalized or "gpt5" in normalized:
        return MODEL_CONFIGS["gpt-5-mini"]

    # DeepSeek-V4-Flash models
    if "deepseek" in normalized or "v4-flash" in normalized or "ds-v4" in normalized:
        return MODEL_CONFIGS["deepseek-v4-flash"]

    # Fallback to default
    return MODEL_CONFIGS["default"]


def get_input_chunk_tokens(model_name: str | None) -> int:
    """Get recommended input chunk size in tokens for a model."""
    return get_model_config(model_name)["input_chunk_tokens"]


def get_input_chunk_chars(model_name: str | None) -> int:
    """Get recommended input chunk size in characters for a model."""
    return get_model_config(model_name)["input_chunk_chars"]


def get_output_max_tokens(model_name: str | None) -> int:
    """Get maximum output tokens for a model."""
    return get_model_config(model_name)["output_max_tokens"]


def get_output_safe_tokens(model_name: str | None) -> int:
    """Get safe output token limit (with buffer) for a model."""
    return get_model_config(model_name)["output_safe_tokens"]


def get_l1_chunk_size(model_name: str | None) -> int:
    """Get number of L1 memories to process per chunk."""
    return get_model_config(model_name)["l1_chunk_size"]


def get_l2_batch_size(model_name: str | None) -> int:
    """Get number of L2 trajectories to process per batch."""
    return get_model_config(model_name)["l2_batch_size"]


def get_l1_max_total(model_name: str | None) -> int:
    """Get maximum total L1 memories to process."""
    return get_model_config(model_name)["l1_max_total"]


def get_l2_common_candidates(model_name: str | None) -> int:
    """Get number of L2 common problem candidates to analyze."""
    return get_model_config(model_name)["l2_common_candidates"]


def get_l2_consecutive_candidates(model_name: str | None) -> int:
    """Get number of L2 consecutive problem candidates to analyze."""
    return get_model_config(model_name)["l2_consecutive_candidates"]


def supports_large_output(model_name: str | None) -> bool:
    """Check if model can handle 100k+ token outputs."""
    return get_model_config(model_name)["supports_large_output"]


def get_model_context_limit(model_name: str | None) -> int:
    """Get model's total context window size in tokens."""
    return get_model_config(model_name)["input_context_window"]


def calculate_adaptive_output_limit(
    model_name: str | None,
    input_tokens: int,
    safety_ratio: float = 0.10,
    minimum_safety_tokens: int = 16_000,
) -> int:
    """Return the largest safe output ceiling for a measured prompt.

    The returned value is bounded by both the model's configured safe output
    limit and the context remaining after the complete input prompt plus a
    safety reserve.  It is a ceiling, not a generation target.
    """
    if not 0 <= safety_ratio < 1:
        raise ValueError("safety_ratio must be in the range [0, 1)")
    if input_tokens < 0:
        raise ValueError("input_tokens must be non-negative")

    config = get_model_config(model_name)
    context_window = config["input_context_window"]
    safety_tokens = max(
        minimum_safety_tokens,
        int(context_window * safety_ratio),
    )
    available_output = max(0, context_window - input_tokens - safety_tokens)
    return min(config["output_safe_tokens"], available_output)


def requires_multi_stage(model_name: str | None) -> bool:
    """Check if model requires multi-stage processing for large outputs."""
    return get_model_config(model_name)["requires_multi_stage"]


def calculate_safe_input_limit(
    model_name: str | None, desired_output_tokens: int | None = None
) -> int:
    """
    Calculate safe input limit ensuring: input + output + overhead <= context window.

    Args:
        model_name: Model name
        desired_output_tokens: Desired output size (None = use model's max)

    Returns:
        Maximum safe input tokens
    """
    config = get_model_config(model_name)
    context_window = config["input_context_window"]

    if desired_output_tokens is None:
        desired_output_tokens = config["output_max_tokens"]

    # Reserve 2% or 10k for prompt overhead (whichever is smaller)
    prompt_overhead = min(10_000, int(context_window * 0.02))

    max_safe_input = context_window - desired_output_tokens - prompt_overhead

    return max(0, max_safe_input)


def get_fallback_chunk_size(
    current_chunk_tokens: int, model_name: str | None = None
) -> int:
    """
    Get fallback chunk size when hitting token limits.

    Strategy: Reduce to half of current chunk size.

    Args:
        current_chunk_tokens: Current chunk size that failed
        model_name: Model name (for logging/validation)

    Returns:
        Reduced chunk size (current_chunk / 2)
    """
    fallback_size = current_chunk_tokens // 2

    # Ensure minimum chunk size (at least 1000 tokens)
    MIN_CHUNK_SIZE = 1_000
    if fallback_size < MIN_CHUNK_SIZE:
        fallback_size = MIN_CHUNK_SIZE

    return fallback_size


def validate_input_output_fit(
    input_tokens: int, output_tokens: int, model_name: str | None
) -> tuple[bool, str]:
    """
    Validate that input + output + overhead fits in context window.

    Args:
        input_tokens: Input size in tokens
        output_tokens: Desired output size in tokens
        model_name: Model name

    Returns:
        Tuple of (fits: bool, message: str)
    """
    config = get_model_config(model_name)
    context_window = config["input_context_window"]
    prompt_overhead = min(10_000, int(context_window * 0.02))

    total_needed = input_tokens + output_tokens + prompt_overhead

    if total_needed <= context_window:
        headroom = context_window - total_needed
        return (
            True,
            f"OK Fits: {total_needed:,} <= {context_window:,} (headroom: {headroom:,})",
        )
    else:
        overflow = total_needed - context_window
        return (
            False,
            f"NO Overflow: {total_needed:,} > {context_window:,} (over by: {overflow:,})",
        )


# Convenience function to print model comparison
def print_model_comparison():
    """Print comparison table of model configurations."""
    print("\n" + "=" * 80)
    print("MODEL TOKEN CONFIGURATION COMPARISON")
    print("=" * 80)

    headers = ["Model", "Input Chunk", "Output Max", "L1 Chunk", "Large Output"]
    col_widths = [15, 15, 15, 12, 15]

    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * 80)

    # Print each model
    for model_key in ["minimax-m2.5", "glm-5.2", "gpt-5-mini", "gpt-5.4", "deepseek-v4-flash"]:
        config = MODEL_CONFIGS[model_key]
        row = [
            model_key,
            f"{config['input_chunk_tokens']:,}",
            f"{config['output_max_tokens']:,}",
            str(config["l1_chunk_size"]),
            "OK" if config["supports_large_output"] else "NO",
        ]
        print(" | ".join(val.ljust(w) for val, w in zip(row, col_widths)))

    print("=" * 80)
    print()


if __name__ == "__main__":
    # Test and demonstrate usage
    print_model_comparison()

    # Test model resolution
    test_models = [
        "glm-5.2",
        "openrouter/z-ai/glm-5.2",
        "minimax-m2.5",
        "openrouter/minimax/minimax-m2.5",
        "gpt-5-mini",
        "gpt-5.4",
        "gpt5.4",
        "openrouter/deepseek/deepseek-v4-flash",
        "unknown-model",
    ]

    print("\nModel Resolution Tests:")
    print("-" * 80)
    for model in test_models:
        config = get_model_config(model)
        print(
            f"{model:40} -> Input: {config['input_chunk_tokens']:>8,} | Output: {config['output_max_tokens']:>8,}"
        )
