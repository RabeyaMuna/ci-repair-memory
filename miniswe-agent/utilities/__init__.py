# utilities/__init__.py
"""
Utilities package for mini-swe-agent-ci-based.

This package provides reusable utilities for:
- LLM providers and token tracking
- LLM invocation with robust retry logic
- Diff chunking and summarization
- Prompt building for CI analysis
- Dependency evidence extraction
"""

# Import commonly used functions for easier access
from utilities.diff_chunker import (
    estimate_tokens,
    chunk_diff_by_files,
    summarize_commit_changes,
    decide_chunking_strategy,
    merge_groups,
)

from utilities.prompt_builder import (
    FileSelectionPromptBuilder,
    CommitAnalysisPromptBuilder,
    truncate_text,
    format_validations,
    extract_failure_info,
)

from utilities.ci_cache import (
    load_structured_ci_failure,
    load_validation_sequence,
    save_structured_ci_failure_cache,
    save_validation_sequence_cache,
)

from utilities.dataset_fetcher import (
    fetch_dataset,
    filter_by_repos,
    save_issues_to_jsonl,
    matches_repo_filter,
)

from utilities.llm_invoker import (
    invoke_llm_with_retry,
    invoke_json,
)

from utilities.llm_chunking import (
    invoke_with_chunking_fallback,
    split_items_by_tokens,
    estimate_item_tokens,
    get_model_token_limits,
)

from utilities.model_registry import (
    configure_model_environment,
    resolve_model_alias,
)

from utilities.model_token_config import (
    get_model_config,
)

from utilities.error_handler import (
    save_error_to_execption,
    load_error_from_execption,
    has_error_in_execption,
    clear_error_from_execption,
    get_all_errors,
    create_error_dict,
)

__all__ = [
    # Diff chunking
    "estimate_tokens",
    "chunk_diff_by_files",
    "summarize_commit_changes",
    "decide_chunking_strategy",
    "merge_groups",
    # Prompt building
    "FileSelectionPromptBuilder",
    "CommitAnalysisPromptBuilder",
    "truncate_text",
    "format_validations",
    "extract_failure_info",
    # CI cache
    "load_structured_ci_failure",
    "load_validation_sequence",
    "save_structured_ci_failure_cache",
    "save_validation_sequence_cache",
    # Dataset fetching
    "fetch_dataset",
    "filter_by_repos",
    "save_issues_to_jsonl",
    "matches_repo_filter",
    # LLM invocation with retry logic
    "invoke_llm_with_retry",
    "invoke_json",
    # LLM chunking with fallback
    "invoke_with_chunking_fallback",
    "split_items_by_tokens",
    "estimate_item_tokens",
    "get_model_token_limits",
    # Model configuration
    "configure_model_environment",
    "resolve_model_alias",
    "get_model_config",
    # Error handling
    "save_error_to_execption",
    "load_error_from_execption",
    "has_error_in_execption",
    "clear_error_from_execption",
    "get_all_errors",
    "create_error_dict",
]
