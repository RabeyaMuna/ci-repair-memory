"""
Retry wrapper with exponential backoff for rate limits and timeouts.
Ensures no data is lost due to transient API errors.
"""

import time
import logging
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)


def retry_with_backoff(
    func: Callable,
    max_retries: int = 10,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    backoff_factor: float = 2.0,
    retry_on_timeout: bool = True,
    operation_name: str = "LLM call"
) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts (default: 10)
        initial_delay: Initial delay in seconds (default: 2s)
        max_delay: Maximum delay between retries (default: 120s = 2 minutes)
        backoff_factor: Multiplier for delay (default: 2x)
        retry_on_timeout: Whether to retry on timeout errors (default: True)
        operation_name: Name of operation for logging

    Returns:
        Result of the function call

    Raises:
        Exception: If all retries are exhausted
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            result = func()
            if attempt > 0:
                logger.info(f"✓ {operation_name} succeeded after {attempt} retries")
            return result

        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            # Check if it's a retryable error
            is_rate_limit = '429' in error_str or 'rate' in error_str or 'too many' in error_str
            is_timeout = '504' in error_str or 'timeout' in error_str or 'timed out' in error_str
            is_server_error = '500' in error_str or '502' in error_str or '503' in error_str

            should_retry = is_rate_limit or is_server_error or (retry_on_timeout and is_timeout)

            if not should_retry:
                logger.error(f"✗ {operation_name} failed with non-retryable error: {e}")
                raise

            if attempt == max_retries - 1:
                logger.error(f"✗ {operation_name} failed after {max_retries} retries: {e}")
                raise

            # Log the retry with clear information
            if is_rate_limit:
                logger.warning(f"⏳ Rate limit hit on {operation_name} (attempt {attempt + 1}/{max_retries}). Sleeping {delay:.1f}s...")
            elif is_timeout:
                logger.warning(f"⏳ Timeout on {operation_name} (attempt {attempt + 1}/{max_retries}). Sleeping {delay:.1f}s...")
            else:
                logger.warning(f"⏳ Server error on {operation_name} (attempt {attempt + 1}/{max_retries}). Sleeping {delay:.1f}s...")

            # Sleep with exponential backoff
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    # Should never reach here, but just in case
    raise last_exception if last_exception else Exception(f"{operation_name} failed after {max_retries} retries")


def with_retry(
    max_retries: int = 10,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    retry_on_timeout: bool = True,
    operation_name: Optional[str] = None
):
    """
    Decorator for automatic retry with exponential backoff.

    Usage:
        @with_retry(max_retries=5, initial_delay=1.0)
        def my_api_call():
            return llm.invoke(...)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            return retry_with_backoff(
                func=lambda: func(*args, **kwargs),
                max_retries=max_retries,
                initial_delay=initial_delay,
                max_delay=max_delay,
                retry_on_timeout=retry_on_timeout,
                operation_name=op_name
            )
        return wrapper
    return decorator
