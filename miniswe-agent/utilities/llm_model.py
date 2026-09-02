"""
LitellmModel - Unified LLM Wrapper
==================================

Provides a consistent interface for LLM calls across the codebase.
Handles model credentials, token limits, retry logic, and error handling.

Usage:
    llm = LitellmModel("openrouter/minimax/minimax-m2.5")
    result = llm.invoke("Your prompt here")
    content = result.content
"""

import logging
import os
import time
import warnings
from typing import Any

import litellm

from utilities.model_registry import resolve_model_alias
from utilities.model_token_config import get_output_safe_tokens
from utilities.run_metrics import safe_metrics_call

# Suppress Pydantic serialization warnings from LiteLLM responses
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
    module="pydantic.main"
)

LOGGER = logging.getLogger(__name__)


class LitellmModel:
    """
    Small invoke-compatible wrapper for LLM calls.

    Features:
    - Automatic credential detection based on model name
    - Auto-detection of token limits based on model
    - Consistent error handling and logging
    - Compatible with CILogAnalyzer and other utilities
    """

    def __init__(self, model_name: str, metrics_recorder: Any = None):
        self.model_name = self._normalize_model_name(model_name)
        self.api_key, self.api_base = self._model_credentials(self.model_name)
        self.metrics_recorder = metrics_recorder

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        """Normalize and resolve model aliases."""
        raw_model_name = str(model_name or "").strip()
        resolved = resolve_model_alias(model_name)
        if resolved:
            return resolved
        return raw_model_name

    @staticmethod
    def _model_credentials(model_name: str) -> tuple[str | None, str | None]:
        """Get API credentials based on model name."""
        lowered = str(model_name or "").lower()

        # OpenRouter models
        if lowered.startswith("openrouter/"):
            return (
                os.getenv("OPENROUTER_API_KEY"),
                os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            )

        # GLM models
        if "glm" in lowered or "z-ai" in lowered:
            return (
                os.getenv("GLM_API_KEY"),
                os.getenv("GLM_BASE_URL") or "https://api.z.ai/api/paas/v4",
            )

        # Minimax models (via OpenRouter)
        if "minimax" in lowered:
            return (
                os.getenv("OPENROUTER_API_KEY"),
                os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            )

        # OpenAI models must never inherit OpenRouter credentials merely because
        # both keys are present in the project .env file. An explicit
        # ``openrouter/`` prefix above remains the opt-in route through
        # OpenRouter.
        openai_name = lowered.removeprefix("openai/")
        if openai_name.startswith(("gpt-", "chatgpt-", "codex-")) or (
            len(openai_name) > 1
            and openai_name[0] == "o"
            and openai_name[1].isdigit()
        ):
            return (
                os.getenv("OPENAI_API_KEY"),
                os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            )

        # Do not silently send unknown models to a billable provider. Let
        # LiteLLM resolve an explicitly prefixed provider or report a clear
        # configuration error.
        return (None, None)

    def invoke(
        self,
        prompt: Any,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ):
        """
        Invoke the LLM with a prompt.

        Args:
            prompt: String prompt or list of messages
            max_tokens: Maximum output tokens (auto-detected if None)
            reasoning_effort: Optional provider-normalized reasoning level
            response_format: Optional structured-output format

        Returns:
            Result object with .content and .raw_response attributes
        """
        # Convert prompt to messages format
        if isinstance(prompt, list):
            messages = [
                {
                    "role": "user",
                    "content": str(getattr(message, "content", message)),
                }
                for message in prompt
            ]
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        try:
            start_time = time.time()

            # Auto-detect max_tokens based on model if not specified
            if max_tokens is None:
                try:
                    max_tokens = get_output_safe_tokens(self.model_name)
                    LOGGER.debug(
                        f"Auto-detected max_tokens={max_tokens} for model={self.model_name}"
                    )
                except Exception:
                    max_tokens = 16000  # Fallback

            # Special handling for z-ai models
            if str(self.model_name).lower().startswith("zai/"):
                max_tokens = min(int(max_tokens), 120000)

            # Build completion kwargs
            # GPT-5 models require temperature=1, others can use temperature=0
            model_lower = str(self.model_name).lower()
            temperature = 1 if "gpt-5" in model_lower else 0

            # For GLM models, use openai/ prefix with custom base URL
            # This tells LiteLLM to use OpenAI-compatible format
            model_for_litellm = self.model_name
            if "glm" in model_lower and not model_lower.startswith("openai/"):
                # Use the actual model name from env or default
                actual_model = os.getenv("GLM_MODEL_NAME", "glm-5.2")
                # Strip any provider prefix for the actual API call
                if "/" in actual_model:
                    actual_model = actual_model.split("/")[-1]
                model_for_litellm = f"openai/{actual_model}"

            # GPT-5.x models use max_completion_tokens, older models use max_tokens
            if "gpt-5" in model_lower:
                completion_kwargs = {
                    "model": model_for_litellm,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "timeout": int(os.getenv("LITELLM_TIMEOUT", "600")),
                }
            else:
                completion_kwargs = {
                    "model": model_for_litellm,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": int(os.getenv("LITELLM_TIMEOUT", "600")),
                }
            if reasoning_effort:
                completion_kwargs["reasoning_effort"] = reasoning_effort
            if response_format:
                completion_kwargs["response_format"] = response_format
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            if self.api_base:
                completion_kwargs["api_base"] = self.api_base

            # Make API call
            safe_metrics_call(
                self.metrics_recorder,
                "begin_api_call",
                phase="context_llm",
                model=self.model_name,
            )
            response = litellm.completion(**completion_kwargs)
            elapsed = time.time() - start_time

            if self.metrics_recorder is not None:
                safe_metrics_call(
                    self.metrics_recorder,
                    "record_response",
                    response=response,
                    phase="context_llm",
                    model=self.model_name,
                    duration_seconds=elapsed,
                )

            # Check finish_reason
            finish_reason = getattr(response.choices[0], "finish_reason", None)

            # Log token usage
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", "?")
                completion_tokens = getattr(usage, "completion_tokens", "?")
                total_tokens = getattr(usage, "total_tokens", "?")
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                if isinstance(prompt_details, dict):
                    cached_tokens = prompt_details.get("cached_tokens")
                    cache_write_tokens = prompt_details.get("cache_write_tokens")
                else:
                    cached_tokens = getattr(prompt_details, "cached_tokens", None)
                    cache_write_tokens = getattr(
                        prompt_details, "cache_write_tokens", None
                    )
                completion_details = getattr(
                    usage, "completion_tokens_details", None
                )
                if isinstance(completion_details, dict):
                    reasoning_tokens = completion_details.get("reasoning_tokens")
                else:
                    reasoning_tokens = getattr(
                        completion_details, "reasoning_tokens", None
                    )
                reasoning_suffix = (
                    f", reasoning={reasoning_tokens}"
                    if reasoning_tokens is not None
                    else ""
                )
                cache_suffix = (
                    f", cached_input={cached_tokens}"
                    if cached_tokens is not None
                    else ""
                )
                if cache_write_tokens is not None:
                    cache_suffix += f", cache_write={cache_write_tokens}"
                print(
                    f"      [API] finish_reason={finish_reason}, tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}{cache_suffix}{reasoning_suffix}"
                )
            else:
                print(f"      [API] finish_reason={finish_reason}, no usage data")

            # Handle error finish_reason
            if finish_reason == "error":
                error_msg = getattr(response, "error", "Unknown error")
                if hasattr(response, "_hidden_params"):
                    error_msg += f" | Details: {response._hidden_params}"
                LOGGER.error(f"LLM error after {elapsed:.1f}s. Error: {error_msg}")
                print(f"    FAIL LLM Error ({elapsed:.1f}s): {error_msg}")

            # Handle length limit
            elif finish_reason == "length":
                LOGGER.warning(
                    f"Hit length limit: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
                )
                print("    WARNING Length limit hit!")
                print(
                    f"       Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
                )
                print(f"       max_tokens setting: {max_tokens or 16000}")
                print(
                    "       Generation exhausted its output budget; inspect reasoning usage before splitting input"
                )

            # Return result
            class Result:
                content = response.choices[0].message.content or ""
                raw_response = response

            return Result()

        except Exception as e:
            elapsed = time.time() - start_time if "start_time" in locals() else 0.0
            if self.metrics_recorder is not None:
                safe_metrics_call(
                    self.metrics_recorder,
                    "record_api_call",
                    phase="context_llm",
                    model=self.model_name,
                    duration_seconds=elapsed,
                    status="failed",
                    error=str(e),
                )
            LOGGER.error(f"LiteLLM API call failed: {type(e).__name__}: {e}")
            print(f"    FAIL API Error: {type(e).__name__}: {str(e)[:200]}")

            # Return error result
            class Result:
                content = ""
                error = str(e)
                raw_response = None

            return Result()

    def __call__(self, prompt: Any):
        """
        Make LitellmModel callable for compatibility with CILogAnalyzer.

        Example:
            llm = LitellmModel("minimax2.5")
            content = llm("What is 2+2?")
        """
        result = self.invoke(prompt)
        return result.content
