# utilities/model_token_limits.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class ModelLimits:
    context_window: int
    reserved_output: int = 8_000
    safety_buffer: int = 2_000
    max_output: Optional[int] = None  # hard cap for completion tokens (if known)

    def prompt_budget(self) -> int:
        return max(8_000, self.context_window - self.reserved_output - self.safety_buffer)

    def output_budget(self, prompt_tokens: int) -> int:
        """
        Max output tokens allowed given prompt_tokens, respecting context window.
        """
        remaining = self.context_window - prompt_tokens - self.safety_buffer
        remaining = max(0, remaining)
        if self.max_output is not None:
            return min(remaining, self.max_output)
        return remaining

MODEL_LIMITS: Dict[str, ModelLimits] = {
    # OpenAI
    "gpt-4o-mini": ModelLimits(context_window=128_000, reserved_output=8_000, max_output=16_384),
    "gpt-4o":      ModelLimits(context_window=128_000, reserved_output=8_000, max_output=16_384),
    "gpt-4.1":     ModelLimits(context_window=1_000_000, reserved_output=16_000, max_output=32_768),
    "gpt-5-mini":  ModelLimits(context_window=400_000, reserved_output=16_000, max_output=128_000),
    "gpt-5.1":     ModelLimits(context_window=400_000, reserved_output=16_000, max_output=128_000),

    # DeepSeek (adjust if needed)
    "deepseek-chat":  ModelLimits(context_window=128_000, reserved_output=8_000, max_output=8_000),
    "deepseek-coder": ModelLimits(context_window=128_000, reserved_output=8_000, max_output=8_000),
    "deepseek-v4-flash": ModelLimits(context_window=163_840, reserved_output=8_000, max_output=8_000),  # Actual limit via OpenRouter
}

def get_model_limits(model_name: Optional[str]) -> ModelLimits:
    key = (model_name or "").strip()
    return MODEL_LIMITS.get(key, ModelLimits(context_window=128_000, reserved_output=8_000, max_output=16_384))

def get_prompt_token_budget(model_name: Optional[str]) -> int:
    return get_model_limits(model_name).prompt_budget()

def get_max_output_tokens(model_name: Optional[str], prompt_tokens: int) -> int:
    return get_model_limits(model_name).output_budget(prompt_tokens)

def get_chunk_threshold_simple(model_name: Optional[str]) -> int:
    limits = get_model_limits(model_name)
    ctx = limits.context_window

    # DeepSeek V4 Flash: 160k context, use conservative threshold
    if model_name and "deepseek-v4-flash" in model_name.lower():
        return 80_000  # Half of 160k context window

    # Other DeepSeek models: use smaller chunks to reduce input token costs
    if model_name and "deepseek" in model_name.lower():
        return 30_000  # More aggressive chunking for cost efficiency

    # Large-context models (GPT-5-mini, GPT-5.1, GPT-4.1, etc.)
    if ctx >= 300_000:
        return 200_000

    # Standard 128k models
    return 70_000 

def is_large_context_model(model_name: Optional[str]) -> bool:
    limits = get_model_limits(model_name)
    return limits.context_window >= 300_000

def get_fl_chunk_threshold(model_name: Optional[str]) -> tuple[int, int]:
    """
    Get fault localization chunk size and overlap thresholds for a given model.

    Returns:
        tuple[chunk_size, overlap]: (lines per chunk, overlap lines)
    """
    limits = get_model_limits(model_name)
    ctx = limits.context_window

    # DeepSeek V4 Flash: 160k context, use moderate chunks
    if model_name and "deepseek-v4-flash" in model_name.lower():
        return (400, 50)  # Moderate for 160k context

    # Other DeepSeek models: moderate chunks
    if model_name and "deepseek" in model_name.lower():
        return (300, 50)  # Standard for 128k context

    # Large-context models (GPT-5-mini, GPT-5.1, GPT-4.1, etc.)
    if ctx >= 300_000:
        return (600, 50)  # Large chunks for 300k+ context

    # Standard 128k models (GPT-4o, GPT-4o-mini)
    return (300, 50)  # Default for most models
