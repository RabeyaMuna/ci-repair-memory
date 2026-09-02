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

    # Large-context models (GPT-5-mini, GPT-5.1, GPT-4.1, etc.)
    if ctx >= 300_000:
        return 200_000

    # Standard 128k models
    return 70_000 

def is_large_context_model(model_name: Optional[str]) -> bool:
    limits = get_model_limits(model_name)
    return limits.context_window >= 300_000
