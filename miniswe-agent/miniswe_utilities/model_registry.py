"""Shared model aliases for CI-Repair-Bench commands.

The codebase passes model names through LiteLLM. For OpenRouter-hosted models,
LiteLLM expects the provider prefix: ``openrouter/<openrouter-model-slug>``.
"""

from __future__ import annotations

import os
import re


CANONICAL_MINIMAX_MODEL = "openrouter/minimax/minimax-m2.5"
CANONICAL_ZAI_GLM_MODEL = "zai/glm-5.2"
LEGACY_GLM_MODEL = "glm-4-plus"
# GPT-5.4 models (using official OpenAI API - always use latest)
# All aliases map to gpt-5.4-mini (latest version)
CANONICAL_GPT5_MINI_MODEL = "gpt-5.4-mini"
CANONICAL_GPT54_MODEL = "gpt-5.4-mini"
# DeepSeek-V4-Flash via OpenRouter
CANONICAL_DEEPSEEK_V4_FLASH_MODEL = "openrouter/deepseek/deepseek-v4-flash"

MINIMAX_ALIASES = {
    "minimax",
    "minimax2.5",
    "minimax-2.5",
    "minimaxm2.5",
    "minimax-m2.5",
    "minimax_m2.5",
}

GLM_ALIASES = {
    "glm",
    "gml",
    "gml5.2",
    "gml-5.2",
    "glm5.2",
    "glm-5.2",
    "glm_5.2",
    "glm_5_2",
    "glm-5-2",
}

GPT5_MINI_ALIASES = {
    "gpt-5-mini",
    "gpt5-mini",
    "gpt5mini",
    "gpt_5_mini",
}

GPT54_ALIASES = {
    "gpt-5.4",
    "gpt5.4",
    "gpt-5.4-mini",
    "gpt5.4-mini",
    "gpt-5.4-mini-2026-03-17",
    "gpt5.4-mini-2026-03-17",
    "gpt54",
}

DEEPSEEK_V4_FLASH_ALIASES = {
    "deepseek-v4-flash",
}


_ALIASES = {
    **{alias: CANONICAL_MINIMAX_MODEL for alias in MINIMAX_ALIASES},
    "minimax/minimax-m2.5": CANONICAL_MINIMAX_MODEL,
    "openrouter/minimax/minimax-m2.5": CANONICAL_MINIMAX_MODEL,
    **{alias: CANONICAL_ZAI_GLM_MODEL for alias in GLM_ALIASES},
    "z-ai/glm-5.2": CANONICAL_ZAI_GLM_MODEL,
    "zai/glm-5.2": CANONICAL_ZAI_GLM_MODEL,
    "openrouter/z-ai/glm-5.2": CANONICAL_ZAI_GLM_MODEL,
    # Keep the old GLM value valid for existing experiments.
    LEGACY_GLM_MODEL: LEGACY_GLM_MODEL,
    # GPT-5 models
    **{alias: CANONICAL_GPT5_MINI_MODEL for alias in GPT5_MINI_ALIASES},
    **{alias: CANONICAL_GPT54_MODEL for alias in GPT54_ALIASES},
    # DeepSeek-V4-Flash models
    **{alias: CANONICAL_DEEPSEEK_V4_FLASH_MODEL for alias in DEEPSEEK_V4_FLASH_ALIASES},
    "deepseek/deepseek-v4-flash": CANONICAL_DEEPSEEK_V4_FLASH_MODEL,
    "openrouter/deepseek/deepseek-v4-flash": CANONICAL_DEEPSEEK_V4_FLASH_MODEL,
}


def resolve_model_alias(model_name: str | None) -> str | None:
    """Resolve a user-facing model name to the provider string used by LiteLLM."""
    if model_name is None:
        return None

    raw = str(model_name).strip()
    if not raw:
        return raw

    lowered = raw.lower()

    if lowered in GLM_ALIASES:
        if os.getenv("GLM_MODEL_NAME"):
            return os.getenv("GLM_MODEL_NAME", "").strip()
        return CANONICAL_ZAI_GLM_MODEL

    if lowered in MINIMAX_ALIASES and os.getenv("MINIMAX_MODEL_NAME"):
        return os.getenv("MINIMAX_MODEL_NAME", "").strip()

    if lowered in _ALIASES:
        return _ALIASES[lowered]

    if lowered.startswith("minimax/"):
        return f"openrouter/{raw}"

    if lowered.startswith(("z-ai/", "zai/")):
        return raw

    if lowered.startswith("deepseek/"):
        return f"openrouter/{raw}"

    return raw


def model_output_name(model_name: str | None) -> str:
    """Return a stable, filesystem-friendly folder name for result paths."""
    resolved = resolve_model_alias(model_name) or "unknown-model"
    original = (model_name or "").lower().strip()

    if resolved == CANONICAL_MINIMAX_MODEL:
        return "minimax-m2.5"
    if resolved == CANONICAL_ZAI_GLM_MODEL:
        return "glm-5.2"
    if resolved == LEGACY_GLM_MODEL:
        return LEGACY_GLM_MODEL
    if resolved == CANONICAL_DEEPSEEK_V4_FLASH_MODEL:
        return "deepseek-v4-flash"

    # For GPT-5 models: preserve the original input to distinguish gpt-5.4-mini vs gpt-5-mini
    # Even though they resolve to the same underlying model, use different output names
    if resolved == "gpt-5.4-mini":
        # Check original input to determine output name
        if "5.4" in original or "5-4" in original:
            return "gpt-5.4-mini"
        else:
            return "gpt-5-mini"

    name = resolved
    if name.startswith("openrouter/"):
        name = name.removeprefix("openrouter/")
    name = name.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "model"


def configure_model_environment(model_name: str | None) -> str | None:
    """Resolve a model and expose it to memory/decomposition helpers."""
    resolved = resolve_model_alias(model_name)
    if resolved:
        os.environ["MEMCI_LLM_MODEL"] = resolved
    return resolved
