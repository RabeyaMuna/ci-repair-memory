# utilities/llm_provider.py

import inspect
import os
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

if TYPE_CHECKING:
    from utilities.token_tracker import TokenTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# API Keys and URLs for MiniMax and GLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://api.z.ai/api/paas/v4")
GLM_MODEL_NAME = os.getenv("GLM_MODEL_NAME", "glm-5.2")

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass
class LLMInfo:
    """Configuration for one logical LLM."""

    provider: str
    model_name: str
    temperature: float = 0.0
    base_url: str | None = None
    api_key_env: str | None = None  # which env var holds the key
    api_key: str | None = None  # direct key value (optional)


# Registry of models (MiniMax-M2.5 and GLM-5.2)
LLM_REGISTRY: Dict[str, LLMInfo] = {
    # MiniMax M2.5 via OpenRouter
    "minimax/minimax-m2.5": LLMInfo(
        provider="openrouter",
        model_name="minimax/minimax-m2.5",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "minimax-m2.5": LLMInfo(
        provider="openrouter",
        model_name="minimax/minimax-m2.5",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "MiniMax-M2.5": LLMInfo(
        provider="openrouter",
        model_name="minimax/minimax-m2.5",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    # GLM-5.2 via native Z.ai-compatible endpoint
    "glm-5.2": LLMInfo(
        provider="zai",
        model_name=GLM_MODEL_NAME,
        temperature=0.0,
        base_url=GLM_BASE_URL,
        api_key=GLM_API_KEY,
    ),
    "z-ai/glm-5.2": LLMInfo(
        provider="zai",
        model_name=GLM_MODEL_NAME,
        temperature=0.0,
        base_url=GLM_BASE_URL,
        api_key=GLM_API_KEY,
    ),
    "GLM-5.2": LLMInfo(
        provider="zai",
        model_name=GLM_MODEL_NAME,
        temperature=0.0,
        base_url=GLM_BASE_URL,
        api_key=GLM_API_KEY,
    ),
    # GPT-5.4-mini via OpenAI
    "gpt-5.4-mini": LLMInfo(
        provider="openai",
        model_name="gpt-5.4-mini",
        temperature=0.0,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    ),
    # DeepSeek-V4-Flash via OpenRouter
    "deepseek-v4-flash": LLMInfo(
        provider="openrouter",
        model_name="deepseek/deepseek-v4-flash",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "deepseek/deepseek-v4-flash": LLMInfo(
        provider="openrouter",
        model_name="deepseek/deepseek-v4-flash",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
    "openrouter/deepseek/deepseek-v4-flash": LLMInfo(
        provider="openrouter",
        model_name="deepseek/deepseek-v4-flash",
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    ),
}


# Aliases for convenience
MODEL_ALIASES: Dict[str, str] = {
    "minimax-m2.5": "minimax/minimax-m2.5",
    "minimax2.5": "minimax/minimax-m2.5",  # Added for convenience
    "MiniMax-M2.5": "minimax/minimax-m2.5",
    "MiniMax M2.5": "minimax/minimax-m2.5",
    "glm": "glm-5.2",
    "glm5.2": "glm-5.2",
    "gml": "glm-5.2",
    "gml5.2": "glm-5.2",
    "gml-5.2": "glm-5.2",
    "GLM": "glm-5.2",
    "GLM-5.2": "glm-5.2",
    # GPT-5.4 aliases (all map to gpt-5.4-mini)
    "gpt-5.4": "gpt-5.4-mini",
    "gpt5.4": "gpt-5.4-mini",
    "gpt5.4-mini": "gpt-5.4-mini",
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
    "gpt5.4-mini-2026-03-17": "gpt-5.4-mini",
    "gpt54": "gpt-5.4-mini",
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt5-mini": "gpt-5.4-mini",
    "gpt5mini": "gpt-5.4-mini",
    "gpt_5_mini": "gpt-5.4-mini",
    # DeepSeek aliases
    "deepseek-v4-flash": "deepseek-v4-flash",
}


def resolve_model_key(model_key: str | None) -> str:
    """Resolve model key with aliases."""
    raw = str(model_key or "").strip()
    return MODEL_ALIASES.get(raw, raw)


def get_default_model_key(default: str | None = None) -> str:
    """Return an explicit caller-provided model key."""
    if not default:
        raise ValueError("Model is required. Use minimax2.5 or glm5.2.")
    return default


def filesystem_safe_model_key(model_key: str | None) -> str:
    """Convert model key to filesystem-safe string."""
    return str(model_key or "").strip().replace("/", "__")


class TrackedLLM:
    """
    Transparent wrapper around ChatOpenAI that intercepts invoke() calls
    and records token usage in a TokenTracker.

    IMPORTANT: Also adds a `memci_model_key` attribute so downstream code
    can extract the logical model name for token limit configuration.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tracker: "TokenTracker",
        agent_name: str,
        model_key: str = "",
    ):
        self._llm = llm
        self._tracker = tracker
        self._agent_name = agent_name
        self._model_key = model_key

        # CRITICAL: Store model_key so L2 analysis can extract it
        self.memci_model_key = model_key

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        response = self._llm.invoke(input, *args, **kwargs)

        # Extract token counts
        input_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            input_tokens = um.get("input_tokens", 0)
            output_tokens = um.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            tu = response.response_metadata.get("token_usage", {})
            input_tokens = tu.get("prompt_tokens", 0)
            output_tokens = tu.get("completion_tokens", 0)

        # Extract prompt text
        prompt_text = ""
        try:
            if isinstance(input, str):
                prompt_text = input
            elif isinstance(input, list):
                parts = []
                for msg in input:
                    if hasattr(msg, "content"):
                        parts.append(str(msg.content))
                    elif isinstance(msg, dict):
                        parts.append(str(msg.get("content", "")))
                prompt_text = "\n".join(parts)
        except Exception:
            pass

        # Extract response text
        response_text = ""
        try:
            if hasattr(response, "content"):
                response_text = str(response.content)
            elif isinstance(response, str):
                response_text = response
        except Exception:
            pass

        # Auto-detect calling method
        call_site = self._agent_name
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_name = frame.f_back.f_code.co_name
                call_site = f"{self._agent_name}.{caller_name}"
        except Exception:
            pass

        self._tracker.record(
            agent=self._agent_name,
            call_site=call_site,
            model=self._model_key,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt_text,
            response=response_text,
        )
        return response

    # Forward all other attributes to underlying LLM
    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def get_llm(model_key: str) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance for the given model_key.

    The returned LLM will have a `memci_model_key` attribute
    that can be used to extract the model name for configuration.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    model_key = resolve_model_key(model_key)

    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model_key: {model_key}")

    info = LLM_REGISTRY[model_key]

    # Decide API key (MiniMax, GLM, or OpenAI)
    api_key = None
    if info.api_key_env:
        api_key = os.getenv(info.api_key_env)
    elif info.api_key:
        api_key = info.api_key
    elif info.provider == "openrouter":
        if "minimax" in model_key.lower():
            api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
        else:
            api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    elif info.provider == "zai":
        api_key = os.getenv("GLM_API_KEY") or GLM_API_KEY
    elif info.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY

    kwargs = {
        "model": info.model_name,
        "temperature": info.temperature,
        "request_timeout": 120,
        "max_retries": 1,
    }

    if api_key:
        kwargs["api_key"] = api_key

    base_url = info.base_url
    if info.provider == "openrouter":
        if "minimax" in model_key.lower():
            base_url = (
                os.getenv("OPENROUTER_BASE_URL")
                or base_url
                or "https://openrouter.ai/api/v1"
            )
    elif info.provider == "zai":
        base_url = (
            os.getenv("GLM_BASE_URL") or base_url or "https://api.z.ai/api/paas/v4"
        )
    elif info.provider == "openai":
        base_url = (
            os.getenv("OPENAI_BASE_URL") or base_url or "https://api.openai.com/v1"
        )

    if base_url:
        kwargs["base_url"] = base_url

    if not kwargs.get("api_key"):
        raise ValueError(
            f"Missing API key for model '{model_key}'. "
            f"Provider={info.provider}. Check your .env configuration."
        )

    llm = ChatOpenAI(**kwargs)

    # CRITICAL: Attach model_key to LLM object
    # Use object.__setattr__ to bypass Pydantic validation
    object.__setattr__(llm, 'memci_model_key', model_key)

    return llm


def get_tracked_llm(
    model_key: str,
    tracker: "TokenTracker",
    agent_name: str,
) -> TrackedLLM:
    """
    Return a TrackedLLM that wraps ChatOpenAI and records token usage.

    The returned LLM will have a `memci_model_key` attribute for
    extracting the model name.
    """
    llm = get_llm(model_key)
    return TrackedLLM(
        llm=llm, tracker=tracker, agent_name=agent_name, model_key=model_key
    )
