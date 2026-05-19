# utilities/llm_provider.py

import inspect
import os
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

if TYPE_CHECKING:
    from utilities.token_tracker import TokenTracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # you must set this in .env
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "dummy-local-key")  # fallback
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://openrouter.ai/api/v1")
MEMCI_LLM_MODEL = os.getenv("MEMCI_LLM_MODEL", "").strip()

@dataclass
class LLMInfo:
    """Configuration for one logical LLM."""
    provider: str
    model_name: str
    temperature: float = 0.0
    base_url: str | None = None
    api_key_env: str | None = None  # which env var holds the key (if not OPENAI_API_KEY)
    api_key: str | None = None      # direct key value (optional)


# Here you list *all* models you want to use in your system
LLM_REGISTRY: Dict[str, LLMInfo] = {
    # OpenAI models
    "gpt-4o-mini": LLMInfo(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    "gpt4o": LLMInfo(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    "gpt-4.1": LLMInfo(
        provider="openai",
        model_name="gpt-4.1",  # change to the exact model name you use
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    
    "gpt-5.1": LLMInfo(
        provider="openai",
        model_name="gpt-5.1",   # exact API name
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    
    "gpt-5-mini": LLMInfo(
        provider="openai",
        model_name="gpt-5-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),

    # DeepSeek via OpenAI-compatible API
    "deepseek-coder": LLMInfo(
        provider="deepseek",
        model_name="deepseek-coder", # Changed to the correct model name for the API
        temperature=0.0,
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
    ),

    
    "deepseek-chat": LLMInfo(
        provider="deepseek",
        model_name="deepseek-chat",
        temperature=0.1,
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
    ),

    # MiniMax via OpenRouter
    "minimax/minimax-m2.5": LLMInfo(
        provider="openrouter",
        model_name="minimax/minimax-m2.5",
        temperature=0.0,
        base_url=MINIMAX_BASE_URL,
        api_key=MINIMAX_API_KEY,
    ),
    "MiniMax-M2.5": LLMInfo(
        provider="openrouter",
        model_name="minimax/minimax-m2.5",
        temperature=0.0,
        base_url=MINIMAX_BASE_URL,
        api_key=MINIMAX_API_KEY,
    ),
    "minimax-m2.5": LLMInfo(
        provider="openrouter",
        model_name="minimax/minimax-m2.5",
        temperature=0.0,
        base_url=MINIMAX_BASE_URL,
        api_key=MINIMAX_API_KEY,
    ),


}


MODEL_ALIASES: Dict[str, str] = {
    "minimax-m2.5": "minimax/minimax-m2.5",
    "MiniMax-M2.5": "minimax/minimax-m2.5",
    "MiniMax M2.5": "minimax/minimax-m2.5",
}


def resolve_model_key(model_key: str | None) -> str:
    raw = str(model_key or "").strip()
    if not raw and MEMCI_LLM_MODEL:
        raw = MEMCI_LLM_MODEL
    return MODEL_ALIASES.get(raw, raw)


def get_default_model_key(default: str = "gpt-5-mini") -> str:
    configured = str(MEMCI_LLM_MODEL or "").strip()
    return configured or default


def filesystem_safe_model_key(model_key: str | None) -> str:
    return str(model_key or "").strip().replace("/", "__")


class TrackedLLM:
    """
    Transparent wrapper around ChatOpenAI that intercepts every invoke() call,
    extracts token-usage metadata from the response, and records it in a
    TokenTracker instance.

    Drop-in replacement: agents continue to call ``self.llm.invoke(...)``
    unchanged — no agent code modifications are required.

    Auto-detects the calling method name from the Python call stack so each
    call is labelled with the agent method that triggered it
    (e.g. ``ci_log_analysis``, ``_select_suspicious_files``).
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        tracker: "TokenTracker",
        agent_name: str,
        model_name: str = "",
    ):
        self._llm = llm
        self._tracker = tracker
        self._agent_name = agent_name
        self._model_name = model_name

    # ------------------------------------------------------------------
    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        response = self._llm.invoke(input, *args, **kwargs)

        # --- extract token counts from response metadata ---------------
        input_tokens = 0
        output_tokens = 0

        # LangChain ≥ 0.2 exposes usage_metadata directly on AIMessage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            input_tokens = um.get("input_tokens", 0)
            output_tokens = um.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            tu = response.response_metadata.get("token_usage", {})
            input_tokens = tu.get("prompt_tokens", 0)
            output_tokens = tu.get("completion_tokens", 0)

        # --- extract full prompt text ----------------------------------
        prompt_text = ""
        try:
            if isinstance(input, str):
                prompt_text = input
            elif isinstance(input, list):
                # LangChain message list: concat content of each message
                parts = []
                for msg in input:
                    if hasattr(msg, "content"):
                        parts.append(str(msg.content))
                    elif isinstance(msg, dict):
                        parts.append(str(msg.get("content", "")))
                prompt_text = "\n".join(parts)
        except Exception:
            pass

        # --- extract full response text --------------------------------
        response_text = ""
        try:
            if hasattr(response, "content"):
                response_text = str(response.content)
            elif isinstance(response, str):
                response_text = response
        except Exception:
            pass

        # --- auto-detect the agent method that called invoke() ---------
        call_site = self._agent_name  # fallback
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_name = frame.f_back.f_code.co_name
                # Skip generic wrappers; use the first real agent method
                call_site = f"{self._agent_name}.{caller_name}"
        except Exception:
            pass

        self._tracker.record(
            agent=self._agent_name,
            call_site=call_site,
            model=self._model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt_text,
            response=response_text,
        )
        return response

    # Forward every other attribute access (e.g. .model_name, .temperature)
    # to the underlying ChatOpenAI instance so duck-typing is preserved.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def get_llm(model_key: str) -> ChatOpenAI:
    """
    Return a ChatOpenAI-like instance for the given logical model_key.
    Same function is used everywhere in the system.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    model_key = resolve_model_key(model_key)

    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model_key: {model_key}")

    info = LLM_REGISTRY[model_key]

    # Decide which API key to use
    api_key = None
    if info.api_key_env:
        api_key = os.getenv(info.api_key_env)
    elif info.api_key:
        api_key = info.api_key
    elif info.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    elif info.provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    elif info.provider == "openrouter":
        api_key = os.getenv("MINIMAX_API_KEY") or MINIMAX_API_KEY

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
        base_url = os.getenv("MINIMAX_BASE_URL") or base_url or "https://openrouter.ai/api/v1"
    if base_url:
        kwargs["base_url"] = base_url

    if not kwargs.get("api_key"):
        raise ValueError(
            f"Missing API key for model '{model_key}'. "
            f"Provider={info.provider}. Check your .env configuration."
        )

    return ChatOpenAI(**kwargs)


def get_tracked_llm(
    model_key: str,
    tracker: "TokenTracker",
    agent_name: str,
) -> TrackedLLM:
    """
    Return a TrackedLLM that wraps the ChatOpenAI for ``model_key`` and
    automatically records every invoke() call in ``tracker``.

    Parameters
    ----------
    model_key  : key from LLM_REGISTRY (e.g. "gpt-5-mini")
    tracker    : a TokenTracker instance shared across the pipeline
    agent_name : human-readable agent label (e.g. "CILogAnalyzerLLM")
    """
    llm = get_llm(model_key)
    return TrackedLLM(llm=llm, tracker=tracker, agent_name=agent_name, model_name=model_key)
