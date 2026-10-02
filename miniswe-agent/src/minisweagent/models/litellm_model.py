import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

        # Fix max_tokens -> max_completion_tokens for GPT-4o and newer models
        self._fix_max_tokens_parameter()

    @staticmethod
    def _requires_max_completion_tokens(model_name: str) -> bool:
        """
        Check if a model requires max_completion_tokens instead of max_tokens.

        Returns True for:
        - GPT-5 series (gpt-5.x, gpt-5.4-mini, etc.)
        - GPT-4o series (gpt-4o, gpt-4o-mini, chatgpt-4o-latest, etc.)
        - o-series reasoning models (o1, o3, o3-mini, etc.)

        Returns False for:
        - Codex models (codex, code-davinci-002, etc.)
        - GPT-4 non-o models (gpt-4, gpt-4-turbo, etc.)
        - GPT-3.5 models (gpt-3.5-turbo, etc.)
        - All other models
        """
        normalized = str(model_name).lower().removeprefix("openai/").removeprefix("azure/")

        # GPT-5 series
        if normalized.startswith("gpt-5"):
            return True

        # GPT-4o series (but not gpt-4 without 'o')
        if normalized.startswith("gpt-4o") or normalized.startswith("chatgpt-4o"):
            return True

        # o-series reasoning models (o1, o3, etc.)
        # Must start with 'o' followed by a digit
        if len(normalized) > 1 and normalized[0] == "o" and normalized[1].isdigit():
            return True

        # All other models use max_tokens
        return False

    def _fix_max_tokens_parameter(self):
        """Convert max_tokens to max_completion_tokens for compatible models."""
        if not self._requires_max_completion_tokens(self.config.model_name):
            # Model uses max_tokens (codex, gpt-4, gpt-3.5, etc.) - no conversion needed
            return

        if "max_tokens" in self.config.model_kwargs:
            max_tokens_value = self.config.model_kwargs.pop("max_tokens")
            self.config.model_kwargs["max_completion_tokens"] = max_tokens_value
            logger.info(
                "[Model Init] Converted max_tokens=%s -> max_completion_tokens for: %s",
                max_tokens_value,
                self.config.model_name
            )

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            # Fix max_tokens in kwargs too (they can override model_kwargs)
            if self._requires_max_completion_tokens(self.config.model_name) and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                logger.debug("[Query] Converted max_tokens in kwargs for: %s", self.config.model_name)

            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def invoke(self, messages):
        """
        LangChain-compatible invoke method for CI log analyzer.

        Accepts either:
        - A list of message dicts or LangChain message objects
        - A plain string prompt

        Returns an object with a .content attribute.
        """
        # Convert to message format
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            msg_list = []
            for m in messages:
                # Handle LangChain messages
                if hasattr(m, 'content'):
                    role = "assistant" if hasattr(m, '__class__') and 'AI' in m.__class__.__name__ else "user"
                    msg_list.append({"role": role, "content": m.content})
                elif isinstance(m, dict):
                    msg_list.append(m)
                else:
                    msg_list.append({"role": "user", "content": str(m)})
        else:
            msg_list = [{"role": "user", "content": str(messages)}]

        # Call query method
        response = self.query(msg_list)

        # Return object with .content attribute
        class Response:
            def __init__(self, content):
                self.content = content

        return Response(response.get("content", ""))

    def _calculate_cost(self, response) -> dict[str, float]:
        """
        Calculate cost with priority:
        1. Actual cost from API response (OpenRouter provides this)
        2. litellm's calculator (for models in its database)
        3. Manual calculation using custom pricing
        4. 0.0 if ignore_errors is set
        """
        cost = 0.0

        # Try to get actual cost from OpenRouter API response
        try:
            if hasattr(response, '_hidden_params') and response._hidden_params:
                actual_cost = response._hidden_params.get('response_cost')
                if actual_cost and float(actual_cost) > 0:
                    return {"cost": float(actual_cost)}
        except (AttributeError, KeyError, ValueError, TypeError):
            pass

        # Try litellm's built-in calculator
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost > 0.0:
                return {"cost": cost}
        except Exception:
            pass

        # Try manual calculation with custom pricing
        try:
            # Get token usage
            usage = getattr(response, 'usage', None)
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)

                # Custom pricing for unmapped models
                model_pricing = {
                    "deepseek/deepseek-v4-flash": (0.000000165, 0.0000006),
                    "openrouter/deepseek/deepseek-v4-flash": (0.000000165, 0.0000006),
                    "minimax/minimax-m2.5": (0.0000002, 0.0000006),
                    "openrouter/minimax/minimax-m2.5": (0.0000002, 0.0000006),
                    "z-ai/glm-5.2": (0.0000004186, 0.000001316),
                    "openrouter/z-ai/glm-5.2": (0.0000004186, 0.000001316),
                    "glm-5.2": (0.0000004186, 0.000001316),
                }

                if self.config.model_name in model_pricing:
                    input_cost, output_cost = model_pricing[self.config.model_name]
                    cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
                    if cost > 0:
                        return {"cost": cost}
        except Exception:
            pass

        # If all methods failed and ignore_errors is NOT set, raise error
        if cost <= 0.0 and self.config.cost_tracking != "ignore_errors":
            msg = (
                f"Error calculating cost for model {self.config.model_name}: Model not in pricing database. "
                "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                "Alternatively check the 'Cost tracking' section in the documentation at "
                "https://klieret.short.gy/mini-local-models. "
                " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
            )
            logger.critical(msg)
            raise RuntimeError(msg)

        return {"cost": cost}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
