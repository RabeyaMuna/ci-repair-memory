"""
Tracked LLM Wrapper - Drop-in replacement for LLM calls with automatic tracking
"""

import time
from typing import Any, List, Optional
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from .llm_tracker import tracker


class TrackedLLM:
    """
    Wrapper around ChatOpenAI that automatically tracks token usage, cost, and timing.

    Usage:
        # Instead of:
        llm = get_llm("gpt-4o")
        response = llm.invoke([HumanMessage(content=prompt)])

        # Use:
        llm = TrackedLLM(get_llm("gpt-4o"), model_name="gpt-4o", module="patch_generation")
        response = llm.invoke([HumanMessage(content=prompt)])
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        model_name: str,
        module: str = "unknown",
        function: str = "llm_call"
    ):
        """
        Args:
            llm: The underlying ChatOpenAI instance
            model_name: Model name for tracking (e.g., "gpt-4o", "deepseek-v4-flash")
            module: Module name for tracking (e.g., "patch_generation", "fault_localization")
            function: Function name for tracking (optional)
        """
        self.llm = llm
        self.model_name = model_name
        self.module = module
        self.function = function

    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invoke the LLM with automatic tracking.
        Returns the full response object (not just .content) so usage metadata is preserved.
        """
        start_time = time.time()
        error = None
        success = True

        try:
            # Call the actual LLM
            response = self.llm.invoke(messages, **kwargs)
            duration = time.time() - start_time

            # Extract token usage from response metadata
            input_tokens = 0
            output_tokens = 0

            # Try different metadata formats (OpenAI vs DeepSeek)
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
            elif hasattr(response, 'usage_metadata'):
                input_tokens = response.usage_metadata.get('input_tokens', 0)
                output_tokens = response.usage_metadata.get('output_tokens', 0)

            # Track the call
            tracker.track_call(
                module=self.module,
                function=self.function,
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration=duration,
                success=True
            )

            return response

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            success = False

            # Track the failed call
            tracker.track_call(
                module=self.module,
                function=self.function,
                model=self.model_name,
                input_tokens=0,
                output_tokens=0,
                duration=duration,
                success=False,
                error=error
            )

            raise

    def __getattr__(self, name):
        """Forward all other attributes to the underlying LLM"""
        return getattr(self.llm, name)


def create_tracked_llm(
    llm: ChatOpenAI,
    model_name: str,
    module: str = "unknown",
    function: str = "llm_call"
) -> TrackedLLM:
    """
    Factory function to create a tracked LLM wrapper.

    Args:
        llm: The underlying ChatOpenAI instance
        model_name: Model name for tracking
        module: Module name for tracking
        function: Function name for tracking

    Returns:
        TrackedLLM instance
    """
    return TrackedLLM(llm, model_name, module, function)
