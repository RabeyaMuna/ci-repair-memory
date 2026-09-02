"""
utilities/llm_invoker.py - Robust LLM invocation with comprehensive retry logic

Centralizes retry logic for LLM API calls with handling for:
- Rate limits (exponential backoff)
- Timeouts (retry with increased timeout)
- Connection errors (brief retry)
- Empty responses
- Malformed JSON

Import and use across all scripts for consistent error handling.
"""
import re
import json
import logging
import os
import time
from typing import Any

try:
    import demjson3  # type: ignore
except Exception:
    demjson3 = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class LLMTransientConnectionError(RuntimeError):
    """A transport failure that remained unavailable after bounded retries."""


def _requires_max_completion_tokens(model_name: str) -> bool:
    """Check if model requires max_completion_tokens instead of max_tokens."""
    if not model_name:
        return False
    normalized = str(model_name).lower().removeprefix("openai/").removeprefix("azure/")
    # GPT-5 series
    if normalized.startswith("gpt-5"):
        return True
    # GPT-4o series
    if normalized.startswith("gpt-4o") or normalized.startswith("chatgpt-4o"):
        return True
    # o-series (o1, o3, etc.)
    if len(normalized) > 1 and normalized[0] == "o" and normalized[1].isdigit():
        return True
    return False


def get_model_max_output_tokens(model_name: str, default: int = 16000) -> int:
    """
    Get the maximum output tokens for a model based on its capabilities.

    Args:
        model_name: Model identifier (e.g., "deepseek-v4-flash", "gpt-4o-mini")
        default: Default value if model is unknown

    Returns:
        Maximum output tokens for the model
    """
    if not model_name:
        return default

    normalized = str(model_name).lower().removeprefix("openai/").removeprefix("azure/")

    # DeepSeek models - massive output capacity
    if "deepseek" in normalized:
        if "v4" in normalized or "v3" in normalized:
            return 384000  # DeepSeek V3/V4: 384K output
        return 100000  # Older DeepSeek: conservative 100K

    # GPT-5 series - large output
    if normalized.startswith("gpt-5"):
        return 64000  # GPT-5: ~64K output

    # GPT-4o series
    if normalized.startswith("gpt-4o"):
        if "mini" in normalized:
            return 16000  # GPT-4o-mini: 16K output
        return 16000  # GPT-4o: 16K output

    # o-series reasoning models
    if len(normalized) > 1 and normalized[0] == "o" and normalized[1].isdigit():
        return 100000  # o1/o3 series: 100K output

    # Claude models
    if "claude" in normalized or "sonnet" in normalized or "opus" in normalized:
        return 8192  # Claude: 8K output (most variants)

    # MiniMax models
    if "minimax" in normalized:
        return 32768  # MiniMax M2.5: 32K max output (204K context total)

    # GLM models
    if "glm" in normalized:
        return 131072  # GLM 5.2: 131K max output (1M context total)

    # Gemini models
    if "gemini" in normalized:
        return 8192  # Gemini: 8K output

    # Default for unknown models
    return default


# Minimal health check prompt to test API connectivity
HEALTH_CHECK_PROMPT = "Reply with exactly: OK"


def _invoke_llm(llm: Any, prompt: str, **kwargs: Any) -> Any:
    """Invoke either an ``.invoke()`` model wrapper or a plain callable.

    Context extraction deliberately uses ``callable(prompt) -> str`` while
    other pipeline stages use LangChain-style objects.  Keeping that adapter
    distinction here prevents callers from having to wrap the same model each
    time they use the shared retry logic.
    """
    invoke = getattr(llm, "invoke", None)
    if callable(invoke):
        return invoke(prompt, **kwargs)
    if callable(llm):
        return llm(prompt)
    raise TypeError("LLM must be callable or expose an invoke() method")


# JSON parsing instructions for repair prompts
STRICT_JSON_RULES = """
🚨 CRITICAL JSON FORMAT 🚨
You MUST follow these rules EXACTLY:

1. Return ONLY raw JSON - nothing else
2. FIRST character MUST be { or [
3. LAST character MUST be } or ]
4. NO markdown code fences (```json or ```)
5. NO backticks ` of any kind
6. NO explanations before or after the JSON
7. NO comments inside the JSON
8. Must be parseable by json.loads() directly

❌ WRONG: ```json {...}``` or "Here is the JSON: {...}"
✅ CORRECT: {...}
""".strip()


def _clean_json_control_characters(content: str) -> str:
    """
    Escape unescaped control characters inside JSON strings.

    Fixes "Invalid control character" errors by escaping newlines, tabs, etc.
    that appear inside JSON string values.
    """
    import re

    # First, fix trailing commas before closing brackets
    # This handles ", }" and ", ]" patterns
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Now escape control characters inside strings
    result = []
    in_string = False
    escape_next = False

    for char in content:
        if escape_next:
            result.append(char)
            escape_next = False
        elif char == '\\':
            result.append(char)
            escape_next = True
        elif char == '"' and not escape_next:
            result.append(char)
            in_string = not in_string
        elif in_string and char in '\n\r\t\b\f':
            # Escape control characters inside strings
            escape_map = {
                '\n': '\\n',
                '\r': '\\r',
                '\t': '\\t',
                '\b': '\\b',
                '\f': '\\f'
            }
            result.append(escape_map.get(char, char))
        else:
            result.append(char)

    return ''.join(result)


def _load_json_flexible(content: str) -> Any:
    """
    Flexibly parse JSON from LLM output.

    Tries multiple parsing strategies:
    1. Standard json.loads
    2. Clean control characters and retry
    3. Extract from markdown code blocks
    4. demjson3 for lenient parsing

    Returns:
        Parsed JSON object/array, or [] if all attempts fail
    """
    if not content or not content.strip():
        return []

    content = content.strip()

    # Strip markdown code fences and explanatory text (common LLM mistakes)
    # Handle multiple formats:
    # - ```json {...} ```
    # - ``` {...} ```
    # - "Here is the JSON: {...}"
    # - {...} followed by explanatory text

    # Remove leading text before JSON
    # Find first { or [ that starts the actual JSON
    json_start = -1
    for i, char in enumerate(content):
        if char in ('{', '['):
            json_start = i
            break

    if json_start > 0:
        # There's text before the JSON, remove it
        content = content[json_start:]

    # Strip markdown fences if present
    if content.startswith("```"):
        lines = content.split('\n')
        # Remove first line (```json or ```)
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = '\n'.join(lines).strip()

    # Remove trailing text after JSON
    # Find the last valid closing bracket
    json_end = -1
    brace_count = 0
    bracket_count = 0
    for i, char in enumerate(content):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and bracket_count == 0:
                json_end = i
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if brace_count == 0 and bracket_count == 0:
                json_end = i

    if json_end > 0 and json_end < len(content) - 1:
        # There's text after the JSON, remove it
        content = content[:json_end + 1]

    content = content.strip()

    # Fix common LLM mistakes: Missing opening brace
    # If content starts with "field_name": [...] wrap in {...}
    if content.startswith('"') and '":' in content[:100]:
        try:
            # Find the last valid closing bracket (] or })
            # This handles cases where there's extra text after the JSON
            last_bracket_idx = max(
                content.rfind(']'),
                content.rfind('}')
            )

            if last_bracket_idx > 0:
                # Trim content to the last valid bracket
                trimmed = content[:last_bracket_idx + 1].strip()

                # Wrap with opening and closing braces
                wrapped = '{' + trimmed + '}'
                return json.loads(wrapped)
        except (json.JSONDecodeError, ValueError):
            pass  # Continue with normal parsing

    last_json_err = None
    last_demjson3_err = None

    # Try standard JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        last_json_err = e
        # If error is "Extra data", try to extract just the JSON part
        if "Extra data" in str(e):
            try:
                # Find the position where valid JSON ends
                # JSONDecodeError.pos gives us where the error occurred
                if hasattr(e, 'pos'):
                    valid_json = content[:e.pos].strip()
                    return json.loads(valid_json)
            except Exception:
                pass  # Continue to other strategies
    except Exception as e:
        last_json_err = e

    # Try cleaning control characters and parse again
    try:
        cleaned = _clean_json_control_characters(content)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Handle "Extra data" error after cleaning
        if "Extra data" in str(e) and hasattr(e, 'pos'):
            try:
                valid_json = cleaned[:e.pos].strip()
                return json.loads(valid_json)
            except Exception:
                pass
    except Exception:
        pass  # Continue to next strategy

    # Try closing truncated JSON structures (improved for nested structures)
    try:
        cleaned = _clean_json_control_characters(content)

        # Remove any incomplete trailing strings/values
        # Find the last complete comma, closing brace, or closing bracket
        last_valid_pos = max(
            cleaned.rfind(','),
            cleaned.rfind('}'),
            cleaned.rfind(']'),
            cleaned.rfind('"')
        )

        # If we found a potentially truncated part, try removing it
        if last_valid_pos > 0 and last_valid_pos < len(cleaned) - 1:
            # Check if there's incomplete text after the last valid delimiter
            trailing = cleaned[last_valid_pos + 1:].strip()
            if trailing and not trailing.startswith((']', '}')):
                # Likely truncated - remove incomplete trailing part
                cleaned = cleaned[:last_valid_pos + 1].rstrip(',').strip()

        # Count open/close brackets and braces with proper nesting
        open_braces = cleaned.count('{') - cleaned.count('}')
        open_brackets = cleaned.count('[') - cleaned.count(']')

        # Close any unclosed structures in proper order (innermost first)
        if open_braces > 0 or open_brackets > 0:
            closed = cleaned

            # Smart closing: track what needs to be closed based on last seen opener
            stack = []
            in_string = False
            escape = False

            for char in cleaned:
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}' and stack and stack[-1] == '}':
                    stack.pop()
                elif char == ']' and stack and stack[-1] == ']':
                    stack.pop()

            # Close in reverse order of what was opened
            while stack:
                closed += stack.pop()

            return json.loads(closed)
    except Exception:
        pass  # Continue to next strategy

    # Try extracting first complete JSON object/array (handles text before/after)
    try:
        # Find first { or [
        for start_char in ['{', '[']:
            start_idx = content.find(start_char)
            if start_idx == -1:
                continue

            # Track brace/bracket balance to find matching close
            balance = 0
            end_char = '}' if start_char == '{' else ']'

            for i in range(start_idx, len(content)):
                if content[i] == start_char:
                    balance += 1
                elif content[i] == end_char:
                    balance -= 1
                    if balance == 0:
                        # Found complete JSON object/array
                        json_str = content[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break  # Try other strategies
            break
    except Exception:
        pass  # Continue to next strategy

    # Try extracting from markdown code blocks
    try:

        # More flexible patterns to handle various markdown formats
        patterns = [
            r"```json\s*(.*?)\s*```",         # ```json ... ```
            r"```\s*(.*?)\s*```",              # ``` ... ```
            r"^`{1,3}json\s*(.*?)`{1,3}$",    # Handle 1-3 backticks
            r"^\s*\[.*\]\s*$",                 # Raw array
            r"^\s*\{.*\}\s*$",                 # Raw object
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
            if matches:
                candidate = matches[0].strip()
                # Skip if candidate is empty or only whitespace
                if not candidate:
                    continue
                try:
                    objects = json.loads(candidate)
                    if isinstance(objects, list) and len(objects) > 1:
                        return objects
                    if len(objects) == 1:
                        return objects[0]
                    return objects
                except json.JSONDecodeError:
                    continue  # Try next pattern
    except Exception:
        pass

    # Try demjson3 for lenient parsing
    try:
        if demjson3 is not None:
            result = demjson3.decode(content)
            if result is not None:
                return result
    except Exception as exc:
        last_demjson3_err = exc

    # All parsing failed
    preview = content[:500] if len(content) > 500 else content
    parse_error = ValueError(
        f"JSON parse failed: json={last_json_err}; demjson3={last_demjson3_err}\n"
        f"Content preview (first 500 chars):\n{preview}"
    )
    LOGGER.warning("%s", parse_error)
    return []


def _check_api_health(llm: Any, verbose: bool = False) -> bool:
    """
    Quick health check to see if API is responsive.

    Sends a minimal prompt to test connectivity.
    Returns True if API responds, False otherwise.

    Args:
        llm: LLM instance
        verbose: Print status messages

    Returns:
        True if API is healthy, False otherwise
    """
    try:
        if verbose:
            print("          -> Testing API connectivity with health check...")

        # Use a minimal prompt to check connectivity (very fast)
        try:
            model_name = getattr(llm, 'model_name', '') or getattr(llm, 'model', '')
            if _requires_max_completion_tokens(model_name):
                response = _invoke_llm(
                    llm, HEALTH_CHECK_PROMPT, max_completion_tokens=8
                )
            else:
                response = _invoke_llm(llm, HEALTH_CHECK_PROMPT, max_tokens=100)
        except TypeError:
            response = _invoke_llm(llm, HEALTH_CHECK_PROMPT)
        content = str(getattr(response, "content", response) or "").strip()

        # Any response means API is back
        if content:
            if verbose:
                print("          -> OK API is responsive!")
            return True
        else:
            if verbose:
                print("          -> FAIL API returned empty response")
            return False

    except Exception as e:
        if verbose:
            print(f"          -> FAIL API still unreachable: {type(e).__name__}")
        return False


def invoke_llm_with_retry(
    llm: Any,
    prompt: str,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    response_format: dict[str, Any] | None = None,
    length_signal: str = "SPLIT_REQUIRED",
    parse_json: bool = True,
    json_repair_prompt_template: str | None = None,
    max_rate_limit_retries: int = 3,
    rate_limit_backoff_base: int = 60,
    max_timeout_retries: int = 2,
    timeout_multiplier: float = 2.0,
    max_connection_retries: int = 5,
    connection_retry_delay: int = 10,
    verbose: bool = True,
    _retry_count: int = 0,
    _rate_limit_retry: int = 0,
    _connection_retry: int = 0,
) -> Any:
    """
    Invoke LLM with comprehensive retry logic and error handling.

    Args:
        llm: Language model instance with .invoke() method
        prompt: The prompt to send to the LLM
        max_tokens: Maximum tokens for response (optional)
        reasoning_effort: Optional provider-normalized reasoning level
        response_format: Optional structured-output format
        length_signal: Value returned when generation exhausts its output budget
        parse_json: Whether to parse response as JSON (default: True)
        json_repair_prompt_template: Custom repair prompt template with {content} placeholder
        max_rate_limit_retries: Maximum retries for rate limit errors (default: 3)
        rate_limit_backoff_base: Base wait time in seconds for rate limits (default: 60)
        max_timeout_retries: Maximum retries for timeout errors (default: 2)
        timeout_multiplier: Timeout multiplier for retries (default: 2.0)
        max_connection_retries: Maximum retries for connection errors (default: 5)
        connection_retry_delay: Base wait time in seconds for connection retries with exponential backoff (default: 10)
                                First retry: 10s, second: 20s, third: 40s, fourth: 80s, fifth: 160s
        verbose: Print status messages (default: True)
        _retry_count: Internal - current timeout retry count
        _rate_limit_retry: Internal - current rate limit retry count
        _connection_retry: Internal - current connection retry count

    Returns:
        - If parse_json=True: Parsed JSON object/array or [] on failure
        - If parse_json=False: Raw string content or "" on failure
        - Special return value "SPLIT_REQUIRED" signals caller should split input

    Retry Strategy:
        - Rate limits: Exponential backoff (60s, 120s, 180s, ...)
        - Timeouts: Retry once with increased timeout, then return SPLIT_REQUIRED
        - Connection errors: Brief wait (5s) and retry up to 2 times
        - Empty content: Check finish_reason, return SPLIT_REQUIRED if length limit
        - Malformed JSON: One repair attempt with small prompt
    """
    prompt_size_kb = len(prompt) / 1024

    if verbose and prompt_size_kb > 80:
        print(
            f"        WARNING  Large prompt: {prompt_size_kb:.1f}KB - may cause API errors"
        )

    # ============================================================
    # STEP 1: Invoke LLM with error handling
    # ============================================================
    try:
        try:
            invoke_kwargs: dict[str, Any] = {}
            if max_tokens:
                # GPT-5/GPT-4o require max_completion_tokens, not max_tokens
                model_name = getattr(llm, 'model_name', '') or getattr(llm, 'model', '')
                if _requires_max_completion_tokens(model_name):
                    invoke_kwargs["max_completion_tokens"] = max_tokens
                else:
                    invoke_kwargs["max_tokens"] = max_tokens
            if reasoning_effort:
                invoke_kwargs["reasoning_effort"] = reasoning_effort
            if response_format:
                invoke_kwargs["response_format"] = response_format
            response = _invoke_llm(llm, prompt, **invoke_kwargs)
        except TypeError:
            # Compatibility fallback for wrappers whose ``invoke`` method only
            # accepts the prompt. Retrying with the same token keyword would
            # reproduce the TypeError and get converted into empty content.
            response = _invoke_llm(llm, prompt)

        # Some wrappers return an empty Result with an embedded error instead
        # of raising. Promote that error so transient transport failures enter
        # the retry/backoff path instead of looking like empty model output.
        response_error = str(getattr(response, "error", "") or "").strip()
        if response_error:
            raise RuntimeError(response_error)

        content = str(getattr(response, "content", response) or "").strip()

    except Exception as exc:
        error_msg = str(exc)
        error_type = type(exc).__name__

        # --------------------------------------------------------
        # ERROR TYPE 1: Rate Limit
        # --------------------------------------------------------
        if "rate limit" in error_msg.lower():
            if _rate_limit_retry < max_rate_limit_retries:
                wait_time = rate_limit_backoff_base * (_rate_limit_retry + 1)

                if verbose:
                    print(
                        f"        FAIL Rate limit (attempt {_rate_limit_retry + 1}/{max_rate_limit_retries})"
                    )
                    print(f"          -> Waiting {wait_time} seconds before retry...")

                LOGGER.warning(
                    f"Rate limit hit, waiting {wait_time}s before retry "
                    f"{_rate_limit_retry + 1}/{max_rate_limit_retries}: {exc}"
                )

                time.sleep(wait_time)

                if verbose:
                    print(f"          -> Retrying after {wait_time}s wait...")

                return invoke_llm_with_retry(
                    llm=llm,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                    length_signal=length_signal,
                    parse_json=parse_json,
                    json_repair_prompt_template=json_repair_prompt_template,
                    max_rate_limit_retries=max_rate_limit_retries,
                    rate_limit_backoff_base=rate_limit_backoff_base,
                    max_timeout_retries=max_timeout_retries,
                    timeout_multiplier=timeout_multiplier,
                    max_connection_retries=max_connection_retries,
                    connection_retry_delay=connection_retry_delay,
                    verbose=verbose,
                    _retry_count=_retry_count,
                    _rate_limit_retry=_rate_limit_retry + 1,
                    _connection_retry=_connection_retry,
                )
            else:
                if verbose:
                    print(
                        f"        FAIL Rate limit persists after {max_rate_limit_retries} retries"
                    )
                    print("          -> Giving up on this chunk")

                LOGGER.error(
                    f"Rate limit persists after {max_rate_limit_retries} retries, giving up"
                )
                return [] if parse_json else ""

        # --------------------------------------------------------
        # ERROR TYPE 2: Timeout
        # --------------------------------------------------------
        elif "timeout" in error_msg.lower() or "Timeout" in error_type:
            if _retry_count < max_timeout_retries:
                if verbose:
                    print(
                        f"        FAIL API Timeout (attempt {_retry_count + 1}): {error_type}"
                    )
                    print(f"          Prompt size: {prompt_size_kb:.1f}KB")
                    print(
                        f"          -> Retrying with increased timeout ({timeout_multiplier}x)..."
                    )

                LOGGER.warning(
                    f"LLM API timeout on attempt {_retry_count + 1}, retrying with increased timeout: {exc}"
                )

                # Temporarily increase timeout
                original_timeout = int(os.getenv("LITELLM_TIMEOUT", "900"))
                new_timeout = int(original_timeout * timeout_multiplier)
                os.environ["LITELLM_TIMEOUT"] = str(new_timeout)

                try:
                    time.sleep(2)  # Brief pause before retry
                    result = invoke_llm_with_retry(
                        llm=llm,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        response_format=response_format,
                        length_signal=length_signal,
                        parse_json=parse_json,
                        json_repair_prompt_template=json_repair_prompt_template,
                        max_rate_limit_retries=max_rate_limit_retries,
                        rate_limit_backoff_base=rate_limit_backoff_base,
                        max_timeout_retries=max_timeout_retries,
                        timeout_multiplier=timeout_multiplier,
                        max_connection_retries=max_connection_retries,
                        connection_retry_delay=connection_retry_delay,
                        verbose=verbose,
                        _retry_count=_retry_count + 1,
                        _rate_limit_retry=_rate_limit_retry,
                        _connection_retry=_connection_retry,
                    )
                    return result
                finally:
                    # Restore original timeout
                    os.environ["LITELLM_TIMEOUT"] = str(original_timeout)
            else:
                if verbose:
                    print(
                        f"        FAIL API Timeout (attempt {_retry_count + 1}): {error_type}"
                    )
                    print(f"          Prompt size: {prompt_size_kb:.1f}KB")
                    print(
                        "          -> Timeout persists after retry, splitting chunk..."
                    )

                LOGGER.warning(f"LLM API timeout after retry, triggering split: {exc}")
                return "SPLIT_REQUIRED"

        # --------------------------------------------------------
        # ERROR TYPE 3: Connection/Network Errors
        # --------------------------------------------------------
        elif any(
            keyword in error_msg.lower()
            for keyword in [
                "peer closed connection",
                "connection",
                "network",
                "incomplete chunked",
                "openrouterexception",
                "apierror",
            ]
        ):
            if _connection_retry < max_connection_retries:
                # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                max_wait_time = connection_retry_delay * (2**_connection_retry)

                if verbose:
                    print(
                        f"        FAIL API Connection Error (attempt {_connection_retry + 1}/{max_connection_retries})"
                    )
                    print(f"          Error: {error_type}: {str(exc)[:200]}")
                    print(
                        f"          -> Adaptive wait: checking connection every 5s (max {max_wait_time}s)..."
                    )

                LOGGER.warning(
                    f"Connection error, adaptive wait up to {max_wait_time}s before retry "
                    f"{_connection_retry + 1}/{max_connection_retries}: {exc}"
                )

                # Adaptive waiting: check health every 5 seconds
                elapsed = 0
                check_interval = 5
                while elapsed < max_wait_time:
                    time.sleep(check_interval)
                    elapsed += check_interval

                    # Check if API is back
                    if _check_api_health(llm, verbose=verbose):
                        if verbose:
                            print(
                                f"          -> Connection restored after {elapsed}s! Retrying now..."
                            )
                        break
                    else:
                        if verbose and elapsed < max_wait_time:
                            print(
                                f"          -> Still waiting... ({elapsed}s/{max_wait_time}s)"
                            )

                if elapsed >= max_wait_time and verbose:
                    print(f"          -> Retrying after full {max_wait_time}s wait...")

                return invoke_llm_with_retry(
                    llm=llm,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    response_format=response_format,
                    length_signal=length_signal,
                    parse_json=parse_json,
                    json_repair_prompt_template=json_repair_prompt_template,
                    max_rate_limit_retries=max_rate_limit_retries,
                    rate_limit_backoff_base=rate_limit_backoff_base,
                    max_timeout_retries=max_timeout_retries,
                    timeout_multiplier=timeout_multiplier,
                    max_connection_retries=max_connection_retries,
                    connection_retry_delay=connection_retry_delay,
                    verbose=verbose,
                    _retry_count=_retry_count,
                    _rate_limit_retry=_rate_limit_retry,
                    _connection_retry=_connection_retry + 1,
                )
            else:
                if verbose:
                    print(
                        f"        FAIL Connection error persists after {max_connection_retries} retries"
                    )
                    print(f"          Error: {error_type}: {str(exc)[:200]}")
                    print("          -> Giving up on this chunk")

                LOGGER.error(
                    f"Connection error persists after {max_connection_retries} retries: {exc}"
                )
                raise LLMTransientConnectionError(
                    "LLM connection remained unavailable after "
                    f"{max_connection_retries} retries: {exc}"
                ) from exc

        # --------------------------------------------------------
        # ERROR TYPE 4: Malformed/Truncated JSON
        # --------------------------------------------------------
        elif (
            "Unable to get json response" in error_msg or "Expecting value" in error_msg
        ):
            if verbose:
                print("        FAIL API returned malformed/truncated JSON")
                print(f"          Prompt size: {prompt_size_kb:.1f}KB")
                print("          -> Chunk too large, reduce max_files_per_chunk")

            LOGGER.error(f"Malformed JSON response: {exc}")
            return [] if parse_json else ""

        # --------------------------------------------------------
        # ERROR TYPE 5: Other API Errors
        # --------------------------------------------------------
        else:
            if verbose:
                print(f"        FAIL API Error: {error_type}: {str(exc)[:200]}")

            LOGGER.error(f"LLM API call failed: {error_type}: {exc}")
            return [] if parse_json else ""

    # ============================================================
    # STEP 2: Handle empty content
    # ============================================================
    if not content:
        error_details = ""
        finish_reason = None

        # Extract error details from response
        if hasattr(response, "error"):
            error_details = f"Error: {response.error}"
        elif hasattr(response, "raw_response") and response.raw_response:
            raw = response.raw_response
            if hasattr(raw, "error"):
                error_details = f"Error: {raw.error}"
            if hasattr(raw.choices[0], "finish_reason"):
                finish_reason = raw.choices[0].finish_reason
                error_details += f" | Finish reason: {finish_reason}"

        if verbose:
            print("        FAIL LLM returned empty content")
            print(
                f"          Prompt size: {len(prompt)} chars ({prompt_size_kb:.1f}KB)"
            )
            if error_details:
                print(f"          {error_details}")

        # Special handling for length limit
        if finish_reason == "length":
            if verbose:
                print(
                    f"          -> Returning {length_signal!r}; output budget was exhausted"
                )
            LOGGER.warning("Length limit hit; generation exhausted output budget")
            return length_signal

        LOGGER.warning(
            f"LLM empty content. Prompt size: {len(prompt)} chars. {error_details}"
        )
        return [] if parse_json else ""

    # ============================================================
    # STEP 3: Return raw content if JSON parsing not requested
    # ============================================================
    if not parse_json:
        return content

    # ============================================================
    # STEP 4: Parse JSON
    # ============================================================
    parsed = _load_json_flexible(content)

    # Successfully parsed (including valid empty arrays/objects)
    if parsed is not None:
        # Check if content was literally "[]" or "{}" - valid empty responses
        content_stripped = content.strip()
        if content_stripped in ('[]', '{}') or (content_stripped.startswith('[') and content_stripped.endswith(']')) or (content_stripped.startswith('{') and content_stripped.endswith('}')):
            # Valid JSON structure (even if empty)
            return parsed
        # Non-empty parsed result
        if parsed not in ([], {}):
            return parsed

    # ============================================================
    # STEP 5: Attempt JSON repair
    # ============================================================
    LOGGER.warning(
        f"Initial JSON parse failed, attempting repair. Content preview: {content[:200]}"
    )

    # Use custom repair prompt template if provided
    if json_repair_prompt_template:
        repair_prompt = json_repair_prompt_template.format(content=content[:24000])
    else:
        repair_prompt = f"""{STRICT_JSON_RULES}

Repair the following model output into valid JSON only.
Preserve all recoverable keys and values.
If the output is truncated, close the current JSON structure conservatively and omit incomplete trailing items.

--- MODEL OUTPUT TO REPAIR ---
{content[:24000]}
"""

    try:
        repair_max_tokens = min(max_tokens or 8_000, 8_000)
        try:
            repaired_response = _invoke_llm(
                llm,
                repair_prompt,
                max_tokens=repair_max_tokens,
                reasoning_effort="low" if reasoning_effort else None,
                response_format=response_format,
            )
        except TypeError:
            repaired_response = _invoke_llm(llm, repair_prompt)
        repaired_content = str(
            getattr(repaired_response, "content", repaired_response) or ""
        ).strip()
        repaired = _load_json_flexible(repaired_content)

        if repaired not in (None, [], {}):
            LOGGER.info("Recovered malformed JSON with repair prompt")
            return repaired
    except Exception as exc:
        LOGGER.warning("JSON repair prompt failed: %s", exc)

    LOGGER.warning(
        f"All JSON parsing attempts failed. Returning empty. Original content length: {len(content)}"
    )
    # Ensure we always return a valid type (never None)
    return parsed if parsed is not None else []


# Convenience alias for backward compatibility
invoke_json = invoke_llm_with_retry
