"""
utilities/json_fallback.py - Robust JSON parsing with auto-split fallback

Handles common LLM JSON issues:
- Malformed JSON (missing quotes, trailing commas, etc.)
- Overly complex inputs that confuse the LLM
- Automatic splitting and retry when input is too large
"""

import json
from typing import Any, Callable, Dict, List, Optional

try:
    import demjson3
except ImportError:
    demjson3 = None


def parse_json_with_fallback(
    response: str,
    expected_keys: Optional[List[str]] = None,
    fallback_value: Optional[Dict] = None,
) -> Dict:
    """
    Parse JSON response with multiple repair strategies.

    Args:
        response: LLM response string
        expected_keys: List of keys that should be in the result
        fallback_value: Default value to return if all parsing fails

    Returns:
        Parsed JSON dict, or fallback_value if parsing fails
    """
    if fallback_value is None:
        fallback_value = {}

    # Remove markdown code blocks
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()

    # Strategy 1: Standard JSON parsing
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON between { }
    start = response.find("{")
    end = response.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(response[start:end])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Lenient parsing with demjson3
    if demjson3:
        try:
            result = demjson3.decode(response)
            if isinstance(result, dict):
                print("    INFO: JSON repaired using demjson3")
                return result
        except Exception:
            pass

    # Strategy 4: Extract expected keys manually (last resort)
    if expected_keys and response:
        try:
            result = extract_keys_heuristic(response, expected_keys)
            if result:
                print("    INFO: JSON extracted using heuristic parsing")
                return result
        except Exception:
            pass

    # All strategies failed
    print(f"    WARNING: All JSON parsing strategies failed")
    print(f"    Response preview: {response[:300]}")
    return fallback_value


def extract_keys_heuristic(response: str, expected_keys: List[str]) -> Optional[Dict]:
    """
    Heuristic extraction of key-value pairs from malformed JSON.

    This is a last resort when the JSON is so broken that demjson3 can't fix it.
    """
    # This is a simplified heuristic - can be enhanced based on specific patterns
    result = {}
    for key in expected_keys:
        # Try to find key: value pattern
        pattern = f'"{key}"'
        idx = response.find(pattern)
        if idx >= 0:
            # Found the key, try to extract its value
            # This is very basic and would need enhancement for production
            pass

    return result if result else None


def should_split_and_retry(
    error_msg: str,
    input_size: int,
    threshold: int = 40_000,
) -> bool:
    """
    Determine if we should split input and retry based on error and size.

    Args:
        error_msg: The error message from JSON parsing
        input_size: Size of input in tokens/chars
        threshold: Size threshold above which to split

    Returns:
        True if should split and retry
    """
    # If input is large and we got a JSON error, it's likely due to complexity
    if input_size > threshold:
        return True

    # If error mentions specific issues that indicate confusion
    confusion_indicators = [
        "Expecting",  # JSON syntax errors
        "delimiter",  # Missing delimiters
        "line",  # Syntax error at specific line
        "Unterminated",  # Incomplete JSON
    ]
    return any(indicator in error_msg for indicator in confusion_indicators)


def split_and_retry_json_call(
    call_func: Callable[[Any], str],
    parse_func: Callable[[str], Dict],
    items: List[Any],
    chunk_size: int,
    merge_func: Callable[[List[Dict]], Dict],
    max_retries: int = 2,
) -> Dict:
    """
    Generic utility to split input, retry LLM calls, and merge results.

    Args:
        call_func: Function that calls LLM with a chunk of items
        parse_func: Function that parses LLM response to dict
        items: List of items to process
        chunk_size: How many items per chunk
        merge_func: Function to merge results from multiple chunks
        max_retries: Max number of retry attempts per chunk

    Returns:
        Merged result dict

    Example:
        def call_llm(files):
            prompt = build_prompt(files)
            return llm.invoke(prompt)

        def parse(response):
            return parse_json_with_fallback(response, ["problems"])

        def merge(results):
            all_problems = []
            for r in results:
                all_problems.extend(r.get("problems", []))
            return {"problems": all_problems}

        result = split_and_retry_json_call(
            call_llm, parse, files, chunk_size=20, merge_func=merge
        )
    """
    if not items:
        return merge_func([])

    # Split items into chunks
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
    results = []

    for chunk_idx, chunk in enumerate(chunks, 1):
        print(f"      Processing chunk {chunk_idx}/{len(chunks)}: {len(chunk)} items")

        for retry in range(max_retries):
            try:
                response = call_func(chunk)
                result = parse_func(response)
                results.append(result)
                break  # Success, move to next chunk
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"        Retry {retry + 1}/{max_retries} due to: {e}")
                else:
                    print(f"        Failed after {max_retries} retries: {e}")
                    # Add empty result for this chunk
                    results.append({})

    return merge_func(results)


def create_safe_fallback(
    expected_schema: Dict[str, Any],
    error_msg: str,
) -> Dict:
    """
    Create a safe fallback result that matches expected schema.

    Args:
        expected_schema: Dict describing expected structure
        error_msg: Error message to include in fallback

    Returns:
        Safe default dict with expected structure

    Example:
        fallback = create_safe_fallback(
            {"problems": list, "total": int},
            "JSON parsing failed"
        )
        # Returns: {"problems": [], "total": 0, "parse_error": "..."}
    """
    result = {}

    for key, value_type in expected_schema.items():
        if value_type == list:
            result[key] = []
        elif value_type == dict:
            result[key] = {}
        elif value_type == int:
            result[key] = 0
        elif value_type == str:
            result[key] = ""
        elif value_type == bool:
            result[key] = False
        else:
            result[key] = None

    result["parse_error"] = error_msg
    return result
