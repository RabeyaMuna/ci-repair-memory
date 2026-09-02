from __future__ import annotations

import json
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

from utilities.model_token_config import (
    calculate_adaptive_output_limit,
    get_input_chunk_tokens,
)
from utilities.llm_invoker import invoke_llm_with_retry
from utilities.text_normalizer import normalize_log_text

# ── Optional third-party dependencies ─────────────────────────────────────────
try:
    import demjson3  # type: ignore

    _DEMJSON3_AVAILABLE = True
except ImportError:
    demjson3 = None  # type: ignore
    _DEMJSON3_AVAILABLE = False

try:
    import tiktoken  # type: ignore

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:
    BM25Okapi = None  # type: ignore

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_core.messages import HumanMessage  # type: ignore
except ImportError:

    class HumanMessage:  # type: ignore
        def __init__(self, content: str):
            self.content = content


# ── JSON Repair Utilities ─────────────────────────────────────────────────────
def _repair_truncated_json(content: str) -> str:
    """
    Attempt to repair truncated JSON by closing open structures.

    Handles:
    - Incomplete strings
    - Unclosed arrays
    - Unclosed objects
    - Mixed nesting
    - Mid-sentence truncations in array values
    - Incomplete key-value pairs
    """
    if not content.strip():
        return content

    content = content.strip()

    # Handle unclosed strings (odd number of quotes)
    if content.count('"') % 2 != 0:
        # Find the last unclosed quote
        last_quote_idx = content.rfind('"')
        if last_quote_idx >= 0:
            # Count quotes before to determine if this is opening or closing
            quotes_before = content[:last_quote_idx].count('"')
            if quotes_before % 2 == 0:
                # This is an opening quote - incomplete string
                before_quote = content[:last_quote_idx].rstrip()

                # Check if this is a key or value in a key-value pair
                # Look for : before the quote
                check_colon = before_quote.rstrip()
                if check_colon and check_colon[-1] == ':':
                    # This is a value - remove the entire key-value pair
                    # Find the key by looking backwards
                    # Pattern: "key": "truncated
                    colon_idx = before_quote.rfind(':')
                    if colon_idx > 0:
                        # Find the key before the colon
                        before_colon = before_quote[:colon_idx].rstrip()
                        # Remove the key (last quoted string before colon)
                        last_key_quote = before_colon.rfind('"')
                        if last_key_quote > 0:
                            # Find the opening quote of the key
                            key_start = before_colon[:last_key_quote].rfind('"')
                            if key_start >= 0:
                                # Remove from key start
                                content = before_colon[:key_start].rstrip()
                                # Remove trailing comma if present
                                if content and content[-1] == ',':
                                    content = content[:-1].rstrip()
                else:
                    # This might be in an array or the key itself
                    # Just remove the incomplete string
                    if before_quote and before_quote[-1] in ',:':
                        before_quote = before_quote[:-1].rstrip()
                    content = before_quote

    # Count open/close brackets and braces AFTER fixing strings
    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')

    # Remove trailing comma
    content = content.rstrip(' ,\n\t')

    # Add closing brackets/braces
    content += ']' * open_brackets
    content += '}' * open_braces

    return content


def _clean_malformed_json(content: str) -> str:
    """
    Clean common LLM JSON errors:
    - Markdown fences
    - Trailing commas
    - Missing commas
    - Extra text before/after JSON
    - Truncated JSON
    """
    # STEP 1: Repair truncated JSON first
    content = _repair_truncated_json(content)

    # STEP 2: Remove markdown fences
    content = re.sub(r"```(?:json)?\s*\n?(.*?)\n?```", r"\1", content, flags=re.DOTALL)

    # Fix trailing commas before } or ]
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    # Fix missing commas between string values
    content = re.sub(r'"\s+"', '", "', content)

    # Fix double commas
    content = re.sub(r",\s*,", ",", content)

    # Fix missing commas between } and {
    content = re.sub(r"}\s*{", "}, {", content)
    content = re.sub(r"}\s*\[", "}, [", content)
    content = re.sub(r"]\s*{", "], {", content)

    # Extract JSON from surrounding text (find first { or [ to last } or ])
    content = content.strip()
    start_brace = content.find("{")
    start_bracket = content.find("[")

    # Determine which comes first
    if start_brace == -1 and start_bracket == -1:
        return content
    elif start_brace == -1:
        start = start_bracket
        end_char = "]"
    elif start_bracket == -1:
        start = start_brace
        end_char = "}"
    else:
        start = min(start_brace, start_bracket)
        end_char = "}" if start == start_brace else "]"

    # Find matching closing brace/bracket
    end = content.rfind(end_char)
    if end > start:
        content = content[start : end + 1]

    return content.strip()


# ── Fallbacks for missing 'utilities.*' modules ────────────────────────────────

# utilities.constant.ERROR_KEYWORDS
ERROR_KEYWORDS: list[str] = [
    "error",
    "Error",
    "ERROR",
    "errors",
    "Errors",
    "ERRORS",
    "failed",
    "Failed",
    "FAILED",
    "failure",
    "Failure",
    "FAILURE",
    "exception",
    "Exception",
    "EXCEPTION",
    "traceback",
    "Traceback",
    "fatal",
    "Fatal",
    "FATAL",
    "critical",
    "Critical",
    "CRITICAL",
    "AssertionError",
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "TypeError",
    "ValueError",
    "RuntimeError",
    "no module named",
    "not found",
    "permission denied",
    "syntax error",
    "timeout",
    "timed out",
    "connection refused",
]


def load_config() -> dict[str, Any]:
    """Fallback for utilities.load_config.load_config()."""
    out_folder = os.environ.get(
        "MEMCI_OUT_FOLDER",
        os.path.join(tempfile.gettempdir(), "memci_out"),
    )
    return {
        "exception_dir": os.path.join(out_folder, "exceptions"),
        "out_folder": out_folder,
    }


def chunk_log_by_tokens(
    text: str,
    max_tokens: int | None = None,
    overlap: int = 200,
    model: str = "",
    token_counter: Callable[[str], int] | None = None,
) -> list[str]:
    """
    Split a log chronologically using measured token counts.

    ``token_counter`` lets callers use the same tokenizer used to decide that
    chunking is necessary.  The optional argument preserves compatibility with
    existing callers.  Without a tokenizer, use a conservative estimate for
    log-like text instead of the unsafe four-characters-per-token assumption.
    """
    if max_tokens is None:
        try:
            max_tokens = get_input_chunk_tokens(model)
        except Exception:
            max_tokens = 70_000  # Fallback
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    def count_tokens(value: str) -> int:
        if not value:
            return 0
        if token_counter is not None:
            return max(1, token_counter(value))
        # CI logs contain punctuation, paths, and identifiers that tokenize
        # more densely than prose. Two chars/token is intentionally cautious.
        return max(1, (len(value) + 1) // 2)

    if count_tokens(text) <= max_tokens:
        return [text]

    def split_oversized_text(value: str) -> list[str]:
        """Split a single oversized line using token-counted binary search."""
        pieces: list[str] = []
        start = 0
        while start < len(value):
            low = start + 1
            high = len(value)
            best = start + 1
            while low <= high:
                middle = (low + high) // 2
                if count_tokens(value[start:middle]) <= max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            pieces.append(value[start:best])
            start = best
        return pieces

    units: list[str] = []
    for line in text.splitlines(keepends=True):
        if count_tokens(line) > max_tokens:
            units.extend(split_oversized_text(line))
        else:
            units.append(line)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("".join(current))

            overlap_units: list[str] = []
            overlap_tokens = 0
            for previous in reversed(current):
                previous_tokens = count_tokens(previous)
                if overlap_tokens + previous_tokens > overlap:
                    break
                overlap_units.append(previous)
                overlap_tokens += previous_tokens
            overlap_units.reverse()

            if overlap_tokens + unit_tokens <= max_tokens:
                current = overlap_units
                current_tokens = overlap_tokens
            else:
                current = []
                current_tokens = 0

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("".join(current))

    return chunks or [text]


def get_chunk_threshold_simple(model_name: str = "") -> int:
    """
    Fallback for utilities.model_token_limits.get_chunk_threshold_simple().

    Now model-aware: returns model-specific threshold.
    """
    try:
        return get_input_chunk_tokens(model_name)
    except Exception:
        return 70_000  # Fallback


class CILogAnalyzerLLM:
    def __init__(
        self,
        repo_path: str,
        ci_log: list[dict[str, Any]],
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: ChatOpenAI,
        model_name: str,
        task_id: str,
    ):
        self.config = load_config()
        self.repo_path = repo_path
        self.ci_log = ci_log
        self.sha_fail = sha_fail
        self.workflow = workflow
        self.workflow_path = workflow_path
        self.task_id = task_id

        self.llm = llm
        self.model_name = model_name
        self._encoder = self._get_encoder()

        self.error_details: list[dict[str, Any]] = []

    def ci_log_analysis(self) -> list[dict[str, Any]]:
        """
        Analyze CI logs using LLM
        """
        print("Running Tool: LLM-based CI Log Analysis")
        results: list[dict[str, Any]] = []
        THRESHOLD = get_chunk_threshold_simple(self.model_name)
        chunk_tracker = []

        for step in self.ci_log:
            step_name = step.get("step_name", "unknown_step")
            log = step.get("log", "")
            print(f"\nProcessing Step: {step_name}")

            try:
                # Handle different log formats
                if isinstance(log, str):
                    log_text = log
                elif isinstance(log, list):
                    # List of strings: join them
                    if log and isinstance(log[0], str):
                        log_text = "\n".join(log)
                    # List of dicts with 'log' key: extract and join
                    elif log and isinstance(log[0], dict):
                        log_text = "\n".join(item.get("log", "") for item in log if isinstance(item, dict))
                    else:
                        log_text = ""
                else:
                    log_text = str(log) if log else ""

                # Normalize text to handle ANY encoding issues (BOM, control chars, etc.)
                log_text = normalize_log_text(log_text)

                if not log_text or len(log_text.strip()) == 0:
                    print(f"  ⚠️  Skipping step '{step_name}': empty log content")
                    continue

                total_tokens = self._estimate_tokens(log_text)

                print(f"Token count for '{step_name}': {total_tokens}")

                if total_tokens > THRESHOLD:
                    raw_chunks = chunk_log_by_tokens(
                        log_text,
                        max_tokens=THRESHOLD,
                        overlap=200,
                        model=self.model_name,
                        token_counter=self._estimate_tokens,
                    )
                    print(
                        f"Chunking activated: {len(raw_chunks)} chunks created for step '{step_name}'"
                    )

                    chunk_tracker.append((step_name, len(raw_chunks)))

                    if len(raw_chunks) > 6:
                        chunks = self._filter_chunks(raw_chunks)
                    else:
                        chunks = raw_chunks
                else:
                    chunks = [log_text]
                    print(f"No chunking needed for '{step_name}'")

                self._save_chunk_tracker(chunk_tracker)

                step_chunks = []

                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i + 1}/{len(chunks)}...")

                    prompt = f"""
You are CI Log Analyzer. Analyze the following CI log chunk and extract structured information while staying strictly faithful to the text.

You may be dealing with ANY CI system and ANY tools.

DO NOT guess. DO NOT invent missing context. DO NOT give advice.

────────────────────────────────
1) SUMMARY (SHORT, HIGH-SIGNAL)

Write a concise 2–5 sentence summary describing:
- The purpose of this CI log chunk
- Whether failures or errors occur
- The nature of those failures (if present)
- Whether execution appears to continue or stop

Rules:
- Omit repetitive and low-signal output
- Mention repeated failures only once
- Stay strictly faithful to the log

────────────────────────────────
2) RELEVANT FILES (ALL FILE PATHS MENTIONED)

Extract ALL file paths that appear anywhere in the chunk.
For each file:
- Normalize to repo-relative paths when possible
- Explain WHY the file appears using evidence from the log
- State whether it is directly related to a failure, a warning, or normal execution

Rules:
- Deduplicate files
- If the same file appears in multiple contexts, merge the reasons
- Do NOT invent relevance

────────────────────────────────
3) FAILURE SIGNALS (EXTRACT IDENTIFIERS FOR PATTERN MATCHING)

For each DISTINCT failure, extract a concise signal that can be used for pattern matching and debugging.

Format: '<tool> [version] <error_code>: <error_msg> at <file>:<line> [key_tokens]'

Include when available:
- Tool/validator name (mypy, pytest, ruff, npm, etc.)
- Tool version if shown in logs
- Error code (F401, arg-type, E501, etc.)
- Key error message (brief)
- File path and line number
- Key tokens/symbols that identify the issue

Examples:
- 'mypy 1.8.0 error [arg-type]: Incompatible type for argument 1 at helpers.py:42 [joinpath]'
- 'pytest 7.4.3: test_login FAILED [AssertionError: expected 200, got 401]'
- 'ruff 0.1.9 F401: Unused import at utils.py:15 [numpy.typing.DTypeLike]'
- 'npm install: ERESOLVE dependency conflict [react@18 vs react@17]'

If the chunk contains NO failures, return an empty array [].

────────────────────────────────
4) RELEVANT FAILURES (NORMALIZED, NATURAL LANGUAGE — CRITICAL)

For each DISTINCT failure observed in this chunk, write ONE concise natural-language description.

Each failure description MUST:
- Describe WHAT failed (trial / test / command / step / job, if identifiable)
- State the EXACT error type and message as shown (e.g., ValueError)
- State WHERE it occurred (file and line if shown; normalize paths)
- Explain HOW the failure is evidenced in the log (warnings, traceback mention, failure message)
- Mention repetition if the same failure occurs multiple times

STRICT RULES:
- Do NOT copy raw stack traces
- Do NOT include long multiline blocks
- Do NOT repeat identical failures multiple times
- Do NOT speculate about root cause beyond what the log states
- Keep each failure description under 500 characters

If the chunk contains NO failures, return an empty array [].

────────────────────────────────
PATH NORMALIZATION

Remove common CI workspace prefixes when possible, such as:
- /home/runner/work/<repo>/<repo>/
- /workspace/
- /__w/<repo>/<repo>/
- /opt/.../<repo>/

If unsure, keep the original path.

────────────────────────────────
OUTPUT FORMAT (STRICT)

Return ONLY valid JSON with EXACTLY this structure.
Do NOT add extra keys.
Do NOT use ``` or ```json.

{{
  "step_name": "{step_name}",
  "summary": "Detailed, faithful narrative summary of everything shown in this chunk.",
  "failure_signals": [
    "Format: '<tool> [version] <error_code>: <error_msg> at <file>:<line> [key_tokens]', any code tokens that are buggy, and any other identifiers that can help pattern matching and debugging.",
    "Example: 'mypy 1.8.0 error [arg-type]: Incompatible type at helpers.py:42 [joinpath]'"
  ],
  "relevant_files": [
    {{
      "file": "repo/relative/path.ext",
      "reason": "Evidence-based explanation of why this file appears in the log and whether it relates to a failure."
    }}
  ],
  "relevant_failures": [
    "Natural-language, evidence-based description of a distinct failure."
  ]
}}

Rules:
- **CRITICAL**: Include ALL files that have DIRECT or INDIRECT relevance to the CI failure
- Include files from ALL error types (test failures, setup errors, infrastructure errors, dependency errors, etc.)
- **MUST extract files from:**
  * ANY traceback or exception (Python, Java, etc.) - extract the file path from stack traces
  * Error messages that mention file paths
  * Files being tested/executed when failure occurred
  * Setup/installation scripts that failed (e.g., install.py, setup.py, build scripts)
  * Configuration files involved in infrastructure failures (e.g., CI config, dependency files)
  * ANY file mentioned in logs where an error/failure occurred, regardless of error category
- **Examples of files to include:**
  * Test failures: test files + source files mentioned in test output
  * Setup/installation errors: installation scripts, setup.py, requirements files
  * Build errors: build scripts, Makefile, build configuration
  * Infrastructure errors: CI workflow files, environment setup scripts
  * Dependency errors: pyproject.toml, package.json, requirements.txt, lock files
- If no files exist in the logs, return: "relevant_files": []
- If no failures exist, return: "relevant_failures": [] AND "failure_signals": []
- Output plain JSON only — no text before or after.

────────────────────────────────
CI LOG CHUNK
{chunk}
"""

                    max_output_tokens = self._adaptive_output_limit(prompt)
                    response = self.llm.invoke(
                        [HumanMessage(content=prompt)],
                        max_tokens=max_output_tokens,
                    ).content
                    content = self.load_json_maybe_fenced(response)
                    if not content or not content.strip():
                        continue

                    try:
                        cleaned_json = json.loads(content)
                    except json.JSONDecodeError:
                        if demjson3 is not None:
                            try:
                                cleaned_json = demjson3.decode(content)
                            except Exception as dec_err:
                                print(
                                    f"[WARN] demjson3 failed for chunk {i + 1}: {dec_err}"
                                )
                                continue
                        else:
                            print(
                                f"[WARN] json.loads failed for chunk {i + 1} and demjson3 not installed; skipping."
                            )
                            continue

                    # Decide whether to skip this chunk
                    no_failures = not cleaned_json.get(
                        "relevant_failures"
                    )  # [] or missing -> True
                    no_files = not cleaned_json.get(
                        "relevant_files"
                    )  # [] or missing -> True
                    empty_summary = not (cleaned_json.get("summary") or "").strip()

                    if no_failures and no_files and empty_summary:
                        print(
                            f"Skipping chunk {i + 1}/{len(chunks)}: no failure evidence found."
                        )
                        continue

                    cleaned_json.update(
                        self._build_chunk_metadata(
                            step_name=step_name,
                            chunk=chunk,
                            chunk_index=i,
                            total_chunks=len(chunks),
                        )
                    )
                    step_chunks.append(cleaned_json)

                results.append(
                    {
                        "step_name": step_name,
                        "chunks": step_chunks,
                        "step_document": self._build_step_document(
                            step_name, step_chunks
                        ),
                    }
                )

            except Exception as e:
                print(f"[ERROR] Processing step '{step_name}': {str(e)}")
                results.append({"step_name": step_name, "chunks": [], "error": str(e)})

        return results

    # ------------------------------------------------------------------
    def _check_if_fallback_needed(
        self, step_payload_json: str, model_name: str
    ) -> bool:
        """
        Check if iterative fallback summarization is needed.

        Returns True if combined chunks exceed 50% of model context limit.
        """
        try:
            # Get model context limit
            from utilities.model_token_config import get_model_context_limit

            context_limit = get_model_context_limit(model_name)

            # Estimate tokens in payload
            tokens = self._estimate_tokens(step_payload_json)

            # Add prompt overhead (~2000 tokens)
            total_tokens = tokens + 2000

            # Check if exceeds 50% of context
            usage_ratio = total_tokens / context_limit

            if usage_ratio > 0.5:
                print(f"[FALLBACK] Payload size: {total_tokens:,} tokens ({usage_ratio*100:.1f}% of {context_limit:,})")
                print(f"[FALLBACK] Enabling iterative summarization")
                return True

            return False
        except Exception as e:
            # If check fails, don't use fallback
            print(f"[FALLBACK] Check failed: {e}, using standard approach")
            return False

    # ------------------------------------------------------------------
    def _iterative_summarization(
        self, step_name: str, chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Iterative summarization for large log steps.

        Process chunks incrementally, building up overall failure info.
        This is a fallback strategy for very large logs that exceed
        50% of model context limit.
        """
        print(f"[FALLBACK] Processing {len(chunks)} chunks iteratively")

        # Start with empty summary
        accumulated_summary = {
            "step_name": step_name,
            "sha_fail": self.sha_fail,
            "log_content": "",
            "error_context": [],
            "relevant_files": [],
            "error_types": [],
        }

        # Process chunks in batches
        for i, chunk in enumerate(chunks):
            print(f"[FALLBACK] Processing chunk {i+1}/{len(chunks)}")

            # Build incremental prompt
            prompt = f"""
You are a CI log analyzer working incrementally.

You have a PARTIAL summary of a CI step failure, and you are receiving ONE MORE CHUNK of log data.

## Current Summary (so far):
{json.dumps(accumulated_summary, indent=2, ensure_ascii=False)}

## New Chunk to Integrate:
{json.dumps(chunk, indent=2, ensure_ascii=False)}

## Task:
Update the summary by integrating information from the new chunk.

Rules:
- Append new error_context entries (deduplicate similar ones)
- Add new relevant_files (deduplicate by path)
- Add new error_types (deduplicate by category+subcategory)
- Extend log_content with new information
- Keep step_name and sha_fail unchanged

Output STRICT JSON matching this schema:
{{
  "step_name": "{step_name}",
  "sha_fail": "{self.sha_fail}",
  "log_content": "...",
  "error_context": [...],
  "relevant_files": [...],
  "error_types": [...]
}}

Return ONLY JSON, no markdown fences, no extra text.
"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)]).content
                content = self.load_json_maybe_fenced(response)

                if not content or not content.strip():
                    print(f"[FALLBACK] Empty response for chunk {i+1}, skipping")
                    continue

                try:
                    updated_summary = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        cleaned = _clean_malformed_json(content)
                        updated_summary = json.loads(cleaned)
                    except json.JSONDecodeError:
                        if demjson3 is not None:
                            try:
                                cleaned = _clean_malformed_json(content)
                                updated_summary = demjson3.decode(cleaned)
                            except Exception:
                                print(f"[FALLBACK] JSON parse failed for chunk {i+1}, skipping")
                                continue
                        else:
                            print(f"[FALLBACK] JSON parse failed for chunk {i+1}, skipping")
                            continue

                # Update accumulated summary
                accumulated_summary = updated_summary

            except Exception as e:
                print(f"[FALLBACK] Error processing chunk {i+1}: {e}")
                continue

        print(f"[FALLBACK] Iterative summarization complete")
        return accumulated_summary

    # ------------------------------------------------------------------
    def generate_log_summary(self, all_step_outputs) -> list[dict[str, Any]]:
        """
        Generate a structured final error summary from error details,
        workflow tools, and validation checks.
        """
        print(" Running Tool: _generate_summary")
        log_details = []
        for step in all_step_outputs:
            step_name = step.get("step_name", "UNKNOWN_STEP")
            chunks = step.get("chunks", [])

            step_payload = {
                "step_name": step_name,
                "chunks": chunks,
            }
            step_payload_json = json.dumps(step_payload, indent=2, ensure_ascii=False)

            # Check if fallback needed (payload > 50% of model context)
            if self._check_if_fallback_needed(step_payload_json, self.model_name):
                try:
                    summary = self._iterative_summarization(step_name, chunks)
                    log_details.append(summary)
                    continue  # Skip standard processing
                except Exception as e:
                    print(f"[FALLBACK] Iterative summarization failed: {e}")
                    print(f"[FALLBACK] Falling back to standard processing")
                    # Continue with standard processing below

            prompt = f"""
You are a CI log analyzer.

You receive pre-processed CI log information for a FAILED CI RUN, for a SINGLE CI STEP.

## Input Details
--- PRE-PROCESSED Log information for Given STEP DATA (JSON) ---
{step_payload_json}

Using ONLY this information for THIS STEP (all its chunks), produce a structured summary
of the CI failure for this step using the following STRICT JSON schema
(**do not add or remove top-level keys**)

{{
"step_name": "{step_name}",
"sha_fail": "{self.sha_fail}",
"log_content": "<Explain overall details of the CI log in natural language>",
"error_context": [
    "English explanation(s) of the root cause(s) visible of CI workflow failed supported by log evidence. Provide detail and evident reasons of the given step failure."
],
"relevant_files": [
    {{
    "file": "path/to/file.py",
    "line_number": 123,
    "reason": "Short explanation of why this file is tied to the failure."
    }},
],
"error_types": [
    {{
    "category": "High-level category, e.g. 'Test Failure', 'Runtime Error', 'Dependency Error', 'Configuration Error', 'Code Formatting', 'Type Checking'",
    "subcategory": "More specific description, e.g. 'Runtime Error – ValueError in Optuna objective', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
    "evidence": 'Short quote or paraphrase from THIS CHUNK that justifies this classification.'
    }}
]
}}

### Rules (IMPORTANT)

- "step_name": the CI step name (use the input step_name).
- "sha_fail": the failing commit SHA (given).
- "log_content":
  - A concise but informative natural-language description of what happened in this CI STEP.
  - You may integrate information across ALL chunks for this step.

- "error_context":
  - A list of English explanations of the root cause(s) of the CI failure related to THIS STEP.
  - Each entry must be supported by evidence in the summaries and/or relevant_failures.
  - If nothing meaningful appears, use an empty list [].

- "relevant_files":
  - **CRITICAL**: Include ALL files with DIRECT or INDIRECT relevance to THIS STEP's failure
  - Consider all chunk-level data ("relevant_files", "relevant_failures", and summaries)
  - **MUST INCLUDE files from:**
      * ANY traceback, exception, or stack trace (extract file paths from ALL tracebacks)
      * Failing tests, assertion errors, runtime exceptions
      * Setup/installation scripts that failed (install.py, setup.py, build.sh, etc.)
      * Configuration files in dependency/environment errors (pyproject.toml, package.json, etc.)
      * Build/compilation errors (Makefile, build scripts, source files)
      * Files explicitly mentioned in error messages or failure logs
      * Test files that were being executed when failure occurred (even for infrastructure failures)
      * Source files imported/used by tests that failed or were running at failure time
  - **DO NOT exclude files just because:**
      * They are setup/installation scripts
      * The error is "infrastructure" or "environment" related
      * The failure happened during setup rather than test execution
      * The file is a dependency/configuration file
  - For infrastructure failures (DNS, cache, CUDA version, etc.): include BOTH:
      * Files that caused or contributed to the infrastructure failure (e.g., installation scripts)
      * Test files that were queued or executing when the external failure occurred
  - Deduplicate by "file" path. If the same file appears with different reasons, merge
    them into one concise, evidence-based "reason".
  - "line_number":
      * Use the failing line number if it is clearly shown in the logs (e.g., from traceback)
      * otherwise null.
  - If no file appears in any log/error/traceback, return "relevant_files": [].
  - Provide evidence-based "reason" explaining how this file is tied to the CI failure.

- "error_types":
  - Describe the kinds of errors visible in this STEP:
      * Test failures,
      * Runtime exceptions,
      * Dependency / environment issues,
      * Configuration / CI YAML problems,
      * Code formatting or linting issues,
      * Type checking errors, etc.
  - Each entry MUST include:
      * "category"  (broad bucket),
      * "subcategory" (more specific),
      * "evidence"   (short quote or paraphrase from summaries / relevant_failures).

### Global Rules
1. Use ONLY the information from the STEP JSON shown below.
2. Use null for any unknown scalar values (e.g., line_number if not visible).
3. Do NOT add extra top-level keys.
4. Return STRICT JSON ONLY — no markdown, no comments, no natural language outside JSON.

### Output Rules (STRICT)
- Output MUST be a single raw JSON object.
- Do NOT wrap the JSON in triple backticks.
- Do NOT include ```json or any other marker/fence.
- Do NOT add any text before or after the JSON.

### OUTPUT GUIDELINES
- Keep log_content CONCISE - include only essential information
- Include ALL important error causes in error_context (don't artificially limit)
- Include ALL relevant files that are clearly tied to failures
- Include ALL distinct error types found
- Be thorough but concise - output only what's needed (not padding)
"""

            try:
                # Use all output capacity that safely remains after the
                # measured prompt and context reserve. This is a ceiling, not
                # a generation target.
                max_tokens = self._adaptive_output_limit(prompt)
                response = invoke_llm_with_retry(
                    llm=self.llm,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    parse_json=False,
                )
                content = self.load_json_maybe_fenced(response)

                if not content or not content.strip():
                    raise ValueError(
                        "LLM returned an empty response for generate_log_summary"
                    )

                # Check if LLM returned an error response
                if isinstance(content, dict) and "error" in content:
                    raise ValueError(f"LLM returned error: {content.get('error')}")

                try:
                    summary = json.loads(content)
                except json.JSONDecodeError as json_err:
                    # Try repairing truncated/malformed JSON
                    try:
                        cleaned = _clean_malformed_json(content)
                        summary = json.loads(cleaned)
                        print(f"  ✓ Repaired malformed JSON for step '{step_name}'")
                    except json.JSONDecodeError:
                        # Fall back to demjson3 for lenient parsing
                        if demjson3 is not None:
                            try:
                                cleaned = _clean_malformed_json(content)
                                summary = demjson3.decode(cleaned)
                                print(f"  ✓ Parsed with demjson3 for step '{step_name}'")
                            except Exception as dec_err:
                                # Log the exact parse error with context
                                error_context = content[:500] if len(content) > 500 else content
                                raise ValueError(
                                    f"JSON parse failed after all attempts.\n"
                                    f"Original error: {json_err}\n"
                                    f"Demjson3 error: {dec_err}\n"
                                    f"Content preview: {error_context}"
                                )
                        else:
                            # No demjson3, provide detailed error
                            error_context = content[:500] if len(content) > 500 else content
                            raise ValueError(
                                f"JSON parse failed (demjson3 not installed).\n"
                                f"Error: {json_err}\n"
                                f"Content preview: {error_context}\n"
                                f"Hint: Install demjson3 for more lenient JSON parsing"
                            )

                # Validate that we got a valid summary structure
                if not isinstance(summary, dict):
                    raise ValueError(f"Expected dict, got {type(summary)}")

                # Check for required fields
                if "step_name" not in summary:
                    summary["step_name"] = step_name

                log_details.append(summary)
            except Exception as e:
                # If one chunk fails, log and continue
                self._log_error(
                    method="generate_log_summary",
                    error=e,
                    step=f"{step_name}",
                )

        return log_details

    # ------------------------------------------------------------------
    def full_content_summary(
        self, log_details: list[dict[str, Any]], workflow_details: Any
    ) -> dict[str, Any]:
        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read step-level CI job log summaries and workflow details, then produce a **single, structured, evidence-based JSON summary** that explains why the CI run failed and clearly classifies the errors by category and subcategory.

---

## INPUTS

1. CI Log Details (from step analysis, list of objects):

Each element in `log_details` corresponds to ONE CI step and typically includes:
- "step_name": name of the CI step.
- "sha_fail": the failing commit SHA.
- "log_content": natural-language description of what happened in this step.
- "error_context": list of step-level explanations of root causes in this step.
- "relevant_files": list of files tied to the failure in this step, each with:
  - "file"
  - "line_number" (may be null)
  - "issue_type": short failure classification for this file (may be null)
  - "failed_cmd": the command that triggered the failure for this file (may be null)
  - "failed_tool": the tool that reported the failure for this file (may be null)
  - "reason"
- "error_types": list of error classifications for this step, each with:
  - "category"
  - "subcategory"
  - "evidence"

Full step-level details:
{json.dumps(log_details, indent=2, ensure_ascii=False)}

2. Workflow Details:

Parsed CI workflow (e.g., GitHub Actions YAML) including jobs, steps, and commands.
Use this to map failing steps to their jobs and commands when possible.

Full workflow details:
{json.dumps(workflow_details, indent=2, ensure_ascii=False)}

---

## OUTPUT FORMAT (strict JSON only)

Return a SINGLE aggregated summary for the entire failed run using this exact structure:

{{
  "error_context": [
    "Plain-English explanation(s) of the root cause(s), supported by log evidence. Mention all the steps involved in the failure and how and why it failed."
  ],
  "failure_signals": [
    "Concise observable pattern that identifies the failure. INCLUDE when available: tool/validator, version, error code, key tokens/symbols, error message, location.",
    "Format: '<tool> [version] <error_code>: <error_msg> at <file>:<line> [key_tokens]'",
    "Examples:",
    "  - 'mypy 1.8.0 error [arg-type]: Incompatible type for argument 1 (got Optional[Any], expected str) at helpers.py:42 [joinpath]'",
    "  - 'pytest 7.4.3: 3 test failures in test_auth.py::test_login [AssertionError: expected 200, got 401]'",
    "  - 'ruff 0.1.9 F401: Unused import in 5 files [numpy.typing.DTypeLike]'",
    "  - 'npm install: ERESOLVE dependency conflict [react@18 vs react@17]'",
    "Include these identifiers to help pattern matching and debugging."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "issue_type": "Short failure classification for this file, e.g. 'Test Failure', 'Import Error', 'Type Error', 'Dependency Error', 'Lint Error'. Use null if not clearly tied to a failure.",
      "failed_cmd": "The exact command that triggered the failure for this file, e.g. 'pytest tests/', 'python -m mypy src/'. Use null if not identifiable.",
      "failed_tool": "The tool that reported or caused the failure for this file, e.g. 'pytest', 'mypy', 'flake8', 'pip'. Use null if not identifiable.",
      "reason": "Short evidence-based explanation of why this file is tied to the failure."
    }}
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Code Formatting', 'Dependency Error', 'Test Failure', 'Runtime Error', 'Type Checking', 'Configuration Error'",
      "subcategory": "More specific type under that category, e.g. 'Unused Import', 'Line Length Exceeded', 'ImportError: No module named X', 'AssertionError', 'Missing dependency', 'Mypy type mismatch'",
      "evidence": "Brief quote or paraphrase from logs that proves this classification."
    }}
  ],
  "failed_job": [
    {{
      "job": "Job name or ID",
      "step": "Step name that failed",
      "command": "Exact command or action that caused the failure"
    }}
  ]
}}

---

## INSTRUCTIONS

1. **Aggregate across ALL steps (do not treat them independently).**
   - Read every element in `log_details`.
   - Combine their information into ONE global summary for the entire CI run.

2. **Failure Signals (aggregate from chunks).**
   - COLLECT all "failure_signals" from the chunk-level data in log_details
   - Deduplicate identical signals
   - If chunks provide no signals, extract them from error_types and relevant_failures
   - Format: '<tool> [version] <error_code>: <error_msg> at <file>:<line> [key_tokens]'

3. **Error Context (global explanation).**
   - Use English sentences summarizing the main root cause(s) of the failure.
   - Explicitly reference the step names involved and has failed in that steps, e.g.:
     - "In step 'Install dependencies', pip failed due to missing package X..."
     - "In step 'Run tests', pytest reported failing tests because..."
   - If multiple steps contribute to the failure, mention each clearly.
   - Base your explanations on `log_content`, `error_context`, and `error_types` from the steps.

3. **Failure Signals (NEW - concise observable patterns).**
   - Extract CONCISE, OBSERVABLE patterns that identify the failure.
   - Format: "tool: specific error pattern at location"
   - Examples:
     * "mypy: X | Y union syntax requires Python 3.10 at cloudinit/distros/__init__.py:154,158"
     * "pylint E1131: unsupported-binary-operation at cloudinit/distros/__init__.py:154,158"
     * "pytest: 3 tests failed in test_auth.py::test_login"
     * "ruff F401: unused imports in 5 files"
   - Be SPECIFIC: include tool name, error code/pattern, file paths, line numbers.
   - Keep signals SHORT (1 sentence per signal).
   - Each signal should be independently recognizable.
   - any token of code that is buggy

4. **Relevant Files (deduplicated across steps).**
   - Consider all `relevant_files` from all steps.
   - Deduplicate by `"file"` path: each file should appear at most once in the final list.
   - If the same file appears in multiple steps, merge the reasons into a single concise,
     evidence-based `"reason"` that reflects all relevant contexts.
   - Use `"line_number":` the most specific failing line if available; otherwise `null`.
   - For each file, analyze the failure evidence together with the matching workflow job/step/command to infer:
     - `"failed_cmd"` from the exact workflow `run` command or `uses` action tied to the failing step.
     - `"failed_tool"` from the reporting tool or command invoked in that step (for example `pytest`, `mypy`, `ruff`, `flake8`, `pip`, `npm`, `go test`).
     - `"issue_type"` from the file-specific failure mode described in the logs (for example test failure, lint error, import error, type error, dependency error).
   - Prefer file-specific values from the step analysis when present. Otherwise, infer them from the workflow step that produced the file-level failure. Use `null` only when the value cannot be supported by the logs or workflow.
   - The `"reason"` for each file must explain why that file is tied to the failure and, when possible, mention the step/tool/command that connects the file to the failing workflow step.
   - Include only files that are clearly tied to failures, errors, or critical warnings.
   - If no such file exists, return `"relevant_files": []`.

5. **Error Types (all distinct types found).**
   - Look at all `error_types` from all steps.
   - Aggregate them into a list of distinct (category, subcategory) pairs.
   - For each distinct pair, keep one entry with:
     - `"category"` and `"subcategory"` exactly once.
     - `"evidence"` summarizing or quoting representative log evidence (you may combine evidence from multiple steps briefly).

6. **Failed Job (use workflow + step info).**
   - Use both `log_details` and `workflow_details` to identify which job(s) and step(s) failed.
   - Match step names from `log_details` to steps in the workflow (by their `"name"` field) to infer:
     - `"job"`: the job display name or ID (from the workflow, if available; otherwise null).
     - `"step"`: the failing step name from `log_details`.
     - `"command"`:
       - The value of `"run"` if present, or `"uses"` if it is an action reference.
       - If no command can be found, use `null`.
   - If multiple jobs/steps clearly fail, include multiple entries in `"failed_job"`.
   - Reuse this same step-to-workflow mapping when populating each file's `"failed_cmd"` and `"failed_tool"` so the file-level metadata stays consistent with the recorded failed job/step.

7. **Output Rules**
   - Return **only valid JSON** — no markdown, commentary, or code fences.
   - Do not hallucinate; use `null` for unknown values (e.g., line_number, job, command).
   - Merge duplicates carefully (same file paths, same category/subcategory pairs).
   - Ensure every item is concise, evidence-based, and traceable to the logs and/or workflow.
"""
        try:
            max_tokens = self._adaptive_output_limit(prompt)
            response = invoke_llm_with_retry(
                llm=self.llm,
                prompt=prompt,
                max_tokens=max_tokens,
                parse_json=False,
            )
            content = self.load_json_maybe_fenced(response)

            if not content or not content.strip():
                raise ValueError(
                    "LLM returned an empty response for full_content_summary"
                )

            try:
                summary = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Try cleaning malformed JSON first
                try:
                    cleaned = _clean_malformed_json(content)
                    summary = json.loads(cleaned)
                    print(f"  ✓ Repaired malformed JSON in full_content_summary")
                except json.JSONDecodeError:
                    # Fall back to demjson3
                    if demjson3 is not None:
                        try:
                            cleaned = _clean_malformed_json(content)
                            summary = demjson3.decode(cleaned)
                            print(f"  ✓ Parsed with demjson3 in full_content_summary")
                        except Exception as dec_err:
                            error_context = content[:500] if len(content) > 500 else content
                            raise ValueError(
                                f"JSON parse failed after all attempts.\n"
                                f"Original error: {json_err}\n"
                                f"Demjson3 error: {dec_err}\n"
                                f"Content preview: {error_context}"
                            )
                    else:
                        error_context = content[:500] if len(content) > 500 else content
                        raise ValueError(
                            f"JSON parse failed (demjson3 not installed).\n"
                            f"Error: {json_err}\n"
                            f"Content preview: {error_context}\n"
                            f"Hint: Install demjson3 for more lenient JSON parsing"
                        )
            print(" Completed: _generate_summary")

            summary["sha_fail"] = self.sha_fail
            summary["id"] = self.task_id

            return summary

        except Exception as e:
            error_dir = os.path.join(
                self.config["exception_dir"], "interrupted_error_log"
            )
            os.makedirs(error_dir, exist_ok=True)

            error_data = {
                "sha_fail": self.sha_fail,
                "error": str(e),
                "tool": "ErrorContextExtractionAgent.run",
            }

            error_file = os.path.join(error_dir, f"{self.sha_fail}_error.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=4)

            return {"error": f"Failed to generate summary: {e}"}

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        print(f"Fully Autonomous Execution for Commit: {self.sha_fail}")

        # Check if logs are available (use self.ci_log for CILogAnalyzerLLM)
        logs_to_check = getattr(self, 'logs', None) or getattr(self, 'ci_log', None)
        if not logs_to_check or (isinstance(logs_to_check, str) and len(logs_to_check.strip()) == 0):
            return {
                "error": "No CI logs available in dataset for this issue.",
                "sha_fail": self.sha_fail,
                "id": self.task_id,
                "reason": "empty_logs",
            }

        selected_logs = self.ci_log_analysis()
        if not selected_logs:
            return {
                "error": "CI log analysis produced no usable step data.",
                "sha_fail": self.sha_fail,
                "id": self.task_id,
                "reason": "no_steps_extracted",
            }

        log_details = self.generate_log_summary(selected_logs)
        if not log_details:
            return {
                "error": "Failed to generate structured summary from CI log steps.",
                "sha_fail": self.sha_fail,
                "id": self.task_id,
                "reason": "summary_generation_failed",
                "steps_count": len(selected_logs),
            }

        generated_summary = self.full_content_summary(
            log_details, workflow_details=self.workflow
        )
        # Removed: chunk_summaries, step_documents, overall_ci_summary, analysis_document
        # Reason: Redundant - structured fields already contain all info
        # Added: failure_signals (concise observable patterns)
        return generated_summary

    # ------------------------------------------------------------------
    def _get_encoder(self):
        """Safely get a tiktoken encoder for the model."""
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for a given text using the cached encoder."""
        if text is None:
            return 0
        if self._encoder is None:
            # CI logs tokenize more densely than ordinary prose. Keep the
            # fallback conservative when the tokenizer is unavailable.
            return (len(text) + 1) // 2
        return len(self._encoder.encode(text))

    def _adaptive_output_limit(self, prompt: str) -> int:
        """Calculate a capability-aware output ceiling for a complete prompt."""
        prompt_tokens = self._estimate_tokens(prompt)
        output_tokens = calculate_adaptive_output_limit(
            self.model_name,
            prompt_tokens,
            safety_ratio=0.10,
        )
        if output_tokens <= 0:
            raise ValueError(
                f"Prompt does not fit {self.model_name!r} after its context safety reserve "
                f"({prompt_tokens:,} measured input tokens)"
            )
        return output_tokens

    def _log_error(self, method: str, error: Exception, step: str = ""):
        base_dir = os.path.join(self.config["exception_dir"], "interrupted_error_log")
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"{self.sha_fail}_{method}_{int(time.time())}_error.json"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "commit": self.sha_fail,
                    "method": method,
                    "step": step,
                    "error": str(error),
                },
                f,
                indent=2,
            )
        print(f"[ERROR LOGGED] {file_path}")

    def _filter_chunks(self, raw_chunks: list[str]) -> list[str]:
        """
        Keep:
        - All chunks in the first (n-6) that contain any ERROR_KEYWORDS (by word)
        - Always keep the last 6 chunks (serial order preserved)
        """
        n_chunks = len(raw_chunks)

        cutoff = n_chunks - 4
        filtered_chunks: list[str] = []

        # 1) Check the first (n-6) chunks and keep only those with error keywords
        for idx, chunk in enumerate(raw_chunks[:cutoff]):
            for line_no, line in enumerate(chunk.splitlines(), start=1):
                hits = self.is_line_error(line, ERROR_KEYWORDS)
                if hits:
                    # optional debug:
                    # print(f"[FILTER] chunk {idx}, line {line_no}: hits={hits}")
                    filtered_chunks.append(chunk)
                    break  # done with this chunk

        # 2) Append the last 4 chunks unconditionally (preserve serial order)
        filtered_chunks.extend(raw_chunks[cutoff:])

        if len(filtered_chunks) > 20:
            print(
                f"After filtering, too many chunks ({len(filtered_chunks)}). Truncating to last 12 "
                f"(checked first {cutoff}, always kept last 4)"
            )
            filtered_chunks = filtered_chunks[-12:]

        print(
            f"Filtered from {n_chunks} -> {len(filtered_chunks)} chunks "
            f"(checked first {cutoff}, always kept last 4)"
        )

        return filtered_chunks

    def _save_chunk_tracker(self, chunk_tracker: list[tuple]):
        debug_dir = os.path.join(self.config["out_folder"], "chunk_tracking")
        os.makedirs(debug_dir, exist_ok=True)

        file_path = os.path.join(debug_dir, "chunk_tracker.json")

        # Load existing data if file exists
        existing: list[dict[str, Any]] = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except Exception:
                existing = []

        # Append this run
        existing.append({"sha_fail": self.sha_fail, "chunks": chunk_tracker})

        # Write back pretty JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        print(f"[CHUNK TRACKER SAVED] {file_path}")

    def is_line_error(self, line: str, keywords) -> list[str]:
        """
        Return list of keywords that appear as separate words or exact phrases.
        - Single-word keywords: must match a whole word in the line (case-sensitive).
        - Multi-word keywords: matched as exact substrings (case-sensitive).
        - Special rule: if we see 'error' or 'errors' but the previous word is 'no'
        (e.g. 'no error', 'No errors'), we IGNORE that match.
        """
        hits = []

        # Very simple tokenization: split on whitespace, strip basic punctuation
        raw_tokens = line.split()
        tokens = [tok.strip("[]():,") for tok in raw_tokens]

        # 1) Phrase keywords -> use substring
        for kw in keywords:
            if " " in kw:
                if kw in line:  # exact phrase, case-sensitive
                    hits.append(kw)

        # 2) Single-word keywords -> must match full token
        # We also apply the "no error" rule here.
        for idx, tok in enumerate(tokens):
            for kw in keywords:
                if " " in kw:
                    continue  # phrases already handled

                if tok == kw:
                    # Special: ignore "error"/"errors" if previous token is 'no' (any case)
                    if kw in ("error", "errors") and idx > 0:
                        prev_tok = tokens[idx - 1]
                        if prev_tok.lower() == "no":
                            # e.g. "no error", "No errors" -> not a real error
                            continue

                    hits.append(kw)

        # Remove duplicates while preserving order
        seen = set()
        unique_hits = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                unique_hits.append(h)

        return unique_hits

    def load_json_maybe_fenced(self, text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            # drop opening fence line (``` or ```json)
            lines = lines[1:] if lines else lines
            # drop closing fence line if present
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _build_chunk_metadata(
        self,
        *,
        step_name: str,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
    ) -> dict[str, Any]:
        token_count = self._estimate_tokens(chunk)
        token_keywords = self._extract_chunk_keywords(chunk)
        code_context = self._extract_code_context(chunk)
        return {
            "chunk_index": chunk_index + 1,
            "chunk_total": total_chunks,
            "chunk_token_count": token_count,
            "token_keywords": token_keywords,
            "code_context": code_context,
            "chunk_document": self._build_chunk_document(
                step_name=step_name,
                chunk_index=chunk_index + 1,
                total_chunks=total_chunks,
                token_count=token_count,
                token_keywords=token_keywords,
                code_context=code_context,
            ),
        }

    def _extract_chunk_keywords(self, chunk: str, limit: int = 12) -> list[str]:
        candidates: list[str] = []
        for match in re.finditer(r"[A-Za-z0-9_./:-]{3,}", chunk):
            token = match.group(0).strip()
            lower = token.lower()
            if lower.startswith(("http://", "https://")):
                continue
            if token.isdigit():
                continue
            if "/" in token or "." in token or "_" in token or "::" in token:
                candidates.append(token)
                continue
            if any(ch.isupper() for ch in token[1:]):
                candidates.append(token)
                continue
            if lower in {
                "error",
                "errors",
                "failed",
                "failure",
                "traceback",
                "exception",
                "warning",
            }:
                candidates.append(token)

        seen = set()
        ordered: list[str] = []
        for token in candidates:
            if token not in seen:
                seen.add(token)
                ordered.append(token)
            if len(ordered) >= limit:
                break
        return ordered

    def _extract_code_context(self, chunk: str, limit: int = 5) -> list[str]:
        lines: list[str] = []
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if len(line) > 160:
                line = line[:157] + "..."
            if self._looks_like_code_or_trace(line):
                lines.append(line)
            if len(lines) >= limit:
                break
        return lines

    def _looks_like_code_or_trace(self, line: str) -> bool:
        code_markers = (
            "Traceback",
            'File "',
            " line ",
            "Error:",
            "Exception",
            "AssertionError",
            "FAILED",
            "E   ",
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "raise ",
            "pytest",
            "mypy",
            "ruff",
            "flake8",
        )
        if any(marker in line for marker in code_markers):
            return True
        return bool(
            re.search(
                r"[A-Za-z0-9_./-]+\.(py|js|ts|tsx|jsx|java|go|rb|php|yml|yaml|json|toml|ini|cfg)(:\d+)?",
                line,
            )
        )

    def _build_chunk_document(
        self,
        *,
        step_name: str,
        chunk_index: int,
        total_chunks: int,
        token_count: int,
        token_keywords: list[str],
        code_context: list[str],
    ) -> str:
        parts = [
            f"step={step_name}",
            f"chunk={chunk_index}/{total_chunks}",
            f"tokens={token_count}",
        ]
        if token_keywords:
            parts.append("keywords=" + ", ".join(token_keywords[:8]))
        if code_context:
            parts.append("code=" + " | ".join(code_context[:3]))
        return " ; ".join(parts)

    def _build_step_document(
        self, step_name: str, step_chunks: list[dict[str, Any]]
    ) -> str:
        lines = [f"step: {step_name}"]
        for chunk in step_chunks:
            summary = str(chunk.get("summary") or "").strip()
            failure_signals = chunk.get("failure_signals") or []
            failures = chunk.get("relevant_failures") or []
            files = [
                str(item.get("file") or "").strip()
                for item in (chunk.get("relevant_files") or [])
                if isinstance(item, dict)
            ]
            doc = str(chunk.get("chunk_document") or "").strip()
            if doc:
                lines.append(doc)
            if summary:
                lines.append(f"summary: {summary}")
            if failure_signals:
                lines.append(
                    "failure_signals: "
                    + " | ".join(
                        str(item).strip()
                        for item in failure_signals[:5]
                        if str(item).strip()
                    )
                )
            if failures:
                lines.append(
                    "failures: "
                    + " | ".join(
                        str(item).strip() for item in failures[:3] if str(item).strip()
                    )
                )
            if files:
                lines.append("files: " + ", ".join(path for path in files[:6] if path))
        return "\n".join(lines).strip()

    def _flatten_chunk_summaries(
        self, selected_logs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for step in selected_logs:
            step_name = str(step.get("step_name") or "unknown_step")
            for chunk in step.get("chunks", []) or []:
                if not isinstance(chunk, dict):
                    continue
                flattened.append(
                    {
                        "step_name": step_name,
                        "chunk_index": chunk.get("chunk_index"),
                        "chunk_total": chunk.get("chunk_total"),
                        "chunk_token_count": chunk.get("chunk_token_count"),
                        "token_keywords": chunk.get("token_keywords") or [],
                        "summary": chunk.get("summary") or "",
                        "failure_signals": chunk.get("failure_signals") or [],
                        "relevant_failures": chunk.get("relevant_failures") or [],
                        "relevant_files": chunk.get("relevant_files") or [],
                        "code_context": chunk.get("code_context") or [],
                        "chunk_document": chunk.get("chunk_document") or "",
                    }
                )
        return flattened


# ─────────────────────────────────────────────────────────────────────────────
# CILogAnalyzer — public adapter used by ci_context.py
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_logs(logs: Any) -> list[dict[str, Any]]:
    """
    Normalise raw CI logs to the ``[{step_name, log}]`` format that
    ``CILogAnalyzerLLM`` expects.
    """
    if isinstance(logs, list):
        out: list[dict[str, Any]] = []
        for i, item in enumerate(logs):
            if isinstance(item, dict):
                step_name = str(
                    item.get("step_name") or item.get("name") or f"step_{i + 1}"
                )
                log_text = str(
                    item.get("log") or item.get("text") or item.get("output") or ""
                )
            else:
                step_name = f"step_{i + 1}"
                log_text = str(item)
            out.append({"step_name": step_name, "log": log_text})
        return out

    # Plain string — treat as a single step
    return [{"step_name": "ci_log", "log": str(logs or "")}]


def _make_langchain_llm(llm: Any) -> Any:
    """
    Return a LangChain-compatible LLM object from *llm*.

    Supports:
      • Already LangChain (has ``invoke([HumanMessage(...)])``): returned as-is.
      • Callable / litellm-style (``llm(prompt) -> str``): wrapped in a thin
        shim so ``CILogAnalyzerLLM`` can call ``.invoke([HumanMessage(…)])``.
    """
    if llm is None:
        raise ValueError("llm must not be None for CILogAnalyzerLLM")

    # Test whether it already speaks LangChain.
    if callable(getattr(llm, "invoke", None)):
        return llm

    # Wrap a plain callable
    class _LLMShim:
        def __init__(self, fn: Any) -> None:
            self._fn = fn

        def invoke(self, messages: Any, **kwargs: Any) -> Any:
            # Extract text from a list of messages or pass through a plain string
            if isinstance(messages, list):
                content = " ".join(getattr(m, "content", str(m)) for m in messages)
            else:
                content = str(messages)
            try:
                # Context callables created by CIBench accept generation
                # ceilings. Older/plain callables remain supported below.
                result = self._fn(content, **kwargs)
            except TypeError:
                result = self._fn(content)

            # Return an object with a .content attribute
            class _Resp:
                def __init__(self, c: str) -> None:
                    self.content = c

            return _Resp(str(result) if result is not None else "")

    return _LLMShim(llm)


class CILogAnalyzer:
    """
    Public adapter for ``CILogAnalyzerLLM``, used by ``ci_context.py``.

    Accepts the interface that ``ci_context._run_log_analysis()`` uses:

        analyzer = CILogAnalyzer(
            logs=logs,           # str | list[{step_name, log}]
            sha_fail=sha_fail,
            workflow=workflow,
            workflow_path=workflow_path,
            llm=llm,             # LangChain ChatModel | callable | None
            model_name=model,
            task_id=task_id,
        )
        result = analyzer.run()  # -> same schema as CILogAnalyzerLLM.run()

    Raises ``RuntimeError`` when ``llm`` is ``None`` so that the caller's
    fallback (``_minimal_log_analysis``) is triggered automatically.
    """

    def __init__(
        self,
        *,
        logs: Any,
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: Any,
        model_name: str,
        task_id: str,
        output_dir: str | Path | None = None,
    ) -> None:
        if llm is None:
            raise RuntimeError(
                "CILogAnalyzer requires an LLM; llm=None -> using fallback analysis."
            )

        ci_log = _normalize_logs(logs)
        lc_llm = _make_langchain_llm(llm)

        self._inner = CILogAnalyzerLLM(
            repo_path=".",
            ci_log=ci_log,
            sha_fail=sha_fail,
            workflow=workflow,
            workflow_path=workflow_path,
            llm=lc_llm,
            model_name=model_name,
            task_id=task_id,
        )
        if output_dir is not None:
            output_path = Path(output_dir)
            self._inner.config["out_folder"] = str(output_path)
            self._inner.config["exception_dir"] = str(output_path / "exceptions")

    def run(self) -> dict[str, Any]:
        return self._inner.run()


# ═════════════════════════════════════════════════════════════════════════════
# Adapter functions (moved from ci_context.py for backward compatibility)
# ═════════════════════════════════════════════════════════════════════════════


def _run_log_analysis(
    instance: dict[str, Any],
    llm: Any,
    model: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run CI log analysis on an instance with caching.

    Checks cache first for faster reuse across ablations and agents.
    If not cached, generates analysis and saves to shared cache.

    Args:
        instance: Benchmark instance with logs, workflow, sha_fail
        llm: LLM instance
        model: Model name
        output_dir: Optional output directory for intermediate files

    Returns:
        Analysis result with error_context, relevant_files, error_types, failed_job
    """
    sha_fail = str(instance.get("sha_fail") or "")
    task_id = str(instance.get("instance_id") or instance.get("id") or sha_fail)
    workflow = instance.get("workflow") or ""
    workflow_path = str(instance.get("workflow_path") or "")
    logs = instance.get("logs") or instance.get("log") or ""

    # Use shared cache utility - checks cache first, generates if missing
    from utilities.ci_cache import load_structured_ci_failure

    result = load_structured_ci_failure(
        sha_fail=sha_fail,
        issue_id=task_id,
        logs=logs,
        workflow=workflow,
        workflow_path=workflow_path,
        llm=llm,
        model_name=model,
    )

    # If cache utility didn't return a result, fall back to direct analysis
    # (shouldn't happen, but defensive programming)
    if not result:
        print(f"    Warning: Cache utility returned empty result, running analyzer directly")
        analyzer = CILogAnalyzer(
            logs=logs,
            sha_fail=sha_fail,
            workflow=workflow,
            workflow_path=workflow_path,
            llm=llm,
            model_name=model,
            task_id=task_id,
            output_dir=output_dir,
        )
        result = analyzer.run()

    return result


def _log_analysis_to_context(
    log_result: dict[str, Any],
    instance: dict[str, Any],
    workflow_profile: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Convert log analysis result to context dict.

    Args:
        log_result: Output from _run_log_analysis
        instance: Original benchmark instance
        workflow_profile: Optional workflow profile (installation_cmd, validation_cmd)

    Returns:
        Context dict with overall_failure_reasons, effected_files, failed_jobs
    """
    if workflow_profile is None:
        workflow_profile = {}

    sha_fail = str(log_result.get("sha_fail") or instance.get("sha_fail") or "")
    task_id = str(
        log_result.get("id")
        or instance.get("instance_id")
        or instance.get("id")
        or sha_fail
    )
    repo_name = str(instance.get("repo_name") or "")
    repo_owner = str(instance.get("repo_owner") or "")
    repo = str(instance.get("repo") or f"{repo_owner}/{repo_name}".strip("/"))
    workflow_path = str(instance.get("workflow_path") or "")
    workflow_name = (
        str(instance.get("workflow_name") or "")
        or os.path.splitext(os.path.basename(workflow_path))[0]
    )

    # Map error_context -> overall_failure_reasons
    error_context = log_result.get("error_context") or []
    overall_failure_reasons = [
        str(item).strip() for item in error_context if str(item).strip()
    ]

    # Map error_types -> overall_error_types
    error_types_list = log_result.get("error_types") or []
    overall_error_types = []
    for et in error_types_list:
        if isinstance(et, dict):
            cat = str(et.get("category") or "").strip()
            subcat = str(et.get("subcategory") or "").strip()
            evidence = str(et.get("evidence") or "").strip()
            parts = [p for p in [cat, subcat, evidence] if p]
            if parts:
                overall_error_types.append(" | ".join(parts))

    # Map relevant_files -> effected_files
    relevant_files = log_result.get("relevant_files") or []
    effected_files = []
    for rf in relevant_files:
        if isinstance(rf, dict):
            effected_files.append(
                {
                    "file": rf.get("file", ""),
                    "reason": rf.get("reason", ""),
                    "issue_type": rf.get("issue_type"),
                    "failed_cmd": rf.get("failed_cmd"),
                    "failed_tool": rf.get("failed_tool"),
                    "line_number": rf.get("line_number"),
                }
            )

    # Map failed_job
    failed_job_data = (
        log_result.get("failed_job") or log_result.get("failed_jobs") or []
    )
    if not isinstance(failed_job_data, list):
        failed_job_data = [failed_job_data] if failed_job_data else []
    failed_jobs = []
    for fj in failed_job_data:
        if isinstance(fj, dict):
            failed_jobs.append(
                {
                    "job": fj.get("job", ""),
                    "step": fj.get("step", ""),
                    "command": fj.get("command", ""),
                }
            )

    return {
        "id": task_id,
        "sha_fail": sha_fail,
        "repo": repo,
        "repo_name": repo_name,
        "repo_owner": repo_owner,
        "workflow_name": workflow_name,
        "workflow_path": workflow_path,
        "overall_failure_reasons": overall_failure_reasons,
        "overall_error_types": overall_error_types,
        "effected_files": effected_files,
        "failed_jobs": failed_jobs,
        "workflow_profile": workflow_profile,
        "_log_analysis": log_result,
    }
