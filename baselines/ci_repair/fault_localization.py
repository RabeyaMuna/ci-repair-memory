import os
import json
import subprocess
import sys
import math
import re
import yaml
import time
from pathlib import Path
import demjson3
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from utilities.model_token_limits import get_prompt_token_budget
from utilities.load_config import load_config
from utilities.snippet_extractor import extract_snippet_from_line_range, find_line_range
from utilities.symbols_outline import build_outline, format_outline
from utilities.chunking_logic import chunk_log_by_tokens, chunk_lines_with_overlap, estimate_tokens


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

api_key = os.getenv("OPENAI_API_KEY")


class FaultLocalization:
    def __init__(
        self,
        sha_fail: str,
        repo_path: str,
        error_logs: dict,
        workflow: str,
        llm: ChatOpenAI,
        model_name: Optional[str] = None,
        changed_files_info: Optional[dict] = None,
    ):
        """
        FaultLocalization agent.

        changed_files_info format (from collect_changed_files_for_fail_and_parent):
        {
          "sha_fail": "<sha_fail>",
          "changed_files": [
            {
              "commit": "<commit_sha>",
              "file_path": "<path/to/file>",
              "diff": "<unified diff>",
            },
            ...
          ]
        }
        """
        # Unpack OmegaConf + project root if your loader returns a tuple
        cfg_result = load_config()
        if isinstance(cfg_result, tuple) and len(cfg_result) == 2:
            self.config, self.project_root = cfg_result
        else:
            self.config, self.project_root = cfg_result, None

        self.error_logs = error_logs or {}
        self.changed_files_info = changed_files_info or {"changed_files": []}

        # Use correct defaults/types
        self.error_context = self.error_logs.get("error_context", [])         # list
        self.error_types = self.error_logs.get("error_types", [])             # list

        # Handle both singular/plural to be robust
        self.failed_jobs = self.error_logs.get(
            "failed_jobs",
            self.error_logs.get("failed_job", []),
        )  # list

        # Fix tuple-as-key bug
        self.relevant_files = self.error_logs.get("relevant_files", [])       # list

        self._has_checked_out = False
        self.workflow = workflow
        self.repo_path = repo_path
        self.failed_commit = sha_fail
        self.id = self.error_logs.get("id", "")

        self.model_name = model_name
        self.llm = llm

        self.parser = JsonOutputParser()

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> Dict:
        try:
            self._checkout_failed_commit_once()

            print("[Step 3] Running final fault localization...")
            suspecious_files = self.select_suspecious_files()

            result = self._final_fault_localization(suspecious_files)
            result["id"] = self.id 
            return result

        except Exception as e:
            base_dir = os.path.join(
                self.config["exception_dir"],
                "interrupted_fault_localization",
            )
            os.makedirs(base_dir, exist_ok=True)
            filepath = os.path.join(base_dir, f"{self.failed_commit}_bug.json")
            error_info = {
                "sha_fail": self.failed_commit,
                "error": str(e),
                "tool": "FaultLocalization",
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=4)
            return error_info

    # ------------------------------------------------------------------ #
    # Git helpers
    # ------------------------------------------------------------------ #

    def _checkout_failed_commit_once(self):
        try:
            subprocess.run(
                ["git", "checkout", self.failed_commit],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
            )
            self._has_checked_out = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git checkout failed: {e.stderr.decode()}")

    # ------------------------------------------------------------------ #
    # Suspicious file selection
    # ------------------------------------------------------------------ #

    def select_suspecious_files(self) -> List[Dict[str, Any]]:
        """
        1) Start from log_analyzer `relevant_files` (filtered by extension).
        2) Add extra files from `changed_files_info["changed_files"]` if the LLM
        says the diff is suspicious for the failed jobs.

        Returns a list of dicts that at least contain "file" so that
        _final_fault_localization can resolve paths.
        """
        suspecious_files: List[Dict[str, Any]] = []

        # 1) Relevant files from error_logs
        for item in self.relevant_files:
            file_path = (item.get("file") or item.get("path") or "").strip()
            if not file_path:
                continue

            ext = Path(file_path).suffix.lower()
            if ext not in {".py", ".txt", ".toml"}:
                continue

            suspecious_files.append({"file": file_path})

        # Build a set of already-selected file paths from suspecious_files
        seen_paths: set[str] = set()
        for entry in suspecious_files:
            p = (entry.get("file") or entry.get("path") or "").strip()
            if p:
                seen_paths.add(p)

        # 2) Changed files from changed_files_info
        changed_files_list = self.changed_files_info.get("changed_files", []) or []
        if not changed_files_list:
            return suspecious_files

        failed_jobs_text = json.dumps(self.failed_jobs, indent=2, ensure_ascii=False)

        for item in changed_files_list:
            file_path = (item.get("file_path") or "").strip()
            if not file_path:
                continue

            # If this file is already in suspecious_files, skip immediately
            if file_path in seen_paths:
                # Already included from relevant_files or earlier changed_files
                continue

            ext = Path(file_path).suffix.lower()
            if ext not in {".py", ".txt"}:
                continue

            changed_content = item.get("diff", "")

            prompt = f"""
You are a **Suspicious File Selector** for CI failures.

Goal:
Given a file path, the unified diff of that file for a failed commit, and the CI failed jobs description, decide whether this file's changes are likely responsible for (or closely related to) the CI failure. 
If you have strong evident of the change of code(diff) in the failed commit is the reason of failed jobs, only then set `is_suspicious` as true. Do not speculate.

Return **only** a JSON object with this exact schema, as plain text:

{{
  "is_suspicious": true or false
}}

Hard rules:
- Do NOT add any markdown fences (no ```json, no ```).
- Do NOT add any extra keys, comments, or explanation.
- Do NOT add any surrounding text before or after the JSON.
- The response must be a single valid JSON object only.

========================================
FILE PATH:
{file_path}

UNIFIED DIFF FOR THIS FILE:
{changed_content}

FAILED JOBS (CI context):
{failed_jobs_text}
========================================
"""

            try:
                raw_response = self.llm.invoke(prompt).content.strip()
                if raw_response.startswith("```"):
                    raw_response = raw_response.strip("` \n")

                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError:
                    parsed = demjson3.decode(raw_response)

                if isinstance(parsed, dict) and parsed.get("is_suspicious") is True:
                    suspecious_files.append({"file": file_path})
                    seen_paths.add(file_path)  # so we never process it again
                    print(f"[Selector] Marked '{file_path}' as suspicious based on diff.")
                else:
                    print(f"[Selector] '{file_path}' not suspicious.")
            except Exception as e:
                print(f"[Selector] Error deciding for {file_path}: {e}")

        return suspecious_files


    # ------------------------------------------------------------------ #
    # Final fault localization over selected files
    # ------------------------------------------------------------------ #

    def _final_fault_localization(self, suspecious_files: list) -> Dict[str, Any]:
        print("[Tool] read_error_file called")

        fault_localization: List[Dict[str, Any]] = []
        for item in suspecious_files:
            file_path = (item.get("file") or item.get("path") or "").strip()
            if not file_path:
                continue

            ext = Path(file_path).suffix.lower()
            if ext not in {".py", ".toml", ".txt"}:
                continue

            resolved = self.find_full_file_path(file_path)
            if resolved["status"] != "found":
                print(f"[WARN] Could not resolve path for {file_path}")
                continue

            full_file_path = resolved["full_path"]

            content = self._read_file_content(full_file_path)
            if not content:
                continue

            file_type = self.detect_file_type(file_path)
            outline = build_outline(content) if file_type == "python" else ""
            numbered_full_content = self._numbered_file_content(content)

            chunks = self._chunk_file(numbered_full_content)
            num_chunks = len(chunks)

            # Per-chunk strict FL
            all_faults: List[Dict[str, Any]] = []

            for idx, ch in enumerate(chunks):
                file_summary = (
                    f"File: {file_path}\n"
                    f"Full Path: {full_file_path}\n\n"
                    f"**File Outline (symbol → [start–end] lines)**\n{outline}\n\n"
                    f"File Type: {file_type}\n"
                    f"Chunk index: {idx+1} of {num_chunks}\n"
                    f"Chunk lines ({ch['valid_start']}–{ch['valid_end']}):\n\n"
                    f"```{file_type}\n{ch['content']}\n```"
                )

                faults = self.fault_localization_based_on_ci_log(
                    file_path=file_path,
                    full_file_path=full_file_path,
                    file_summary=file_summary,
                    valid_start=ch["valid_start"],
                    valid_end=ch["valid_end"],
                    original_content=content,
                    chunk_idx=idx,
                    num_chunks=num_chunks,
                    faults=all_faults,
                    outline=outline,
                    chunk=ch["content"]
                )

                if faults:
                    all_faults.extend(faults)
            
            if all_faults:
                fault_localization.append(
                    {
                        "file_path": file_path,
                        "full_file_path": full_file_path,
                        "faults": all_faults,
                    }
                )

        results = {
            "sha_fail": self.failed_commit,
            "fault_localization_data": fault_localization,
        }

        return results

    # ------------------------------------------------------------------ #
    # Core FL for one chunk
    # ------------------------------------------------------------------ #
    
    def prompt_for_fault_localization(
    self,
    *,
    file_summary: str,
    valid_start: int,
    valid_end: int,
    faults: list,
    ) -> str:
        """
        Build the strict fault-localization prompt for a single (sub)chunk.

        - file_summary: includes file metadata + outline + the code chunk (already fenced).
        - valid_start/valid_end: absolute (file-level) line numbers for this chunk window.
        - faults: previously detected faults (used only to avoid duplicates).
        """
        return f"""
You are a **Strict Fault Localization Agent**.

Your task:
Given a numbered source-code CHUNK, the CI error context, workflow jobs, and a file outline, identify **all** distinct faults that explain the CI failure — not just the first one.
Use the outline only to determine the correct "fault_localization_level" (method/class/import_block/file/line) and to understand broader structure beyond the chunk.
Do NOT expand "line_range" to outline boundaries. Return ALL distinct faults present in this chunk that directly explain CI failures (do not stop at the first fault).
Output must be **valid JSON only** (array or empty array). No markdown, no commentary, no extra text.

==============================================================================
INPUT
------------------------------------------------------------------------------
SOURCE CODE (numbered lines; 1-based) with Outline of the file is given:
{file_summary}

Each outline entry contains:
- name: symbol or construct name (function/class/import)
- type: one of "method" | "class" | "import_block"
- start_line: first numbered line of the element
- end_line: last numbered line of the element

Use this outline ONLY to determine which scope a fault belongs to ("fault_localization_level").
Do NOT expand "line_range" to outline boundaries.

FAILED JOBS:
{self.failed_jobs}

ERROR TYPES:
{self.error_types}

ERROR CONTEXT:
{self.error_context}

CHUNK WINDOW: lines {valid_start}–{valid_end}

==============================================================================
RULES
------------------------------------------------------------------------------
R1. Detect All Matches
- Read every numbered line between {valid_start}–{valid_end}.
- Add every new fault that directly explains CI messages or rule codes (ruff, pylint, mypy, pytest, etc.) or reasons of Failed Jobs.
- Include formatting, linting, typing, runtime, and test failures if indicated by the logs.
- Do NOT stop after finding one issue; return every distinct fault in the chunk that matches CI evidence.

R2. Verify in Code
- Each fault must be observable in these lines or provably absent (for missing imports/symbols).
- Confirm CI log claims (missing symbol, unused import, annotation absence, etc.) against code shown.

R3. Outline-Based Scope Classification (NO expansion)
- For each detected fault, choose a "line_range" that covers the faulty line(s) and any necessary adjacent lines
    to show the complete faulty statement (it does NOT need to be the smallest possible range).
- "line_range" MUST remain within the CHUNK WINDOW {valid_start}–{valid_end}.
- Determine "fault_localization_level" by finding the smallest outline entry (tightest [start_line–end_line]) that fully contains the provided "line_range", with LINE-FIRST rules:

    • If the "line_range" is fully contained by a method/function outline entry
      → set "fault_localization_level" to "method".

    • Else if the "line_range" is fully contained by a class outline entry
      AND the faulty lines are NOT inside any method/function child of that class
      (i.e., the fault is at class scope: class docstring / decorators / class vars)
      → set "fault_localization_level" to "class".

    • Else if the "line_range" is fully contained by an import_block outline entry
      (or the faulty lines are import statements)
      → set "fault_localization_level" to "import_block".

    • Else
      → set "fault_localization_level" to "line".

  Hard constraints:
    - Do NOT use "file" as a fault_localization_level.
    - "line_range" MUST stay minimal (only the faulty statement + a little context), never the entire class/file.
    - If you cannot point to exact faulty line(s) within this chunk, do NOT return the fault.

- If outline is missing/empty/unusable, you may fall back to "line" when appropriate.
- If multiple faults occur in the SAME outline element, you may merge them into one JSON object and combine reasons,
    BUT do NOT merge unrelated faults and do NOT expand "line_range" to the element boundary.

R4. Evidence-Based Reasoning — MUST include WHAT + FIX + WHERE (Hard Requirement)
For EACH returned fault object, the "reason" field MUST include ALL of the following:

(1) WHAT (exact issue description)
- Describe precisely what is wrong in the code.
- Tie the issue to CI evidence when available (error message and/or rule code such as F401, E1101, mypy error, pytest failure).

(2) FIX (concrete action required)
- State exactly what needs to be changed using imperative language.
- Examples:
    • "Remove the unused import X"
    • "Add the missing import for Y"
    • "Change the argument type from X to Y"
    • "Update the function call to use Z"
    • "Fix formatting to satisfy rule E501"

(3) WHERE (exact line numbers)
- Explicitly state the exact line number(s) in THIS chunk where the issue occurs.
- Use ONLY one of these formats:
    • "Fault at line N"
    • "Fault at lines N–M"
    • "Fault at line N (and line K)"
- The referenced line numbers MUST fall within the returned "line_range".

DO NOT be vague or generic. DO NOT say "the class has a problem" or "there is a docstring issue". Provide specific details citing CI evidence and exact line numbers. If you cannot identify the exact faulty line(s) in this chunk, DO NOT return the fault.


R5. Line Range Integrity (Issue Range, not outline range)
- "line_range" must include the faulty line(s). It does NOT need to match outline boundaries.
- Always use the displayed (numbered) line indices, not inferred offsets.

R6. Fault Type & Level
- Choose "issue_type" precisely (formatting, linting, type_error, runtime_error, test_failure, dependency_error, docstring, complexity, other).
- Set "fault_localization_level" based on the outline containment rules in R3: line | method | class | import_block | file.

R7. Extended Reason Context
- In "reason", you may mention related decorators, helper calls, or affected functions that clarify the cause.
- Do NOT claim you inspected code outside this chunk. You may reference the outline structurally only.

R8. Missing Elements
- If CI cites a missing construct (e.g., import, symbol, type hint), confirm it is missing in the shown chunk scope if applicable,
    and record it with the correct "fault_localization_level" using outline-based classification.

R9. Output Contract (Hard)
- Return **strict JSON only**:
    • Either [] or a JSON array of objects matching the schema below.
    • No markdown fences, comments, or trailing commas.
- Do NOT include "code_snippet" — it will be added later by the caller.

R10. Line Number Formatting (Critical)
- The "line_range" array MUST contain plain decimal integers with NO leading zeros.
- If the source shows "0007:" or "0042:", you MUST output 7 or 42 in JSON.
- Examples:
    • Correct: "line_range": [1, 15]
    • Correct: "line_range": [7, 42]
    • INCORRECT: "line_range": [0001, 0015]
    • INCORRECT: "line_range": ["0001", "0015"]   (do NOT quote them)
- Always map the zero-padded display index NNNN to its integer value.

==============================================================================
OUTPUT SCHEMA (JSON array)
------------------------------------------------------------------------------
[
{{
    "line_range": [start_line, end_line],
    "reason": "Comprehensive explanation citing CI log messages and rule codes. If merged, include concise bullet-like sub-fault summaries. Mention relevant line numbers that contain the issue(s).",
    "issue_type": "formatting | linting | type_error | runtime_error | test_failure | dependency_error | docstring | complexity | other",
    "fault_localization_level": "line | method | class | import_block | file"
}},
...
]

==============================================================================
CHECKLIST BEFORE RETURNING
------------------------------------------------------------------------------
1) All new faults in {valid_start}–{valid_end} are included (do not stop at first).
2) Duplicates are avoided; only merge faults when they are the SAME underlying issue in the SAME outline element.
3) "line_range" includes the faulty line(s) (and needed statement context) and stays within {valid_start}–{valid_end}.
4) Each "reason" references concrete CI evidence or rule code AND points to specific line numbers in this chunk.
5) No duplicates of ALREADY DETECTED FAULTS.
6) Output is valid JSON only — no markdown, prose, or trailing commas.
7) If nothing new is found, return [].
""".strip()

                                      
    def fault_localization_based_on_ci_log(
    self,
    *,
    file_path: str,
    full_file_path: str,
    file_summary: str,
    valid_start: int,
    valid_end: int,
    original_content: str,
    chunk_idx: int,
    num_chunks: int,
    faults: list,
    outline: List[Dict[str, Any]],
    chunk: str,
    ) -> list:
        fault_locations: List[Dict[str, Any]] = []
        token_limit = get_prompt_token_budget(self.model_name)

        prompt = self.prompt_for_fault_localization(
            file_summary=file_summary,
            valid_start=valid_start,
            valid_end=valid_end,
            faults=faults,
        )

        print(f"[Chunk {chunk_idx+1}/{num_chunks}] Analyzing lines {valid_start}-{valid_end}...")

        def _invoke_and_parse(p: str):
            raw_response = self.llm.invoke([HumanMessage(content=p)]).content.strip()
            if raw_response.startswith("```"):
                raw_response = raw_response.strip("` \n")

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                parsed = demjson3.decode(raw_response)

            if not isinstance(parsed, list) or not parsed:
                return []
            return parsed

        model_for_count = self.model_name or "gpt-4o-mini"

        # -----------------------------
        # Case 1: prompt fits -> normal path
        # -----------------------------
        if estimate_tokens(prompt, model=model_for_count) <= token_limit:
            try:
                parsed_result = _invoke_and_parse(prompt)
            except Exception as e:
                self._save_fault_localization_error(
                    self.failed_commit,
                    e,
                    tool_name="fault_localization_based_on_ci_log",
                    prompt_name="llm_invocation",
                )
                print(f"[Chunk {chunk_idx}] LLM invocation error: {e}")
                return []
        else:
            # -----------------------------
            # Case 2: prompt too big -> subchunk the PROVIDED chunk only
            # sub_lines = (n_lines // 2) + 50 ; overlap stays 50
            # -----------------------------
            n_lines = len((chunk or "").splitlines())
            sub_lines = max(1, (n_lines // 2) + 50)

            print(
                f"[Chunk {chunk_idx+1}] Prompt too large "
                f"({estimate_tokens(prompt, model=model_for_count)} > {token_limit}). "
                f"Sub-chunking chunk content: {n_lines} lines -> {sub_lines} lines/chunk with 50 overlap."
            )

            subchunks = chunk_lines_with_overlap(
                chunk,
                lines_per_chunk=sub_lines,
                overlap=50,
            )

            parsed_result = []

            for sub_i, (sub_start_1b, sub_end_1b, sub_text) in enumerate(subchunks):
                # sub_start_1b/sub_end_1b are line numbers *within this chunk* (1-based)
                abs_start = (valid_start - 1) + sub_start_1b
                abs_end = (valid_start - 1) + sub_end_1b

                sub_file_summary = (
                    f"File: {file_path}\n"
                    f"Full Path: {full_file_path}\n\n"
                    f"**File Outline (symbol → [start–end] lines)**\n{outline}\n\n"
                    f"Chunk index: {chunk_idx+1} of {num_chunks} (sub {sub_i+1} of {len(subchunks)})\n"
                    f"Chunk lines ({abs_start}–{abs_end}):\n\n"
                    f"```text\n{sub_text}\n```"
                )

                sub_prompt = self.prompt_for_fault_localization(
                    file_summary=sub_file_summary,
                    valid_start=abs_start,
                    valid_end=abs_end,
                    faults=faults,
                )

                if estimate_tokens(sub_prompt, model=model_for_count) > token_limit:
                    print(
                        f"[Chunk {chunk_idx+1}] Skipping subchunk {sub_i+1}: "
                        f"prompt still too large ({estimate_tokens(sub_prompt, model=model_for_count)} > {token_limit})."
                    )
                    continue

                try:
                    sub_parsed = _invoke_and_parse(sub_prompt)
                except Exception as e:
                    self._save_fault_localization_error(
                        self.failed_commit,
                        e,
                        tool_name="fault_localization_based_on_ci_log",
                        prompt_name="llm_invocation_subchunk",
                    )
                    print(f"[Chunk {chunk_idx+1}] Subchunk LLM error: {e}")
                    continue

                parsed_result.extend(sub_parsed)

            if not parsed_result:
                print(f"[Chunk {chunk_idx}] No faults found.")
                return []

        # -----------------------------
        # Process parsed_result (unchanged)
        # -----------------------------
        for fault in parsed_result:
            line_range = fault.get("line_range")
            fault_level = fault.get("fault_localization_level")

            if not line_range:
                continue

            start, end = line_range
            if valid_start <= start and end <= valid_end:
                extended_range = self._expand_line_range_with_outline(
                    line_range=line_range,
                    outline=outline,
                    fault_level=fault_level,
                )
                print("\n--- Before Line range ---", line_range)
                fault["line_range"] = extended_range

                snippet = extract_snippet_from_line_range(
                    original_file_content=original_content,
                    line_range=extended_range,
                )
                fault["code_snippet"] = snippet

                fault_locations.append(fault)
                print("\n--- Fault Detected ---")
                print("Code Snippet:\n", snippet)
                print("After Extending, Line Range:", extended_range)
                print("---------------------\n")
            else:
                print(
                    f"[Chunk {chunk_idx+1}] Skipping fault outside chunk range "
                    f"{valid_start}-{valid_end}: {line_range}"
                )

        return fault_locations



    # ------------------------------------------------------------------ #
    # Path & file helpers
    # ------------------------------------------------------------------ #

    def find_full_file_path(self, file_path: str) -> dict:
        """
        Find the best matching full path for a given relative file_path inside repo_path.
        Prioritizes the candidate whose relative path best matches the requested file_path.
        """
        normalized = os.path.normpath(file_path.split(":", 1)[0].strip())
        abs_path = os.path.join(self.repo_path, normalized)
        file_name = os.path.basename(normalized)

        try:
            if os.path.exists(abs_path):
                return {"status": "found", "full_path": abs_path}

            candidates = []
            for root, _, files in os.walk(self.repo_path):
                if file_name in files:
                    candidate = os.path.join(root, file_name)
                    rel_candidate = os.path.relpath(candidate, self.repo_path)
                    score = len(
                        os.path.commonprefix(
                            [rel_candidate[::-1], normalized[::-1]]
                        )
                    )
                    candidates.append((score, candidate))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_match = candidates[0][1]
                return {
                    "status": "found",
                    "full_path": best_match,
                    "all_candidates": [c[1] for c in candidates],
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

        return {"status": "not_found"}

    def _read_file_content(self, resolved_path: str) -> str:
        """Return file content as a string, or '' if file is missing/unreadable."""
        if not os.path.exists(resolved_path):
            print(f"[WARN] File not found: {resolved_path}")
            return ""

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] Could not read {resolved_path}: {e}")
            return ""

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #

    def _call_llm_directly(self, prompt: str) -> dict:
        chunks = chunk_log_by_tokens(prompt, max_tokens=90000, model=self.model_name)

        for chunk in chunks:
            try:
                raw_response = self.llm.invoke([HumanMessage(content=chunk)]).content.strip()
                raw_response = re.sub(
                    r"^```(?:json)?\s*|```$",
                    "",
                    raw_response.strip(),
                    flags=re.DOTALL,
                )
                return self.safe_parse_json(raw_response)
            except Exception as e:
                print(f"[ERROR] LLM call failed while parsing: {e}")
                raise

    def safe_parse_json(self, text: str) -> Dict:
        try:
            return self.parser.parse(text)
        except Exception as e:
            print("[!] Primary parser failed. Attempting cleanup...")
            try:
                cleaned = text.strip().strip("```json").strip("```").strip()
                return json.loads(cleaned)
            except Exception as second_error:
                print("[!] Fallback parsing failed.")
                raise ValueError(f"JSON parsing failed: {e}\n\nRaw:\n{text}")

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    def _numbered_file_content(self, content: str, offset: int = 0) -> str:
        return "\n".join(
            f"{idx+1:04d}: {line}" for idx, line in enumerate(content.splitlines())
        )

    def detect_file_type(self, file_path: str) -> str:
        """Detect programming/config language based on file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".dockerfile": "dockerfile",
            ".rst": "restructuredtext",
            ".md": "markdown",
        }

        if ext == "" and Path(file_path).name.lower() == "dockerfile":
            return "dockerfile"
        return mapping.get(ext, "text")

    def _chunk_file(self, file_content: str) -> List[Dict[str, Any]]:
        chunk_size = 300
        overlap = 50
        lines = file_content.splitlines()
        total_lines = len(lines)
        chunks: List[Dict[str, Any]] = []

        num_chunks = (
            math.ceil(total_lines / (chunk_size - overlap))
            if chunk_size > overlap
            else 1
        )

        for chunk_idx in range(num_chunks):
            start_idx = max(0, chunk_idx * chunk_size - overlap)
            end_idx = min(start_idx + chunk_size + overlap, total_lines)

            valid_start = start_idx + overlap if chunk_idx != 0 else start_idx
            valid_end = end_idx - overlap if chunk_idx != num_chunks - 1 else end_idx

            chunk_lines = lines[start_idx:end_idx]

            chunks.append(
                {
                    "content": "\n".join(chunk_lines),
                    "line_range": (start_idx + 1, end_idx),
                    "valid_start": valid_start + 1,
                    "valid_end": valid_end,
                }
            )

        return chunks

    def _save_fault_localization_error(
        self,
        sha: str,
        error: Exception,
        tool_name: str = "",
        prompt_name: str = "",
        extra_context: Optional[dict] = None,
    ):
        """
        Save detailed fault localization error info to JSON file.
        """
        base_dir = os.path.join(
            self.config["exception_dir"], "interrupted_fault_localization"
        )
        os.makedirs(base_dir, exist_ok=True)

        fname = f"{self.failed_commit}.json"
        filepath = os.path.join(base_dir, fname)

        error_info = {
            "sha_fail": sha or self.failed_commit,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "tool": tool_name,
            "prompt_name": prompt_name,
            "Agent": "FaultLocalization",
        }

        if extra_context:
            error_info["context"] = extra_context

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=4)

        return error_info

    def _expand_line_range_with_outline(
    self,
    line_range: List[int],
    outline: List[dict],
    fault_level: Optional[str] = None,
    ) -> List[int]:
        """
        Expand/normalize [start, end] using the outline.

        Contract by fault_level:
        - "method": return the entire containing function/method (tightest).
        - "import_block": return the containing import block (tightest).
        - "line": return [start-4, end+4] (clamp start to >= 1).
        - "class": return the dedicated class per rules (as before).
        - anything else / None: return tightest containing node of any kind.
        """
        if not line_range:
            return line_range

        start, end = line_range
        if not (isinstance(start, int) and isinstance(end, int)):
            return line_range

        # Normalize ordering defensively
        if end < start:
            start, end = end, start

        # ---------------------------
        # LINE LEVEL: +/- 4 lines context
        # ---------------------------
        if fault_level == "line":
            s = max(1, start - 4)
            e = end + 4
            # guarantee original is contained
            return [s, e] if s <= start and end <= e else [start, end]

        # If outline missing/unusable, we can't expand to method/class/import boundaries
        if not outline:
            return [start, end]

        # Flatten with parent pointers (without mutating the original outline dicts).
        flat: List[dict] = []

        def visit(node: dict, parent: Optional[dict] = None):
            if not isinstance(node, dict):
                return
            n = dict(node)  # copy to avoid side effects
            n["_parent"] = parent
            children = node.get("children") or []
            n["children"] = children
            flat.append(n)
            for child in children:
                visit(child, n)

        for node in outline:
            visit(node, None)

        def contains(node: dict, s: int, e: int) -> bool:
            a, b = node.get("start"), node.get("end")
            return isinstance(a, int) and isinstance(b, int) and a <= s and e <= b

        candidates = [n for n in flat if contains(n, start, end)]
        if not candidates:
            return [start, end]

        def tightest(nodes: List[dict]) -> dict:
            return min(nodes, key=lambda n: (int(n["end"]) - int(n["start"]), int(n["start"])))

        # ---------------------------
        # METHOD LEVEL: tightest containing function/method
        # ---------------------------
        if fault_level == "method":
            funcs = [c for c in candidates if c.get("kind") in {"func", "method"}]
            if funcs:
                chosen = tightest(funcs)
                expanded = [int(chosen["start"]), int(chosen["end"])]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]
            return [start, end]

        # ---------------------------
        # IMPORT BLOCK LEVEL: tightest containing import block
        # ---------------------------
        if fault_level == "import_block":
            imps = [c for c in candidates if c.get("kind") == "import_block"]
            if imps:
                chosen = tightest(imps)
                expanded = [int(chosen["start"]), int(chosen["end"])]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]
            return [start, end]

        # ---------------------------
        # CLASS LEVEL: dedicated class per rules (your existing logic)
        # ---------------------------
        if fault_level == "class":
            class_candidates = [c for c in candidates if c.get("kind") == "class"]
            if not class_candidates:
                return [start, end]

            owning = tightest(class_candidates)
            cls_start, cls_end = int(owning["start"]), int(owning["end"])

            parent = owning.get("_parent")
            if parent and parent.get("kind") == "class":
                expanded = [cls_start, cls_end]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            direct_nested = [
                ch for ch in (owning.get("children") or [])
                if isinstance(ch, dict)
                and ch.get("kind") == "class"
                and isinstance(ch.get("start"), int)
                and isinstance(ch.get("end"), int)
            ]
            direct_nested.sort(key=lambda n: int(n["start"]))

            if not direct_nested:
                expanded = [cls_start, cls_end]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            first_child_start = int(direct_nested[0]["start"])
            header_end = first_child_start - 1

            if cls_start <= start and end <= header_end:
                expanded = [cls_start, header_end]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            return [start, end]

        # ---------------------------
        # DEFAULT: tightest containing node of any kind
        # ---------------------------
        chosen = tightest(candidates)
        expanded = [int(chosen["start"]), int(chosen["end"])]
        return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]
