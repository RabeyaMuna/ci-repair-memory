import os
import re
import sys
import demjson3
from pathlib import Path
import json
import subprocess
import logging
import shlex
import traceback
import py_compile
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from utilities.chunking_logic import chunk_log_by_tokens
from utilities.load_config import load_config
from utilities.symbols_outline import get_outline_for_file
from utilities.chunking_logic import chunk_lines_with_overlap
from utilities.model_token_limits import get_prompt_token_budget, get_max_output_tokens


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger.setLevel(logging.INFO)

automated_commands_available = [
  {
    "tool": "ruff",
    "purpose": (
      "Unified linter and formatter for Python. "
      "Can replace flake8/pyflakes/pycodestyle, handle many import issues, "
      "and format code (similar to black)."
    ),
    "install_command": "pip install ruff",
    "check_command": "ruff check {{file_or_dir}}",
    "fix_command": "ruff check --fix {{file_or_dir}}",
    "format_command": "ruff format {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "black",
    "purpose": "Code formatter enforcing uniform style and indentation",
    "install_command": "pip install black",
    "check_command": "black --check {{file_or_dir}}",
    "fix_command": "black {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "isort",
    "purpose": "Sort and group imports automatically",
    "install_command": "pip install isort",
    "check_command": "isort --check-only {{file_or_dir}}",
    "fix_command": "isort {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "flake8",
    "purpose": "Linting tool for style and logical errors (PEP8, unused imports, etc.)",
    "install_command": "pip install flake8",
    "check_command": "flake8 {{file_or_dir}}",
    "fix_command": "ruff check --fix {{file_or_dir}}",  # delegate fixes to Ruff
    "file_pattern": "*.py"
  },
  {
    "tool": "autopep8",
    "purpose": "Automatically formats Python code to be PEP8 compliant",
    "install_command": "pip install autopep8",
    "check_command": "autopep8 --diff {{file_or_dir}}",
    "fix_command": "autopep8 --in-place --aggressive {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "yapf",
    "purpose": "Code formatter (alternative to black/autopep8)",
    "install_command": "pip install yapf",
    "check_command": "yapf --diff {{file_or_dir}}",
    "fix_command": "yapf -i {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "pylint",
    "purpose": "Comprehensive linter for detecting code smells and logical issues",
    "install_command": "pip install pylint",
    "check_command": "pylint {{file_or_dir}}",
    "fix_command": "ruff check --fix {{file_or_dir}}",  # fixes delegated to Ruff
    "file_pattern": "*.py"
  },
  {
    "tool": "mypy",
    "purpose": "Static type checker for Python",
    "install_command": "pip install mypy",
    "check_command": "mypy {{file_or_dir}}",
    "fix_command": "",
    "file_pattern": "*.py"
  },
  {
    "tool": "pytest",
    "purpose": "Run unit and integration tests",
    "install_command": "pip install pytest",
    "check_command": "pytest {{file_or_dir}}",
    "fix_command": "",
    "file_pattern": "tests/*.py"
  },
  {
    "tool": "bandit",
    "purpose": "Security linter (checks for hardcoded secrets, dangerous calls, etc.)",
    "install_command": "pip install bandit",
    "check_command": "bandit -r {{file_or_dir}}",
    "fix_command": "",
    "file_pattern": "*.py"
  },
  {
    "tool": "codespell",
    "purpose": "Spell checker for comments and docstrings",
    "install_command": "pip install codespell",
    "check_command": "codespell {{file_or_dir}}",
    "fix_command": "codespell -w {{file_or_dir}}",
    "file_pattern": ["*.py", "*.md", "*.rst"]
  },
  {
    "tool": "docformatter",
    "purpose": "Format docstrings to follow PEP 257 conventions",
    "install_command": "pip install docformatter",
    "check_command": "docformatter --check {{file_or_dir}}",
    "fix_command": "docformatter --in-place {{file_or_dir}}",
    "file_pattern": "*.py"
  }
]


class DiffOutput(BaseModel):
    diff: str


class PatchGeneration:
    """
    PatchGeneration applies a two-phase CI repair pipeline:
    1️ Try automated tool-based fixes (ruff, black, isort, etc.)
    2️ If unavailable or unsuccessful, generate localized patches via LLM
    3️ Produce a validated unified diff and reset repo state
    """

    def __init__(
        self,
        bug_report: Dict,
        repo_path: str,
        task_id: str,
        error_details: Optional[Dict] = None,
        workflow_path: str = "",
        workflow: str = "",
        llm: ChatOpenAI = None,
        model_name: str = None,
        tracker=None,           # utilities.token_tracker.TokenTracker | None
    ):
        self.config = load_config()
        self.bug_report = bug_report
        self.repo_path = repo_path
        self.task_id = task_id
        self.error_details = error_details or {}
        self.sha_fail = bug_report.get("sha_fail")
        self.workflow_path = workflow_path
        self.workflow = workflow
        self.llm = llm
        self.parser = JsonOutputParser()
        self.patch_results: List[Dict[str, Any]] = []
        self.original_content = None
        self.model_name = model_name
        self._tracker = tracker

    # =========================================================
    # --------------------- CORE METHODS ----------------------
    # =========================================================

    def _record_tool_call(
        self,
        cmd,            # list[str] or str
        call_type: str,
        proc,           # subprocess.CompletedProcess
        elapsed: float,
    ) -> None:
        """Push one subprocess result into the shared tracker (no-op if no tracker)."""
        if self._tracker is None:
            return
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self._tracker.record_tool_call(
            agent="PatchGeneration",
            command=cmd_str,
            call_type=call_type,
            success=(proc.returncode == 0),
            return_code=proc.returncode,
            elapsed_sec=elapsed,
        )

    def _call_llm_directly(self, prompt: str) -> Any:
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            match = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
            extracted = match.group(1).strip() if match else response
            
            try:
                data = json.loads(extracted)
            except Exception:
                data = demjson3.decode(extracted)

            return data

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw LLM Output:\n{response}")
            self._save_interrupted_patch_error(
                "_call_llm_directly",
                file_path="(llm)",
                e=e,
                extra={"response_snippet": (response or "")[:2000]},
            )
        return None

    def _extract_outline_block_for_fault(
        self,
        outline: List[Dict[str, Any]],
        file_content: str,
        line_range: List[int]
    ) -> Dict[str, Any]:
        """
        Given an outline and a fault's line_range, find the enclosing block (class/func/const)
        and extract its full code from file_content. Returns a dict with snippet and metadata.
        """
        if not line_range or not isinstance(line_range, (list, tuple)) or len(line_range) != 2:
            return {"original_snippet": "", "block_info": None}

        fault_start, fault_end = line_range
        lines = file_content.splitlines()
        best_match = None
        best_span = float("inf")

        def search_outline(nodes: List[Dict[str, Any]]):
            nonlocal best_match, best_span
            for node in nodes:
                start, end = node["start"], node["end"]
                if start <= fault_start <= end or start <= fault_end <= end:
                    span = end - start
                    if span < best_span:
                        best_match = node
                        best_span = span
                if node.get("children"):
                    search_outline(node["children"])

        search_outline(outline)

        if best_match:
            start, end = best_match["start"], best_match["end"]
            snippet = "\n".join(lines[start - 1:end])
            return {
                "original_snippet": snippet,
                "block_info": {
                    "name": best_match["name"],
                    "kind": best_match["kind"],
                    "start": start,
                    "end": end,
                },
            }

        snippet = "\n".join(lines[fault_start - 1:fault_end])
        return {
            "original_snippet": snippet,
            "block_info": {
                "name": "top_level",
                "kind": "unknown",
                "start": fault_start,
                "end": fault_end,
            },
        }
        
    def _estimate_tokens(self, text: str) -> int:
        """Best-effort token estimator."""
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.encoding_for_model(self.model_name or "gpt-4o-mini")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def _dedupe_preserve_order(self, responses: List[Any], full_path: str) -> Dict[str, List[str]]:
        """
        responses: list of LLM responses; each element may be:
        - {} dict
        - {"installation_commands":[...], "fix_commands":[...]}
        - [{...},{...}] (rare, but supported upstream)
        Returns:
        {"installation_commands": [...], "fix_commands": [...]}
        """
        def normalize_cmd(cmd: str) -> str:
            cmd = (cmd or "").strip()
            if not cmd:
                return ""
            
            cmd = cmd.replace("{{file_or_dir}}", full_path).strip()
            cmd = " ".join(cmd.split())
            return cmd

        results: Dict[str, List[str]] = {"installation_commands": [], "fix_commands": []}
        seen_install = set()
        seen_fix = set()

        def add_install(c: str) -> None:
            c2 = normalize_cmd(c)
            if c2 and c2 not in seen_install:
                seen_install.add(c2)
                results["installation_commands"].append(c2)

        def add_fix(c: str) -> None:
            c2 = normalize_cmd(c)
            if c2 and c2 not in seen_fix:
                seen_fix.add(c2)
                results["fix_commands"].append(c2)

        def absorb_one(res: Any) -> None:
            if not res:
                return
            if isinstance(res, list):
                for item in res:
                    absorb_one(item)
                return
            if not isinstance(res, dict):
                return

            for c in (res.get("installation_commands") or []):
                add_install(c)
            for c in (res.get("fix_commands") or []):
                add_fix(c)

        for r in responses:
            absorb_one(r)

        return results

    def _build_automated_fix_prompt(self, *, full_path: str, faults_payload: Any, pyproject_cfg: str) -> str:
        return f"""
You are a CI Repair Analyst AI.

Your task is to decide whether the CI failure is 100% deterministically fixable
by running existing automated tools only (linters, formatters, static analyzers)
that are part of the CI validation for the failed job.

If there is any doubt, you MUST return an empty JSON object: {{}}.

---

CONTEXT INFORMATION

1) FILE INFO
Full Path: {full_path}

2) FAULT CONTEXT (Detected by Fault Localization)
{json.dumps(faults_payload, indent=2)}

3) CI ERROR LOGS
{json.dumps(self.error_details, indent=2)}

4) FAILED JOB VALIDATION COMMANDS (source of truth)
These are the exact validation tools/commands executed in the failed CI job:
{json.dumps(self.error_details.get("failed_job", {}), indent=2)}

5) AUTOMATED TOOLS REFERENCE
Verified tools and valid commands:
{json.dumps(automated_commands_available, indent=2)}

6) PROJECT CONFIG (pyproject.toml raw content, may be empty)
{pyproject_cfg}

---

OUTPUT FORMAT (STRICT JSON ONLY)

Return EXACTLY one of the following:

A) If automated tooling CAN deterministically fix this CI failure:
{{
"installation_commands": ["..."],
"fix_commands": ["..."]
}}

B) If automated tooling CANNOT fix it, OR if you are not 100% sure, OR if there are no commands to run:
{{}}

Rules:
- Return the MINIMAL set of commands to fix the failure.
- Prefer only relevant tools (highest confidence) rather than chaining multiple tools.
- If Ruff is present in FAILED JOB VALIDATION COMMANDS, ALWAYS prefer Ruff over isort/black/flake8 for Python formatting/import issues.
- Do NOT include commands that are not required for this specific failure.
- Output ONLY valid JSON (no markdown, no backticks, no extra text).
- Do not invent tools.
- ONLY use commands that appear verbatim in FAILED JOB VALIDATION COMMANDS (source of truth).
- If FAILED JOB VALIDATION COMMANDS does not include a tool, you MUST NOT suggest it.
- NEVER install runtime dependencies (e.g. huggingface-hub).
- installation_commands may ONLY install tools listed in AUTOMATED TOOLS REFERENCE.

""".strip()

    # =========================================================
    # ---------------- AUTOMATED TOOL FIX ---------------------
    # =========================================================

    def _load_pyproject(self) -> str:
        """
        Read pyproject.toml as plain text and return its contents.
        No TOML parsing, no tomllib/tomli – just raw text.

        Returns:
            The file content as a string, or "" if the file does not
            exist or cannot be read.
        """
        path = os.path.join(self.repo_path, "pyproject.toml")

        if not os.path.exists(path):
            logger.info("No pyproject.toml found for this repo.")
            return ""

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info("Loaded pyproject.toml successfully.")
                return content
        except Exception as e:
            logger.warning(f"Failed to read pyproject.toml: {e}")
            return ""


    def _try_automated_fix(self, faults: List[Dict[str, Any]], full_path: str) -> bool:
        """Try to fix the file automatically using appropriate tools (ruff, black, isort, etc.)"""
        pyproject_cfg = self._load_pyproject()
        TOKEN_LIMIT = get_prompt_token_budget(self.model_name)

        collected: List[Any] = []  # collect raw LLM outputs here

        base_prompt = self._build_automated_fix_prompt(
            full_path=full_path, faults_payload=faults, pyproject_cfg=pyproject_cfg
        )

        base_tokens = self._estimate_tokens(base_prompt)
        if base_tokens <= TOKEN_LIMIT:
            res = self._call_llm_directly(base_prompt)
            collected.append(res)
        else:
            logger.warning(
                f"Base context alone is {base_tokens} tokens (>= {TOKEN_LIMIT}). "
                "Proceeding with per-fault prompts."
            )

            for i, fault in enumerate(faults):
                single_prompt = self._build_automated_fix_prompt(
                    full_path=full_path, faults_payload=[fault], pyproject_cfg=pyproject_cfg
                )
                single_tokens = self._estimate_tokens(single_prompt)

                if single_tokens <= TOKEN_LIMIT:
                    part_res = self._call_llm_directly(single_prompt)
                    collected.append(part_res)
                    continue

                logger.warning(f"[fault {i}] exceeds {TOKEN_LIMIT} tokens. Proceeding with snippet chunking.")

                fault_meta = dict(fault)
                full_snippet = fault_meta.pop("code_snippet", "")

                snippet_chunks = chunk_log_by_tokens(
                    full_snippet,
                    max_tokens=60000,
                    model=self.model_name,
                )

                for j, snip_chunk in enumerate(snippet_chunks):
                    fault_piece = dict(fault_meta)
                    fault_piece["code_snippet"] = snip_chunk
                    fault_piece["chunk_info"] = {
                        "chunk_index": j,
                        "chunk_count": len(snippet_chunks),
                        "chunk_field": "code_snippet",
                    }

                    piece_prompt = self._build_automated_fix_prompt(
                        full_path=full_path, faults_payload=[fault_piece], pyproject_cfg=pyproject_cfg
                    )
                    piece_res = self._call_llm_directly(piece_prompt)
                    collected.append(piece_res)

        # ---- FINAL: aggregate unique commands (once) ----
        results = self._dedupe_preserve_order(collected, full_path)

        if not results["installation_commands"] and not results["fix_commands"]:
            logger.warning(f"No automated commands returned for automated fix in {full_path}")
            return False
        
        # ---- Execute install commands (with trusted-host fallback) ----
                # ---- Execute install commands (no sys.executable; trusted-host fallback) ----
        for cmd in results["installation_commands"]:
            try:
                logger.info(f"Installing tool: {cmd}")
                tokens = shlex.split(cmd)

                # 1) Try original command as returned (e.g., "pip install docformatter")
                try:
                     # --- IGNORE ---
                    _t0 = time.time()
                    proc = subprocess.run(
                        tokens,
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self._record_tool_call(tokens, "install", proc, time.time() - _t0)
                except FileNotFoundError:
                    # e.g., 'pip' not found -> try python -m pip install ...
                    if len(tokens) >= 3 and tokens[0] == "pip" and tokens[1] == "install":
                        pkgs = tokens[2:]
                        fallback = ["python", "-m", "pip", "install"] + pkgs
                        logger.info(f"'pip' not found; retrying via: {' '.join(fallback)}")
                        _t0 = time.time()
                        proc = subprocess.run(
                            fallback,
                            cwd=self.repo_path,
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        self._record_tool_call(fallback, "install", proc, time.time() - _t0)
                    else:
                        raise

                if proc.returncode == 0:
                    logger.info(f"Tool installation succeeded: {cmd}")
                    continue

                logger.warning(
                    f"Tool installation failed (normal): {cmd}\n"
                    f"Return code: {proc.returncode}\n"
                    f"STDOUT:\n{proc.stdout}\n"
                    f"STDERR:\n{proc.stderr}"
                )

                # 2) If it's a pip install, retry with trusted-host (still NOT sys.executable)
                if len(tokens) >= 3 and tokens[0] == "pip" and tokens[1] == "install":
                    pkgs = tokens[2:]
                    retry_cmd = (
                        ["python", "-m", "pip", "install",
                         "--trusted-host", "pypi.org",
                         "--trusted-host", "files.pythonhosted.org"]
                        + pkgs
                    )
                    logger.info(f"Retrying with trusted-host: {' '.join(retry_cmd)}")

                    _t0 = time.time()
                    proc2 = subprocess.run(
                        retry_cmd,
                        cwd=self.repo_path,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    self._record_tool_call(retry_cmd, "install", proc2, time.time() - _t0)
                    if proc2.returncode == 0:
                        logger.info(f"Tool installation succeeded (trusted-host): {cmd}")
                    else:
                        logger.warning(
                            f"Tool installation failed (trusted-host): {cmd}\n"
                            f"Return code: {proc2.returncode}\n"
                            f"STDOUT:\n{proc2.stdout}\n"
                            f"STDERR:\n{proc2.stderr}"
                        )

            except Exception as e:
                logger.error(f"Tool installation errored ({cmd}): {e}")
                self._save_interrupted_patch_error("_try_automated_fix:install", full_path, e, extra={"cmd": cmd})
                logger.exception("Tool installation errored (%s: %s) cmd=%r", type(e).__name__, e, cmd)



        # ---- Execute fix commands ----
        any_success = False
        for cmd in results["fix_commands"]:
            try:
                logger.info(f"Running automated fix command: {cmd}")
                _t0 = time.time()
                fix_proc = subprocess.run(
                    shlex.split(cmd),
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self._record_tool_call(cmd, "lint_fix", fix_proc, time.time() - _t0)

                if fix_proc.returncode == 0:
                    logger.info(f"Automated fix succeeded: {cmd}")
                    any_success = True
                else:
                    logger.warning(f"Command failed: {cmd}\n{fix_proc.stderr}")
            except Exception as e:
                self._save_interrupted_patch_error("_try_automated_fix:fix", full_path, e, extra={"cmd": cmd})
                logger.error(f"Failed running command {cmd}: {e}")

        # If automated fix succeeded, return True so caller skips LLM fallback
        return any_success


    def _llm_patch_prompt(self, *, full_path: str, faults_payload: Any, outline: str) -> str:
        return f"""
    You are a Senior Software Engineer responsible for repairing code faults by fixing any kind of issues in the code.
    Each fault entry describes a specific issue detected by CI validation tools.

    ---

    ### FILE PATH
    {full_path}

    ### FILE OUTLINE
    {outline}

    ### FAULT LOCALIZATION DATA
    {json.dumps(faults_payload, indent=2)}

    ### Failed Job
    {json.dumps(self.error_details.get("failed_job", {}), indent=2)}

    ---

    ### INSTRUCTIONS
    - For each issue, identify the **exact original snippet** to fix.
    - ALWAYS return full, syntactically valid code blocks so that we can replace the given one with updated one.
    - Do NOT remove or modify unrelated code.
    - Keep indentation and structure consistent with the original code.
    - Do NOT return the entire file content, only the minimally necessary blocks which is given with fixation.
    - IMPORTANT: If there is **no required modification** in the provided `code_snippet`, return an **empty JSON list**: [].

    ---

    ### RESPONSE RULES (CRITICAL)
    - Your **entire** response must be a single JSON value.
    - Do NOT include markdown headings, backticks, or any text before or after the JSON.
    - Do NOT wrap the JSON in ```json``` fences.
    - No explanations outside the JSON object. No markdown, commentary, or code fences.
    - Code format should be preserved as-is; return patched code without any corruption.

    ### RESPONSE FORMAT
    Return one of:
    1) []   (if no changes are needed)
    2) [
        {{
        "original_snippet": "<exact code block that was modified>",
        "fixed_snippet": "<corrected version of that code block>"
        }}
    ]
    """.strip()

    # =========================================================
    # ------------------- LLM PATCH FIX ------------------------
    # =========================================================

    def _generate_llm_patch(
    self,
    faults: List[Dict[str, Any]],
    file_path: str,
    full_path: str,
    original_content: str,
    ) -> bool:
        """
        Generate snippet-level patches using LLM and apply them to the file.
        - Try combined faults prompt if within token limit
        - Else per-fault
        - If a per-fault prompt is too large, chunk that fault's code_snippet and try per chunk
        After collecting all patches for the file, apply them to the file content and write once.
        """
        outline = get_outline_for_file(full_path) or "No outline available."
        token_limit = get_prompt_token_budget(self.model_name)

        collected_patches: List[Dict[str, Any]] = []

        # --------------------------
        # 1) Combined faults prompt
        # --------------------------
        full_prompt = self._llm_patch_prompt(
            full_path=full_path, faults_payload=faults, outline=outline
        )

        try:
            combined_tokens = self._estimate_tokens(full_prompt)

            if combined_tokens <= token_limit:
                res = self._call_llm_directly(full_prompt)
                    
                if res and isinstance(res, list):
                    for item in res:
                        if isinstance(item, dict):
                            collected_patches.append(item)
                else:
                    logger.warning(
                        f"No valid snippet patches returned by LLM for {file_path} (combined)."
                    )
            else:
                logger.warning(
                    f"{file_path}: combined prompt too large ({combined_tokens} > {token_limit}); "
                    "proceeding per-fault."
                )

                # --------------------------
                # 2) Per-fault prompts
                # --------------------------
                for i, fault in enumerate(faults):
                    fault_prompt = self._llm_patch_prompt(
                        full_path=full_path, faults_payload=fault, outline=outline
                    )

                    fault_tokens = self._estimate_tokens(fault_prompt)

                    # If fault prompt fits -> call directly
                    if fault_tokens <= token_limit:
                        fault_res = self._call_llm_directly(fault_prompt)
                        if fault_res and isinstance(fault_res, list):
                            for item in fault_res:
                                if isinstance(item, dict):
                                    collected_patches.append(item)
                        else:
                            logger.warning(f"{file_path}: fault {i} returned no valid patches.")
                        continue

                    # --------------------------
                    # 3) Fault too large -> chunk code_snippet only
                    # --------------------------
                    logger.warning(
                        f"{file_path}: fault {i} prompt too large ({fault_tokens} > {token_limit}); "
                        "chunking code_snippet."
                    )

                    snippet = (fault.get("code_snippet") or "")
                    if not snippet.strip():
                        logger.warning(f"{file_path}: fault {i} has empty code_snippet; skipping.")
                        continue

                    fault_meta = dict(fault)
                    fault_meta.pop("code_snippet", None)

                    # Chunk by lines: 300 lines per chunk, 50 line overlap
                    snippet_chunks = chunk_lines_with_overlap(
                        snippet,
                        lines_per_chunk=200,
                        overlap=50,
                    )

                    chunk_count = len(snippet_chunks)
                    for j, chunk_tuple in enumerate(snippet_chunks):
                        # chunk_lines_with_overlap returns (start_line, end_line, chunk_text)
                        snip_chunk = chunk_tuple[2]

                        fault_piece = dict(fault_meta)
                        fault_piece["code_snippet"] = snip_chunk
                        fault_piece["chunk_info"] = {
                            "chunk_index": j,
                            "chunk_count": chunk_count,
                            "chunk_field": "code_snippet",
                        }

                        chunk_prompt = self._llm_patch_prompt(
                            full_path=full_path, faults_payload=fault_piece, outline=outline
                        )

                        chunk_tokens = self._estimate_tokens(chunk_prompt)
                        if chunk_tokens > token_limit:
                            logger.warning(
                                f"{file_path}: fault {i} chunk {j} still too large "
                                f"({chunk_tokens} > {token_limit}); skipping chunk."
                            )
                            continue

                        chunk_res = self._call_llm_directly(chunk_prompt)
                        if chunk_res and isinstance(chunk_res, list):
                            for item in chunk_res:
                                if isinstance(item, dict):
                                    collected_patches.append(item)
                        else:
                            logger.warning(
                                f"{file_path}: fault {i} chunk {j} returned no valid patches."
                            )

            # --------------------------
            # Apply ALL collected patches at the end
            # --------------------------
            if not collected_patches:
                logger.warning(f"No valid snippet patches returned by LLM for {file_path}")
                return False

            updated_content = original_content
            any_applied = False

            for idx, patch in enumerate(collected_patches, start=1):
                original_snippet = (patch.get("original_snippet") or "")
                fixed_snippet = (patch.get("fixed_snippet") or "")

                if not original_snippet.strip() or not fixed_snippet.strip():
                    logger.warning(f"Patch {idx} missing snippet data, skipping.")
                    continue

                replaced_content = self._replace_snippet_in_code(
                    updated_content, original_snippet, fixed_snippet
                )

                if replaced_content == updated_content:
                    logger.warning(f"Patch {idx}: snippet not found in {file_path}, skipping.")
                    continue

                updated_content = replaced_content
                any_applied = True
                logger.info(f"Patch {idx} applied to {file_path}")

            if not any_applied:
                logger.warning(
                    f"{file_path}: patches were generated but none applied (snippets not found)."
                )
                return False

            # Write once
            if not self._write_updated_file(full_path, updated_content):
                logger.error(f"{file_path}: failed to write updated file.")
                return False

            return True

        except Exception as e:
            logger.error(f"LLM patch generation failed for {file_path}: {e}")
            self._save_interrupted_patch_error("_generate_llm_patch", full_path, e)
            return False

    # =========================================================
    # ---------------------- REPLACEMENT ----------------------
    # =========================================================

    def _replace_snippet_in_code(
        self, code: str, original_snippet: str, fixed_snippet: str
    ) -> str:
        """Replace a snippet of code safely with normalization."""
        if not original_snippet.strip():
            return code

        if original_snippet in code:
            return code.replace(original_snippet, fixed_snippet, 1)

        norm_code = "\n".join(line.rstrip() for line in code.splitlines())
        norm_snippet = "\n".join(line.rstrip() for line in original_snippet.splitlines())
        if norm_snippet in norm_code:
            logger.info("Snippet found via normalized search.")
            return norm_code.replace(norm_snippet, fixed_snippet, 1)

        logger.debug("Snippet not found after normalization.")
        return code

    def _write_updated_file(self, full_path: str, content: str) -> bool:
        """Write updated content to disk with newline normalization."""
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                for line in content.splitlines(keepends=True):
                    f.write(line if line.endswith("\n") else line + "\n")
            logger.info(f"Updated file written successfully: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write updated file {full_path}: {e}")
            return False

    # =========================================================
    # -------------------- PATCH PROCESS ----------------------
    # =========================================================

    def _fix_code(self) -> List[Dict[str, Any]]:
        """Apply automated + LLM fixes sequentially."""
        faults_data = self.bug_report.get("fault_localization_data", [])
        for faults in faults_data:
            file_path = faults["file_path"]
            full_path = faults["full_file_path"]
            faults = faults["faults"]

            if faults == []:
                logger.info(f"No faults listed for {file_path}, skipping.")
                continue

            if not os.path.exists(full_path):
                logger.warning(f"Invalid file path: {full_path}")
                continue

            original = self._read_file(full_path)
            if not original:
                logger.warning(f"Could not read file: {full_path}")
                continue

            # 1) Try automated fix first
            automated_ok = self._try_automated_fix(faults, full_path)

            if not automated_ok:
                logger.info(f"Falling back to LLM patch for {file_path}")
                patched_applied = self._generate_llm_patch(
                    faults, file_path, full_path, original
                )

                # Only if LLM actually changed the file do we consider running Ruff
                if patched_applied:
                    self._safe_format_python(full_path)

            modified_content = self._read_file(full_path)

            if modified_content != original:
                self.patch_results.append(
                    {
                        "file_path": file_path,
                        "full_file_path": full_path,
                        "original_content": original,
                        "fixed_content": modified_content,
                        "fix_method": "automated_or_llm",
                    }
                )

        return self.patch_results
    
    def _ruff_config_error(self, out: str) -> bool:
        s = (out or "").lower()
        return (
            ("failed to parse" in s and "pyproject.toml" in s)
            or "unknown rule selector" in s
            or "toml parse error" in s
        )

    def _run_cmd(self, cmd: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _safe_format_python(self, full_path: str) -> None:
        """Format a Python file respecting project settings when possible; never hard-fail."""
        if Path(full_path).suffix != ".py":
            return

        tools_dir = os.path.join(self.repo_path, ".ci_tools")
        os.makedirs(tools_dir, exist_ok=True)

        def _run_py_module(module: str, args: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
            """Run `python -m module ...` with repo-local PYTHONPATH and cwd=self.repo_path."""
            env = os.environ.copy()
            env["PYTHONPATH"] = tools_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

            return subprocess.run(
                [sys.executable, "-m", module] + args,
                cwd=self.repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

        def _pip_install_repo_local(pkgs: List[str], timeout: int = 300) -> bool:
            """Install packages into repo-local tools_dir using --target."""
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "--disable-pip-version-check",
                "--target",
                tools_dir,
            ] + pkgs

            proc = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode == 0:
                return True

            # optional trusted-host retry (keeps same repo-local target)
            cmd_retry = cmd[:]
            cmd_retry.insert(cmd_retry.index("install") + 1, "--trusted-host")
            cmd_retry.insert(cmd_retry.index("install") + 2, "pypi.org")
            cmd_retry.insert(cmd_retry.index("install") + 3, "--trusted-host")
            cmd_retry.insert(cmd_retry.index("install") + 4, "files.pythonhosted.org")

            proc2 = subprocess.run(
                cmd_retry,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc2.returncode != 0:
                logger.warning(
                    f"Repo-local pip install failed for {pkgs}\n"
                    f"STDERR:\n{proc2.stderr}\nSTDOUT:\n{proc2.stdout}"
                )
                return False

            return True

        # --------------------
        # 1) Try Ruff first
        # --------------------
        try:
            if _pip_install_repo_local(["ruff"], timeout=300):
                _t0 = time.time()
                check = _run_py_module("ruff", ["check", full_path], timeout=300)
                self._record_tool_call(["python", "-m", "ruff", "check", full_path], "lint_check", check, time.time() - _t0)
                combined = (check.stdout or "") + "\n" + (check.stderr or "")

                # Your helper catches parse/config errors (like TC001 / TOML issues)
                if self._ruff_config_error(combined):
                    logger.warning(
                        "Skipping Ruff due to Ruff config/pyproject parse error:\n"
                        f"{check.stderr}"
                    )
                else:
                    # 0) Never run formatters on invalid Python (prevents "corruption" cases)
                    try:
                        py_compile.compile(full_path, doraise=True)
                    except Exception as e:
                        logger.warning(f"Skipping formatting; invalid Python: {full_path}\n{e}")
                        return

                    # 1) OPTIONAL: attempt safe autofixes (won't rename vars; may still return 1)
                    _t0 = time.time()
                    fix = _run_py_module(
                        "ruff", ["check", "--force-exclude", "--fix", full_path], timeout=300
                    )
                    self._record_tool_call(["python", "-m", "ruff", "check", "--fix", full_path], "lint_fix", fix, time.time() - _t0)
                    if fix.returncode != 0 and (fix.stdout or fix.stderr):
                        # Not fatal for formatting; Ruff puts diagnostics in STDOUT.
                        logger.info(
                            f"Ruff check --fix reported remaining issues for {full_path} (rc={fix.returncode}):\n"
                            f"STDOUT:\n{fix.stdout}\nSTDERR:\n{fix.stderr}"
                        )

                    # 2) Format (this is what fixes indentation/line breaks)
                    _t0 = time.time()
                    fmt = _run_py_module(
                        "ruff", ["format", "--force-exclude", full_path], timeout=300
                    )
                    self._record_tool_call(["python", "-m", "ruff", "format", full_path], "format", fmt, time.time() - _t0)
                    if fmt.returncode != 0:
                        logger.warning(
                            f"Ruff format failed for {full_path}:\nSTDOUT:\n{fmt.stdout}\nSTDERR:\n{fmt.stderr}"
                        )
                        return

                    # 3) Imports-only fix (keeps imports grouped at top; doesn't touch variables)
                    _t0 = time.time()
                    imp = _run_py_module(
                        "ruff", ["check", "--select", "I", "--fix", "--force-exclude", full_path], timeout=300
                    )
                    self._record_tool_call(["python", "-m", "ruff", "check", "--select", "I", "--fix", full_path], "import_sort", imp, time.time() - _t0)
                    if imp.returncode != 0 and (imp.stdout or imp.stderr):
                        logger.info(
                            f"Ruff import pass reported issues for {full_path} (rc={imp.returncode}):\n"
                            f"STDOUT:\n{imp.stdout}\nSTDERR:\n{imp.stderr}"
                        )

                    # 4) Format again after import edits (keeps spacing consistent)
                    _t0 = time.time()
                    fmt2 = _run_py_module(
                        "ruff", ["format", "--force-exclude", full_path], timeout=300
                    )
                    self._record_tool_call(["python", "-m", "ruff", "format", full_path], "format", fmt2, time.time() - _t0)
                    if fmt2.returncode != 0:
                        logger.warning(
                            f"Ruff format (second pass) failed for {full_path}:\nSTDOUT:\n{fmt2.stdout}\nSTDERR:\n{fmt2.stderr}"
                        )
                        return

                    print(f"[FORMAT] Ruff format + import ordering completed for {full_path}")
                    return

            else:
                logger.warning("Failed to install Ruff repo-locally; falling back to isort/black.")
        except Exception as e:
            logger.warning(f"Ruff formatting attempt errored; falling back: {e}")

                        # --------------------
        # 2) Fallback: isort + black
        # # --------------------
        # try:
        #     ok = _pip_install_repo_local(["isort", "black"], timeout=300)
        #     if not ok:
        #         logger.warning("Failed to install isort/black repo-locally; skipping formatting.")
        #         return

        #     isort_proc = _run_py_module("isort", [full_path], timeout=300)
        #     if isort_proc.returncode != 0:
        #         logger.warning(f"isort failed for {full_path}:\n{isort_proc.stderr}")

        #     black_proc = _run_py_module("black", [full_path], timeout=300)
        #     if black_proc.returncode != 0:
        #         logger.warning(f"black failed for {full_path}:\n{black_proc.stderr}")
        # except Exception as e:
        #     logger.warning(f"isort/black fallback failed for {full_path}: {e}")

    # =========================================================
    # ---------------------- DIFF CREATION --------------------
    # =========================================================

    def _format_diff(self, patch_results) -> Optional[DiffOutput]:
        if not patch_results:
            raise RuntimeError("No valid files after patching.")

        valid_files = []
        for r in patch_results:
            try:
                rel = os.path.relpath(r["full_file_path"], self.repo_path)
                valid_files.append(rel)
            except Exception as e:
                logger.warning(f"Could not compute relative path: {e}")

        subprocess.run(["git", "add"] + valid_files, cwd=self.repo_path, check=True)
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        diff = proc.stdout
        if not self._is_diff_valid(diff):
            raise RuntimeError("Diff failed validation (git apply --check).")

        logger.info("Diff successfully validated.")
        return DiffOutput(diff=diff)

    def _is_diff_valid(self, diff_text: str) -> bool:
        temp = os.path.join(self.repo_path, "temp_validation.diff")
        with open(temp, "w", encoding="utf-8") as f:
            f.write(diff_text)
        proc = subprocess.run(
            ["git", "apply", "--check", "--3way", temp],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        os.remove(temp)
        return proc.returncode == 0

    # =========================================================
    # ---------------------- UTILITIES ------------------------
    # =========================================================

    def _read_file(self, full_path: str) -> str:
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Cannot read file {full_path}: {e}")
            return ""

    # =========================================================
    # ------------------ GIT OPERATIONS ------------------------
    # =========================================================

    def _reset_to_failed_commit(self):
        logger.info(f"Resetting repo to commit {self.sha_fail}")
        subprocess.run(["git", "reset", "--hard"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "checkout", self.sha_fail], cwd=self.repo_path, check=True)

    def _check_current_commit(self) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == self.sha_fail

    # 2) ADD these methods inside PatchGeneration (e.g., under CORE METHODS)

    def _interrupted_dir(self) -> str:
        base = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/exceptions/interrupted_patches"
        os.makedirs(base, exist_ok=True)
        return base

    def _save_interrupted_patch_error(
        self,
        method: str,
        file_path: str,
        e: BaseException,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a record to:
        .../baselines/exceptions/interrupted_patches/<sha_fail>.jsonl
        """
        try:
            folder = self._interrupted_dir()
            sha = (self.sha_fail or "unknown_sha").strip() or "unknown_sha"
            sha = re.sub(r"[^a-zA-Z0-9._-]+", "_", sha)

            out_path = os.path.join(folder, f"{sha}.jsonl")

            record: Dict[str, Any] = {
                "sha_fail": self.sha_fail,
                "id": self.task_id,
                "method": method,
                "file_path": file_path,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            if extra:
                record["extra"] = extra

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        except Exception as inner:
            logger.exception(
                "Failed to save interrupted_patches record (%s: %s)",
                type(inner).__name__,
                inner,
            )

    # =========================================================
    # ---------------------- MAIN ENTRY -----------------------
    # =========================================================

    def run(self) -> Dict[str, Any]:
        """Main entrypoint for patch generation and diff validation."""
        if not self._check_current_commit():
            self._reset_to_failed_commit()

        patches = self._fix_code()

        try:
            diff_output = self._format_diff(patches)
        except Exception as e:
            logger.error(f"Failed diff creation: {e}")
            diff_output = None

        self._reset_to_failed_commit()

        return {
            "id": self.task_id,
            "sha_fail": self.sha_fail,
            "diff": diff_output.diff if diff_output else "",
        }
