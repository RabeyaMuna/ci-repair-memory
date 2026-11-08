import os
import re
import sys
import demjson3
import json
import subprocess
import logging
import shlex
import traceback
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
from utilities.tracked_llm import TrackedLLM


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
        "purpose": "Unified Python linter and formatter (combines flake8, isort, black)",
        "install_command": "pip install ruff",
        "fix_command": "ruff check --fix {{file_or_dir}}",
        "format_command": "ruff format {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["linting errors", "import sorting, order, grouping", "code formatting"],
    },
    {
        "tool": "black",
        "purpose": "Python code formatter (opinionated, PEP 8 compliant)",
        "install_command": "pip install black",
        "fix_command": "black {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["code formatting", "line length", "indentation"],
    },
    {
        "tool": "docstrfmt",
        "purpose": "RST file formatter (mainly for .rst documentation files, also formats RST in docstrings)",
        "install_command": "pip install docstrfmt",
        "fix_command": "docstrfmt {{file_or_dir}}",
        "file_pattern": ["*.rst", "*.py"],
        "fixes": ["RST heading style", "RST formatting", "RST syntax"],
        "note": "Primarily for *.rst files. For Python docstrings use docformatter instead.",
    },
    {
        "tool": "docformatter",
        "purpose": "Python docstring formatter (PEP 257 compliant, for docstrings inside .py files)",
        "install_command": "pip install docformatter",
        "fix_command": "docformatter --in-place --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["docstring formatting", "PEP 257 compliance", "docstring wrapping"],
        "note": "For Python docstrings. For RST files use docstrfmt instead.",
    },
    {
        "tool": "mdformat",
        "purpose": "Markdown file formatter",
        "install_command": "pip install mdformat",
        "fix_command": "python -m mdformat {{file_or_dir}}",
        "file_pattern": "*.md",
        "fixes": ["markdown formatting", "heading style", "list formatting"],
    },
    {
        "tool": "codespell",
        "purpose": "Spell checker for code and documentation",
        "install_command": "pip install codespell",
        "fix_command": "codespell -w {{file_or_dir}}",
        "file_pattern": ["*.py", "*.md", "*.rst", "*.txt"],
        "fixes": ["spelling errors", "typos"],
    },
    {
        "tool": "taplo",
        "purpose": "TOML file formatter",
        "install_command": "cargo install taplo-cli",
        "fix_command": "taplo fmt {{file_or_dir}}",
        "file_pattern": "*.toml",
        "fixes": ["TOML formatting", "pyproject.toml formatting"],
        "note": "Requires Rust/Cargo to install",
    },
    {
        "tool": "autopep8",
        "purpose": "Python code formatter (less opinionated than black)",
        "install_command": "pip install autopep8",
        "fix_command": "autopep8 --in-place --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["PEP 8 violations", "code formatting"],
    },
    {
        "tool": "autoflake",
        "purpose": "Removes unused imports and variables",
        "install_command": "pip install autoflake",
        "fix_command": "autoflake --in-place --remove-all-unused-imports --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
        "fixes": ["unused imports", "unused variables"],
    },
    {
    "tool": "yamllint",
    "purpose": "YAML linter for syntax, indentation, whitespace, and style violations",
    "install_command": "pip install yamllint",
    "fix_command": None,
    "file_pattern": ["*.yml", "*.yaml"],
    "fixes": [],
    "note": "Detects YAML violations but does not automatically modify files.",
},
{
    "tool": "yamlfmt",
    "purpose": "YAML formatter",
    "install_command": "go install github.com/google/yamlfmt/cmd/yamlfmt@latest",
    "fix_command": "yamlfmt {{file_or_dir}}",
    "file_pattern": ["*.yml", "*.yaml"],
    "fixes": [
        "YAML formatting",
        "indentation",
        "trailing whitespace",
    ],
},
{
    "tool": "prettier",
    "purpose": "YAML, JSON, Markdown, and other file formatter",
    "install_command": "npm install -g prettier",
    "fix_command": "prettier --write {{file_or_dir}}",
    "file_pattern": ["*.yml", "*.yaml", "*.json", "*.md"],
    "fixes": [
        "YAML indentation",
        "trailing whitespace",
        "YAML formatting",
        "YAML style"
    ],
    "note": "Can automatically format GitHub Actions workflow YAML files.",
},
{
    "tool": "yapf",
    "purpose": "Python code formatter",
    "install_command": "pip install yapf",
    "fix_command": "yapf -i {{file_or_dir}}",
    "file_pattern": "*.py",
    "fixes": ["code formatting", "style violations"],
},
{
    "tool": "pre-commit-black-jupyter",
    "purpose": "Update and run the Black Jupyter notebook pre-commit hook",
    "install_command": "pre-commit install",
    "fix_command": (
        "pre-commit autoupdate && "
        "pre-commit clean && "
        "pre-commit run black-jupyter --all-files"
    ),
    "file_pattern": "*.ipynb",
    "fixes": [
        "Jupyter notebook formatting",
        "Black pre-commit hook compatibility",
    ],
},
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
        model_name: str = None
    ):
        self.config = load_config()
        self.bug_report = bug_report
        self.repo_path = repo_path
        self.task_id = task_id
        self.error_details = error_details or {}
        self.sha_fail = bug_report.get("sha_fail")
        self.workflow_path = workflow_path
        self.workflow = workflow
        # Wrap LLM with tracking
        self.llm = TrackedLLM(llm, model_name=model_name, module="patch_generation") if llm else None
        self.parser = JsonOutputParser()
        self.patch_results: List[Dict[str, Any]] = []
        self.original_content = None
        self.model_name = model_name

    # =========================================================
    # --------------------- CORE METHODS ----------------------
    # =========================================================

    def _call_llm_directly(self, prompt: str) -> Any:
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

            # Try to extract JSON from markdown code blocks first
            match = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
            else:
                # Try to find JSON array or object in the response
                # Look for outermost [...] or {...}
                array_match = re.search(r'(\[.*\])', response, re.DOTALL)
                object_match = re.search(r'(\{.*\})', response, re.DOTALL)

                # Use whichever match comes first in the response
                if array_match and object_match:
                    if array_match.start() < object_match.start():
                        extracted = array_match.group(1)
                    else:
                        extracted = object_match.group(1)
                elif array_match:
                    extracted = array_match.group(1)
                elif object_match:
                    extracted = object_match.group(1)
                else:
                    extracted = response

            # Try standard json.loads first
            try:
                data = json.loads(extracted)
                return data
            except json.JSONDecodeError as json_err:
                # If json.loads fails, try demjson3 which is more lenient
                logger.debug(f"json.loads failed: {json_err}, trying demjson3")
                try:
                    data = demjson3.decode(extracted)
                    return data
                except Exception as demjson_err:
                    logger.error(f"Both json.loads and demjson3 failed")
                    logger.debug(f"Extracted content:\n{extracted[:1000]}")
                    raise demjson_err

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
        """Try to fix the file automatically using appropriate tools (ruff, black, isort, etc.)

        DISABLED: Automated fixes often create massive whole-file reformatting diffs
        that don't fix the actual bug. Rely on targeted LLM patches instead.
        """
        return False  # Skip automated fixes to avoid massive formatting diffs

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
        
        # Setup venv for running commands
        venv_dir = self._ensure_venv_and_gitignore()
        if not venv_dir:
            logger.warning("Could not setup venv for automated fixes")
            return False

        # Get venv python/pip
        if sys.platform == "win32":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
            venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")
            venv_pip = os.path.join(venv_dir, "bin", "pip")

        # ---- Execute install commands in venv ----
        for cmd in results["installation_commands"]:
            try:
                logger.info(f"Installing in venv: {cmd}")
                tokens = shlex.split(cmd)

                # Replace 'pip install' with venv pip
                if len(tokens) >= 2 and tokens[0] == "pip" and tokens[1] == "install":
                    pkgs = tokens[2:]
                    install_cmd = [venv_pip, "install"] + pkgs
                else:
                    # Other commands run with venv python
                    install_cmd = [venv_python, "-m"] + tokens

                proc = subprocess.run(
                    install_cmd,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if proc.returncode == 0:
                    logger.info(f"✓ Installed: {cmd}")
                else:
                    logger.warning(f"Install failed: {cmd}\n{proc.stderr}")

            except Exception as e:
                logger.error(f"Install error ({cmd}): {e}")

        # ---- Execute fix commands in venv ----
        # Read original content before fixes
        original_before_fix = self._read_file(full_path)

        for cmd in results["fix_commands"]:
            try:
                logger.info(f"Running fix in venv: {cmd}")
                tokens = shlex.split(cmd)

                # Run command with venv python
                fix_cmd = [venv_python, "-m"] + tokens

                fix_proc = subprocess.run(
                    fix_cmd,
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if fix_proc.returncode == 0:
                    logger.info(f"✓ Fix succeeded: {cmd}")
                else:
                    logger.warning(f"Fix failed: {cmd}\n{fix_proc.stderr}")

            except Exception as e:
                logger.error(f"Fix error ({cmd}): {e}")

        # Check if file actually changed
        modified_after_fix = self._read_file(full_path)
        file_changed = modified_after_fix != original_before_fix

        if not file_changed:
            logger.info(f"Automated fix ran but file unchanged, falling back to LLM")

        return file_changed


    def _llm_git_diff_prompt(self, *, file_path: str, full_path: str, file_content: str, faults_payload: Any, outline: str) -> str:
        """Prompt LLM to generate a git diff patch (MiniSWE-Agent approach)."""
        return f"""
You are a Senior Software Engineer responsible for fixing code faults in CI failures.
Generate a git diff format patch to fix the issues described below.

---

### FILE TO FIX
File: {file_path}
Path: {full_path}

### FILE CONTENT
```
{file_content}
```

### FILE OUTLINE (if applicable)
{outline}

### FAULT LOCALIZATION DATA
{json.dumps(faults_payload, indent=2)}

### FAILED JOB CONTEXT
{json.dumps(self.error_details.get("failed_job", {}), indent=2)}

---

### INSTRUCTIONS
1. Analyze the faults and determine the minimal changes needed to fix them
2. Generate a COMPLETE git diff patch in standard unified diff format
3. The patch MUST be applicable with `git apply`
4. Include context lines (3 lines before/after changes)
5. Preserve exact indentation and line endings
6. If no changes needed, return "NO_CHANGES_NEEDED"

### REQUIRED FORMAT
Your response must be ONLY the git diff patch, starting with:
```
diff --git a/{file_path} b/{file_path}
index <hash>..<hash> <mode>
--- a/{file_path}
+++ b/{file_path}
@@ -<start>,<count> +<start>,<count> @@
 <context lines>
-<removed lines>
+<added lines>
 <context lines>
```

Do NOT include:
- Explanations or comments
- Markdown fences (```diff)
- Multiple file diffs (only fix THIS file)
- Any text before or after the diff

If no changes are needed, respond with exactly: NO_CHANGES_NEEDED
""".strip()

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

    def _apply_git_diff_patch(self, diff_content: str) -> bool:
        """Apply a git diff patch using git apply (MiniSWE-Agent approach)."""
        try:
            # Clean up diff content
            diff = diff_content.strip()

            # Check if no changes needed
            if diff == "NO_CHANGES_NEEDED" or not diff:
                logger.info("LLM indicated no changes needed")
                return False

            # Remove markdown fences if present
            if diff.startswith("```"):
                lines = diff.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                diff = "\n".join(lines)

            # Validate it looks like a git diff
            if not diff.startswith("diff --git"):
                logger.warning(f"Response doesn't look like a git diff: {diff[:100]}")
                return False

            # Write diff to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(diff)
                patch_file = f.name

            try:
                # Try to apply the patch
                result = subprocess.run(
                    ["git", "apply", "--check", patch_file],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    logger.warning(f"Patch validation failed: {result.stderr}")
                    return False

                # Apply the patch
                result = subprocess.run(
                    ["git", "apply", patch_file],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True
                )

                logger.info("✓ Git diff patch applied successfully")
                return True

            finally:
                # Clean up temp file
                try:
                    os.unlink(patch_file)
                except:
                    pass

        except Exception as e:
            logger.error(f"Error applying git diff patch: {e}")
            return False

    # =========================================================
    # ------------------- LLM PATCH FIX ------------------------
    # =========================================================

    def _generate_git_diff_patch(
        self,
        faults: List[Dict[str, Any]],
        file_path: str,
        full_path: str,
        original_content: str,
    ) -> bool:
        """
        Generate and apply patch using git diff format (MiniSWE-Agent approach).
        Works for ALL file types including notebooks, YAML, JSON, requirements.txt, etc.
        """
        outline = get_outline_for_file(full_path) or "No outline available."

        # Build prompt for git diff
        prompt = self._llm_git_diff_prompt(
            file_path=file_path,
            full_path=full_path,
            file_content=original_content,
            faults_payload=faults,
            outline=outline
        )

        try:
            # Call LLM to generate diff
            response = self.llm.invoke(prompt).content.strip()

            logger.info(f"Git diff approach for {file_path}")

            # Apply the diff
            if self._apply_git_diff_patch(response):
                logger.info(f"✓ Git diff patch applied to {file_path}")
                return True
            else:
                logger.warning(f"Git diff patch failed for {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error in git diff patch generation for {file_path}: {e}")
            return False

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
        """Replace a snippet of code safely with normalization and fuzzy matching."""
        if not original_snippet.strip():
            return code

        # 1. Try exact match
        if original_snippet in code:
            return code.replace(original_snippet, fixed_snippet, 1)

        # 2. Try normalized match (strip trailing whitespace)
        norm_code = "\n".join(line.rstrip() for line in code.splitlines())
        norm_snippet = "\n".join(line.rstrip() for line in original_snippet.splitlines())
        if norm_snippet in norm_code:
            logger.info("Snippet found via normalized search.")
            return norm_code.replace(norm_snippet, fixed_snippet, 1)

        # 3. Fallback: Try fuzzy matching
        try:
            from difflib import SequenceMatcher

            snippet_lines = original_snippet.splitlines()
            code_lines = code.splitlines()

            if not snippet_lines:
                return code

            best_ratio = 0.0
            best_start = -1
            snippet_len = len(snippet_lines)

            # Sliding window to find best match
            for i in range(len(code_lines) - snippet_len + 1):
                window = code_lines[i:i + snippet_len]
                matcher = SequenceMatcher(None, snippet_lines, window)
                ratio = matcher.ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i

            # Apply if similarity is high enough (85%+)
            if best_ratio >= 0.85 and best_start >= 0:
                logger.info(f"Snippet found via fuzzy match (similarity: {best_ratio:.2%})")

                # Replace the matched section
                fixed_lines = fixed_snippet.splitlines()
                new_lines = (
                    code_lines[:best_start] +
                    fixed_lines +
                    code_lines[best_start + snippet_len:]
                )
                return "\n".join(new_lines)

            logger.debug(f"Fuzzy match failed (best similarity: {best_ratio:.2%})")

        except Exception as e:
            logger.debug(f"Fuzzy matching error: {e}")

        return code

    def _write_notebook_with_patches(self, notebook_path: str, patched_code: str) -> bool:
        """Write patched code back to notebook cells."""
        try:
            # Load original notebook
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            # Parse patched code into cells
            # Format: # Cell N\ncode\n\n# Cell N+1\ncode...
            cell_blocks = []
            current_cell_code = []

            for line in patched_code.split("\n"):
                if line.startswith("# Cell "):
                    # Start of new cell
                    if current_cell_code:
                        cell_blocks.append("\n".join(current_cell_code).strip())
                        current_cell_code = []
                elif line.strip() or current_cell_code:  # Skip leading empty lines
                    current_cell_code.append(line)

            # Don't forget the last cell
            if current_cell_code:
                cell_blocks.append("\n".join(current_cell_code).strip())

            # Update notebook cells with patched code
            code_cell_idx = 0
            cells = notebook.get("cells", [])

            for cell in cells:
                if cell.get("cell_type") == "code":
                    if code_cell_idx < len(cell_blocks):
                        # Convert patched code to notebook source format
                        patched_source = cell_blocks[code_cell_idx]

                        # Split into lines with preserved newlines (notebook format)
                        if patched_source:
                            source_lines = [line + "\n" for line in patched_source.split("\n")]
                            # Remove \n from last line
                            if source_lines:
                                source_lines[-1] = source_lines[-1].rstrip("\n")
                            cell["source"] = source_lines
                        else:
                            cell["source"] = []

                        code_cell_idx += 1

            # Write updated notebook
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)

            logger.info(f"Updated notebook written successfully: {notebook_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write notebook {notebook_path}: {e}")
            return False

    def _write_updated_file(self, full_path: str, content: str) -> bool:
        """Write updated content to disk with newline normalization."""
        try:
            # Special handling for Jupyter notebooks
            if full_path.endswith('.ipynb'):
                return self._write_notebook_with_patches(full_path, content)

            # Regular file writing
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
                # 2) Try git diff approach (works for all file types)
                logger.info(f"Trying git diff approach for {file_path}")
                patched_applied = self._generate_git_diff_patch(
                    faults, file_path, full_path, original
                )

                # 3) Fallback to snippet-based if git diff fails
                if not patched_applied:
                    logger.info(f"Git diff failed, falling back to snippet-based patch for {file_path}")
                    patched_applied = self._generate_llm_patch(
                        faults, file_path, full_path, original
                    )

                # Skip formatting to avoid massive whole-file diffs
                # The LLM should generate targeted fixes only
                # if patched_applied and full_path.endswith('.py'):
                #     self._simple_format_python(full_path)

            modified_content = self._read_file(full_path)

            # Check if file was actually modified
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
                logger.info(f"✓ Successfully modified: {file_path}")
            else:
                # File unchanged - revert any partial changes to keep repo clean
                logger.warning(f"File unchanged, reverting any partial changes: {file_path}")
                try:
                    subprocess.run(
                        ["git", "checkout", "HEAD", "--", file_path],
                        cwd=self.repo_path,
                        capture_output=True,
                        timeout=30
                    )
                except Exception as e:
                    logger.warning(f"Could not revert {file_path}: {e}")

        return self.patch_results
    

    def _run_cmd(self, cmd: List[str], timeout: int = 300) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _ensure_venv_and_gitignore(self) -> str:
        """
        Create/use virtual environment: venv-{repo_name}
        No pyproject.toml checks needed - isolated venv handles everything.
        Returns path to venv directory.
        """
        # Use format: venv-{repo_name}
        repo_name = os.path.basename(self.repo_path)
        venv_name = f"venv-{repo_name}"
        venv_dir = os.path.join(self.repo_path, venv_name)

        # Reuse if exists, create if not
        if os.path.exists(venv_dir):
            logger.info(f"Reusing existing venv: {venv_name}")
        else:
            logger.info(f"Creating venv: {venv_name}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", venv_dir],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True
                )
                logger.info(f"✓ Created venv: {venv_name}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to create venv: {e.stderr}")
                return None

        # Ensure venv pattern is in .gitignore (won't be staged/committed)
        gitignore_path = os.path.join(self.repo_path, ".gitignore")

        # Read existing .gitignore
        existing_content = ""
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                existing_content = f.read()

        # Add venv-*/ pattern if not already there
        if "venv-" not in existing_content:
            with open(gitignore_path, 'a') as f:
                f.write(f"\n# CI venv (not committed)\nvenv-*/\n")
            logger.info(f"Added venv-*/ to .gitignore")

        return venv_dir

    def _simple_format_python(self, full_path: str) -> None:
        """Simple formatting: pip install ruff, then ruff format. Ignore if not applicable."""
        if not full_path.endswith('.py'):
            return

        venv_dir = self._ensure_venv_and_gitignore()
        if not venv_dir:
            return

        # Get venv paths
        if sys.platform == "win32":
            venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_pip = os.path.join(venv_dir, "bin", "pip")
            venv_python = os.path.join(venv_dir, "bin", "python")

        try:
            # Install ruff (ignore if fails)
            subprocess.run(
                [venv_pip, "install", "-q", "ruff"],
                cwd=self.repo_path,
                capture_output=True,
                timeout=300
            )

            # Run ruff format (ignore if fails)
            subprocess.run(
                [venv_python, "-m", "ruff", "format", full_path],
                cwd=self.repo_path,
                capture_output=True,
                timeout=300
            )
        except Exception:
            pass  # Ignore all errors

    # =========================================================
    # ---------------------- DIFF CREATION --------------------
    # =========================================================

    def _format_diff(self, patch_results) -> Optional[DiffOutput]:
        if not patch_results:
            raise RuntimeError("No valid files after patching.")

        # Files/directories to EXCLUDE from patches (should not be committed)
        EXCLUDED_PATTERNS = [
            '.github/workflows/',  # CI workflow files
            '.github/actions/',    # GitHub Actions
            '__pycache__/',        # Python cache
            '.pytest_cache/',      # Pytest cache
            '.ruff_cache/',        # Ruff cache
            '.mypy_cache/',        # Mypy cache
            'venv/',               # Virtual environments
            '.venv/',
            'env/',
            '.env/',
            'virtualenv/',
            'build/',              # Build artifacts
            'dist/',               # Distribution artifacts
            '.tox/',               # Tox environments
            'node_modules/',       # Node modules
            '.git/',               # Git directory
            '.DS_Store',           # macOS files
        ]

        valid_files = []
        for r in patch_results:
            try:
                rel = os.path.relpath(r["full_file_path"], self.repo_path)

                # Check if file matches any excluded pattern
                should_exclude = False
                for pattern in EXCLUDED_PATTERNS:
                    if pattern in rel or rel.endswith(pattern.rstrip('/')):
                        logger.warning(f"⚠️  EXCLUDING file from patch: {rel} (matches {pattern})")
                        should_exclude = True
                        break

                if not should_exclude:
                    valid_files.append(rel)

            except Exception as e:
                logger.warning(f"Could not compute relative path: {e}")

        if not valid_files:
            logger.error("All files were excluded! No valid files to patch.")
            raise RuntimeError("All files excluded from patch.")

        # Use -f to force adding even if files are in .gitignore (repo may ignore its own source)
        subprocess.run(["git", "add", "-f"] + valid_files, cwd=self.repo_path, check=True)
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        diff = proc.stdout

        # Debug: log diff info before validation
        logger.info(f"Generated diff: {len(diff)} chars, {diff.count('diff --git')} files")

        if not self._is_diff_valid(diff):
            logger.error(f"Diff validation failed. First 1000 chars of diff:\n{diff[:1000]}")
            raise RuntimeError("Diff failed validation (git apply --check).")

        logger.info("Diff successfully validated.")
        return DiffOutput(diff=diff)

    def _is_diff_valid(self, diff_text: str) -> bool:
        # Use absolute path for temp file
        temp_abs = os.path.abspath(os.path.join(self.repo_path, "temp_validation.diff"))
        with open(temp_abs, "w", encoding="utf-8") as f:
            f.write(diff_text)

        # Git runs in repo_path, so use just the filename
        temp_name = "temp_validation.diff"
        proc = subprocess.run(
            ["git", "apply", "--check", "--3way", temp_name],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            logger.warning(f"git apply --check failed: {proc.stderr[:500]}")

        os.remove(temp_abs)
        return proc.returncode == 0

    # =========================================================
    # ---------------------- UTILITIES ------------------------
    # =========================================================

    def _parse_notebook_to_code(self, notebook_path: str) -> str:
        """Parse Jupyter notebook and extract code cells as Python code."""
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = json.load(f)

            code_lines = []
            cells = notebook.get("cells", [])

            for idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])

                    # Handle both list and string formats
                    if isinstance(source, list):
                        cell_code = "".join(source)
                    else:
                        cell_code = source

                    if cell_code.strip():
                        code_lines.append(f"# Cell {idx + 1}")
                        code_lines.append(cell_code.rstrip())
                        code_lines.append("")

            return "\n".join(code_lines)
        except Exception as e:
            logger.error(f"Failed to parse notebook {notebook_path}: {e}")
            return ""

    def _read_file(self, full_path: str) -> str:
        try:
            # Special handling for Jupyter notebooks
            if full_path.endswith('.ipynb'):
                return self._parse_notebook_to_code(full_path)

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
