import argparse
import os
import json
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_LOG_PATHS = [
    REPO_ROOT / "baselines" / "results" / "deepseek-coder_bm25" / "log_details.json",
    REPO_ROOT / "baselines" / "results" / "deepseek-coder_llm" / "log_details.json",
    REPO_ROOT / "baselines" / "results" / "gpt-5-mini_bm25" / "log_details.json",
]
DEFAULT_OUTPUT_JSONL = REPO_ROOT / "ci_failure_error_types_deepseek.jsonl"

# Use whatever DeepSeek chat model you want
MODEL_NAME = "deepseek-chat"   # change if you use another DeepSeek model


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a CI failure classification assistant.

You will receive ONE JSON object describing a CI failure for a specific commit (sha_fail).
Your job is to decide one or more error categories and subcategories for that failure.

You MUST output a single JSON object with this exact structure:

{
  "error_category": [...],
  "error_subcategory": [...]
}

CONSTRAINTS ON OUTPUT
- "error_category" MUST be an array of strings.
- "error_subcategory" MUST be an array of strings.
- The two arrays MUST have the SAME LENGTH.
  - For each index i:
      error_category[i]  ↔  error_subcategory[i]
    together describe ONE specific reason for failure.
- Each (category, subcategory) pair MUST be unique (no duplicates).
- If you truly cannot infer any reason, return:
  {
    "error_category": [],
    "error_subcategory": []
  }
- DO NOT include any extra keys, text, or comments. Output ONLY the JSON object.

ALLOWED ERROR CATEGORIES (USE EXACTLY THESE STRINGS)

1. "Code Formatting"
2. "Code Linting"
3. "Configuration Error"
4. "Environment Error"
5. "Syntax Error"
6. "Dependency Issues"
7. "Package Installation Error"
8. "Runtime Error"
9. "Test Failure"
10. "Assertion Error"
11. "Documentation or Docstring Error"
12. "Type Checking Error"

You may use the same category multiple times with different subcategories if there are multiple distinct reasons within that category.

CATEGORY GUIDELINES & EXAMPLES

1. Code Formatting
   - Problems related to automatic code formatting or style formatting rules.
   - Typical evidence:
     - Tools like black, ruff format, clang-format, Prettier in "check" mode failing.
     - Ruff I001 (“import block is un-sorted or un-formatted”).
     - Indentation-only / whitespace-only style problems enforced by a formatter.
   - Example subcategories:
     - "Ruff import block formatting error"
     - "Code formatter style violation"
     - "Indentation formatting mismatch"

2. Code Linting
   - Static analysis / lint failures where code is syntactically valid, but rules are violated.
   - Tools: ruff (lint mode), flake8, pylint, ESLint, golangci-lint, mypy as linter, etc.
   - Examples:
     - Unused imports or variables.
     - Naming convention violations.
     - Type-checking issues when used as lint.
   - Example subcategories:
     - "Ruff lint rule violation"
     - "Pylint unused import error"
     - "ESLint rule violation"

3. Configuration Error
   - Misconfigured CI, build, or test configuration.
   - Examples:
     - Invalid GitHub Actions YAML / GitLab CI / Jenkinsfile.
     - Wrong test path or pattern in the config.
     - Misconfigured build tool options (CMake flags, Gradle/Maven profile).
     - Misconfigured workflow steps, wrong working directory, wrong shell.
   - Example subcategories:
     - "Incorrect GitHub Actions workflow path"
     - "Invalid test discovery pattern"
     - "Misconfigured build command in CI"

4. Environment Error
   - Issues with the execution environment (OS, hardware, system tools, permissions).
   - Examples:
     - Missing system binary (e.g. “bash: gcc: command not found”).
     - Permission denied on file or directory.
     - Out-of-memory or disk full errors.
     - Missing or invalid environment variables required at runtime.
   - Example subcategories:
     - "Missing system binary in CI image"
     - "Insufficient permissions on file"
     - "Missing required environment variable"

5. Syntax Error
   - Language syntax errors in source code or configuration files.
   - Examples:
     - Python SyntaxError, JavaScript syntax error.
     - YAML indentation or structure errors that are clearly syntax related.
   - Example subcategories:
     - "Python syntax error in module"
     - "YAML syntax error in workflow"
     - "Invalid JSON syntax in config"

6. Dependency Issues
   - Problems with dependencies already installed or resolved, but failing at resolution/import time.
   - Examples:
     - Import/module not found at runtime ("ModuleNotFoundError", "No module named X").
     - Version conflicts ("could not resolve dependency", incompatible library versions).
     - Missing shared library (.so / .dll) at runtime.
   - Example subcategories:
     - "Missing Python module at runtime"
     - "Library version conflict"
     - "Missing shared library dependency"

7. Package Installation Error
   - Failures during installation of packages (before code/tests run).
   - Examples:
     - pip install, conda install, npm install, yarn, Maven/Gradle download failures.
     - Build failures while compiling a dependency during install.
     - Network errors during package download.
   - Example subcategories:
     - "pip install command failed"
     - "Dependency wheel build error"
     - "Network failure during package install"
   - RULE OF THUMB:
     - If the error happens during an explicit install step (e.g. `pip install`, `npm install`,
       or installing dependencies for tox/poetry/conda environments), prefer
       "Package Installation Error" over "Dependency Issues".

8. Runtime Error
   - Non-assertion exceptions or crashes while running the code or tests.
   - Examples:
     - Null/None/attribute error, index error, type error at runtime.
     - Segmentation faults, unhandled exceptions (excluding pure assertion failures).
     - Timeouts that indicate runtime problems (e.g. infinite loop).
   - Example subcategories:
     - "Unhandled exception during test run"
     - "Segmentation fault in application"
     - "Timeout due to long-running process"

9. Test Failure
9. Test Failure
   - Tests executed but did not pass for any reason.
   - This category MUST be used whenever a test command fails
     (e.g. `pytest`, `tox -e test`, `npm test`, `jest`, `mvn test`, `gradle test`,
      `go test`, `ctest`, etc.), regardless of the underlying cause
     (assertion, runtime error, dependency failure during tests, timeout, etc.).
   - Examples:
     - Test suite exits with non-zero status.
     - Test setup/teardown failures.
     - Failing unit, integration, or end-to-end tests.
   - Example subcategories:
     - "Pytest test suite failed"
     - "Unit tests did not pass"
     - "Integration test job failure"
     - "End-to-end test timeout"
   - Tests executed but did not pass, for reasons other than explicit assertion errors
     (or when the cause is only known as “test failure”).
   - Examples:
     - Test suite reports failing tests without clear assertion message.
     - Test setup/teardown failures.
     - Flaky test timeouts.
   - Example subcategories:
     - "General pytest test failure"
     - "Test setup fixture failure"
     - "End-to-end test timeout"
   - NOTE: If a specific AssertionError is clearly shown, prefer "Assertion Error"
     for that reason. "Test Failure" can be used for more general or non-assert failures. If Test failed, Use it and Sub category use the specific reason like Assertionerror, or Coverage issues

10. Assertion Error
    - Failures caused explicitly by assertion conditions not being met.
    - Usually messages like "AssertionError", "Expected X but got Y".
    - Typically inside unit tests, but can also be application assertions.
    - Example subcategories:
      - "Pytest assertion failure"
      - "Unit test expectation mismatch"
      - "Assertion failed in integration test"

11. Documentation or Docstring Error
    - Failures due to documentation checks, docstrings, or API docs not matching expectations.
    - Examples:
      - Docstring style checker failures.
      - Sphinx or MkDocs build errors due to missing or invalid docs.
      - Linting of docstrings only.
    - Example subcategories:
      - "Docstring style rule violation"
      - "Sphinx documentation build error"
      - "Missing required parameter documentation"

12. Type Checking Error
    - Failures caused by static type checkers or type systems, distinct from general linting.
    - Tools: mypy, pyright, TypeScript compiler (`tsc`), mypyc, or type-checking modes of other tools.
    - Examples:
      - Incompatible types in function calls (`Argument 1 has incompatible type`).
      - Missing return type annotations when required by strict type mode.
      - Type mismatch between variable declarations and assigned values.
    - Example subcategories:
      - "Mypy static type mismatch error"
      - "Pyright incompatible argument type"
      - "TypeScript type checking failure"


INPUT FORMAT

You will receive ONE JSON object with this structure:

{
  "sha_fail": "....",
  "error_context": [
    "... natural language description of the failure ...",
    "..."
  ],
  "relevant_files": [
    {
      "file": "...",
      "line_number": 123 or null,
      "reason": "..."
    },
    ...
  ],
  "error_types": [
    {
      "category": "...",
      "subcategory": "...",
      "evidence": "..."
    },
    ...
  ],
  "failed_job": [
    {
      "job": "...",
      "step": "...",
      "command": "..."
    },
    ...
  ]
}

NOTES:
- The "error_types" list may contain pre-existing labels like "Dependency Error"
  or "Missing dependency or installation failure". Treat them as HINTS only:
  you MUST map to one of the allowed categories listed above.
- Use "Package Installation Error" when the problem clearly happens during installing
  dependencies (pip install, tox environment setup, etc.).

YOUR TASK

1. Carefully read:
   - The high-level explanation in "error_context".
   - The files listed in "relevant_files" and why they are relevant.
   - The evidence in "error_types".
   - The job/step/command in "failed_job".

2. Identify all distinct reasons that caused or directly contributed to this CI failure.

3. For each reason:
   - Choose exactly one allowed error_category.
   - Create a short, specific error_subcategory (3–8 words),
     e.g.:
       - "Tox dependency installation failure"
       - "Ruff import block formatting error"
       - "Pytest assertion failure"

4. Avoid overlapping or redundant subcategories. Prefer specific and concise wording.

5. Return ONLY the JSON object:

{
  "error_category": [...],
  "error_subcategory": [...]
}

## Output Rules:
DO NOT wrap the JSON in backticks or any kind of markdown code fence.

DO NOT return any markdown formatting at all.

Output ONLY the raw JSON object, nothing before or after it.
"""


# ============================================================
# LOAD LOG FILES AND INDEX BY sha_fail
# ============================================================

def load_error_entries_index(path: Path):
    index = {}

    if not path.exists():
        print(f"[WARN] Log file not found: {path}")
        return index

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[WARN] Expected a list in {path}, got {type(data)}")
        return index

    for entry in data:
        if not isinstance(entry, dict):
            continue
        sha = entry.get("sha_fail")
        if not sha:
            continue
        sha = str(sha)
        index.setdefault(sha, []).append(entry)

    print(f"[INFO] Loaded {len(index)} sha_fail entries from {path.name}")
    return index


# ============================================================
# MERGE MULTIPLE ENTRIES FOR SAME sha_fail
# ============================================================

def merge_error_entries(entries):
    if not entries:
        return None

    sha_fail = str(entries[0].get("sha_fail", ""))

    error_context = []
    relevant_files = []
    error_types = []
    failed_job = []

    for e in entries:
        if isinstance(e.get("error_context"), list):
            error_context.extend(e["error_context"])
        if isinstance(e.get("relevant_files"), list):
            relevant_files.extend(e["relevant_files"])
        if isinstance(e.get("error_types"), list):
            error_types.extend(e["error_types"])
        if isinstance(e.get("failed_job"), list):
            failed_job.extend(e["failed_job"])

    return {
        "sha_fail": sha_fail,
        "error_context": error_context,
        "error_types": error_types,
        "failed_job": failed_job,
    }


# ============================================================
# CALL DEEPSEEK
# ============================================================

def classify_failure_with_llm(
    error_entry: dict,
    client: OpenAI,
    model_name: str,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(error_entry, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        print("[WARN] Could not parse model response as JSON; returning empty arrays.")
        print("Raw response:", raw)
        return {"error_category": [], "error_subcategory": []}

    cats = parsed.get("error_category", [])
    subs = parsed.get("error_subcategory", [])

    if not isinstance(cats, list) or not isinstance(subs, list):
        print("[WARN] Model response did not return lists; returning empty arrays.")
        return {"error_category": [], "error_subcategory": []}

    cleaned_cats, cleaned_subs, seen = [], [], set()
    for c, s in zip(cats, subs):
        if not isinstance(c, str) or not isinstance(s, str):
            continue
        pair = (c.strip(), s.strip())
        if pair in seen:
            continue
        seen.add(pair)
        cleaned_cats.append(pair[0])
        cleaned_subs.append(pair[1])

    return {
        "error_category": cleaned_cats,
        "error_subcategory": cleaned_subs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to lca_dataset.parquet (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--log-path",
        dest="log_paths",
        action="append",
        type=Path,
        help=(
            "Path to a log_details.json file. Repeat for multiple files. "
            "Defaults to the standard baseline result logs."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable name holding the DeepSeek API key",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.deepseek.com",
        help="Base URL for the OpenAI-compatible API",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Model name to call (default: {MODEL_NAME})",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    log_paths = args.log_paths or DEFAULT_LOG_PATHS
    output_jsonl = args.output_jsonl
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} environment variable is not set.")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    df = pd.read_parquet(dataset_path)

    if "id" not in df.columns:
        raise ValueError("Dataset must have an 'id' column.")
    if "sha_fail" not in df.columns:
        raise ValueError("Dataset must have a 'sha_fail' column.")

    combined_index = {}
    for log_path in log_paths:
        idx = load_error_entries_index(log_path)
        for sha_fail, entries in idx.items():
            combined_index.setdefault(sha_fail, []).extend(entries)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for _, row in df.iterrows():
            row_id = str(row["id"])
            sha_fail = str(row["sha_fail"])

            entries = combined_index.get(sha_fail, [])
            if not entries:
                result = {"error_category": [], "error_subcategory": []}
            else:
                merged_entry = merge_error_entries(entries)
                if merged_entry is None:
                    result = {"error_category": [], "error_subcategory": []}
                else:
                    result = classify_failure_with_llm(
                        merged_entry,
                        client=client,
                        model_name=args.model,
                    )

            record = {
                "id": row_id,
                "sha_fail": sha_fail,
                "error_category": result["error_category"],
                "error_subcategory": result["error_subcategory"],
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Done. Wrote results to {output_jsonl}")


if __name__ == "__main__":
    main()
