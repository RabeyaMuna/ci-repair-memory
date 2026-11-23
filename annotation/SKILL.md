---
name: "ci-failure-annotation"
description: "Annotate a CI failure instance with correct failure types and sub-types by checking the ground-truth diff against the CI workflow and auditing the model-generated classification."
---

# CI Failure Annotation

Each instance comes with a CI workflow YAML, the failed jobs/steps, a ground-truth diff (the fix that made CI pass), and a system-generated (model) classification with evidence. Work out the correct Failure Type(s) and Sub-type(s), and judge each part of the model classification.

## Evidence priority

1. Ground-truth diff (strongest). Every changed line fixes something that was broken. For each hunk, ask what would have failed without the change.
2. CI workflow YAML. Defines which tools run and their scope (paths, excludes, flags, matrix). A failure type is only possible if a tool in the workflow can produce it.
3. Model-generated evidence (weak). A hint only, never proof. It often over-labels (Type Checking with no mypy, Configuration Error for application code) and misses Assertion Error.
4. Failed job/step names (unreliable). Composite steps hide which tool failed, and labels can mislead. Never decide from the step name alone.

## No sequence assumption

Developers push repeatedly, and each push can fail at a different verification. The diff can hold fixes for several tools in any order.

- Do not drop a type because its step runs after the failing step or was skipped.
- Include a type when the diff has a concrete fix of that kind AND a workflow tool would flag it.
- Exclude a type when no workflow tool checks for it.

## Procedure

1. Map the workflow. List each verification tool, its scope, path filters and excludes (e.g. `grep -v '/tests/'`, `extend-exclude`), the Python/OS matrix, and install steps.
2. Classify each diff hunk: what was wrong, which workflow tool catches it (none means the hunk is incidental), whether the file is in that tool's scope, then the taxonomy type and a sub-type.
3. Merge into unique failure types and unique sub-types.
4. Audit the model classification: mark each type CORRECT or WRONG with a one-line reason, and list missed types as MISSING.

## Error taxonomy

| Failure Type | Definition | Typical tools / signals |
|---|---|---|
| Code Formatting | A formatter would rewrite the code: whitespace, blank lines, wrapping, quotes, import order | black, ruff format, isort, yapf, autopep8, prettier |
| Linting | Static rule violation: unused import or variable, bare except, `== None`, E402, complexity | ruff check, flake8, pylint, pyflakes, eslint |
| Syntax Error | Code cannot be parsed: IndentationError, invalid syntax | Python import, ruff check E999, compile step |
| Type Checking | Type annotation or type mismatch flagged by a type checker | mypy, pyright, pytype (must be in the workflow) |
| Runtime Error | Exception during execution: TypeError, AttributeError, KeyError, ValueError, ImportError, NameError, ValidationError | pytest errors, app or CLI startup |
| Test Failure | A test fails for any reason | pytest, unittest, nose, tox |
| Assertion Error | A test fails because an assert did not hold (wrong value or output) | Always paired with Test Failure |
| Dependency Issues | Version conflict, incompatible release, missing package, pin or bump | pyproject deps, requirements, lock files |
| Package Install Error | The install or build step itself fails: resolver error, wheel build, missing system lib | pip, poetry, uv install steps |
| Configuration Error | Misconfigured CI YAML or tool config | workflow files, pyproject `[tool.*]`, setup.cfg, tox.ini, pre-commit |
| Environment Error | Runner, OS, Python version, secrets, env vars, network access, missing system binaries | matrix-specific failures, missing env vars, external URLs |
| Doc/Docstring | Docstring style or format, or documentation files failing doc tooling | docformatter, pydocstyle, ruff D rules, Sphinx, RST/MD linters |

## Hard rules

1. ruff check is Linting. ruff format, black and isort are Code Formatting. Never swap them.
2. Type Checking requires a type checker in the workflow.
3. Configuration Error covers only CI YAML and tool configuration. It never covers application code that handles config data, and never covers dependency bumps or pins (those are Dependency Issues).
4. A failed assert in a test means both Test Failure and Assertion Error.
5. An exception raised during a test means Test Failure plus Runtime Error. Add Assertion Error only if an assert also fails on a wrong value.
6. Comment-only edits are not Code Formatting unless a workflow tool enforces them.
7. Files excluded from a tool's scope cannot cause that tool's failure.
8. ruff format silently skips a file with a syntax error. ruff check catches it (E999), and pytest fails at import. Label it Syntax Error, plus Test Failure if tests run.
9. Test changes that align expectations with production code point to Test Failure and Assertion Error. Test changes that add mocks, API keys or offline data in place of real calls point to Environment Error or Runtime Error.
10. When a code change and a dependency pin both fix the same break (e.g. a library API change), label both Runtime Error and Dependency Issues.

## Sub-type rules

- 2 to 6 words, transferable, one per distinct root cause, no near-duplicates.
- Describe the defect, not the tool or its consequence. Good: unused import, bare except clause, KeyError on wrong dict path, trailing whitespace, None for non-Optional field. Bad: ruff check violation, error detected by mypy in helpers.py.
- Order sub-types to follow the order of the failure types.

## Output format

**Workflow tools:** tool list and scope notes, one line

**Diff to evidence:**
- `file`: what changed, the tool that would flag it, the Failure Type (mark incidental changes "no CI impact")

**Model classification audit:**
- Type: CORRECT or WRONG (reason)
- MISSING: Type (reason)

**Failure Types:** comma-separated
**Sub-types:** comma-separated

## Example

The workflow runs only `ruff check` on non-test files (`grep -v '/tests/'`). The diff removes unused imports in src, changes a bare `except:` to `except json.JSONDecodeError:`, and changes `== None` to `is None` in tests. The model says Linting, Code Formatting, Type Checking, Assertion Error.

Audit: Linting CORRECT. Code Formatting WRONG (no formatter in the workflow). Type Checking WRONG (no mypy). Assertion Error WRONG (tests are excluded and pytest does not run).

**Failure Types:** Linting
**Sub-types:** unused import, bare except clause

## Final check

- Every failure type maps to both a real workflow tool and a concrete diff hunk.
- No type is included only because the model or the step name said so.
- No type is dropped only because of step order.
- Sub-types are short, distinct, describe defects and transferable.