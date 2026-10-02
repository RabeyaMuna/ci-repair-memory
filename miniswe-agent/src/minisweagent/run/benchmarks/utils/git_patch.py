"""Lossless Git patch capture and validation in an isolated index."""

import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


class PatchValidationError(ValueError):
    """A patch cannot faithfully represent or apply the requested changes."""


class UnresolvedMergeConflict(PatchValidationError):
    """The agent must reconcile the checkout's unmerged index entries."""


def conflicted_paths(repo: Path) -> list[str]:
    return list(dict.fromkeys(
        os.fsdecode(entry.split(b"\t", 1)[1])
        for entry in run_git(repo, "ls-files", "--unmerged", "-z").stdout.split(b"\0") if entry
    ))


def check_resolved_files(repo: Path, paths: list[str]) -> None:
    """Check files known to have conflicted, including after the agent stages them."""
    for name in paths:
        path = repo / name
        if path.is_file() and not path.is_symlink():
            contents = path.read_bytes()
            if re.search(rb"^<{7,} [^\n]*\n.*?^={7,}\r?\n.*?^>{7,} [^\n]*(?:\n|$)", contents, re.MULTILINE | re.DOTALL):
                raise UnresolvedMergeConflict(f"Unresolved merge conflicts: markers remain in {name}")


def run_git(repo: Path, *args: str, input: bytes | None = None, env=None, check=True):
    git_env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
    ):
        git_env.pop(name, None)
    if env is not None:
        git_env.update(env)
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input,
        env=git_env,
        capture_output=True,
        timeout=120,
    )
    if check and result.returncode:
        raise PatchValidationError(
            result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result


@contextmanager
def patch_repository(repo: Path, base_commit: str):
    """Borrow source objects read-only; keep index and new objects temporary."""
    repo = Path(repo).resolve()
    base = (
        run_git(repo, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
        .stdout.strip()
        .decode()
    )
    objects = run_git(repo, "rev-parse", "--git-path", "objects").stdout.strip()
    object_path = (repo / os.fsdecode(objects)).resolve()
    object_format = (
        run_git(repo, "rev-parse", "--show-object-format").stdout.strip().decode()
    )
    with tempfile.TemporaryDirectory(prefix="ci-patch-") as directory:
        isolated = Path(directory)
        run_git(isolated, "init", "-q", f"--object-format={object_format}")
        # An alternates file avoids path-list quoting issues in environment variables.
        (isolated / ".git/objects/info/alternates").write_bytes(
            os.fsencode(object_path) + b"\n"
        )
        run_git(isolated, "read-tree", base)
        yield isolated, base


def diff_trees(repo: Path, base: str, target: str) -> str:
    """Let Git encode the patch; preserve its exact bytes and line endings."""
    args = (
        "-c",
        "core.quotePath=true",
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-relative",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--unified=3",
        "--submodule=short",
        base,
        target,
        "--",
    )
    raw = run_git(repo, *args).stdout
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # info/attributes has higher precedence than repository .gitattributes.
        # This repository is temporary; binary records preserve non-UTF-8 bytes.
        (repo / ".git/info/attributes").write_text("* binary\n", encoding="utf-8")
        return run_git(repo, *args).stdout.decode("utf-8")


def check_round_trip(
    repo: Path, base: str, patch: str, expected_tree: str | None = None
):
    """Apply to the failed commit, refusing conflicts and any tree mismatch."""
    run_git(repo, "read-tree", base)
    if patch:
        raw = patch.encode("utf-8")
        run_git(
            repo,
            "apply",
            "--cached",
            "--check",
            "--3way",
            "--whitespace=nowarn",
            input=raw,
        )
        run_git(repo, "apply", "--cached", "--3way", "--whitespace=nowarn", input=raw)
    actual = run_git(repo, "write-tree").stdout.strip().decode()
    if expected_tree is not None and actual != expected_tree:
        raise PatchValidationError(
            "Applied patch does not reproduce the complete final Git tree"
        )
    return actual


def should_exclude_path(path: str) -> bool:
    """Check if a path should be excluded from patches (auto-generated dirs)."""
    excluded_patterns = [
        # Python virtual environments
        '.venv', 'venv', '.env', 'env',
        # Python caches and build artifacts
        '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
        'dist', 'build', '.eggs', '*.egg-info', '.tox',
        # Node.js
        'node_modules', '.npm', '.yarn',
        # IDE and editor
        '.idea', '.vscode', '.vs',
        # OS
        '.DS_Store', 'Thumbs.db',
        # Other build artifacts
        'target', '.gradle', '.mvn',
    ]

    path_parts = path.split('/')
    for part in path_parts:
        # Check exact matches
        if part in excluded_patterns:
            return True
        # Check prefixes (e.g., .venv-wandb-44)
        for pattern in excluded_patterns:
            if pattern.startswith('.') and part.startswith(pattern):
                return True
            # Handle wildcard patterns
            if '*' in pattern and part.endswith(pattern.replace('*', '')):
                return True

    return False


def collect_workspace_patch(repo: Path, base_commit: str) -> str:
    """Capture committed, staged, unstaged and nonignored new files against base.

    The original checkout and index are never modified. An empty patch means
    the final tree equals the failed commit; errors are never an empty patch.
    """
    repo = Path(repo).resolve()
    if run_git(repo, "ls-files", "--unmerged", "-z").stdout:
        raise UnresolvedMergeConflict(
            "Resolve the checkout's merge conflicts before generating a patch"
        )
    merge_head = run_git(repo, "rev-parse", "--git-path", "MERGE_HEAD").stdout.strip()
    if (repo / os.fsdecode(merge_head)).exists():
        # Staging alone does not resolve a merge. Ignore unrelated whitespace
        # warnings and only reject newly introduced conflict markers here.
        check = run_git(repo, "diff", "--check", "HEAD", check=False)
        if b"leftover conflict marker" in check.stdout:
            raise UnresolvedMergeConflict("Unresolved merge conflicts: new markers remain in the active merge")
    sparse = run_git(repo, "config", "--bool", "core.sparseCheckout", check=False)
    if sparse.stdout.strip() == b"true":
        raise PatchValidationError(
            "Patch capture requires a complete checkout, not a sparse checkout"
        )
    with patch_repository(repo, base_commit) as (isolated, base):
        # Seed from the agent's index, including staged additions that may now
        # match an ignore rule. Export entries to support split indexes too.
        entries = run_git(repo, "ls-files", "--stage", "-z").stdout
        run_git(isolated, "read-tree", "--empty")
        run_git(isolated, "update-index", "-z", "--index-info", input=entries)
        # Keep the source repository's attributes, filters and filemode config,
        # but direct all index/object writes into the temporary repository.
        env = {
            "GIT_INDEX_FILE": str(isolated / ".git/index"),
            "GIT_OBJECT_DIRECTORY": str(isolated / ".git/objects"),
        }
        run_git(repo, "add", "-A", "--", ".", env=env)

        # Filter out auto-generated directories and files
        entries_after_add = run_git(isolated, "ls-files", "--stage", "-z").stdout
        filtered_entries = []
        for entry in entries_after_add.split(b"\0"):
            if not entry:
                continue
            # Entry format: <mode> <object> <stage>\t<path>
            parts = entry.split(b"\t", 1)
            if len(parts) == 2:
                path = os.fsdecode(parts[1])
                if not should_exclude_path(path):
                    filtered_entries.append(entry)

        # Rebuild index with filtered entries
        if filtered_entries:
            filtered_data = b"\0".join(filtered_entries) + b"\0"
            run_git(isolated, "read-tree", "--empty")
            run_git(isolated, "update-index", "-z", "--index-info", input=filtered_data)
        for entry in run_git(isolated, "ls-files", "--stage", "-z").stdout.split(b"\0"):
            if entry.startswith(b"160000 "):
                path = os.fsdecode(entry.split(b"\t", 1)[1])
                submodule = repo / path
                if (submodule / ".git").exists() and run_git(
                    submodule,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                ).stdout:
                    raise PatchValidationError(
                        f"Uncommitted submodule changes cannot be stored in the parent patch: {path}"
                    )
        target = run_git(isolated, "write-tree").stdout.strip().decode()
        patch = diff_trees(isolated, base, target)
        check_round_trip(isolated, base, patch, target)
        return patch


def validate_against_commit(patch: str, repo: Path, base_commit: str) -> None:
    with patch_repository(repo, base_commit) as (isolated, base):
        check_round_trip(isolated, base, patch)
