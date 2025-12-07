import os
import subprocess
import time


# Heuristics for transient network / remote errors
TRANSIENT_ERROR_KEYWORDS = [
    "Recv failure",
    "Connection reset by peer",
    "fatal: expected flush after ref listing",
    "The requested URL returned error: 5",   # any 5xx
    "Operation timed out",
]


def _looks_transient_error(stderr: str) -> bool:
    stderr_low = stderr.lower()
    return any(k.lower() in stderr_low for k in TRANSIENT_ERROR_KEYWORDS)


def run_cmd(args, cwd=None, retries=1, retry_delay=5.0):
    """
    Run a command and raise with readable stderr on failure.

    - Detects stale .git/index.lock errors and removes the lock.
    - Retries on transient network / RPC errors (curl 56, expected flush, etc.).
    """
    cmd_str = " ".join(args)

    for attempt in range(retries + 1):
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.returncode == 0:
            return stdout.strip()

        # 1) Handle stale .git/index.lock once
        if (
            "index.lock" in stderr
            and "File exists" in stderr
            and cwd is not None
        ):
            git_dir = os.path.join(cwd, ".git")
            lock_path = os.path.join(git_dir, "index.lock")
            if os.path.exists(lock_path):
                print(f"[run_cmd] Removing stale git lock and retrying: {lock_path}")
                os.remove(lock_path)
                # after removing lock, immediately retry if we still have retries
                if attempt < retries:
                    time.sleep(retry_delay)
                    continue

        # 2) Retry on transient network errors
        if _looks_transient_error(stderr) and attempt < retries:
            print(f"[run_cmd] Transient error detected, retrying ({attempt+1}/{retries})...")
            time.sleep(retry_delay)
            continue

        # 3) If we get here, either no retries left or non-transient error
        raise RuntimeError(
            f"Command failed: {cmd_str}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    # Should never reach here
    raise RuntimeError(f"Command failed after retries: {cmd_str}")


def ensure_repo_at_commit(repo_url: str, repo_path: str, commit_sha: str) -> None:
    """
    Ensure that `repo_path` is a clone of `repo_url` and checked out at `commit_sha`
    in a clean, detached state.

    Changes vs your original:
    - No `git fetch --all`.
    - Only fetch the specific commit if we don't already have it.
    - Uses shallow, filtered fetch to reduce data and avoid RPC errors.
    """
    # 1) Clone if missing
    if not os.path.exists(repo_path):
        parent = os.path.dirname(repo_path)
        os.makedirs(parent, exist_ok=True)
        print(f"[ensure_repo_at_commit] Cloning {repo_url} -> {repo_path}")
        run_cmd(
            ["git", "clone", "--no-tags", "--filter=blob:none", "--depth", "100", repo_url, repo_path],
            cwd=parent,
            retries=3,
        )

    # 2) Basic sanity check: must be a git repo
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        raise RuntimeError(
            f"[ensure_repo_at_commit] Directory exists but is not a git repository: {repo_path}"
        )

    # 3) Make sure origin URL is correct (in case repo moved)
    run_cmd(["git", "remote", "set-url", "origin", repo_url], cwd=repo_path)

    # 4) Check if we already have the commit locally
    have_commit = True
    try:
        # `cat-file -e` returns 0 if object exists
        run_cmd(["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"], cwd=repo_path, retries=0)
    except RuntimeError:
        have_commit = False

    # 5) If commit is missing, fetch JUST that commit (shallow)
    if not have_commit:
        print(f"[ensure_repo_at_commit] Fetching commit {commit_sha}")
        run_cmd(
            [
                "git",
                "fetch",
                "--no-tags",
                "--filter=blob:none",
                "--depth",
                "200",
                "origin",
                commit_sha,
            ],
            cwd=repo_path,
            retries=3,
        )

    # 6) Reset and clean to the desired commit
    print(f"[ensure_repo_at_commit] Resetting and cleaning {repo_path} to {commit_sha}")
    run_cmd(["git", "reset", "--hard", commit_sha], cwd=repo_path, retries=1)
    run_cmd(["git", "clean", "-fdx"], cwd=repo_path, retries=1)

    # Detached HEAD is implicit after reset to a raw commit hash
    print(f"[ensure_repo_at_commit] Ready at {commit_sha}")
