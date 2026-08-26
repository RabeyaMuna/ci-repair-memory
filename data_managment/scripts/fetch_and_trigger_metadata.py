#!/usr/bin/env python3
"""
Fetch metadata for all instances and save to single JSON file.
If metadata missing, trigger workflow in fork to reproduce and fetch.
Output: data_managment/results/all_instances_metadata.json
"""
import os
import sys
import io
import re
import json
import base64
import requests
import subprocess
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from ruamel.yaml import YAML

# Load data-management settings and one shared GitHub credential.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_MANAGEMENT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = DATA_MANAGEMENT_DIR / 'config.yaml'

CONFIG = {}
if CONFIG_FILE.exists():
    config_yaml = YAML(typ='safe')
    with CONFIG_FILE.open('r', encoding='utf-8') as f:
        CONFIG = config_yaml.load(f) or {}

GITHUB_CONFIG = CONFIG.get('github', {})
TOKEN_VARIABLE = GITHUB_CONFIG.get('token_variable', 'GITHUB_TOKEN')
CONFIG_FORK_USER = GITHUB_CONFIG.get('benchmark_owner')

env_file_setting = GITHUB_CONFIG.get('env_file', '../.env')
ENV_FILE = (DATA_MANAGEMENT_DIR / env_file_setting).resolve()

# Explicit shell environment wins; otherwise read the selected variable from
# the configured env file. The token is never copied into config.yaml.
GITHUB_TOKEN = os.environ.get(TOKEN_VARIABLE)
if not GITHUB_TOKEN and ENV_FILE.exists():
    with ENV_FILE.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith(f'{TOKEN_VARIABLE}=') and not line.startswith('#'):
                GITHUB_TOKEN = line.split('=', 1)[1].strip()
                break

if not GITHUB_TOKEN:
    print(
        f"ERROR: {TOKEN_VARIABLE} not found in the environment or {ENV_FILE} "
        f"(configured by {CONFIG_FILE})"
    )
    sys.exit(1)


def git_auth_env() -> Dict[str, str]:
    """Use the same PAT for HTTPS Git that is used for GitHub API calls."""
    encoded = base64.b64encode(
        f'x-access-token:{GITHUB_TOKEN}'.encode('utf-8')
    ).decode('ascii')
    env = os.environ.copy()
    env.update({
        'GIT_TERMINAL_PROMPT': '0',
        'GIT_CONFIG_COUNT': '1',
        'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
        'GIT_CONFIG_VALUE_0': f'Authorization: Basic {encoded}',
    })
    return env

HEADERS = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'Authorization': f'Bearer {GITHUB_TOKEN}'
}

# Default paths
DATASET_PATH = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet'
OUTPUT_DIR = '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/data_managment/results'
OUTPUT_FILE = 'all_instances_metadata.json'
TRIGGER_FILE = 'triggered_waiting.json'

def get_workflow_runs_for_commit(repo_owner: str, repo_name: str, sha: str, workflow_file: str) -> List[Dict]:
    """Get workflow runs for a specific commit."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_file}/runs'
    params = {'event': 'push', 'per_page': 100}

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return []

    runs = response.json().get('workflow_runs', [])
    return [run for run in runs if run['head_sha'] == sha]

def get_workflow_runs_for_branch(repo_owner: str, repo_name: str, branch_name: str, workflow_file: str) -> List[Dict]:
    """Get workflow runs for a specific branch (used for triggered runs, whose head_sha is a new empty commit, not the original sha)."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_file}/runs'
    params = {'event': 'push', 'branch': branch_name, 'per_page': 100}

    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return []

    return response.json().get('workflow_runs', [])

def fetch_jobs_for_run(repo_owner: str, repo_name: str, run_id: int) -> List[Dict]:
    """Fetch jobs for a workflow run."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/jobs'
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        return response.json().get('jobs', [])
    return []

def fetch_commit_metadata(repo_owner: str, repo_name: str, sha: str) -> Optional[Dict]:
    """Fetch commit metadata."""
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{sha}'
    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        commit_data = response.json()
        return {
            'sha': sha,
            'message': commit_data['commit']['message'],
            'author': commit_data['commit']['author']['name'],
            'date': commit_data['commit']['author']['date'],
            'url': commit_data['html_url']
        }
    return None

def get_commits_between(repo_owner: str, repo_name: str, sha_fail: str, sha_success: str) -> List[str]:
    """Ordered list of commit SHAs from sha_fail through sha_success (both inclusive).

    Uses GitHub's compare API, whose 'commits' list excludes the base (sha_fail)
    and ends with the head (sha_success). Capped at GitHub's 250-commit compare
    limit; prints a warning if the range is truncated.
    """
    url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/compare/{sha_fail}...{sha_success}'
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return [sha_fail, sha_success]

    data = response.json()
    commits = [c['sha'] for c in data.get('commits', [])]
    if data.get('total_commits', len(commits)) > len(commits):
        print(f"    ⚠ compare truncated: {data.get('total_commits')} commits, only {len(commits)} returned")

    if not commits or commits[-1] != sha_success:
        commits.append(sha_success)
    return [sha_fail] + commits

def edit_workflow_push(workflow_file):
    """Edit workflow.yaml so that it would be run on push only."""
    yaml = YAML()
    with open(workflow_file, "r") as file:
        yaml_data = yaml.load(file)

    yaml_data["on"] = "push"

    with open(workflow_file, "w") as file:
        yaml.dump(yaml_data, file)

def extract_referenced_workflows(workflow_file):
    """Find reusable workflow references to preserve them."""
    yaml = YAML()
    with open(workflow_file, "r") as file:
        yaml_data = yaml.load(file)
    referenced = set()

    def _scan(value):
        if isinstance(value, str):
            matches = re.findall(r"\.github/workflows/([\w\-.\/]+(?:\.yml|\.yaml))", value)
            for m in matches:
                referenced.add(os.path.basename(m))
        elif isinstance(value, list):
            for item in value:
                _scan(item)
        elif isinstance(value, dict):
            for _, v in value.items():
                _scan(v)

    _scan(yaml_data)
    return referenced

def delete_unreferenced_workflows(workflow_dir, referenced_files):
    """Delete only workflows not referenced anywhere."""
    existing_files = [
        f for f in os.listdir(workflow_dir)
        if os.path.isfile(os.path.join(workflow_dir, f)) and f.endswith((".yml", ".yaml"))
    ]
    for filename in existing_files:
        if filename not in referenced_files:
            try:
                os.remove(os.path.join(workflow_dir, filename))
            except Exception as e:
                pass  # Silently skip if can't delete

def ensure_commit_in_fork(repo_owner: str, repo_name: str, sha: str, fork_user: str) -> bool:
    """Ensure commit exists in fork by fetching from upstream if needed.

    Returns True if commit is available, False otherwise.
    """
    fork_dir = f'/tmp/{fork_user}_{repo_name}_trigger'
    original_dir = os.getcwd()

    try:
        # Clone fork if needed
        if not os.path.exists(fork_dir):
            subprocess.run(
                f'git clone https://github.com/{fork_user}/{repo_name}.git {fork_dir}',
                shell=True, check=True, capture_output=True, text=True,
                env=git_auth_env()
            )

        os.chdir(fork_dir)

        # Check if commit already exists in fork
        result = subprocess.run(
            f'git cat-file -e {sha}', shell=True, capture_output=True
        )
        if result.returncode == 0:
            os.chdir(original_dir)
            return True

        # Commit not in fork - try fetching from upstream
        upstream_url = f'https://github.com/{repo_owner}/{repo_name}.git'
        result = subprocess.run('git remote', shell=True, capture_output=True, text=True)
        if 'upstream' not in result.stdout:
            subprocess.run(f'git remote add upstream {upstream_url}', shell=True, check=True, capture_output=True)
        else:
            subprocess.run(f'git remote set-url upstream {upstream_url}', shell=True, check=True, capture_output=True)

        # Fetch the specific commit
        subprocess.run(f'git fetch upstream {sha}', shell=True, check=True, capture_output=True)

        # Verify it's now available
        result = subprocess.run(
            f'git cat-file -e {sha}', shell=True, capture_output=True
        )
        os.chdir(original_dir)
        return result.returncode == 0

    except Exception:
        os.chdir(original_dir)
        return False


def trigger_workflow_in_fork(repo_owner: str, repo_name: str, sha: str,
                             fork_user: str, commit_type: str, workflow_filename: str,
                             workflow_content: str) -> Dict:
    """Trigger workflow in fork with ONLY the target workflow enabled.

    Always returns a dict: {'branch_name', 'trigger_commit', 'error'}.
    On failure branch_name/trigger_commit are None and 'error' holds the reason.
    trigger_commit is the new empty commit pushed on top of `sha` (its head_sha
    on GitHub will differ from `sha` itself).
    """
    # Ensure commit exists in fork before attempting trigger
    if not ensure_commit_in_fork(repo_owner, repo_name, sha, fork_user):
        print(f"✗ Commit {sha[:7]} not available in fork")
        return {'branch_name': None, 'trigger_commit': None, 'error': f'Commit {sha[:7]} not in fork or upstream'}

    print(f"→ Trigger...", end=' ', flush=True)

    fork_dir = f'/tmp/{fork_user}_{repo_name}_trigger'
    original_dir = os.getcwd()

    try:
        # Clone fork if needed
        if not os.path.exists(fork_dir):
            subprocess.run(
                f'git clone https://github.com/{fork_user}/{repo_name}.git {fork_dir}',
                shell=True, check=True, capture_output=True, text=True,
                env=git_auth_env()
            )

        os.chdir(fork_dir)

        # Add/repoint upstream remote (fork_dir may be reused across different repos)
        upstream_url = f'https://github.com/{repo_owner}/{repo_name}.git'
        result = subprocess.run('git remote', shell=True, capture_output=True, text=True)
        if 'upstream' not in result.stdout:
            subprocess.run(f'git remote add upstream {upstream_url}', shell=True, check=True, capture_output=True)
        else:
            subprocess.run(f'git remote set-url upstream {upstream_url}', shell=True, check=True, capture_output=True)

        # Full (unshallow) fetch of all branches/tags from upstream
        subprocess.run('git fetch upstream --tags --force --prune', shell=True, check=True, capture_output=True)
        subprocess.run('git fetch upstream --unshallow', shell=True, capture_output=True)

        # Fallback: fetch the exact commit directly in case it isn't reachable from a branch tip
        subprocess.run(f'git fetch upstream {sha}', shell=True, capture_output=True)

        # Discard any leftover local state, then hard-checkout the exact commit
        branch_name = f'trigger__{sha[:7]}__{commit_type}'
        subprocess.run(f'git branch -D {branch_name}', shell=True, stderr=subprocess.DEVNULL)
        subprocess.run('git clean -fdx', shell=True, capture_output=True)
        subprocess.run(f'git checkout -f -B {branch_name} {sha}', shell=True, check=True, capture_output=True)
        subprocess.run(f'git reset --hard {sha}', shell=True, check=True, capture_output=True)

        # Setup workflow: write target workflow with push trigger only
        workflow_dir = os.path.join(fork_dir, ".github", "workflows")
        os.makedirs(workflow_dir, exist_ok=True)

        target_workflow_file = os.path.join(workflow_dir, workflow_filename)

        # Write workflow content
        if workflow_content:
            yaml = YAML()
            yaml.preserve_quotes = True
            yaml_data = yaml.load(io.StringIO(workflow_content))
            with open(target_workflow_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f)

        # Edit to push trigger only
        edit_workflow_push(target_workflow_file)

        # Extract referenced workflows
        referenced_files = extract_referenced_workflows(target_workflow_file)
        referenced_files.add(workflow_filename)

        # Delete all other workflows
        delete_unreferenced_workflows(workflow_dir, referenced_files)

        # Commit changes
        subprocess.run('git add .github/workflows/', shell=True, check=True)
        subprocess.run(
            f'git commit --allow-empty -m "Trigger {workflow_filename} for {sha[:7]}"',
            shell=True, check=True, capture_output=True
        )

        # Push to trigger
        subprocess.run(
            f'git push -f origin {branch_name}', shell=True, check=True,
            capture_output=True, env=git_auth_env()
        )

        # Capture the actual pushed commit sha (the empty "trigger" commit, not the original sha)
        trigger_commit = subprocess.run(
            'git rev-parse HEAD', shell=True, check=True, capture_output=True, text=True
        ).stdout.strip()

        os.chdir(original_dir)
        print(f"✓")
        return {'branch_name': branch_name, 'trigger_commit': trigger_commit, 'error': None}

    except Exception as e:
        os.chdir(original_dir)
        detail = e.stderr.decode(errors='replace').strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        print(f"✗ {detail}")
        return {'branch_name': None, 'trigger_commit': None, 'error': detail}

def process_jobs_to_json_format(jobs: List[Dict]) -> Dict:
    """Process jobs into JSON format, including runner and job identity info needed for polling."""
    overall_jobs = []
    failed_jobs = []
    total_steps = 0

    for job in jobs:
        step_names = [step['name'] for step in job.get('steps', [])]
        total_steps += len(step_names)

        job_entry = {
            'job_id': job['id'],
            'job_name': job['name'],
            'status': job.get('status'),
            'conclusion': job.get('conclusion'),
            'runner_name': job.get('runner_name'),
            'runner_group_name': job.get('runner_group_name'),
            'labels': job.get('labels', []),
            'started_at': job.get('started_at'),
            'completed_at': job.get('completed_at'),
            'html_url': job.get('html_url'),
            'step_names': step_names
        }
        overall_jobs.append(job_entry)

        if job['conclusion'] == 'failure':
            failed_step_names = [step['name'] for step in job.get('steps', [])
                                if step.get('conclusion') == 'failure']
            failed_jobs.append({
                'job_id': job['id'],
                'job_name': job['name'],
                'runner_name': job.get('runner_name'),
                'step_names': failed_step_names if failed_step_names else step_names
            })

    return {
        'overall_jobs': overall_jobs,
        'overall_jobs_count': len(overall_jobs),
        'overall_steps_count': total_steps,
        'failed_jobs': failed_jobs,
        'no_failed_jobs': len(failed_jobs)
    }

def fetch_metadata_for_commit(repo_owner: str, repo_name: str, sha: str,
                              workflow_file: str, commit_label: str,
                              fork_user: str = None, trigger: bool = False,
                              workflow_content: str = None) -> Dict:
    """Fetch metadata for one commit. Trigger in fork if not found and trigger=True."""
    print(f"    {commit_label} ({sha[:7]})...", end=' ')

    commit_type = commit_label.lower()

    # Get commit info
    commit_info = fetch_commit_metadata(repo_owner, repo_name, sha)
    if not commit_info:
        print("✗ Commit not found")
        return {
            'original_commit': sha,
            'commit_type': commit_type,
            'action': workflow_file,
            'overall_jobs': [],
            'overall_jobs_count': 0,
            'overall_steps_count': 0,
            'failed_jobs': [],
            'no_failed_jobs': 0,
            'triggered': False
        }

    # Find workflow run in original repo
    runs = get_workflow_runs_for_commit(repo_owner, repo_name, sha, workflow_file)

    if not runs:
        # Trigger in user's fork (not original repo) — push and return immediately,
        # do NOT block waiting for the run to finish. A separate pass
        # (fetch_triggered_results.py) polls triggered_waiting.json later.
        if trigger and fork_user:
            print(f"→ Trigger in fork...", end=' ', flush=True)

            trigger_result = trigger_workflow_in_fork(repo_owner, repo_name, sha, fork_user, commit_label.lower(),
                                             workflow_file, workflow_content or '')

            if trigger_result.get('error'):
                return {
                    'original_commit': sha,
                    'commit_type': commit_type,
                    'action': workflow_file,
                    'commit_message': commit_info['message'],
                    'overall_jobs': [],
                    'overall_jobs_count': 0,
                    'overall_steps_count': 0,
                    'failed_jobs': [],
                    'no_failed_jobs': 0,
                    'triggered': False,
                    'trigger_failed': True,
                    'trigger_error': trigger_result['error']
                }

            print(f"✓ pushed, pending")
            return {
                'original_commit': sha,
                'commit_type': commit_type,
                'action': workflow_file,
                'commit_message': commit_info['message'],
                'commit_author': commit_info['author'],
                'commit_date': commit_info['date'],
                'triggered': True,
                'triggered_in_fork': f"{fork_user}/{repo_name}",
                'triggered_branch': trigger_result['branch_name'],
                'triggered_commit': trigger_result['trigger_commit'],
                'pending': True,
                'overall_jobs': [],
                'overall_jobs_count': 0,
                'overall_steps_count': 0,
                'failed_jobs': [],
                'no_failed_jobs': 0
            }
        else:
            print("✗ No run")
            return {
                'original_commit': sha,
                'commit_type': commit_type,
                'action': workflow_file,
                'commit_message': commit_info['message'],
                'overall_jobs': [],
                'overall_jobs_count': 0,
                'overall_steps_count': 0,
                'failed_jobs': [],
                'no_failed_jobs': 0,
                'triggered': False
            }

    run = runs[0]

    # Fetch jobs
    jobs = fetch_jobs_for_run(repo_owner, repo_name, run['id'])
    if not jobs:
        print("✗ No jobs")
        return {
            'original_commit': sha,
            'commit_type': commit_type,
            'action': workflow_file,
            'commit_message': commit_info['message'],
            'overall_jobs': [],
            'overall_jobs_count': 0,
            'overall_steps_count': 0,
            'failed_jobs': [],
            'no_failed_jobs': 0,
            'triggered': False
        }

    jobs_data = process_jobs_to_json_format(jobs)
    print(f"✓ {len(jobs)} jobs, {len(jobs_data['failed_jobs'])} failed")

    return {
        'original_commit': sha,
        'commit_type': commit_type,
        'action': workflow_file,
        'commit_message': commit_info['message'],
        'commit_author': commit_info['author'],
        'commit_date': commit_info['date'],
        'workflow_run_id': run['id'],
        'workflow_conclusion': run['conclusion'],
        'run_html_url': run.get('html_url'),
        'run_number': run.get('run_number'),
        'triggered': False,
        **jobs_data
    }

def fetch_instance_metadata(instance_id: str, df: pd.DataFrame,
                           fork_user: str = None, trigger: bool = False) -> Dict:
    """Fetch metadata for one instance. Trigger workflows if metadata missing."""
    row = df[df['id'] == instance_id].iloc[0]

    print(f"\n  ID {instance_id}: {row['repo_owner']}/{row['repo_name']}")

    metadata = {
        'id': instance_id,
        'repo': f"{row['repo_owner']}/{row['repo_name']}",
        'repo_owner': row['repo_owner'],
        'repo_name': row['repo_name'],
        'workflow_name': row['workflow_name'],
        'workflow_file': row['workflow_filename'],
        'sha_fail': row['sha_fail'],
        'sha_success': row['sha_success'],
        'compare': f"https://github.com/{row['repo_owner']}/{row['repo_name']}/compare/{row['sha_fail']}...{row['sha_success']}",
    }

    # All commits from sha_fail through sha_success (inclusive), not just the two endpoints
    workflow_content = row.get('workflow', '')
    commit_shas = get_commits_between(row['repo_owner'], row['repo_name'], row['sha_fail'], row['sha_success'])
    print(f"    {len(commit_shas)} commit(s) between sha_fail and sha_success")

    commit_metadata = []
    for idx, sha in enumerate(commit_shas):
        if idx == 0:
            label = 'FAIL'
        elif idx == len(commit_shas) - 1:
            label = 'SUCCESS'
        else:
            label = f'COMMIT_{idx}'

        commit_metadata.append(fetch_metadata_for_commit(
            row['repo_owner'], row['repo_name'], sha,
            row['workflow_filename'], label,
            fork_user=fork_user, trigger=trigger,
            workflow_content=workflow_content
        ))

    metadata['commit_metadata'] = commit_metadata

    return metadata

def is_pending(metadata: Dict) -> bool:
    """True if an instance still has one or more triggered-but-unresolved commits."""
    return any(item.get('pending') for item in metadata['commit_metadata'])

def is_instance_valid(metadata: Dict) -> bool:
    """True if the instance has useful metadata (at least sha_fail OR sha_success).

    Changed from requiring BOTH endpoints to accepting partial data. An instance
    is valid if:
    - No commits are pending or trigger_failed (those need retry)
    - At least ONE of sha_fail or sha_success has job data

    This allows saving partial metadata instead of marking instances as missing
    when we have data for some commits.
    """
    commit_metadata = metadata.get('commit_metadata', [])
    if not commit_metadata:
        return False

    fail_commit = commit_metadata[0]
    success_commit = commit_metadata[-1]

    # Reject if any endpoint is pending or failed to trigger
    for item in (fail_commit, success_commit):
        if item.get('pending') or item.get('trigger_failed'):
            return False

    # Accept if at least ONE endpoint has job data
    fail_has_data = fail_commit.get('overall_jobs_count', 0) > 0
    success_has_data = success_commit.get('overall_jobs_count', 0) > 0

    return fail_has_data or success_has_data

def build_trigger_entry(row: pd.Series, metadata: Dict) -> Dict:
    """Build a triggered_waiting.json entry: instance + per-commit trigger/result state.

    Consumed later by fetch_triggered_results.py, which polls each pending
    commit's branch, fills in run/job results, and once all commits are
    resolved, appends the completed instance to all_instances_metadata.json.
    """
    return {
        'id': metadata['id'],
        'sha_fail': metadata['sha_fail'],
        'sha_success': metadata['sha_success'],
        'compare': metadata['compare'],
        'repo_owner': row['repo_owner'],
        'repo_name': row['repo_name'],
        'workflow_name': row['workflow_name'],
        'workflow_file': row['workflow_filename'],
        'commit_metadata': metadata['commit_metadata']
    }

def load_trigger_entries(path: str) -> List[Dict]:
    """Load triggered_waiting.json (list of pending instance entries)."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)

def save_trigger_entries(entries: List[Dict], path: str) -> None:
    """Persist triggered_waiting.json immediately (used after each pending instance)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2)

def save_all_metadata(all_metadata: List[Dict], output_path: str) -> None:
    """Persist all_metadata to disk immediately (used after each instance)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_and_trigger_metadata.py [OPTIONS] <id1> [id2] ...")
        print("   OR: python fetch_and_trigger_metadata.py [OPTIONS] --all")
        print("\nOptions:")
        print("  --trigger            Trigger workflows in fork if metadata missing")
        print("  --fork-user=USER     Override github.benchmark_owner from config.yaml")
        print("\nExamples:")
        print("  python fetch_and_trigger_metadata.py --all")
        print("  python fetch_and_trigger_metadata.py --all --trigger")
        return

    # Parse arguments
    trigger = False
    fork_user = CONFIG_FORK_USER
    instance_ids = []

    for arg in sys.argv[1:]:
        if arg == '--trigger':
            trigger = True
        elif arg.startswith('--fork-user='):
            fork_user = arg.split('=')[1]
        elif arg == '--all':
            instance_ids = None  # Will be set to all IDs below
        elif not arg.startswith('--'):
            instance_ids.append(arg) if instance_ids is not None else None

    if trigger and not fork_user:
        print("ERROR: Set github.benchmark_owner in config.yaml or pass --fork-user=USER")
        return

    # Load dataset
    df = pd.read_parquet(DATASET_PATH)

    # Get instance IDs
    if instance_ids is None:
        instance_ids = df['id'].tolist()

    # Load existing data — keep only fully-resolved (valid) instances as "done";
    # anything with empty jobs/failed triggers is dropped so it gets retried.
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    metadata_by_id = {}

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        for item in loaded:
            if is_instance_valid(item):
                metadata_by_id[item['id']] = item
        print(f"Loaded {len(loaded)} existing instances ({len(metadata_by_id)} valid, "
              f"{len(loaded) - len(metadata_by_id)} incomplete — will retry)")

    # Load already-triggered-but-unresolved instances (skip re-triggering them)
    trigger_path = os.path.join(OUTPUT_DIR, TRIGGER_FILE)
    trigger_entries = load_trigger_entries(trigger_path)
    pending_ids = {entry['id'] for entry in trigger_entries}
    if trigger_entries:
        print(f"Loaded {len(trigger_entries)} pending triggered instance(s) from {trigger_path}")

    print(f"\nProcessing {len(instance_ids)} instances...")
    if trigger:
        print(f"Trigger mode: ON (fork: {fork_user})")
    print('='*80)

    # Process each instance
    new_count = 0
    triggered_count = 0
    retry_count = 0
    interrupted = False

    try:
        for instance_id in instance_ids:
            if instance_id in metadata_by_id:
                print(f"\n  ID {instance_id}: Already processed (valid data), skipping")
                continue
            if instance_id in pending_ids:
                print(f"\n  ID {instance_id}: Already triggered, pending result fetch — skipping")
                continue

            try:
                metadata = fetch_instance_metadata(instance_id, df, fork_user=fork_user, trigger=trigger)

                if is_pending(metadata):
                    # Push already happened; don't block waiting — record for later polling.
                    row = df[df['id'] == instance_id].iloc[0]
                    trigger_entries.append(build_trigger_entry(row, metadata))
                    pending_ids.add(instance_id)
                    save_trigger_entries(trigger_entries, trigger_path)
                    triggered_count += 1
                elif is_instance_valid(metadata):
                    metadata_by_id[instance_id] = metadata
                    new_count += 1
                    save_all_metadata(list(metadata_by_id.values()), output_path)
                else:
                    # Trigger/commit fetch failed outright (e.g. permission error) —
                    # don't persist as done; next run will retry it from scratch.
                    print(f"    ⚠ ID {instance_id}: unresolved failure(s), will retry next run")
                    retry_count += 1

            except Exception as e:
                print(f"\n  ✗ ID {instance_id}: Error - {e}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n⚠️  Interrupted by user — progress already saved, resume later to continue.")

    # Final save (redundant safety net; loop already saves after each instance)
    save_all_metadata(list(metadata_by_id.values()), output_path)
    save_trigger_entries(trigger_entries, trigger_path)

    print("\n" + "="*80)
    if interrupted:
        print("⚠️  Run was interrupted — rerun the same command to resume remaining instances.")
    print(f"✓ Processed: {new_count} new instances")
    print(f"✓ Total: {len(metadata_by_id)} instances")
    if retry_count > 0:
        print(f"⚠️  Unresolved (will retry next run): {retry_count} instances")
    if triggered_count > 0:
        print(f"⏱️  Triggered (pending): {triggered_count} instances")
        print(f"✓ Saved pending list to: {trigger_path}")
        print(f"\nOnce their workflow runs finish, run:")
        print(f"  python data_managment/scripts/fetch_triggered_results.py")
    print(f"✓ Saved to: {output_path}")

if __name__ == '__main__':
    main()
