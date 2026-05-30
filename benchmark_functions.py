import os
import io
import re
import json
import git
import requests
import subprocess
from git import GitCommandError
from ruamel.yaml import YAML
  
def edit_workflow_push(workflow_file):
    """
    editing workflow.yaml so, that it would be run on push
    """

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
                print(f"[WARN] Could not delete {filename}: {e}")

def copy_and_edit_workflow_file(datapoint, repo):
    """
    Copies and minimally edits a workflow file:
      - Adds 'push' trigger if missing.
      - Normalizes deprecated runners only if found.
      - Keeps formatting, comments, and structure identical.
    """

    workflow_dir = os.path.join(repo.working_dir, ".github", "workflows")
    os.makedirs(workflow_dir, exist_ok=True)

    workflow_path = datapoint.get("workflow_path")
    workflow_content = datapoint.get("workflow")

    if not workflow_path or not os.path.isfile(os.path.join(repo.working_dir, workflow_path)):
        print(f"[WARN] Workflow path invalid or missing for {datapoint['id']}: {workflow_path}")
        return

    target_file = os.path.join(workflow_dir, os.path.basename(workflow_path))

    # --- Load workflow YAML ---
    if workflow_content:
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml_data = yaml.load(io.StringIO(workflow_content))
        
        with open(target_file, "w", encoding="utf-8") as f:
          yaml.dump(yaml_data, f)
    
    edit_workflow_push(target_file)
    
    reference_files = extract_referenced_workflows(target_file)

    reference_files.add(os.path.basename(target_file))

    delete_unreferenced_workflows(workflow_dir, reference_files)

    print(f"[INFO] Workflow updated: {target_file}")


def rename_precommit_files(repo_path):
    """
    rename pre-commit.yaml, so it will be run on push
    """
    workflow_dir = os.path.join(repo_path, ".github/workflows")
    
    for filename in os.listdir(workflow_dir):
        file_path = os.path.join(workflow_dir, filename)
        if os.path.isfile(file_path):
            if "pre-commit" in filename.lower():
                os.rename(
                    file_path, file_path.lower().replace("pre-commit", "precommit")
                )


def push_repo(repo, credentials, benchmark_owner, user_branch_name):
    """
    Pushes the corrected repo, return commit sha to use it for getting results
    """

    # TODO think about adding only changed files
    repo.git.add(".")
    repo.git.add(update=True)
    repo.index.commit(user_branch_name)
    username = credentials["username"]
    token = credentials["token"]
    try:
        repo.delete_remote("origin")
    except:
        pass
    origin_url = (
        f"https://{username}:{token}@github.com/{benchmark_owner}/{repo.name}.git"
    )
    try:
        origin = repo.remote("origin")
        origin.set_url(origin_url)
    except Exception:
        origin = repo.create_remote("origin", origin_url)
    repo.git.push("--force", "--set-upstream", origin, repo.head.ref)
    # Tried this, but it did not work - returned an error
    """
    cmdline: git push -u origin test_user
    stderr: 'gh auth git-credential: "erase" operation not supported
    remote: Invalid username or password.
    """
    # origin = repo.remote("origin")
    # with repo.git.custom_environment(GIT_USERNAME=username, GIT_PASSWORD=token):
    #     repo.git.push("-u", "origin", "test_user")
    #     origin.push()
    
    commit_hash = repo.head.commit.hexsha
    return commit_hash


def get_repo(datapoint, repos_folder, test_username, benchmark_owner, credentials):
    """
    clones repo and switches it to the required commit
    creates branch with username
    """
    repos_folder = os.path.abspath(repos_folder)
    id = datapoint["id"]
    username = credentials["username"]
    token = credentials["token"]
    model_name = credentials["model"]
    repo_name, repo_owner = datapoint["repo_name"], datapoint["repo_owner"]
    # TODO add original branch name to new_branch_name?
    new_branch_name = f"{test_username}__{model_name}__id_{id}"
    commit_hash = datapoint["sha_fail"]
    repo_path = os.path.join(repos_folder, f"{repo_owner}__{repo_name}")
    repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
    origin_url = (
        f"https://{username}:{token}@github.com/{benchmark_owner}/{repo_name}.git"
    )
    if (not os.path.exists(repo_path)) or (not os.listdir(repo_path)):
        repo = git.Repo.clone_from(repo_url, repo_path, depth=1)  # branch=commit_hash
    else:
        repo = git.Repo(repo_path)
    try:
        origin = repo.remote("origin")
    except:
        origin = repo.create_remote("origin", url=origin_url)
    repo.git.fetch("origin", commit_hash)
    try:
        repo.git.reset("--hard", commit_hash)
    except Exception as e:
        print(e)
        repo.git.checkout(commit_hash)
    # remove excessive files
    repo.git.clean("-fdx")
    if not any((h for h in repo.heads if h.name == new_branch_name)):
        # repo.delete_head("test_user", force=True)
        repo.create_head(new_branch_name, force=True)
    # TODO note that you should ban usage of the .git folder.
    # You need flag "-B" to checkout to the current state. Otherwise, the old brach state would be used
    repo.git.checkout("-B", new_branch_name)
    repo.name, repo.owner = repo_name, repo_owner
    
    return repo, new_branch_name


def get_run_data(repo_name, commit_sha, credentials, config):
    token = credentials["token"]
    headers = {"Authorization": f"token {token}"}

    jobs_url = f"https://api.github.com/repos/{config.benchmark_owner}/{repo_name}/commits/{commit_sha}/check-runs"
    
    response = requests.get(jobs_url, headers=headers)
    data = response.json()
    try:
        run_url = data["check_runs"][0]["html_url"]
        job_url = "/".join(run_url.split("/")[:-2])
        conclusions = [run["conclusion"] for run in data["check_runs"]]
        statuses = [run["status"] for run in data["check_runs"]]
        completed = [status == "completed" for status in statuses]
    except:
        print(f"Error in requesting jobs url {jobs_url}")
        print(data)
        job_url = ""
        conclusion = "error"
        return job_url, conclusion

    if not all(completed):
        conclusion = "waiting"
    elif "failure" in conclusions:
        conclusion = "failure"
    elif all([conclusion == "success" for conclusion in conclusions]):
        conclusion = "success"
    else:
        log_file_path = os.path.join(config.out_folder, "out_logs.txt")
        with open(log_file_path, "a") as f:
            f.write("--------------------DP BEGIN----------------------- \n")
            f.write(str(statuses) + "\n")
            f.write(str(conclusions) + "\n")
            f.write(str(data) + "\n")
            f.write("---------------------DP END------------------------- \n")
            conclusion = "error"
    
    return job_url, conclusion


def fix_none(datapoint, repo_path, repo=None, out_folder=None):
    return None


def fix_apply_diff(datapoint, repo_path, repo, out_folder):
    commit_sha = datapoint["sha_fail"][:7]
    # Absolute path so repo.git.apply (cwd=repo dir) can find the file.
    diff_path = os.path.abspath(os.path.join(out_folder, f"{commit_sha}.diff"))
    with open(diff_path, "w") as f:
        f.write(datapoint["diff"])

    try:
        repo.git.apply(diff_path)
    except GitCommandError as err:
        print(f"Sha = {datapoint['sha_fail']}")
        print(f"An error occurred while running the git command: {err}")
    os.remove(diff_path)
    return None


def process_datapoint(datapoint, fix_repo_function, config, credentials):
    """
    fix_repo_function - function that takes repo path and datapoint, repo object and out_folder.
    it should edit the repo in the folder, nothing to return
    credentials are passed in the following format:
    {'token': token, 'username': username}
    """

    # TODO think, what to do if test_username (which converts to a branch) is already present
    repo, user_branch_name = get_repo(
        datapoint,
        config.repos_folder,
        config.test_username,
        config.benchmark_owner,
        credentials,
    )

    # Prepares workflow file Moves target workflow file to the .github/workflows
    copy_and_edit_workflow_file(datapoint, repo)
    # Fixing the repo. fix_repo_function is provided by user.
    fix_repo_function(datapoint, repo.working_dir, repo, config.out_folder)
    # Push the corrected repo
    commit_sha = push_repo(repo, credentials, config.benchmark_owner, user_branch_name)
    job_identificator = {
        "repo_name": repo.name,
        "commit": commit_sha,
        "id": datapoint["id"],
        "sha_original": datapoint["sha_fail"],
        "branch_name": user_branch_name,
        "workflow": datapoint.get("workflow_path", "")
    }

    return job_identificator


def get_results(job_identificator, config, credentials):
    # We have to make some pause to get result or even url, unless it sees no runs
    repo_name = job_identificator["repo_name"]
    commit_sha = job_identificator["commit"]
    job_url, conclusion = get_run_data(repo_name, commit_sha, credentials, config)
    
    return job_url, conclusion


def dataset_to_json(dataset):
    json_list = []
    for item in dataset:
        json_list.append(item)
    
    return json_list


def _load_patch_records(out_folder):
    out_folder = os.path.abspath(out_folder)
    env_patch_file = os.environ.get("CIBENCH_PATCH_FILE") or os.environ.get("GENERATED_PATCHES_PATH")
    patch_candidates = [
        os.path.join(out_folder, "generated_patches.json"),
        os.path.join(out_folder, "preds.json"),
    ]
    if env_patch_file:
        patch_candidates.append(os.path.abspath(env_patch_file))

    existing_patch_files = [path for path in patch_candidates if os.path.exists(path)]
    if not existing_patch_files:
        raise FileNotFoundError(
            "[ERROR] Patch file not found. Checked: " + ", ".join(patch_candidates)
        )

    patch_records = []
    for patch_file in existing_patch_files:
        patch_records.extend(_read_patch_records_from_file(patch_file))
    return patch_records


def _read_patch_records_from_file(patch_file):
    with open(patch_file, "r", encoding="utf-8") as f:
        try:
            raw_patches = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Patch file empty or invalid, using empty list: {patch_file}")
            return []

    if isinstance(raw_patches, list):
        return [patch for patch in raw_patches if isinstance(patch, dict)]

    if isinstance(raw_patches, dict):
        patch_records = []
        for key, value in raw_patches.items():
            if isinstance(value, dict):
                patch = dict(value)
                patch.setdefault("id", key)
                patch.setdefault("sha_fail", key)
            else:
                patch = {"id": key, "sha_fail": key, "diff": str(value or "")}
            patch_records.append(patch)
        return patch_records

    print(f"[WARN] Unsupported patch file format in {patch_file}: {type(raw_patches).__name__}")
    return []


def fix_apply_generated_patch(datapoint, repo_path, repo, out_folder):
    out_folder = os.path.abspath(out_folder)
    patches = _load_patch_records(out_folder)

    current_id  = datapoint["id"]
    sha_fail    = datapoint.get("sha_fail", "")
    patch_data = next(
        (
            p
            for p in patches
            if (
                ids_match(p.get("id"), current_id)
                or ids_match(p.get("sha_fail"), sha_fail)
            )
            and p.get("diff", "").strip()
        ),
        None,
    )

    if not patch_data:
        print(f"[SKIP] No patch found for ID {current_id} / sha_fail {sha_fail[:12]}")
        return

    # Sanity check: the patch must have been generated for the same sha_fail
    # that the repo is currently at. If they differ, the diff will not apply.
    patch_sha = patch_data.get("sha_fail", "")
    if patch_sha and sha_fail and patch_sha != sha_fail:
        print(
            f"[SKIP] ID {current_id}: patch sha_fail {patch_sha[:12]} does not match "
            f"datapoint sha_fail {sha_fail[:12]} — skipping."
        )
        return

    temp_diff_path = os.path.join(out_folder, f"temp_{current_id}.diff")

    # Write patch to temp file
    # Ensure trailing newline — git apply requires it; missing \n → "corrupt patch"
    diff_content = patch_data["diff"]
    if not diff_content.endswith("\n"):
        diff_content += "\n"

    with open(temp_diff_path, "w", encoding="utf-8") as f:
        f.write(diff_content)

    # Pre-check patch validity using the same 3-way mode used for apply.
    check = subprocess.run(
        ["git", "apply", "--check", "--3way", temp_diff_path],
        cwd=repo_path,
        capture_output=True,
        text=True
    )

    if check.returncode != 0:
        print(f"[SKIP] Patch for ID {current_id} failed `git apply --check --3way`")
        print(f"[DEBUG] Git Error:\n{check.stderr.strip()}")
        # Numbered lines so "corrupt patch at line N" can be pinpointed
        diff_lines = patch_data["diff"].splitlines()
        print(f"[DEBUG] Patch ({len(diff_lines)} lines):")
        for lineno, line in enumerate(diff_lines, 1):
            print(f"  {lineno:4d} | {line}")
        os.remove(temp_diff_path)
        return

    # Apply patch if valid
    try:
        subprocess.run(["git", "apply", "--3way", temp_diff_path], cwd=repo_path, check=True)
        print(f"[SUCCESS] Applied patch for ID {current_id}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to apply patch for ID {current_id}")
        print(f"[DEBUG] {e.stderr}")
    finally:
        if os.path.exists(temp_diff_path):
            os.remove(temp_diff_path)


def ids_match(a, b) -> bool:
    # If same type, just compare directly
    if type(a) is type(b):
        return a == b

    # If types differ, handle int<->str specifically
    if isinstance(a, int) and isinstance(b, str):
        return str(a) == b.strip()
    if isinstance(a, str) and isinstance(b, int):
        return a.strip() == str(b)

    # Fallback for weird cases – you can decide to be stricter here
    return False
