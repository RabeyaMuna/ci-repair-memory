# ============================================================
#  verify_successful_commits.py — Verify SHA Success Commits
# ============================================================
"""
This script tests if sha_success commits actually pass CI.
It checks out the sha_success commit (instead of sha_fail),
applies the workflow file, and triggers CI without any patch.

Usage:
  python maintenance/verify_successful_commits.py --ids 72,75,81,82
"""

import os
import sys
import argparse

# Add parent directory to path to import benchmark
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from benchmark import CIFixBenchmark
from benchmark.benchmark_functions import (
    get_repo,
    copy_and_edit_workflow_file,
    push_repo,
)
import git

# ============================================================
#  Configuration
# ============================================================
parser = argparse.ArgumentParser(
    description="Verify that sha_success commits pass CI"
)
parser.add_argument(
    "--ids",
    required=True,
    help="Comma-separated dataset IDs to verify",
)
args = parser.parse_args()

# Get project root directory (parent of maintenance)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "config.yaml")
config = OmegaConf.load(config_path)

# Model name for verification mode
model_name = "verify_success"

# Initialize benchmark object
CIBenchPython = CIFixBenchmark(model_name, config_path)

# ============================================================
#  Load Dataset
# ============================================================
dataset_info = os.path.join(config.get("base_dir"), "dataset", "lca_dataset.parquet")

# Load dataset
all_ids = [row["id"] for row in CIBenchPython.get_dataset(dataset_info=dataset_info)]
requested_ids = args.ids.strip()
selected_ids = [item.strip() for item in requested_ids.split(",") if item.strip()]

print(f"[VERIFY] Testing sha_success commits for IDs: {', '.join(selected_ids)}")
print(f"[VERIFY] Total IDs to verify: {len(selected_ids)}")

# ============================================================
#  Custom Function to Verify Success Commits
# ============================================================
def verify_success_commit(datapoint, repo_path, repo, out_folder):
    """
    Verify sha_success by checking it out and triggering CI.
    No patch is applied - just test if the success commit passes.

    Returns:
        True if sha_success was checked out successfully
        None if there was an error or sha_success doesn't exist
    """
    current_id = datapoint["id"]
    sha_success = datapoint.get("sha_success", "")
    sha_fail = datapoint.get("sha_fail", "")

    if not sha_success:
        print(f"[SKIP] ID {current_id}: No sha_success available")
        return None

    print(f"\n{'='*80}")
    print(f"[VERIFY] ID {current_id}")
    print(f"{'='*80}")
    print(f"  SHA Fail:    {sha_fail[:12]}")
    print(f"  SHA Success: {sha_success[:12]}")

    try:
        # Checkout sha_success instead of sha_fail
        print(f"[VERIFY] Checking out sha_success {sha_success[:12]}...")
        repo.git.fetch("origin", sha_success)
        repo.git.reset("--hard", sha_success)
        repo.git.clean("-fdx")

        print(f"[VERIFY] Successfully checked out sha_success")
        print(f"[VERIFY] Workflow will be copied and push triggered (no patch applied)")

        # The workflow file will be copied by copy_and_edit_workflow_file
        # No patch is applied - we're testing the success commit as-is
        return True

    except Exception as e:
        print(f"[ERROR] ID {current_id}: Failed to checkout sha_success: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
#  Run Verification
# ============================================================
print("\n" + "="*80)
print("STARTING VERIFICATION OF SUCCESSFUL COMMITS")
print("="*80)
print(f"IDs to verify: {', '.join(selected_ids)}")
print(f"Output folder: {config.get('out_folder')}")
print("="*80 + "\n")

CIBenchPython.eval_dataset(
    fix_repo_function=verify_success_commit,
    dataset_info=dataset_info,
    num_dp=None,
    ids_list=selected_ids,
    force_download=False,
    resume=False,
)

# ============================================================
#  Get Results
# ============================================================
print("\n" + "="*80)
print("COLLECTING VERIFICATION RESULTS")
print("="*80)

CIBenchPython.get_results()

print("\n" + "="*80)
print("VERIFICATION COMPLETE!")
print("="*80)
print(f"Results saved to: {config.get('out_folder')}/jobs_ids_verify_success.jsonl")
print(f"Check results with: {config.get('out_folder')}/jobs_results_verify_success.jsonl")
print("="*80)
