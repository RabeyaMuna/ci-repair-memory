#!/usr/bin/env python3
"""
CI-Repair-Bench Health Check System
====================================
Automated quarterly maintenance to validate benchmark instance reproducibility.

This script:
1. Re-evaluates all benchmark instances by triggering CI workflows
2. Applies ground-truth patches and verifies complete workflow passes
3. Flags cancelled, invalid, timed-out, or failed executions
4. Generates detailed reports for manual inspection
5. Tracks changes for versioned benchmark releases
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import benchmark modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import CIFixBenchmark
from benchmark.benchmark_functions import (
    get_repo,
    copy_and_edit_workflow_file,
    ensure_workflow_enabled,
    push_repo,
    get_run_data,
)
from omegaconf import OmegaConf


class HealthCheckStatus:
    """Status codes for health check results"""
    PASS = "pass"  # Workflow passed with ground truth
    FAIL = "fail"  # Workflow failed even with ground truth
    CANCELLED = "cancelled"  # Workflow was cancelled
    TIMEOUT = "timeout"  # Workflow timed out
    INVALID = "invalid"  # Instance is invalid (repo deleted, etc.)
    ERROR = "error"  # Unexpected error during check


class BenchmarkHealthChecker:
    """
    Automated health checker for CI-Repair-Bench instances.
    Validates that ground-truth patches still work correctly.
    """

    def __init__(self, config_path: str, output_dir: str):
        self.config = OmegaConf.load(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this health check run
        self.run_timestamp = datetime.now(timezone.utc).isoformat()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Results tracking
        self.results = []
        self.flagged_instances = []

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load benchmark dataset"""
        print(f"[INFO] Loading dataset from: {dataset_path}")
        if dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        print(f"[INFO] Loaded {len(df)} instances")
        return df

    def check_instance(self, datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a single benchmark instance by:
        1. Cloning the repo at the failing commit
        2. Applying the ground-truth patch
        3. Triggering the CI workflow
        4. Checking if it passes
        """
        instance_id = datapoint["id"]
        repo_name = datapoint.get("repo", "unknown")

        print(f"\n[CHECK] Instance {instance_id} ({repo_name})")

        result = {
            "id": instance_id,
            "repo": repo_name,
            "sha_fail": datapoint.get("sha_fail", ""),
            "workflow": datapoint.get("workflow_path", ""),
            "status": HealthCheckStatus.ERROR,
            "message": "",
            "workflow_url": "",
            "conclusion": "",
            "checked_at": self.run_timestamp,
        }

        try:
            # Clone repository at failing commit
            credentials = {
                "username": self.config.benchmark_owner,
                "token": self.config.github_token,
            }

            repo = get_repo(
                datapoint,
                self.config.repos_folder,
                self.config.get("test_username", self.config.username_gh),
                self.config.benchmark_owner,
                credentials,
            )

            # Prepare workflow file
            copy_and_edit_workflow_file(datapoint, repo)

            # Apply ground-truth patch
            patch_content = datapoint.get("diff_fix", "")
            if not patch_content:
                result["status"] = HealthCheckStatus.INVALID
                result["message"] = "No ground-truth patch available"
                return result

            # Apply patch to repository
            patch_file = Path(repo.working_dir) / "ground_truth.patch"
            patch_file.write_text(patch_content)

            try:
                repo.git.apply(str(patch_file))
                print(f"[OK] Ground-truth patch applied")
            except Exception as e:
                result["status"] = HealthCheckStatus.FAIL
                result["message"] = f"Ground-truth patch failed to apply: {str(e)}"
                return result

            # Ensure workflow is enabled
            workflow_path = datapoint.get("workflow_path", "")
            if workflow_path:
                ensure_workflow_enabled(
                    repo.name, workflow_path, credentials, self.config
                )

            # Push and trigger workflow
            branch_name = f"health_check_{instance_id}_{self.run_id}"
            commit_sha = push_repo(repo, credentials, self.config.benchmark_owner, branch_name)

            print(f"[OK] Pushed to branch {branch_name}, commit {commit_sha[:8]}")

            # Wait for workflow to complete and get results
            # Note: This uses existing polling logic from benchmark
            import time
            max_wait = 600  # 10 minutes
            poll_interval = 30  # 30 seconds
            elapsed = 0

            workflow_url = ""
            conclusion = ""

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                try:
                    workflow_url, conclusion = get_run_data(
                        repo.name, commit_sha, credentials, self.config
                    )

                    if conclusion and conclusion != "waiting":
                        break

                except Exception as e:
                    print(f"[WARN] Error polling workflow: {e}")
                    continue

            result["workflow_url"] = workflow_url
            result["conclusion"] = conclusion

            # Determine status based on workflow conclusion
            if conclusion == "success":
                result["status"] = HealthCheckStatus.PASS
                result["message"] = "Ground-truth patch passes CI"
            elif conclusion == "failure":
                result["status"] = HealthCheckStatus.FAIL
                result["message"] = "Ground-truth patch failed CI"
            elif conclusion == "cancelled":
                result["status"] = HealthCheckStatus.CANCELLED
                result["message"] = "Workflow was cancelled"
            elif conclusion == "timed_out":
                result["status"] = HealthCheckStatus.TIMEOUT
                result["message"] = "Workflow timed out"
            elif not conclusion or conclusion == "waiting":
                result["status"] = HealthCheckStatus.TIMEOUT
                result["message"] = f"Workflow did not complete within {max_wait}s"
            else:
                result["status"] = HealthCheckStatus.ERROR
                result["message"] = f"Unknown conclusion: {conclusion}"

        except Exception as e:
            result["status"] = HealthCheckStatus.ERROR
            result["message"] = f"Unexpected error: {str(e)}"
            print(f"[ERROR] {e}")

        return result

    def run_health_check(self, dataset_path: str, instance_ids: Optional[List[str]] = None):
        """
        Run health check on all instances (or specified subset)

        Args:
            dataset_path: Path to benchmark dataset
            instance_ids: Optional list of instance IDs to check (None = all)
        """
        # Load dataset
        df = self.load_dataset(dataset_path)

        # Filter to specified IDs if provided
        if instance_ids:
            df = df[df["id"].astype(str).isin(instance_ids)]
            print(f"[INFO] Checking {len(df)} specified instances")

        # Run health check on each instance
        print(f"\n{'='*60}")
        print(f"Starting health check on {len(df)} instances")
        print(f"{'='*60}\n")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Health Check"):
            datapoint = row.to_dict()
            result = self.check_instance(datapoint)
            self.results.append(result)

            # Flag problematic instances
            if result["status"] != HealthCheckStatus.PASS:
                self.flagged_instances.append(result)

        # Generate reports
        self.generate_reports()

    def generate_reports(self):
        """Generate detailed health check reports"""

        # Summary statistics
        total = len(self.results)
        status_counts = {}
        for result in self.results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        # Save full results
        results_file = self.output_dir / f"health_check_{self.run_id}_full.json"
        with open(results_file, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "timestamp": self.run_timestamp,
                "total_instances": total,
                "status_summary": status_counts,
                "results": self.results,
            }, f, indent=2)

        print(f"\n[SAVED] Full results: {results_file}")

        # Save flagged instances (need manual inspection)
        flagged_file = self.output_dir / f"health_check_{self.run_id}_flagged.json"
        with open(flagged_file, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "timestamp": self.run_timestamp,
                "total_flagged": len(self.flagged_instances),
                "flagged_instances": self.flagged_instances,
            }, f, indent=2)

        print(f"[SAVED] Flagged instances: {flagged_file}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"HEALTH CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Run ID: {self.run_id}")
        print(f"Timestamp: {self.run_timestamp}")
        print(f"Total instances checked: {total}")
        print(f"\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {status:12s}: {count:4d} ({percentage:5.1f}%)")

        print(f"\nFlagged for manual inspection: {len(self.flagged_instances)}")

        if self.flagged_instances:
            print(f"\nFlagged instance IDs:")
            for result in self.flagged_instances[:10]:  # Show first 10
                print(f"  - ID {result['id']:4s} ({result['repo']:30s}): {result['status']} - {result['message'][:50]}")
            if len(self.flagged_instances) > 10:
                print(f"  ... and {len(self.flagged_instances) - 10} more (see {flagged_file})")

        print(f"\n{'='*60}\n")

        return results_file, flagged_file


def main():
    parser = argparse.ArgumentParser(
        description="Run automated health check on CI-Repair-Bench instances"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to benchmark config file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to benchmark dataset (parquet or json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="health_check/reports",
        help="Directory to save health check reports",
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="*",
        help="Specific instance IDs to check (optional, default: all)",
    )

    args = parser.parse_args()

    # Initialize health checker
    checker = BenchmarkHealthChecker(args.config, args.output_dir)

    # Run health check
    checker.run_health_check(args.dataset, args.ids)

    print(f"[DONE] Health check complete. Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
