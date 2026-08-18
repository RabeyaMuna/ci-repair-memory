#!/usr/bin/env python3
"""
Fetch detailed validation steps from GitHub Actions workflow runs.

This script:
1. Reads job results from JSONL files (e.g., jobs_failure_diff.jsonl)
2. Uses GitHub API to fetch all jobs and steps for each workflow run
3. Creates a comprehensive mapping of all validation steps (passed, failed, cancelled)
4. Separately tracks failed steps for easy mapping with CI logs

Output: validation_steps_detailed.json
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime


class ValidationStepsFetcher:
    def __init__(self, github_token: str, results_dir: str, benchmark_owner: str, input_dir: str = None):
        """
        Initialize the fetcher.

        Args:
            github_token: GitHub personal access token
            results_dir: Directory for output files
            benchmark_owner: GitHub username that owns the forked repos
            input_dir: Directory containing input JSONL files (default: dataset/)
        """
        self.github_token = github_token
        self.results_dir = Path(results_dir)
        self.input_dir = Path(input_dir) if input_dir else Path(results_dir).parent / "dataset"
        self.benchmark_owner = benchmark_owner
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github+json"
        }
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = None

    def check_rate_limit(self):
        """Check and handle GitHub API rate limits."""
        if self.rate_limit_remaining < 10:
            if self.rate_limit_reset:
                wait_time = max(0, self.rate_limit_reset - time.time())
                print(f"[RATE LIMIT] Waiting {wait_time:.0f}s for rate limit reset...")
                time.sleep(wait_time + 1)
            else:
                print("[RATE LIMIT] Low on requests, waiting 60s...")
                time.sleep(60)

    def update_rate_limit(self, response: requests.Response):
        """Update rate limit info from response headers."""
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
        if 'X-RateLimit-Reset' in response.headers:
            self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])

    def extract_run_id(self, url: str) -> Optional[str]:
        """
        Extract workflow run ID from GitHub Actions URL.

        Example: https://github.com/RabeyaMuna/wandb/actions/runs/32019351034
        Returns: "32019351034"
        """
        match = re.search(r'/actions/runs/(\d+)', url)
        return match.group(1) if match else None

    def fetch_workflow_jobs(self, repo_name: str, run_id: str) -> List[Dict]:
        """
        Fetch all jobs for a workflow run from GitHub API.

        Args:
            repo_name: Repository name (e.g., "wandb")
            run_id: Workflow run ID

        Returns:
            List of job dictionaries with steps
        """
        self.check_rate_limit()

        url = f"https://api.github.com/repos/{self.benchmark_owner}/{repo_name}/actions/runs/{run_id}/jobs"

        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            self.update_rate_limit(response)

            if response.status_code == 200:
                data = response.json()
                return data.get('jobs', [])
            elif response.status_code == 404:
                print(f"[WARN] Run {run_id} not found for {repo_name}")
                return []
            elif response.status_code == 403:
                print(f"[ERROR] Rate limit or access denied for {repo_name}/{run_id}")
                return []
            else:
                print(f"[ERROR] HTTP {response.status_code} for {repo_name}/{run_id}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Network error fetching {repo_name}/{run_id}: {e}")
            return []

    def process_job_steps(self, job: Dict) -> Dict:
        """
        Process a job and extract step information.

        Args:
            job: Job dictionary from GitHub API

        Returns:
            Dictionary with job and step details
        """
        job_name = job.get('name', 'unknown')
        job_status = job.get('status', 'unknown')
        job_conclusion = job.get('conclusion', 'unknown')

        steps = []
        all_step_names = []
        all_step_names_short = []  # Just step names without job prefix
        failed_step_names = []
        failed_step_names_short = []

        for step in job.get('steps', []):
            step_name = step.get('name', 'unknown')
            step_status = step.get('status', 'unknown')
            step_conclusion = step.get('conclusion', 'unknown')

            # Create multiple formats for flexible matching:
            # 1. Full format: "job_name: step_name" - for detailed tracking
            full_step_name = f"{job_name}: {step_name}"

            # 2. Short format: just "step_name" - matches some CI log formats
            short_step_name = step_name

            step_info = {
                "job_name": job_name,
                "step_name": step_name,
                "full_name": full_step_name,
                "short_name": short_step_name,
                "status": step_status,
                "conclusion": step_conclusion,
                "number": step.get('number', 0),
                "started_at": step.get('started_at'),
                "completed_at": step.get('completed_at')
            }

            steps.append(step_info)
            all_step_names.append(full_step_name)
            all_step_names_short.append(short_step_name)

            # Track failed steps (failure or cancelled)
            if step_conclusion in ['failure', 'cancelled', 'timed_out']:
                failed_step_names.append(full_step_name)
                failed_step_names_short.append(short_step_name)

        return {
            "job_name": job_name,
            "job_id": job.get('id'),
            "job_status": job_status,
            "job_conclusion": job_conclusion,
            "steps": steps,
            "all_step_names": all_step_names,
            "all_step_names_short": all_step_names_short,
            "failed_step_names": failed_step_names,
            "failed_step_names_short": failed_step_names_short,
            # Add job-level tracking for CI log matching
            "job_failed": job_conclusion in ['failure', 'cancelled', 'timed_out']
        }

    def process_workflow_run(self, job_data: Dict) -> Dict:
        """
        Process a single workflow run and extract all validation steps.

        Args:
            job_data: Dictionary from jobs_*.jsonl

        Returns:
            Detailed validation steps information
        """
        issue_id = job_data.get('id')
        repo_name = job_data.get('repo_name')
        commit = job_data.get('commit')
        url = job_data.get('url', '')
        conclusion = job_data.get('conclusion', 'unknown')

        print(f"[INFO] Processing ID {issue_id} ({repo_name})")

        # Extract run ID from URL
        run_id = self.extract_run_id(url)
        if not run_id:
            print(f"[WARN] Could not extract run ID from URL: {url}")
            return None

        # Fetch jobs from GitHub API
        jobs = self.fetch_workflow_jobs(repo_name, run_id)

        if not jobs:
            print(f"[WARN] No jobs found for ID {issue_id}")
            return None

        # Process all jobs and their steps
        all_validation_steps = []
        failed_validation_steps = []
        all_job_names = []  # Job-level names for CI log matching
        failed_job_names = []  # Failed job names for CI log matching
        job_details = []

        for job in jobs:
            job_info = self.process_job_steps(job)
            job_details.append(job_info)

            # Add to all validation steps
            all_validation_steps.extend(job_info['all_step_names'])
            failed_validation_steps.extend(job_info['failed_step_names'])

            # Track job-level names (for CI log matching)
            job_name = job_info['job_name']
            all_job_names.append(job_name)
            if job_info['job_failed']:
                failed_job_names.append(job_name)

        # Create summary
        result = {
            "id": issue_id,
            "repo_name": repo_name,
            "commit": commit,
            "sha_original": job_data.get('sha_original'),
            "branch_name": job_data.get('branch_name'),
            "workflow": job_data.get('workflow'),
            "workflow_run_url": url,
            "run_id": run_id,
            "overall_conclusion": conclusion,
            "pushed_at": job_data.get('pushed_at'),
            "fetched_at": datetime.utcnow().isoformat() + "Z",

            # Summary statistics
            "summary": {
                "total_jobs": len(jobs),
                "failed_jobs": len(failed_job_names),
                "total_validation_steps": len(all_validation_steps),
                "failed_steps_count": len(failed_validation_steps),
                "success_steps_count": len(all_validation_steps) - len(failed_validation_steps)
            },

            # Job-level tracking (for CI log matching)
            "all_job_names": all_job_names,
            "failed_job_names": failed_job_names,

            # Step-level tracking (detailed validation)
            "all_validation_steps": all_validation_steps,
            "failed_validation_steps": failed_validation_steps,

            # Detailed breakdown
            "jobs": job_details
        }

        return result

    def read_jsonl(self, file_path: Path) -> List[Dict]:
        """Read a JSONL file."""
        data = []
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            return data

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Skipping invalid JSON line: {e}")
        return data

    def process_all_results(self, input_files: List[str] = None) -> List[Dict]:
        """
        Process all result files and fetch validation steps.

        Args:
            input_files: List of JSONL file names to process.
                        If None, processes all jobs_*.jsonl files

        Returns:
            List of validation step data for all issues
        """
        if input_files is None:
            input_files = [
                'jobs_results_diff.jsonl',
                'jobs_failure_diff.jsonl',
                'jobs_success_diff.jsonl',
                'jobs_ids_diff.jsonl'
            ]

        all_results = []
        seen_ids = set()

        for filename in input_files:
            # Read from input_dir (dataset/)
            file_path = self.input_dir / filename
            if not file_path.exists():
                print(f"[SKIP] File not found: {file_path}")
                continue

            print(f"\n[INFO] Processing {filename} from {self.input_dir}")
            job_data_list = self.read_jsonl(file_path)

            for job_data in job_data_list:
                issue_id = job_data.get('id')

                # Skip duplicates
                if issue_id in seen_ids:
                    continue
                seen_ids.add(issue_id)

                # Skip if no URL (not pushed yet)
                if not job_data.get('url'):
                    print(f"[SKIP] ID {issue_id} has no workflow URL")
                    continue

                # Process this workflow run
                result = self.process_workflow_run(job_data)
                if result:
                    all_results.append(result)

                # Be nice to GitHub API
                time.sleep(0.5)

        return all_results

    def prepare_dataset_features(self, results: List[Dict]) -> List[Dict]:
        """
        Prepare data in NEW CLEAN STRUCTURE (4 columns).

        - overall_jobs: [{"job_name": str, "steps": [str]}]
        - failed_jobs: [{"job_name": str, "steps": [str]}]  (same structure, only failures)
        - total_jobs: int
        - total_steps: int
        """
        dataset_features = []

        for result in results:
            issue_id = result['id']

            # Build overall_jobs structure
            overall_jobs = []
            total_steps_count = 0

            for job_detail in result.get('jobs', []):
                job_name = job_detail['job_name']
                all_step_names = job_detail.get('all_step_names_short', [])

                overall_jobs.append({
                    "job_name": job_name,
                    "steps": all_step_names
                })
                total_steps_count += len(all_step_names)

            # Build failed_jobs structure (same format as overall_jobs, only failures)
            # IMPORTANT: If overall workflow succeeded, failed_jobs should be empty!
            failed_jobs = []

            overall_conclusion = result.get('overall_conclusion', 'unknown')
            if overall_conclusion != 'success':
                # Only build failed_jobs if workflow didn't succeed
                for job_detail in result.get('jobs', []):
                    job_name = job_detail['job_name']
                    failed_step_names = job_detail.get('failed_step_names_short', [])

                    # Only include jobs that have failed steps
                    if failed_step_names:
                        failed_jobs.append({
                            "job_name": job_name,
                            "steps": failed_step_names
                        })

            feature = {
                "id": issue_id,
                "sha_original": result.get('sha_original'),
                "commit": result.get('commit'),

                # NEW CLEAN STRUCTURE (4 columns)
                "overall_jobs": overall_jobs,
                "failed_jobs": failed_jobs,
                "total_jobs": len(overall_jobs),
                "total_steps": total_steps_count,

                # Extra metadata
                "workflow_run_url": result.get('workflow_run_url'),
                "overall_conclusion": result.get('overall_conclusion')
            }

            dataset_features.append(feature)

        return dataset_features

    def save_results(self, results: List[Dict], output_file: str = "validation_steps_detailed.json"):
        """Save results to JSON files."""
        output_path = self.results_dir / output_file

        # Save full detailed results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Saved {len(results)} detailed validation mappings to {output_path}")

        # Save dataset-ready features
        dataset_features = self.prepare_dataset_features(results)
        dataset_output = self.results_dir / "validation_features_for_dataset.json"

        with open(dataset_output, 'w', encoding='utf-8') as f:
            json.dump(dataset_features, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Saved {len(dataset_features)} dataset features to {dataset_output}")

        # Print summary statistics
        total_steps = sum(r['summary']['total_validation_steps'] for r in results)
        total_failed = sum(r['summary']['failed_steps_count'] for r in results)

        print(f"\n{'='*60}")
        print("VALIDATION STEPS SUMMARY")
        print('='*60)
        print(f"Total issues processed: {len(results)}")
        print(f"Total validation steps: {total_steps}")
        print(f"Total failed steps: {total_failed}")
        print(f"Overall success rate: {((total_steps - total_failed) / total_steps * 100):.1f}%")
        print('='*60)

        # Print sample of dataset features
        print(f"\n{'='*60}")
        print("SAMPLE VALIDATION DATA (NEW STRUCTURE - 4 columns)")
        print('='*60)
        if dataset_features:
            sample = dataset_features[0]
            print(f"ID: {sample['id']}")
            print(f"Commit: {sample['commit']}")

            print(f"\n1. overall_jobs: (total_jobs={sample['total_jobs']})")
            for job in sample['overall_jobs'][:2]:
                steps_preview = job['steps'][:3] + ['...'] if len(job['steps']) > 3 else job['steps']
                print(f"   - {job['job_name']}: {steps_preview}")
            if len(sample['overall_jobs']) > 2:
                print(f"   ... ({len(sample['overall_jobs']) - 2} more jobs)")

            print(f"\n2. failed_jobs:")
            if sample['failed_jobs']:
                for job in sample['failed_jobs']:
                    print(f"   - {job['job_name']}: {job['steps']}")
            else:
                print(f"   (none - all jobs passed!)")

            print(f"\n3. total_jobs: {sample['total_jobs']}")
            print(f"4. total_steps: {sample['total_steps']}")

            print(f"\nConclusion: {sample['overall_conclusion']}")
        print('='*60)


def main():
    """Main entry point."""
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config.yaml"

    if config_path.exists():
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        github_token = config.get("github_token") or config.get("GITHUB_TOKEN")
        benchmark_owner = config.get("benchmark_owner", "RabeyaMuna")
        results_dir = config.get("out_folder", "results")
    else:
        # Fallback to environment variables
        github_token = os.environ.get("GITHUB_TOKEN")
        benchmark_owner = os.environ.get("BENCHMARK_OWNER", "RabeyaMuna")
        results_dir = "results"

    if not github_token:
        print("[ERROR] GitHub token not found!")
        print("Set GITHUB_TOKEN in config.yaml or environment variable")
        return 1

    # Directories
    dataset_dir = str(Path(__file__).parent.parent.parent / "dataset")

    print("=" * 60)
    print("Validation Steps Fetcher")
    print("=" * 60)
    print(f"Input directory: {dataset_dir}")
    print(f"Output directory: {results_dir}")
    print(f"Benchmark owner: {benchmark_owner}")
    print(f"GitHub token: {github_token[:8]}...")
    print()
    print("Fetching validation steps from GitHub API...")
    print("(Will read from dataset/jobs_*.jsonl files)")
    print()

    # Initialize fetcher
    fetcher = ValidationStepsFetcher(
        github_token=github_token,
        results_dir=results_dir,
        benchmark_owner=benchmark_owner,
        input_dir=dataset_dir
    )

    # Process all results
    results = fetcher.process_all_results()

    # Save results
    if results:
        fetcher.save_results(results)
    else:
        print("[WARN] No results to save")

    return 0


if __name__ == "__main__":
    exit(main())
