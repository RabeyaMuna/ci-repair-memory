#!/usr/bin/env python3
"""
Unified Workflow Manager for CI Benchmark Data Management

Orchestrates all data management operations:
- Setup permanent benchmark branches
- Fetch metadata (with auto-trigger if needed)
- Monitor CI health
- Update dataset

Usage:
    python workflow_manager.py setup                # One-time setup
    python workflow_manager.py fetch-metadata       # Fetch/update metadata
    python workflow_manager.py monitor              # Monitor CI health
    python workflow_manager.py update-dataset       # Update dataset
    python workflow_manager.py run-all              # Full pipeline
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class WorkflowManager:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.dataset_dir = self.base_dir.parent / "dataset"

        # Script paths
        self.setup_script = self.base_dir / "setup_benchmark_branches.py"
        self.fetch_script = self.base_dir / "fetch_commit_metadata.py"
        self.trigger_script = self.base_dir / "trigger_ci_for_commits.py"
        self.update_dataset_script = self.base_dir / "update_failed_logs.py"

        # Result files
        self.branches_file = self.results_dir / "branches" / "benchmark_branches.json"
        self.metadata_file = self.results_dir / "metadata" / "commit_job_metadata.json"
        self.missing_ids_file = self.results_dir / "metadata" / "missing_metadata_ids.json"

    def run_command(self, cmd: list, description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n{'='*80}")
        print(f"🚀 {description}")
        print(f"{'='*80}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def check_prerequisites(self) -> bool:
        """Check if required files exist."""
        dataset_file = self.dataset_dir / "lca_dataset.parquet"

        if not dataset_file.exists():
            print(f"❌ Dataset not found: {dataset_file}")
            return False

        print(f"✓ Dataset found: {dataset_file}")
        return True

    def setup_branches(self) -> bool:
        """Setup permanent benchmark branches."""
        if not self.setup_script.exists():
            print(f"❌ Setup script not found: {self.setup_script}")
            return False

        return self.run_command(
            ["python", str(self.setup_script)],
            "Setting up permanent benchmark branches"
        )

    def fetch_metadata(self, ids: Optional[list] = None) -> bool:
        """Fetch commit metadata."""
        if not self.fetch_script.exists():
            print(f"❌ Fetch script not found: {self.fetch_script}")
            return False

        return self.run_command(
            ["python", str(self.fetch_script)],
            "Fetching commit metadata"
        )

    def trigger_missing_commits(self) -> bool:
        """Trigger CI for commits without metadata."""
        if not self.trigger_script.exists():
            print(f"❌ Trigger script not found: {self.trigger_script}")
            return False

        if not self.missing_ids_file.exists():
            print("✓ No missing metadata - skipping trigger")
            return True

        # Check if there are any missing IDs
        with open(self.missing_ids_file) as f:
            missing_ids = json.load(f)

        if not missing_ids:
            print("✓ No missing metadata - skipping trigger")
            return True

        print(f"Found {len(missing_ids)} issues with missing metadata")

        return self.run_command(
            ["python", str(self.trigger_script)],
            "Triggering CI for missing commits"
        )

    def update_dataset(self) -> bool:
        """Update dataset with metadata."""
        if not self.update_dataset_script.exists():
            print(f"❌ Update script not found: {self.update_dataset_script}")
            return False

        if not self.metadata_file.exists():
            print(f"❌ Metadata file not found: {self.metadata_file}")
            print("   Run 'fetch-metadata' first")
            return False

        return self.run_command(
            ["python", str(self.update_dataset_script)],
            "Updating dataset with metadata"
        )

    def monitor_ci_health(self) -> bool:
        """Monitor CI workflow health."""
        print("\n{'='*80}")
        print("🔍 CI Health Monitoring")
        print("{'='*80}")
        print("\n⚠️  Monitoring feature coming soon!")
        print("   This will check:")
        print("   - Workflow still exists")
        print("   - Same failure patterns")
        print("   - Ground truth validation")
        return True

    def status(self):
        """Show current status of all components."""
        print(f"\n{'='*80}")
        print("📊 CI Benchmark Data Management Status")
        print(f"{'='*80}\n")

        # Check branches
        if self.branches_file.exists():
            with open(self.branches_file) as f:
                branches = json.load(f)
            created = sum(1 for b in branches if b.get("status") == "created")
            print(f"✓ Benchmark Branches: {created}/{len(branches)} created")
        else:
            print("❌ Benchmark Branches: Not setup (run 'setup')")

        # Check metadata
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                metadata = json.load(f)
            with_meta = sum(1 for issue in metadata if any(
                len(c.get('metadata', [])) > 0 for c in issue.get('commits', [])
            ))
            print(f"✓ Metadata: {len(metadata)} issues, {with_meta} with metadata")
        else:
            print("❌ Metadata: Not fetched (run 'fetch-metadata')")

        # Check missing IDs
        if self.missing_ids_file.exists():
            with open(self.missing_ids_file) as f:
                missing = json.load(f)
            print(f"⚠️  Missing Metadata: {len(missing)} issues need triggering")
        else:
            print("❓ Missing IDs: No data yet")

        # Check dataset
        dataset_file = self.dataset_dir / "lca_dataset.parquet"
        if dataset_file.exists():
            import pandas as pd
            df = pd.read_parquet(dataset_file)
            has_logs = 'logs' in df.columns
            print(f"✓ Dataset: {len(df)} issues, logs column: {has_logs}")
        else:
            print("❌ Dataset: Not found")

        print(f"\n{'='*80}\n")

    def run_all(self) -> bool:
        """Run complete pipeline."""
        print("\n" + "="*80)
        print("🚀 Running Complete CI Benchmark Pipeline")
        print("="*80 + "\n")

        steps = [
            ("Prerequisites", lambda: self.check_prerequisites()),
            ("Setup Branches", lambda: self.setup_branches()),
            ("Fetch Metadata", lambda: self.fetch_metadata()),
            ("Trigger Missing", lambda: self.trigger_missing_commits()),
            ("Re-fetch Metadata", lambda: self.fetch_metadata()),
            ("Update Dataset", lambda: self.update_dataset()),
        ]

        for step_name, step_func in steps:
            print(f"\n{'─'*80}")
            print(f"Step: {step_name}")
            print(f"{'─'*80}")

            success = step_func()

            if not success:
                print(f"\n❌ Pipeline failed at: {step_name}")
                return False

            print(f"✓ {step_name} completed")

        print(f"\n{'='*80}")
        print("✅ Complete pipeline finished successfully!")
        print(f"{'='*80}\n")

        # Show final status
        self.status()
        return True


def print_usage():
    """Print usage information."""
    print("""
CI Benchmark Workflow Manager

Usage:
    python workflow_manager.py <command>

Commands:
    setup               Setup permanent benchmark branches (one-time)
    fetch-metadata      Fetch commit metadata for all issues
    trigger-missing     Trigger CI for commits without metadata
    update-dataset      Update dataset with collected metadata
    monitor             Monitor CI workflow health
    status              Show current status
    run-all             Run complete pipeline

Examples:
    # Initial setup
    python workflow_manager.py setup
    python workflow_manager.py fetch-metadata

    # Update metadata
    python workflow_manager.py fetch-metadata

    # Trigger missing and update
    python workflow_manager.py trigger-missing
    python workflow_manager.py fetch-metadata
    python workflow_manager.py update-dataset

    # Full pipeline
    python workflow_manager.py run-all

    # Check status
    python workflow_manager.py status
    """)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()
    manager = WorkflowManager()

    commands = {
        "setup": manager.setup_branches,
        "fetch-metadata": manager.fetch_metadata,
        "trigger-missing": manager.trigger_missing_commits,
        "update-dataset": manager.update_dataset,
        "monitor": manager.monitor_ci_health,
        "status": manager.status,
        "run-all": manager.run_all,
        "help": print_usage,
    }

    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print_usage()
        return

    # Execute command
    if command == "status":
        commands[command]()
    elif command == "help":
        commands[command]()
    else:
        success = commands[command]()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
