#!/usr/bin/env python3
"""
Automated BRaIn FL Pipeline
Handles missing repos, fetching, indexing, and FL execution
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import sys


class AutoBRaInFL:
    """Automated BRaIn FL pipeline"""

    def __init__(self, model_name: str, limit: int = 150):
        self.model_name = model_name
        self.limit = limit

        # Paths
        self.base_dir = Path(__file__).parent.parent.parent
        self.repo_dir = self.base_dir / "baselines" / "repo_cloned"
        self.log_details_path = (
            self.base_dir / "baselines" / "results" / model_name / "log_details.json"
        )

        # GitHub org for cloning
        self.github_org = "RabeyaMuna"

    def load_log_details(self) -> List[Dict]:
        """Load log_details.json"""
        print(f"[LOAD] Loading log_details from {self.model_name}...")

        if not self.log_details_path.exists():
            print(f"[ERROR] File not found: {self.log_details_path}")
            sys.exit(1)

        with open(self.log_details_path) as f:
            data = json.load(f)

        print(f"   Total entries: {len(data)}")
        print(f"   Processing first: {self.limit}")
        return data[:self.limit]

    def find_repo_for_commit(self, sha: str) -> str:
        """Find which repo contains a commit"""
        for repo_path in self.repo_dir.iterdir():
            if not repo_path.is_dir() or repo_path.name.startswith('.'):
                continue

            try:
                result = subprocess.run(
                    ["git", "cat-file", "-t", sha],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return repo_path.name
            except:
                continue

        return None

    def identify_repos_needed(
        self,
        log_details: List[Dict]
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Identify which repos are needed and which commits are missing

        Returns:
            (repo_to_commits, missing_commits)
        """
        print(f"\n🔍 Identifying required repositories...")

        repo_to_commits = {}  # repo_name -> [commits]
        missing_commits = []

        for i, entry in enumerate(log_details):
            sha_fail = entry.get("sha_fail", "")
            if not sha_fail:
                continue

            if (i + 1) % 20 == 0:
                print(f"   Checked {i+1}/{len(log_details)}...")

            # Use repo_name from entry if available (enriched data)
            repo_name = entry.get("repo_name")

            # If not in entry, search for it
            if not repo_name:
                repo_name = self.find_repo_for_commit(sha_fail)

            if repo_name:
                if repo_name not in repo_to_commits:
                    repo_to_commits[repo_name] = []
                repo_to_commits[repo_name].append(sha_fail)
            else:
                missing_commits.append(sha_fail)

        print(f"\n[OK] Found {len(repo_to_commits)} repos with commits")
        print(f"[WARN]  Missing {len(missing_commits)} commits")

        return repo_to_commits, missing_commits

    def try_fetch_missing_commits(
        self,
        missing_commits: List[str]
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Try to fetch missing commits by running git fetch on all repos

        Returns:
            (newly_found_repo_to_commits, still_missing)
        """
        if not missing_commits:
            return {}, []

        print(f"\n[FETCH] Fetching commits to find missing {len(missing_commits)} commits...")

        # Fetch all repos
        for i, repo_path in enumerate(self.repo_dir.iterdir()):
            if not repo_path.is_dir() or repo_path.name.startswith('.'):
                continue

            print(f"   [{i+1}] Fetching {repo_path.name}...")

            try:
                subprocess.run(
                    ["git", "fetch", "--all", "--tags", "--prune", "--quiet"],
                    cwd=repo_path,
                    timeout=120,
                    capture_output=True
                )
            except:
                pass

        # Re-check missing commits
        print(f"\n🔍 Re-checking missing commits...")
        newly_found = {}
        still_missing = []

        for sha in missing_commits:
            repo_name = self.find_repo_for_commit(sha)
            if repo_name:
                if repo_name not in newly_found:
                    newly_found[repo_name] = []
                newly_found[repo_name].append(sha)
            else:
                still_missing.append(sha)

        print(f"[OK] Found {len(missing_commits) - len(still_missing)} commits after fetch")
        print(f"[WARN]  Still missing: {len(still_missing)} commits")

        return newly_found, still_missing

    def get_repo_name_from_commit(self, sha: str) -> str:
        """
        Try to guess repo name from commit
        This is a fallback - we can't always determine this
        """
        # Would need additional metadata or API calls
        # For now, return None
        return None

    def clone_missing_repos(self, repo_names: List[str]) -> Dict[str, bool]:
        """Clone missing repositories"""
        print(f"\n[FETCH] Cloning {len(repo_names)} missing repositories...")

        results = {}

        for i, repo_name in enumerate(repo_names, 1):
            repo_url = f"https://github.com/{self.github_org}/{repo_name}.git"
            repo_path = self.repo_dir / repo_name

            if repo_path.exists():
                print(f"[{i}/{len(repo_names)}] [OK] {repo_name} already exists")
                results[repo_name] = True
                continue

            print(f"[{i}/{len(repo_names)}] Cloning {repo_name}...")

            try:
                result = subprocess.run(
                    ["git", "clone", "--filter=blob:none", repo_url, str(repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    print(f"   [OK] Cloned successfully")
                    results[repo_name] = True
                else:
                    print(f"   [ERROR] Failed: {result.stderr[:150]}")
                    results[repo_name] = False

            except subprocess.TimeoutExpired:
                print(f"   [ERROR] Timed out")
                results[repo_name] = False
            except Exception as e:
                print(f"   [ERROR] Error: {e}")
                results[repo_name] = False

        return results

    def run_indexing(self) -> bool:
        """Run the indexing script"""
        print(f"\n🔍 Indexing repositories into Elasticsearch...")
        print(f"   This may take 20-40 minutes for {self.limit} entries...")

        script_dir = self.base_dir / "baselines" / "BRaIn"
        script = script_dir / "index_repositories.py"

        cmd = [
            "python", str(script),
            "--model", self.model_name,
            "--limit", str(self.limit)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=script_dir,
                timeout=3600  # 1 hour
            )

            if result.returncode == 0:
                print("[OK] Indexing completed successfully")
                return True
            else:
                print(f"[WARN]  Indexing completed with warnings (code: {result.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print("[ERROR] Indexing timed out")
            return False
        except Exception as e:
            print(f"[ERROR] Indexing error: {e}")
            return False

    def run_brain_fl(self) -> bool:
        """Run BRaIn FL"""
        print(f"\n🧠 Running BRaIn Fault Localization...")

        script_dir = self.base_dir / "baselines" / "BRaIn"
        script = script_dir / "run_brain_fl_full.py"

        cmd = [
            "python", str(script),
            self.model_name,
            "--limit", str(self.limit)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=script_dir,
                timeout=7200  # 2 hours
            )

            if result.returncode == 0:
                print("[OK] BRaIn FL completed successfully")
                return True
            else:
                print(f"[WARN]  BRaIn FL completed with warnings (code: {result.returncode})")
                return False

        except subprocess.TimeoutExpired:
            print("[ERROR] BRaIn FL timed out")
            return False
        except Exception as e:
            print(f"[ERROR] BRaIn FL error: {e}")
            return False

    def run(self, skip_index: bool = False, skip_fl: bool = False):
        """Run the complete pipeline"""
        print("=" * 70)
        print("🧠 Automated BRaIn FL Pipeline")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Limit: {self.limit}")
        print(f"Repo dir: {self.repo_dir}")
        print()

        # Step 1: Load log_details
        log_details = self.load_log_details()

        # Step 2: Identify repos needed
        repo_to_commits, missing_commits = self.identify_repos_needed(log_details)

        print(f"\n[STATS] Summary:")
        print(f"   Repos with commits: {len(repo_to_commits)}")
        print(f"   Total commits found: {sum(len(c) for c in repo_to_commits.values())}")
        print(f"   Missing commits: {len(missing_commits)}")

        # Step 3: Try fetching to find missing commits
        if missing_commits:
            newly_found, still_missing = self.try_fetch_missing_commits(missing_commits)

            # Update repo_to_commits with newly found
            for repo, commits in newly_found.items():
                if repo not in repo_to_commits:
                    repo_to_commits[repo] = []
                repo_to_commits[repo].extend(commits)

            if still_missing:
                print(f"\n[WARN]  Warning: {len(still_missing)} commits still not found")
                print(f"   First 5: {', '.join(s[:7] for s in still_missing[:5])}")
                print(f"   These entries will be skipped during indexing")

        # Step 4: Run indexing
        if not skip_index:
            success = self.run_indexing()
            if not success:
                print("\n[ERROR] Indexing failed. Fix issues before running FL.")
                return 1
        else:
            print("\n⏭️  Skipping indexing")

        # Step 5: Run BRaIn FL
        if not skip_fl:
            success = self.run_brain_fl()
            if not success:
                print("\n[WARN]  BRaIn FL had issues")
                return 1
        else:
            print("\n⏭️  Skipping BRaIn FL")

        print("\n" + "=" * 70)
        print("[OK] Pipeline complete!")
        print("=" * 70)
        print(f"\nResults saved to:")
        print(f"  baselines/results/{self.model_name}/brain_fl_results.json")

        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated BRaIn FL Pipeline"
    )
    parser.add_argument(
        "model_name",
        help="Model name (e.g., gpt-5-mini_llm)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Number of entries to process (default: 150)"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing step"
    )
    parser.add_argument(
        "--skip-fl",
        action="store_true",
        help="Skip BRaIn FL step"
    )

    args = parser.parse_args()

    pipeline = AutoBRaInFL(args.model_name, args.limit)
    return pipeline.run(
        skip_index=args.skip_index,
        skip_fl=args.skip_fl
    )


if __name__ == "__main__":
    sys.exit(main())
