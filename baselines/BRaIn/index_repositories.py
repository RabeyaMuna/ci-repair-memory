"""
Index repositories for BRaIn FL
Indexes source files from each sha_fail commit into Elasticsearch
"""
import json
import os
import subprocess
from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import argparse
from typing import Dict, Optional


class RepositoryIndexer:
    """Indexes repository files into Elasticsearch"""

    def __init__(self, es_host="localhost", es_port=9200):
        self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
        self.index_name = "ci_repair_repos"

    def is_indexed(self, sha_fail: str) -> bool:
        """Check if a commit is already indexed"""
        try:
            result = self.es.count(
                index=self.index_name,
                body={"query": {"term": {"sha_fail": sha_fail}}}
            )
            return result['count'] > 0
        except:
            return False

    def create_index(self):
        """Create Elasticsearch index with proper mappings"""
        mapping = {
            "mappings": {
                "properties": {
                    "sha_fail": {"type": "keyword"},
                    "file_path": {"type": "keyword"},
                    "file_content": {"type": "text", "analyzer": "standard"},
                    "file_extension": {"type": "keyword"},
                    "repo_name": {"type": "keyword"},
                    "project": {"type": "keyword"}
                }
            }
        }

        # Create index if doesn't exist (don't delete existing)
        if not self.es.indices.exists(index=self.index_name):
            print(f"Creating index: {self.index_name}")
            self.es.indices.create(index=self.index_name, body=mapping)
        else:
            print(f"Using existing index: {self.index_name}")

    def index_repository(self, repo_path: Path, sha_fail: str, entry_id: str):
        """Index all source files from a repository at specific commit"""
        print(f"  Indexing repo at {sha_fail[:7]}...")

        # Save current branch/commit for later restoration
        try:
            original_ref = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            ).stdout.strip()
        except:
            original_ref = None

        # Checkout commit
        try:
            subprocess.run(
                ["git", "checkout", "-f", sha_fail],
                cwd=repo_path,
                capture_output=True,
                check=True,
                timeout=30
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
            print(f"    [ERROR] Failed to checkout {sha_fail[:7]}: {stderr[:200]}")
            return 0
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] Checkout timed out for {sha_fail[:7]}")
            return 0

        # Find all source files
        extensions = [".py", ".txt", ".toml", ".md", ".rst", ".yaml", ".yml", ".json"]
        source_files = []

        for ext in extensions:
            result = subprocess.run(
                ["find", ".", "-name", f"*{ext}", "-type", "f"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
            source_files.extend(files)

        # Index files
        docs = []
        for file_path in source_files:
            try:
                full_path = repo_path / file_path.lstrip('./')

                # Skip if file doesn't exist or is broken symlink
                if not full_path.exists() or not full_path.is_file():
                    continue

                # Skip large files
                if full_path.stat().st_size > 1_000_000:  # 1MB limit
                    continue

                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                doc = {
                    "_index": self.index_name,
                    "_source": {
                        "sha_fail": sha_fail,
                        "file_path": str(file_path),
                        "file_content": content,
                        "file_extension": Path(file_path).suffix,
                        "repo_name": repo_path.name,
                        "project": "ci-repair",
                        "entry_id": entry_id
                    }
                }
                docs.append(doc)

            except Exception as e:
                # Skip files that can't be read (broken symlinks, permissions, etc.)
                continue

        # Bulk index
        if docs:
            try:
                success, _ = bulk(self.es, docs)
                print(f"    [OK] Indexed {success} files")
            except Exception as e:
                print(f"    [ERROR] Bulk index failed: {e}")
                success = 0
        else:
            print(f"    [WARN]  No files to index")
            success = 0

        # Restore original branch/commit
        if original_ref and original_ref != "HEAD":
            try:
                subprocess.run(
                    ["git", "checkout", "-f", original_ref],
                    cwd=repo_path,
                    capture_output=True,
                    timeout=30
                )
            except:
                pass  # Non-critical, just leaves repo at indexed commit

        return success


def load_dataset_mapping(base_dir: Path) -> Dict[str, str]:
    """
    Load mapping of sha_fail -> repo_name from dataset

    Returns:
        Dict mapping commit SHA to repo name
    """
    dataset_file = base_dir / "dataset" / "validation_steps_detailed.json"

    if not dataset_file.exists():
        print(f"[WARN]  Dataset not found at {dataset_file}")
        return {}

    try:
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)

        # Create mapping: commit -> repo_name
        mapping = {}
        for entry in dataset:
            commit = entry.get("commit", "")
            repo_name = entry.get("repo_name", "")
            if commit and repo_name:
                mapping[commit] = repo_name

        print(f"[LOAD] Loaded {len(mapping)} commit->repo mappings from dataset")
        return mapping
    except Exception as e:
        print(f"[WARN]  Failed to load dataset mapping: {e}")
        return {}


def clone_missing_repo(repo_name: str, base_dir: Path, github_org: str = "RabeyaMuna") -> Optional[Path]:
    """
    Clone a missing repository

    Args:
        repo_name: Name of repository to clone
        base_dir: Base directory
        github_org: GitHub organization/user

    Returns:
        Path to cloned repo, or None if failed
    """
    repos_dir = base_dir / "baselines" / "repo_cloned"
    repo_path = repos_dir / repo_name

    if repo_path.exists():
        return repo_path

    print(f"  [FETCH] Cloning missing repository: {repo_name}")

    repo_url = f"https://github.com/{github_org}/{repo_name}.git"

    try:
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", repo_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"  [OK] Cloned {repo_name} successfully")
            return repo_path
        else:
            print(f"  [ERROR] Failed to clone {repo_name}: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Clone timed out for {repo_name}")
        return None
    except Exception as e:
        print(f"  [ERROR] Clone error for {repo_name}: {e}")
        return None


def fetch_repo_commits(repo_path: Path) -> bool:
    """
    Fetch all commits from a repository

    Args:
        repo_path: Path to repository

    Returns:
        True if successful
    """
    try:
        subprocess.run(
            ["git", "fetch", "--all", "--tags", "--prune", "--quiet"],
            cwd=repo_path,
            timeout=120,
            capture_output=True
        )
        return True
    except:
        return False


def get_repo_path(sha_fail: str, base_dir: Path = None,
                  commit_to_repo: Dict[str, str] = None,
                  auto_clone: bool = True) -> Path:
    """
    Map sha_fail to repository path by searching through all repos

    Args:
        sha_fail: The commit SHA to find
        base_dir: Base directory containing repos (default: auto-detect)

    Returns:
        Path to the repository containing the commit

    Raises:
        FileNotFoundError: If no repository contains the commit
    """
    # Determine repos directory
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    # Try common repo locations
    possible_dirs = [
        base_dir / "repo_cloned",
        base_dir / "repo",
        base_dir.parent / "repo"
    ]

    repos_dir = None
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            repos_dir = dir_path
            break

    if repos_dir is None:
        raise FileNotFoundError(
            f"Repository directory not found. Tried: {', '.join(str(d) for d in possible_dirs)}"
        )

    # Check all repos for this commit
    checked = 0
    for repo_dir in sorted(repos_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
            continue

        checked += 1
        try:
            result = subprocess.run(
                ["git", "cat-file", "-t", sha_fail],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10  # Increased timeout
            )
            if result.returncode == 0:
                return repo_dir
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    raise FileNotFoundError(
        f"Repository for commit {sha_fail[:7]} not found "
        f"(checked {checked} repos in {repos_dir})"
    )


def main():
    parser = argparse.ArgumentParser(description="Index repositories for BRaIn FL")
    parser.add_argument("--log-details", default=None, help="Path to log_details.json")
    parser.add_argument("--model", default="gpt-5-mini_llm", help="Model name (e.g., gpt-5-mini_llm)")
    parser.add_argument("--limit", type=int, default=150, help="Number of repos to index")
    parser.add_argument("--skip-indexed", action="store_true", default=True, help="Skip already indexed commits (default: True)")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing of all commits")
    parser.add_argument("--no-auto-clone", action="store_true", help="Disable automatic cloning of missing repos")

    args = parser.parse_args()

    print("[START] Repository Indexing for BRaIn FL")
    print("=" * 50)

    # Determine base directory
    base_dir = Path(__file__).parent.parent.parent

    # Load dataset mapping for auto-cloning
    auto_clone = not args.no_auto_clone
    commit_to_repo = {}

    if auto_clone:
        commit_to_repo = load_dataset_mapping(base_dir)
        if commit_to_repo:
            print(f"[OK] Auto-clone enabled ({len(commit_to_repo)} repos mapped)")
        else:
            print("[WARN]  Auto-clone enabled but no dataset mapping loaded")
    else:
        print("[WARN]  Auto-clone disabled")

    # Determine log_details path
    if args.log_details:
        log_details_path = args.log_details
    else:
        log_details_path = f"../results/{args.model}/log_details.json"

    # Load log_details
    print(f"[LOAD] Loading {log_details_path}...")
    if not os.path.exists(log_details_path):
        print(f"[ERROR] File not found: {log_details_path}")
        return 1

    with open(log_details_path) as f:
        log_details = json.load(f)

    print(f"   Total entries: {len(log_details)}")
    print(f"   Indexing first: {args.limit}")
    print()

    # Initialize indexer
    indexer = RepositoryIndexer()

    # Create index (don't delete existing)
    indexer.create_index()

    # Count what needs indexing
    skip_indexed = args.skip_indexed and not args.force_reindex

    if skip_indexed:
        print("[CHECK] Checking which commits are already indexed...")

    # Index repositories
    total_files = 0
    failed = 0
    skipped = 0

    for i, entry in enumerate(log_details[:args.limit]):
        entry_id = entry.get("id", str(i))
        sha_fail = entry.get("sha_fail", "")

        print(f"[{i+1}/{args.limit}] Entry ID={entry_id}, SHA={sha_fail[:7]}")

        # Skip if already indexed
        if skip_indexed and indexer.is_indexed(sha_fail):
            print(f"  [SKIP] Already indexed")
            skipped += 1
            continue

        try:
            repo_path = get_repo_path(
                sha_fail,
                base_dir=base_dir,
                commit_to_repo=commit_to_repo,
                auto_clone=auto_clone
            )
            print(f"  Found repo: {repo_path}")

            num_files = indexer.index_repository(repo_path, sha_fail, entry_id)
            total_files += num_files

        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            failed += 1
            continue
        except Exception as e:
            print(f"  [ERROR] Error: {e}")
            failed += 1
            continue

    print()
    print("=" * 50)
    print(f"[OK] Indexing Complete!")
    print(f"   Total files indexed: {total_files}")
    print(f"   Skipped (already indexed): {skipped}/{args.limit}")
    print(f"   Failed: {failed}/{args.limit}")
    print(f"   Newly indexed: {args.limit - skipped - failed}/{args.limit}")
    print()
    print("Next: python run_brain_fl_full.py gpt-5-mini_llm --limit 10")


if __name__ == "__main__":
    main()
