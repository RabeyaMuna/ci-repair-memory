"""Path configuration for shared resources.

This module provides centralized path management for the multi-agent benchmark.
All paths are relative to the project root, which is one level up from miniswe-agent/.
"""

from pathlib import Path

# Project root is one level up from miniswe-agent/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
REPO_ROOT = PROJECT_ROOT / "repo"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"


def get_memory_root():
    """Get shared memory directory.

    Returns:
        Path: Shared memory directory (data/trs/)
    """
    return DATA_ROOT / "trs"


def get_results_dir(agent_name: str, model_name: str, ablation_level: str):
    """Get results directory for specific configuration.

    Args:
        agent_name: Agent scaffold name (e.g., 'miniswe-agent')
        model_name: Model name ('minimax', 'glm', 'kimi', etc.)
        ablation_level: Memory ablation level ('baseline', 'L1', 'L1_L2', 'L1_L2_L3')

    Returns:
        Path: Results directory (created if doesn't exist)

    Example:
        >>> get_results_dir("miniswe-agent", "minimax", "L1_L2_L3")
        Path('.../results/miniswe-agent/minimax/L1_L2_L3')
    """
    results_dir = RESULTS_ROOT / agent_name / model_name / ablation_level
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_repo_dir(repo_identifier: str):
    """Get testbed repository directory.

    Args:
        repo_identifier: Repository identifier (e.g., 'owner__repo')

    Returns:
        Path: Repository directory
    """
    return REPO_ROOT / repo_identifier


def get_data_root():
    """Get shared data directory.

    Returns:
        Path: Data root directory
    """
    return DATA_ROOT


def get_repo_root():
    """Get shared repository root.

    Returns:
        Path: Repository root directory
    """
    return REPO_ROOT


def get_scripts_root():
    """Get shared scripts directory.

    Returns:
        Path: Scripts root directory
    """
    return SCRIPTS_ROOT


# Backward compatibility aliases
def get_project_root():
    """Get project root directory.

    Returns:
        Path: Project root directory
    """
    return PROJECT_ROOT
