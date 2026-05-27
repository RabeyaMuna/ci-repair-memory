import os
from pathlib import Path

from omegaconf import OmegaConf, open_dict

# Project root = three levels up from this file:
# baselines/utilities/load_config.py -> baselines/utilities -> baselines -> CI-REPAIR-BENCH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_PATHS = {
    "repos_folder": "repo",
    "out_folder": "results",
    "data_cache_dir": "dataset",
    "baseline_repo_folder": "baselines/repo_cloned",
    "baseline_output_folder": "baselines/results",
    "exception_dir": "baselines/exceptions",
    "changed_files_folder": "baselines/changed_files",
    "base_dir": ".",
    "project_result_dir": "baselines/results",
    "memory_retrieval_log_path": "baselines/results/memory_retrieval_log.jsonl",
    "memory_chroma_dir": "baselines/results/chroma_memory",
}


def _is_placeholder(value: str) -> bool:
    return value.startswith("<") and value.endswith(">")


def _resolve_project_path(value: str | os.PathLike | None) -> str | None:
    if value is None:
        return None

    raw = str(value).strip()
    if not raw or _is_placeholder(raw):
        return raw

    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def normalize_config_paths(config):
    """
    Make path-like config values portable across clones.

    Relative values in config.yaml are resolved from the repository root rather
    than the current shell directory. Missing path keys get repo-relative
    defaults so a fresh dev-server clone does not need machine-specific paths.
    """
    with open_dict(config):
        for key, default_value in DEFAULT_PATHS.items():
            value = config.get(key)
            if value is None or str(value).strip() == "":
                value = default_value
            resolved = _resolve_project_path(value)
            if resolved is not None:
                config[key] = resolved
    return config


def load_config(config_path: str | Path | None = None):
    """
    Load the Hydra/OmegaConf config for CI-REPAIR-BENCH.

    - By default, loads `<PROJECT_ROOT>/config.yaml`.
    - Optionally accepts an explicit config_path override.
    """
    if config_path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    return normalize_config_paths(OmegaConf.load(str(path)))
