import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf


DATASET_REPO_ID = "ci-benchmark-user/ci-repair-bench"
DATASET_FILENAME = "ci_repair_dataset.parquet"


def _token_from_config(project_root: Path) -> Optional[str]:
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        return None
    config = OmegaConf.load(str(config_path))
    return os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")


def get_ci_repair_dataset_path(project_root: Path) -> str:
    load_dotenv(project_root / ".env")
    token = os.getenv("HF_TOKEN") or _token_from_config(project_root)
    return hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=DATASET_FILENAME,
        repo_type="dataset",
        token=token,
    )


if __name__ == "__main__":
    print(get_ci_repair_dataset_path(Path(__file__).resolve().parents[1]))
