from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_project_env(start_dir: str | Path | None = None) -> Path | None:
    """Load the nearest project .env without overriding existing shell vars."""
    current = Path(start_dir or os.getcwd()).resolve()
    for candidate_dir in (current, *current.parents):
        dotenv_path = candidate_dir / ".env"
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path, override=False)
            return dotenv_path
    return None
