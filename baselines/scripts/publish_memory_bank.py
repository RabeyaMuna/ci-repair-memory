#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "baselines" / "results" / "memory_seed_preparation"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish prepared L1/L2/L3 memory files into the baseline memory bank location.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = OmegaConf.load(str(args.config))
    target_root = Path(str(config.project_result_dir))
    target_root.mkdir(parents=True, exist_ok=True)

    copied = {}
    for name in ("failure_memory.json", "repo_memory.json", "cross_memory.json", "summary.json"):
        src = args.source_dir / name
        if not src.exists():
            continue
        dst = target_root / name
        shutil.copy2(src, dst)
        copied[name] = str(dst)

    manifest = {
        "source_dir": str(args.source_dir),
        "target_root": str(target_root),
        "copied": copied,
    }
    (target_root / "memory_bank_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
