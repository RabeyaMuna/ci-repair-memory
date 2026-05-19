#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = ROOT / "baselines"
if str(BASELINES_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINES_DIR))

from utilities.load_config import load_config
from utilities.memory_plugin import MemoryPlugin


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def changed_file_payload(changed_files_dir: Path, sha_fail: str):
    path = changed_files_dir / f"{sha_fail}.json"
    if not path.exists():
        return {"changed_files": []}
    return load_json(path)


def main() -> int:
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "baselines" / "results" / "gpt-5-mini_llm_baseline")
    config = load_config()
    plugin = MemoryPlugin(config=config, result_dir=str(result_dir))
    plugin.enabled = True

    log_details_path = result_dir / "log_details.json"
    fl_path = result_dir / "fault_localization.json"
    patch_path = result_dir / "generated_patches.json"
    changed_dir = Path(str(config.changed_files_folder))

    if not log_details_path.exists() or not patch_path.exists():
        raise FileNotFoundError("Expected log_details.json and generated_patches.json in result_dir")

    log_details = {item.get("sha_fail"): item for item in load_json(log_details_path)}
    fl_records = {}
    if fl_path.exists():
        fl_records = {item.get("sha_fail"): item for item in load_json(fl_path)}
    patch_records = load_json(patch_path)

    existing = {item.get("sha_fail") for item in plugin.memories}
    added = 0
    for patch in patch_records:
        sha_fail = patch.get("sha_fail")
        if not sha_fail or sha_fail in existing:
            continue
        log_item = log_details.get(sha_fail)
        if not log_item:
            continue
        plugin.save_memory_entry(
            task_id=str(log_item.get("id", "")),
            sha_fail=sha_fail,
            repo_name=str(log_item.get("repo_name", "")),
            repo_owner=str(log_item.get("repo_owner", "")),
            workflow_path=str(log_item.get("workflow_path", "")),
            workflow=str(log_item.get("workflow", "")),
            log_analysis_result=log_item,
            changed_files_info=changed_file_payload(changed_dir, sha_fail),
            fault_localizer=fl_records.get(sha_fail),
            patch_generator=patch,
        )
        existing.add(sha_fail)
        added += 1

    print(f"[memory] store={plugin.store_path}")
    print(f"[memory] added_entries={added}")
    print(f"[memory] total_entries={len(plugin.memories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
