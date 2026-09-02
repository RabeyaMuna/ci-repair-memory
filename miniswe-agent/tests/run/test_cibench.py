from types import SimpleNamespace
from unittest.mock import patch

import json

from minisweagent.run.benchmarks.cibench import (
    attach_fault_localization,
    format_fault_localization,
    load_fault_localization,
    process_instance,
)


def test_fault_localization_is_loaded_and_attached_by_sha(tmp_path):
    fl_path = tmp_path / "fault_localization.json"
    fl_path.write_text(
        json.dumps([{"sha_fail": "abc", "fault_localization_data": []}]),
        encoding="utf-8",
    )
    instances = [{"instance_id": "issue-1", "sha_fail": "abc"}]

    matched = attach_fault_localization(instances, load_fault_localization(fl_path))

    assert matched == 1
    assert instances[0]["_fault_localization"]["sha_fail"] == "abc"


def test_fault_localization_prompt_omits_machine_specific_full_path():
    rendered = format_fault_localization(
        {
            "fault_localization_data": [
                {
                    "file_path": "src/app.py",
                    "full_file_path": "/old/machine/src/app.py",
                    "faults": [
                        {
                            "line_range": [10, 12],
                            "fault_localization_level": "method",
                            "reason": "likely failing method",
                            "code_snippet": "def broken(): pass",
                        }
                    ],
                }
            ]
        }
    )

    assert "src/app.py:10-12" in rendered
    assert "likely failing method" in rendered
    assert "/old/machine" not in rendered


def test_direct_fault_localization_mode_skips_context_llm(tmp_path):
    instance = {
        "id": "issue-1",
        "sha_fail": "abc",
        "repo_owner": "owner",
        "repo_name": "repo",
        "_fault_localization": {
            "sha_fail": "abc",
            "fault_localization_data": [],
        },
    }
    progress = SimpleNamespace(
        on_instance_start=lambda *_: None,
        update_instance_status=lambda *_: None,
        on_instance_end=lambda *_: None,
    )
    config = {"model": {"model_name": "gpt-5-mini"}}

    with (
        patch(
            "minisweagent.run.benchmarks.cibench.setup_local_environment",
            side_effect=RuntimeError("stop before repair agent"),
        ),
    ):
        process_instance(
            instance,
            tmp_path,
            config,
            progress,
            repair_model="gpt-5-mini",
            fault_localization_path="fault_localization.json",
        )
