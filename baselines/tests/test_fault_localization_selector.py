import json
from types import SimpleNamespace

from ci_repair.fault_localization import FaultLocalization


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        payload = [
            {"file_path": "src/changed.py", "is_suspicious": True},
            {"file_path": "src/other.py", "is_suspicious": False},
        ]
        return SimpleNamespace(content=json.dumps(payload))


def test_selector_batches_multiple_changed_files_in_one_call():
    selector = FaultLocalization.__new__(FaultLocalization)
    selector.relevant_files = [{"file": "src/from_log.py"}]
    selector.changed_files_info = {
        "changed_files": [
            {"file_path": "src/from_log.py", "diff": "already selected"},
            {"file_path": "src/changed.py", "diff": "+ broken change"},
            {"file_path": "src/other.py", "diff": "+ unrelated change"},
        ]
    }
    selector.failed_jobs = [{"job": "lint", "step": "ruff"}]
    selector.model_name = "gpt-5-mini"
    selector.llm = RecordingLLM()

    selected = selector.select_suspecious_files()

    assert selected == [{"file": "src/from_log.py"}, {"file": "src/changed.py"}]
    assert len(selector.llm.prompts) == 1
    assert "src/changed.py" in selector.llm.prompts[0]
    assert "src/other.py" in selector.llm.prompts[0]
