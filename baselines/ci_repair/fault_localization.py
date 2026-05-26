import os
import json
import subprocess
import sys
import math
import re
import yaml
import time
from pathlib import Path
import demjson3
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from utilities.model_token_limits import get_prompt_token_budget
from utilities.load_config import load_config
from utilities.snippet_extractor import extract_snippet_from_line_range, find_line_range
from utilities.symbols_outline import build_outline, format_outline
from utilities.chunking_logic import chunk_log_by_tokens, chunk_lines_with_overlap, estimate_tokens
from utilities.model_token_limits import is_large_context_model


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

api_key = os.getenv("OPENAI_API_KEY")


class FaultLocalization:
    def __init__(
        self,
        sha_fail: str,
        repo_path: str,
        error_logs: dict,
        workflow: str,
        llm: ChatOpenAI,
        model_name: Optional[str] = None,
        changed_files_info: Optional[dict] = None,
        memory_plugin: Optional[Any] = None,
        memory_context: Optional[Dict[str, Any]] = None,
    ):
        """
        FaultLocalization agent.

        changed_files_info format (from collect_changed_files_for_fail_and_parent):
        {
          "sha_fail": "<sha_fail>",
          "changed_files": [
            {
              "commit": "<commit_sha>",
              "file_path": "<path/to/file>",
              "diff": "<unified diff>",
            },
            ...
          ]
        }
        """
        # Unpack OmegaConf + project root if your loader returns a tuple
        cfg_result = load_config()
        if isinstance(cfg_result, tuple) and len(cfg_result) == 2:
            self.config, self.project_root = cfg_result
        else:
            self.config, self.project_root = cfg_result, None

        self.error_logs = error_logs or {}
        self.changed_files_info = changed_files_info or {"changed_files": []}

        # Use correct defaults/types
        self.error_context = self.error_logs.get("error_context", [])         # list
        self.error_types = self.error_logs.get("error_types", [])             # list

        # Handle both singular/plural to be robust
        self.failed_jobs = self.error_logs.get(
            "failed_jobs",
            self.error_logs.get("failed_job", []),
        )  # list

        # Fix tuple-as-key bug
        self.relevant_files = self.error_logs.get("relevant_files", [])       # list

        self._has_checked_out = False
        self.workflow = workflow
        self.repo_path = repo_path
        self.failed_commit = sha_fail
        self.id = self.error_logs.get("id", "")

        self.model_name = model_name
        self.llm = llm
        self.memory_plugin = memory_plugin
        if self.memory_plugin is not None:
            try:
                self.memory_plugin.set_llm(llm)
            except Exception:
                pass
        self.memory_context = memory_context or {"enabled": False, "matches": []}

        # Build the issue-level memory query once; used as base for per-file retrieval.
        self._memory_query: Optional[Dict[str, Any]] = None
        if self.memory_plugin is not None and self.memory_plugin.is_enabled():
            try:
                repo_name = os.path.basename(self.repo_path.rstrip("/\\"))
                self._memory_query = self.memory_plugin.build_query(
                    task_id=str(self.id),
                    sha_fail=sha_fail,
                    repo_name=repo_name,
                    workflow_path="",
                    workflow=self.workflow or "",
                    log_analysis_result=self.error_logs,
                    changed_files_info=None,  # per-file targeting happens in retrieve_for_file()
                )
            except Exception as _qe:
                print(f"[Memory] Could not build query: {_qe}")

        self.parser = JsonOutputParser()

    def _normalize_path_key(self, value: Any) -> str:
        return os.path.normpath(str(value or "").strip())

    def _normalize_previous_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_level": str(experience.get("memory_level", "")).strip(),
            "memory_rank_score": float(experience.get("memory_rank_score", 0.0) or 0.0),
            "failure_pattern": str(experience.get("failure_pattern", "")).strip(),
            "memory_reason": str(experience.get("memory_reason", "")).strip(),
            "inspection_hint": str(experience.get("inspection_hint", "")).strip(),
            "confidence": str(experience.get("confidence", "")).strip() or "medium",
        }

    def _normalize_file_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        previous = []
        for item in entry.get("previous_experiences", []) or []:
            if isinstance(item, dict):
                previous.append(self._normalize_previous_experience(item))
        return {
            "file": (entry.get("file") or entry.get("path") or "").strip(),
            "reason": str(entry.get("reason", "")).strip(),
            "issue_type": str(entry.get("issue_type") or "").strip(),
            "previous_experiences": previous,
        }

    def _merge_text_values(self, left: str, right: str) -> str:
        values = []
        for text in (left, right):
            cleaned = str(text or "").strip()
            if cleaned and cleaned not in values:
                values.append(cleaned)
        return " | ".join(values)

    def _merge_previous_experiences(
        self,
        left: List[Dict[str, Any]],
        right: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[tuple, Dict[str, Any]] = {}
        for item in [*(left or []), *(right or [])]:
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_previous_experience(item)
            key = (
                normalized.get("memory_level", ""),
                normalized.get("failure_pattern", ""),
                normalized.get("memory_reason", ""),
                normalized.get("inspection_hint", ""),
            )
            current = merged.get(key)
            if current is None:
                merged[key] = normalized
                continue
            current["memory_rank_score"] = max(
                float(current.get("memory_rank_score", 0.0) or 0.0),
                float(normalized.get("memory_rank_score", 0.0) or 0.0),
            )
            if not current.get("failure_pattern"):
                current["failure_pattern"] = normalized.get("failure_pattern", "")
            if not current.get("memory_reason"):
                current["memory_reason"] = normalized.get("memory_reason", "")
            if not current.get("inspection_hint"):
                current["inspection_hint"] = normalized.get("inspection_hint", "")
        ordered = list(merged.values())
        ordered.sort(key=lambda item: float(item.get("memory_rank_score", 0.0) or 0.0), reverse=True)
        return ordered[:4]

    def _merge_file_entry(self, base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        merged["file"] = (base.get("file") or extra.get("file") or "").strip()
        merged["reason"] = self._merge_text_values(base.get("reason", ""), extra.get("reason", ""))
        base_issue = str(base.get("issue_type") or "").strip()
        extra_issue = str(extra.get("issue_type") or "").strip()
        merged["issue_type"] = base_issue if base_issue else extra_issue
        merged["previous_experiences"] = self._merge_previous_experiences(
            base.get("previous_experiences", []) or [],
            extra.get("previous_experiences", []) or [],
        )
        return merged

    def _dedupe_file_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for entry in entries:
            normalized = self._normalize_file_entry(entry)
            key = self._normalize_path_key(normalized.get("file", ""))
            if not key:
                continue
            if key not in merged:
                merged[key] = normalized
                order.append(key)
            else:
                merged[key] = self._merge_file_entry(merged[key], normalized)
        return [merged[key] for key in order]

    def _build_entry_from_relevant_file(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self._normalize_file_entry(
            {
                "file": (item.get("file") or item.get("path") or "").strip(),
                "reason": item.get("reason", ""),
                "issue_type": item.get("issue_type", ""),
                "previous_experiences": [],
            }
        )

    def _build_memory_experiences_for_file(
        self,
        *,
        file_path: str,
        retrieval_result: Dict[str, Any],
        file_context: str = "",
    ) -> List[Dict[str, Any]]:
        if self.memory_plugin is None:
            return []
        try:
            analysis = self.memory_plugin.analyze_relevance_for_file(
                file_path=file_path,
                file_context=file_context,
                retrieval_result=retrieval_result,
            )
        except Exception as exc:
            print(f"[Memory] Could not build previous_experiences for {file_path}: {exc}")
            return []

        experiences: List[Dict[str, Any]] = []
        for item in analysis.get("selected_items", []) or []:
            if not isinstance(item, dict):
                continue
            experiences.append(
                self._normalize_previous_experience(
                    {
                        "memory_level": item.get("memory_level", ""),
                        "memory_rank_score": item.get("similarity_score", 0.0),
                        "failure_pattern": item.get("failure_pattern", ""),
                        "memory_reason": item.get("failure_reason") or item.get("justification", ""),
                        "inspection_hint": item.get("localization_hint") or item.get("fix_direction", ""),
                        "confidence": item.get("relevance", "") or "medium",
                    }
                )
            )
        for item in analysis.get("selected_context", []) or []:
            if not isinstance(item, dict):
                continue
            experiences.append(
                self._normalize_previous_experience(
                    {
                        "memory_level": item.get("memory_level", ""),
                        "memory_rank_score": item.get("score", 0.0),
                        "failure_pattern": item.get("failure_pattern", ""),
                        "memory_reason": item.get("fl_guidance", ""),
                        "inspection_hint": item.get("fix_strategy", ""),
                        "confidence": "medium",
                    }
                )
            )
        return self._merge_previous_experiences([], experiences)

    def _build_memory_entry_for_file(
        self,
        *,
        file_path: str,
        reason: str,
        issue_type: str,
        retrieval_result: Dict[str, Any],
        file_context: str = "",
    ) -> Dict[str, Any]:
        return self._normalize_file_entry(
            {
                "file": file_path,
                "reason": reason,
                "issue_type": issue_type,
                "previous_experiences": self._build_memory_experiences_for_file(
                    file_path=file_path,
                    retrieval_result=retrieval_result,
                    file_context=file_context,
                ),
            }
        )

    def _compact_previous_experience_text(self, item: Dict[str, Any]) -> str:
        parts = []
        if item.get("memory_level"):
            parts.append(f"level={item['memory_level']}")
        if item.get("memory_rank_score"):
            parts.append(f"score={float(item['memory_rank_score']):.2f}")
        if item.get("failure_pattern"):
            parts.append(f"failure_pattern={item['failure_pattern']}")
        if item.get("memory_reason"):
            parts.append(f"reason={item['memory_reason']}")
        if item.get("inspection_hint"):
            parts.append(f"hint={item['inspection_hint']}")
        if item.get("confidence"):
            parts.append(f"confidence={item['confidence']}")
        return " | ".join(parts)

    def _format_file_entry_context(self, file_entry: Optional[Dict[str, Any]]) -> str:
        if not file_entry:
            return "No file-level organizer summary."
        lines = [
            f"file={file_entry.get('file', '')}",
            f"reason={file_entry.get('reason', '') or 'None'}",
            f"issue_type={file_entry.get('issue_type', '') or 'other'}",
        ]
        previous = file_entry.get("previous_experiences", []) or []
        if not previous:
            lines.append("previous_experiences=None")
            return "\n".join(lines)
        lines.append("previous_experiences:")
        for idx, item in enumerate(previous[:3], start=1):
            lines.append(f"  {idx}. {self._compact_previous_experience_text(item)}")
        return "\n".join(lines)

    def _enqueue_or_merge_file(
        self,
        *,
        queue: List[Dict[str, Any]],
        queued_entries: Dict[str, Dict[str, Any]],
        entry: Dict[str, Any],
        processed_files: set[str],
    ) -> None:
        normalized = self._normalize_file_entry(entry)
        key = self._normalize_path_key(normalized.get("file", ""))
        if not key or key in processed_files:
            return
        if key in queued_entries:
            merged = self._merge_file_entry(queued_entries[key], normalized)
            queued_entries[key] = merged
            for index, existing in enumerate(queue):
                if self._normalize_path_key(existing.get("file", "")) == key:
                    queue[index] = merged
                    break
            return
        queued_entries[key] = normalized
        queue.append(normalized)

    def _build_changed_files_selection_prompt(
        self,
        *,
        changed_files: List[Dict[str, Any]],
        failed_jobs_text: str,
    ) -> str:
        files_payload = []
        for item in changed_files:
            files_payload.append(
                {
                    "file": item.get("file_path", ""),
                    "diff": item.get("diff", ""),
                }
            )

        return f"""
You are a **Suspicious File Selector** for CI failures.

Goal:
Given a set of changed files with unified diffs and the CI failed jobs description, return ONLY the files whose changes are likely responsible for, or closely related to, the CI failure.
Select a file only when the diff provides strong evidence that it is relevant to the failed jobs. Do not speculate.

Return **only** a JSON array, as plain text:

[
  {{
    "file": "path/to/file",
    "reason": "short concrete reason tied to the failed jobs and diff",
    "issue_type": "formatting | linting | type_error | runtime_error | test_failure | dependency_error | docstring | complexity | other"
  }}
]

Hard rules:
- Do NOT add markdown fences.
- Do NOT add extra text before or after the JSON.
- Return `[]` if none of the files are suspicious.
- Only include files from the provided changed-files list.
- Keep each reason short, concrete, and tied to the diff plus failed jobs.

FAILED JOBS (CI context):
{failed_jobs_text}

CHANGED FILES WITH DIFFS:
{json.dumps(files_payload, indent=2, ensure_ascii=False)}
""".strip()

    def _parse_suspicious_files_response(self, raw_response: str) -> List[Dict[str, Any]]:
        text = (raw_response or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = demjson3.decode(text)

        if not isinstance(parsed, list):
            return []

        out: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            file_path = (item.get("file") or "").strip()
            if not file_path:
                continue
            out.append(
                self._normalize_file_entry(
                    {
                        "file": file_path,
                        "reason": item.get("reason", ""),
                        "issue_type": item.get("issue_type", "other"),
                        "previous_experiences": [],
                    }
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> Dict:
        try:
            self._force_clean_and_checkout(self.failed_commit)
            self._checkout_failed_commit_once()
            self._ensure_repo_at_commit(self.failed_commit, require_clean=True)

            print("[Step 1] Running File Selection...")
            suspecious_files = self.select_suspecious_files()
            suspecious_files = self._apply_memory_to_suspicious_files(suspecious_files)

            # Memory-mode fallback: if both selection steps found nothing,
            # try to derive candidate files directly from error_context so
            # fault localization can still run rather than produce empty results.
            if not suspecious_files and self.memory_plugin.is_enabled():
                print("[FL] No suspicious files after selection + memory augmentation. "
                      "Memory mode — falling back to error-context file extraction.")
                suspecious_files = self._fallback_files_from_error_context()

            print("[Step 2] Running final fault localization...")
            result = self._final_fault_localization(suspecious_files)
            result["id"] = self.id 
            return result

        except Exception as e:
            base_dir = os.path.join(
                self.config["exception_dir"],
                "interrupted_fault_localization",
            )
            os.makedirs(base_dir, exist_ok=True)
            filepath = os.path.join(base_dir, f"{self.failed_commit}_bug.json")
            error_info = {
                "sha_fail": self.failed_commit,
                "error": str(e),
                "tool": "FaultLocalization",
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=4)
            return error_info

    # ------------------------------------------------------------------ #
    # Git helpers
    # ------------------------------------------------------------------ #

    def _checkout_failed_commit_once(self):
        try:
            subprocess.run(
                ["git", "checkout", self.failed_commit],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
            )
            self._has_checked_out = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git checkout failed: {e.stderr.decode()}")
    
    def _force_clean_and_checkout(self, sha: str) -> None:
        """Hard reset, remove untracked files/dirs, and detach-checkout the exact sha."""
        def _run(args: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

        # Make sure we are in a repo
        _run(["rev-parse", "--is-inside-work-tree"])

        # 1) Discard local changes
        _run(["reset", "--hard"])

        # 2) Remove untracked files/dirs (keeps ignored files)
        _run(["clean", "-fd"])

        # If you ALSO want to remove ignored files (like .venv, build artifacts), use:
        # _run(["clean", "-fdx"])

        # 3) Checkout the exact commit in detached HEAD
        _run(["checkout", "--detach", sha])

        # 4) Verify exact sha + clean tree
        head = _run(["rev-parse", "HEAD"]).stdout.strip()
        if head != sha:
            raise RuntimeError(f"Repo HEAD mismatch after checkout. expected={sha} got={head}")

        status = _run(["status", "--porcelain"]).stdout.strip()
        if status:
            raise RuntimeError(
                f"Working tree not clean after force clean at {sha}.\n{status}"
            )

    def _ensure_repo_at_commit(self, sha: str, *, require_clean: bool = False) -> None:
        """Ensure repo HEAD is exactly at sha. Optionally require clean working tree."""
        def _run(args: list[str]) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

        # 1) Verify HEAD
        head = _run(["rev-parse", "HEAD"]).stdout.strip()
        if head != sha:
            # Force checkout to the exact sha
            _run(["checkout", "--detach", sha])
            head2 = _run(["rev-parse", "HEAD"]).stdout.strip()
            if head2 != sha:
                raise RuntimeError(f"Repo HEAD mismatch after checkout. expected={sha} got={head2}")

        # 2) Optionally verify clean working tree
        if require_clean:
            status = _run(["status", "--porcelain"]).stdout.strip()
            if status:
                raise RuntimeError(
                    f"Working tree not clean at {sha}. "
                    f"Refusing to proceed because files may not match the commit.\n{status}"
                )

    # ------------------------------------------------------------------ #
    # Suspicious file selection
    # ------------------------------------------------------------------ #

    def select_suspecious_files(self) -> List[Dict[str, Any]]:
        """
        1) Start from log_analyzer `relevant_files` (filtered by extension).
        2) Add extra files from `changed_files_info["changed_files"]` if the LLM
        says the diff is suspicious for the failed jobs.

        Returns a list of dicts that at least contain "file" so that
        _final_fault_localization can resolve paths.
        """
        suspecious_files: List[Dict[str, Any]] = []

        # 1) Relevant files from error_logs
        for item in self.relevant_files:
            file_path = (item.get("file") or item.get("path") or "").strip()
            if not file_path:
                continue

            ext = Path(file_path).suffix.lower()
            if ext not in {".py", ".txt", ".toml", ".md", ".rst", ".in", ".cfg", ".ini", ".yaml", ".yml"}:
                continue

            suspecious_files.append(self._build_entry_from_relevant_file(item))

        # Build a set of already-selected file paths from suspecious_files
        seen_paths: set[str] = set()
        for entry in suspecious_files:
            p = (entry.get("file") or entry.get("path") or "").strip()
            if p:
                seen_paths.add(p)

        # 2) Changed files from changed_files_info
        changed_files_list = self.changed_files_info.get("changed_files", []) or []
        if not changed_files_list:
            return suspecious_files

        failed_jobs_text = json.dumps(self.failed_jobs, indent=2, ensure_ascii=False)

        eligible_changed_files: List[Dict[str, Any]] = []
        for item in changed_files_list:
            file_path = (item.get("file_path") or "").strip()
            if not file_path or file_path in seen_paths:
                continue

            ext = Path(file_path).suffix.lower()
            if ext not in {".py", ".txt", ".toml", ".md", ".rst", ".in", ".cfg", ".ini", ".yaml", ".yml"}:
                continue

            eligible_changed_files.append(
                {
                    "file_path": file_path,
                    "diff": item.get("diff", ""),
                }
            )

        if not eligible_changed_files:
            return self._dedupe_file_entries(suspecious_files)

        model_for_count = self.model_name or "gpt-4o-mini"
        token_limit = int(get_prompt_token_budget(self.model_name) * 0.8)

        def _select_from_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            prompt = self._build_changed_files_selection_prompt(
                changed_files=batch,
                failed_jobs_text=failed_jobs_text,
            )
            raw_response = self.llm.invoke(prompt).content.strip()
            return self._parse_suspicious_files_response(raw_response)

        full_prompt = self._build_changed_files_selection_prompt(
            changed_files=eligible_changed_files,
            failed_jobs_text=failed_jobs_text,
        )

        selected_from_changed: List[Dict[str, Any]] = []
        used_full_batch = False
        if estimate_tokens(full_prompt, model=model_for_count) <= token_limit:
            try:
                selected_from_changed = _select_from_batch(eligible_changed_files)
                used_full_batch = True
            except Exception as e:
                print(f"[Selector] Batch selection failed; falling back to chunked mode: {e}")

        if not used_full_batch:
            current_batch: List[Dict[str, Any]] = []
            for item in eligible_changed_files:
                trial_batch = current_batch + [item]
                trial_prompt = self._build_changed_files_selection_prompt(
                    changed_files=trial_batch,
                    failed_jobs_text=failed_jobs_text,
                )
                if current_batch and estimate_tokens(trial_prompt, model=model_for_count) > token_limit:
                    try:
                        selected_from_changed.extend(_select_from_batch(current_batch))
                    except Exception as e:
                        print(f"[Selector] Chunked batch selection failed: {e}")
                    current_batch = [item]
                else:
                    current_batch = trial_batch

            if current_batch:
                try:
                    selected_from_changed.extend(_select_from_batch(current_batch))
                except Exception as e:
                    print(f"[Selector] Final chunked batch selection failed: {e}")

        for entry in selected_from_changed:
            file_path = (entry.get("file") or "").strip()
            if not file_path or file_path in seen_paths:
                continue
            suspecious_files.append(entry)
            seen_paths.add(file_path)
            print(f"[Selector] Marked '{file_path}' as suspicious based on batched diff analysis.")

        return self._dedupe_file_entries(suspecious_files)

    def _apply_memory_to_suspicious_files(
        self,
        suspicious_files: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        matches = self.memory_context.get("matches", []) or []
        if not matches or self.memory_plugin is None:
            return self._dedupe_file_entries(suspicious_files)

        augmented = self.memory_plugin.augment_suspicious_files(
            suspicious_files=suspicious_files,
            changed_files_info=self.changed_files_info,
            retrieval_result=self.memory_context,
        )
        normalized_augmented: List[Dict[str, Any]] = []
        existing_paths = {
            self._normalize_path_key(item.get("file", "")) for item in suspicious_files if item.get("file")
        }
        for item in augmented:
            file_path = (item.get("file") or item.get("path") or "").strip()
            if not file_path:
                continue
            key = self._normalize_path_key(file_path)
            if key in existing_paths:
                normalized_augmented.append(item)
                continue
            normalized_augmented.append(
                self._build_memory_entry_for_file(
                    file_path=file_path,
                    reason="Retrieved from similar prior failures for the current repository and CI context.",
                    issue_type="other",
                    retrieval_result=self.memory_context,
                )
            )

        for item in normalized_augmented:
            file_path = (item.get("file") or "").strip()
            if not file_path:
                continue
            try:
                analysis = self.memory_plugin.analyze_relevance_for_file(
                    file_path=file_path,
                    file_context="",
                    retrieval_result=self.memory_context,
                )
            except Exception:
                continue
            for ref in analysis.get("dependent_files", []) or []:
                if not isinstance(ref, dict):
                    continue
                dep_file = (ref.get("file") or "").strip()
                if not dep_file:
                    continue
                normalized_augmented.append(
                    self._build_memory_entry_for_file(
                        file_path=dep_file,
                        reason=str(ref.get("reason") or "Related file suggested by previous similar failures.").strip(),
                        issue_type="other",
                        retrieval_result=self.memory_context,
                    )
                )

        deduped = self._dedupe_file_entries(normalized_augmented)
        ranked = self.memory_plugin.rank_files(deduped, self.memory_context)

        print(
            f"[Memory] Retrieved {len(matches)} similar cases. "
            f"Candidate files after memory augmentation: {len(ranked)}"
        )
        return self._dedupe_file_entries(ranked)

    def _fallback_files_from_error_context(self) -> List[Dict[str, Any]]:
        """
        Memory-mode fallback when both select_suspecious_files() and
        _apply_memory_to_suspicious_files() return nothing.

        Strategy (no extra LLM call):
        1. Collect explicit "file" fields from error_context entries.
        2. Collect file paths from relevant_files (may have been filtered
           earlier by extension rules).
        3. Scan the serialised error_context text for any *.py path-like
           strings (e.g. from traceback lines).
        4. Verify each candidate actually exists in the repo and return
           only real files.
        """
        candidates: List[str] = []

        # 1) Explicit file fields in error_context
        for entry in (self.error_context or []):
            if not isinstance(entry, dict):
                continue
            for key in ("file", "file_path", "files"):
                val = entry.get(key)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())
                elif isinstance(val, list):
                    candidates.extend(str(v).strip() for v in val if v)

        # 2) relevant_files from log analysis (may have been excluded by
        #    extension filter in select_suspecious_files)
        for item in (self.relevant_files or []):
            if isinstance(item, dict):
                fp = (item.get("file") or item.get("path") or "").strip()
                if fp:
                    candidates.append(fp)

        # 3) Regex scan of the serialised error_context for .py paths
        ctx_text = json.dumps(self.error_context or [], ensure_ascii=False)
        for m in re.finditer(r'[\w./\\-]+\.py', ctx_text):
            candidates.append(m.group(0))

        # Primary error type for the issue_type field
        primary_issue = (
            self.error_types[0].get("error_type", "")
            if self.error_types and isinstance(self.error_types[0], dict)
            else ""
        )

        seen: set = set()
        result: List[Dict[str, Any]] = []
        for fp in candidates:
            fp = fp.strip().lstrip("/").replace("\\", "/")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            full_path = os.path.join(self.repo_path, fp)
            if os.path.isfile(full_path):
                result.append(self._normalize_file_entry({
                    "file": fp,
                    "reason": (
                        "Derived from error context — no diff-based or memory-augmented "
                        "suspicious files were found; using error log file references as fallback."
                    ),
                    "issue_type": primary_issue,
                    "previous_experiences": [],
                }))
                print(f"[FL Fallback] Added '{fp}' from error context.")

        if not result:
            print("[FL Fallback] No verifiable files found from error context either.")
        return result


    # ------------------------------------------------------------------ #
    # Final fault localization over selected files
    # ------------------------------------------------------------------ #

    def _final_fault_localization(self, suspecious_files: list) -> Dict[str, Any]:
        print("[Tool] read_error_file called")

        fault_localization: List[Dict[str, Any]] = []
        queue: List[Dict[str, Any]] = self._dedupe_file_entries(list(suspecious_files))
        queued_entries: Dict[str, Dict[str, Any]] = {
            self._normalize_path_key(item.get("file", "")): item
            for item in queue
            if item.get("file")
        }
        processed_files: set[str] = set()

        while queue:
            item = queue.pop(0)
            file_path = (item.get("file") or item.get("path") or "").strip()
            if not file_path:
                continue
            normalized_file = os.path.normpath(file_path)
            queued_entries.pop(normalized_file, None)
            if normalized_file in processed_files:
                continue

            ext = Path(file_path).suffix.lower()
            if ext not in {".py", ".toml", ".txt"}:
                continue

            resolved = self.find_full_file_path(file_path)
            if resolved["status"] != "found":
                print(f"[WARN] Could not resolve path for {file_path}")
                continue

            full_file_path = resolved["full_path"]

            content = self._read_file_content(full_file_path)
            if not content:
                continue

            file_type = self.detect_file_type(file_path)
            outline = build_outline(content) if file_type == "python" else ""
            numbered_full_content = self._numbered_file_content(content)

            chunks = self._chunk_file(numbered_full_content)
            num_chunks = len(chunks)

            # Steps 2–8: Per-file memory retrieval using the file's own context.
            # This runs the full pipeline: embed file context → cosine similarity
            # retrieval from L1/L2/L3 → ranking → LLM selection (lazy, in prompt).
            per_file_memory = self.memory_context  # fallback to issue-level
            if self.memory_plugin is not None and self._memory_query is not None:
                try:
                    per_file_memory = self.memory_plugin.retrieve_for_file(
                        file_path=file_path,
                        file_context=content[:1500],
                        issue_query=self._memory_query,
                    )
                except Exception as _me:
                    print(f"[Memory] Per-file retrieval failed for {file_path}: {_me}")

            # Per-chunk strict FL
            all_faults: List[Dict[str, Any]] = []
            print(f"[FL] Analyzing file: {full_file_path} ({num_chunks} chunks)")
            for idx, ch in enumerate(chunks):
                file_summary = (
                    f"File: {file_path}\n"
                    f"Full Path: {full_file_path}\n\n"
                    f"**File Outline (symbol → [start–end] lines)**\n{outline}\n\n"
                    f"File Type: {file_type}\n"
                    f"Chunk index: {idx+1} of {num_chunks}\n"
                    f"Chunk lines ({ch['valid_start']}–{ch['valid_end']}):\n\n"
                    f"```{file_type}\n{ch['content']}\n```"
                )

                faults = self.fault_localization_based_on_ci_log(
                    file_path=file_path,
                    full_file_path=full_file_path,
                    file_summary=file_summary,
                    valid_start=ch["valid_start"],
                    valid_end=ch["valid_end"],
                    original_content=content,
                    chunk_idx=idx,
                    num_chunks=num_chunks,
                    outline=outline,
                    chunk=ch["content"],
                    per_file_memory=per_file_memory,
                    file_entry=item,
                )

                if faults:
                    all_faults.extend(faults)
                    if self.memory_plugin is not None and (per_file_memory.get("selected_memory_levels") or []):
                        suggested = self.memory_plugin.get_additional_files_for_file(
                            file_path=file_path,
                            file_context=file_summary,
                            retrieval_result=per_file_memory,
                        )
                        for ref in suggested:
                            candidate = (ref.get("file") or "").strip()
                            if not candidate:
                                continue
                            resolved_candidate = self.find_full_file_path(candidate)
                            if resolved_candidate.get("status") != "found":
                                continue
                            self._enqueue_or_merge_file(
                                queue=queue,
                                queued_entries=queued_entries,
                                processed_files=processed_files,
                                entry={
                                    "file": candidate,
                                    "reason": ref.get("reason", ""),
                                    "issue_type": "other",
                                    "previous_experiences": [
                                        {
                                            "memory_level": "L2",
                                            "memory_rank_score": per_file_memory.get("weighted_similarity", 0.0),
                                            "failure_pattern": ref.get("failure_pattern", ""),
                                            "memory_reason": ref.get("reason", ""),
                                            "inspection_hint": ref.get("fix_strategy", ""),
                                            "confidence": "medium",
                                        }
                                    ],
                                },
                            )

            if all_faults:
                fault_localization.append(
                    {
                        "file_path": file_path,
                        "full_file_path": full_file_path,
                        "faults": all_faults,
                    }
                )
            processed_files.add(normalized_file)

        results = {
            "sha_fail": self.failed_commit,
            "fault_localization_data": fault_localization,
        }

        return results

    # ------------------------------------------------------------------ #
    # Core FL for one chunk
    # ------------------------------------------------------------------ #
    
    def prompt_for_fault_localization(
    self,
    *,
    file_path: str,
    file_summary: str,
    valid_start: int,
    valid_end: int,
    per_file_memory: Optional[Dict[str, Any]] = None,
    file_entry: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the strict fault-localization prompt for a single (sub)chunk.

        - file_summary: includes file metadata + outline + the code chunk (already fenced).
        - valid_start/valid_end: absolute (file-level) line numbers for this chunk window.
        - per_file_memory: retrieval result from retrieve_for_file() for this specific file.
        """
        memory_context_text = self._memory_context_for_file(
            file_path,
            file_summary,
            per_file_memory,
            file_entry=file_entry,
        )
        return f"""
You are a **Strict Fault Localization Agent**.

Your task:
Given a numbered source-code CHUNK, CI error context, workflow jobs, and a file outline,
identify **ALL distinct faults** in THIS chunk that are relevant to the CI failures. Analysis each line of code based CI error context and failed jobs to detect faults.

IMPORTANT:
• Do NOT stop after the first fault.
• Do NOT speculate beyond the shown lines.
• Do NOT expand scope beyond numeric containment rules.
• Output **valid JSON only** (array or empty array). No prose, no markdown.

==============================================================================
INPUT
------------------------------------------------------------------------------
SOURCE CODE (numbered lines; 1-based) with Outline:
{file_summary}

Each outline entry contains:
- kind: "func" | "class" | "import_block" | "const"
- name
- start
- end

FAILED JOBS:
{self.failed_jobs}

ERROR CONTEXT FROM CI LOGS:
{json.dumps(self.error_context, indent=2, ensure_ascii=False)}

ERROR TYPES:
{json.dumps(self.error_types, indent=2, ensure_ascii=False)}

RELEVANT FILES FROM LOGS:
{json.dumps(self.relevant_files, indent=2, ensure_ascii=False)}

RETRIEVED PREVIOUS EXPERIENCE:
{memory_context_text}

CHUNK WINDOW: lines {valid_start}–{valid_end}

==============================================================================
CORE DETECTION RULES
------------------------------------------------------------------------------

R1. Exhaustive Fault Detection (MANDATORY)
- Scan EVERY line from {valid_start}–{valid_end}.
- Add EVERY distinct fault that directly explains CI failures
- Continue scanning even after finding a valid fault.

R2. Evidence-Only Rule
- Every fault MUST be directly supported by:
  • CI log message OR
  •reason of the failure of given failed jobs
- No speculative or “possible” issues.

R3. Code Verification
- Each fault must be observable in this chunk OR
  provably absent (e.g., missing import or symbol).
- If CI claims something missing, confirm absence here.

==============================================================================
OUTLINE-BASED SCOPE CLASSIFICATION (STRICT — NUMERIC ONLY)
------------------------------------------------------------------------------

Numeric containment definition:
An outline entry CONTAINS [s,e] IFF entry.start <= s AND e <= entry.end

Apply rules IN ORDER. First match wins.

1) import_block
   If [s,e] is inside an outline entry with kind == "import_block"

2) method
   If [s,e] is inside an outline entry with kind == "func"

3) class
   If [s,e] is inside an outline entry with kind == "class"
   AND NOT inside any func entry

4) line (DEFAULT)
   If [s,e] is not inside ANY outline entry

- NEVER infer ownership.
- NEVER expand line_range to outline boundaries.
- NEVER escalate scope.

==============================================================================
DECORATOR RULE (CRITICAL)
------------------------------------------------------------------------------
- Decorator lines are OUTSIDE a function unless their line number
  is numerically inside func.start–func.end.
- If decorator line < func.start → fault_localization_level MUST be "line".

==============================================================================
FAULT OBJECT REQUIREMENTS (HARD)
------------------------------------------------------------------------------

For EACH fault, "reason" MUST include ALL THREE:

1) WHAT
   - Exact issue description.
   - Cite CI message or rule code explicitly.

2) FIX
   - Concrete imperative fix.
   - Example: "Add missing import X", "Remove unused variable Y".

3) WHERE
   - Exact line numbers in THIS chunk.
   - Allowed formats ONLY:
       • Fault at line N
       • Fault at lines N–M
       • Fault at line N (and line K)

If you cannot name exact line numbers → DO NOT return the fault.

==============================================================================
LINE RANGE RULES
------------------------------------------------------------------------------
- "line_range" MUST always be exactly TWO integers: [start, end].
- For a single faulty line N, use [N, N] — NEVER [N] alone.
- For a range of lines N to M, use [N, M].
- Must stay within {valid_start}–{valid_end}.
- Use displayed line numbers as integers (no leading zeros).
- INVALID: [10]  VALID: [10, 10]  VALID: [10, 25]

==============================================================================
FAULT MERGING RULE
------------------------------------------------------------------------------
- Only merge faults if:
  • Same underlying issue AND
  • Same outline element AND
  • Same fix.
- Otherwise, return separate objects.

==============================================================================
OUTPUT CONTRACT (STRICT)
------------------------------------------------------------------------------
Return ONLY:
- []  (if no new faults), OR
- A JSON array of objects:

[
  {{
    "line_range": [start, end],
    "reason": "...",
    "issue_type": "short description of the issue type as returned by analysis, e.g. type_error, test_failure, dependency_error",
    "fault_localization_level": "line | method | class | import_block"
  }}
]

No markdown. No comments. No extra keys. No trailing commas.

==============================================================================
FINAL CHECK
------------------------------------------------------------------------------
- All faults in this chunk are included
- No speculation
- Numeric containment strictly respected
- Exact line numbers cited
- Treat retrieved memory as compact hints only. Ignore any prior experience that is not directly supported by the current CI logs and current code in this chunk.
- Use retrieved memory only as supporting evidence. Current CI logs and current code always win.
- Valid JSON only
""".strip()
                            
    def fault_localization_based_on_ci_log(
    self,
    *,
    file_path: str,
    full_file_path: str,
    file_summary: str,
    valid_start: int,
    valid_end: int,
    original_content: str,
    chunk_idx: int,
    num_chunks: int,
    outline: List[Dict[str, Any]],
    chunk: str,
    per_file_memory: Optional[Dict[str, Any]] = None,
    file_entry: Optional[Dict[str, Any]] = None,
    ) -> list:
        fault_locations: List[Dict[str, Any]] = []
        token_limit = get_prompt_token_budget(self.model_name)

        prompt = self.prompt_for_fault_localization(
            file_path=file_path,
            file_summary=file_summary,
            valid_start=valid_start,
            valid_end=valid_end,
            per_file_memory=per_file_memory,
            file_entry=file_entry,
        )

        print(f"[Chunk {chunk_idx+1}/{num_chunks}] Analyzing lines {valid_start}-{valid_end}...")

        def _invoke_and_parse(p: str):
            raw_response = self.llm.invoke([HumanMessage(content=p)]).content.strip()
            if raw_response.startswith("```"):
                lines = raw_response.splitlines()
                lines = lines[1:]  # drop opening fence line (``` or ```json)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw_response = "\n".join(lines).strip()

            if not raw_response:
                return []

            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError:
                try:
                    parsed = demjson3.decode(raw_response)
                except Exception:
                    return []

            if not isinstance(parsed, list) or not parsed:
                return []
            return parsed

        model_for_count = self.model_name or "gpt-4o-mini"

        # -----------------------------
        # Case 1: prompt fits -> normal path
        # -----------------------------
        if estimate_tokens(prompt, model=model_for_count) <= token_limit:
            try:
                parsed_result = _invoke_and_parse(prompt)
            except Exception as e:
                self._save_fault_localization_error(
                    self.failed_commit,
                    e,
                    tool_name="fault_localization_based_on_ci_log",
                    prompt_name="llm_invocation",
                )
                print(f"[Chunk {chunk_idx}] LLM invocation error: {e}")
                return []
        else:
            # -----------------------------
            # Case 2: prompt too big -> subchunk the PROVIDED chunk only
            # sub_lines = (n_lines // 2) + 50 ; overlap stays 50
            # -----------------------------
            n_lines = len((chunk or "").splitlines())
            sub_lines = max(1, (n_lines // 2) + 50)

            print(
                f"[Chunk {chunk_idx+1}] Prompt too large "
                f"({estimate_tokens(prompt, model=model_for_count)} > {token_limit}). "
                f"Sub-chunking chunk content: {n_lines} lines -> {sub_lines} lines/chunk with 50 overlap."
            )

            subchunks = chunk_lines_with_overlap(
                chunk,
                lines_per_chunk=sub_lines,
                overlap=50,
            )

            parsed_result = []

            for sub_i, (sub_start_1b, sub_end_1b, sub_text) in enumerate(subchunks):
                # sub_start_1b/sub_end_1b are line numbers *within this chunk* (1-based)
                abs_start = (valid_start - 1) + sub_start_1b
                abs_end = (valid_start - 1) + sub_end_1b

                sub_file_summary = (
                    f"File: {file_path}\n"
                    f"Full Path: {full_file_path}\n\n"
                    f"**File Outline (symbol → [start–end] lines)**\n{outline}\n\n"
                    f"Chunk index: {chunk_idx+1} of {num_chunks} (sub {sub_i+1} of {len(subchunks)})\n"
                    f"Chunk lines ({abs_start}–{abs_end}):\n\n"
                    f"```text\n{sub_text}\n```"
                )

                sub_prompt = self.prompt_for_fault_localization(
                    file_path=file_path,
                    file_summary=sub_file_summary,
                    valid_start=abs_start,
                    valid_end=abs_end,
                    per_file_memory=per_file_memory,
                    file_entry=file_entry,
                )

                if estimate_tokens(sub_prompt, model=model_for_count) > token_limit:
                    print(
                        f"[Chunk {chunk_idx+1}] Skipping subchunk {sub_i+1}: "
                        f"prompt still too large ({estimate_tokens(sub_prompt, model=model_for_count)} > {token_limit})."
                    )
                    continue

                try:
                    sub_parsed = _invoke_and_parse(sub_prompt)
                except Exception as e:
                    self._save_fault_localization_error(
                        self.failed_commit,
                        e,
                        tool_name="fault_localization_based_on_ci_log",
                        prompt_name="llm_invocation_subchunk",
                    )
                    print(f"[Chunk {chunk_idx+1}] Subchunk LLM error: {e}")
                    continue

                parsed_result.extend(sub_parsed)

            if not parsed_result:
                print(f"[Chunk {chunk_idx}] No faults found.")
                return []

        # -----------------------------
        # Process parsed_result (unchanged)
        # -----------------------------
        for fault in parsed_result:
            line_range = fault.get("line_range")
            fault_level = fault.get("fault_localization_level")

            if not line_range or len(line_range) < 2:
                continue

            start, end = line_range[0], line_range[1]
            if valid_start <= start and end <= valid_end:
                extended_range = self._expand_line_range_with_outline(
                    line_range=line_range,
                    outline=outline,
                    fault_level=fault_level,
                )
                print("\n--- Before Line range ---", line_range)
                fault["line_range"] = extended_range

                snippet = extract_snippet_from_line_range(
                    original_file_content=original_content,
                    line_range=extended_range,
                )

                fault["code_snippet"] = snippet

                fault_locations.append(fault)
                print("\n--- Fault Detected ---")
                print("Code Snippet:\n", snippet)
                print("After Extending, Line Range:", extended_range)
                print("---------------------\n")
            else:
                print(
                    f"[Chunk {chunk_idx+1}] Skipping fault outside chunk range "
                    f"{valid_start}-{valid_end}: {line_range}"
                )

        return fault_locations

    def _memory_context_for_file(
        self,
        file_path: str,
        file_context: str = "",
        per_file_memory: Optional[Dict[str, Any]] = None,
        file_entry: Optional[Dict[str, Any]] = None,
    ) -> str:
        sections = [self._format_file_entry_context(file_entry)]

        # Use per-file retrieval result if provided; fall back to issue-level context.
        ctx = per_file_memory if per_file_memory is not None else self.memory_context
        if ctx.get("selected_memory_levels") and self.memory_plugin is not None:
            sections.append(self.memory_plugin.format_for_file_prompt(file_path, ctx, file_context=file_context))
        else:
            sections.append("No similar prior failures retrieved.")
        return "\n\n".join(section for section in sections if section.strip())

    # ------------------------------------------------------------------ #
    # Path & file helpers
    # ------------------------------------------------------------------ #

    def find_full_file_path(self, file_path: str) -> dict:
        """
        Find the best matching full path for a given relative file_path inside repo_path.
        Prioritizes the candidate whose relative path best matches the requested file_path.
        """
        normalized = os.path.normpath(file_path.split(":", 1)[0].strip())
        abs_path = os.path.join(self.repo_path, normalized)
        file_name = os.path.basename(normalized)

        try:
            if os.path.exists(abs_path):
                return {"status": "found", "full_path": abs_path}

            candidates = []
            for root, _, files in os.walk(self.repo_path):
                if file_name in files:
                    candidate = os.path.join(root, file_name)
                    rel_candidate = os.path.relpath(candidate, self.repo_path)
                    score = len(
                        os.path.commonprefix(
                            [rel_candidate[::-1], normalized[::-1]]
                        )
                    )
                    candidates.append((score, candidate))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_match = candidates[0][1]
                return {
                    "status": "found",
                    "full_path": best_match,
                    "all_candidates": [c[1] for c in candidates],
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

        return {"status": "not_found"}

    def _read_file_content(self, resolved_path: str) -> str:
        """Return file content as a string, or '' if file is missing/unreadable."""
        try:
            self._ensure_repo_at_commit(self.failed_commit, require_clean=True)
        except RuntimeError:
            # Repo is dirty (e.g. leftover patch state); attempt force clean and continue.
            try:
                self._force_clean_and_checkout(self.failed_commit)
            except Exception as clean_err:
                print(f"[WARN] Could not restore repo to {self.failed_commit}: {clean_err}")
                return ""

        if not os.path.exists(resolved_path):
            print(f"[WARN] File not found: {resolved_path}")
            return ""

        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] Could not read {resolved_path}: {e}")
            return ""

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #

    def _call_llm_directly(self, prompt: str) -> dict:
        chunks = chunk_log_by_tokens(prompt, max_tokens=90000, model=self.model_name)

        for chunk in chunks:
            try:
                raw_response = self.llm.invoke([HumanMessage(content=chunk)]).content.strip()
                raw_response = re.sub(
                    r"^```(?:json)?\s*|```$",
                    "",
                    raw_response.strip(),
                    flags=re.DOTALL,
                )
                return self.safe_parse_json(raw_response)
            except Exception as e:
                print(f"[ERROR] LLM call failed while parsing: {e}")
                raise

    def safe_parse_json(self, text: str) -> Dict:
        try:
            return self.parser.parse(text)
        except Exception as e:
            print("[!] Primary parser failed. Attempting cleanup...")
            try:
                cleaned = text.strip().strip("```json").strip("```").strip()
                return json.loads(cleaned)
            except Exception as second_error:
                print("[!] Fallback parsing failed.")
                raise ValueError(f"JSON parsing failed: {e}\n\nRaw:\n{text}")

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    def _numbered_file_content(self, content: str, offset: int = 0) -> str:
        return "\n".join(
            f"{idx+1:04d}: {line}" for idx, line in enumerate(content.splitlines())
        )

    def detect_file_type(self, file_path: str) -> str:
        """Detect programming/config language based on file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".dockerfile": "dockerfile",
            ".rst": "restructuredtext",
            ".md": "markdown",
        }

        if ext == "" and Path(file_path).name.lower() == "dockerfile":
            return "dockerfile"
        return mapping.get(ext, "text")

    def _chunk_file(self, file_content: str) -> List[Dict[str, Any]]:
        if is_large_context_model(self.model_name):
            chunk_size = 600
            overlap = 50
        else:
            chunk_size = 300
            overlap = 50
        lines = file_content.splitlines()
        total_lines = len(lines)
        chunks: List[Dict[str, Any]] = []

        if total_lines == 0:
            return chunks

        step = max(1, chunk_size - overlap)

        slice_start = 0
        while slice_start < total_lines:
            slice_end = min(slice_start + chunk_size, total_lines)

            # NO-GAP VALID WINDOW:
            # include the leading overlap in the current chunk
            # only trim the trailing overlap (except last chunk)
            valid_start = slice_start  # 0-based inclusive
            valid_end = slice_end if slice_end == total_lines else (slice_end - overlap)

            # defensive
            if valid_end < valid_start:
                valid_start = slice_start
                valid_end = slice_end

            chunk_lines = lines[slice_start:slice_end]

            chunks.append(
                {
                    "content": "\n".join(chunk_lines),
                    "line_range": (slice_start + 1, slice_end),      # absolute 1-based
                    "valid_start": valid_start + 1,                 # absolute 1-based
                    "valid_end": valid_end,                         # absolute 1-based
                }
            )

            slice_start += step

        return chunks


    def _save_fault_localization_error(
        self,
        sha: str,
        error: Exception,
        tool_name: str = "",
        prompt_name: str = "",
        extra_context: Optional[dict] = None,
    ):
        """
        Save detailed fault localization error info to JSON file.
        """
        base_dir = os.path.join(
            self.config["exception_dir"], "interrupted_fault_localization"
        )
        os.makedirs(base_dir, exist_ok=True)

        fname = f"{self.failed_commit}.json"
        filepath = os.path.join(base_dir, fname)

        error_info = {
            "sha_fail": sha or self.failed_commit,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "tool": tool_name,
            "prompt_name": prompt_name,
            "Agent": "FaultLocalization",
        }

        if extra_context:
            error_info["context"] = extra_context

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=4)

        return error_info

    def _expand_line_range_with_outline(
    self,
    line_range: List[int],
    outline: List[dict],
    fault_level: Optional[str] = None,
    ) -> List[int]:
        """
        Expand/normalize [start, end] using the outline.

        Contract by fault_level:
        - "method": return the entire containing function/method (tightest).
        - "import_block": return the containing import block (tightest).
        - "line": if inside decorator_block -> return that block; else return +/-5 lines.
        - "class": return the dedicated class per rules (as before).
        - anything else / None: return tightest containing node of any kind.
        """

        if not line_range or len(line_range) < 2:
            # Single-element range: treat as [N, N]
            if line_range and len(line_range) == 1:
                line_range = [line_range[0], line_range[0]]
            else:
                return line_range

        start, end = line_range[0], line_range[1]
        if not (isinstance(start, int) and isinstance(end, int)):
            return line_range

        # Normalize ordering defensively
        if end < start:
            start, end = end, start

        # ------------------------------------------------------------------
        # LINE LEVEL:
        # 1) If (start,end) is inside a decorator_block, return decorator_block span
        # 2) Otherwise, return +/-5 lines around the original range
        # ------------------------------------------------------------------
        if fault_level == "line":
            if outline:
                # flatten outline (no side effects)
                flat_nodes: List[dict] = []

                def _visit(node: dict):
                    if not isinstance(node, dict):
                        return
                    flat_nodes.append(node)
                    for ch in (node.get("children") or []):
                        _visit(ch)

                for node in outline:
                    _visit(node)

                def _contains(node: dict, s: int, e: int) -> bool:
                    a, b = node.get("start"), node.get("end")
                    return isinstance(a, int) and isinstance(b, int) and a <= s and e <= b

                dec_blocks = [
                    n for n in flat_nodes
                    if n.get("kind") == "decorator_block" and _contains(n, start, end)
                ]

                if dec_blocks:
                    # tightest decorator block (smallest span, then earliest start)
                    chosen = min(
                        dec_blocks,
                        key=lambda n: (int(n["end"]) - int(n["start"]), int(n["start"]))
                    )
                    expanded = [int(chosen["start"]), int(chosen["end"])]
                    # ensure it still contains original range
                    return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            # fallback: +/- 5 lines
            s = max(1, start - 5)
            e = end + 5
            return [s, e] if s <= start and end <= e else [start, end]

        # If outline missing/unusable, we can't expand to method/class/import boundaries
        if not outline:
            return [start, end]

        # Flatten with parent pointers (without mutating the original outline dicts).
        flat: List[dict] = []

        def visit(node: dict, parent: Optional[dict] = None):
            if not isinstance(node, dict):
                return
            n = dict(node)  # copy to avoid side effects
            n["_parent"] = parent
            children = node.get("children") or []
            n["children"] = children
            flat.append(n)
            for child in children:
                visit(child, n)

        for node in outline:
            visit(node, None)

        def contains(node: dict, s: int, e: int) -> bool:
            a, b = node.get("start"), node.get("end")
            return isinstance(a, int) and isinstance(b, int) and a <= s and e <= b

        candidates = [n for n in flat if contains(n, start, end)]
        if not candidates:
            return [start, end]

        def tightest(nodes: List[dict]) -> dict:
            return min(nodes, key=lambda n: (int(n["end"]) - int(n["start"]), int(n["start"])))

        # ---------------------------
        # METHOD LEVEL: tightest containing function/method
        # ---------------------------
        if fault_level == "method":
            funcs = [c for c in candidates if c.get("kind") in {"func", "method"}]
            if funcs:
                chosen = tightest(funcs)
                expanded = [int(chosen["start"]), int(chosen["end"])]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]
            return [start, end]

        # ---------------------------
        # IMPORT BLOCK LEVEL: tightest containing import block
        # ---------------------------
        if fault_level == "import_block":
            imps = [c for c in candidates if c.get("kind") == "import_block"]
            if imps:
                chosen = tightest(imps)
                expanded = [int(chosen["start"]), int(chosen["end"])]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]
            return [start, end]

        # ---------------------------
        # CLASS LEVEL: dedicated class per rules (your existing logic)
        # ---------------------------
        if fault_level == "class":
            class_candidates = [c for c in candidates if c.get("kind") == "class"]
            if not class_candidates:
                return [start, end]

            owning = tightest(class_candidates)
            cls_start, cls_end = int(owning["start"]), int(owning["end"])

            parent = owning.get("_parent")
            if parent and parent.get("kind") == "class":
                expanded = [cls_start, cls_end]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            direct_nested = [
                ch for ch in (owning.get("children") or [])
                if isinstance(ch, dict)
                and ch.get("kind") == "class"
                and isinstance(ch.get("start"), int)
                and isinstance(ch.get("end"), int)
            ]
            direct_nested.sort(key=lambda n: int(n["start"]))

            if not direct_nested:
                expanded = [cls_start, cls_end]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            first_child_start = int(direct_nested[0]["start"])
            header_end = first_child_start - 1

            if cls_start <= start and end <= header_end:
                expanded = [cls_start, header_end]
                return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]

            return [start, end]

        # ---------------------------
        # DEFAULT: tightest containing node of any kind
        # ---------------------------
        chosen = tightest(candidates)
        expanded = [int(chosen["start"]), int(chosen["end"])]
        return expanded if expanded[0] <= start and end <= expanded[1] else [start, end]
