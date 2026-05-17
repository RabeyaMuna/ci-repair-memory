from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_path(path: str) -> str:
    return (path or "").strip().lstrip("/").replace("\\", "/")


def _basename(path: str) -> str:
    return os.path.basename(_normalize_path(path))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_./-]+", (text or "").lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    lset = {x for x in left if x}
    rset = {x for x in right if x}
    if not lset and not rset:
        return 1.0
    if not lset or not rset:
        return 0.0
    return len(lset & rset) / len(lset | rset)


def _weighted_average(parts: List[Tuple[float, float]]) -> float:
    total_weight = sum(weight for weight, _ in parts)
    if total_weight == 0:
        return 0.0
    return sum(weight * value for weight, value in parts) / total_weight


def extract_files_from_diff(diff_text: str) -> List[str]:
    files: List[str] = []
    for match in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text or "", re.MULTILINE):
        fp = match.group(1).strip()
        if fp:
            files.append(_normalize_path(fp))
    return files


def _stringify_failed_jobs(failed_jobs: Any) -> str:
    try:
        return json.dumps(failed_jobs or [], ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(failed_jobs or "")


def _extract_job_tokens(failed_jobs: Any) -> List[str]:
    tokens: List[str] = []
    for item in _safe_list(failed_jobs):
        if isinstance(item, dict):
            for key in ("job_name", "name", "job", "command", "step", "tool"):
                value = item.get(key)
                if value:
                    tokens.extend(_tokenize(str(value)))
        else:
            tokens.extend(_tokenize(str(item)))
    return tokens


def _extract_log_file_paths(log_details: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for item in log_details.get("relevant_files", []) or []:
        if isinstance(item, dict):
            path = item.get("file") or item.get("path")
            if path:
                paths.append(_normalize_path(path))
        elif isinstance(item, str):
            paths.append(_normalize_path(item))
    return [p for p in paths if p]


def _extract_changed_file_paths(changed_files_info: Optional[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for item in (changed_files_info or {}).get("changed_files", []) or []:
        if isinstance(item, dict):
            fp = item.get("file_path")
            if fp:
                paths.append(_normalize_path(fp))
    return [p for p in paths if p]


def _clip(text: str, limit: int = 2000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class MemoryPlugin:
    """
    Hierarchical memory for CI repair.

    L1: issue-type filter
    L2: structural reranking (files, paths, workflow)
    L3: semantic reranking (log/workflow text tokens)
    """

    def __init__(self, config, result_dir: str):
        self.config = config
        self.result_dir = result_dir
        self.enabled = bool(self._cfg("memory_enabled", False))
        self.top_k = int(self._cfg("memory_top_k", 3))
        self.issue_threshold = float(self._cfg("memory_issue_threshold", 0.2))
        self.similarity_threshold = float(self._cfg("memory_similarity_threshold", 0.55))
        self.store_path = str(
            self._cfg(
                "memory_store_path",
                os.path.join(result_dir, "memory_store.jsonl"),
            )
        )
        self.retrieval_log_path = str(
            self._cfg(
                "memory_retrieval_log_path",
                os.path.join(result_dir, "memory_retrieval_log.jsonl"),
            )
        )
        self.memories = self._load_memories()

    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            value = getattr(self.config, key, default)
        return default if value is None else value

    def _load_memories(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.store_path):
            return []

        records: List[Dict[str, Any]] = []
        with open(self.store_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
        return records

    def is_enabled(self) -> bool:
        return self.enabled

    def build_query(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_name: str,
        workflow_path: str,
        workflow: str,
        log_analysis_result: Dict[str, Any],
        changed_files_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        error_context = log_analysis_result.get("error_context", []) or []
        return {
            "task_id": task_id,
            "sha_fail": sha_fail,
            "repo_name": repo_name,
            "workflow_path": workflow_path,
            "error_types": _safe_list(log_analysis_result.get("error_types", [])),
            "failed_jobs": log_analysis_result.get("failed_jobs", log_analysis_result.get("failed_job", [])),
            "relevant_files": _extract_log_file_paths(log_analysis_result),
            "changed_files": _extract_changed_file_paths(changed_files_info),
            "error_context_text": json.dumps(error_context, ensure_ascii=False),
            "workflow_text": workflow or "",
        }

    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return self._empty_result(query, "memory_disabled")
        if not self.memories:
            return self._empty_result(query, "memory_empty")

        l1_candidates: List[Dict[str, Any]] = []
        for memory in self.memories:
            if memory.get("sha_fail") == query.get("sha_fail"):
                continue
            issue_score = self._issue_score(query, memory)
            if issue_score < self.issue_threshold:
                continue
            l1_candidates.append(
                {
                    "memory": memory,
                    "issue_score": round(issue_score, 4),
                }
            )

        ranked: List[Dict[str, Any]] = []
        for candidate in l1_candidates:
            memory = candidate["memory"]
            structural = self._structural_score(query, memory)
            semantic = self._semantic_score(query, memory)
            overall = _weighted_average(
                [
                    (0.45, candidate["issue_score"]),
                    (0.35, structural),
                    (0.20, semantic),
                ]
            )
            if overall < self.similarity_threshold:
                continue
            candidate.update(
                {
                    "structural_score": round(structural, 4),
                    "semantic_score": round(semantic, 4),
                    "overall_score": round(overall, 4),
                }
            )
            ranked.append(candidate)

        ranked.sort(
            key=lambda item: (
                item["overall_score"],
                item["structural_score"],
                item["semantic_score"],
            ),
            reverse=True,
        )

        top = ranked[: self.top_k]
        result = {
            "enabled": True,
            "query": {
                "task_id": query.get("task_id"),
                "sha_fail": query.get("sha_fail"),
                "repo_name": query.get("repo_name"),
            },
            "thresholds": {
                "issue_threshold": self.issue_threshold,
                "similarity_threshold": self.similarity_threshold,
            },
            "matches": [self._format_match(item) for item in top],
        }
        self._append_jsonl(self.retrieval_log_path, result)
        return result

    def _empty_result(self, query: Dict[str, Any], reason: str) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "reason": reason,
            "query": {
                "task_id": query.get("task_id"),
                "sha_fail": query.get("sha_fail"),
                "repo_name": query.get("repo_name"),
            },
            "thresholds": {
                "issue_threshold": self.issue_threshold,
                "similarity_threshold": self.similarity_threshold,
            },
            "matches": [],
        }
        if self.enabled:
            self._append_jsonl(self.retrieval_log_path, result)
        return result

    def _issue_score(self, query: Dict[str, Any], memory: Dict[str, Any]) -> float:
        q_errors = [_token for item in query.get("error_types", []) for _token in _tokenize(str(item))]
        m_errors = [_token for item in memory.get("error_types", []) for _token in _tokenize(str(item))]
        q_jobs = _extract_job_tokens(query.get("failed_jobs", []))
        m_jobs = _extract_job_tokens(memory.get("failed_jobs", []))
        q_workflow = _tokenize(query.get("workflow_path", ""))
        m_workflow = _tokenize(memory.get("workflow_path", ""))
        return _weighted_average(
            [
                (0.5, _jaccard(q_errors, m_errors)),
                (0.3, _jaccard(q_jobs, m_jobs)),
                (0.2, _jaccard(q_workflow, m_workflow)),
            ]
        )

    def _structural_score(self, query: Dict[str, Any], memory: Dict[str, Any]) -> float:
        q_relevant = query.get("relevant_files", [])
        q_changed = query.get("changed_files", [])
        m_relevant = memory.get("relevant_files", [])
        m_changed = memory.get("changed_files", [])
        m_gt = memory.get("ground_truth_files", [])
        q_basenames = [_basename(p) for p in q_relevant + q_changed]
        m_basenames = [_basename(p) for p in m_relevant + m_changed + m_gt]
        return _weighted_average(
            [
                (0.45, _jaccard(q_relevant, m_relevant + m_gt)),
                (0.35, _jaccard(q_basenames, m_basenames)),
                (0.20, _jaccard(_tokenize(query.get("workflow_path", "")), _tokenize(memory.get("workflow_path", "")))),
            ]
        )

    def _semantic_score(self, query: Dict[str, Any], memory: Dict[str, Any]) -> float:
        q_error = _token_set(query.get("error_context_text", ""))
        m_error = _token_set(memory.get("error_context_text", ""))
        q_workflow = _token_set(query.get("workflow_text", ""))
        m_workflow = _token_set(memory.get("workflow_text", ""))
        return _weighted_average(
            [
                (0.7, _jaccard(q_error, m_error)),
                (0.3, _jaccard(q_workflow, m_workflow)),
            ]
        )

    def _format_match(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        memory = candidate["memory"]
        return {
            "sha_fail": memory.get("sha_fail"),
            "id": memory.get("id"),
            "repo_name": memory.get("repo_name"),
            "workflow_path": memory.get("workflow_path"),
            "issue_score": candidate["issue_score"],
            "structural_score": candidate["structural_score"],
            "semantic_score": candidate["semantic_score"],
            "overall_score": candidate["overall_score"],
            "error_types": memory.get("error_types", []),
            "relevant_files": memory.get("relevant_files", [])[:5],
            "changed_files": memory.get("changed_files", [])[:5],
            "ground_truth_files": memory.get("ground_truth_files", [])[:5],
            "repair_summary": memory.get("repair_summary", ""),
        }

    def rank_files(self, candidate_files: List[Dict[str, Any]], retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = retrieval_result.get("matches", []) or []
        if not matches:
            return candidate_files

        score_by_file: Counter[str] = Counter()
        for match in matches:
            boost = float(match.get("overall_score", 0.0))
            for path in match.get("ground_truth_files", []):
                score_by_file[_normalize_path(path)] += boost * 1.0
                score_by_file[_basename(path)] += boost * 0.75
            for path in match.get("relevant_files", []):
                score_by_file[_normalize_path(path)] += boost * 0.4
                score_by_file[_basename(path)] += boost * 0.25

        ranked: List[Tuple[float, int, Dict[str, Any]]] = []
        for index, item in enumerate(candidate_files):
            path = _normalize_path(item.get("file") or item.get("path") or "")
            bonus = score_by_file.get(path, 0.0) + score_by_file.get(_basename(path), 0.0)
            enriched = dict(item)
            if bonus:
                enriched["memory_rank_score"] = round(bonus, 4)
            ranked.append((bonus, -index, enriched))

        ranked.sort(reverse=True)
        return [item for _, _, item in ranked]

    def augment_suspicious_files(
        self,
        suspicious_files: List[Dict[str, Any]],
        changed_files_info: Optional[Dict[str, Any]],
        retrieval_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        matches = retrieval_result.get("matches", []) or []
        if not matches:
            return suspicious_files

        existing = {_normalize_path(item.get("file") or item.get("path") or "") for item in suspicious_files}
        existing.discard("")

        memory_targets = {
            _normalize_path(path)
            for match in matches
            for path in match.get("ground_truth_files", [])
        }
        memory_basenames = {_basename(path) for path in memory_targets}

        augmented = list(suspicious_files)
        for item in (changed_files_info or {}).get("changed_files", []) or []:
            path = _normalize_path(item.get("file_path", ""))
            if not path or path in existing:
                continue
            if path in memory_targets or _basename(path) in memory_basenames:
                augmented.append(
                    {
                        "file": path,
                        "memory_source": "retrieved_ground_truth",
                    }
                )
                existing.add(path)
        return augmented

    def format_for_prompt(self, retrieval_result: Dict[str, Any]) -> str:
        matches = retrieval_result.get("matches", []) or []
        if not matches:
            return "No similar memory retrieved."

        lines = [
            "Retrieved similar past CI failures ranked by hierarchical similarity.",
            f"Similarity threshold: {retrieval_result.get('thresholds', {}).get('similarity_threshold', self.similarity_threshold)}",
        ]
        for index, match in enumerate(matches, start=1):
            lines.append(
                (
                    f"[Memory {index}] sha={match.get('sha_fail')} "
                    f"score={match.get('overall_score')} "
                    f"(L1={match.get('issue_score')}, L2={match.get('structural_score')}, L3={match.get('semantic_score')})"
                )
            )
            lines.append(f"error_types={match.get('error_types', [])}")
            lines.append(f"ground_truth_files={match.get('ground_truth_files', [])}")
            lines.append(f"relevant_files={match.get('relevant_files', [])}")
            lines.append(f"repair_summary={match.get('repair_summary', '')}")
        return "\n".join(lines)

    def save_memory_entry(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_name: str,
        repo_owner: str,
        workflow_path: str,
        workflow: str,
        log_analysis_result: Dict[str, Any],
        changed_files_info: Optional[Dict[str, Any]],
        fault_localizer: Optional[Dict[str, Any]],
        patch_generator: Optional[Dict[str, Any]],
    ) -> None:
        if not self.enabled:
            return

        diff_text = (patch_generator or {}).get("diff", "")
        ground_truth_files = extract_files_from_diff(diff_text)
        fault_files = [
            _normalize_path(item.get("file_path", ""))
            for item in (fault_localizer or {}).get("fault_localization_data", []) or []
            if item.get("file_path")
        ]
        entry = {
            "memory_version": "v1",
            "id": task_id,
            "sha_fail": sha_fail,
            "repo_name": repo_name,
            "repo_owner": repo_owner,
            "workflow_path": workflow_path,
            "workflow_text": _clip(workflow, 3000),
            "error_types": _safe_list(log_analysis_result.get("error_types", [])),
            "failed_jobs": log_analysis_result.get("failed_jobs", log_analysis_result.get("failed_job", [])),
            "error_context_text": _clip(
                json.dumps(log_analysis_result.get("error_context", []), ensure_ascii=False),
                3000,
            ),
            "relevant_files": _extract_log_file_paths(log_analysis_result),
            "changed_files": _extract_changed_file_paths(changed_files_info),
            "fault_localization_files": fault_files,
            "ground_truth_files": ground_truth_files,
            "repair_summary": self._summarize_repair(diff_text, ground_truth_files),
        }

        self._append_jsonl(self.store_path, entry)
        self.memories.append(entry)

    def _summarize_repair(self, diff_text: str, files: List[str]) -> str:
        snippets = []
        for line in (diff_text or "").splitlines():
            if line.startswith("@@") or line.startswith("+") or line.startswith("-"):
                snippets.append(line)
            if len("\n".join(snippets)) > 800:
                break
        summary = {
            "files": files[:5],
            "diff_excerpt": _clip("\n".join(snippets), 800),
        }
        return json.dumps(summary, ensure_ascii=False)

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
