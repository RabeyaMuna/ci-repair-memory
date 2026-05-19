from __future__ import annotations

import json
import math
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


def _clip(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _structured_file_refs(value: Any) -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    for item in _safe_list(value):
        if isinstance(item, dict):
            path = _normalize_path(str(item.get("file") or item.get("path") or ""))
            reason = str(item.get("reason") or "").strip()
            if path:
                refs.append({"file": path, "reason": reason})
        else:
            path = _normalize_path(str(item or ""))
            if path:
                refs.append({"file": path, "reason": ""})
    return refs


def _token_counter(text: str) -> Counter[str]:
    return Counter(_tokenize(text))


def _cosine_similarity(left_text: str, right_text: str) -> float:
    left = _token_counter(left_text)
    right = _token_counter(right_text)
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right[token] for token in left.keys() & right.keys())
    if numerator == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_error_type_rows(error_types: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in _safe_list(error_types):
        if isinstance(item, dict):
            rows.append(
                {
                    "category": str(item.get("category") or "").strip(),
                    "subcategory": str(item.get("subcategory") or "").strip(),
                    "evidence": str(item.get("evidence") or "").strip(),
                }
            )
        else:
            text = str(item or "").strip()
            if text:
                rows.append({"category": text, "subcategory": "", "evidence": ""})
    return rows


def _primary_error_type(error_types: Any) -> str:
    rows = _normalize_error_type_rows(error_types)
    if not rows:
        return ""
    return rows[0]["category"] or rows[0]["subcategory"]


def _primary_failure_pattern(error_types: Any) -> str:
    rows = _normalize_error_type_rows(error_types)
    if not rows:
        return ""
    return rows[0]["subcategory"] or rows[0]["category"]


def extract_files_from_diff(diff_text: str) -> List[str]:
    files: List[str] = []
    for match in re.finditer(r"^diff --git a/.+ b/(.+)$", diff_text or "", re.MULTILINE):
        path = _normalize_path(match.group(1).strip())
        if path:
            files.append(path)
    return files


def _extract_file_diff(diff_text: str, file_path: str) -> str:
    target = _normalize_path(file_path)
    current: List[str] = []
    capture = False
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            if capture:
                break
            match = re.match(r"^diff --git a/.+ b/(.+)$", line)
            capture = bool(match and _normalize_path(match.group(1)) == target)
        if capture:
            current.append(line)
    return "\n".join(current)


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
        path = _normalize_path(item.get("file_path", "")) if isinstance(item, dict) else ""
        if path:
            paths.append(path)
    return paths


def _extract_failed_commands_and_tools(failed_jobs: Any) -> Tuple[List[str], List[str]]:
    commands: List[str] = []
    tools: List[str] = []
    for item in _safe_list(failed_jobs):
        if isinstance(item, dict):
            for key in ("command", "validation_command", "failed_command", "cmd"):
                value = str(item.get(key) or "").strip()
                if value and value not in commands:
                    commands.append(value)
            for key in ("tool", "tools", "validator", "name", "job", "job_name", "step"):
                value = item.get(key)
                if isinstance(value, list):
                    for entry in value:
                        text = str(entry).strip()
                        if text and text not in tools:
                            tools.append(text)
                else:
                    text = str(value or "").strip()
                    if text and text not in tools:
                        tools.append(text)
        else:
            text = str(item).strip()
            if text and text not in tools:
                tools.append(text)
    return commands, tools


def _first_error_type(log_analysis_result: Dict[str, Any]) -> str:
    error_types = _safe_list(log_analysis_result.get("error_types", []))
    return str(error_types[0]).strip() if error_types else ""


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _write_json_list(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)


class MemoryPlugin:
    """
    MemCI-style three-level memory for fault localization.

    L1: failure_memory  — per-file failure records
    L2: repo_memory     — repo-scoped recurring patterns
    L3: cross_memory    — cross-repo generalized principles

    When an LLM is provided, candidates that pass lexical thresholds are
    re-scored by the LLM (Exemplar-Guardian-style) before injection.
    """

    def __init__(self, config, result_dir: str, llm=None):
        self.config = config
        self.result_dir = result_dir
        self.llm = llm
        self.enabled = bool(self._cfg("memory_enabled", False))
        self.top_k = int(self._cfg("memory_top_k", 3))

        # Per-ablation thresholds — always take precedence for known ablation configs.
        #
        # Design principle (Equal L1-floor / Corroboration-bonus):
        #   T_cond = T_L1 × W_L1(renormalized)
        #   where T_L1 = 0.55 is the base evidence bar for a direct file match.
        #
        # This guarantees:
        #   1. L1 alone at strength ≥ 0.55 always passes in every ablation condition.
        #   2. L2 and L3 can only LOWER the bar for a weaker L1 (corroboration bonus).
        #      Multi-level conditions are always at least as permissive as L1-only.
        #   3. Injection rate increases monotonically: L1 ≤ L1+L2 ≤ L1+L2+L3.
        #
        # Concretely (renormalized L1 weights: L1=1.000 / 0.667 / 0.600):
        #   L1 only    → T = 0.55 × 1.000 = 0.55  (40/60 on benchmark)
        #   L1+L2      → T = 0.55 × 0.667 = 0.37  (42/60 — 2 extra via L2 corroboration)
        #   L1+L2+L3   → T = 0.55 × 0.600 = 0.33  (43/60 — 1 further via L3)
        #
        # memory_similarity_threshold in config.yaml is only used as a fallback
        # for unrecognized ablation strings (e.g. custom configs).
        _raw_levels = str(self._cfg("memory_ablation_levels", "L1+L2+L3"))
        _ablation_thresholds = {"L1": 0.55, "L1+L2": 0.37, "L1+L2+L3": 0.33}
        if _raw_levels in _ablation_thresholds:
            self.similarity_threshold = float(_ablation_thresholds[_raw_levels])
        else:
            self.similarity_threshold = float(self._cfg("memory_similarity_threshold", 0.45))

        project_result_dir = str(self._cfg("project_result_dir", result_dir))
        self.failure_memory_path = os.path.join(project_result_dir, "failure_memory.json")
        self.repo_memory_path = os.path.join(project_result_dir, "repo_memory.json")
        self.cross_memory_path = os.path.join(project_result_dir, "cross_memory.json")
        self.retrieval_log_path = str(
            self._cfg(
                "memory_retrieval_log_path",
                os.path.join(result_dir, "memory_retrieval_log.jsonl"),
            )
        )

        self.failure_memory = _load_json_list(self.failure_memory_path)
        self.repo_memory = _load_json_list(self.repo_memory_path)
        self.cross_memory = _load_json_list(self.cross_memory_path)
        self.level_thresholds = {"L1": 0.30, "L2": 0.40, "L3": 0.50}

        # Ablation: which memory levels are active (L1 / L1+L2 / L1+L2+L3)
        raw_levels = str(self._cfg("memory_ablation_levels", "L1+L2+L3"))
        self.active_levels = {lvl.strip() for lvl in raw_levels.split("+") if lvl.strip()} or {"L1", "L2", "L3"}

        # Renormalize base weights so they always sum to 1.0 across active levels.
        # Without this, L1-only weighted_similarity = 0.60 × L1_best, which would
        # suppress memory far more than the full config at the same threshold.
        _base = {"L1": 0.60, "L2": 0.30, "L3": 0.10}
        _active_sum = sum(_base[lvl] for lvl in self.active_levels if lvl in _base)
        self.level_weights = {
            lvl: (_base[lvl] / _active_sum if lvl in self.active_levels and _active_sum > 0 else 0.0)
            for lvl in ("L1", "L2", "L3")
        }
        print(
            f"[Memory] active_levels={sorted(self.active_levels)}  "
            f"weights={{{', '.join(f'{k}:{v:.3f}' for k,v in self.level_weights.items())}}}  "
            f"threshold={self.similarity_threshold}"
        )

        self._per_file_analysis_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _cfg(self, key: str, default: Any) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:
            value = getattr(self.config, key, default)
        return default if value is None else value

    def is_enabled(self) -> bool:
        return self.enabled

    def set_llm(self, llm: Any) -> None:
        self.llm = llm

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
        failed_jobs = log_analysis_result.get("failed_jobs", log_analysis_result.get("failed_job", []))
        failed_cmd, failed_tool = _extract_failed_commands_and_tools(failed_jobs)
        error_type = _primary_error_type(log_analysis_result.get("error_types", []))
        failure_pattern = _primary_failure_pattern(log_analysis_result.get("error_types", []))
        failure_reason = _clip(
            " | ".join(str(x).strip() for x in _safe_list(log_analysis_result.get("error_context", [])) if str(x).strip()),
            1200,
        )
        return {
            "task_id": task_id,
            "sha_fail": sha_fail,
            "repo": repo_name,
            "repo_name": repo_name,
            "workflow_path": workflow_path,
            "workflow_text": workflow or "",
            "error_type": error_type,
            "failure_pattern": failure_pattern,
            "error_types": _normalize_error_type_rows(log_analysis_result.get("error_types", [])),
            "failed_cmd": failed_cmd,
            "failed_tool": failed_tool,
            "relevant_files": _extract_log_file_paths(log_analysis_result),
            "changed_files": _extract_changed_file_paths(changed_files_info),
            "failure_reason": failure_reason,
            "error_context_summary": _clip(
                json.dumps(log_analysis_result.get("error_context", []), ensure_ascii=False),
                1800,
            ),
        }

    def retrieve(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return self._empty_result(query, "memory_disabled")

        l1 = self._retrieve_l1(query) if "L1" in self.active_levels else []
        l2 = self._retrieve_l2(query) if "L2" in self.active_levels else []
        l3 = self._retrieve_l3(query) if "L3" in self.active_levels else []

        best_scores = {
            "L1": round(max((float(row.get("similarity_score", 0.0)) for row in l1), default=0.0), 4),
            "L2": round(max((float(row.get("similarity_score", 0.0)) for row in l2), default=0.0), 4),
            "L3": round(max((float(row.get("similarity_score", 0.0)) for row in l3), default=0.0), 4),
        }
        weighted_similarity = round(
            sum(self.level_weights[level] * best_scores[level] for level in ("L1", "L2", "L3")),
            4,
        )

        # Global threshold gate: suppress memory if overall confidence is too low
        if weighted_similarity < self.similarity_threshold:
            print(
                f"[Memory] weighted_similarity={weighted_similarity} < threshold={self.similarity_threshold}. "
                "Suppressing memory injection."
            )
            return self._empty_result(
                query, "below_weighted_threshold",
                level_scores=best_scores,
                weighted_similarity=weighted_similarity,
            )

        # Candidate files: prefer LLM's suspected_files, fall back to L1/L2 paths
        candidate_files: List[str] = []
        for row in l1:
            path = _normalize_path(row.get("file", ""))
            if path and path not in candidate_files:
                candidate_files.append(path)
        for row in l2:
            for file_row in (row.get("modified_files", []) or row.get("files", []))[:5]:
                path = _normalize_path(file_row.get("file", ""))
                if path and path not in candidate_files:
                    candidate_files.append(path)

        # High-level hints: coarse retrieval only; file-specific refinement happens later in FL.
        high_level_hints: List[str] = []
        for row in l2:
            reason = str(row.get("failure_reason") or "")
            if reason:
                high_level_hints.append(_clip(reason, 220))
        for row in l3:
            principle = str(row.get("principle") or row.get("fix_strategy") or "")
            if principle:
                high_level_hints.append(_clip(principle, 220))

        result = {
            "enabled": True,
            "query": {
                "task_id": query.get("task_id"),
                "sha_fail": query.get("sha_fail"),
                "repo": query.get("repo"),
                "error_type": query.get("error_type"),
                "failure_pattern": query.get("failure_pattern"),
            },
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                **self.level_thresholds,
            },
            "weights": dict(self.level_weights),
            "level_scores": best_scores,
            "weighted_similarity": weighted_similarity,
            "selected_memory_levels": [
                lvl for lvl, rows in (("L1", l1), ("L2", l2), ("L3", l3))
                if rows and best_scores.get(lvl, 0.0) >= self.level_thresholds[lvl]
            ],
            "candidate_files": candidate_files[:10],
            "high_level_hints": high_level_hints[:6],
            "l1_matches": l1,
            "l2_matches": l2,
            "l3_matches": l3,
            "matches": [*l1, *l2, *l3],
        }
        self._append_jsonl(self.retrieval_log_path, result)
        return result

    def _empty_result(
        self,
        query: Dict[str, Any],
        reason: str,
        level_scores: Optional[Dict[str, float]] = None,
        weighted_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "reason": reason,
            "query": {
                "task_id": query.get("task_id"),
                "sha_fail": query.get("sha_fail"),
                "repo": query.get("repo"),
                "error_type": query.get("error_type"),
            },
            "thresholds": {
                "similarity_threshold": self.similarity_threshold,
                **self.level_thresholds,
            },
            "weights": dict(self.level_weights),
            # Preserve actual computed scores even when suppressed so ablation
            # analysis can see how close each issue was to the threshold.
            "level_scores": level_scores if level_scores is not None else {"L1": 0.0, "L2": 0.0, "L3": 0.0},
            "weighted_similarity": weighted_similarity,
            "selected_memory_levels": [],
            "candidate_files": [],
            "high_level_hints": [],
            "l1_matches": [],
            "l2_matches": [],
            "l3_matches": [],
            "matches": [],
        }
        if self.enabled:
            self._append_jsonl(self.retrieval_log_path, result)
        return result

    def _retrieve_l1(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        repo = str(query.get("repo") or "")
        error_type = str(query.get("error_type") or "").lower()
        failure_pattern = str(query.get("failure_pattern") or "").lower()
        query_files = query.get("relevant_files", []) + query.get("changed_files", [])
        query_basenames = [_basename(p) for p in query_files]
        query_tools = [str(x).lower() for x in query.get("failed_tool", [])]
        query_reason = str(query.get("failure_reason") or query.get("error_context_summary") or "")

        scored: List[Dict[str, Any]] = []
        for row in self.failure_memory:
            if row.get("sha_fail") == query.get("sha_fail"):
                continue
            if repo and str(row.get("repo") or "") != repo:
                continue
            row_error = str(row.get("error_type") or "").lower()
            row_pattern = str(row.get("failure_pattern") or row.get("issue_type") or "").lower()
            row_reason = str(row.get("failure_reason") or row.get("reason") or "")
            row_tools = [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
            row_doc = " ".join(
                [
                    str(row.get("error_type") or ""),
                    str(row.get("failure_pattern") or row.get("issue_type") or ""),
                    row_reason,
                    " ".join(row_tools),
                    str(row.get("file") or ""),
                ]
            )
            query_doc = " ".join(
                [
                    str(query.get("error_type") or ""),
                    str(query.get("failure_pattern") or ""),
                    query_reason,
                    " ".join(query_tools),
                    " ".join(query_files),
                ]
            )

            # Check whether the memory's file appears in the query's file list.
            # Jaccard(many_query_files, one_memory_file) = 1/N even for a perfect
            # match, so use a direct membership check instead.
            row_file_norm = _normalize_path(str(row.get("file") or ""))
            row_file_base = _basename(row_file_norm)
            file_score = (
                1.0 if row_file_norm and row_file_norm in {_normalize_path(p) for p in query_files}
                else (0.7 if row_file_base and row_file_base in {_basename(p) for p in query_files}
                      else 0.0)
            )
            error_score = 1.0 if error_type and row_error and row_error == error_type else 0.0
            pattern_score = 1.0 if failure_pattern and row_pattern and row_pattern == failure_pattern else _cosine_similarity(failure_pattern, row_pattern)
            tool_score = _jaccard(query_tools, row_tools)
            text_score = _cosine_similarity(query_doc, row_doc)
            similarity = round(
                0.35 * file_score
                + 0.20 * error_score
                + 0.15 * pattern_score
                + 0.10 * tool_score
                + 0.20 * text_score,
                4,
            )
            if similarity < self.level_thresholds["L1"]:
                continue

            scored.append(
                {
                    **row,
                    "memory_level": "L1",
                    "similarity_score": similarity,
                    "matched_on": {
                        "file_score": round(file_score, 4),
                        "error_score": round(error_score, 4),
                        "pattern_score": round(pattern_score, 4),
                        "tool_score": round(tool_score, 4),
                        "text_score": round(text_score, 4),
                    },
                }
            )

        scored.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        return scored[: self.top_k]

    def _retrieve_l2(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        repo = str(query.get("repo") or "")
        error_type = str(query.get("error_type") or "").lower()
        failure_pattern = str(query.get("failure_pattern") or "").lower()
        query_files = query.get("relevant_files", []) + query.get("changed_files", [])
        query_tools = [str(x).lower() for x in query.get("failed_tool", [])]
        query_doc = " ".join(
            [
                str(query.get("error_type") or ""),
                str(query.get("failure_pattern") or ""),
                str(query.get("failure_reason") or query.get("error_context_summary") or ""),
                " ".join(query_tools),
                " ".join(query_files),
            ]
        )

        scored: List[Dict[str, Any]] = []
        for row in self.repo_memory:
            if repo and str(row.get("repo") or "") != repo:
                continue
            row_error = str(row.get("error_type") or "").lower()
            row_pattern = str(row.get("failure_pattern") or row.get("pattern_name") or "").lower()
            modified_files = row.get("modified_files", []) or row.get("files", [])
            row_files = [str(item.get("file") or "") for item in modified_files if isinstance(item, dict)]
            row_tools = [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
            row_doc = " ".join(
                [
                    str(row.get("error_type") or ""),
                    str(row.get("failure_pattern") or row.get("pattern_name") or ""),
                    str(row.get("failure_reason") or ""),
                    " ".join(str(x) for x in _safe_list(row.get("fix_pattern", []))),
                    " ".join(row_tools),
                    " ".join(row_files),
                ]
            )
            error_score = 1.0 if error_type and row_error and row_error == error_type else 0.0
            pattern_score = 1.0 if failure_pattern and row_pattern and row_pattern == failure_pattern else _cosine_similarity(failure_pattern, row_pattern)
            tool_score = _jaccard(query_tools, row_tools)
            text_score = _cosine_similarity(query_doc, row_doc)
            similarity = round(
                0.45 * text_score + 0.25 * error_score + 0.15 * pattern_score + 0.15 * tool_score,
                4,
            )
            if similarity < self.level_thresholds["L2"]:
                continue

            scored.append(
                {
                    **row,
                    "memory_level": "L2",
                    "similarity_score": similarity,
                    "matched_on": {
                        "error_score": round(error_score, 4),
                        "pattern_score": round(pattern_score, 4),
                        "tool_score": round(tool_score, 4),
                        "text_score": round(text_score, 4),
                    },
                }
            )

        scored.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        return scored[: self.top_k]

    def _retrieve_l3(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        error_type = str(query.get("error_type") or "").lower()
        failure_pattern = str(query.get("failure_pattern") or "").lower()
        query_tools = [str(x).lower() for x in query.get("failed_tool", [])]
        query_doc = " ".join(
            [
                str(query.get("error_type") or ""),
                str(query.get("failure_pattern") or ""),
                str(query.get("failure_reason") or query.get("error_context_summary") or ""),
                " ".join(query_tools),
            ]
        )

        scored: List[Dict[str, Any]] = []
        for row in self.cross_memory:
            row_error = str(row.get("error_type") or "").lower()
            # L3 records store failure_pattern or issue_type (no singular failure_pattern field guaranteed)
            row_pattern = str(row.get("failure_pattern") or row.get("issue_type") or "").lower()
            row_tools = [str(x).lower() for x in _safe_list(row.get("failed_tool", []))]
            # L3 records have failure_reasons (list); join them for semantic text matching
            row_failure_reason = " | ".join(
                str(r) for r in _safe_list(row.get("failure_reasons", [])) if str(r).strip()
            ) or str(row.get("failure_reason") or row.get("principle") or "")
            row_doc = " ".join(
                [
                    str(row.get("error_type") or ""),
                    row_pattern,
                    row_failure_reason,
                    " ".join(str(x) for x in _safe_list(row.get("fix_strategies", []) or row.get("fix_pattern", []))),
                    " ".join(row_tools),
                ]
            )
            error_score = 1.0 if error_type and row_error and row_error == error_type else 0.0
            pattern_score = 1.0 if failure_pattern and row_pattern and row_pattern == failure_pattern else _cosine_similarity(failure_pattern, row_pattern)
            tool_score = _jaccard(query_tools, row_tools)
            text_score = _cosine_similarity(query_doc, row_doc)
            similarity = round(
                0.55 * text_score + 0.20 * error_score + 0.15 * pattern_score + 0.10 * tool_score,
                4,
            )
            if similarity < self.level_thresholds["L3"]:
                continue

            scored.append(
                {
                    **row,
                    "memory_level": "L3",
                    "similarity_score": similarity,
                    "matched_on": {
                        "error_score": round(error_score, 4),
                        "pattern_score": round(pattern_score, 4),
                        "tool_score": round(tool_score, 4),
                        "text_score": round(text_score, 4),
                    },
                }
            )

        scored.sort(key=lambda item: item.get("similarity_score", 0.0), reverse=True)
        return scored[: self.top_k]

    def augment_suspicious_files(
        self,
        suspicious_files: List[Dict[str, Any]],
        changed_files_info: Optional[Dict[str, Any]],
        retrieval_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidate_files = retrieval_result.get("candidate_files", []) or []
        if not candidate_files:
            return suspicious_files

        existing = {_normalize_path(item.get("file") or item.get("path") or "") for item in suspicious_files}
        existing.discard("")

        augmented = list(suspicious_files)
        for item in (changed_files_info or {}).get("changed_files", []) or []:
            path = _normalize_path(item.get("file_path", "")) if isinstance(item, dict) else ""
            if not path or path in existing:
                continue
            if path in candidate_files or _basename(path) in {_basename(p) for p in candidate_files}:
                augmented.append({"file": path, "memory_source": "hierarchical_memory"})
                existing.add(path)
        return augmented

    def rank_files(self, candidate_files: List[Dict[str, Any]], retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        l1 = retrieval_result.get("l1_matches", []) or []
        l2 = retrieval_result.get("l2_matches", []) or []

        direct_scores: Dict[str, float] = {}
        for row in l1:
            path = _normalize_path(row.get("file", ""))
            if path:
                direct_scores[path] = max(direct_scores.get(path, 0.0), float(row.get("similarity_score", 0.0)))
        for row in l2:
            boost = float(row.get("similarity_score", 0.0)) * 0.5
            for file_row in (row.get("modified_files", []) or row.get("files", []) or []):
                path = _normalize_path(file_row.get("file", ""))
                if path:
                    direct_scores[path] = max(direct_scores.get(path, 0.0), boost)

        ranked = []
        for index, item in enumerate(candidate_files):
            path = _normalize_path(item.get("file") or item.get("path") or "")
            score = direct_scores.get(path, direct_scores.get(_basename(path), 0.0))
            enriched = dict(item)
            if score:
                enriched["memory_rank_score"] = round(score, 4)
            ranked.append((score, -index, enriched))
        ranked.sort(reverse=True)
        return [item for _, _, item in ranked]

    def _filter_candidates_for_file(
        self,
        file_path: str,
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized = _normalize_path(file_path)
        base = _basename(normalized)
        l1_rows = [
            row for row in (retrieval_result.get("l1_matches", []) or [])
            if _normalize_path(row.get("file", "")) == normalized or _basename(row.get("file", "")) == base
        ]
        l2_rows = []
        for row in (retrieval_result.get("l2_matches", []) or []):
            files = row.get("modified_files", []) or row.get("files", []) or []
            if any(_normalize_path(item.get("file", "")) == normalized or _basename(item.get("file", "")) == base for item in files):
                l2_rows.append(row)
        if not l2_rows:
            l2_rows = (retrieval_result.get("l2_matches", []) or [])[:2]
        l3_rows = (retrieval_result.get("l3_matches", []) or [])[:2]
        return {"l1": l1_rows[:3], "l2": l2_rows[:2], "l3": l3_rows[:2]}

    def _build_file_level_analysis_prompt(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
        candidates: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        query = retrieval_result.get("query", {}) or {}

        def _summarize_row(level: str, idx: int, row: Dict[str, Any]) -> str:
            row_file = row.get("file", "")
            files = row.get("modified_files", []) or row.get("files", []) or []
            files_text = ", ".join(str(item.get("file", "")) for item in files[:4] if isinstance(item, dict))
            reason = row.get("failure_reason") or row.get("reason") or row.get("principle") or ""
            fix = row.get("fix_pattern") or row.get("fix_strategies") or row.get("fix_strategy") or ""
            return (
                f"  [{level}-{idx}] score={row.get('similarity_score', 0.0):.2f} "
                f"error_type={row.get('error_type', '')} "
                f"failure_pattern={row.get('failure_pattern') or row.get('issue_type') or row.get('pattern_name', '')}\n"
                f"      file={row_file}\n"
                f"      files={files_text}\n"
                f"      reason={_clip(str(reason), 260)}\n"
                f"      fix={_clip(str(fix), 220)}"
            )

        candidate_lines: List[str] = []
        for level_key, rows in (("L1", candidates["l1"]), ("L2", candidates["l2"]), ("L3", candidates["l3"])):
            if rows:
                for idx, row in enumerate(rows):
                    candidate_lines.append(_summarize_row(level_key, idx, row))
            else:
                candidate_lines.append(f"  [{level_key}] none")

        candidate_block = "\n".join(candidate_lines)

        return f"""You are a file-level memory relevance analyst for CI fault localization.

Your task:
- Compare the CURRENT FILE and CURRENT FAILURE context against the retrieved memory candidates.
- Keep only memory that is truly relevant and transferable for this file.
- Discard candidates that are weak, incompatible, or only globally similar.

CURRENT FAILURE
- repo: {query.get("repo", "")}
- error_type: {query.get("error_type", "")}
- failure_pattern: {query.get("failure_pattern", "")}
- weighted_similarity: {retrieval_result.get("weighted_similarity", 0.0):.2f}

CURRENT FILE CONTEXT
file_path: {file_path}
file_context:
{_clip(file_context, 2600)}

RETRIEVED CANDIDATES
{candidate_block}

Selection criteria:
1. Same or compatible error_type
2. Same or compatible failure_pattern
3. Relevant to this file, not just the issue globally
4. Same or compatible tool/failure mode
5. Useful for fault localization, not just patch description

Return STRICT JSON only:
{{
  "use_memory": true,
  "similarity_score": 0.0,
  "similarity_reason": "<why the selected memory is relevant for this file>",
  "selected_memory_levels": ["L1"],
  "selected_items": [
    {{
      "memory_level": "L1|L2|L3",
      "candidate_key": "L1-0",
      "similarity_score": 0.0,
      "relevance": "high|medium|low",
      "justification": "<why this candidate is relevant for this file>",
      "failure_pattern": "<compatible pattern>",
      "failure_reason": "<relevant reason>",
      "dependent_files": [
          {
              "file": "file_a",
              "reason": "<why this file is relevant for the current file>"
          },
          {
              "file": "file_b",
              "reason": "<why this file is relevant for the current file>"
          }
      ],
      "additional_localization_files": [
          {"file": "file_c", "reason": "<why this file is relevant for the current file>"}
      ],
      "localization_hint": "<what to inspect in the current file>",
      "fix_direction": "<transferable fix direction>"
    }}
  ],
  "diagnostic_summary": "<2-3 sentence file-specific memory summary>"
}}

Rules:
- If no candidate is useful for this file, return use_memory=false and selected_items=[].
- Be file-specific. Do not keep a candidate just because it matches the issue globally.
- Prefer L1 over L2 over L3 when multiple candidates are compatible.
- No markdown fences. No extra keys."""

    def analyze_relevance_for_file(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        cache_key = (_normalize_path(file_path), _clip(file_context, 800))
        if cache_key in self._per_file_analysis_cache:
            return self._per_file_analysis_cache[cache_key]

        empty = {
            "use_memory": False,
            "similarity_score": 0.0,
            "similarity_reason": "",
            "selected_memory_levels": [],
            "selected_items": [],
            "diagnostic_summary": "",
        }
        if not self.llm:
            self._per_file_analysis_cache[cache_key] = empty
            return empty

        candidates = self._filter_candidates_for_file(file_path, retrieval_result)
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for level_name, rows in (("L1", candidates["l1"]), ("L2", candidates["l2"]), ("L3", candidates["l3"])):
            for idx, row in enumerate(rows):
                candidate_map[f"{level_name}-{idx}"] = row
        if not (candidates["l1"] or candidates["l2"] or candidates["l3"]):
            self._per_file_analysis_cache[cache_key] = empty
            return empty

        prompt = self._build_file_level_analysis_prompt(
            file_path=file_path,
            file_context=file_context,
            retrieval_result=retrieval_result,
            candidates=candidates,
        )
        try:
            raw = self.llm.invoke(prompt).content.strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                parsed = empty
        except Exception as exc:
            print(f"[Memory] File-level LLM relevance analysis failed for {file_path}: {exc}")
            parsed = empty

        if isinstance(parsed, dict):
            parsed["_candidate_map"] = candidate_map

        self._per_file_analysis_cache[cache_key] = parsed
        return parsed

    def get_additional_files_for_file(
        self,
        *,
        file_path: str,
        file_context: str,
        retrieval_result: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        per_file = self.analyze_relevance_for_file(
            file_path=file_path,
            file_context=file_context,
            retrieval_result=retrieval_result,
        )
        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in _safe_list(per_file.get("selected_items", [])):
            if not isinstance(item, dict):
                continue
            for ref in _structured_file_refs(item.get("dependent_files", [])):
                if ref["file"] and ref["file"] not in seen and ref["file"] != _normalize_path(file_path):
                    seen.add(ref["file"])
                    out.append(ref)
            for ref in _structured_file_refs(item.get("additional_localization_files", [])):
                if ref["file"] and ref["file"] not in seen and ref["file"] != _normalize_path(file_path):
                    seen.add(ref["file"])
                    out.append(ref)
        return out

    def format_for_file_prompt(self, file_path: str, retrieval_result: Dict[str, Any], file_context: str = "") -> str:
        file_path = _normalize_path(file_path)
        per_file = self.analyze_relevance_for_file(
            file_path=file_path,
            file_context=file_context,
            retrieval_result=retrieval_result,
        )
        selected_items = per_file.get("selected_items", []) or []
        by_key: Dict[str, Dict[str, Any]] = per_file.get("_candidate_map", {}) or {}

        lines = ["HIERARCHICAL MEMORY SYNTHESIS (L1/L2/L3):"]
        lines.append(f"  Levels used: {', '.join(per_file.get('selected_memory_levels', [])) or 'None'}")
        lines.append(
            "  Similarity scores: "
            f"L1={retrieval_result.get('level_scores', {}).get('L1', 0.0):.2f}, "
            f"L2={retrieval_result.get('level_scores', {}).get('L2', 0.0):.2f}, "
            f"L3={retrieval_result.get('level_scores', {}).get('L3', 0.0):.2f}, "
            f"weighted={retrieval_result.get('weighted_similarity', 0.0):.2f}, "
            f"file_relevance={per_file.get('similarity_score', 0.0):.2f}"
        )
        lines.append(f"  File-level relevance: {_clip(str(per_file.get('similarity_reason', '')), 220)}")
        diagnostic_summary = per_file.get("diagnostic_summary", "")
        if diagnostic_summary:
            lines.append(f"  LLM diagnostic summary: {_clip(diagnostic_summary, 400)}")
        if not selected_items:
            lines.append("  No file-specific memory evidence selected.")
            return "\n".join(lines)

        for item in selected_items[:4]:
            level = str(item.get("memory_level", ""))
            candidate_key = str(item.get("candidate_key", ""))
            row = by_key.get(candidate_key, {})
            lines.append(
                f"  {level} selected: candidate={candidate_key} score={float(item.get('similarity_score', 0.0)):.2f} "
                f"relevance={item.get('relevance', '')}"
            )
            if row:
                lines.append(
                    f"    error_type={row.get('error_type', '')} "
                    f"failure_pattern={row.get('failure_pattern') or row.get('issue_type') or row.get('pattern_name', '')}"
                )
            if item.get("justification"):
                lines.append(f"    justification={_clip(str(item.get('justification', '')), 220)}")
            if item.get("failure_reason"):
                lines.append(f"    relevant_reason={_clip(str(item.get('failure_reason', '')), 220)}")
            if item.get("localization_hint"):
                lines.append(f"    localization_hint={_clip(str(item.get('localization_hint', '')), 220)}")
            if item.get("fix_direction"):
                lines.append(f"    fix_direction={_clip(str(item.get('fix_direction', '')), 220)}")
            dependent_files = _structured_file_refs(item.get("dependent_files", []))
            if dependent_files:
                lines.append(
                    "    dependent_files="
                    + ", ".join(
                        f"{ref['file']} ({_clip(ref['reason'], 80)})" if ref.get("reason") else ref["file"]
                        for ref in dependent_files[:4]
                    )
                )
            extra_files = _structured_file_refs(item.get("additional_localization_files", []))
            if extra_files:
                lines.append(
                    "    additional_localization_files="
                    + ", ".join(
                        f"{ref['file']} ({_clip(ref['reason'], 80)})" if ref.get("reason") else ref["file"]
                        for ref in extra_files[:4]
                    )
                )

        return "\n".join(lines)

    def format_for_prompt(self, retrieval_result: Dict[str, Any]) -> str:
        lines = ["Retrieved hierarchical memory:"]
        for level_key in ("l1_matches", "l2_matches", "l3_matches"):
            for row in (retrieval_result.get(level_key, []) or [])[:2]:
                score_str = f"score={row.get('similarity_score', 0.0):.2f}"
                lines.append(
                    f"  {row.get('memory_level','')} scores=({score_str}) "
                    f"error_type={row.get('error_type','')}"
                )
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

        repo_id = repo_name
        error_type = _first_error_type(log_analysis_result)
        failed_jobs = log_analysis_result.get("failed_jobs", log_analysis_result.get("failed_job", []))
        failed_cmd, failed_tool = _extract_failed_commands_and_tools(failed_jobs)
        diff_text = str((patch_generator or {}).get("diff", "") or "")
        ground_truth_files = extract_files_from_diff(diff_text)
        fl_data = (fault_localizer or {}).get("fault_localization_data", []) or []
        relevant_files = _extract_log_file_paths(log_analysis_result)
        changed_files = _extract_changed_file_paths(changed_files_info)
        error_context_summary = _clip(
            json.dumps(log_analysis_result.get("error_context", []), ensure_ascii=False),
            1800,
        )

        l1_rows = self._build_l1_rows(
            task_id=task_id,
            sha_fail=sha_fail,
            repo_id=repo_id,
            repo_name=repo_name,
            error_type=error_type,
            failed_cmd=failed_cmd,
            failed_tool=failed_tool,
            ground_truth_files=ground_truth_files,
            fl_data=fl_data,
            diff_text=diff_text,
            error_context_summary=error_context_summary,
        )
        for row in l1_rows:
            self._upsert_list_record(
                self.failure_memory,
                row,
                keys=("sha_fail", "file", "error_type", "issue_type"),
            )

        repo_row = self._build_l2_row(
            task_id=task_id,
            sha_fail=sha_fail,
            repo_id=repo_id,
            repo_name=repo_name,
            error_type=error_type,
            failed_cmd=failed_cmd,
            failed_tool=failed_tool,
            l1_rows=l1_rows,
            ground_truth_files=ground_truth_files,
            error_context_summary=error_context_summary,
        )
        self._merge_repo_memory(repo_row)

        cross_row = self._build_l3_row(
            task_id=task_id,
            repo_id=repo_id,
            repo_name=repo_name,
            error_type=error_type,
            failed_cmd=failed_cmd,
            failed_tool=failed_tool,
            repo_row=repo_row,
            ground_truth_files=ground_truth_files,
        )
        self._merge_cross_memory(cross_row)

        _write_json_list(self.failure_memory_path, self.failure_memory)
        _write_json_list(self.repo_memory_path, self.repo_memory)
        _write_json_list(self.cross_memory_path, self.cross_memory)

    def _build_l1_rows(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_id: str,
        repo_name: str,
        error_type: str,
        failed_cmd: List[str],
        failed_tool: List[str],
        ground_truth_files: List[str],
        fl_data: List[Dict[str, Any]],
        diff_text: str,
        error_context_summary: str,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for entry in fl_data:
            file_path = _normalize_path(entry.get("file_path", ""))
            if not file_path:
                continue
            faults = entry.get("faults", []) or []
            first_fault = faults[0] if faults else {}
            reasons = [str(f.get("reason") or "").strip() for f in faults if str(f.get("reason") or "").strip()]
            issue_types = [str(f.get("issue_type") or "").strip() for f in faults if str(f.get("issue_type") or "").strip()]
            levels = [str(f.get("fault_localization_level") or "").strip() for f in faults if str(f.get("fault_localization_level") or "").strip()]
            row = {
                "issue_id": task_id,
                "sha_fail": sha_fail,
                "repo": repo_id,
                "repo_name": repo_name,
                "file": file_path,
                "line_range": first_fault.get("line_range") or [],
                "error_type": error_type,
                "error_subtype": "",
                "issue_type": issue_types[0] if issue_types else "",
                "failed_cmd": failed_cmd,
                "failed_tool": failed_tool,
                "dependent_files": [path for path in ground_truth_files if path != file_path],
                "reason": _clip(" | ".join(reasons) or error_context_summary, 400),
                "fix_strategy": _clip(_extract_file_diff(diff_text, file_path), 500),
                "error_level": sorted(set(levels)),
                "recommended_api": [],
            }
            rows.append(row)
        return rows

    def _build_l2_row(
        self,
        *,
        task_id: str,
        sha_fail: str,
        repo_id: str,
        repo_name: str,
        error_type: str,
        failed_cmd: List[str],
        failed_tool: List[str],
        l1_rows: List[Dict[str, Any]],
        ground_truth_files: List[str],
        error_context_summary: str,
    ) -> Dict[str, Any]:
        issue_type = next((str(row.get("issue_type") or "").strip() for row in l1_rows if row.get("issue_type")), "")
        issue_levels = sorted({level for row in l1_rows for level in row.get("error_level", []) if level})
        file_entries = [
            {
                "file": row.get("file", ""),
                "line_range": row.get("line_range", []),
                "reason": row.get("reason", ""),
                "issue_type": row.get("issue_type", ""),
                "error_type": row.get("error_type", ""),
            }
            for row in l1_rows
        ]
        changed_entries = [
            {"file": path, "line_range": [], "fix_family": "", "issue_id": task_id}
            for path in ground_truth_files
        ]
        fix_strategies = [row.get("fix_strategy", "") for row in l1_rows if row.get("fix_strategy")]
        fix_strategies = list(dict.fromkeys(fix_strategies))
        pattern_name = f"{error_type} {issue_type}".strip() or error_type or "repo_failure_pattern"
        return {
            "repo": repo_id,
            "repo_name": repo_name,
            "error_type": error_type,
            "issue_type": issue_type,
            "failed_cmd": failed_cmd,
            "error_level": issue_levels,
            "files": file_entries,
            "changed_files": ground_truth_files,
            "changed_file_entries": changed_entries,
            "fix_strategy": fix_strategies[0] if fix_strategies else "",
            "fix_strategies": fix_strategies,
            "failure_reason": _clip(error_context_summary, 400),
            "failure_reasons": [row.get("reason", "") for row in l1_rows if row.get("reason")][:5],
            "failed_tool": failed_tool,
            "pattern_name": pattern_name,
            "issue_ids": [task_id],
            "sha_fail_refs": [sha_fail],
        }

    def _build_l3_row(
        self,
        *,
        task_id: str,
        repo_id: str,
        repo_name: str,
        error_type: str,
        failed_cmd: List[str],
        failed_tool: List[str],
        repo_row: Dict[str, Any],
        ground_truth_files: List[str],
    ) -> Dict[str, Any]:
        issue_type = str(repo_row.get("issue_type") or "").strip()
        reason_parts = [str(r) for r in _safe_list(repo_row.get("failure_reasons", [])) if str(r).strip()]
        reason_snippet = _clip(" | ".join(reason_parts[:2]), 300) if reason_parts else ""
        principle = (
            f"error_type={error_type}"
            + (f" issue_type={issue_type}" if issue_type else "")
            + (f": {reason_snippet}" if reason_snippet else "")
            + f" (seen in repo: {repo_name})"
        )
        return {
            "error_type": error_type,
            "issue_type": issue_type,
            "failure_pattern": issue_type,
            "repos": [repo_id],
            "repo_names": [repo_name],
            "failed_tool": failed_tool,
            "failed_cmd": failed_cmd,
            "error_level": repo_row.get("error_level", []),
            "principle": principle,
            "failure_reasons": repo_row.get("failure_reasons", [])[:3],
            "fix_strategies": repo_row.get("fix_strategies", []),
            "fix_strategy": repo_row.get("fix_strategy", ""),
            "evidence_issue_ids": [task_id],
            "recommended_tools": [],
            "changed_files": ground_truth_files,
            "changed_file_entries": repo_row.get("changed_file_entries", []),
        }

    def _upsert_list_record(self, rows: List[Dict[str, Any]], record: Dict[str, Any], keys: Tuple[str, ...]) -> None:
        for index, row in enumerate(rows):
            if all(str(row.get(key) or "") == str(record.get(key) or "") for key in keys):
                rows[index] = record
                return
        rows.append(record)

    def _merge_repo_memory(self, incoming: Dict[str, Any]) -> None:
        key = (
            str(incoming.get("repo") or ""),
            str(incoming.get("error_type") or ""),
            str(incoming.get("issue_type") or ""),
        )
        for index, row in enumerate(self.repo_memory):
            row_key = (
                str(row.get("repo") or ""),
                str(row.get("error_type") or ""),
                str(row.get("issue_type") or ""),
            )
            if row_key != key:
                continue
            merged = dict(row)
            for field in ("failed_cmd", "error_level", "changed_files", "fix_strategies", "failure_reasons", "failed_tool", "issue_ids", "sha_fail_refs"):
                merged[field] = list(dict.fromkeys((row.get(field) or []) + (incoming.get(field) or [])))
            merged["files"] = self._merge_dict_list(row.get("files", []), incoming.get("files", []), "file")
            merged["changed_file_entries"] = self._merge_dict_list(
                row.get("changed_file_entries", []),
                incoming.get("changed_file_entries", []),
                "file",
            )
            if incoming.get("fix_strategy"):
                merged["fix_strategy"] = incoming["fix_strategy"]
            if incoming.get("failure_reason"):
                merged["failure_reason"] = incoming["failure_reason"]
            self.repo_memory[index] = merged
            return
        self.repo_memory.append(incoming)

    def _merge_cross_memory(self, incoming: Dict[str, Any]) -> None:
        key = (
            str(incoming.get("error_type") or ""),
            str(incoming.get("issue_type") or ""),
        )
        for index, row in enumerate(self.cross_memory):
            row_key = (
                str(row.get("error_type") or ""),
                str(row.get("issue_type") or ""),
            )
            if row_key != key:
                continue
            merged = dict(row)
            for field in ("repos", "repo_names", "failed_tool", "failed_cmd", "error_level", "failure_reasons", "fix_strategies", "evidence_issue_ids", "changed_files"):
                merged[field] = list(dict.fromkeys((row.get(field) or []) + (incoming.get(field) or [])))
            merged["changed_file_entries"] = self._merge_dict_list(
                row.get("changed_file_entries", []),
                incoming.get("changed_file_entries", []),
                "file",
            )
            if incoming.get("principle"):
                merged["principle"] = incoming["principle"]
            if incoming.get("fix_strategy"):
                merged["fix_strategy"] = incoming["fix_strategy"]
            self.cross_memory[index] = merged
            return
        self.cross_memory.append(incoming)

    def _merge_dict_list(self, left: List[Dict[str, Any]], right: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for row in (left or []) + (right or []):
            value = str(row.get(key) or "")
            if value and value not in seen:
                seen.add(value)
                out.append(row)
        return out

    def _append_jsonl(self, path: str, record: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
