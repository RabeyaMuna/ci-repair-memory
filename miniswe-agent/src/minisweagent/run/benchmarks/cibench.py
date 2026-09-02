"""
cibench.py
==========
Run mini-SWE-agent on CI-failure repair instances in batch mode.

Each instance is processed entirely locally:
  - The repository is cloned from GitHub at run time (no Docker images needed).
  - The failing commit (sha_fail) is checked out inside the clone.
  - The agent runs in a LocalEnvironment against that checkout.

Input dataset  (JSONL or HuggingFace)
--------------------------------------
Each instance must contain:
  instance_id   str   unique identifier
  sha_fail      str   the commit SHA where CI failed
  repo_owner    str
  repo_name     str
  workflow_path str   e.g. ".github/workflows/test.yml"
  workflow_name str   e.g. "test"
  workflow      str   full workflow YAML content
  logs          any   raw CI logs  (str | list[{step_name,log}])

Output predictions  (preds.json)
---------------------------------
{
  "<instance_id>": {
    "id":       "<instance_id>",
    "sha_fail": "<sha>",
    "diff":     "<git diff output - the patch>"
  },
  ...
}

Each instance also gets a trajectory saved to:
  <output_dir>/<sha_fail>/<sha_fail>.traj.json

Usage examples
--------------
  # Run on all instances in a JSONL file
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/

  # Custom model
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/ \\
    -m anthropic/claude-opus-4

  # Filter / slice
  mini-swe-agent cibench --dataset ci_data.jsonl --output results/ \\
    --filter "^owner-repo" --slice 0:10
"""

import concurrent.futures
import json
import os
import re
import demjson3
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
import typer
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.benchmarks.utils.common import ProgressTrackingAgent

from minisweagent.run.benchmarks.utils.patch_merger import (
    merge_duplicate_patches,
    detect_duplicate_patches,
)

from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.project_env import load_project_env
from minisweagent.utils.serialize import UNSET, recursive_merge
from utilities.model_registry import configure_model_environment, resolve_model_alias
from utilities.run_metrics import (
    RunMetricsRecorder,
    completed_instance_ids,
    metric_instance_ids,
    safe_metrics_call,
)

# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "cibench.yaml"
_OUTPUT_FILE_LOCK = threading.Lock()
app = typer.Typer(rich_markup_mode="rich", add_completion=False)


def _discover_project_root() -> Path:
    """Find the CI-Repair-Bench workspace that owns shared benchmark data."""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "dataset").is_dir() and (parent / "baselines").is_dir():
            return parent
    # Fallback for a standalone mini-swe-agent checkout.
    return current_file.parents[4]


PROJECT_ROOT = _discover_project_root()
REPO_CACHE_ROOT = Path(
    os.getenv("MSWEA_REPO_CACHE_ROOT") or (PROJECT_ROOT / "repo")
).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper for network operations
# ─────────────────────────────────────────────────────────────────────────────
def _retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions=(Exception,),
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exception = exc
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                logger.error(
                    f"All {max_retries + 1} attempts failed. Last error: {exc}"
                )

    raise last_exception or RuntimeError("Retry failed with no exception captured")

# Automated fix tools for mechanical/static problems (formatting, linting, style)
AUTOMATED_TOOLS = [
    {
        "tool": "ruff",
        "purpose": "Unified Python linter and formatter",
        "install_command": "pip install ruff",
        "fix_command": "ruff check --fix {{file_or_dir}}",
        "file_pattern": "*.py",
    },
    {
        "tool": "black",
        "purpose": "Python code formatter",
        "install_command": "pip install black",
        "fix_command": "black {{file_or_dir}}",
        "file_pattern": "*.py",
    },
    {
        "tool": "isort",
        "purpose": "Python import sorter",
        "install_command": "pip install isort",
        "fix_command": "isort {{file_or_dir}}",
        "file_pattern": "*.py",
    },
    {
        "tool": "docstrfmt",
        "purpose": "RST documentation and docstring formatter",
        "install_command": "pip install docstrfmt",
        "fix_command": "docstrfmt {{file_or_dir}}",
        "file_pattern": ["*.rst", "*.py"],
    },
    {
        "tool": "docformatter",
        "purpose": "Python docstring formatter (PEP 257)",
        "install_command": "pip install docformatter",
        "fix_command": "docformatter --in-place --recursive {{file_or_dir}}",
        "file_pattern": "*.py",
    },
    {
        "tool": "docstrfmt",
        "purpose": "RST documentation and docstring formatter",
        "install_command": "pip install docstrfmt",
        "fix_command": "docstrfmt {{file_or_dir}}",
        "file_pattern": ["*.rst", "*.py"],
    },
    {
        "tool": "mdformat",
        "purpose": "Markdown formatter",
        "install_command": "pip install mdformat",
        "fix_command": "python -m mdformat {{file_or_dir}}",
        "file_pattern": "*.md",
    },
    {
        "tool": "codespell",
        "purpose": "Spell checker for code",
        "install_command": "pip install codespell",
        "fix_command": "codespell -w {{file_or_dir}}",
        "file_pattern": ["*.py", "*.md", "*.rst"],
    },
    {
        "tool": "taplo",
        "purpose": "TOML formatter",
        "install_command": "cargo install taplo-cli",
        "fix_command": "taplo fmt {{file_or_dir}}",
        "file_pattern": "*.toml",
    },
]


def _make_context_llm(
    config: Dict[str, Any],
    context_model: Optional[str] = None,
    metrics_recorder: Any = None,
) -> Any:
    """
    Build a plain callable (prompt: str) -> str for Phase A and Phase C
    using chat completion directly.

    The repair-agent model wrappers may add the bash tool and parse bash actions.
    Context extraction prompts only need raw text/JSON, and some OpenRouter
    models do not expose tool-calling endpoints.
    """
    model_config = config.get("model", {})
    model_name = resolve_model_alias(context_model or model_config.get("model_name"))
    model_kwargs = dict(model_config.get("model_kwargs") or {})
    # A stalled response body must not block one benchmark instance for the
    # provider's much longer default. This does not shorten input or output;
    # it only bounds an unresponsive network request.
    model_kwargs.setdefault(
        "timeout", int(os.getenv("CIBENCH_CONTEXT_LLM_TIMEOUT", "120"))
    )

    # These options belong to the repair agent, which is given a Bash tool.
    # Phase A/C context extraction is a plain text completion, so forwarding
    # tool controls without a tools array makes OpenAI reject the request.
    for tool_only_param in (
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
    ):
        model_kwargs.pop(tool_only_param, None)

    # GPT-5 and o-series reasoning models should use the provider's default
    # sampling behavior. The shared CI config's temperature is intended for
    # MiniMax and can be invalid for OpenAI reasoning models.
    lowered_model_name = str(model_name or "").lower().removeprefix("openai/")
    is_gpt5_or_o_series = lowered_model_name.startswith("gpt-5") or (
        len(lowered_model_name) > 1
        and lowered_model_name[0] == "o"
        and lowered_model_name[1].isdigit()
    )

    # GPT-4o and newer models (GPT-5, o-series) use 'max_completion_tokens' instead of 'max_tokens'
    is_gpt4o_or_newer = (
        is_gpt5_or_o_series
        or lowered_model_name.startswith("gpt-4o")
        or lowered_model_name.startswith("chatgpt-4o")
    )

    if is_gpt5_or_o_series:
        model_kwargs.pop("temperature", None)

    if is_gpt4o_or_newer and "max_tokens" in model_kwargs:
        # Convert max_tokens to max_completion_tokens for newer OpenAI models
        model_kwargs["max_completion_tokens"] = model_kwargs.pop("max_tokens")
        logger.info(
            "[CIBench] Converted max_tokens -> max_completion_tokens for %s",
            model_name
        )

    if not model_name:
        logger.warning("[CIBench] Could not build context LLM: no model configured")
        return None

    def _call(prompt: str, **generation_kwargs: Any) -> str:
        call_kwargs = dict(model_kwargs)
        for key in ("max_tokens", "max_completion_tokens", "reasoning_effort"):
            value = generation_kwargs.get(key)
            if value is not None:
                call_kwargs[key] = value

        if is_gpt4o_or_newer and "max_tokens" in call_kwargs:
            call_kwargs["max_completion_tokens"] = call_kwargs.pop("max_tokens")

        def _make_llm_call():
            call_started = time.time()
            safe_metrics_call(
                metrics_recorder,
                "begin_api_call",
                phase="context_llm",
                model=str(model_name),
            )
            try:
                response = litellm.completion(
                    model=str(model_name),
                    messages=[{"role": "user", "content": prompt}],
                    **call_kwargs,
                )
            except BaseException as exc:
                if metrics_recorder is not None:
                    safe_metrics_call(
                        metrics_recorder,
                        "record_api_call",
                        phase="context_llm",
                        model=str(model_name),
                        duration_seconds=time.time() - call_started,
                        status=(
                            "interrupted"
                            if isinstance(exc, KeyboardInterrupt)
                            else "failed"
                        ),
                        error=str(exc),
                    )
                raise
            if metrics_recorder is not None:
                safe_metrics_call(
                    metrics_recorder,
                    "record_response",
                    response=response,
                    phase="context_llm",
                    model=str(model_name),
                    duration_seconds=time.time() - call_started,
                )
            content = str(response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("LLM returned empty content")
            return content

        try:
            return _retry_with_backoff(
                _make_llm_call,
                max_retries=int(os.getenv("CIBENCH_CONTEXT_LLM_RETRIES", "1")),
                initial_delay=2.0,
                backoff_multiplier=2.0,
                exceptions=(Exception,),
            )
        except Exception as exc:
            logger.warning("[CIBench] context LLM call failed after retries: %s", exc)
            return ""

    return _call


def _resolve_context_model(context_model: Optional[str], config: Dict[str, Any]) -> str:
    """Use the repair model for every LLM stage unless explicitly overridden."""
    return str(resolve_model_alias(context_model or str(config["model"]["model_name"])))


_HELP_TEXT = """
Run mini-SWE-agent on CI-failure repair instances.

Each instance is run locally: the repo is cloned from GitHub, the failing
commit is checked out, and the agent edits the code in place.

Input: a JSONL dataset (one JSON object per line) or a HuggingFace dataset path.
Output: per-instance patch in preds.json, format {id, sha_fail, diff}.

[bold green]Required:[/bold green] --fault-localization /path/to/fault_localization.json
"""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

_seed_log_index: Optional[Dict[str, Dict[str, Any]]] = None


def _load_seed_log_index() -> Dict[str, Dict[str, Any]]:
    """
    Load data/trs/seed_log_details.json and index by sha_fail and id.
    Contains pre-computed CILogAnalyzer output (error_context, error_types,
    relevant_files, failed_job) nested under an 'analysis' key.
    Cached after first load.
    """
    global _seed_log_index
    if _seed_log_index is not None:
        return _seed_log_index

    seed_path = PROJECT_ROOT / "data" / "trs" / "seed_log_details.json"
    if not seed_path.exists():
        _seed_log_index = {}
        return {}

    try:
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else list(data.values())
        index: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            sha = str(rec.get("sha_fail") or "")
            iid = str(rec.get("id") or rec.get("instance_id") or "")
            if sha:
                index[sha] = rec
            if iid:
                index[iid] = rec
        _seed_log_index = index
        logger.info("[CIBench] Loaded seed_log_details index: %d records", len(index))
    except Exception as exc:
        logger.warning("[CIBench] Could not load seed_log_details.json: %s", exc)
        _seed_log_index = {}

    return _seed_log_index


def _inject_precomputed_analysis(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Look up the instance in seed_log_details.json and inject the pre-computed
    CILogAnalyzer output as top-level fields so Phase A is skipped entirely.

    seed_log_details stores analysis under a nested 'analysis' key:
      {
        "sha_fail": "...",
        "analysis": {
          "error_context":  [str]   -> overall_failure_reasons
          "error_types":    [dict]  -> overall_error_types (already set)
          "relevant_files": [dict]  -> effected_files
          "failed_job":     [dict]  -> failed_jobs
        }
      }

    Mapping to top-level fields that _has_precomputed_analysis checks for:
        analysis.error_context  -> overall_failure_reasons
        analysis.relevant_files -> effected_files
        analysis.failed_job     -> failed_jobs
        analysis.error_types    -> error_types (for Phase A query building)
    """
    index = _load_seed_log_index()
    if not index:
        return inst

    sha = str(inst.get("sha_fail") or "")
    iid = str(inst.get("instance_id") or inst.get("id") or "")
    seed_rec = index.get(sha) or index.get(iid)

    if not seed_rec:
        return inst

    analysis = seed_rec.get("analysis") or {}
    if not analysis:
        return inst

    inst = dict(inst)

    # Map nested analysis fields to the top-level fields the pipeline expects
    if not inst.get("overall_failure_reasons") and analysis.get("error_context"):
        inst["overall_failure_reasons"] = [
            str(x) for x in analysis["error_context"] if str(x).strip()
        ]

    if not inst.get("effected_files") and analysis.get("relevant_files"):
        inst["effected_files"] = analysis["relevant_files"]

    if not inst.get("failed_jobs") and analysis.get("failed_job"):
        inst["failed_jobs"] = analysis["failed_job"]

    if not inst.get("error_types") and analysis.get("error_types"):
        inst["error_types"] = analysis["error_types"]

    logger.debug(
        "[CIBench] Injected pre-computed analysis for sha=%s: "
        "reasons=%d  files=%d  jobs=%d",
        sha[:12],
        len(inst.get("overall_failure_reasons") or []),
        len(inst.get("effected_files") or []),
        len(inst.get("failed_jobs") or []),
    )
    return inst


def _has_local_precomputed_analysis(inst: Dict[str, Any]) -> bool:
    return bool(
        inst.get("overall_failure_reasons")
        and inst.get("effected_files")
        and inst.get("failed_jobs")
    )


def _prepare_local_instance(
    inst: Dict[str, Any], *, allow_hf_enrichment: bool
) -> Dict[str, Any]:
    normalized = _normalize_instance(inst)
    injected = _inject_precomputed_analysis(normalized)
    if _has_local_precomputed_analysis(injected) or not allow_hf_enrichment:
        return injected
    enriched = _enrich_from_hf(injected)
    return _inject_precomputed_analysis(enriched)


def _normalize_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names so both eval_dataset.jsonl and eval_issues.json
    work without any changes to the rest of the pipeline.

    Field aliases handled:
        id               -> instance_id        (eval_issues.json uses "id")
        workflow_filename -> workflow_path      (if workflow_path is missing)
        error_type list  -> overall_error_types (eval_issues has a list, not a string)
    """
    if not isinstance(inst, dict):
        return inst
    inst = dict(inst)

    # Prefer "id" over "instance_id" for benchmark consistency
    # (Benchmarks use numeric/string IDs like "102", "121", not repo@sha format)
    if "id" in inst:
        inst["instance_id"] = str(inst["id"])
    elif "instance_id" not in inst:
        # Fallback: generate from repo@sha if neither exists
        repo = f"{inst.get('repo_owner', '')}/{inst.get('repo_name', '')}".strip("/")
        sha = str(inst.get("sha_fail", ""))[:12]
        inst["instance_id"] = f"{repo}@{sha}" if repo and sha else "unknown"

    # workflow_filename -> workflow_path
    if "workflow_path" not in inst and "workflow_filename" in inst:
        wf = str(inst["workflow_filename"])
        inst["workflow_path"] = (
            f".github/workflows/{wf}" if not wf.startswith(".github") else wf
        )

    # error_type as list -> overall_error_types
    # eval_issues.json stores ["Dependency Issues", "Syntax Error", ...]
    # The pipeline uses overall_error_types (list) and error_type (str)
    et = inst.get("error_type")
    if isinstance(et, list):
        inst["overall_error_types"] = et
        inst["error_type"] = et[0] if et else ""
    elif isinstance(et, str) and et:
        inst.setdefault("overall_error_types", [et])

    return inst


_HF_DATASET_NAME = "ci-benchmark-user/ci-repair-bench"
_hf_index: Optional[Dict[str, Dict[str, Any]]] = None  # module-level cache


def _load_hf_index() -> Dict[str, Dict[str, Any]]:
    """
    Load the full HuggingFace dataset once and index it by sha_fail and id.
    Cached after first load so subsequent calls are instant.
    """
    global _hf_index
    if _hf_index is not None:
        return _hf_index

    try:
        from datasets import load_dataset  # type: ignore

        logger.info(
            "[CIBench] Loading HuggingFace dataset %s for instance enrichment...",
            _HF_DATASET_NAME,
        )
        # Suppress noisy HuggingFace "Repo card metadata block was not found" warnings
        import logging as _logging

        _hf_logger = _logging.getLogger("datasets")
        _prev_level = _hf_logger.level
        _hf_logger.setLevel(_logging.ERROR)

        # Try common split names - dataset uses 'train' as the only split
        # Try once - if it fails, use local data (no retries)
        ds = None
        for split_name in ("test", "train", "validation", "all"):
            try:
                ds = load_dataset(_HF_DATASET_NAME, split=split_name)
                break
            except Exception:
                continue

        _hf_logger.setLevel(_prev_level)

        if ds is None:
            raise RuntimeError(f"No usable split found in {_HF_DATASET_NAME}")

        index: Dict[str, Dict[str, Any]] = {}
        for row in ds:
            row = dict(row)
            sha = str(row.get("sha_fail") or "")
            iid = str(row.get("instance_id") or row.get("id") or "")
            if sha:
                index[sha] = row  # match by sha_fail
            if iid:
                index[iid] = row  # match by id / instance_id
            # Also index by numeric string of id (eval_issues stores id as string)
            raw_id = row.get("id")
            if raw_id is not None:
                index[str(int(raw_id))] = row
        _hf_index = index
        logger.info("[CIBench] HuggingFace index built: %d records", len(index))
        return index
    except Exception as exc:
        logger.warning(
            "[CIBench] Could not load HuggingFace dataset (%s) - using local data only",
            exc,
        )
        _hf_index = {}
        return {}


def _enrich_from_hf(inst: Dict[str, Any]) -> Dict[str, Any]:
    """
    Look up the instance in the HuggingFace dataset by sha_fail or id.
    If found, merge HF record (base) with local record (overlay) so that
    any pre-computed analysis fields from HF are used, but local overrides
    (like error_type list) are preserved.
    Returns the enriched instance.
    """
    index = _load_hf_index()
    if not index:
        return inst

    sha = str(inst.get("sha_fail") or "")
    iid = str(inst.get("instance_id") or inst.get("id") or "")

    hf_row = index.get(sha) or index.get(iid)
    if not hf_row:
        logger.debug(
            "[CIBench] No HF match for sha=%s id=%s - using local data", sha[:12], iid
        )
        return inst

    # Merge: HF record is the base (has pre-computed fields like overall_failure_reasons),
    # local inst overlays its own fields (logs, error_type list, etc.)
    merged = {**hf_row, **inst}
    return _normalize_instance(merged)


def load_ci_instances(dataset_path: str, split: str = "test") -> List[Dict[str, Any]]:
    """
    Load CI benchmark instances.

    Supports three sources:
        .json   - JSON array (eval_issues.json) - IDs are looked up in HuggingFace
                  to enrich with pre-computed analysis fields
        .jsonl  - one JSON object per line (eval_dataset.jsonl)
        str     - HuggingFace dataset name (loaded directly)

    For .json files: each instance is enriched by fetching the matching
    full record from HuggingFace by sha_fail/id, so the pipeline always
    gets complete data regardless of what the local file contains.
    """
    p = Path(dataset_path)

    if p.exists():
        if p.suffix.lower() == ".parquet":
            try:
                import pandas as pd

                frame = pd.read_parquet(p)
                instances = [_normalize_instance(record) for record in frame.to_dict("records")]
                logger.info("Loaded %d instances from Parquet dataset %s", len(instances), p)
                return instances
            except Exception as e:
                raise ValueError(f"Could not load Parquet dataset '{dataset_path}': {e}") from e

        raw_text = p.read_text(encoding="utf-8").strip()

        # JSON array - enrich each instance from HuggingFace
        if raw_text.startswith("["):
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse JSON array in {dataset_path}: {e}"
                ) from e
            instances = []
            for x in data:
                if not isinstance(x, dict):
                    continue
                instances.append(_prepare_local_instance(x, allow_hf_enrichment=True))
            logger.info(
                "Loaded %d instances from %s (enriched from HuggingFace)",
                len(instances),
                dataset_path,
            )
            return instances

        # JSONL - one object per line.  Filtered eval files may contain only
        # the compact local issue rows, so normalize, enrich, and inject the
        # same precomputed analysis used for JSON arrays.
        instances_: List[Dict[str, Any]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    for x in obj:
                        if not isinstance(x, dict):
                            continue
                        instances_.append(
                            _prepare_local_instance(x, allow_hf_enrichment=True)
                        )
                elif isinstance(obj, dict):
                    instances_.append(
                        _prepare_local_instance(obj, allow_hf_enrichment=True)
                    )
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed line in %s: %s", dataset_path, e)
        logger.info(
            "Loaded %d instances from %s (enriched from HuggingFace/local cache)",
            len(instances_),
            dataset_path,
        )
        return instances_

    # HuggingFace dataset name passed directly
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_path, split=split)
        instances_ = [_normalize_instance(dict(x)) for x in ds]  # type: ignore
        logger.info(
            "Loaded %d instances from HuggingFace dataset %s",
            len(instances_),
            dataset_path,
        )
        return instances_
    except Exception as e:
        raise ValueError(
            f"Could not load dataset from '{dataset_path}'. "
            f"Expected a .json / .jsonl file path or a HuggingFace dataset name. Error: {e}"
        ) from e


def filter_instances(
    instances: List[Dict[str, Any]],
    *,
    filter_spec: str = "",
    slice_spec: str = "",
    shuffle: bool = False,
) -> List[Dict[str, Any]]:
    import random

    if shuffle:
        instances = sorted(instances, key=lambda x: str(x.get("instance_id", "")))
        random.seed(42)
        random.shuffle(instances)

    if filter_spec:
        before = len(instances)
        instances = [
            inst
            for inst in instances
            if re.match(filter_spec, str(inst.get("instance_id", "")))
        ]
        logger.info(
            "Filter '%s': %d -> %d instances", filter_spec, before, len(instances)
        )

    if slice_spec:
        before = len(instances)
        parts = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*parts)]
        logger.info(
            "Slice '%s': %d -> %d instances", slice_spec, before, len(instances)
        )

    return instances


def load_fault_localization(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load reference-framework FL records, indexed by SHA and instance id."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("instance_id", key)
                records.append(record)
    elif isinstance(payload, list):
        records = [record for record in payload if isinstance(record, dict)]
    else:
        raise ValueError("fault-localization input must be a JSON list or object")

    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for key in (record.get("sha_fail"), record.get("instance_id"), record.get("id")):
            if key:
                indexed[str(key)] = record
    return indexed


def attach_fault_localization(
    instances: List[Dict[str, Any]], indexed: Dict[str, Dict[str, Any]]
) -> int:
    """Attach FL to dataset rows without changing the benchmark dataset."""
    matched = 0
    for instance in instances:
        record = None
        for key in (instance.get("sha_fail"), instance.get("instance_id"), instance.get("id")):
            if key and str(key) in indexed:
                record = indexed[str(key)]
                break
        if record is not None:
            instance["_fault_localization"] = record
            matched += 1
    return matched


def format_fault_localization(record: Dict[str, Any]) -> str:
    """Render predicted localization as explicit, non-ground-truth agent context."""
    locations = record.get("fault_localization_data") or []
    if not locations:
        return "No suspicious locations were predicted by the fault localizer."

    lines = [
        "The following locations were predicted by the fault-localization stage.",
        "They are hints, not ground truth; verify them against the repository and CI evidence.",
    ]
    for location in locations:
        file_path = location.get("file_path") or location.get("file") or "unknown"
        faults = location.get("faults") or []
        if not faults:
            lines.append(f"- {file_path} (file-level prediction; no narrower fault supplied)")
            continue
        for fault in faults:
            line_range = fault.get("line_range") or []
            where = file_path
            if len(line_range) >= 2:
                where += f":{line_range[0]}-{line_range[1]}"
            level = fault.get("fault_localization_level") or "unspecified"
            reason = fault.get("reason") or "No reason supplied"
            lines.append(f"- {where} [{level}]: {reason}")
            snippet = str(fault.get("code_snippet") or "").strip()
            if snippet:
                lines.append(f"  Predicted snippet:\n```\n{snippet}\n```")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Predictions file helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_preds(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def update_preds_file(
    output_path: Path,
    instance_id: str,
    sha_fail: str,
    diff: str,
    cost: float = 0.0,
) -> None:
    """
    Write / update the prediction for one instance (thread-safe).

    Automatically merges duplicate file patches in the diff to prevent
    merge conflicts during application.

    Args:
        output_path: Path to predictions file
        instance_id: Instance identifier
        sha_fail: Failing commit SHA
        diff: Patch diff
        cost: API cost for this instance (in dollars)
    """
    # Detect and merge duplicate patches
    if diff:
        duplicates = detect_duplicate_patches(diff)
        duplicate_files = [f for f, count in duplicates.items() if count > 1]

        if duplicate_files:
            logger.warning(
                "[CIBench] Instance %s has duplicate patches for %d file(s): %s",
                instance_id,
                len(duplicate_files),
                ", ".join(duplicate_files),
            )
            logger.info(
                "[CIBench] Merging duplicate patches for instance %s", instance_id
            )

            # Merge duplicates
            try:
                diff = merge_duplicate_patches(diff)
                logger.info(
                    "[CIBench] Successfully merged duplicate patches for instance %s",
                    instance_id,
                )
            except Exception as e:
                logger.error(
                    "[CIBench] Failed to merge duplicate patches for instance %s: %s",
                    instance_id,
                    str(e),
                )
                # Continue with original diff (will likely fail during application)

    with _OUTPUT_FILE_LOCK:
        data = _read_preds(output_path)
        data[instance_id] = {
            "id": instance_id,
            "sha_fail": sha_fail,
            "diff": diff or "",
            "cost": round(cost, 6),  # Round to 6 decimal places ($0.000001 precision)
        }
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def update_cost_tracking(
    output_dir: Path,
    instance_id: str,
    cost: float,
    model_name: str,
    ablation: str,
    direction: str,
) -> None:
    """
    Update cost tracking file with instance cost.

    Structure persists across interruptions and supports reruns.
    When instance is rerun, its cost is replaced with new value.

    Args:
        output_dir: Output directory containing costs.json
        instance_id: Instance identifier
        cost: API cost for this instance (dollars)
        model_name: Model used (e.g., "deepseek-v4-flash")
        ablation: Stable experiment configuration label
        direction: Experiment direction label
    """
    costs_path = output_dir / "costs.json"

    # Create configuration key
    config_key = f"{ablation}_{direction}_{model_name}".replace("/", "_")

    with _OUTPUT_FILE_LOCK:
        # Load existing costs
        if costs_path.exists():
            try:
                costs_data = json.loads(costs_path.read_text(encoding="utf-8"))
            except Exception:
                costs_data = {}
        else:
            costs_data = {}

        # Initialize configuration if not exists
        if config_key not in costs_data:
            costs_data[config_key] = {
                "model": model_name,
                "ablation": ablation,
                "direction": direction,
                "instances": {},
            }

        # Update instance cost (replace if rerun)
        costs_data[config_key]["instances"][instance_id] = round(cost, 6)

        # Save
        costs_path.write_text(
            json.dumps(costs_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def get_total_cost(output_dir: Path, model_name: str, ablation: str, direction: str) -> dict:
    """
    Calculate total cost for a configuration from costs.json.

    Returns:
        {
            "total_cost": float,
            "n_instances": int,
            "avg_cost": float,
            "instances": dict
        }
    """
    costs_path = output_dir / "costs.json"
    config_key = f"{ablation}_{direction}_{model_name}".replace("/", "_")

    if not costs_path.exists():
        return {"total_cost": 0.0, "n_instances": 0, "avg_cost": 0.0, "instances": {}}

    try:
        costs_data = json.loads(costs_path.read_text(encoding="utf-8"))
        config = costs_data.get(config_key, {})
        instances = config.get("instances", {})

        total_cost = sum(instances.values())
        n_instances = len(instances)
        avg_cost = total_cost / n_instances if n_instances > 0 else 0.0

        return {
            "total_cost": total_cost,
            "n_instances": n_instances,
            "avg_cost": avg_cost,
            "instances": instances,
        }
    except Exception:
        return {"total_cost": 0.0, "n_instances": 0, "avg_cost": 0.0, "instances": {}}


def remove_from_preds_file(output_path: Path, instance_id: str) -> None:
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        data = _read_preds(output_path)
        if instance_id in data:
            del data[instance_id]
            output_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )


def update_problems_file(
    output_path: Path,
    instance_id: str,
    problems: list,
) -> None:
    """
    Save the problems list that was provided to the agent for this instance.

    This helps track what the agent was asked to fix, separate from the final patch.
    Format: {"id": instance_id, "problems": [...]}
    """
    with _OUTPUT_FILE_LOCK:
        data = _read_preds(output_path)
        data[instance_id] = {
            "id": instance_id,
            "problems": problems or [],
        }
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Local environment setup  (clone -> checkout -> LocalEnvironment)
# ─────────────────────────────────────────────────────────────────────────────


def _load_install_commands_from_cache(sha_fail: str) -> list:
    """
    Load installation commands from workflow_validation_cache.json.

    Returns list of installation commands extracted from the validation_sequence.
    These are the EXACT commands used in CI, ensuring compatibility.

    Checks BOTH locations: data/ (user) and data/trs/ (generated)
    """
    # Try both cache locations
    cache_paths = [
        PROJECT_ROOT / "data" / "workflow_validation_cache.json",
        PROJECT_ROOT / "data" / "trs" / "workflow_validation_cache.json",
    ]

    cache_path = None
    for path in cache_paths:
        if path.exists():
            cache_path = path
            break

    if not cache_path:
        return []

    try:
        with open(cache_path) as f:
            cache = json.load(f)
        logger.debug(f"[CIBench] Loaded workflow cache from {cache_path}")

        # Find entry matching this sha_fail
        for entry in cache:
            if entry.get("sha_fail") == sha_fail:
                # Extract installation_cmd from validation_sequence
                commands = []
                for step in entry.get("validation_sequence", []):
                    install_cmd = step.get("installation_cmd", "").strip()
                    if install_cmd and install_cmd != "":
                        # Split multiple commands (comma-separated)
                        for cmd in install_cmd.split(","):
                            cmd = cmd.strip()
                            # Skip action references (not shell commands)
                            if not cmd.startswith("actions/") and not cmd.startswith(
                                "astral-sh/"
                            ):
                                commands.append(cmd)

                if commands:
                    logger.info(
                        f"[CIBench] Loaded {len(commands)} install command(s) from cache for {sha_fail[:8]}"
                    )
                return commands

        logger.info(f"[CIBench] No cache entry found for {sha_fail[:8]}")
        return []

    except Exception as e:
        logger.warning(f"[CIBench] Failed to load install commands from cache: {e}")
        return []


def _try_install_dependencies(testbed_path: Path, sha_fail: str = "") -> None:
    """
    Cache-aware dependency installation for 106+ repos and 567 issues.

    Strategy:
    1. Load install commands from workflow_validation_cache.json (BEST!)
       - These are the EXACT commands from CI workflows
       - Already tested and verified to work
    2. Fallback to smart project detection
    3. Continues on failure (resilient)

    Why cache-first?
    - Pre-extracted from actual CI workflows
    - Handles edge cases (poetry, conda, custom scripts, uv, etc.)
    - Works for all 106 repos automatically
    - No YAML parsing needed
    """
    import subprocess

    logger.info("[CIBench] Starting dependency installation...")

    # ── Strategy 1: Load install commands from cache (BEST) ──────────────────
    install_commands = _load_install_commands_from_cache(sha_fail)

    if install_commands:
        for i, cmd in enumerate(install_commands, 1):
            logger.info(f"[CIBench] Cache command {i}/{len(install_commands)}: {cmd}")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=testbed_path,
                    shell=True,  # Commands may have pipes, &&, etc.
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if result.returncode == 0:
                    logger.info("[CIBench] OK Cache install command succeeded")
                    # Continue to next command (don't return - install all)
                else:
                    logger.warning(
                        f"[CIBench] Cache command failed (code {result.returncode}), trying next..."
                    )
            except Exception as e:
                logger.warning(f"[CIBench] Cache command error: {e}, trying next...")

        # If we executed cache commands, return (don't try fallback)
        logger.info("[CIBench] OK Executed all cache install commands")
        return

    # ── Strategy 2: Smart project detection (FALLBACK) ───────────────────────
    logger.info("[CIBench] No cache commands found, trying smart detection...")

    # Detect project files
    project_files = {
        "pyproject.toml": (testbed_path / "pyproject.toml").exists(),
        "setup.py": (testbed_path / "setup.py").exists(),
        "requirements.txt": (testbed_path / "requirements.txt").exists(),
        "package.json": (testbed_path / "package.json").exists(),
        "package-lock.json": (testbed_path / "package-lock.json").exists(),
        "yarn.lock": (testbed_path / "yarn.lock").exists(),
        "pnpm-lock.yaml": (testbed_path / "pnpm-lock.yaml").exists(),
        "Gemfile": (testbed_path / "Gemfile").exists(),
        "Cargo.toml": (testbed_path / "Cargo.toml").exists(),
        "go.mod": (testbed_path / "go.mod").exists(),
        "pom.xml": (testbed_path / "pom.xml").exists(),
        "build.gradle": (testbed_path / "build.gradle").exists(),
    }

    detected_files = [k for k, v in project_files.items() if v]
    if detected_files:
        logger.info(f"[CIBench] Detected project files: {', '.join(detected_files)}")
    else:
        logger.info(
            "[CIBench] No standard project files detected - skipping installation"
        )
        return

    def _try_command(cmd: list, description: str, timeout: int = 180) -> bool:
        """Try a command, return True if successful"""
        try:
            logger.info(f"[CIBench] Trying: {description}")
            # CRITICAL: Use python -m pip instead of pip to ensure we use testbed's Python
            # This ensures dependencies install to the same Python the agent will use
            if cmd[0] == "pip":
                cmd = ["python", "-m", "pip"] + cmd[1:]

            result = subprocess.run(
                cmd, cwd=testbed_path, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                logger.info(f"[CIBench] OK {description} succeeded")
                return True
            else:
                logger.warning(
                    f"[CIBench] {description} failed (return code {result.returncode})"
                )
                return False
        except Exception as e:
            logger.warning(f"[CIBench] {description} error: {e}")
            return False

    # Python projects - ROOT level
    if project_files["pyproject.toml"] or project_files["setup.py"]:
        if _try_command(["pip", "install", "-e", "."], "pip install -e ."):
            return

    if project_files["requirements.txt"]:
        if _try_command(
            ["pip", "install", "-r", "requirements.txt"],
            "pip install -r requirements.txt",
        ):
            return

    # Python projects - MONOREPO (DYNAMIC - works for ANY repo structure!)
    # Use 'find' to discover ALL Python packages regardless of directory names
    logger.info("[CIBench] Searching for Python packages (dynamic discovery)...")
    packages_found = []

    try:
        # Find all pyproject.toml files (excluding root, venv, .git, node_modules)
        result = subprocess.run(
            [
                "find",
                ".",
                "-name",
                "pyproject.toml",
                "-not",
                "-path",
                "./pyproject.toml",
                "-not",
                "-path",
                "*/.venv/*",
                "-not",
                "-path",
                "*/venv/*",
                "-not",
                "-path",
                "*/.git/*",
                "-not",
                "-path",
                "*/node_modules/*",
                "-maxdepth",
                "4",
            ],
            cwd=testbed_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

        for line in result.stdout.strip().split("\n"):
            if line and line.strip():
                pkg_path = testbed_path / Path(line).parent
                if pkg_path.is_dir():
                    packages_found.append(pkg_path)

        # Also find setup.py files (avoid duplicates)
        result = subprocess.run(
            [
                "find",
                ".",
                "-name",
                "setup.py",
                "-not",
                "-path",
                "./setup.py",
                "-not",
                "-path",
                "*/.venv/*",
                "-not",
                "-path",
                "*/venv/*",
                "-not",
                "-path",
                "*/.git/*",
                "-not",
                "-path",
                "*/node_modules/*",
                "-maxdepth",
                "4",
            ],
            cwd=testbed_path,
            capture_output=True,
            text=True,
            timeout=10,
        )

        for line in result.stdout.strip().split("\n"):
            if line and line.strip():
                pkg_path = testbed_path / Path(line).parent
                if pkg_path.is_dir() and pkg_path not in packages_found:
                    packages_found.append(pkg_path)

    except Exception as e:
        logger.warning(f"[CIBench] Dynamic package search failed: {e}")

    if packages_found:
        logger.info(
            f"[CIBench] Found {len(packages_found)} Python package(s) in monorepo"
        )
        installed_count = 0

        # Try installing ALL packages (don't stop on first success!)
        for pkg_path in packages_found:
            pkg_name = pkg_path.name
            has_poetry = (pkg_path / "poetry.lock").exists()

            # Prefer poetry for poetry projects
            if has_poetry:
                logger.info(f"[CIBench] Installing {pkg_name} with poetry...")
                result = subprocess.run(
                    ["poetry", "install"],
                    cwd=pkg_path,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                if result.returncode == 0:
                    logger.info(f"[CIBench] OK {pkg_name} installed with poetry")
                    installed_count += 1
                    continue
                else:
                    logger.warning(
                        f"[CIBench] Poetry failed for {pkg_name}, trying pip..."
                    )

            # Fallback to pip install -e
            logger.info(f"[CIBench] Installing {pkg_name} with pip...")
            result = subprocess.run(
                ["python", "-m", "pip", "install", "-e", "."],
                cwd=pkg_path,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode == 0:
                logger.info(f"[CIBench] OK {pkg_name} installed")
                installed_count += 1
            else:
                logger.warning(
                    f"[CIBench] Failed to install {pkg_name} (continuing...)"
                )

        if installed_count > 0:
            logger.info(
                f"[CIBench] OK Installed {installed_count}/{len(packages_found)} monorepo packages"
            )
            return

    # Node.js projects
    if project_files["package.json"]:
        if project_files["pnpm-lock.yaml"]:
            if _try_command(["pnpm", "install"], "pnpm install"):
                return
        if project_files["yarn.lock"]:
            if _try_command(["yarn", "install"], "yarn install"):
                return
        if _try_command(["npm", "install"], "npm install"):
            return

    # Ruby projects
    if project_files["Gemfile"]:
        if _try_command(["bundle", "install"], "bundle install"):
            return

    # Rust projects
    if project_files["Cargo.toml"]:
        if _try_command(["cargo", "build"], "cargo build", timeout=300):
            return

    # Go projects
    if project_files["go.mod"]:
        if _try_command(["go", "mod", "download"], "go mod download"):
            return

    # Java/Maven projects
    if project_files["pom.xml"]:
        if _try_command(
            ["mvn", "dependency:resolve"], "mvn dependency:resolve", timeout=300
        ):
            return

    # Gradle projects
    if project_files["build.gradle"]:
        if _try_command(
            ["./gradlew", "dependencies"], "gradle dependencies", timeout=300
        ):
            return

    logger.info(
        "[CIBench] [WARN] Could not install dependencies - agent will work without them"
    )


def _create_isolated_venv_if_needed(
    testbed_path: Path, venv_name: str, logger: Any
) -> Tuple[str, str]:
    """
    Create an isolated virtual environment for the issue if not already present.

    This ensures each issue has its own isolated Python environment,
    preventing dependency conflicts between different issues even on the same repo.

    The venv is created as .venv-<repo_name>-<issue_id> inside the testbed directory.
    Agent will handle dependency installation when needed.

    Args:
        testbed_path: Path to the testbed directory
        venv_name: Full venv name (e.g., ".venv-crewai-102")
        logger: Logger instance

    Returns:
        Tuple of (path to venv's Python binary, venv_name) for activation
    """
    import sys
    import subprocess

    # Use provided venv name (already includes repo and issue ID)
    # CRITICAL: Ensure venv_path is absolute to avoid creation in wrong location
    venv_path = (testbed_path / venv_name).resolve()
    venv_python = venv_path / "bin" / "python"

    # DEBUG: Log the actual paths being used
    logger.info(f"[CIBench] DEBUG: testbed_path = {testbed_path}")
    logger.info(f"[CIBench] DEBUG: venv_name = {venv_name}")
    logger.info(f"[CIBench] DEBUG: venv_path (absolute) = {venv_path}")

    # Skip if venv already exists and is valid
    if venv_path.exists() and venv_python.exists():
        logger.info(f"[CIBench] Using existing virtual environment: {venv_name}")
        return str(venv_python), venv_name

    logger.info(f"[CIBench] Creating isolated virtual environment: {venv_name}")

    try:
        # Create virtual environment (fast, no dependencies)
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            cwd=str(testbed_path),
            capture_output=True,
            text=True,
            timeout=60,
            check=True
        )

        logger.info(f"[CIBench] ✓ Virtual environment ready: {venv_name}")
        logger.info(f"[CIBench]   Agent can install dependencies as needed using:")
        logger.info(f"[CIBench]   source {venv_name}/bin/activate && pip install <package>")

        # DEBUG: Verify the venv actually exists at the expected location
        if venv_path.exists():
            logger.info(f"[CIBench] DEBUG: Venv verified at: {venv_path.resolve()}")
        else:
            logger.warning(f"[CIBench] DEBUG: WARNING - Venv NOT found at expected path: {venv_path.resolve()}")

        return str(venv_python), venv_name

    except subprocess.TimeoutExpired:
        logger.warning(f"[CIBench] Virtual environment creation timed out (non-blocking)")
        return "", ""
    except Exception as e:
        logger.warning(f"[CIBench] Could not create virtual environment (non-blocking): {e}")
        return "", ""


def setup_local_environment(
    config: Dict[str, Any],
    instance: Dict[str, Any],
    instance_dir: Path,
) -> Tuple[LocalEnvironment, Path]:
    """
    Prepare a local repository checkout and return *(env, testbed_path)*.

    - Keeps a shared network clone cache at ``<project_root>/repo/``.
    - Creates ``<instance_dir>/testbed/`` from that cache when needed.
    - Checks out the failing commit (``sha_fail``).
    - **Auto-creates an isolated .venv-<repo_name>-<issue_id> per issue.**
    - Venv persists across reruns (preserved by git clean -e .venv*).
    - Agent handles all dependency installation as needed.
    - Wraps the directory in a ``LocalEnvironment`` using settings from the
      ``environment`` section of the config (timeout, interpreter, env vars).

    If the testbed directory already contains a ``.git`` folder (re-run),
    the local working copy is reused so previous work is preserved, including
    the issue-specific virtual environment and any agent-installed packages.
    """
    repo_owner = instance.get("repo_owner", "")
    repo_name = instance.get("repo_name", "")
    sha_fail = str(instance.get("sha_fail") or "")

    # CRITICAL: Always use absolute paths to avoid venv creation in wrong location
    testbed_path = (instance_dir / "testbed").resolve()
    cache_dir_name = f"{repo_owner}__{repo_name}"
    repo_cache_path = REPO_CACHE_ROOT / cache_dir_name
    clone_url = f"https://github.com/{repo_owner}/{repo_name}.git"

    def _run_git(
        args: list[str], cwd: Path, timeout: int = 300
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _ensure_commit_available(
        repo_path: Path, commit: str, remote: str = "origin"
    ) -> None:
        if not commit:
            return
        verify = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repo_path)
        if verify.returncode == 0:
            return
        fetch = _run_git(["fetch", "--quiet", remote, commit], repo_path)
        if fetch.returncode != 0:
            raise RuntimeError(
                f"git fetch failed for commit {commit} in {repo_path}:\n"
                f"{fetch.stderr[:800]}"
            )
        verify = _run_git(["rev-parse", "--verify", f"{commit}^{{commit}}"], repo_path)
        if verify.returncode != 0:
            raise RuntimeError(
                f"commit {commit} still unavailable in {repo_path} after fetch:\n"
                f"{verify.stderr[:800]}"
            )

    def _must_run_git(
        args: list[str], cwd: Path, error_label: str, timeout: int = 300
    ) -> None:
        result = _run_git(args, cwd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"{error_label} in {cwd}:\n{result.stderr[:800] or result.stdout[:800]}"
            )

    def _prepare_worktree(repo_path: Path, commit: str) -> None:
        if not commit:
            raise RuntimeError(
                f"Missing sha_fail for worktree preparation in {repo_path}"
            )
        _must_run_git(
            ["config", "user.email", "ci-repair@mini-swe-agent"],
            repo_path,
            "git config user.email failed",
        )
        _must_run_git(
            ["config", "user.name", "mini-swe-agent"],
            repo_path,
            "git config user.name failed",
        )
        _must_run_git(
            ["checkout", "--detach", "--force", commit],
            repo_path,
            f"git checkout {commit} failed",
        )
        _must_run_git(
            ["reset", "--hard", commit],
            repo_path,
            f"git reset --hard {commit} failed",
        )
        _must_run_git(
            ["clean", "-fdx", "-e", ".venv*"],
            repo_path,
            "git clean -fdx (preserving .venv*)",
        )

    # ── Shared repo cache ─────────────────────────────────────────────────────
    REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not (repo_cache_path / ".git").exists():
        logger.info(
            "[CIBench] Cloning %s -> shared cache %s", clone_url, repo_cache_path
        )

        def _do_clone():
            result = subprocess.run(
                ["git", "clone", "--quiet", clone_url, str(repo_cache_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                error_msg = result.stderr[:800]
                # Check if it's a network-related error that we should retry
                if any(
                    err in error_msg.lower()
                    for err in [
                        "could not resolve host",
                        "connection",
                        "network",
                        "timeout",
                        "temporary failure",
                    ]
                ):
                    raise RuntimeError(
                        f"Network error during git clone for {repo_owner}/{repo_name}:\n{error_msg}"
                    )
                else:
                    # Non-network error, don't retry
                    raise RuntimeError(
                        f"git clone failed for {repo_owner}/{repo_name}:\n{error_msg}"
                    )
            return result

        _retry_with_backoff(
            _do_clone,
            max_retries=3,
            initial_delay=2.0,
            backoff_multiplier=2.0,
            exceptions=(RuntimeError,),
        )
    else:
        logger.info("[CIBench] Reusing shared cache at %s", repo_cache_path)

        def _do_fetch():
            fetch_result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_cache_path),
                    "fetch",
                    "--all",
                    "--tags",
                    "--prune",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if fetch_result.returncode != 0:
                error_msg = fetch_result.stderr[:800]
                # Check if it's a network-related error
                if any(
                    err in error_msg.lower()
                    for err in [
                        "could not resolve host",
                        "connection",
                        "network",
                        "timeout",
                        "temporary failure",
                    ]
                ):
                    raise RuntimeError(
                        f"Network error during git fetch for {repo_cache_path}:\n{error_msg}"
                    )
                else:
                    # Non-network error, just log warning
                    logger.warning(
                        "[CIBench] git fetch failed for shared cache %s:\n%s",
                        repo_cache_path,
                        error_msg,
                    )
            return fetch_result

        try:
            _retry_with_backoff(
                _do_fetch,
                max_retries=3,
                initial_delay=2.0,
                backoff_multiplier=2.0,
                exceptions=(RuntimeError,),
            )
        except RuntimeError:
            # Fetch failure is not critical, we can continue with cached version
            pass
    _ensure_commit_available(repo_cache_path, sha_fail, remote="origin")

    # ── Per-instance working copy ─────────────────────────────────────────────
    if not (testbed_path / ".git").exists():
        testbed_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[CIBench] Creating local working copy %s -> %s",
            repo_cache_path,
            testbed_path,
        )
        result = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                str(repo_cache_path),
                str(testbed_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"local git clone failed for {repo_owner}/{repo_name} from cache:\n"
                f"{result.stderr[:800]}"
            )
    else:
        logger.info("[CIBench] Reusing existing working copy at %s", testbed_path)
    _ensure_commit_available(testbed_path, sha_fail, remote="origin")
    _prepare_worktree(testbed_path, sha_fail)

    # ── Auto-create isolated virtual environment for this issue ────────────────
    # Create a unique venv per issue to avoid dependency conflicts between issues
    instance_id = str(instance.get("instance_id", "")).replace("/", "_").replace("@", "_")
    issue_venv_name = f".venv-{repo_name}-{instance_id}"

    # DEBUG: Log what we're passing to venv creation
    logger.info(f"[CIBench] DEBUG: About to create venv with:")
    logger.info(f"[CIBench] DEBUG:   testbed_path = {testbed_path}")
    logger.info(f"[CIBench] DEBUG:   testbed_path (absolute) = {testbed_path.resolve()}")
    logger.info(f"[CIBench] DEBUG:   issue_venv_name = {issue_venv_name}")

    venv_python, venv_name = _create_isolated_venv_if_needed(testbed_path, issue_venv_name, logger)

    # No pre-installation - agent handles dependencies as needed
    logger.info(
        "[CIBench] Agent will install dependencies as needed in the isolated venv"
    )

    # ── Build LocalEnvironment ────────────────────────────────────────────────
    # Strip environment_class from the env config dict (LocalEnvironment doesn't accept it).
    env_cfg: Dict[str, Any] = {
        k: v
        for k, v in config.get("environment", {}).items()
        if k != "environment_class"
    }
    env_cfg["cwd"] = str(testbed_path)
    env_cfg.setdefault("timeout", 120)

    # Activate isolated venv if it was created
    if venv_python and venv_name:
        import os
        # CRITICAL: Use absolute path for venv activation
        venv_path = (testbed_path / venv_name).resolve()
        venv_bin = venv_path / "bin"

        # Set up environment variables to activate venv
        env_vars = env_cfg.get("env", {})
        env_vars["VIRTUAL_ENV"] = str(venv_path)
        env_vars["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        # Unset PYTHONHOME to avoid conflicts
        env_vars["PYTHONHOME"] = ""
        env_cfg["env"] = env_vars

        logger.info(f"[CIBench] Activated issue-specific virtual environment: {venv_name}")
        logger.info(f"[CIBench] DEBUG: Activated venv at: {venv_path}")

    env = LocalEnvironment(**env_cfg)

    # ── Git setup + checkout sha_fail ─────────────────────────────────────────
    startup_tpl = config.get("run", {}).get("startup_command", "")
    if startup_tpl:
        try:
            rendered = Template(startup_tpl, undefined=StrictUndefined).render(
                **instance
            )
        except Exception:
            # Fallback: manual substitution if instance has unexpected keys
            rendered = startup_tpl.replace("{{sha_fail}}", sha_fail)
        out = env.execute({"command": rendered})
        if out.get("returncode", 0) != 0:
            raise RuntimeError(
                f"startup command failed for {instance.get('instance_id')}:\n"
                f"{out.get('output', '')[:800]}"
            )

    return env, testbed_path


# ─────────────────────────────────────────────────────────────────────────────
# Sequential Repair: Fix problems one at a time
# ─────────────────────────────────────────────────────────────────────────────


def _structured_strategy_text(strategy: Any) -> str:
    """Render structured repair guidance without discarding its fields."""
    if not strategy:
        return ""
    if not isinstance(strategy, dict):
        return str(strategy)

    lines = []
    summary = str(strategy.get("summary") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")

    actions = strategy.get("actions") or []
    if not isinstance(actions, list):
        actions = [actions]
    for index, action in enumerate(actions, 1):
        if action:
            lines.append(f"Action {index}: {action}")

    validation_cmd = str(
        strategy.get("validation_cmd")
        or strategy.get("verification_cmd")
        or ""
    ).strip()
    if validation_cmd:
        lines.append(f"Validate: {validation_cmd}")

    pitfalls = strategy.get("pitfalls") or []
    if not isinstance(pitfalls, list):
        pitfalls = [pitfalls]
    for pitfall in pitfalls:
        if pitfall:
            lines.append(f"Avoid: {pitfall}")

    return "\n".join(lines)


def _analyze_repair_with_llm(
    problem: Dict[str, Any], context_llm: Any
) -> Dict[str, Any]:
    """Analyze problem and generate repair plan (automated tool or manual fix)."""

    problem_statement = problem.get("problem_statement") or problem.get("problem", "")
    error_details = problem.get("error_details") or problem.get("failure_signals", [])
    root_cause = problem.get("root_cause", "")
    fix_strategy = problem.get("fix_strategy") or _structured_strategy_text(
        problem.get("repair_strategy")
    )
    files = problem.get("files", [])

    file_items = files if isinstance(files, list) else [files]
    file_lines = []
    for file_entry in file_items:
        if isinstance(file_entry, dict):
            path = file_entry.get("path") or file_entry.get("file")
            if path:
                file_lines.append(f"  - {path}")
        elif file_entry:
            file_lines.append(f"  - {file_entry}")
    files_str = "\n".join(file_lines) if file_lines else "  (not specified)"

    detail_lines = []
    for detail in error_details if isinstance(error_details, list) else [error_details]:
        if isinstance(detail, dict):
            detail_text = "; ".join(
                f"{key}: {value}" for key, value in detail.items() if value
            )
            if detail_text:
                detail_lines.append(f"  - {detail_text}")
        elif detail:
            detail_lines.append(f"  - {detail}")
    error_details_str = "\n".join(detail_lines)

    # Build prompt
    prompt_parts = [f"PROBLEM:\n{problem_statement}"]
    if error_details_str:
        prompt_parts.append(f"ERROR DETAILS:\n{error_details_str}")
    if root_cause:
        prompt_parts.append(f"ROOT CAUSE:\n{root_cause}")
    if fix_strategy:
        prompt_parts.append(f"PREVIOUS EXPERIENCE:\n{fix_strategy}")
    prompt_parts.append(f"FILES:\n{files_str}")
    prompt_parts.append(f"AUTOMATED TOOLS:\n{json.dumps(AUTOMATED_TOOLS, indent=2)}")

    prompt = f"""Analyze this CI problem and root cause, then return the shortest complete repair plan.

{chr(10).join(prompt_parts)}

INSTRUCTIONS:
1. Work with the provided sections; some may be missing
2. **CRITICAL - PREVIOUS EXPERIENCE MUST BE USED**:
   - PREVIOUS EXPERIENCE contains actual successful repairs for similar problems
   - Check if any files in FILES are mentioned in PREVIOUS EXPERIENCE's "How it was fixed" section
   - If a file is mentioned with specific changes (e.g., "exit_code_test.py: Refactored title extraction logic to split file content into lines"), you MUST use that EXACT approach
   - Your Edit instruction must describe the same changes, not a different fix
   - Example: PREVIOUS EXPERIENCE says "Refactored title extraction: split entire file content into lines, access second line as title with fallback to first line" → Your Edit line MUST say "Refactor title extraction logic: split file content into lines, access lines[1] as title with fallback to lines[0]"
   - DO NOT ignore PREVIOUS EXPERIENCE - it shows proven solutions
3. First infer the actual root-cause change from PROBLEM, ERROR DETAILS, ROOT CAUSE, and PREVIOUS EXPERIENCE; use FILES only as supporting evidence or tool targets
4. Distinguish root-cause files from affected validation files:
   - Root-cause files are explicitly named as needing a value, dependency, API, logic, or content change
   - Affected validation files are files that failed or would be reformatted after the root-cause fix
   - Do not create manual edits for affected validation files unless the evidence says what content or logic must change there
4. Treat file extensions as hints for possible tools, not proof that a tool can fix the issue
5. For each affected file or path, classify the required work as:
   - automated: a listed tool can safely make the required edit without knowing project intent
   - manual: the fix requires intent, API knowledge, dependency/config decisions, content changes, or code semantics
   - cleanup: an automated tool should run only after manual edits to format or sort the result
6. Choose the repair type from those classifications:
   - automated_tool: every required change is automated; no manual edit or semantic decision is needed
   - manual_fix: every core change is manual; automated tools, if any, are only cleanup after edits
   - hybrid: at least one core change is manual and at least one separate core change is automated
7. Use automated_tool only when the evidence says the issue is purely mechanical, such as formatting, import sorting, whitespace, markdown/RST heading formatting, or another listed tool fix
8. Do NOT use automated_tool for syntax errors, parser failures, runtime exceptions, missing names/imports, wrong values, broken links/refs requiring target selection, version/config changes, or failed assertions unless the evidence explicitly says a listed tool can repair that exact issue
9. Choose the tool that can repair the issue, not merely the tool that reported it; if the reporter cannot parse the file or cannot infer intent, the core fix is manual
10. When using automated tools, select only from AUTOMATED TOOLS and match by both purpose and file_pattern
11. If the root cause is a formatter/tool version in config files, manually edit only the config files with the version change, then run the relevant formatter on the affected files it controls
12. If .py files are listed but the evidence is about doc formatting or generated formatting output, prefer an automated formatter that matches the evidence; choose a manual Python edit only when the evidence names a Python logic/content change
13. If manual Python edits are required, run ruff cleanup afterward; if Python files only need formatting/docstring cleanup, treat that as automated rather than manual
14. For automated_tool, choose the Run target from FILES first; if FILES is missing, use paths from PROBLEM/ERROR DETAILS/ROOT CAUSE
15. For automated_tool, if one file is affected use that file; if multiple files share a narrow directory use that directory; otherwise list concrete paths
16. For automated_tool, output Install and Run lines using exact install_command and fix_command from AUTOMATED TOOLS, replacing {{file_or_dir}} with concrete targets
17. NEVER output placeholders like <tool>, <path>, {{file_or_dir}}; always use concrete values from the evidence
18. For manual_fix, include Locate, Analyze, Change, Impact, and Edit lines
19. Locate must say how to find all occurrences using symbols, error text, config keys, file paths, or line numbers from the evidence
20. Edit must list each required file:line when available; otherwise list the concrete file or directory pattern
21. Add automated cleanup commands for formatting AFTER manual edits:
    - Python cleanup: "Install: pip install ruff" and "Cleanup: ruff check --fix <concrete targets>"
    - RST cleanup: use docstrfmt only when the manual RST edit can benefit from formatting
    - TOML/Markdown cleanup: use taplo/mdformat only for formatting cleanup, not value/content decisions
    - IMPORTANT: Cleanup commands run AFTER the manual edit is made, not before
    - If cleanup fails due to missing dependencies, the manual fix is still valid
22. For hybrid, put manual core edits first, then automated core fixes or cleanup grouped by tool and target
23. Keep the plan short and actionable; do not include validation commands, test commands, markdown headings, or explanatory background

OUTPUT FORMAT:
Return ONLY the raw JSON object. Do NOT wrap in markdown code fences (```json). Do NOT add explanations.

Automated tool only:
{{
  "type": "automated_tool",
  "repair_plan": "Install: exact install_command\\nRun: exact fix_command with concrete target"
}}

Manual fix only:
{{
  "type": "manual_fix",
  "repair_plan": "Locate: ...\\nAnalyze: ...\\nChange: ...\\nImpact: ...\\nEdit: concrete file:line or path\\nInstall: cleanup install command if needed\\nCleanup: cleanup command if needed"
}}

Hybrid:
{{
  "type": "hybrid",
  "repair_plan": "MANUAL:\\nLocate: ...\\nAnalyze: ...\\nEdit: concrete file:line or path\\n\\nAUTOMATED:\\nInstall: exact install_command(s)\\nRun: exact fix_command(s) with concrete targets"
}}"""

    try:
        response = context_llm(prompt).strip()
        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback 1: Try demjson3 for more lenient parsing
            try:
                return demjson3.decode(response)
            except Exception:
                pass

            # Fallback 2: Extract JSON from markdown fences if present
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}")

    # Fallback to manual fix
    return {
        "type": "manual_fix",
        "repair_plan": fix_strategy or "Analyze error and apply fix",
    }


def _build_single_problem_task(
    problem: Dict[str, Any],
    problem_num: int,
    total_problems: int,
    repo_name: str,
    issue_id: str = "",
) -> str:
    """Build focused repair task with automated tool detection or manual repair instructions."""

    # Sanitize issue_id for filesystem-safe venv name (must match setup_local_environment)
    if issue_id:
        issue_id = issue_id.replace("/", "_").replace("@", "_")

    # Extract problem fields
    problem_statement = problem.get("problem_statement") or problem.get("problem", "")
    error_details = problem.get("error_details") or problem.get("failure_signals", [])
    root_cause = problem.get("root_cause", "")
    fix_strategy = problem.get("fix_strategy") or _structured_strategy_text(
        problem.get("repair_strategy")
    )
    files = problem.get("files", [])
    issue_type = problem.get("issue_type") or problem.get("problem_type", "")
    failure_type = problem.get("failure_type") or problem.get("error_type") or ""
    verification_cmd = (
        problem.get("verification_cmd")
        or problem.get("validation_cmd")
        or ""
    )

    # Combine problem_statement and error_details if both exist
    full_problem = problem_statement
    if error_details and isinstance(error_details, list):
        for detail in error_details:
            if isinstance(detail, dict):
                msg = detail.get("message", "")
                if msg and msg not in full_problem:
                    full_problem += f"\n{msg}"
            elif detail and str(detail) not in full_problem:
                full_problem += f"\n{detail}"

    repair_type = "manual"
    repair_plan = fix_strategy or "Analyze error and apply fix"

    # Format repair plan based on type
    if repair_type == "automated_tool":
        # Add strong directives for automated tool repairs
        repair_plan_formatted = f"""**AUTOMATED FIX - DO NOT EDIT FILES MANUALLY**

This problem can be solved completely by running these commands:

{repair_plan}

[WARN] CRITICAL INSTRUCTIONS:
1. First inspect the tool configuration and confirm that every target path exists
2. Confirm the reported failure is within that tool's configured scope
3. If the tool is unavailable, install it in the repository-local environment
   or tool cache rather than globally; preserve the requested version when one
   is specified
4. Execute the Run/Fix command with the verified target paths
5. Inspect the resulting diff and run the relevant check command when feasible
6. DO NOT manually edit files when the confirmed problem is purely mechanical

The automated tool should make the mechanical changes, but you must verify its
scope, output, and resulting diff before finishing."""
    else:
        # Manual fix or hybrid - use plan as-is
        repair_plan_formatted = repair_plan

    # Format files list
    files_str = (
        "\n".join(f"- {f}" for f in (files if isinstance(files, list) else [files]))
        if files
        else "N/A"
    )

    return f"""Problem {problem_num}/{total_problems}

{failure_type} | {issue_type}
{full_problem}

Root Cause: {root_cause}

Affected Files:
{files_str}

Repair Plan:
{repair_plan_formatted}

Validation Command:
{verification_cmd or "Determine from the CI workflow and run the relevant check."}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: ISOLATED ENVIRONMENT - USE REPO-SPECIFIC VIRTUAL ENVIRONMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This issue has an ISOLATED virtual environment: .venv-{repo_name}{f"-{issue_id}" if issue_id else ""}/

⚠️  MANDATORY: ALL installations and executions MUST use this venv!

For INSTALLING packages (if repair plan says "install X"):
  ✓ CORRECT:   .venv-{repo_name}{f"-{issue_id}" if issue_id else ""}/bin/pip install <package>
  ✗ WRONG:     pip install <package>  # This affects other issues!

For RUNNING Python scripts:
  ✓ CORRECT:   .venv-{repo_name}{f"-{issue_id}" if issue_id else ""}/bin/python script.py
  ✗ WRONG:     python script.py  # May use wrong environment!

For RUNNING commands (pytest, mypy, flake8, etc.):
  ✓ CORRECT:   source .venv-{repo_name}{f"-{issue_id}" if issue_id else ""}/bin/activate && pytest
  ✗ WRONG:     pytest  # May use globally installed version!

WHY THIS MATTERS:
- Each issue has its own isolated environment
- Installing in one venv does NOT affect other issues
- What you install for THIS issue stays in THIS issue only
- Prevents dependency conflicts between different issues/repos

ALWAYS activate before running ANY command:
  source .venv-{repo_name}{f"-{issue_id}" if issue_id else ""}/bin/activate && <your command>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: If this is an infrastructure problem that CANNOT be fixed by code changes, skip it immediately:
- Rate limiting (HTTP 429, API limits)
- Network timeouts or connection failures
- External service outages (GitHub Actions, Docker Hub, npm registry)
- Download failures for external dependencies/actions
- Transient errors not caused by code

For infrastructure problems: Respond with "This is an infrastructure issue, not fixable by code" and submit an empty patch.
"""


def _combine_partial_fixes(partial_fixes: List[Dict[str, Any]]) -> str:
    """
    Combine multiple partial fixes into a unified diff.
    """
    if not partial_fixes:
        return ""

    unified = []
    for fix in partial_fixes:
        patch = fix.get("patch", "")
        if patch:
            # Don't add comments - just append patch directly for valid unified diff
            unified.append(patch)

    # Join with double newline to separate patches cleanly
    return "\n\n".join(unified)


def _collect_final_workspace_diff(testbed_path: Path) -> str:
    """
    Return one final diff from the checkout after all sequential repair steps.

    Partial per-problem submissions are not composable when multiple problems
    touch the same file. The checkout is the source of truth because each agent
    run mutates the same working tree.

    CRITICAL FIX: Handle new files properly by adding them as intent-to-add
    so they appear in the diff with proper 'new file mode' headers.
    """
    try:
        # CRITICAL: Add untracked files as "intent to add" so they appear in diff
        # This ensures new files created by the agent get proper "new file mode" headers
        # Without this, patches fail with "does not exist in index" errors
        add_result = subprocess.run(
            ["git", "add", "-N", "."],
            cwd=testbed_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if add_result.returncode != 0:
            logger.warning(
                "[CIBench] git add -N failed (non-critical): %s",
                add_result.stderr or add_result.stdout,
            )

        # Generate diff with full index info to properly handle new files
        result = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "--full-index"],
            cwd=testbed_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        logger.warning("[CIBench] Failed to collect final workspace diff: %s", exc)
        return ""

    if result.returncode != 0:
        logger.warning(
            "[CIBench] git diff failed while collecting final workspace diff: %s",
            result.stderr or result.stdout,
        )
        return ""

    diff = result.stdout or ""

    # Validate the generated patch can actually apply
    if diff:
        validation_result = subprocess.run(
            ["git", "apply", "--check", "--3way"],
            input=diff,
            text=True,
            cwd=testbed_path,
            capture_output=True,
            timeout=30,
        )
        if validation_result.returncode != 0:
            logger.warning(
                "[CIBench] Generated patch may not apply cleanly:\n%s",
                validation_result.stderr[:500],
            )
            # Log but don't fail - the patch might still work with different apply flags
            # or the validation might be overly strict

    parts = diff.split("\n")
    while parts and parts[-1] == "":
        parts.pop()
    return ("\n".join(parts) + "\n") if parts else ""


def _analyze_and_group_problems(
    problems: List[Dict[str, Any]], context_llm: Any
) -> List[Dict[str, Any]]:
    """
    Two-step grouping:
    1. Pre-group by validation_cmd + failure_type
    2. LLM decides if each group can merge
    """
    if not problems or len(problems) <= 1:
        return problems

    logger.info(f"[CIBench] Analyzing {len(problems)} problems for grouping...")

    # Step 1: Pre-group by validation_cmd + failure_type
    groups = {}
    for problem in problems:
        validation_cmd = problem.get("verification_cmd", "")
        failure_type = problem.get("failure_type") or problem.get("error_type", "")
        group_key = f"{validation_cmd}|{failure_type}"

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(problem)

    logger.info(f"[CIBench] Pre-grouped into {len(groups)} groups by type")

    # Step 2: For each group, ask LLM if they can merge
    final_problems = []
    for group_key, group_problems in groups.items():
        if len(group_problems) == 1:
            # Single problem, no merge needed
            final_problems.append(group_problems[0])
        else:
            # Multiple problems, ask LLM if they can merge
            logger.info(
                f"[CIBench] Analyzing group with {len(group_problems)} problems..."
            )
            merged = _analyze_group_for_merge(group_problems, context_llm)
            final_problems.extend(merged)

    logger.info(
        f"[CIBench] After grouping: {len(problems)} → {len(final_problems)} problems"
    )
    return final_problems


def _analyze_group_for_merge(
    group_problems: List[Dict[str, Any]], context_llm: Any
) -> List[Dict[str, Any]]:
    """
    LLM analyzes if problems in group are similar/related and can be merged.
    Returns: list of problems (merged or separate)
    """
    if not context_llm:
        return group_problems

    # Build analysis prompt
    problems_summary = []
    for i, p in enumerate(group_problems, 1):
        problems_summary.append(
            {
                "id": i,
                "problem_statement": p.get("problem_statement", "")[:200],
                "root_cause": p.get("root_cause", "")[:200],
                "files": p.get("files", []),
            }
        )

    prompt = f"""Analyze if these {len(group_problems)} problems can be merged into one.

PROBLEMS (same validation type and failure type):
{json.dumps(problems_summary, indent=2)}

TASK: Decide if these problems are similar and related enough to fix together.

Merge if:
- Same root cause (e.g., all IndexError, all type errors)
- Related files (same file or related files)
- Can be fixed with ONE unified approach

[WARN] CRITICAL: Do NOT merge if:
- Different root causes
- Unrelated files
- Require different fix approaches
- **Solutions are MUTUALLY EXCLUSIVE** (e.g., one says "change code", other says "change config")
- **Solutions CONFLICT** (e.g., "fix by replacing type A with B" vs "fix by adding plugin")
- One problem says "Alternative fix path" or "Option 1/Option 2"

CONFLICTING SOLUTIONS CAUSE AGENT TO GET STUCK IN INFINITE LOOP!
If solutions conflict, MUST keep separate.

OUTPUT (JSON only):
{{
  "can_merge": true/false,
  "reason": "why they can/cannot merge (mention if solutions conflict)"
}}"""

    try:
        response = context_llm(prompt)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group(0))
            if analysis.get("can_merge"):
                logger.info(
                    f"[CIBench] OK Merging {len(group_problems)} problems: {analysis.get('reason')}"
                )
                return [_merge_problems(group_problems)]
            else:
                logger.info(
                    f"[CIBench] FAIL Keeping separate: {analysis.get('reason')}"
                )
                return group_problems
    except Exception as e:
        logger.warning(f"[CIBench] LLM merge analysis failed: {e}")

    # Fallback: keep separate
    return group_problems


def _merge_problems(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple problems into one combined problem.
    Combines: problem_statement, root_cause, files, fixes
    """
    merged = {
        "problem_id": f"merged_{problems[0].get('problem_id', 1)}",
        "source": problems[0].get("source", ""),
        "verification_cmd": problems[0].get("verification_cmd", ""),
        "error_type": problems[0].get("error_type", ""),
        "failure_type": problems[0].get("failure_type", ""),
        "issue_type": problems[0].get("issue_type", ""),
    }

    # Combine problem statements
    statements = [
        p.get("problem_statement", "") for p in problems if p.get("problem_statement")
    ]
    merged["problem_statement"] = "Multiple related issues:\n" + "\n".join(
        f"{i + 1}. {s}" for i, s in enumerate(statements)
    )

    # Combine root causes
    causes = [p.get("root_cause", "") for p in problems if p.get("root_cause")]
    if causes:
        merged["root_cause"] = "\n".join(set(causes))  # Deduplicate
    else:
        merged["root_cause"] = ""

    # Combine fix strategies
    fixes = [p.get("fix_strategy", "") for p in problems if p.get("fix_strategy")]
    if fixes:
        merged["fix_strategy"] = "\n".join(set(fixes))
    else:
        merged["fix_strategy"] = ""

    # Combine files (deduplicate)
    all_files = []
    for p in problems:
        files = p.get("files", [])
        if isinstance(files, list):
            all_files.extend(files)
    merged["files"] = list(set(all_files))

    # Combine error details
    all_errors = []
    for p in problems:
        errors = p.get("error_details", [])
        if isinstance(errors, list):
            all_errors.extend(errors)
    merged["error_details"] = all_errors

    logger.info(f"[CIBench] Merged {len(problems)} problems into 1")
    return merged


def _run_sequential_repair(
    agent: Any,
    problems: List[Dict[str, Any]],
    testbed_path: Path,
    progress_manager: Any,
    instance_id: str,
    repo_name: str = "",
) -> Tuple[Dict[str, Any], str]:
    """
    Fix problems sequentially, one at a time.

    For each problem:
    1. Build focused task (ONLY this problem)
    2. Agent fixes THIS problem
    3. Verify validation passes
    4. Save partial fix or stop

    Supports two problem formats:
    - Old format: {status, verification_cmd, validation_stage, check_first, ...}
    - New format: {problem_id, is_primary, problem_statement, repair_plan, verification}

    Args:
        installation_cmd: Optional list of installation commands to run before validation

    Returns:
        (info_dict, unified_diff)
    """
    partial_fixes = []
    total_problems_initial = len(problems)
    per_problem_timeout = 600  # 10 minutes per problem

    logger.info(
        f"[CIBench] Starting sequential repair: {total_problems_initial} problems"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # NOTE: Agent handles all validation internally
    # ═══════════════════════════════════════════════════════════════════════════
    # The agent will:
    # - Check if problems exist
    # - Identify what needs fixing
    # - Skip or fix based on current state
    # - Handle validation with proper environment setup
    #
    # We only keep immediate exit detection and consecutive failure detection
    # to prevent cascade failures.

    total_problems = total_problems_initial

    for i, problem in enumerate(problems, 1):
        logger.info(f"[CIBench] {'=' * 60}")
        logger.info(f"[CIBench] Problem {i}/{total_problems}")
        logger.info(f"[CIBench] {'=' * 60}")
        # Support both old and new problem formats

        # New simplified format
        problem_id = problem.get("problem_id", i)
        source = str(problem.get("source", "")).strip()
        # No more nested verification dict - fields are flat
        validation_cmd = problem.get("verification_cmd", "")
        # No check_first - agent decides on its own
        status = "CI FAILURE" if source == "ci failure" else "PREVIOUS EXPERIENCE"
        stage = ""

        logger.info(f"[CIBench] Problem ID: {problem_id}")
        logger.info(f"[CIBench] Source: {source or 'unknown'}")
        # problem_statement is now a string, not a dict
        problem_desc = problem.get("problem_statement", "")
        if isinstance(problem_desc, str):
            logger.info(f"[CIBench] Description: {problem_desc[:100]}")
        else:
            # Very old format (dict) - shouldn't happen with our new code
            logger.info(
                f"[CIBench] Description: {problem_desc.get('description', '')[:100]}"
            )

        logger.info(f"[CIBench] Status: {status}")
        logger.info(f"[CIBench] Stage: {stage}")
        logger.info(f"[CIBench] Validation: {validation_cmd}")

        # Build focused task for THIS problem only
        single_task = _build_single_problem_task(
            problem, i, total_problems, repo_name, instance_id
        )

        logger.info("[CIBench] Task preview (first 300 chars):")
        logger.info(single_task[:300] + "...")

        # Update progress
        progress_manager.update_instance_status(
            instance_id, f"Fixing problem {i}/{total_problems}"
        )

        # Agent fixes THIS problem (with timeout to avoid getting stuck)
        logger.info(f"[CIBench] Running agent for problem {i}...")

        try:
            start_time = time.time()
            previous_start_time = getattr(agent, "_start_time", start_time)
            previous_wall_limit = getattr(agent.config, "wall_time_limit_seconds", 0)

            # Each atomic problem is a fresh agent run in the same repaired
            # workspace. Reset per-run counters so an earlier problem cannot
            # exhaust the next problem's step/cost budget.
            agent.n_calls = 0
            agent.cost = 0.0

            # ALWAYS set timeout for THIS problem (don't rely on previous state)
            agent.config.wall_time_limit_seconds = per_problem_timeout
            agent._start_time = start_time

            try:
                info = agent.run(single_task)
            finally:
                # Restore original timeout (for next problem to start fresh)
                agent.config.wall_time_limit_seconds = previous_wall_limit
                agent._start_time = previous_start_time

            elapsed = time.time() - start_time
            logger.info(f"[CIBench] Problem {i} completed in {elapsed:.1f}s")

            # ═══════════════════════════════════════════════════════════════
            # CRITICAL: Detect immediate exit (agent state corruption)
            # ═══════════════════════════════════════════════════════════════
            if elapsed < 1.0:
                logger.error(
                    f"[CIBench] [WARN] Problem {i} completed in {elapsed:.3f}s - SUSPICIOUS IMMEDIATE EXIT!"
                )
                logger.error(
                    "[CIBench]    This indicates agent state corruption or configuration issue"
                )
                logger.error(f"[CIBench]    Exit status: {info.get('exit_status')}")
                logger.error(
                    f"[CIBench]    Submission length: {len(info.get('submission', ''))}"
                )
                logger.error(
                    f"[CIBench]    Agent start_time: {getattr(agent, '_start_time', 'unknown')}"
                )
                logger.error(
                    f"[CIBench]    Agent timeout: {getattr(agent.config, 'wall_time_limit_seconds', 'unknown')}"
                )
                logger.error(
                    "[CIBench] [WARN] STOPPING sequential repair to prevent cascade failures"
                )
                # Break the loop - agent is broken, don't try more problems
                break

            exit_status = info.get("exit_status")

            # Extract patch
            raw_submission = info.get("submission") or ""
            patch = _extract_diff(raw_submission)

            if not patch or not patch.strip():
                logger.warning(f"[CIBench] SKIP Problem {i}: Agent returned no patch")
                logger.warning(f"[CIBench]    Exit status: {exit_status}")
                logger.warning(f"[CIBench]    Elapsed time: {elapsed:.1f}s")
                logger.warning(f"[CIBench]    Submission length: {len(raw_submission)}")

                # If we've had multiple consecutive no-patch failures, stop
                recent_failures = sum(
                    1 for f in partial_fixes[-3:] if f.get("patch", "").strip() == ""
                )
                if recent_failures >= 3:
                    logger.error(
                        "[CIBench] [WARN] 3+ consecutive no-patch failures - agent may be broken"
                    )
                    logger.error("[CIBench] [WARN] STOPPING to prevent wasted time")
                    break

                # Don't break - continue to next problem
                continue

            logger.info(
                f"[CIBench] OK Problem {i}: Agent generated patch ({len(patch)} chars)"
            )

            # Save patch (no validation - agent's loop handles self-correction)
            logger.info(
                f"[CIBench] Saving patch for problem {i} (agent self-corrects in loop)"
            )
            partial_fixes.append(
                {
                    "problem_id": i,
                    "problem_number": problem.get("problem_number", i),
                    "status": status,
                    "validation_stage": stage,
                    "patch": patch,
                    "exit_status": exit_status,
                }
            )

        except Exception as e:
            logger.error(f"[CIBench] ERROR Problem {i}: {e} - SKIPPING to next problem")
            # Don't break - continue to next problem
            continue

        finally:
            # Small delay between problems to allow cleanup and prevent timeout cascading
            if i < total_problems:
                logger.info("[CIBench] Waiting 5 seconds before next problem...")
                time.sleep(5)

    # The working tree is the source of truth. Concatenating per-problem
    # submissions can create duplicate diffs for the same file.
    unified_diff = _collect_final_workspace_diff(testbed_path)

    if unified_diff:
        logger.info(
            "[CIBench] Collected final workspace diff after sequential repair (%d chars)",
            len(unified_diff),
        )
    elif partial_fixes:
        logger.warning(
            "[CIBench] Final workspace diff is empty despite %d submitted partial patch(es); "
            "falling back to concatenated partial diffs",
            len(partial_fixes),
        )
        unified_diff = _combine_partial_fixes(partial_fixes)

        # CRITICAL: Merge duplicate file patches to avoid conflicts
        from minisweagent.run.benchmarks.utils.patch_merger import (
            detect_duplicate_patches,
            merge_duplicate_patches,
        )

        duplicates = detect_duplicate_patches(unified_diff)
        if duplicates:
            logger.warning(
                "[CIBench] Detected duplicate patches for %d file(s): %s",
                len(duplicates),
                list(duplicates.keys()),
            )
            logger.info("[CIBench] Merging duplicate patches...")
            try:
                unified_diff = merge_duplicate_patches(
                    unified_diff, repo_path=testbed_path
                )
                logger.info(
                    "[CIBench] Successfully merged duplicate patches into unified diff"
                )
            except Exception as e:
                logger.error(f"[CIBench] Failed to merge patches: {e}")
                # Keep original even if merge fails

    logger.info("[CIBench] Sequential repair complete:")
    logger.info(f"[CIBench]   Total problems: {total_problems}")
    logger.info(f"[CIBench]   Fixed problems: {len(partial_fixes)}")
    logger.info(
        f"[CIBench]   Success rate: {len(partial_fixes)}/{total_problems} = {100 * len(partial_fixes) / total_problems:.1f}%"
    )
    logger.info(f"[CIBench]   Unified diff: {len(unified_diff)} chars")

    # Build info dict
    info = {
        "exit_status": "submitted" if unified_diff else "failed",
        "submission": unified_diff,
        "sequential_repair": {
            "total_problems": total_problems,
            "fixed_problems": len(partial_fixes),
            "success_rate": len(partial_fixes) / total_problems
            if total_problems > 0
            else 0,
            "partial_fixes": partial_fixes,
        },
    }

    return info, unified_diff


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance processing
# ─────────────────────────────────────────────────────────────────────────────


def process_instance(
    instance: Dict[str, Any],
    output_dir: Path,
    config: Dict[str, Any],
    progress_manager: RunBatchProgressManager,
    *,
    repair_model: str,
    fault_localization_path: Optional[str] = None,
    direction: str = "unknown",
) -> None:
    """
    Full pipeline for one CI instance:
      pre-process logs -> clone repo -> checkout sha_fail -> run agent -> save diff
    """
    instance_id = str(instance.get("instance_id") or instance.get("id") or "unknown")
    sha_fail = str(instance.get("sha_fail") or "")
    dir_name = sha_fail or instance_id  # prefer sha_fail as the directory key
    instance_dir = output_dir / dir_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    metrics = RunMetricsRecorder(
        output_dir,
        instance_id,
        agent="miniswe-agent",
        model=repair_model,
        direction=direction,
        ablation="fault_localization_only",
    )

    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{dir_name}.traj.json").unlink(missing_ok=True)

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pre-processing CI logs")

    # Capture starting cost for this instance
    instance_start_cost = GLOBAL_MODEL_STATS.cost

    agent = None
    exit_status: Optional[str] = None
    diff = ""
    problems: List[Dict[str, Any]] = []
    extra_info: Dict[str, Any] = {}
    fault_localization = instance.get("_fault_localization")
    if fault_localization is not None:
        extra_info["fault_localization"] = {
            "source": fault_localization_path,
            "data": fault_localization,
        }
    prediction_ready = True

    # ── Cost and Time Tracking ────────────────────────────────────────────────
    import time
    start_time = time.time()
    cost_tracking = {
        "model": repair_model,
        "direction": direction,
        "ablation": "fault_localization_only",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_calls": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost_usd": 0.0,
    }

    # ── Phase 1: build problem input ─────────────────────────────────────────
    # In the external FL baseline, the preceding framework has already done
    # localization. Pass that FL document directly to mini-SWE-agent.
    if fault_localization is not None:
        task = (
            f"# CI Repair from Predicted Fault Localization\n\n"
            f"Repository: {instance.get('repo_owner', '')}/{instance.get('repo_name', '')}\n"
            f"Failing commit: {sha_fail}\n\n"
            f"## Predicted fault-localization document\n"
            f"{format_fault_localization(fault_localization)}"
        )
        problems = [
            {
                "problem_id": "external-fault-localization",
                "source": "ci failure",
                "problem_statement": task,
                "verification_cmd": "Inspect the repository and run the most relevant available checks.",
            }
        ]
        extra_info["input_mode"] = "fault_localization_only"
        logger.info(
            "[CIBench] Direct FL mode for %s; skipping CI/workflow analysis",
            instance_id,
        )
    # ── Phase 2: clone repo + checkout sha_fail ───────────────────────────────
    progress_manager.update_instance_status(instance_id, "Cloning repository")

    try:
        env, testbed_path = setup_local_environment(config, instance, instance_dir)

        # Fix max_tokens -> max_completion_tokens for GPT-4o and newer models
        model_config = dict(config.get("model", {}))
        model_kwargs = dict(model_config.get("model_kwargs", {}))
        model_name = str(resolve_model_alias(model_config.get("model_name", ""))).lower().removeprefix("openai/")

        is_gpt4o_or_newer = (
            model_name.startswith("gpt-5")
            or model_name.startswith("gpt-4o")
            or model_name.startswith("chatgpt-4o")
            or (len(model_name) > 1 and model_name[0] == "o" and model_name[1].isdigit())
        )

        if is_gpt4o_or_newer and "max_tokens" in model_kwargs:
            model_kwargs["max_completion_tokens"] = model_kwargs.pop("max_tokens")
            model_config["model_kwargs"] = model_kwargs
            logger.info(
                "[CIBench] Converted max_tokens -> max_completion_tokens for agent model: %s",
                model_name
            )

        model_ = get_model(config=model_config)
        agent = ProgressTrackingAgent(
            model_,
            env,
            progress_manager=progress_manager,
            instance_id=instance_id,
            metrics_recorder=metrics,
            **config.get("agent", {}),
        )
        # Inject the real checkout path so {{testbed_path}} resolves in templates
        agent.extra_template_vars["testbed_path"] = str(testbed_path)

        # ── Phase 3: run agent ────────────────────────────────────────────────
        progress_manager.update_instance_status(instance_id, "Running agent")

        # ══════════════════════════════════════════════════════════════════
        # SEQUENTIAL REPAIR: Fix problems one-by-one, collect ONE final diff
        # ══════════════════════════════════════════════════════════════════
        if problems:
            # SEQUENTIAL REPAIR MODE: Fix problems one at a time in same workspace
            # Agent fixes Problem 1 → modifies files
            # Agent fixes Problem 2 → modifies more files
            # ...
            # Agent fixes Problem 8 → modifies more files
            # THEN: Collect ONE unified diff from workspace with ALL changes
            logger.info(
                f"[CIBench] Sequential repair mode: {len(problems)} problems to fix"
            )

            repo_name = instance.get("repo_name", "")
            info, diff = _run_sequential_repair(
                agent=agent,
                problems=problems,
                testbed_path=testbed_path,
                progress_manager=progress_manager,
                instance_id=instance_id,
                repo_name=repo_name,
            )
            exit_status = info.get("exit_status")

            # Save sequential repair metadata
            sequential_metadata = info.get("sequential_repair", {})
            extra_info["sequential_repair"] = sequential_metadata
            logger.info(
                f"[CIBench] Sequential repair: {sequential_metadata.get('fixed_problems', 0)}/"
                f"{sequential_metadata.get('total_problems', 0)} problems fixed"
            )
        else:
            # Never replace failed/empty decomposition with a generic task: that
            # produces a patch unrelated to the atomic CI problems and then marks
            # the issue complete. Leave it retryable instead.
            prediction_ready = False
            exit_status = "problem_generation_failed"
            extra_info.setdefault(
                "problem_generation_error",
                "No fault-localization task was available for this instance",
            )
            logger.error("[CIBench] %s", extra_info["problem_generation_error"])

    except Exception as exc:
        logger.error(
            "[CIBench] Agent run failed for %s: %s", instance_id, exc, exc_info=True
        )
        exit_status = type(exc).__name__
        diff = ""
        prediction_ready = False
        extra_info.update(
            {"traceback": traceback.format_exc(), "exception_str": str(exc)}
        )

    finally:
        active_exception = sys.exc_info()[1]
        if isinstance(active_exception, KeyboardInterrupt):
            prediction_ready = False
            exit_status = "interrupted"

        # ── Finalize Cost Tracking ────────────────────────────────────────────
        end_time = time.time()
        cost_tracking["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cost_tracking["total_duration_seconds"] = round(end_time - start_time, 2)

        # Get agent cost if available
        if agent is not None and hasattr(agent, "cost"):
            cost_tracking["agent_cost_usd"] = round(float(agent.cost), 8)

        completed = bool(
            prediction_ready
            and str(exit_status or "").lower() == "submitted"
            and str(diff or "").strip()
        )
        if not completed:
            prediction_ready = False
        metrics_status = (
            "interrupted"
            if isinstance(active_exception, KeyboardInterrupt)
            else ("completed" if completed else "failed")
        )
        attempt_metrics = safe_metrics_call(
            metrics,
            "finish",
            status=metrics_status,
            error=(str(active_exception) if active_exception is not None else None),
            metadata={"exit_status": exit_status, "patch_generated": bool(str(diff).strip())},
        )
        attempt_totals = attempt_metrics.get("totals", {})
        cost_tracking.update(
            {
                "api_calls": attempt_metrics.get("api_calls", []),
                "total_input_tokens": attempt_totals.get("input_tokens", 0),
                "total_cached_input_tokens": attempt_totals.get("cached_input_tokens", 0),
                "total_cache_write_input_tokens": attempt_totals.get(
                    "cache_write_input_tokens", 0
                ),
                "total_output_tokens": attempt_totals.get("output_tokens", 0),
                "total_reasoning_output_tokens": attempt_totals.get(
                    "reasoning_output_tokens", 0
                ),
                "total_api_time_seconds": attempt_totals.get("api_time_seconds", 0),
                "total_cost_usd": attempt_totals.get("cost_usd", 0.0),
                "unpriced_api_calls": attempt_totals.get("unpriced_api_calls", 0),
                "cost_complete": attempt_totals.get("unpriced_api_calls", 0) == 0,
                "attempt_id": attempt_metrics.get("attempt_id"),
                "attempt_status": metrics_status,
            }
        )

        # Add to extra_info for saving
        extra_info["cost_tracking"] = cost_tracking

        if agent is not None:
            traj_path = instance_dir / f"{dir_name}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": diff,
                        "sha_fail": sha_fail,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
            logger.info("[CIBench] Saved trajectory to '%s'", traj_path)

        if prediction_ready:
            # Calculate cost for this instance
            instance_cost = GLOBAL_MODEL_STATS.cost - instance_start_cost

            update_preds_file(output_dir / "preds.json", instance_id, sha_fail, diff, cost=instance_cost)

            # Update persistent cost tracking (survives interruptions, handles reruns)
            update_cost_tracking(
                output_dir=output_dir,
                instance_id=instance_id,
                cost=instance_cost,
                model_name=repair_model,
                ablation="fault_localization_only",
                direction=direction,
            )

            # Also save the problems that were provided to the agent
            update_problems_file(output_dir / "problems.json", instance_id, problems)
        else:
            logger.error(
                "[CIBench] Not recording %s in preds.json; problem generation must be retried",
                instance_id,
            )
        progress_manager.on_instance_end(instance_id, exit_status)


def _extract_diff(submission: str) -> str:
    """
    Pull the git diff out of the agent's final submission string.

    The agent echoes ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` then cats the
    patch file.  We take everything after that sentinel (or the full string
    if the sentinel is absent).

    NOTE: We deliberately avoid str.strip() here because it removes trailing
    blank context lines (" \\n") from the diff, causing git to report
    "corrupt patch" (hunk header claims N lines but body only has N-1).
    Instead we lstrip leading whitespace and remove only truly empty trailing
    lines so that blank context lines like " " are preserved.

    Also detects duplicate file patches (multiple diffs for the same file)
    and logs a warning - these will be merged later in update_preds_file().
    """
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if sentinel in submission:
        diff = submission[submission.index(sentinel) + len(sentinel) :].lstrip()
    else:
        diff = submission.lstrip()
    # Strip only completely empty trailing lines, not blank context lines (" ")
    parts = diff.split("\n")
    while parts and parts[-1] == "":
        parts.pop()

    diff = ("\n".join(parts) + "\n") if parts else ""

    # Detect duplicate patches (will be merged in update_preds_file)
    if diff:
        duplicates = detect_duplicate_patches(diff)
        duplicate_files = [f for f, count in duplicates.items() if count > 1]
        if duplicate_files:
            logger.debug(
                "[CIBench] Detected %d file(s) with duplicate patches: %s (will be merged)",
                len(duplicate_files),
                ", ".join(duplicate_files[:3])
                + ("..." if len(duplicate_files) > 3 else ""),
            )

    return diff


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    dataset: str = typer.Option(
        "ci-benchmark-user/ci-repair-bench", "--dataset",
        help="Path to JSONL dataset or HuggingFace dataset name",
        rich_help_panel="Data selection",
    ),
    split: str = typer.Option(
        "train", "--split",
        help="Dataset split (for HuggingFace datasets)",
        rich_help_panel="Data selection",
    ),
    output: str = typer.Option(
        "", "--output",
        help="Output directory for predictions and trajectories",
        rich_help_panel="Basic",
    ),
    workers: int = typer.Option(
        1, "--workers",
        help="Parallel worker threads (each clones to its own instance_dir)",
        rich_help_panel="Basic",
    ),
    model_name: Optional[str] = typer.Option(
        None, "-m", "--model",
        help="LLM model to use for the repair agent",
        rich_help_panel="Basic",
    ),
    model_class: Optional[str] = typer.Option(
        None, "--model-class",
        help="Model class (e.g. 'anthropic' or full import path)",
        rich_help_panel="Advanced",
    ),
    config_spec: List[str] = typer.Option(
        [str(DEFAULT_CONFIG_FILE)], "--config",
        help="Config file(s) or key=value overrides (merged left to right)",
        rich_help_panel="Basic",
    ),
    filter_spec: str = typer.Option(
        "", "--filter",
        help="Regex filter on instance_id",
        rich_help_panel="Data selection",
    ),
    shuffle: bool = typer.Option(
        False, "--shuffle",
        help="Shuffle instances before slicing/filtering",
        rich_help_panel="Data selection",
    ),
    redo_existing: bool = typer.Option(
        False, "--redo-existing",
        help="Re-run instances already present in preds.json",
        rich_help_panel="Data selection",
    ),
    fault_localization: str = typer.Option(
        ..., "--fault-localization",
        help="Reference-framework fault_localization.json to pass to the repair agent",
        rich_help_panel="Data selection",
    ),
    direction: str = typer.Option(
        "unknown", "--direction",
        help="Experiment direction recorded in cost/time metrics",
        rich_help_panel="Advanced",
    ),
) -> None:
    # fmt: on
    """Run mini-SWE-agent on CI failure instances (batch mode, local environment)."""
    project_env_path = load_project_env()
    if project_env_path:
        logger.info("[CIBench] Loaded project env from %s", project_env_path)
    model_name = resolve_model_alias(model_name) if model_name else model_name

    # ── Output directory ──────────────────────────────────────────────────────
    if not output:
        output = f"ci_repair_results_{int(time.time())}"
    output_path = Path(output)

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("[CIBench] Results will be saved to %s", output_path)
    add_file_handler(output_path / "cibench.log")

    # ── Load dataset ──────────────────────────────────────────────────────────
    instances = load_ci_instances(dataset, split=split)
    instances = filter_instances(
        instances, filter_spec=filter_spec, slice_spec="", shuffle=shuffle
    )

    fault_localization_path = Path(fault_localization).resolve()
    indexed_fl = load_fault_localization(fault_localization_path)
    matched_fl = attach_fault_localization(instances, indexed_fl)
    if matched_fl == 0:
        raise typer.BadParameter(
            "No dataset instances matched the supplied fault localization by sha_fail or id",
            param_hint="--fault-localization",
        )
    unmatched = len(instances) - matched_fl
    logger.info(
        "[CIBench] Attached fault localization to %d/%d instances (%d unmatched)",
        matched_fl,
        len(instances),
        unmatched,
    )
    if unmatched:
        logger.warning("[CIBench] Skipping %d instances without fault-localization data", unmatched)
        instances = [i for i in instances if "_fault_localization" in i]
    with (output_path / "fault_localization_input.json").open("w", encoding="utf-8") as f:
        json.dump(
            [i["_fault_localization"] for i in instances],
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not redo_existing and (output_path / "preds.json").exists():
        predictions = _read_preds(output_path / "preds.json")
        metrics_completed = completed_instance_ids(output_path)
        metrics_known = metric_instance_ids(output_path)
        existing = {
            str(instance_id)
            for instance_id, prediction in predictions.items()
            if str((prediction or {}).get("diff") or "").strip()
            and (
                str(instance_id) not in metrics_known
                or str(instance_id) in metrics_completed
            )
        }
        before   = len(instances)
        instances = [i for i in instances if str(i.get("instance_id", "")) not in existing]
        logger.info("[CIBench] Skipping %d already-completed instances", before - len(instances))

    logger.info("[CIBench] Running on %d instances with %d worker(s)", len(instances), workers)

    # ── Config ────────────────────────────────────────────────────────────────
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "model": {
            "model_name":  model_name  or UNSET,
            "model_class": model_class or UNSET,
        },
    })
    config = recursive_merge(*configs)
    config["model"]["model_name"] = configure_model_environment(
        config["model"]["model_name"]
    )
    resolved_model = str(config["model"]["model_name"])
    logger.info(
        "[CIBench] Repair model: %s  class=%s",
        config["model"]["model_name"],
        config["model"].get("model_class") or "auto",
    )

    # ── Progress ──────────────────────────────────────────────────────────────
    progress_manager = RunBatchProgressManager(
        len(instances),
        output_path / f"exit_statuses_{int(time.time())}.yaml",
    )

    def _process(instance: Dict[str, Any]) -> None:
        process_instance(
            instance,
            output_path,
            config,
            progress_manager,
            repair_model=resolved_model,
            fault_localization_path=str(fault_localization_path),
            direction=direction,
        )

    def _process_futures(futures: "dict[concurrent.futures.Future, str]") -> None:
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as exc:
                iid = futures[future]
                logger.error("[CIBench] Uncaught exception for %s: %s", iid, exc, exc_info=True)
                progress_manager.on_uncaught_exception(iid, exc)

    with Live(progress_manager.render_group, refresh_per_second=4):
        if workers <= 1:
            for instance in instances:
                iid = str(instance.get("instance_id", ""))
                try:
                    _process(instance)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    logger.error("[CIBench] Uncaught exception for %s: %s", iid, exc, exc_info=True)
                    progress_manager.on_uncaught_exception(iid, exc)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process, instance): str(instance.get("instance_id", ""))
                    for instance in instances
                }
                try:
                    _process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("[CIBench] Cancelling pending jobs. Press ^C again to exit immediately.")
                    for f in futures:
                        if not f.running() and not f.done():
                            f.cancel()
                    _process_futures(futures)

    # ── Final summary ─────────────────────────────────────────────────────────
    preds = _read_preds(output_path / "preds.json")
    n_patched = sum(1 for v in preds.values() if v.get("diff", "").strip())

    # Get cost data from persistent tracking (survives interruptions)
    cost_data = get_total_cost(
        output_dir=output_path,
        model_name=model_name,
        ablation="fault_localization_only",
        direction=direction,
    )

    logger.info(
        "[CIBench] Done. %d/%d instances produced a non-empty patch. See %s",
        n_patched, len(preds), output_path / "preds.json",
    )
    logger.info(
        "[CIBench] ═══════════════════════════════════════════════════════════════"
    )
    logger.info("[CIBench] COST SUMMARY (Actual API Cost):")
    logger.info("[CIBench]   Total Cost:       $%.6f", cost_data["total_cost"])
    logger.info("[CIBench]   Instances Billed: %d", cost_data["n_instances"])
    logger.info("[CIBench]   Avg Cost/Instance: $%.6f", cost_data["avg_cost"])
    logger.info("[CIBench]   Model:      %s", model_name)
    logger.info("[CIBench]   Input mode: fault_localization_only")
    logger.info("[CIBench]   Direction:  %s", direction)
    logger.info("[CIBench]   Cost File:  %s", output_path / "costs.json")
    logger.info(
        "[CIBench] ═══════════════════════════════════════════════════════════════"
    )


if __name__ == "__main__":
    app()
