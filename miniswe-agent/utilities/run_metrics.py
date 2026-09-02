"""Durable per-instance API cost and timing metrics for benchmark runners.

The ledger is intentionally additive and lives beside the existing benchmark
outputs.  Existing prediction and trajectory formats remain untouched.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import socket
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - benchmark launchers currently target Unix
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
_PROCESS_LOCK = threading.RLock()
_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)
LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_number(value: Any, *, integer: bool = False) -> int | float:
    try:
        if value is None or value == "?":
            return 0
        return int(value) if integer else float(value)
    except (TypeError, ValueError):
        return 0


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_usage(response: Any) -> dict[str, int]:
    """Normalize OpenAI, LiteLLM, LangChain, and OpenRouter usage objects."""
    usage = _get(response, "usage") or _get(response, "usage_metadata") or {}
    response_metadata = _get(response, "response_metadata") or {}
    if not usage and response_metadata:
        usage = _get(response_metadata, "token_usage") or {}

    input_tokens = _get(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _get(usage, "input_tokens", 0)
    output_tokens = _get(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _get(usage, "output_tokens", 0)

    prompt_details = (
        _get(usage, "prompt_tokens_details")
        or _get(usage, "input_tokens_details")
        or {}
    )
    completion_details = (
        _get(usage, "completion_tokens_details")
        or _get(usage, "output_tokens_details")
        or {}
    )
    return {
        "input_tokens": int(_as_number(input_tokens, integer=True)),
        "cached_input_tokens": int(
            _as_number(
                _get(prompt_details, "cached_tokens", _get(usage, "cached_input_tokens", 0)),
                integer=True,
            )
        ),
        "cache_write_input_tokens": int(
            _as_number(
                _get(
                    prompt_details,
                    "cache_write_tokens",
                    _get(usage, "cache_write_input_tokens", 0),
                ),
                integer=True,
            )
        ),
        "output_tokens": int(_as_number(output_tokens, integer=True)),
        "reasoning_output_tokens": int(
            _as_number(
                _get(
                    completion_details,
                    "reasoning_tokens",
                    _get(usage, "reasoning_output_tokens", 0),
                ),
                integer=True,
            )
        ),
    }


def response_cost_usd(response: Any) -> tuple[float, str]:
    """Return provider/LiteLLM supplied cost when it is present."""
    usage = _get(response, "usage") or {}
    for value, source in (
        (_get(usage, "cost"), "provider_usage"),
        (_get(response, "response_cost"), "provider_response"),
        (_get(_get(response, "_hidden_params") or {}, "response_cost"), "litellm_response"),
    ):
        if value is None:
            continue
        cost = float(_as_number(value))
        if cost >= 0:
            return cost, source
    return 0.0, "unavailable"


def estimate_cost_usd(model: str, usage: dict[str, int]) -> tuple[float, str]:
    """Estimate cost from LiteLLM's model registry when no billed cost is returned."""
    pricing_path = os.getenv("RUN_METRICS_PRICING_FILE", "").strip()
    if pricing_path:
        try:
            pricing_data = json.loads(Path(pricing_path).read_text(encoding="utf-8"))
            pricing = pricing_data.get(model) or pricing_data.get(
                str(model).removeprefix("openai/")
            )
            if isinstance(pricing, dict):
                input_rate = float(pricing["input_cost_per_million"])
                cached_rate = float(
                    pricing.get("cached_input_cost_per_million", input_rate)
                )
                cache_write_rate = float(
                    pricing.get("cache_write_input_cost_per_million", input_rate)
                )
                output_rate = float(pricing["output_cost_per_million"])
                cached = usage.get("cached_input_tokens", 0)
                non_cached = max(usage.get("input_tokens", 0) - cached, 0)
                cost = (
                    non_cached * input_rate
                    + cached * cached_rate
                    + usage.get("cache_write_input_tokens", 0) * cache_write_rate
                    + usage.get("output_tokens", 0) * output_rate
                ) / 1_000_000
                return cost, "custom_pricing"
        except Exception as exc:
            LOGGER.warning("Could not use RUN_METRICS_PRICING_FILE: %s", exc)

    try:
        import litellm

        candidates = [str(model or "")]
        if candidates[0].startswith("openai/"):
            candidates.append(candidates[0].removeprefix("openai/"))
        elif "/" not in candidates[0]:
            candidates.append(f"openai/{candidates[0]}")
        for candidate in candidates:
            try:
                input_cost, output_cost = litellm.cost_per_token(
                    model=candidate,
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    cache_creation_input_tokens=usage.get("cache_write_input_tokens", 0),
                    cache_read_input_tokens=usage.get("cached_input_tokens", 0),
                )
                cost = float(input_cost) + float(output_cost)
                if cost > 0:
                    return cost, "litellm_estimate"
            except Exception:
                continue
    except Exception:
        pass
    return 0.0, "unavailable"


def cost_for_response(response: Any, model: str, usage: dict[str, int]) -> tuple[float, str]:
    cost, source = response_cost_usd(response)
    if source != "unavailable":
        return cost, source
    try:
        import litellm

        cost = float(litellm.completion_cost(completion_response=response, model=model))
        if cost > 0:
            return cost, "litellm_response"
    except Exception:
        pass
    return estimate_cost_usd(model, usage)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _load_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "run": {}, "instances": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": SCHEMA_VERSION, "run": {}, "instances": {}}
    if not isinstance(data, dict):
        return {"schema_version": SCHEMA_VERSION, "run": {}, "instances": {}}
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("run", {})
    data.setdefault("instances", {})
    return data


def _attempt_totals(attempt: dict[str, Any]) -> dict[str, Any]:
    calls = [call for call in attempt.get("api_calls", []) if isinstance(call, dict)]
    totals: dict[str, Any] = {
        "api_calls": len(calls),
        "api_time_seconds": round(
            sum(float(_as_number(call.get("duration_seconds"))) for call in calls), 3
        ),
        "cost_usd": round(
            sum(float(_as_number(call.get("cost_usd"))) for call in calls), 8
        ),
        "unpriced_api_calls": sum(
            call.get("cost_source") == "unavailable"
            for call in calls
        ),
    }
    for field in _TOKEN_FIELDS:
        totals[field] = sum(int(_as_number(call.get(field), integer=True)) for call in calls)
    return totals


def _instance_summary(instance_id: str, value: dict[str, Any]) -> dict[str, Any]:
    attempts = [attempt for attempt in value.get("attempts", []) if isinstance(attempt, dict)]
    totals = [_attempt_totals(attempt) for attempt in attempts]
    last = attempts[-1] if attempts else {}
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "attempts": len(attempts),
        "completed_attempts": sum(a.get("status") == "completed" for a in attempts),
        "interrupted_attempts": sum(a.get("status") == "interrupted" for a in attempts),
        "failed_attempts": sum(a.get("status") == "failed" for a in attempts),
        "last_status": last.get("status", "unknown"),
        "total_wall_time_seconds": round(
            sum(float(_as_number(a.get("duration_seconds"))) for a in attempts), 3
        ),
        "total_api_time_seconds": round(sum(t["api_time_seconds"] for t in totals), 3),
        "total_cost_usd": round(sum(t["cost_usd"] for t in totals), 8),
        "unpriced_api_calls": sum(t["unpriced_api_calls"] for t in totals),
    }
    for field in _TOKEN_FIELDS:
        summary[f"total_{field}"] = sum(t[field] for t in totals)
    summary["cost_complete"] = summary["unpriced_api_calls"] == 0
    return summary


def _build_report(data: dict[str, Any]) -> dict[str, Any]:
    details = [
        _instance_summary(str(instance_id), value)
        for instance_id, value in sorted(
            data.get("instances", {}).items(), key=lambda item: str(item[0])
        )
        if isinstance(value, dict)
    ]
    summary: dict[str, Any] = {
        "instances": len(details),
        "completed_instances": sum(item["completed_attempts"] > 0 for item in details),
        "interrupted_instances": sum(item["interrupted_attempts"] > 0 for item in details),
        "failed_instances": sum(item["failed_attempts"] > 0 for item in details),
        "attempts": sum(item["attempts"] for item in details),
        "total_wall_time_seconds": round(
            sum(item["total_wall_time_seconds"] for item in details), 3
        ),
        "total_api_time_seconds": round(
            sum(item["total_api_time_seconds"] for item in details), 3
        ),
        "total_cost_usd": round(sum(item["total_cost_usd"] for item in details), 8),
        "unpriced_api_calls": sum(item["unpriced_api_calls"] for item in details),
    }
    for field in _TOKEN_FIELDS:
        summary[f"total_{field}"] = sum(item[f"total_{field}"] for item in details)
    summary["cost_complete"] = summary["unpriced_api_calls"] == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "run": data.get("run", {}),
        "summary": summary,
        "instances": details,
    }


def _write_report(output_dir: Path, data: dict[str, Any]) -> None:
    report = _build_report(data)
    _atomic_write_text(
        output_dir / "cost_time_report.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    columns = [
        "instance_id",
        "attempts",
        "completed_attempts",
        "interrupted_attempts",
        "failed_attempts",
        "last_status",
        "total_wall_time_seconds",
        "total_api_time_seconds",
        "total_cost_usd",
        "unpriced_api_calls",
        "cost_complete",
        "total_input_tokens",
        "total_cached_input_tokens",
        "total_cache_write_input_tokens",
        "total_output_tokens",
        "total_reasoning_output_tokens",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["instances"])
    _atomic_write_text(output_dir / "cost_time_report.csv", buffer.getvalue())


class RunMetricsRecorder:
    """Concurrency-safe recorder for one instance attempt."""

    def __init__(
        self,
        output_dir: str | Path,
        instance_id: str,
        *,
        agent: str,
        model: str,
        direction: str,
        ablation: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.instance_id = str(instance_id)
        self.run_metadata = {
            "agent": agent,
            "model": str(model or "unknown"),
            "direction": str(direction or "unknown"),
            "ablation": str(ablation or "baseline"),
        }
        self.ledger_path = self.output_dir / "run_metrics.json"
        self.lock_path = self.output_dir / "run_metrics.lock"
        self.attempt_id = uuid.uuid4().hex
        self._started_epoch = time.time()
        self._finished = False
        self._heartbeat_stop = threading.Event()
        self._start_attempt()
        heartbeat_seconds = max(
            float(_as_number(os.getenv("RUN_METRICS_HEARTBEAT_SECONDS", "10"))),
            0.0,
        )
        self._heartbeat_thread: threading.Thread | None = None
        if heartbeat_seconds > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                args=(heartbeat_seconds,),
                name=f"metrics-{self.instance_id}-{self.attempt_id[:8]}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @contextmanager
    def _locked_ledger(self) -> Iterator[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK:
            with self.lock_path.open("a+", encoding="utf-8") as lock_stream:
                if fcntl is not None:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                try:
                    data = _load_ledger(self.ledger_path)
                    yield data
                    data["updated_at"] = _utc_now()
                    _atomic_write_text(
                        self.ledger_path,
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    )
                    _write_report(self.output_dir, data)
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)

    def _find_attempt(self, data: dict[str, Any]) -> dict[str, Any]:
        instance = data.setdefault("instances", {}).setdefault(
            self.instance_id, {"attempts": []}
        )
        for attempt in instance.setdefault("attempts", []):
            if attempt.get("attempt_id") == self.attempt_id:
                return attempt
        raise KeyError(f"Metrics attempt not found: {self.attempt_id}")

    def _start_attempt(self) -> None:
        now = _utc_now()
        with self._locked_ledger() as data:
            data["run"] = {**data.get("run", {}), **self.run_metadata}
            instance = data.setdefault("instances", {}).setdefault(
                self.instance_id, {"attempts": []}
            )
            for previous in instance.setdefault("attempts", []):
                if previous.get("status") == "running":
                    previous["status"] = "interrupted"
                    previous["ended_at"] = previous.get("last_checkpoint_at", now)
                    previous.setdefault(
                        "interruption_reason", "superseded by a resumed attempt"
                    )
                    pending_call = previous.pop("in_progress_api_call", None)
                    if isinstance(pending_call, dict):
                        previous.setdefault("api_calls", []).append(
                            {
                                **pending_call,
                                "recorded_at": previous["ended_at"],
                                "status": "interrupted",
                                "duration_seconds": 0.0,
                                **{field: 0 for field in _TOKEN_FIELDS},
                                "cost_usd": 0.0,
                                "cost_source": "unavailable",
                                "error": "process ended before API usage was checkpointed",
                            }
                        )
                    previous["totals"] = _attempt_totals(previous)
            instance["attempts"].append(
                {
                    "attempt_id": self.attempt_id,
                    "status": "running",
                    "started_at": now,
                    "last_checkpoint_at": now,
                    "duration_seconds": 0.0,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "api_calls": [],
                    "totals": _attempt_totals({}),
                }
            )

    def _heartbeat_loop(self, interval_seconds: float) -> None:
        while not self._heartbeat_stop.wait(interval_seconds):
            try:
                with self._locked_ledger() as data:
                    attempt = self._find_attempt(data)
                    if attempt.get("status") != "running":
                        self._heartbeat_stop.set()
                        return
                    attempt["last_checkpoint_at"] = _utc_now()
                    attempt["duration_seconds"] = round(
                        time.time() - self._started_epoch, 3
                    )
            except Exception as exc:  # pragma: no cover - requires filesystem failure
                LOGGER.warning("Could not write metrics heartbeat: %s", exc)

    def begin_api_call(self, *, phase: str, model: str | None = None) -> dict[str, Any]:
        marker = {
            "started_at": _utc_now(),
            "phase": str(phase),
            "model": str(model or self.run_metadata["model"]),
        }
        with self._locked_ledger() as data:
            attempt = self._find_attempt(data)
            attempt["in_progress_api_call"] = marker
            attempt["last_checkpoint_at"] = marker["started_at"]
            attempt["duration_seconds"] = round(time.time() - self._started_epoch, 3)
        return marker

    def record_api_call(
        self,
        *,
        phase: str,
        model: str | None = None,
        duration_seconds: float = 0.0,
        usage: dict[str, Any] | None = None,
        cost_usd: float = 0.0,
        cost_source: str = "unavailable",
        status: str = "completed",
        error: str | None = None,
    ) -> dict[str, Any]:
        normalized_usage = {
            field: int(_as_number((usage or {}).get(field), integer=True))
            for field in _TOKEN_FIELDS
        }
        call = {
            "recorded_at": _utc_now(),
            "phase": str(phase),
            "model": str(model or self.run_metadata["model"]),
            "status": status,
            "duration_seconds": round(float(duration_seconds), 3),
            **normalized_usage,
            "cost_usd": round(float(cost_usd), 10),
            "cost_source": cost_source,
        }
        if error:
            call["error"] = str(error)
        with self._locked_ledger() as data:
            attempt = self._find_attempt(data)
            attempt.pop("in_progress_api_call", None)
            attempt.setdefault("api_calls", []).append(call)
            attempt["last_checkpoint_at"] = call["recorded_at"]
            attempt["duration_seconds"] = round(time.time() - self._started_epoch, 3)
            attempt["totals"] = _attempt_totals(attempt)
        return call

    def record_response(
        self,
        response: Any,
        *,
        phase: str,
        model: str | None = None,
        duration_seconds: float = 0.0,
        explicit_cost_usd: float | None = None,
    ) -> dict[str, Any]:
        usage = extract_usage(response)
        selected_model = str(model or self.run_metadata["model"])
        if explicit_cost_usd is not None and explicit_cost_usd > 0:
            cost, source = float(explicit_cost_usd), "agent_response"
        else:
            cost, source = cost_for_response(response, selected_model, usage)
        return self.record_api_call(
            phase=phase,
            model=selected_model,
            duration_seconds=duration_seconds,
            usage=usage,
            cost_usd=cost,
            cost_source=source,
        )

    def finish(
        self,
        status: str,
        *,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._finished:
            return self.snapshot()
        self._heartbeat_stop.set()
        now = _utc_now()
        with self._locked_ledger() as data:
            attempt = self._find_attempt(data)
            pending_call = attempt.pop("in_progress_api_call", None)
            if isinstance(pending_call, dict):
                attempt.setdefault("api_calls", []).append(
                    {
                        **pending_call,
                        "recorded_at": now,
                        "status": "interrupted" if status == "interrupted" else "failed",
                        "duration_seconds": 0.0,
                        **{field: 0 for field in _TOKEN_FIELDS},
                        "cost_usd": 0.0,
                        "cost_source": "unavailable",
                        "error": "attempt ended before API usage was checkpointed",
                    }
                )
            attempt["status"] = status
            attempt["ended_at"] = now
            attempt["last_checkpoint_at"] = now
            attempt["duration_seconds"] = round(time.time() - self._started_epoch, 3)
            if error:
                attempt["error"] = str(error)
            if metadata:
                attempt["metadata"] = metadata
            attempt["totals"] = _attempt_totals(attempt)
            result = json.loads(json.dumps(attempt))
        self._finished = True
        return result

    def snapshot(self) -> dict[str, Any]:
        data = _load_ledger(self.ledger_path)
        try:
            return json.loads(json.dumps(self._find_attempt(data)))
        except KeyError:
            return {}


def completed_instance_ids(output_dir: str | Path) -> set[str]:
    """Return instances with at least one durably completed attempt."""
    data = _load_ledger(Path(output_dir) / "run_metrics.json")
    completed: set[str] = set()
    for instance_id, value in data.get("instances", {}).items():
        attempts = value.get("attempts", []) if isinstance(value, dict) else []
        if any(
            isinstance(attempt, dict) and attempt.get("status") == "completed"
            for attempt in attempts
        ):
            completed.add(str(instance_id))
    return completed


def has_instance_metrics(output_dir: str | Path, instance_id: str) -> bool:
    data = _load_ledger(Path(output_dir) / "run_metrics.json")
    return str(instance_id) in data.get("instances", {})


def metric_instance_ids(output_dir: str | Path) -> set[str]:
    data = _load_ledger(Path(output_dir) / "run_metrics.json")
    return {str(instance_id) for instance_id in data.get("instances", {})}


def safe_metrics_call(
    recorder: RunMetricsRecorder | None, method: str, /, **kwargs: Any
) -> dict[str, Any]:
    """Record metrics without allowing an accounting failure to rerun API work."""
    if recorder is None:
        return {}
    try:
        result = getattr(recorder, method)(**kwargs)
        return result if isinstance(result, dict) else {}
    except Exception as exc:  # pragma: no cover - requires filesystem failure
        LOGGER.warning("Could not persist run metrics via %s: %s", method, exc)
        return {}
