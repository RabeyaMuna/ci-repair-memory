"""Shared agent utilities for benchmark runners."""

import time
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from utilities.run_metrics import safe_metrics_call


class ProgressTrackingAgent(DefaultAgent):
    """Agent that reports per-step progress via :class:`RunBatchProgressManager`."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        metrics_recorder: Any = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id
        self.metrics_recorder = metrics_recorder

    def query(self) -> dict:
        started = time.time()
        limit_reached = (
            0 < self.config.step_limit <= self.n_calls
            or 0 < self.config.cost_limit <= self.cost
            or 0
            < self.config.wall_time_limit_seconds
            <= int(time.time() - self._start_time)
        )
        if not limit_reached:
            safe_metrics_call(
                self.metrics_recorder,
                "begin_api_call",
                phase="repair_agent",
                model=str(
                    getattr(getattr(self.model, "config", None), "model_name", "")
                ),
            )
        try:
            message = super().query()
        except BaseException as exc:
            if self.metrics_recorder is not None and not limit_reached:
                safe_metrics_call(
                    self.metrics_recorder,
                    "record_api_call",
                    phase="repair_agent",
                    model=str(getattr(getattr(self.model, "config", None), "model_name", "")),
                    duration_seconds=time.time() - started,
                    status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    error=str(exc),
                )
            raise

        if self.metrics_recorder is not None:
            extra = message.get("extra", {})
            safe_metrics_call(
                self.metrics_recorder,
                "record_response",
                response=extra.get("response", {}),
                phase="repair_agent",
                model=str(getattr(getattr(self.model, "config", None), "model_name", "")),
                duration_seconds=time.time() - started,
                explicit_cost_usd=float(extra.get("cost") or 0.0),
            )
        return message

    def step(self) -> dict:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()
