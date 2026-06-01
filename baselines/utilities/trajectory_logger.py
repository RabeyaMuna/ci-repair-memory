#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Trajectory Logger - Production Level

Similar to mini-swe-agent: saves one trajectory.json per issue.
Minimal, clean, production-ready.

Usage:
    # Initialize for an issue
    logger = TrajectoryLogger(issue_dir)

    # Log steps
    logger.log("ci_log_analysis", {"logs_count": 3}, {"failed_steps": ["test"]})
    logger.log("fault_localization", {...}, {"predicted_files": [...]})
    logger.log("patch_generation", {...}, {"patch_generated": True})

    # Save trajectory
    logger.save()
"""

import json
import os
import time
from typing import Dict, Any, Optional


class TrajectoryLogger:
    """
    Simple trajectory logger for CI repair pipeline

    Saves trajectory.json with all steps for an issue.
    """

    def __init__(self, issue_dir: str, task_id: str, sha_fail: str,
                 repo_name: str, model: str, condition: str, level: Optional[str] = None):
        """
        Initialize logger

        Args:
            issue_dir: Directory to save trajectory
            task_id: Issue task_id
            sha_fail: Failed commit SHA
            repo_name: Repository name
            model: Model name
            condition: "baseline" or "memory"
            level: Memory level (if memory)
        """
        self.issue_dir = issue_dir
        os.makedirs(issue_dir, exist_ok=True)

        # Trajectory data
        self.trajectory = {
            "task_id": task_id,
            "sha_fail": sha_fail,
            "repo_name": repo_name,
            "model": model,
            "condition": condition,
            "level": level,
            "start_time": time.time(),
            "steps": []
        }

    def log(self, action: str, input: Dict[str, Any], output: Dict[str, Any],
            success: bool = True, error: Optional[str] = None,
            duration: Optional[float] = None):
        """
        Log a step

        Args:
            action: Action name (e.g., "ci_log_analysis")
            input: Input data
            output: Output data
            success: Whether step succeeded
            error: Error message (if failed)
            duration: Step duration in seconds
        """
        step = {
            "step": len(self.trajectory["steps"]) + 1,
            "action": action,
            "timestamp": time.time(),
            "input": input,
            "output": output,
            "success": success,
            "error": error,
            "duration": duration
        }

        self.trajectory["steps"].append(step)

    def set_result(self, patch_generated: bool, fl_predictions: list = None,
                   total_tokens: Dict[str, int] = None, cost: float = 0.0):
        """
        Set final results

        Args:
            patch_generated: Whether patch was generated
            fl_predictions: FL predicted files
            total_tokens: Token usage {"input": X, "output": Y}
            cost: Total cost in USD
        """
        self.trajectory["patch_generated"] = patch_generated
        self.trajectory["fl_predictions"] = fl_predictions or []
        self.trajectory["total_tokens"] = total_tokens or {"input": 0, "output": 0}
        self.trajectory["cost_usd"] = cost
        self.trajectory["end_time"] = time.time()
        self.trajectory["duration_sec"] = self.trajectory["end_time"] - self.trajectory["start_time"]

    def save(self):
        """Save trajectory to trajectory.json"""
        trajectory_file = os.path.join(self.issue_dir, "trajectory.json")

        with open(trajectory_file, 'w') as f:
            json.dump(self.trajectory, f, indent=2)

        print(f"[TRAJECTORY] Saved: {trajectory_file}")

    def get_trajectory(self) -> Dict[str, Any]:
        """Get current trajectory data"""
        return self.trajectory.copy()


def create_logger(result_dir: str, model: str, condition: str, level: Optional[str],
                  task_id: str, sha_fail: str, repo_name: str) -> TrajectoryLogger:
    """
    Create trajectory logger with proper directory structure

    Args:
        result_dir: Base results directory
        model: Model name
        condition: "baseline" or "memory"
        level: Memory level (e.g., "L1", "L1L2", "L1L2L3")
        task_id: Task ID
        sha_fail: Failed commit SHA
        repo_name: Repository name

    Returns:
        TrajectoryLogger instance
    """
    # Build directory path
    model_safe = model.replace("/", "-").replace(":", "-")

    if condition == "baseline":
        issue_dir = os.path.join(result_dir, "trajectories", model_safe, "baseline", task_id)
    else:
        issue_dir = os.path.join(result_dir, "trajectories", model_safe, "memory", level, task_id)

    return TrajectoryLogger(issue_dir, task_id, sha_fail, repo_name, model, condition, level)
