#!/usr/bin/env python3
"""
Detailed LLM Usage Tracker - Integrated into all runs.

Automatically saves detailed_usage_<instance_id>.json for EVERY instance with:
- Complete step-by-step breakdown
- Token usage per phase
- Cost per API call
- Full execution timeline

This runs alongside run_metrics.json to provide granular visibility.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


# MiniMax M2.5 pricing (from LiteLLM)
PRICING_MINIMAX_M2_5 = {
    "input_cost_per_token": 3e-07,           # $0.30 per 1M
    "output_cost_per_token": 1.1e-06,        # $1.10 per 1M
    "cache_read_input_token_cost": 1.5e-07,  # $0.15 per 1M
}


class DetailedUsageTracker:
    """
    Tracks ALL LLM calls for one instance with complete breakdown.

    Auto-saves to: <output_dir>/detailed_usage_<instance_id>.json
    """

    def __init__(self, instance_id: str, output_dir: str | Path, model: str = ""):
        self.instance_id = instance_id
        self.output_dir = Path(output_dir)
        self.default_model = model
        self.calls: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.call_sequence = 0  # Track call order

    def record_call(
        self,
        phase: str,
        model: str = "",
        duration_seconds: float = 0.0,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_output_tokens: int = 0,
        cost_usd: float = 0.0,
        cost_source: str = "unknown",
        **metadata
    ):
        """
        Record a single LLM API call.

        Args:
            phase: Step name (e.g., "context_llm", "memory.decomposition")
            model: Model name
            duration_seconds: API call duration
            input_tokens: Input tokens
            cached_input_tokens: Cached input tokens
            output_tokens: Output tokens
            reasoning_output_tokens: Reasoning tokens
            cost_usd: Cost in USD
            cost_source: How cost was determined
            **metadata: Additional fields to store
        """
        self.call_sequence += 1

        call_data = {
            "sequence": self.call_sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "model": model or self.default_model,
            "duration_seconds": round(duration_seconds, 3),
            "tokens": {
                "input": input_tokens,
                "cached_input": cached_input_tokens,
                "uncached_input": input_tokens - cached_input_tokens,
                "output": output_tokens,
                "reasoning_output": reasoning_output_tokens,
                "total": input_tokens + output_tokens + reasoning_output_tokens
            },
            "cost_usd": round(cost_usd, 8),
            "cost_source": cost_source,
            **metadata
        }

        self.calls.append(call_data)

    def save(self):
        """Save detailed usage to shared detailed_usage.json file."""
        shared_file = self.output_dir / "detailed_usage.json"

        # Calculate totals
        total_calls = len(self.calls)
        total_input = sum(c["tokens"]["input"] for c in self.calls)
        total_cached = sum(c["tokens"]["cached_input"] for c in self.calls)
        total_output = sum(c["tokens"]["output"] for c in self.calls)
        total_reasoning = sum(c["tokens"]["reasoning_output"] for c in self.calls)
        total_cost = sum(c["cost_usd"] for c in self.calls)
        total_duration = sum(c["duration_seconds"] for c in self.calls)

        # Group by phase
        phase_stats = {}
        for call in self.calls:
            phase = call["phase"]
            if phase not in phase_stats:
                phase_stats[phase] = {
                    "call_count": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_cost_usd": 0.0,
                    "total_duration_seconds": 0.0,
                    "calls": []
                }

            phase_stats[phase]["call_count"] += 1
            phase_stats[phase]["total_tokens"] += call["tokens"]["total"]
            phase_stats[phase]["input_tokens"] += call["tokens"]["input"]
            phase_stats[phase]["cached_tokens"] += call["tokens"]["cached_input"]
            phase_stats[phase]["output_tokens"] += call["tokens"]["output"]
            phase_stats[phase]["reasoning_tokens"] += call["tokens"]["reasoning_output"]
            phase_stats[phase]["total_cost_usd"] += call["cost_usd"]
            phase_stats[phase]["total_duration_seconds"] += call["duration_seconds"]
            phase_stats[phase]["calls"].append(call["sequence"])

        # Build this instance's data
        instance_data = {
            "instance_id": self.instance_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time_seconds": round(time.time() - self.start_time, 3),

            "summary": {
                "total_api_calls": total_calls,
                "total_tokens": total_input + total_output + total_reasoning,
                "input_tokens": total_input,
                "cached_input_tokens": total_cached,
                "uncached_input_tokens": total_input - total_cached,
                "output_tokens": total_output,
                "reasoning_output_tokens": total_reasoning,
                "total_cost_usd": round(total_cost, 8),
                "total_api_duration_seconds": round(total_duration, 3)
            },

            "phase_breakdown": {
                phase: {
                    k: (round(v, 8) if k.endswith("_usd") or k.endswith("_seconds") else v)
                    for k, v in stats.items()
                    if k != "calls"  # Exclude call list from summary
                }
                for phase, stats in sorted(phase_stats.items())
            },

            "detailed_calls": self.calls
        }

        # Load existing data or create new
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if shared_file.exists():
            try:
                with open(shared_file, 'r') as f:
                    all_data = json.load(f)
                if not isinstance(all_data, list):
                    all_data = []
            except (json.JSONDecodeError, IOError):
                all_data = []
        else:
            all_data = []

        # Remove existing entry for this instance (if rerun)
        all_data = [item for item in all_data if item.get("instance_id") != self.instance_id]

        # Add this instance
        all_data.append(instance_data)

        # Sort by instance_id
        all_data.sort(key=lambda x: str(x.get("instance_id", "")))

        # Save to shared file
        with open(shared_file, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Print summary
        print(f"\n📊 Detailed usage saved: {shared_file}")
        print(f"   Instance {self.instance_id}: {total_calls} calls, {total_input + total_output + total_reasoning:,} tokens, ${total_cost:.6f}")
        print(f"   Total instances in file: {len(all_data)}")

        return shared_file


def calculate_cost_from_usage(
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    reasoning_output_tokens: int,
    model: str = ""
) -> float:
    """
    Calculate cost from token counts.

    Uses MiniMax M2.5 pricing by default:
    - Uncached input: $0.30/1M tokens
    - Cached input: $0.15/1M tokens
    - Output: $1.10/1M tokens
    - Reasoning: $1.10/1M tokens (separate from output)
    """
    uncached_input = input_tokens - cached_input_tokens

    cost = (
        uncached_input * PRICING_MINIMAX_M2_5["input_cost_per_token"] +
        cached_input_tokens * PRICING_MINIMAX_M2_5["cache_read_input_token_cost"] +
        output_tokens * PRICING_MINIMAX_M2_5["output_cost_per_token"] +
        reasoning_output_tokens * PRICING_MINIMAX_M2_5["output_cost_per_token"]
    )

    return cost
