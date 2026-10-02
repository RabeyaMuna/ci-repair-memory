"""
LLM Usage Tracker - Token, Cost, and Timing Tracking
Integrates with all LLM calls to track usage, cost, and performance.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

# Model pricing per 1M tokens (input, output) - Updated 2025
MODEL_PRICING = {
    # OpenAI models
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},  # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 5.00, "output": 15.00},
    "gpt-5-mini": {"input": 1.00, "output": 4.00},
    "gpt-5.1": {"input": 5.00, "output": 15.00},

    # DeepSeek models (via OpenRouter)
    "deepseek-chat": {"input": 0.14, "output": 0.28},  # v2.5
    "deepseek-coder": {"input": 0.14, "output": 0.28},
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},  # v4 via OpenRouter
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},  # OpenRouter format
}


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call"""
    timestamp: str
    module: str
    function: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration_seconds: float
    input_cost: float
    output_cost: float
    total_cost: float
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMTracker:
    """
    Singleton tracker for all LLM usage across the application.
    Thread-safe and provides aggregation by module and model.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.calls: List[LLMCallRecord] = []
        self.start_time = datetime.now().isoformat()
        self._initialized = True

    def track_call(
        self,
        module: str,
        function: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMCallRecord:
        """Track a single LLM call with full metrics"""

        total_tokens = input_tokens + output_tokens

        # Calculate costs
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        record = LLMCallRecord(
            timestamp=datetime.now().isoformat(),
            module=module,
            function=function,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            success=success,
            error=error,
            metadata=metadata
        )

        with self._lock:
            self.calls.append(record)

        return record

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary of all tracked calls"""

        if not self.calls:
            return {
                "total_calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_duration": 0.0
            }

        total_calls = len(self.calls)
        total_tokens = sum(c.total_tokens for c in self.calls)
        total_cost = sum(c.total_cost for c in self.calls)
        total_duration = sum(c.duration_seconds for c in self.calls)
        successful_calls = sum(1 for c in self.calls if c.success)

        # Group by module
        by_module = {}
        for call in self.calls:
            if call.module not in by_module:
                by_module[call.module] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "duration": 0.0
                }
            by_module[call.module]["calls"] += 1
            by_module[call.module]["tokens"] += call.total_tokens
            by_module[call.module]["cost"] += call.total_cost
            by_module[call.module]["duration"] += call.duration_seconds

        # Group by model
        by_model = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "duration": 0.0
                }
            by_model[call.model]["calls"] += 1
            by_model[call.model]["input_tokens"] += call.input_tokens
            by_model[call.model]["output_tokens"] += call.output_tokens
            by_model[call.model]["total_tokens"] += call.total_tokens
            by_model[call.model]["cost"] += call.total_cost
            by_model[call.model]["duration"] += call.duration_seconds

        return {
            "session_start": self.start_time,
            "session_end": datetime.now().isoformat(),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "total_input_tokens": sum(c.input_tokens for c in self.calls),
            "total_output_tokens": sum(c.output_tokens for c in self.calls),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "total_duration_seconds": round(total_duration, 2),
            "average_duration_seconds": round(total_duration / total_calls, 2) if total_calls > 0 else 0,
            "by_module": by_module,
            "by_model": by_model
        }

    def save_report(self, output_path: str, include_all_calls: bool = True):
        """Save detailed tracking report to JSON"""

        summary = self.get_summary()

        report = {
            "summary": summary,
        }

        if include_all_calls:
            report["all_calls"] = [asdict(call) for call in self.calls]

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return output_file

    def print_summary(self):
        """Print formatted summary to console"""

        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("LLM USAGE SUMMARY")
        print("=" * 80)
        print(f"Session: {summary['session_start']} → {summary['session_end']}")
        print(f"\nTotal Calls:     {summary['total_calls']:,}")
        print(f"  Successful:    {summary['successful_calls']:,}")
        print(f"  Failed:        {summary['failed_calls']:,}")
        print(f"\nTotal Tokens:    {summary['total_tokens']:,}")
        print(f"  Input:         {summary['total_input_tokens']:,}")
        print(f"  Output:        {summary['total_output_tokens']:,}")
        print(f"\nTotal Cost:      ${summary['total_cost_usd']:.4f}")
        print(f"Total Duration:  {summary['total_duration_seconds']:.2f}s")
        print(f"Avg Duration:    {summary['average_duration_seconds']:.2f}s per call")

        print("\n" + "-" * 80)
        print("BY MODULE:")
        print("-" * 80)
        for module, stats in sorted(summary['by_module'].items()):
            print(f"  {module:30} {stats['calls']:4} calls | "
                  f"{stats['tokens']:,} tokens | ${stats['cost']:.4f} | "
                  f"{stats['duration']:.2f}s")

        print("\n" + "-" * 80)
        print("BY MODEL:")
        print("-" * 80)
        for model, stats in sorted(summary['by_model'].items()):
            print(f"  {model:20} {stats['calls']:4} calls | "
                  f"{stats['total_tokens']:,} tokens | ${stats['cost']:.4f} | "
                  f"{stats['duration']:.2f}s")

        print("=" * 80 + "\n")

    def reset(self):
        """Reset tracker (useful for testing or multi-run scenarios)"""
        with self._lock:
            self.calls = []
            self.start_time = datetime.now().isoformat()


# Global tracker instance
tracker = LLMTracker()
