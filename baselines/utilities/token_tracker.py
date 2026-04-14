# utilities/token_tracker.py
"""
Token usage and cost tracking for the CI-REPAIR-BENCH pipeline.

Tracks per-agent and overall:
  - Input tokens  (prompt)
  - Output tokens (completion)
  - API call count
  - Estimated cost (USD)

Usage
-----
    from utilities.token_tracker import TokenTracker

    tracker = TokenTracker(model_name="gpt-5-mini")

    # Pass tracker into get_tracked_llm() — it records automatically.
    # At the end of a run:
    tracker.print_summary()
    tracker.save_json("results/token_report.json")
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model pricing  (USD per 1 000 000 tokens — update as providers change)
# ---------------------------------------------------------------------------
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "gpt-4o":         {"input": 2.50,  "output": 10.00},
    "gpt-4.1":        {"input": 2.00,  "output": 8.00},
    "gpt-5-mini":     {"input": 1.10,  "output": 4.40},   # approximate
    "gpt-5.1":        {"input": 2.00,  "output": 8.00},   # approximate
    # DeepSeek
    "deepseek-chat":  {"input": 0.27,  "output": 1.10},
    "deepseek-coder": {"input": 0.27,  "output": 1.10},
}


# ---------------------------------------------------------------------------
# Data model for a single LLM call
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    agent: str           # e.g. "CILogAnalyzerLLM"
    call_site: str       # auto-detected method name, e.g. "ci_log_analysis"
    model: str           # model key / name
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class TokenTracker:
    """Accumulates token usage across the whole pipeline run."""

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.calls: List[CallRecord] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record(
        self,
        agent: str,
        call_site: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record one LLM API call."""
        total = input_tokens + output_tokens
        cost = self._compute_cost(model, input_tokens, output_tokens)
        self.calls.append(
            CallRecord(
                agent=agent,
                call_site=call_site,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
                cost_usd=cost,
            )
        )

    # ------------------------------------------------------------------
    def get_summary(self) -> dict:
        """Return a structured summary dict (suitable for JSON serialisation)."""
        per_agent: Dict[str, dict] = {}

        for c in self.calls:
            if c.agent not in per_agent:
                per_agent[c.agent] = {
                    "api_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "call_sites": {},
                }
            a = per_agent[c.agent]
            a["api_calls"] += 1
            a["input_tokens"] += c.input_tokens
            a["output_tokens"] += c.output_tokens
            a["total_tokens"] += c.total_tokens
            a["cost_usd"] += c.cost_usd

            # per-method breakdown within agent
            cs_key = c.call_site
            if cs_key not in a["call_sites"]:
                a["call_sites"][cs_key] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            cs = a["call_sites"][cs_key]
            cs["calls"] += 1
            cs["input_tokens"] += c.input_tokens
            cs["output_tokens"] += c.output_tokens
            cs["cost_usd"] += c.cost_usd

        # round per-agent
        for a in per_agent.values():
            a["cost_usd"] = round(a["cost_usd"], 6)
            for cs in a["call_sites"].values():
                cs["cost_usd"] = round(cs["cost_usd"], 6)

        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost_usd for c in self.calls)

        return {
            "overall": {
                "api_calls": len(self.calls),
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "cost_usd": round(total_cost, 6),
            },
            "per_agent": per_agent,
            "all_calls": [
                {
                    "seq": i + 1,
                    "agent": c.agent,
                    "call_site": c.call_site,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens,
                    "cost_usd": round(c.cost_usd, 8),
                    "timestamp": c.timestamp,
                }
                for i, c in enumerate(self.calls)
            ],
        }

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print a human-readable token/cost table to stdout."""
        summary = self.get_summary()
        ov = summary["overall"]

        sep = "=" * 64
        print(f"\n{sep}")
        print("  TOKEN & COST SUMMARY")
        print(sep)
        print(f"  Total API calls    : {ov['api_calls']}")
        print(f"  Total input tokens : {ov['input_tokens']:>12,}")
        print(f"  Total output tokens: {ov['output_tokens']:>12,}")
        print(f"  Total tokens       : {ov['total_tokens']:>12,}")
        print(f"  Estimated cost     : ${ov['cost_usd']:>12.6f}  USD")
        print()

        for agent, data in summary["per_agent"].items():
            print(f"  [{agent}]")
            print(f"    API calls    : {data['api_calls']}")
            print(f"    Input tokens : {data['input_tokens']:,}")
            print(f"    Output tokens: {data['output_tokens']:,}")
            print(f"    Total tokens : {data['total_tokens']:,}")
            print(f"    Cost         : ${data['cost_usd']:.6f} USD")
            for site, cs in data["call_sites"].items():
                print(
                    f"      ├─ {site:<40} "
                    f"calls={cs['calls']}  "
                    f"in={cs['input_tokens']:,}  "
                    f"out={cs['output_tokens']:,}  "
                    f"cost=${cs['cost_usd']:.6f}"
                )
            print()

        print(sep + "\n")

    # ------------------------------------------------------------------
    def save_json(self, path: str) -> None:
        """Save the full summary (including every call record) to a JSON file."""
        summary = self.get_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[TokenTracker] Saved token/cost report → {path}")
