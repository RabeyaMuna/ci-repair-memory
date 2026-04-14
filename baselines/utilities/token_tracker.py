# utilities/token_tracker.py
"""
Comprehensive tracking for the CI-REPAIR-BENCH pipeline.

Tracks per-agent and overall:
  LLM API calls
    - Input tokens  (prompt)
    - Output tokens (completion)
    - API call count
    - Estimated cost (USD)

  Subprocess tool calls  (ruff, black, mypy, pip, …)
    - Tool name, command, call type
    - Success / return-code
    - Wall-clock time

Usage
-----
    from utilities.token_tracker import TokenTracker

    tracker = TokenTracker(model_name="gpt-5-mini", log_analyzer_type="llm")

    # LLM calls are captured automatically by TrackedLLM (llm_provider.py).
    # Tool calls are recorded explicitly in patch_generation.py via:
    #   tracker.record_tool_call(...)

    # At the end of a run:
    tracker.print_summary()
    tracker.save_json("results/gpt-5-mini_llm/token_report.json")
"""

from __future__ import annotations

import datetime
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
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LLMCallRecord:
    """One LLM API call."""
    seq: int
    agent: str           # e.g. "CILogAnalyzerLLM"
    call_site: str       # auto-detected method, e.g. "CILogAnalyzerLLM.ci_log_analysis"
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCallRecord:
    """One subprocess tool invocation (ruff, black, pip, …)."""
    seq: int
    agent: str           # e.g. "PatchGeneration"
    tool_name: str       # short name: "ruff", "black", "pip"
    command: str         # full command string as executed
    call_type: str       # "install" | "lint_fix" | "lint_check" | "format" | "import_sort" | "other"
    success: bool
    return_code: int
    elapsed_sec: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class TokenTracker:
    """
    Central accumulator for all LLM API calls and subprocess tool calls
    across the pipeline.
    """

    def __init__(self, model_name: str = "unknown", log_analyzer_type: str = "llm"):
        self.model_name = model_name
        self.log_analyzer_type = log_analyzer_type
        self.started_at: float = time.time()

        self._llm_calls: List[LLMCallRecord] = []
        self._tool_calls: List[ToolCallRecord] = []

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000

    @staticmethod
    def _extract_tool_name(command: str) -> str:
        """
        Derive a short tool name from a command string.
        e.g. "ruff check --fix foo.py" -> "ruff"
             "python -m ruff format foo.py" -> "ruff"
             "pip install black" -> "pip"
        """
        parts = command.strip().split()
        if not parts:
            return "unknown"
        # Handle "python -m <module> ..." -> module name
        if len(parts) >= 3 and parts[0] in ("python", "python3") and parts[1] == "-m":
            return parts[2]
        return parts[0]

    # -----------------------------------------------------------------------
    # LLM call recording
    # -----------------------------------------------------------------------

    def record(
        self,
        agent: str,
        call_site: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record one LLM API call (called by TrackedLLM.invoke)."""
        total = input_tokens + output_tokens
        cost = self._compute_cost(model, input_tokens, output_tokens)
        self._llm_calls.append(
            LLMCallRecord(
                seq=len(self._llm_calls) + 1,
                agent=agent,
                call_site=call_site,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
                cost_usd=cost,
            )
        )

    # -----------------------------------------------------------------------
    # Tool call recording
    # -----------------------------------------------------------------------

    def record_tool_call(
        self,
        agent: str,
        command: str,
        call_type: str,
        success: bool,
        return_code: int,
        elapsed_sec: float,
        tool_name: Optional[str] = None,
    ) -> None:
        """
        Record one subprocess tool invocation.

        Parameters
        ----------
        agent       : agent class name, e.g. "PatchGeneration"
        command     : the full command that was run (string or list joined)
        call_type   : "install" | "lint_fix" | "lint_check" | "format" |
                      "import_sort" | "other"
        success     : True if return_code == 0
        return_code : process return code
        elapsed_sec : wall-clock time in seconds
        tool_name   : optional override; auto-derived from command if omitted
        """
        name = tool_name or self._extract_tool_name(command)
        self._tool_calls.append(
            ToolCallRecord(
                seq=len(self._tool_calls) + 1,
                agent=agent,
                tool_name=name,
                command=command,
                call_type=call_type,
                success=success,
                return_code=return_code,
                elapsed_sec=round(elapsed_sec, 3),
            )
        )

    # -----------------------------------------------------------------------
    # Summary generation
    # -----------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Return a fully structured summary dict (JSON-serializable)."""
        return {
            "run_metadata": {
                "model": self.model_name,
                "log_analyzer_type": self.log_analyzer_type,
                "started_at": datetime.datetime.fromtimestamp(self.started_at).isoformat(),
            },
            "llm": self._llm_summary(),
            "tool_calls": self._tool_summary(),
            "all_llm_calls": [
                {
                    "seq": c.seq,
                    "agent": c.agent,
                    "call_site": c.call_site,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "total_tokens": c.total_tokens,
                    "cost_usd": round(c.cost_usd, 8),
                    "timestamp": c.timestamp,
                }
                for c in self._llm_calls
            ],
            "all_tool_calls": [
                {
                    "seq": c.seq,
                    "agent": c.agent,
                    "tool_name": c.tool_name,
                    "command": c.command,
                    "call_type": c.call_type,
                    "success": c.success,
                    "return_code": c.return_code,
                    "elapsed_sec": c.elapsed_sec,
                    "timestamp": c.timestamp,
                }
                for c in self._tool_calls
            ],
        }

    # ------------------------------------------------------------------
    def _llm_summary(self) -> dict:
        per_agent: Dict[str, dict] = {}

        for c in self._llm_calls:
            if c.agent not in per_agent:
                per_agent[c.agent] = {
                    "api_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "per_call_site": {},
                }
            a = per_agent[c.agent]
            a["api_calls"] += 1
            a["input_tokens"] += c.input_tokens
            a["output_tokens"] += c.output_tokens
            a["total_tokens"] += c.total_tokens
            a["cost_usd"] += c.cost_usd

            cs = a["per_call_site"].setdefault(
                c.call_site,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            cs["calls"] += 1
            cs["input_tokens"] += c.input_tokens
            cs["output_tokens"] += c.output_tokens
            cs["cost_usd"] += c.cost_usd

        # round floats
        for a in per_agent.values():
            a["cost_usd"] = round(a["cost_usd"], 6)
            for cs in a["per_call_site"].values():
                cs["cost_usd"] = round(cs["cost_usd"], 6)

        total_in = sum(c.input_tokens for c in self._llm_calls)
        total_out = sum(c.output_tokens for c in self._llm_calls)
        total_cost = sum(c.cost_usd for c in self._llm_calls)

        return {
            "overall": {
                "api_calls": len(self._llm_calls),
                "input_tokens": total_in,
                "output_tokens": total_out,
                "total_tokens": total_in + total_out,
                "cost_usd": round(total_cost, 6),
            },
            "per_agent": per_agent,
        }

    # ------------------------------------------------------------------
    def _tool_summary(self) -> dict:
        per_agent: Dict[str, dict] = {}
        per_tool: Dict[str, dict] = {}

        for c in self._tool_calls:
            # --- per agent ---
            if c.agent not in per_agent:
                per_agent[c.agent] = {
                    "total_calls": 0,
                    "successful": 0,
                    "failed": 0,
                    "per_tool": {},
                }
            ag = per_agent[c.agent]
            ag["total_calls"] += 1
            if c.success:
                ag["successful"] += 1
            else:
                ag["failed"] += 1

            t_ag = ag["per_tool"].setdefault(
                c.tool_name,
                {"calls": 0, "successful": 0, "failed": 0, "call_types": {}},
            )
            t_ag["calls"] += 1
            if c.success:
                t_ag["successful"] += 1
            else:
                t_ag["failed"] += 1
            t_ag["call_types"][c.call_type] = t_ag["call_types"].get(c.call_type, 0) + 1

            # --- overall per tool ---
            t_ov = per_tool.setdefault(
                c.tool_name,
                {"calls": 0, "successful": 0, "failed": 0, "call_types": {}},
            )
            t_ov["calls"] += 1
            if c.success:
                t_ov["successful"] += 1
            else:
                t_ov["failed"] += 1
            t_ov["call_types"][c.call_type] = t_ov["call_types"].get(c.call_type, 0) + 1

        total = len(self._tool_calls)
        succeeded = sum(1 for c in self._tool_calls if c.success)

        return {
            "overall": {
                "total_calls": total,
                "successful": succeeded,
                "failed": total - succeeded,
            },
            "per_tool": per_tool,
            "per_agent": per_agent,
        }

    # -----------------------------------------------------------------------
    # Human-readable output
    # -----------------------------------------------------------------------

    def print_summary(self) -> None:
        summary = self.get_summary()
        sep = "=" * 68
        thin = "-" * 68

        # ---- LLM section ----
        llm = summary["llm"]
        ov = llm["overall"]
        print(f"\n{sep}")
        print("  LLM API USAGE")
        print(sep)
        print(f"  {'Total API calls':<30}: {ov['api_calls']:>8}")
        print(f"  {'Total input tokens':<30}: {ov['input_tokens']:>12,}")
        print(f"  {'Total output tokens':<30}: {ov['output_tokens']:>12,}")
        print(f"  {'Total tokens':<30}: {ov['total_tokens']:>12,}")
        print(f"  {'Estimated cost (USD)':<30}: ${ov['cost_usd']:>12.6f}")
        print(thin)
        for agent, data in llm["per_agent"].items():
            print(f"\n  [{agent}]")
            print(f"    {'API calls':<26}: {data['api_calls']}")
            print(f"    {'Input tokens':<26}: {data['input_tokens']:,}")
            print(f"    {'Output tokens':<26}: {data['output_tokens']:,}")
            print(f"    {'Total tokens':<26}: {data['total_tokens']:,}")
            print(f"    {'Cost (USD)':<26}: ${data['cost_usd']:.6f}")
            for site, cs in data["per_call_site"].items():
                print(
                    f"      ├─ {site:<44} "
                    f"calls={cs['calls']:>3}  "
                    f"in={cs['input_tokens']:>9,}  "
                    f"out={cs['output_tokens']:>7,}  "
                    f"${cs['cost_usd']:.6f}"
                )

        # ---- Tool calls section ----
        tools = summary["tool_calls"]
        tov = tools["overall"]
        print(f"\n{sep}")
        print("  TOOL CALLS (subprocess)")
        print(sep)
        print(f"  {'Total tool calls':<30}: {tov['total_calls']:>8}")
        print(f"  {'Successful':<30}: {tov['successful']:>8}")
        print(f"  {'Failed':<30}: {tov['failed']:>8}")
        print(thin)
        for tool, td in tools["per_tool"].items():
            print(
                f"  {tool:<20} calls={td['calls']:>3}  "
                f"ok={td['successful']:>3}  fail={td['failed']:>3}  "
                f"types={td['call_types']}"
            )
        if tools["per_agent"]:
            print(thin)
            for agent, ad in tools["per_agent"].items():
                print(f"\n  [{agent}]")
                print(f"    total={ad['total_calls']}  ok={ad['successful']}  fail={ad['failed']}")
                for tn, td in ad["per_tool"].items():
                    print(
                        f"      ├─ {tn:<18} calls={td['calls']:>3}  "
                        f"ok={td['successful']:>3}  fail={td['failed']:>3}  "
                        f"types={td['call_types']}"
                    )
        print(f"\n{sep}\n")

    # -----------------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------------

    def save_json(self, path: str) -> None:
        """Save the full summary (including every individual call record) to JSON."""
        summary = self.get_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[TokenTracker] Saved report → {path}")
