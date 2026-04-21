#!/usr/bin/env python3
"""
Token and tool-call analysis for CI-REPAIR-BENCH results.

Produces:
  1. TOTALS per stage: input tokens, output tokens, total tokens, cost, time
  2. Per-task averages per stage (for LaTeX table)
  3. Tool-call breakdown per tool
  4. LaTeX tabular snippet ready to paste into a paper

Usage:
    python scripts/token_analysis.py
    python scripts/token_analysis.py --output paper/token_table.tex
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Stage display names & order
# ---------------------------------------------------------------------------
STAGE_ORDER = ["CILogAnalyzerLLM", "FaultLocalization", "PatchGeneration"]
STAGE_LABELS = {
    "CILogAnalyzerLLM": "Log Analysis",
    "FaultLocalization": "Fault Local.",
    "PatchGeneration":   "Patch Gen.",
}


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. TOTAL per stage — read directly from token_report per_agent
#    (these are exact, pre-computed sums — no floating-point re-aggregation)
# ---------------------------------------------------------------------------
def extract_stage_totals(token_report: dict) -> dict:
    """
    Returns:
        {agent: {api_calls, input_tokens, output_tokens, total_tokens, cost_usd}}
    """
    totals = {}
    for agent, info in token_report["llm"]["per_agent"].items():
        totals[agent] = {
            "api_calls":     info["api_calls"],
            "input_tokens":  info["input_tokens"],
            "output_tokens": info["output_tokens"],
            "total_tokens":  info["total_tokens"],
            "cost_usd":      info["cost_usd"],
        }
    return totals


# ---------------------------------------------------------------------------
# 2. Per-stage elapsed time — derived from per-task timestamp spans
# ---------------------------------------------------------------------------
def compute_stage_elapsed(traces: list) -> dict:
    """
    For each agent, collect (last_ts - first_ts) per task, then sum across tasks.
    Returns:
        {agent: {total_sec, avg_sec, n_tasks}}
    """
    task_agent_ts: dict = defaultdict(lambda: defaultdict(list))
    for rec in traces:
        task_agent_ts[str(rec["task_id"])][rec["agent"]].append(rec["timestamp"])

    stage_spans: dict = defaultdict(list)
    for tid, agents in task_agent_ts.items():
        for agent, tss in agents.items():
            span = max(tss) - min(tss) if len(tss) > 1 else 0.0
            stage_spans[agent].append(span)

    result = {}
    for agent, spans in stage_spans.items():
        result[agent] = {
            "total_sec": sum(spans),
            "avg_sec":   statistics.mean(spans),
            "n_tasks":   len(spans),
        }
    return result


# ---------------------------------------------------------------------------
# 3. Per-task per-stage token aggregation — for averages
# ---------------------------------------------------------------------------
def aggregate_per_task(traces: list) -> dict:
    """
    Returns:
        {task_id: {agent: {input, output, total, cost, calls}}}
    """
    agg: dict = defaultdict(lambda: defaultdict(
        lambda: dict(input_tokens=0, output_tokens=0,
                     total_tokens=0, cost_usd=0.0, calls=0)
    ))
    for rec in traces:
        tid = str(rec["task_id"])
        a   = agg[tid][rec["agent"]]
        a["input_tokens"]  += rec.get("input_tokens",  0)
        a["output_tokens"] += rec.get("output_tokens", 0)
        a["total_tokens"]  += rec.get("total_tokens",  0)
        a["cost_usd"]      += rec.get("cost_usd",      0.0)
        a["calls"]         += 1
    return agg


def compute_per_stage_avgs(per_task_data: dict, task_tools: dict, stages: list) -> dict:
    bucket: dict = {s: defaultdict(list) for s in stages}

    for tid, agents in per_task_data.items():
        for stage in stages:
            if stage not in agents:
                continue
            m  = agents[stage]
            ps = bucket[stage]
            ps["input_tokens"].append(m["input_tokens"])
            ps["output_tokens"].append(m["output_tokens"])
            ps["total_tokens"].append(m["total_tokens"])
            ps["cost_usd"].append(m["cost_usd"])
            ps["calls"].append(m["calls"])
            if stage == "PatchGeneration" and tid in task_tools:
                tc = task_tools[tid]
                ps["tool_calls"].append(tc["total_calls"])
                ps["tool_success_rate"].append(tc.get("success_rate", 0.0))

    avgs = {}
    for stage in stages:
        ps = bucket[stage]
        n  = len(ps["total_tokens"])
        avgs[stage] = dict(
            n_tasks          = n,
            avg_input        = statistics.mean(ps["input_tokens"])  if ps["input_tokens"]  else 0,
            avg_output       = statistics.mean(ps["output_tokens"]) if ps["output_tokens"] else 0,
            avg_total        = statistics.mean(ps["total_tokens"])  if ps["total_tokens"]  else 0,
            avg_cost         = statistics.mean(ps["cost_usd"])      if ps["cost_usd"]      else 0,
            avg_api_calls    = statistics.mean(ps["calls"])         if ps["calls"]         else 0,
            avg_tool_calls   = statistics.mean(ps["tool_calls"])         if ps.get("tool_calls") else None,
            avg_tool_success = statistics.mean(ps["tool_success_rate"])  if ps.get("tool_success_rate") else None,
        )
    return avgs


def compute_overall_avgs(per_task_data: dict, task_tools: dict) -> dict:
    inputs, outputs, totals, costs = [], [], [], []
    tool_calls, tool_sr = [], []

    for tid, agents in per_task_data.items():
        ti = to = tt = tc = 0.0
        for m in agents.values():
            ti += m["input_tokens"]
            to += m["output_tokens"]
            tt += m["total_tokens"]
            tc += m["cost_usd"]
        inputs.append(ti); outputs.append(to)
        totals.append(tt); costs.append(tc)
        if tid in task_tools:
            tool_calls.append(task_tools[tid]["total_calls"])
            tool_sr.append(task_tools[tid].get("success_rate", 0.0))

    return dict(
        n_tasks          = len(totals),
        avg_input        = statistics.mean(inputs),
        avg_output       = statistics.mean(outputs),
        avg_total        = statistics.mean(totals),
        avg_cost         = statistics.mean(costs),
        total_input      = sum(inputs),
        total_output     = sum(outputs),
        total_tokens     = sum(totals),
        total_cost       = sum(costs),
        avg_tool_calls   = statistics.mean(tool_calls) if tool_calls else None,
        avg_tool_success = statistics.mean(tool_sr)    if tool_sr    else None,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_m(v: float) -> str:
    """Tokens in millions."""
    return f"{v/1e6:.2f}M"

def fmt_k(v: float) -> str:
    return f"{v/1000:.1f}K"

def fmt_h(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}h {m:02d}m {s:02d}s"

def fmt_pct_latex(v) -> str:
    return "--" if v is None else f"{v*100:.1f}\\%"

def fmt_cost_latex(v: float) -> str:
    return f"\\${v:.4f}"


# ---------------------------------------------------------------------------
# Print totals table
# ---------------------------------------------------------------------------
def print_totals(strategy: str, token_report: dict,
                 stage_totals: dict, stage_elapsed: dict) -> None:

    meta  = token_report["run_metadata"]
    g     = token_report["llm"]["overall"]
    tc_g  = token_report["tool_calls"]["overall"]
    total_run_sec = meta.get("elapsed_sec", 0)

    print()
    print("=" * 90)
    print(f"  STRATEGY : {strategy}")
    print(f"  MODEL    : {meta.get('model', '?')}   |   LOG ANALYZER: {meta.get('log_analyzer_type', '?')}")
    print(f"  RUN TIME : {fmt_h(total_run_sec)}  ({total_run_sec/3600:.2f} h wall-clock)")
    print("=" * 90)

    # ---- TOTAL TOKENS + COST PER STAGE ----
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("│  TOTALS per Stage  (exact sums from token_report)                              │")
    print("├──────────────────┬────────────┬───────────┬───────────┬───────────┬────────────┤")
    print("│  Stage           │  API Calls │  Input    │  Output   │  Total    │  Cost      │")
    print("├──────────────────┼────────────┼───────────┼───────────┼───────────┼────────────┤")

    for stage in STAGE_ORDER:
        if stage not in stage_totals:
            continue
        t = stage_totals[stage]
        label = STAGE_LABELS[stage]
        print(f"│  {label:<16} │ {t['api_calls']:>10,} │ {fmt_m(t['input_tokens']):>9} │"
              f" {fmt_m(t['output_tokens']):>9} │ {fmt_m(t['total_tokens']):>9} │"
              f" ${t['cost_usd']:>9.4f} │")

    print("├──────────────────┼────────────┼───────────┼───────────┼───────────┼────────────┤")
    print(f"│  {'TOTAL':<16} │ {g['api_calls']:>10,} │ {fmt_m(g['input_tokens']):>9} │"
          f" {fmt_m(g['output_tokens']):>9} │ {fmt_m(g['total_tokens']):>9} │"
          f" ${g['cost_usd']:>9.4f} │")
    print("└──────────────────┴────────────┴───────────┴───────────┴───────────┴────────────┘")

    # ---- TIME PER STAGE ----
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  TIME per Stage  (sum of per-task spans from step_trace timestamps) │")
    print("├──────────────────┬─────────┬────────────────────────┬──────────────┤")
    print("│  Stage           │  Tasks  │  Total Time            │  Avg / Task  │")
    print("├──────────────────┼─────────┼────────────────────────┼──────────────┤")

    for stage in STAGE_ORDER:
        if stage not in stage_elapsed:
            continue
        e = stage_elapsed[stage]
        label = STAGE_LABELS[stage]
        print(f"│  {label:<16} │ {e['n_tasks']:>7} │ {fmt_h(e['total_sec']):>22} │"
              f" {fmt_h(e['avg_sec']):>12} │")

    print("├──────────────────┼─────────┼────────────────────────┼──────────────┤")
    print(f"│  {'Wall-clock':<16} │         │ {fmt_h(total_run_sec):>22} │              │")
    print("└──────────────────┴─────────┴────────────────────────┴──────────────┘")
    print("  Note: stages run in parallel per task, so sum of stage times > wall-clock.")

    # ---- TOOL CALLS ----
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  TOOL CALLS  (PatchGeneration stage only)                           │")
    print("├────────────────┬────────┬────────────┬──────────┬───────────────── ┤")
    print("│  Tool          │  Calls │  Successful│  Failed  │  Success Rate    │")
    print("├────────────────┼────────┼────────────┼──────────┼───────────────── ┤")
    for tool, info in token_report["tool_calls"]["per_tool"].items():
        sr = info["successful"] / info["calls"] * 100 if info["calls"] else 0
        print(f"│  {tool:<14} │ {info['calls']:>6,} │ {info['successful']:>10,} │"
              f" {info['failed']:>8,} │ {sr:>14.1f}% │")
    print("├────────────────┼────────┼────────────┼──────────┼───────────────── ┤")
    print(f"│  {'TOTAL':<14} │ {tc_g['total_calls']:>6,} │ {tc_g['successful']:>10,} │"
          f" {tc_g['failed']:>8,} │ {tc_g['success_rate']*100:>14.1f}% │")
    print("└────────────────┴────────┴────────────┴──────────┴───────────────── ┘")


# ---------------------------------------------------------------------------
# Print per-task averages
# ---------------------------------------------------------------------------
def print_averages(stage_avgs: dict, overall: dict) -> None:
    print()
    print("┌─────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│  PER-TASK AVERAGES per Stage                                                            │")
    print("├──────────────────┬────────┬──────────┬──────────┬──────────┬──────────┬────────────────┤")
    print("│  Stage           │ Tasks  │ Avg In.  │ Avg Out. │ Avg Tot. │ Avg Cost │ Avg Tools/Succ │")
    print("├──────────────────┼────────┼──────────┼──────────┼──────────┼──────────┼────────────────┤")

    for stage in STAGE_ORDER:
        a = stage_avgs[stage]
        label = STAGE_LABELS[stage]
        tools = f"{a['avg_tool_calls']:.1f}" if a["avg_tool_calls"] is not None else "  --"
        tool_sr = f"{a['avg_tool_success']*100:.0f}%" if a["avg_tool_success"] is not None else "--"
        tools_col = f"{tools} / {tool_sr}"
        print(f"│  {label:<16} │ {a['n_tasks']:>6} │ {fmt_k(a['avg_input']):>8} │"
              f" {fmt_k(a['avg_output']):>8} │ {fmt_k(a['avg_total']):>8} │"
              f" ${a['avg_cost']:>7.4f} │ {tools_col:>14} │")

    print("├──────────────────┼────────┼──────────┼──────────┼──────────┼──────────┼────────────────┤")
    ov = overall
    tools = f"{ov['avg_tool_calls']:.1f}" if ov["avg_tool_calls"] is not None else "  --"
    tool_sr = f"{ov['avg_tool_success']*100:.0f}%" if ov["avg_tool_success"] is not None else "--"
    tools_col = f"{tools} / {tool_sr}"
    print(f"│  {'OVERALL':<16} │ {ov['n_tasks']:>6} │ {fmt_k(ov['avg_input']):>8} │"
          f" {fmt_k(ov['avg_output']):>8} │ {fmt_k(ov['avg_total']):>8} │"
          f" ${ov['avg_cost']:>7.4f} │ {tools_col:>14} │")
    print("└──────────────────┴────────┴──────────┴──────────┴──────────┴──────────┴────────────────┘")

    # Token share per stage
    total_tok = sum(stage_avgs[s]["avg_total"] for s in STAGE_ORDER)
    print()
    print("  Stage token share (% of avg total per task):")
    for stage in STAGE_ORDER:
        share = stage_avgs[stage]["avg_total"] / total_tok * 100 if total_tok else 0
        bar = "█" * int(share / 2)
        print(f"    {STAGE_LABELS[stage]:<20} {share:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# LaTeX table: totals + averages combined
# ---------------------------------------------------------------------------
def generate_latex(strategy: str, stage_totals: dict, stage_elapsed: dict,
                   stage_avgs: dict, overall: dict, token_report: dict) -> str:
    g    = token_report["llm"]["overall"]
    tc_g = token_report["tool_calls"]["overall"]
    n_stages = len(STAGE_ORDER)

    rows = []
    for i, stage in enumerate(STAGE_ORDER):
        t  = stage_totals.get(stage, {})
        e  = stage_elapsed.get(stage, {})
        a  = stage_avgs.get(stage, {})
        label = STAGE_LABELS[stage]

        strat_col = f"\\multirow{{{n_stages}}}{{*}}{{{strategy}}}" if i == 0 else ""
        tools_avg = f"{a.get('avg_tool_calls', 0):.1f}" if a.get("avg_tool_calls") is not None else "--"
        tool_sr   = fmt_pct_latex(a.get("avg_tool_success"))

        row = (
            f"  {strat_col:<38} & {label:<16}"
            f" & {fmt_m(t.get('input_tokens', 0)):>7}"
            f" & {fmt_m(t.get('output_tokens', 0)):>7}"
            f" & {fmt_m(t.get('total_tokens', 0)):>7}"
            f" & {fmt_cost_latex(t.get('cost_usd', 0)):>10}"
            f" & {fmt_k(a.get('avg_input', 0)):>8}"
            f" & {fmt_k(a.get('avg_output', 0)):>8}"
            f" & {fmt_cost_latex(a.get('avg_cost', 0)):>10}"
            f" & {e.get('total_sec', 0)/3600:>5.1f}h"
            f" & {tools_avg:>5}"
            f" & {tool_sr:>8} \\\\"
        )
        rows.append(row)

    # Overall row
    ov = overall
    tools_avg = f"{ov['avg_tool_calls']:.1f}" if ov["avg_tool_calls"] is not None else "--"
    tool_sr   = fmt_pct_latex(ov["avg_tool_success"])
    rows.append("  \\midrule")
    rows.append(
        f"  {'':38} & {'Overall':<16}"
        f" & {fmt_m(g['input_tokens']):>7}"
        f" & {fmt_m(g['output_tokens']):>7}"
        f" & {fmt_m(g['total_tokens']):>7}"
        f" & {fmt_cost_latex(g['cost_usd']):>10}"
        f" & {fmt_k(ov['avg_input']):>8}"
        f" & {fmt_k(ov['avg_output']):>8}"
        f" & {fmt_cost_latex(ov['avg_cost']):>10}"
        f" & --"
        f" & {tools_avg:>5}"
        f" & {tool_sr:>8} \\\\"
    )

    latex = r"""\begin{table*}[t]
\centering
\caption{Token consumption, cost, and tool-call statistics for \textbf{""" + strategy + r"""}.
  \emph{Totals} are exact sums across all tasks.
  \emph{Averages} are per-task means.
  Token counts: M = $\times10^{6}$, K = $\times10^{3}$.}
\label{tab:token_analysis}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l l rrr r rr r r r r}
\toprule
 & & \multicolumn{4}{c}{\textbf{Total (all tasks)}}
   & \multicolumn{3}{c}{\textbf{Avg (per task)}}
   & & \multicolumn{2}{c}{\textbf{Tools}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-9}\cmidrule(lr){11-12}
\textbf{Strategy} & \textbf{Stage}
  & \textbf{In.} & \textbf{Out.} & \textbf{Tot.} & \textbf{Cost}
  & \textbf{Avg In.} & \textbf{Avg Out.} & \textbf{Avg Cost}
  & \textbf{Time}
  & \textbf{Avg} & \textbf{Succ.} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    return latex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Token analysis for CI-REPAIR-BENCH")
    parser.add_argument(
        "--step_trace",
        default="baselines/results/gpt-5-mini_llm/step_trace.json",
    )
    parser.add_argument(
        "--token_report",
        default="baselines/results/gpt-5-mini_llm/token_report.json",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write LaTeX snippet to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    print(f"Loading step_trace  : {args.step_trace}")
    traces = load_json(args.step_trace)

    print(f"Loading token_report: {args.token_report}")
    report = load_json(args.token_report)

    # Read strategy label from run_metadata
    meta     = report.get("run_metadata", {})
    model    = meta.get("model", "unknown")
    log_type = meta.get("log_analyzer_type", "unknown").upper()
    strategy = f"{model} / {log_type}"

    print(f"\n  model            : {model}")
    print(f"  log_analyzer_type: {log_type}")
    print(f"  started_at       : {meta.get('started_at', '?')}")
    print(f"  wall-clock run   : {meta.get('elapsed_sec', 0)/3600:.2f} h")
    print(f"\n  NOTE: cost in token_report = actual API cost already incurred.")
    print(f"  Running this script costs $0.\n")

    # Compute all stats
    stage_totals  = extract_stage_totals(report)
    stage_elapsed = compute_stage_elapsed(traces)
    per_task_data = aggregate_per_task(traces)
    task_tools    = {str(t["task_id"]): t["tool_calls"] for t in report["per_task"]}
    stage_avgs    = compute_per_stage_avgs(per_task_data, task_tools, STAGE_ORDER)
    overall       = compute_overall_avgs(per_task_data, task_tools)

    # Print
    print_totals(strategy, report, stage_totals, stage_elapsed)
    print_averages(stage_avgs, overall)

    # LaTeX
    latex = generate_latex(strategy, stage_totals, stage_elapsed,
                           stage_avgs, overall, report)
    print()
    print("=" * 90)
    print("  LaTeX Table")
    print("=" * 90)
    print(latex)

    if args.output:
        out = Path(args.output)
        out.write_text(latex, encoding="utf-8")
        print(f"LaTeX written to: {out}")


if __name__ == "__main__":
    main()
