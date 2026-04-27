#!/usr/bin/env python3
"""
Comprehensive results analysis for CI-REPAIR-BENCH baselines.

Reads all available result files for every strategy and produces:
  1. Fault Localization metrics table (Exact Match, Hit@k, P/R/F1)
  2. Patch Generation stats (patch count, avg diff size)
  3. Token & tool-call analysis (only for strategies with token_report.json)
  4. LaTeX-ready tables for the paper

Usage (from repo root):
    python baselines/scripts/results_analysis.py
    python baselines/scripts/results_analysis.py --results_dir baselines/results
    python baselines/scripts/results_analysis.py --strategy gpt-4o-mini_bm25
    python baselines/scripts/results_analysis.py --output paper_tables.tex
"""

import argparse
import json
import os
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Strategy display names  (model, log-retrieval-type)
# ---------------------------------------------------------------------------
STRATEGY_LABELS = {
    "gpt-4o-mini_bm25":   ("GPT-4o-mini",     "BM25"),
    "gpt-4o-mini_llm":    ("GPT-4o-mini",     "LLM"),
    "gpt-5-mini_bm25":    ("GPT-4.1-mini",    "BM25"),
    "gpt-5-mini_llm":     ("GPT-4.1-mini",    "LLM"),
    "deepseek-chat_bm25": ("DeepSeek-Chat",   "BM25"),
    "deepseek-chat_llm":  ("DeepSeek-Chat",   "LLM"),
    "deepseek-coder_bm25":("DeepSeek-Coder",  "BM25"),
    "deepseek-coder_llm": ("DeepSeek-Coder",  "LLM"),
}

STAGE_ORDER  = ["CILogAnalyzerLLM", "FaultLocalization", "PatchGeneration"]
STAGE_LABELS = {
    "CILogAnalyzerLLM":  "Log Analysis",
    "FaultLocalization": "Fault Local.",
    "PatchGeneration":   "Patch Gen.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def pct(v):
    """Float fraction → percentage string."""
    return f"{v * 100:.1f}"


def fmt_k(v):
    return f"{v / 1000:.1f}K"


def fmt_pct_latex(v):
    if v is None:
        return "--"
    return f"{v * 100:.1f}\\%"


def fmt_cost(v):
    return f"\\${v:.4f}"


# ---------------------------------------------------------------------------
# 1. Fault Localization metrics
# ---------------------------------------------------------------------------
def load_fl_metrics(results_dir: Path, strategy: str) -> dict | None:
    path = results_dir / strategy / "fl_evaluation.json"
    if not path.exists():
        return None
    data = load_json(path)
    agg  = data["aggregate"]
    return {
        "tasks_evaluated":      agg.get("tasks_evaluated", 0),
        "tasks_skipped":        agg.get("tasks_skipped_no_patch", 0),
        "exact_match_rate":     agg.get("exact_match_rate", 0.0),
        "hit_at_1_rate":        agg.get("hit_at_1_rate",   0.0),
        "hit_at_3_rate":        agg.get("hit_at_3_rate",   0.0),
        "hit_at_5_rate":        agg.get("hit_at_5_rate",   0.0),
        "mean_precision":       agg.get("mean_precision",  0.0),
        "mean_recall":          agg.get("mean_recall",     0.0),
        "mean_f1":              agg.get("mean_f1",         0.0),
    }


# ---------------------------------------------------------------------------
# 2. Patch generation stats
# ---------------------------------------------------------------------------
def load_patch_stats(results_dir: Path, strategy: str) -> dict | None:
    path = results_dir / strategy / "generated_patches.json"
    if not path.exists():
        return None
    patches = load_json(path)

    total     = len(patches)
    non_empty = [p for p in patches if p.get("diff", "").strip()]
    empty     = total - len(non_empty)
    line_counts = [len(p["diff"].split("\n")) for p in non_empty]

    return {
        "total_patches":  total,
        "non_empty":      len(non_empty),
        "empty":          empty,
        "avg_diff_lines": statistics.mean(line_counts)   if line_counts else 0.0,
        "med_diff_lines": statistics.median(line_counts) if line_counts else 0.0,
        "max_diff_lines": max(line_counts)               if line_counts else 0,
    }


# ---------------------------------------------------------------------------
# 3. Token / tool-call analysis  (only for strategies with token_report.json)
# ---------------------------------------------------------------------------
def aggregate_stage_tokens(traces):
    """step_trace list → {task_id: {agent: {metrics}}}"""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(
        lambda: dict(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0, calls=0)
    ))
    for rec in traces:
        tid   = str(rec["task_id"])
        agent = rec["agent"]
        a = agg[tid][agent]
        a["input_tokens"]  += rec.get("input_tokens",  0)
        a["output_tokens"] += rec.get("output_tokens", 0)
        a["total_tokens"]  += rec.get("total_tokens",  0)
        a["cost_usd"]      += rec.get("cost_usd",      0.0)
        a["calls"]         += 1
    return agg


def compute_stage_avgs(stage_tokens, task_tools, stages):
    from collections import defaultdict
    per_stage = {s: defaultdict(list) for s in stages}
    for tid, agents in stage_tokens.items():
        for stage in stages:
            if stage not in agents:
                continue
            m  = agents[stage]
            ps = per_stage[stage]
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
        ps = per_stage[stage]
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


def compute_overall_avgs(stage_tokens, task_tools):
    per_task_input, per_task_output, per_task_total, per_task_cost = [], [], [], []
    per_task_tools, per_task_tool_sr = [], []
    for tid, agents in stage_tokens.items():
        ti = to = tt = tc = 0.0
        for m in agents.values():
            ti += m["input_tokens"]
            to += m["output_tokens"]
            tt += m["total_tokens"]
            tc += m["cost_usd"]
        per_task_input.append(ti)
        per_task_output.append(to)
        per_task_total.append(tt)
        per_task_cost.append(tc)
        if tid in task_tools:
            per_task_tools.append(task_tools[tid]["total_calls"])
            per_task_tool_sr.append(task_tools[tid].get("success_rate", 0.0))
    return dict(
        n_tasks          = len(per_task_total),
        avg_input        = statistics.mean(per_task_input)    if per_task_input    else 0,
        avg_output       = statistics.mean(per_task_output)   if per_task_output   else 0,
        avg_total        = statistics.mean(per_task_total)    if per_task_total    else 0,
        avg_cost         = statistics.mean(per_task_cost)     if per_task_cost     else 0,
        avg_tool_calls   = statistics.mean(per_task_tools)    if per_task_tools    else None,
        avg_tool_success = statistics.mean(per_task_tool_sr)  if per_task_tool_sr  else None,
    )


def load_token_stats(results_dir: Path, strategy: str) -> dict | None:
    step_path   = results_dir / strategy / "step_trace.json"
    report_path = results_dir / strategy / "token_report.json"
    if not (step_path.exists() and report_path.exists()):
        return None

    traces = load_json(step_path)
    report = load_json(report_path)

    stage_tokens = aggregate_stage_tokens(traces)
    task_tools   = {str(t["task_id"]): t["tool_calls"] for t in report["per_task"]}

    stage_avgs = compute_stage_avgs(stage_tokens, task_tools, STAGE_ORDER)
    overall    = compute_overall_avgs(stage_tokens, task_tools)

    return {
        "global":      report["llm"]["overall"],
        "tool_global": report["tool_calls"]["overall"],
        "tool_detail": report["tool_calls"]["per_tool"],
        "stage_avgs":  stage_avgs,
        "overall":     overall,
    }


# ---------------------------------------------------------------------------
# Collect all data
# ---------------------------------------------------------------------------
def collect_all(results_dir: Path, filter_strategy: str | None = None) -> dict:
    strategies = sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(results_dir / d) and not d.startswith(".")
    )
    if filter_strategy:
        strategies = [s for s in strategies if s == filter_strategy]

    data = {}
    for s in strategies:
        data[s] = {
            "fl":    load_fl_metrics(results_dir, s),
            "patch": load_patch_stats(results_dir, s),
            "token": load_token_stats(results_dir, s),
        }
    return data


# ---------------------------------------------------------------------------
# Plain-text printing
# ---------------------------------------------------------------------------
def print_fl_table(data: dict):
    print("\n" + "=" * 100)
    print("  FAULT LOCALIZATION METRICS")
    print("=" * 100)
    hdr = f"  {'Strategy':<25} {'Log':<6} {'Tasks':>6} {'Exact':>7} {'H@1':>6} {'H@3':>6} {'H@5':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print(hdr)
    print("  " + "-" * 98)
    for s, d in data.items():
        fl = d["fl"]
        if fl is None:
            print(f"  {s:<25}  (no fl_evaluation.json)")
            continue
        model, log = STRATEGY_LABELS.get(s, (s, "?"))
        print(
            f"  {model:<25} {log:<6} {fl['tasks_evaluated']:>6} "
            f"{pct(fl['exact_match_rate']):>6}% "
            f"{pct(fl['hit_at_1_rate']):>5}% "
            f"{pct(fl['hit_at_3_rate']):>5}% "
            f"{pct(fl['hit_at_5_rate']):>5}% "
            f"{pct(fl['mean_precision']):>6}% "
            f"{pct(fl['mean_recall']):>6}% "
            f"{pct(fl['mean_f1']):>6}%"
        )


def print_patch_table(data: dict):
    print("\n" + "=" * 80)
    print("  PATCH GENERATION STATS")
    print("=" * 80)
    hdr = f"  {'Strategy':<25} {'Log':<6} {'Total':>7} {'Non-Empty':>10} {'Empty':>6} {'Avg Lines':>10} {'Med Lines':>10}"
    print(hdr)
    print("  " + "-" * 78)
    for s, d in data.items():
        ps = d["patch"]
        if ps is None:
            print(f"  {s:<25}  (no generated_patches.json)")
            continue
        model, log = STRATEGY_LABELS.get(s, (s, "?"))
        print(
            f"  {model:<25} {log:<6} {ps['total_patches']:>7} "
            f"{ps['non_empty']:>10} {ps['empty']:>6} "
            f"{ps['avg_diff_lines']:>10.1f} {ps['med_diff_lines']:>10.1f}"
        )


def print_token_table(data: dict):
    has_token = [(s, d["token"]) for s, d in data.items() if d["token"] is not None]
    if not has_token:
        print("\n  (No token_report.json found for any strategy)")
        return

    print("\n" + "=" * 100)
    print("  TOKEN & TOOL-CALL ANALYSIS  (per-task averages)")
    print("=" * 100)

    for s, tok in has_token:
        model, log = STRATEGY_LABELS.get(s, (s, "?"))
        print(f"\n  Strategy: {model} / {log}   ({tok['overall']['n_tasks']} tasks)")
        g = tok["global"]
        tg = tok["tool_global"]
        print(f"    Global: input={g['input_tokens']:,}  output={g['output_tokens']:,}  "
              f"total={g['total_tokens']:,}  cost=${g['cost_usd']:.4f}")
        print(f"    Tools:  total={tg['total_calls']:,}  succ={tg['successful']:,}  "
              f"fail={tg['failed']:,}  sr={tg['success_rate']*100:.1f}%")
        print()
        hdr = f"    {'Stage':<20} {'Avg In':>9} {'Avg Out':>9} {'Avg Tot':>9} {'Avg Cost':>10} {'API/task':>9} {'Tools/task':>11} {'Tool Succ':>10}"
        print(hdr)
        print("    " + "-" * 95)
        for stage in STAGE_ORDER:
            a     = tok["stage_avgs"][stage]
            tools = f"{a['avg_tool_calls']:.1f}" if a["avg_tool_calls"] is not None else "--"
            tool_sr = f"{a['avg_tool_success']*100:.1f}%" if a["avg_tool_success"] is not None else "--"
            print(
                f"    {STAGE_LABELS[stage]:<20} "
                f"{fmt_k(a['avg_input']):>9} {fmt_k(a['avg_output']):>9} "
                f"{fmt_k(a['avg_total']):>9} ${a['avg_cost']:>9.5f} "
                f"{a['avg_api_calls']:>9.1f} {tools:>11} {tool_sr:>10}"
            )
        print("    " + "-" * 95)
        ov = tok["overall"]
        tools = f"{ov['avg_tool_calls']:.1f}" if ov["avg_tool_calls"] is not None else "--"
        tool_sr = f"{ov['avg_tool_success']*100:.1f}%" if ov["avg_tool_success"] is not None else "--"
        print(
            f"    {'TOTAL':<20} "
            f"{fmt_k(ov['avg_input']):>9} {fmt_k(ov['avg_output']):>9} "
            f"{fmt_k(ov['avg_total']):>9} ${ov['avg_cost']:>9.5f} "
            f"{'':>9} {tools:>11} {tool_sr:>10}"
        )

        print(f"\n    Stage token share:")
        total = sum(tok["stage_avgs"][s]["avg_total"] for s in STAGE_ORDER)
        for stage in STAGE_ORDER:
            share = tok["stage_avgs"][stage]["avg_total"] / total * 100 if total else 0
            print(f"      {STAGE_LABELS[stage]:<20} {share:.1f}%")

        print(f"\n    Tool breakdown:")
        for tool, info in tok["tool_detail"].items():
            sr = info["successful"] / info["calls"] * 100 if info["calls"] else 0
            print(f"      {tool:<15} calls={info['calls']:>5}  succ={info['successful']:>5}  "
                  f"fail={info['failed']:>4}  sr={sr:.1f}%")


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------
def latex_fl_table(data: dict) -> str:
    rows = []
    prev_model = None
    for s, d in data.items():
        fl = d["fl"]
        if fl is None:
            continue
        model, log = STRATEGY_LABELS.get(s, (s, "?"))
        midrule = "    \\midrule\n" if model != prev_model and prev_model is not None else ""
        prev_model = model
        row = (
            f"    {midrule}"
            f"    {model:<22} & {log:<6} "
            f"& {fl['tasks_evaluated']:>5} "
            f"& {pct(fl['exact_match_rate']):>6}\\% "
            f"& {pct(fl['hit_at_1_rate']):>6}\\% "
            f"& {pct(fl['hit_at_3_rate']):>6}\\% "
            f"& {pct(fl['hit_at_5_rate']):>6}\\% "
            f"& {pct(fl['mean_precision']):>6}\\% "
            f"& {pct(fl['mean_recall']):>6}\\% "
            f"& {pct(fl['mean_f1']):>6}\\% \\\\"
        )
        rows.append(row)

    return r"""\begin{table}[t]
\centering
\caption{Fault localization evaluation across all strategies.
         \emph{Exact} = exact file match rate;
         \emph{H@k} = hit rate at rank $k$;
         \emph{P/R/F1} = file-level precision, recall, F1.}
\label{tab:fl_results}
\small
\begin{tabular}{l l r r r r r r r r}
\toprule
\textbf{Model} & \textbf{Log} & \textbf{$N$}
  & \textbf{Exact} & \textbf{H@1} & \textbf{H@3} & \textbf{H@5}
  & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def latex_patch_table(data: dict) -> str:
    rows = []
    prev_model = None
    for s, d in data.items():
        ps = d["patch"]
        if ps is None:
            continue
        model, log = STRATEGY_LABELS.get(s, (s, "?"))
        midrule = "    \\midrule\n" if model != prev_model and prev_model is not None else ""
        prev_model = model
        row = (
            f"    {midrule}"
            f"    {model:<22} & {log:<6} "
            f"& {ps['total_patches']:>5} "
            f"& {ps['non_empty']:>9} "
            f"& {ps['empty']:>5} "
            f"& {ps['avg_diff_lines']:>9.1f} "
            f"& {ps['med_diff_lines']:>9.1f} \\\\"
        )
        rows.append(row)

    return r"""\begin{table}[t]
\centering
\caption{Patch generation statistics across all strategies.
         \emph{N} = total patches generated;
         \emph{Non-Empty} = patches with non-trivial diff content;
         \emph{Avg/Med Lines} = average and median diff line count.}
\label{tab:patch_stats}
\small
\begin{tabular}{l l r r r r r}
\toprule
\textbf{Model} & \textbf{Log} & \textbf{$N$}
  & \textbf{Non-Empty} & \textbf{Empty}
  & \textbf{Avg Lines} & \textbf{Med Lines} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def latex_token_table(data: dict) -> str:
    has_token = [(s, d["token"]) for s, d in data.items() if d["token"] is not None]
    if not has_token:
        return ""

    all_rows = []
    for s, tok in has_token:
        model, log = STRATEGY_LABELS.get(s, (s, "?"))
        n_stages = len(STAGE_ORDER)
        for i, stage in enumerate(STAGE_ORDER):
            a = tok["stage_avgs"][stage]
            strat_col = f"\\multirow{{{n_stages}}}{{*}}{{{model} / {log}}}" if i == 0 else ""
            tools   = f"{a['avg_tool_calls']:.1f}"          if a["avg_tool_calls"]   is not None else "--"
            tool_sr = fmt_pct_latex(a["avg_tool_success"])
            row = (
                f"    {strat_col:<40} & {STAGE_LABELS[stage]:<16} "
                f"& {fmt_k(a['avg_input']):>8} "
                f"& {fmt_k(a['avg_output']):>8} "
                f"& {fmt_k(a['avg_total']):>8} "
                f"& {fmt_cost(a['avg_cost']):>10} "
                f"& {tools:>8} "
                f"& {tool_sr:>8} \\\\"
            )
            all_rows.append(row)
        # Overall row
        ov = tok["overall"]
        tools   = f"{ov['avg_tool_calls']:.1f}"         if ov["avg_tool_calls"]   is not None else "--"
        tool_sr = fmt_pct_latex(ov["avg_tool_success"])
        all_rows.append("    \\midrule")
        all_rows.append(
            f"    {'':40} & {'Overall':<16} "
            f"& {fmt_k(ov['avg_input']):>8} "
            f"& {fmt_k(ov['avg_output']):>8} "
            f"& {fmt_k(ov['avg_total']):>8} "
            f"& {fmt_cost(ov['avg_cost']):>10} "
            f"& {tools:>8} "
            f"& {tool_sr:>8} \\\\"
        )

    return r"""\begin{table}[t]
\centering
\caption{Per-stage average token consumption and tool-call statistics.
         Values are per-task averages.
         \emph{In.}/\emph{Out.}/\emph{Tot.} in units of $10^3$ tokens (K).}
\label{tab:token_analysis}
\small
\begin{tabular}{l l r r r r r r}
\toprule
\textbf{Strategy} & \textbf{Stage}
  & \textbf{Avg In.} & \textbf{Avg Out.} & \textbf{Avg Tot.}
  & \textbf{Avg Cost} & \textbf{Avg Tools} & \textbf{Tool Succ.} \\
\midrule
""" + "\n".join(all_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CI-REPAIR-BENCH results analysis")
    parser.add_argument(
        "--results_dir",
        default="baselines/results",
        help="Path to the results directory (default: baselines/results)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Analyse a single strategy only (e.g. gpt-4o-mini_bm25)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write LaTeX tables to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    print(f"\nCollecting results from: {results_dir.resolve()}")
    data = collect_all(results_dir, filter_strategy=args.strategy)
    print(f"Strategies found: {list(data.keys())}\n")

    # --- plain-text output ---
    print_fl_table(data)
    print_patch_table(data)
    print_token_table(data)

    # --- LaTeX output ---
    latex_parts = [
        "% ============================================================",
        "% Auto-generated by baselines/scripts/results_analysis.py",
        "% ============================================================\n",
        latex_fl_table(data),
        latex_patch_table(data),
        latex_token_table(data),
    ]
    latex = "\n".join(latex_parts)

    print("\n" + "=" * 80)
    print("  LaTeX Tables")
    print("=" * 80)
    print(latex)

    if args.output:
        out = Path(args.output)
        out.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX written to: {out}")


if __name__ == "__main__":
    main()
