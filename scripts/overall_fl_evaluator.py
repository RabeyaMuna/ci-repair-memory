#!/usr/bin/env python3
"""
FL Evaluator for CI-REPAIR-BENCH — paper-ready metrics.

Subcommands
-----------
eval      Evaluate one strategy and save a full report.
tables    Load pre-computed reports and print RQ1 / RQ2 / efficiency tables
          (with optional LaTeX output).

Example — evaluate each strategy (auto-discover all files from the results dir):
  python scripts/overall_fl_evaluator.py eval \\
    --results-dir baselines/results/gpt-5-mini_llm \\
    --label       "GPT-5-mini" --strategy LLM

  python scripts/overall_fl_evaluator.py eval \\
    --results-dir baselines/results/gpt-5-mini_bm25 \\
    --label       "GPT-5-mini" --strategy BM25

Or pass individual files explicitly:
  python scripts/overall_fl_evaluator.py eval \\
    --fl      baselines/results/gpt-5-mini_llm/fault_localization.json \\
    --out     baselines/results/gpt-5-mini_llm/fl_full_report.json

Then print paper tables:
  python scripts/overall_fl_evaluator.py tables \\
    --reports baselines/results/gpt-5-mini_llm/fl_full_report.json \\
              baselines/results/gpt-5-mini_bm25/fl_full_report.json \\
    --latex   paper/rq_tables.tex
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from dataset_source import get_ci_repair_dataset_path


# ─────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────
def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def normalize_path(p: str) -> str:
    return p.strip().replace("\\", "/").lstrip("/")


def safe_str(v: Any) -> str:
    return "" if v is None else str(v)


# ─────────────────────────────────────────────
# Diff → ground-truth parsing
# ─────────────────────────────────────────────
_DIFF_HDR = re.compile(r"^diff --git a/(.+?) b/(.+)$", re.MULTILINE)
_HUNK_RE  = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(diff_text: str) -> Dict[str, List[Tuple[int, int]]]:
    gt: Dict[str, List[Tuple[int, int]]] = {}
    cur: Optional[str] = None
    for line in (diff_text or "").splitlines():
        m = _DIFF_HDR.match(line)
        if m:
            cur = normalize_path(m.group(2))
            gt.setdefault(cur, [])
            continue
        h = _HUNK_RE.match(line)
        if h and cur:
            start = int(h.group(3))
            count = int(h.group(4)) if h.group(4) else 1
            end   = start if count == 0 else start + count - 1
            gt[cur].append((start, end))
    return gt


# ─────────────────────────────────────────────
# Dataset building from parquet
# ─────────────────────────────────────────────
def build_dataset(parquet: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet)
    seen: Set[Tuple[str, str]] = set()
    records = []
    for _, row in df.iterrows():
        iid   = safe_str(row.get("id"))
        sha   = safe_str(row.get("sha_fail"))
        key   = (iid, sha)
        if key in seen:
            continue
        seen.add(key)
        gt = parse_diff(safe_str(row.get("diff", "")))
        records.append({"id": iid, "sha_fail": sha,
                        "gt_files": set(gt.keys()),
                        "gt_ranges": gt})
    records.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else x["id"])
    return records


# ─────────────────────────────────────────────
# Prediction extraction
# ─────────────────────────────────────────────
def build_pred_index(fl_records: List[Dict]) -> Dict[str, Dict]:
    """Index FL records by sha_fail only — the sole reliable join key."""
    idx: Dict[str, Dict] = {}
    for rec in fl_records:
        sha = safe_str(rec.get("sha_fail"))
        if sha:
            idx[sha] = rec
    return idx


def get_pred(sha: str, idx: Dict) -> Optional[Dict]:
    return idx.get(sha)


def pred_files_ranked(rec: Optional[Dict]) -> List[str]:
    """Ordered list of predicted files (rank order = list order)."""
    if not rec:
        return []
    return [
        normalize_path(item.get("file_path", ""))
        for item in (rec.get("fault_localization_data") or [])
        if item.get("file_path")
    ]


def pred_files_with_ranges(rec: Optional[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """Predicted files → line ranges (for location-level F1)."""
    if not rec:
        return {}
    out: Dict[str, List[Tuple[int, int]]] = {}
    for item in (rec.get("fault_localization_data") or []):
        fp = normalize_path(item.get("file_path", ""))
        if not fp:
            continue
        ranges = []
        for fault in (item.get("faults") or []):
            lr = fault.get("line_range")
            if isinstance(lr, list) and len(lr) == 2:
                try:
                    s, e = int(lr[0]), int(lr[1])
                    if s > e:
                        s, e = e, s
                    ranges.append((s, e))
                except Exception:
                    pass
        out[fp] = ranges
    return out


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def hit_at_k(ranked: List[str], gt: Set[str], k: int) -> bool:
    return any(f in gt for f in ranked[:k])


def average_precision(ranked: List[str], gt: Set[str]) -> float:
    """AP for a single task: sum of precision@k at each hit position / |GT|."""
    if not gt:
        return 0.0
    hits, ap = 0, 0.0
    for k, f in enumerate(ranked, 1):
        if f in gt:
            hits += 1
            ap += hits / k
    return ap / len(gt)


def ranges_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_one(gt_files: Set[str], gt_ranges: Dict,
                 ranked: List[str],
                 pred_ranges: Dict) -> Dict[str, Any]:
    pred_files = set(ranked)
    matched    = gt_files & pred_files
    fp_files   = pred_files - gt_files

    file_rec       = len(matched) / len(gt_files)   if gt_files   else 0.0
    file_precision = len(matched) / len(pred_files)  if pred_files else 0.0

    correct_loc = [
        f for f in matched
        if any(ranges_overlap(p, g)
               for p in pred_ranges.get(f, [])
               for g in gt_ranges.get(f, []))
    ]
    fl_rec       = len(correct_loc) / len(gt_files)   if gt_files   else 0.0
    fl_precision = len(correct_loc) / len(pred_files)  if pred_files else 0.0

    # Accuracy: task is correctly localized if ≥1 GT file is predicted (= any hit)
    accuracy = bool(matched)

    return {
        # Top-K
        "hit_at_1":    hit_at_k(ranked, gt_files, 1),
        "hit_at_3":    hit_at_k(ranked, gt_files, 3),
        "hit_at_5":    hit_at_k(ranked, gt_files, 5),
        "exact_match": pred_files == gt_files and bool(gt_files),
        # MAP
        "ap":          round(average_precision(ranked, gt_files), 4),
        # File-level P / R / F1 / Accuracy
        "accuracy":        float(accuracy),
        "file_precision":  round(file_precision, 4),
        "file_recall":     round(file_rec,        4),
        "file_f1":         round(f1(file_precision, file_rec), 4),
        # Location-level P / R / F1
        "fl_precision":    round(fl_precision, 4),
        "fl_recall":       round(fl_rec,       4),
        "fl_f1":           round(f1(fl_precision, fl_rec), 4),
        # Counts
        "num_gt_files":          len(gt_files),
        "num_pred_files":        len(pred_files),
        "num_matched_files":     len(matched),
        "num_false_pos_files":   len(fp_files),
        "num_correct_localized": len(correct_loc),
        "has_prediction":        bool(pred_files),
    }


# ─────────────────────────────────────────────
# Patch statistics
# ─────────────────────────────────────────────
def load_patch_stats(patches_path: Optional[Path],
                     total_tasks: int) -> Dict[str, Any]:
    if patches_path is None or not patches_path.exists():
        return {"generated": 0, "generated_pct": 0.0, "total_tasks": total_tasks}
    patches = load_json(patches_path)
    n = len(patches)
    return {
        "generated":     n,
        "generated_pct": round(n / total_tasks * 100, 1) if total_tasks else 0.0,
        "total_tasks":   total_tasks,
    }


# ─────────────────────────────────────────────
# Token / cost statistics from token_report.json
# ─────────────────────────────────────────────
STAGE_ORDER  = ["CILogAnalyzerLLM", "FaultLocalization", "PatchGeneration"]
STAGE_LABELS = {
    "CILogAnalyzerLLM": "Log Analysis",
    "FaultLocalization": "Fault Local.",
    "PatchGeneration":   "Patch Gen.",
}


def load_token_stats(token_report_path: Optional[Path]) -> Optional[Dict]:
    if token_report_path is None or not token_report_path.exists():
        return None
    r = load_json(token_report_path)
    agents  = r["llm"]["per_agent"]
    overall = r["llm"]["overall"]
    stages  = {}
    for agent, label in STAGE_LABELS.items():
        if agent not in agents:
            continue
        v = agents[agent]
        stages[agent] = {
            "label":        label,
            "input_tokens": v["input_tokens"],
            "output_tokens": v["output_tokens"],
            "total_tokens": v["total_tokens"],
            "cost_usd":     v["cost_usd"],
            "api_calls":    v["api_calls"],
        }
    return {"per_stage": stages, "overall": overall}


# ─────────────────────────────────────────────
# Full evaluation (one strategy)
# ─────────────────────────────────────────────
def run_eval(fl_path: Path, parquet: Path,
             patches_path: Optional[Path],
             token_report_path: Optional[Path],
             label: str, strategy: str) -> Dict[str, Any]:

    fl_records = load_json(fl_path)
    pred_idx   = build_pred_index(fl_records)
    dataset    = build_dataset(parquet)
    n_total    = len(dataset)

    per_issue = []
    agg = defaultdict(float)
    counters = defaultdict(int)

    for rec in dataset:
        iid, sha = rec["id"], rec["sha_fail"]
        pred_rec    = get_pred(sha, pred_idx)
        ranked      = pred_files_ranked(pred_rec)
        pred_ranges = pred_files_with_ranges(pred_rec)
        m           = evaluate_one(rec["gt_files"], rec["gt_ranges"],
                                   ranked, pred_ranges)

        per_issue.append({"id": iid, "sha_fail": sha, **m})

        for key in ("hit_at_1", "hit_at_3", "hit_at_5", "exact_match",
                    "ap", "accuracy",
                    "file_precision", "file_recall", "file_f1",
                    "fl_precision", "fl_recall", "fl_f1"):
            agg[key] += float(m[key])

        if m["has_prediction"]:
            counters["with_prediction"] += 1

    aggregate = {
        # Top-K accuracy (hit rate over ALL tasks)
        "top_1_rate":       round(agg["hit_at_1"]  / n_total * 100, 2),
        "top_3_rate":       round(agg["hit_at_3"]  / n_total * 100, 2),
        "top_5_rate":       round(agg["hit_at_5"]  / n_total * 100, 2),
        "exact_match_rate": round(agg["exact_match"] / n_total * 100, 2),
        # MAP
        "map":              round(agg["ap"] / n_total * 100, 2),
        # File-level accuracy, precision, recall, F1
        "accuracy":         round(agg["accuracy"]       / n_total * 100, 2),
        "file_precision":   round(agg["file_precision"] / n_total * 100, 2),
        "file_recall":      round(agg["file_recall"]    / n_total * 100, 2),
        "file_f1":          round(agg["file_f1"]        / n_total * 100, 2),
        # Location-level precision, recall, F1
        "fl_precision":     round(agg["fl_precision"]   / n_total * 100, 2),
        "fl_recall":        round(agg["fl_recall"]      / n_total * 100, 2),
        "fl_f1":            round(agg["fl_f1"]          / n_total * 100, 2),
        # Coverage
        "total_tasks":      n_total,
        "with_prediction":  counters["with_prediction"],
    }

    patch_stats = load_patch_stats(patches_path, n_total)
    token_stats = load_token_stats(token_report_path)

    report = {
        "label":       label,
        "strategy":    strategy,
        "aggregate":   aggregate,
        "patch_stats": patch_stats,
        "token_stats": token_stats,
        "per_issue":   per_issue,
    }

    # Console summary
    _print_eval_summary(report)
    return report


def _print_eval_summary(r: Dict) -> None:
    a  = r["aggregate"]
    ps = r["patch_stats"]
    label = f"{r['label']} / {r['strategy']}"
    w = 66
    print()
    print("=" * w)
    print(f"  {label}")
    print("=" * w)
    print(f"  Tasks evaluated  : {a['total_tasks']}")
    print(f"  With prediction  : {a['with_prediction']}")
    print()
    print(f"  ── Top-K Accuracy (Fault Localization) ──")
    print(f"  Top-1            : {a['top_1_rate']:6.2f}%")
    print(f"  Top-3            : {a['top_3_rate']:6.2f}%")
    print(f"  Top-5            : {a['top_5_rate']:6.2f}%")
    print(f"  Exact Match      : {a['exact_match_rate']:6.2f}%")
    print()
    print(f"  ── Ranking Quality ──")
    print(f"  MAP              : {a['map']:6.2f}%")
    print()
    print(f"  ── File-level Metrics ──")
    print(f"  Accuracy         : {a['accuracy']:6.2f}%")
    print(f"  Precision        : {a['file_precision']:6.2f}%")
    print(f"  Recall           : {a['file_recall']:6.2f}%")
    print(f"  F1               : {a['file_f1']:6.2f}%")
    print()
    print(f"  ── Location-level Metrics ──")
    print(f"  Precision        : {a['fl_precision']:6.2f}%")
    print(f"  Recall           : {a['fl_recall']:6.2f}%")
    print(f"  F1               : {a['fl_f1']:6.2f}%")
    print()
    print(f"  Patches generated: {ps['generated']} / {ps['total_tasks']}"
          f"  ({ps['generated_pct']:.1f}%)")
    print("=" * w)


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────
def _pct(v: float, dec: int = 2) -> str:
    return f"{v:.{dec}f}%"


def _m(v: float) -> str:
    return f"{v/1e6:.2f}M"


def _cost(v: float) -> str:
    return f"${v:.2f}"


# ─────────────────────────────────────────────
# RQ1 table  —  model comparison
# (one row per model/strategy combination)
# ─────────────────────────────────────────────
def print_rq1_table(reports: List[Dict]) -> None:
    w = 130
    print()
    print("─" * w)
    print("  RQ1 — Fault Localization Results")
    print("─" * w)

    # Table 1: Top-K + MAP
    header1 = ["Model", "Strategy", "Top-1", "Top-3", "Top-5", "MAP", "Patches (gen/total)", "Pass@1"]
    row_fmt1 = "  {:<22} {:<10} {:>8} {:>8} {:>8} {:>8} {:>20} {:>8}"
    print()
    print("  [Table 1: Top-K Accuracy & MAP]")
    print(row_fmt1.format(*header1))
    print("  " + "─" * (w - 2))
    for r in reports:
        a  = r["aggregate"]
        ps = r["patch_stats"]
        patch_col = f"{ps['generated']}/{ps['total_tasks']} ({ps['generated_pct']:.1f}%)"
        print(row_fmt1.format(
            r["label"], r["strategy"],
            _pct(a["top_1_rate"]),
            _pct(a["top_3_rate"]),
            _pct(a["top_5_rate"]),
            _pct(a.get("map", 0.0)),
            patch_col,
            "--",
        ))

    # Table 2: Precision / Accuracy / Recall / F1
    print()
    print("  [Table 2: Precision / Accuracy / Recall / F1 (file-level)]")
    header2 = ["Model", "Strategy", "Accuracy", "Precision", "Recall", "F1"]
    row_fmt2 = "  {:<22} {:<10} {:>10} {:>10} {:>10} {:>10}"
    print(row_fmt2.format(*header2))
    print("  " + "─" * (w - 2))
    for r in reports:
        a = r["aggregate"]
        print(row_fmt2.format(
            r["label"], r["strategy"],
            _pct(a.get("accuracy", 0.0)),
            _pct(a.get("file_precision", 0.0)),
            _pct(a.get("file_recall", 0.0)),
            _pct(a.get("file_f1", 0.0)),
        ))

    print()
    print("─" * w)
    print("  Note: Pass@1 requires CI test execution results (not yet available).")


# ─────────────────────────────────────────────
# RQ2 table  —  strategy comparison
# (relative drop vs. first report = LLM baseline)
# ─────────────────────────────────────────────
def print_rq2_table(reports: List[Dict]) -> None:
    if not reports:
        return
    base = reports[0]["aggregate"]   # LLM baseline

    def rel_drop(baseline: float, current: float) -> str:
        if baseline == 0:
            return "--"
        d = (baseline - current) / baseline * 100
        sign = "+" if d < 0 else "-"
        return f"{sign}{abs(d):.1f}%"

    w = 130
    print()
    print("─" * w)
    print("  RQ2 — Log Analysis Strategy Comparison")
    print("  (Relative drop vs. first listed strategy)")
    print("─" * w)
    row_fmt = "  {:<22} {:<10} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>8} {:>14}"
    header  = ["Model", "Strategy", "Top-1", "Top-3", "Top-5", "MAP",
               "Accuracy", "Precision", "Recall", "F1", "Rel.Drop(Top-1)"]
    print(row_fmt.format(*header))
    print("  " + "─" * (w - 2))

    for i, r in enumerate(reports):
        a = r["aggregate"]
        drop = "--" if i == 0 else rel_drop(base["top_1_rate"], a["top_1_rate"])
        print(row_fmt.format(
            r["label"],
            r["strategy"],
            _pct(a["top_1_rate"]),
            _pct(a["top_3_rate"]),
            _pct(a["top_5_rate"]),
            _pct(a.get("map", 0.0)),
            _pct(a.get("accuracy", 0.0)),
            _pct(a.get("file_precision", 0.0)),
            _pct(a.get("file_recall", 0.0)),
            _pct(a.get("file_f1", 0.0)),
            drop,
        ))

    print("─" * w)


# ─────────────────────────────────────────────
# RQ2 efficiency table  —  token / cost
# ─────────────────────────────────────────────
def print_efficiency_table(reports: List[Dict]) -> None:
    print()
    print("─" * 110)
    print("  RQ2 Efficiency — Token Usage & Cost per Stage")
    print("─" * 110)
    row_fmt = "  {:<10} {:<16} {:>10} {:>10} {:>10} {:>10} {:>10}"
    header  = ["Strategy", "Stage", "In. (M)", "Out. (M)",
               "Tot. (M)", "Cost ($)", "API Calls"]
    print(row_fmt.format(*header))
    print("  " + "─" * 108)

    for r in reports:
        ts = r.get("token_stats")
        if ts is None:
            print(row_fmt.format(r["strategy"], "(no token data)",
                                 "--", "--", "--", "--", "--"))
            continue

        first = True
        for agent in STAGE_ORDER:
            s = ts["per_stage"].get(agent)
            if s is None:
                continue
            strategy_col = r["strategy"] if first else ""
            first = False
            print(row_fmt.format(
                strategy_col,
                s["label"],
                _m(s["input_tokens"]),
                _m(s["output_tokens"]),
                _m(s["total_tokens"]),
                _cost(s["cost_usd"]),
                f"{s['api_calls']:,}",
            ))

        ov = ts["overall"]
        print(row_fmt.format(
            "", "  Total",
            _m(ov["input_tokens"]),
            _m(ov["output_tokens"]),
            _m(ov["total_tokens"]),
            _cost(ov["cost_usd"]),
            f"{ov['api_calls']:,}",
        ))
        print("  " + "─" * 108)


# ─────────────────────────────────────────────
# LaTeX generation
# ─────────────────────────────────────────────
def _latex_pct(v: float) -> str:
    return f"{v:.2f}\\%"


def generate_latex(reports: List[Dict]) -> str:
    if not reports:
        return ""

    base = reports[0]["aggregate"]

    def rel_drop(b: float, c: float) -> str:
        if b == 0:
            return "--"
        d = (b - c) / b * 100
        sign = "+" if d < 0 else "-"
        return f"{sign}{abs(d):.1f}\\%"

    # ── RQ1 / RQ2 combined table ──────────────────────────────────────────
    rq_rows = []
    for i, r in enumerate(reports):
        a  = r["aggregate"]
        ps = r["patch_stats"]
        drop = "--" if i == 0 else rel_drop(base["top_1_rate"], a["top_1_rate"])
        patch_col = f"{ps['generated']}/{ps['total_tasks']}"
        rq_rows.append(
            f"  {r['label']:<20} & {r['strategy']:<6}"
            f" & {_latex_pct(a['top_1_rate'])}"
            f" & {_latex_pct(a['top_3_rate'])}"
            f" & {_latex_pct(a['fl_f1'])}"
            f" & {patch_col}"
            f" & --"
            f" & {drop} \\\\"
        )

    rq_latex = r"""\begin{table}[t]
\centering
\caption{Fault localization Top-$k$ accuracy, FL F1, and patch statistics.
  Top-$k$ is the \% of tasks where a ground-truth file appears in the top-$k$
  predictions. FL F1 is the location-level F1 score.
  Pass@1 requires CI test-run results (not yet available).
  Rel.\ drop is relative to the first strategy (LLM baseline).}
\label{tab:rq_fl}
\small
\begin{tabular}{l l rrr r r r}
\toprule
\textbf{Model} & \textbf{Strategy}
  & \textbf{Top-1} & \textbf{Top-3} & \textbf{FL F1}
  & \textbf{Patches} & \textbf{Pass@1} & \textbf{Rel.Drop} \\
\midrule
""" + "\n".join(rq_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # ── Efficiency table ──────────────────────────────────────────────────
    eff_rows = []
    for r in reports:
        ts = r.get("token_stats")
        if ts is None:
            eff_rows.append(
                f"  {r['strategy']} & (no token data)"
                " & -- & -- & -- & -- & -- \\\\"
            )
            continue
        n = len(STAGE_ORDER)
        for j, agent in enumerate(STAGE_ORDER):
            s = ts["per_stage"].get(agent)
            if s is None:
                continue
            strategy_col = (f"\\multirow{{{n}}}{{*}}{{{r['strategy']}}}"
                            if j == 0 else "")
            eff_rows.append(
                f"  {strategy_col:<38} & {s['label']:<16}"
                f" & {_m(s['input_tokens'])}"
                f" & {_m(s['output_tokens'])}"
                f" & {_m(s['total_tokens'])}"
                f" & \\${s['cost_usd']:.2f}"
                f" & {s['api_calls']:,} \\\\"
            )
        ov = ts["overall"]
        eff_rows.append("  \\midrule")
        eff_rows.append(
            f"  {'':38} & {'Total':<16}"
            f" & {_m(ov['input_tokens'])}"
            f" & {_m(ov['output_tokens'])}"
            f" & {_m(ov['total_tokens'])}"
            f" & \\${ov['cost_usd']:.2f}"
            f" & {ov['api_calls']:,} \\\\"
        )

    eff_latex = r"""\begin{table}[t]
\centering
\caption{Token consumption and API cost per stage.
  In., Out., Tot.\ = input, output, total tokens ($\times10^{6}$).
  Cost = USD.}
\label{tab:rq2_efficiency}
\small
\begin{tabular}{l l rrrr r}
\toprule
\textbf{Strategy} & \textbf{Stage}
  & \textbf{In.} & \textbf{Out.} & \textbf{Tot.} & \textbf{Cost} & \textbf{API} \\
\midrule
""" + "\n".join(eff_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    return rq_latex + "\n" + eff_latex


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def cmd_eval(args: argparse.Namespace) -> None:
    parquet_path = Path(args.parquet) if args.parquet else Path(
        get_ci_repair_dataset_path(Path(__file__).resolve().parents[1])
    )

    # --results-dir: auto-discover all files from the strategy directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_dir():
            raise SystemExit(f"ERROR: --results-dir not found: {results_dir}")
        fl_path      = results_dir / "fault_localization.json"
        patches_path = results_dir / "generated_patches.json"
        tok_path     = results_dir / "token_report.json"
        out_path     = results_dir / "fl_full_report.json"
        if not fl_path.exists():
            raise SystemExit(f"ERROR: fault_localization.json not found in {results_dir}")
        label    = args.label    if args.label    != "Model" else results_dir.name
        strategy = args.strategy if args.strategy != "LLM"   else (
            "BM25" if "bm25" in results_dir.name.lower() else "LLM"
        )
        patches_path = patches_path if patches_path.exists() else None
        tok_path     = tok_path     if tok_path.exists()     else None
    else:
        if not args.fl:
            raise SystemExit("ERROR: provide --results-dir DIR or --fl PATH")
        fl_path      = Path(args.fl)
        patches_path = Path(args.patches)      if args.patches      else None
        tok_path     = Path(args.token_report) if args.token_report else None
        out_path     = Path(args.out)          if args.out          else fl_path.parent / "fl_full_report.json"
        label        = args.label
        strategy     = args.strategy

    report = run_eval(
        fl_path           = fl_path,
        parquet           = parquet_path,
        patches_path      = patches_path,
        token_report_path = tok_path,
        label             = label,
        strategy          = strategy,
    )

    save_json(out_path, report)
    print(f"\n  Full report saved → {out_path}")

    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(report["per_issue"]).to_csv(csv_path, index=False)
    print(f"  Per-issue CSV   → {csv_path}")


def cmd_tables(args: argparse.Namespace) -> None:
    reports = [load_json(Path(p)) for p in args.reports]

    print_rq1_table(reports)
    print_rq2_table(reports)
    print_efficiency_table(reports)

    latex = generate_latex(reports)
    print()
    print("=" * 80)
    print("  LaTeX")
    print("=" * 80)
    print(latex)

    if args.latex:
        out = Path(args.latex)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex, encoding="utf-8")
        print(f"LaTeX written → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FL evaluator for CI-REPAIR-BENCH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── eval ──────────────────────────────────────
    ev = sub.add_parser("eval", help="Evaluate one strategy")
    ev.add_argument("--results-dir", default=None,
                    help="Strategy results directory (auto-discovers all files)")
    ev.add_argument("--fl",           default=None,
                    help="fault_localization.json (ignored when --results-dir is set)")
    ev.add_argument("--parquet",      default=None,
                    help="Optional local parquet override. Defaults to the Hugging Face dataset.")
    ev.add_argument("--patches",      default=None,
                    help="generated_patches.json (optional)")
    ev.add_argument("--token-report", default=None,
                    help="token_report.json (optional, for efficiency table)")
    ev.add_argument("--label",        default="Model",
                    help="Human-readable model name, e.g. 'GPT-5-mini'")
    ev.add_argument("--strategy",     default="LLM",
                    help="Strategy tag, e.g. LLM or BM25")
    ev.add_argument("--out",          default=None,
                    help="Output path for fl_full_report.json (default: <results-dir>/fl_full_report.json)")
    ev.set_defaults(func=cmd_eval)

    # ── tables ────────────────────────────────────
    tb = sub.add_parser("tables", help="Print paper tables from saved reports")
    tb.add_argument("--reports", nargs="+", required=True,
                    help="fl_full_report.json files (one per strategy, LLM first)")
    tb.add_argument("--latex", default=None,
                    help="Optional path to write LaTeX snippet")
    tb.set_defaults(func=cmd_tables)

    # ── legacy single-strategy mode (backward compat) ──
    # Detect old-style args and map to eval subcommand
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in ("eval", "tables"):
        # Legacy mode: inject "eval" as subcommand
        sys.argv.insert(1, "eval")
        # Map old arg names
        argv = sys.argv[1:]
        argv = [a.replace("--fault-localization", "--fl")
                 .replace("--output-json", "--out")
                 for a in argv]
        # Remove --output-csv (now auto-derived from --out)
        filtered = []
        skip_next = False
        for a in argv:
            if skip_next:
                skip_next = False
                continue
            if a == "--output-csv":
                skip_next = True
                continue
            filtered.append(a)
        sys.argv = [sys.argv[0]] + filtered

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
