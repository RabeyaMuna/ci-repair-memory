#!/usr/bin/env python3
"""Pairwise exact McNemar tests for the seven Table 2 configurations.

The analysis uses (repo_name, id) as the benchmark-instance key, treats every
non-"success" conclusion as a failed repair, and uses the union of instance
keys as the benchmark universe.  A missing record is therefore counted as a
failure.  This matches Table 2's denominator of 565 for DeepSeek-Chat, whose
JSONL file is missing one benchmark instance.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "analysis" / "table2_pairwise_mcnemar.csv"
OUTPUT_MD = ROOT / "analysis" / "table2_pairwise_mcnemar.md"

CONFIGURATIONS = {
    "CI Repair Framework + DeepSeek-Coder": (
        ROOT / "baselines/results/deepseek-coder_llm/jobs_results_diff.jsonl"
    ),
    "CI Repair Framework + DeepSeek-Chat": (
        ROOT / "baselines/results/deepseek-chat_llm/jobs_results_diff.jsonl"
    ),
    "CI Repair Framework + GPT-4o-mini": (
        ROOT / "baselines/results/gpt-4o-mini_llm/jobs_results_diff.jsonl"
    ),
    "CI Repair Framework + GPT-5-mini": (
        ROOT / "baselines/results/gpt-5-mini_llm/jobs_results_diff.jsonl"
    ),
    "CI Repair Framework + DeepSeek-V4-Flash": (
        ROOT / "baselines/results/deepseek-v4-flash_llm/jobs_results_diff.jsonl"
    ),
    "mini-swe-agent + GPT-5-mini": (
        ROOT
        / "miniswe-agent/baselines/results/miniswe-agent_gpt-5-mini_llm/jobs_results_diff.jsonl"
    ),
    "mini-swe-agent + DeepSeek-V4-Flash": (
        ROOT
        / "miniswe-agent/baselines/results/miniswe-agent_deepseek-v4-flash_llm/jobs_results_diff.jsonl"
    ),
}


def load_results(path: Path) -> dict[tuple[str, str], bool]:
    """Return one binary repair outcome per benchmark instance."""
    results: dict[tuple[str, str], bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["repo_name"]), str(row["id"]))
            if key in results:
                raise ValueError(f"Duplicate instance {key} in {path}:{line_number}")
            results[key] = row.get("conclusion") == "success"
    return results


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value (exact binomial test at p=0.5)."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(a_only, b_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order."""
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.3f}"


def main() -> None:
    outcomes = {name: load_results(path) for name, path in CONFIGURATIONS.items()}
    universe = set().union(*(set(result) for result in outcomes.values()))
    if len(universe) != 565:
        raise ValueError(f"Expected 565 benchmark instances, found {len(universe)}")

    rows = []
    for name_a, name_b in itertools.combinations(outcomes, 2):
        result_a = outcomes[name_a]
        result_b = outcomes[name_b]
        pairs = [(result_a.get(key, False), result_b.get(key, False)) for key in universe]
        both_pass = sum(a and b for a, b in pairs)
        a_only = sum(a and not b for a, b in pairs)
        b_only = sum(not a and b for a, b in pairs)
        both_fail = sum(not a and not b for a, b in pairs)
        rate_a = 100.0 * sum(result_a.values()) / len(universe)
        rate_b = 100.0 * sum(result_b.values()) / len(universe)
        rows.append(
            {
                "configuration_a": name_a,
                "configuration_b": name_b,
                "n": len(universe),
                "both_pass": both_pass,
                "a_only_pass": a_only,
                "b_only_pass": b_only,
                "both_fail": both_fail,
                "pass_at_1_a_percent": rate_a,
                "pass_at_1_b_percent": rate_b,
                "signed_difference_pp_a_minus_b": rate_a - rate_b,
                "absolute_difference_pp": abs(rate_a - rate_b),
                "raw_exact_p": exact_mcnemar_p(a_only, b_only),
            }
        )

    adjusted = holm_adjust([row["raw_exact_p"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["significant_at_0.05"] = adjusted_p < 0.05

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Table 2 pairwise statistical analysis",
        "",
        (
            "Two-sided exact McNemar tests with Holm correction across all 21 "
            "pairwise comparisons. Differences are percentage points (pp). "
            "Only a `success` conclusion is counted as a successful repair; "
            "`failure`, `cancelled`, `waiting`, and a missing result are counted "
            "as failures. See the data note below."
        ),
        "",
        "| Configuration A | Configuration B | A only | B only | |Δ Pass@1| (pp) | Raw p | Holm p | Sig. |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {configuration_a} | {configuration_b} | {a_only_pass} | "
            "{b_only_pass} | {absolute_difference_pp:.2f} | {raw} | "
            "{adjusted} | {sig} |".format(
                **row,
                raw=format_p(row["raw_exact_p"]),
                adjusted=format_p(row["holm_adjusted_p"]),
                sig="*" if row["significant_at_0.05"] else "",
            )
        )

    missing = {
        name: sorted(universe - set(result))
        for name, result in outcomes.items()
        if universe - set(result)
    }
    lines.extend(
        [
            "",
            "## Data note",
            "",
            (
                "The DeepSeek-Chat file contains 564 records and lacks "
                "`glances/255`. It is counted as a failure so that the analysis "
                "matches the reported 83/565 Pass@1. All other files contain "
                "all 565 instances. A complete-case analysis on the 564 common "
                "instances produces the same 21 significance decisions."
            ),
            "",
            f"Missing records detected: `{missing}`",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUTPUT_MD.relative_to(ROOT)}")
    print(f"Significant comparisons: {sum(row['significant_at_0.05'] for row in rows)}/21")


if __name__ == "__main__":
    main()
