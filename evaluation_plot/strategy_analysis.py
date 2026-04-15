#!/usr/bin/env python3
"""Compute file/location accuracy and patch-vs-success tables per model strategy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_run_name(run_name: str) -> tuple[str, str]:
    if "_" not in run_name:
        return run_name, "unknown"
    model, strategy = run_name.rsplit("_", 1)
    return model, strategy


def normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_unified_diff_new_hunks(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Return changed line intervals on the fail/new side of each file diff."""
    file_hunks: dict[str, list[tuple[int, int]]] = defaultdict(list)
    current_file: str | None = None

    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if not current_file or not line.startswith("@@ "):
            continue

        try:
            header = line.split("@@")[1].strip()
            new_range = header.split(" ")[1]  # "+1210,7"
            new_range = new_range.lstrip("+")
            if "," in new_range:
                start_text, count_text = new_range.split(",", 1)
                start = int(start_text)
                count = int(count_text)
            else:
                start = int(new_range)
                count = 1
            end = start if count <= 1 else start + count - 1
            file_hunks[current_file].append((start, end))
        except (IndexError, ValueError):
            continue

    return dict(file_hunks)


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


def load_oracle_changed_files(changed_files_dir: Path) -> dict[str, dict[str, list[tuple[int, int]]]]:
    oracle: dict[str, dict[str, list[tuple[int, int]]]] = {}

    if not changed_files_dir.exists():
        return oracle

    for path in sorted(changed_files_dir.glob("*.json")):
        payload = load_json(path)
        sha_fail = normalize_id(payload.get("sha_fail"))
        if not sha_fail:
            continue

        file_to_hunks: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for changed in payload.get("changed_files", []):
            file_path = changed.get("file_path")
            diff_text = changed.get("diff", "")
            parsed = parse_unified_diff_new_hunks(diff_text)

            if file_path and file_path in parsed:
                file_to_hunks[file_path].extend(parsed[file_path])
            elif file_path:
                file_to_hunks[file_path]

        oracle[sha_fail] = dict(file_to_hunks)

    return oracle


def load_success_manifest(manifest_path: Path | None) -> dict[str, set[str]]:
    """Manifest format: {run_name: path_to_json_or_jsonl}."""
    run_to_ids: dict[str, set[str]] = {}

    if manifest_path is None or not manifest_path.exists():
        return run_to_ids

    manifest = load_json(manifest_path)
    for run_name, rel_or_abs_path in manifest.items():
        result_path = Path(rel_or_abs_path)
        if not result_path.is_absolute():
            result_path = manifest_path.parent / result_path
        if not result_path.exists():
            continue

        ids: set[str] = set()
        if result_path.suffix == ".jsonl":
            rows = load_jsonl(result_path)
        else:
            payload = load_json(result_path)
            rows = payload if isinstance(payload, list) else [payload]

        for row in rows:
            for key in ("id", "sha_fail", "sha_original"):
                value = normalize_id(row.get(key))
                if value:
                    ids.add(value)
        run_to_ids[run_name] = ids

    return run_to_ids


def metric_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round((numerator / denominator) * 100.0, 2)


def format_pct(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def build_file_location_rows(
    results_root: Path,
    oracle_by_sha: dict[str, dict[str, list[tuple[int, int]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        fl_path = run_dir / "fault_localization.json"
        if not fl_path.exists():
            continue

        model, strategy = split_run_name(run_dir.name)
        payload = load_json(fl_path)
        total = 0
        evaluable = 0
        file_hits = 0
        location_hits = 0

        for item in payload:
            total += 1
            sha_fail = normalize_id(item.get("sha_fail"))
            if not sha_fail or sha_fail not in oracle_by_sha:
                continue

            oracle = oracle_by_sha[sha_fail]
            if not oracle:
                continue

            evaluable += 1
            oracle_files = set(oracle.keys())
            predictions = item.get("fault_localization_data", []) or []
            predicted_files = {
                pred.get("file_path")
                for pred in predictions
                if pred.get("file_path")
            }

            file_hit = bool(predicted_files & oracle_files)
            if file_hit:
                file_hits += 1

            location_hit = False
            for pred in predictions:
                pred_file = pred.get("file_path")
                if pred_file not in oracle:
                    continue
                oracle_hunks = oracle[pred_file]
                for fault in pred.get("faults", []) or []:
                    line_range = fault.get("line_range") or []
                    if len(line_range) != 2:
                        continue
                    start, end = int(line_range[0]), int(line_range[1])
                    if any(ranges_overlap(start, end, h_start, h_end) for h_start, h_end in oracle_hunks):
                        location_hit = True
                        break
                if location_hit:
                    break

            if location_hit:
                location_hits += 1

        rows.append(
            {
                "run_name": run_dir.name,
                "model": model,
                "strategy": strategy,
                "total_instances": total,
                "evaluable_instances": evaluable,
                "file_hits": file_hits,
                "location_hits": location_hits,
                "file_identification_rate": metric_pct(file_hits, evaluable),
                "location_rate": metric_pct(location_hits, evaluable),
            }
        )

    return rows


def build_patch_success_rows(
    results_root: Path,
    success_ids_by_run: dict[str, set[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        patch_path = run_dir / "generated_patches.json"
        if not patch_path.exists():
            continue

        model, strategy = split_run_name(run_dir.name)
        payload = load_json(patch_path)
        patch_ids: set[str] = set()

        for item in payload:
            for key in ("id", "sha_fail"):
                value = normalize_id(item.get(key))
                if value:
                    patch_ids.add(value)

        success_ids = success_ids_by_run.get(run_dir.name, set())
        ci_successes = len(patch_ids & success_ids) if success_ids else None

        rows.append(
            {
                "run_name": run_dir.name,
                "model": model,
                "strategy": strategy,
                "generated_patches": len(patch_ids),
                "ci_repair_successes": ci_successes,
                "patch_to_success_rate": (
                    metric_pct(ci_successes, len(patch_ids))
                    if ci_successes is not None
                    else None
                ),
            }
        )

    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    sep = ["---"] * len(columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="baselines/results",
        help="Directory containing per-run folders like gpt-5-mini_llm.",
    )
    parser.add_argument(
        "--changed-files-dir",
        default="baselines/changed_files",
        help="Directory containing oracle changed-file JSON per sha_fail.",
    )
    parser.add_argument(
        "--success-manifest",
        default=None,
        help="Optional JSON mapping run_name -> success json/jsonl path.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_plot/strategy_tables",
        help="Directory for generated table files.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    changed_files_dir = Path(args.changed_files_dir)
    output_dir = Path(args.output_dir)
    success_manifest = Path(args.success_manifest) if args.success_manifest else None

    oracle_by_sha = load_oracle_changed_files(changed_files_dir)
    success_ids_by_run = load_success_manifest(success_manifest)

    file_rows = build_file_location_rows(results_root, oracle_by_sha)
    patch_rows = build_patch_success_rows(results_root, success_ids_by_run)

    for row in file_rows:
        row["file_identification_rate_str"] = format_pct(row["file_identification_rate"])
        row["location_rate_str"] = format_pct(row["location_rate"])
    for row in patch_rows:
        row["patch_to_success_rate_str"] = format_pct(row["patch_to_success_rate"])

    file_table = markdown_table(
        file_rows,
        [
            ("model", "Model"),
            ("strategy", "Strategy"),
            ("evaluable_instances", "Evaluated"),
            ("file_hits", "Correct Files"),
            ("file_identification_rate_str", "File ID Rate (%)"),
            ("location_hits", "Correct Locations"),
            ("location_rate_str", "Location Rate (%)"),
        ],
    )
    patch_table = markdown_table(
        patch_rows,
        [
            ("model", "Model"),
            ("strategy", "Strategy"),
            ("generated_patches", "Generated Patches"),
            ("ci_repair_successes", "CI Repair Successes"),
            ("patch_to_success_rate_str", "Patch->Success Rate (%)"),
        ],
    )

    write_json(output_dir / "file_location_metrics.json", file_rows)
    write_json(output_dir / "patch_success_metrics.json", patch_rows)
    (output_dir / "file_location_table.md").write_text(file_table, encoding="utf-8")
    (output_dir / "patch_success_table.md").write_text(patch_table, encoding="utf-8")

    print("File/Location table")
    print(file_table)
    print("Patch/Success table")
    print(patch_table)


if __name__ == "__main__":
    main()
