#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_SCRIPTS = PROJECT_ROOT / "baselines" / "scripts"
DEFAULT_OUTPUT = PROJECT_ROOT / "baselines" / "results" / "trs"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "lca_dataset.parquet"
DEFAULT_PATCHES = PROJECT_ROOT / "generated_patches_list" / "generated_patches_success_only.json"
DEFAULT_CACHED_LOG_DETAILS = PROJECT_ROOT / "baselines" / "results" / "MiniMax-M2.5" / "log_details.json"
DEFAULT_EXISTING_MEMORY_BANK = PROJECT_ROOT / "baselines" / "results" / "trs"


def _run(cmd: list[str], *, cwd: Path, capture: bool = False) -> str:
    print(f"[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(PROJECT_ROOT), str(PROJECT_ROOT / "baselines")]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
        env=env,
    )
    if capture:
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        return proc.stdout
    return ""


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _extract_last_json_object(text: str) -> Dict[str, Any]:
    matches = list(re.finditer(r"\{", text or ""))
    for match in reversed(matches):
        candidate = text[match.start():].strip()
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return {}


def _safe_model_dir_key(model_key: str) -> str:
    return str(model_key or "").strip().replace("/", "__")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _memory_issue_shas(memory_bank_dir: Path) -> set[str]:
    rows = _load_json(memory_bank_dir / "failure_memory.json", [])
    return {
        str(row.get("sha_fail") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("sha_fail") or "")
    }


def _row_belongs_to_seed(row: Dict[str, Any], allowed_shas: set[str], allowed_ids: set[str]) -> bool:
    sha_fail = str(row.get("sha_fail") or "")
    if sha_fail and sha_fail in allowed_shas:
        return True
    issue_id = str(row.get("issue_id") or row.get("id") or "")
    if issue_id and issue_id in allowed_ids:
        return True
    task_ids = row.get("task_ids", [])
    if isinstance(task_ids, list) and any(str(task_id) in allowed_ids for task_id in task_ids):
        return True
    evidence_ids = row.get("evidence_issue_ids", [])
    if isinstance(evidence_ids, list) and any(str(task_id) in allowed_ids for task_id in evidence_ids):
        return True
    return False


def _copy_existing_memory_bank(
    source: Path,
    target: Path,
    *,
    allowed_shas: set[str],
    allowed_ids: set[str],
) -> Dict[str, Any]:
    copied = {}
    if not source.exists():
        return copied
    target.mkdir(parents=True, exist_ok=True)
    for name in ("failure_memory.json", "repo_memory.json", "cross_memory.json"):
        src = source / name
        if not src.exists():
            continue
        rows = _load_json(src, [])
        filtered = [
            row for row in rows
            if isinstance(row, dict) and _row_belongs_to_seed(row, allowed_shas, allowed_ids)
        ]
        dst = target / name
        _write_json(dst, filtered)
        copied[name] = {"path": str(dst), "source_rows": len(rows), "copied_rows": len(filtered)}
    return copied


def _hydrate_cached_seed_log_details(
    *,
    seed_rows: List[Dict[str, Any]],
    cached_log_details_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    cached_rows = _load_json(cached_log_details_path, [])
    cached_by_sha = {
        str(row.get("sha_fail") or ""): row
        for row in cached_rows
        if isinstance(row, dict) and str(row.get("sha_fail") or "")
    }


def _cached_success_shas(cached_log_details_path: Path) -> set[str]:
    rows = _load_json(cached_log_details_path, [])
    shas: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        sha_fail = str(row.get("sha_fail") or "")
        if not sha_fail:
            continue
        analysis = row.get("analysis") if isinstance(row.get("analysis"), dict) else row
        if isinstance(analysis, dict) and not analysis.get("error"):
            shas.add(sha_fail)
    return shas
    existing_rows = _load_json(output_path, [])
    by_sha = {
        str(row.get("sha_fail") or ""): row
        for row in existing_rows
        if isinstance(row, dict) and str(row.get("sha_fail") or "")
    }

    hydrated = 0
    missing = []
    for seed in seed_rows:
        sha_fail = str(seed.get("sha_fail") or "")
        if not sha_fail or sha_fail in by_sha:
            continue
        cached = cached_by_sha.get(sha_fail)
        if not cached:
            missing.append(sha_fail)
            continue
        if isinstance(cached.get("analysis"), dict):
            entry = dict(cached)
        else:
            entry = {
                "id": str(seed.get("id") or cached.get("id") or ""),
                "sha_fail": sha_fail,
                "repo_name": str(seed.get("repo_name") or cached.get("repo_name") or ""),
                "repo_owner": str(seed.get("repo_owner") or cached.get("repo_owner") or ""),
                "workflow_name": str(seed.get("workflow_name") or cached.get("workflow_name") or ""),
                "workflow_path": str(seed.get("workflow_path") or cached.get("workflow_path") or ""),
                "workflow": str(seed.get("workflow") or cached.get("workflow") or ""),
                "analysis": cached,
            }
        by_sha[sha_fail] = entry
        hydrated += 1

    output_rows = list(by_sha.values())
    _write_json(output_path, output_rows)
    return {
        "cached_log_details": str(cached_log_details_path),
        "output": str(output_path),
        "hydrated_from_cache": hydrated,
        "missing_from_cache": missing,
        "total_output_rows": len(output_rows),
    }


def _record_key(row: Dict[str, Any], level_name: str) -> str:
    if row.get("record_id"):
        return str(row["record_id"])
    if level_name == "failure_memory.json":
        return "|".join(
            str(row.get(key) or "")
            for key in ("sha_fail", "repo", "file", "failure_pattern", "error_type")
        )
    if level_name == "repo_memory.json":
        return "|".join(
            str(row.get(key) or "")
            for key in ("repo", "error_type", "failure_pattern")
        )
    return "|".join(
        str(row.get(key) or "")
        for key in ("error_type", "issue_type", "failure_pattern", "principle")
    )


def _merge_memory_bank(base_dir: Path, new_dir: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ("failure_memory.json", "repo_memory.json", "cross_memory.json"):
        base_rows = _load_json(base_dir / name, [])
        new_rows = _load_json(new_dir / name, [])
        merged: Dict[str, Dict[str, Any]] = {}
        for row in [*base_rows, *new_rows]:
            if not isinstance(row, dict):
                continue
            merged[_record_key(row, name)] = row
        rows = list(merged.values())
        _write_json(output_dir / name, rows)
        counts[name] = {
            "base": len(base_rows),
            "new": len(new_rows),
            "merged": len(rows),
        }
    _write_json(
        output_dir / "memory_bank_summary.json",
        {
            "source_base_dir": str(base_dir),
            "source_new_dir": str(new_dir),
            "output_dir": str(output_dir),
            "counts": counts,
        },
    )
    return counts


def _build_subset_memory_bank(args: argparse.Namespace, split_dir: Path, memory_bank_dir: Path, staging_dir: Path) -> Dict[str, Any]:
    seed_file = split_dir / "memory_seed_issues.json"
    seed_rows = _load_json(seed_file, [])
    seed_shas = {str(row.get("sha_fail") or "") for row in seed_rows if str(row.get("sha_fail") or "")}
    seed_ids = {str(row.get("id") or "") for row in seed_rows if str(row.get("id") or "")}
    existing_shas = _memory_issue_shas(args.existing_memory_bank)
    missing_seed_rows = [
        row for row in seed_rows
        if str(row.get("sha_fail") or "") not in existing_shas
    ]
    missing_seed_file = split_dir / "memory_seed_issues_missing_from_existing_memory.json"
    _write_json(missing_seed_file, missing_seed_rows)

    copied_existing = _copy_existing_memory_bank(
        args.existing_memory_bank,
        memory_bank_dir,
        allowed_shas=seed_shas,
        allowed_ids=seed_ids,
    )
    cached_shas = _cached_success_shas(args.cached_log_details)
    cache_missing_shas = {
        str(row.get("sha_fail") or "")
        for row in missing_seed_rows
        if str(row.get("sha_fail") or "") and str(row.get("sha_fail") or "") not in cached_shas
    }
    analyzer_seed_rows = [
        row for row in missing_seed_rows
        if str(row.get("sha_fail") or "") in cache_missing_shas
    ]
    analyzer_seed_file = split_dir / "memory_seed_issues_missing_from_cache.json"
    _write_json(analyzer_seed_file, analyzer_seed_rows)

    build_seed_rows = [
        row for row in missing_seed_rows
        if str(row.get("sha_fail") or "") in cached_shas
    ]
    build_seed_file = split_dir / "memory_seed_issues_ready_for_memory_build.json"
    _write_json(build_seed_file, build_seed_rows)

    manifest = {
        "existing_memory_bank": str(args.existing_memory_bank),
        "copied_existing": copied_existing,
        "total_seed_issues": len(seed_rows),
        "already_in_existing_memory": len(seed_rows) - len(missing_seed_rows),
        "missing_from_existing_memory": len(missing_seed_rows),
        "missing_seed_file": str(missing_seed_file),
        "cache_missing_seed_file": str(analyzer_seed_file),
        "ready_for_memory_build_seed_file": str(build_seed_file),
        "cache_only": bool(args.cached_log_only),
        "cached_log_details": str(args.cached_log_details),
        "found_in_cached_log_details": len(build_seed_rows),
    }

    if analyzer_seed_rows and not args.cached_log_only:
        _run(
            [
                args.python_bin,
                str(BASELINE_SCRIPTS / "analyze_memory_seed_issues.py"),
                "--seed-file", str(analyzer_seed_file),
                "--dataset", str(args.dataset),
                "--config", str(args.config),
                "--output-dir", str(memory_bank_dir),
                "--model-key", args.model_key,
                "--shared-log-details", str(args.cached_log_details),
                "--no-local-output",
            ],
            cwd=PROJECT_ROOT,
        )
        cached_shas = _cached_success_shas(args.cached_log_details)
        build_seed_rows = [
            row for row in missing_seed_rows
            if str(row.get("sha_fail") or "") in cached_shas
        ]
        _write_json(build_seed_file, build_seed_rows)
    elif analyzer_seed_rows:
        print(
            "[memory_subset] cache-only mode: skipping log analysis for cache-missing seed issue(s): "
            + ", ".join(str(row.get("id") or row.get("sha_fail") or "") for row in analyzer_seed_rows)
        )

    manifest["analyzer_seed_issues"] = len(analyzer_seed_rows)
    manifest["ready_for_memory_build_issues"] = len(build_seed_rows)

    if build_seed_rows:
        _run(
            [
                args.python_bin,
                str(BASELINE_SCRIPTS / "build_memory_bank_from_seed_analysis.py"),
                "--seed-file", str(build_seed_file),
                "--analysis-file", str(args.cached_log_details),
                "--output-dir", str(staging_dir),
                "--config", str(args.config),
                "--model-key", args.model_key,
            ],
            cwd=PROJECT_ROOT,
        )
        manifest["merge_counts"] = _merge_memory_bank(
            memory_bank_dir,
            staging_dir,
            memory_bank_dir,
        )
    else:
        manifest["merge_counts"] = {}
    manifest["final_memory_bank_dir"] = str(memory_bank_dir)
    _write_json(memory_bank_dir / "memory_subset_build_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run agno/camel/flower memory subset across ablations.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--patches", type=Path, default=DEFAULT_PATCHES)
    parser.add_argument("--python-bin", default=str(PROJECT_ROOT / "baselines" / ".venv" / "bin" / "python"))
    parser.add_argument("--repos", nargs="*", default=["agno", "camel", "flower"], help="Repos to include. Defaults to agno camel flower.")
    parser.add_argument("--extra-repos", nargs="*", default=[], help="Additional repos to include on top of --repos.")
    parser.add_argument("--eval-per-repo", type=int, default=5, help="Hold out this many success issues per repo for eval.")
    parser.add_argument("--eval-max-per-repo", type=int, default=None, help="Deprecated alias for --eval-per-repo.")
    parser.add_argument(
        "--memory-per-repo",
        type=int,
        default=None,
        help="Optional override: use only this many seed issues per repo instead of all non-eval issues.",
    )
    parser.add_argument("--model-key", default=os.getenv("MEMCI_LLM_MODEL", "MiniMax-M2.5"))
    parser.add_argument("--cached-log-details", type=Path, default=DEFAULT_CACHED_LOG_DETAILS)
    parser.add_argument("--existing-memory-bank", type=Path, default=DEFAULT_EXISTING_MEMORY_BANK)
    parser.add_argument(
        "--no-fill-eval-from-dataset",
        action="store_true",
        help="Deprecated compatibility flag. Eval rows are selected from dataset rows not used as memory.",
    )
    parser.add_argument("--cached-log-only", action="store_true", help="Do not run CI log analyzer for seed issues missing from --cached-log-details.")
    parser.add_argument("--skip-memory-build", action="store_true")
    parser.add_argument("--build-memory-only", action="store_true")
    parser.add_argument("--run-baseline", action="store_true")
    args = parser.parse_args()
    if args.eval_max_per_repo is not None:
        args.eval_per_repo = args.eval_max_per_repo

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir
    memory_bank_dir = args.output_dir
    staging_dir = args.output_dir / "_memory_build_staging"
    runs_dir = args.output_dir / "runs"
    trajectories_dir = args.output_dir / "trajectories"
    config_dir = args.output_dir / "configs"

    split_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "prepare_subset_split.py"),
        "--patches", str(args.patches),
        "--dataset", str(args.dataset),
        "--output-dir", str(split_dir),
        "--eval-per-repo", str(args.eval_per_repo),
        "--repos", *args.repos,
    ]
    split_cmd.extend(["--existing-memory-bank", str(args.existing_memory_bank)])
    if args.extra_repos:
        split_cmd.extend(["--extra-repos", *args.extra_repos])
    if args.memory_per_repo is not None:
        split_cmd.extend(["--memory-per-repo", str(args.memory_per_repo)])
    _run(split_cmd, cwd=PROJECT_ROOT)

    seed_file = split_dir / "memory_seed_issues.json"
    eval_file = split_dir / "eval_issues.json"

    memory_build_manifest = None
    if not args.skip_memory_build:
        memory_build_manifest = _build_subset_memory_bank(args, split_dir, memory_bank_dir, staging_dir)

    if args.build_memory_only:
        result = {
            "output_dir": str(args.output_dir),
            "seed_file": str(seed_file),
            "eval_file": str(eval_file),
            "memory_bank_dir": str(memory_bank_dir),
            "memory_build_manifest": memory_build_manifest,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    manifest = {
        "output_dir": str(args.output_dir),
        "seed_file": str(seed_file),
        "eval_file": str(eval_file),
        "memory_bank_dir": str(memory_bank_dir),
        "runs": {},
    }

    ablations = ["L1", "L1+L2", "L1+L2+L3"]
    if args.run_baseline:
        ablations = ["baseline", *ablations]

    base_config = _load_yaml(args.config)
    for ablation in ablations:
        run_root = runs_dir / ablation.replace("+", "_")
        retrieval_log = trajectories_dir / ablation.replace("+", "_") / "memory_retrieval_log.jsonl"
        if retrieval_log.exists():
            retrieval_log.unlink()

        config = dict(base_config)
        config["project_result_dir"] = str(run_root)
        config["memory_bank_dir"] = str(memory_bank_dir)
        config["memory_retrieval_log_path"] = str(retrieval_log)
        config["memory_enabled"] = ablation != "baseline"
        config["memory_writeback_enabled"] = False
        if ablation != "baseline":
            config["memory_ablation_levels"] = ablation

        config_path = config_dir / f"{ablation.replace('+', '_')}.yaml"
        _write_yaml(config_path, config)

        cmd = [
            args.python_bin,
            str(BASELINE_SCRIPTS / "run_repo_eval_subset.py"),
            "--split-file", str(eval_file),
            "--dataset", str(args.dataset),
            "--config", str(config_path),
            "--repos", *args.repos,
            "--max-per-repo", "0",
            "--memory-mode", "baseline" if ablation == "baseline" else "memory",
            "--model-key", args.model_key,
        ]
        if ablation != "baseline":
            cmd.extend(["--ablation-levels", ablation])

        stdout = _run(cmd, cwd=PROJECT_ROOT, capture=True)
        run_info = _extract_last_json_object(stdout)
        result_dir = Path(run_info.get("result_dir") or "")
        if not result_dir:
            raise RuntimeError(f"Could not determine result_dir for {ablation}")

        model_log_dir = run_root / _safe_model_dir_key(args.model_key)
        shared_log_details = model_log_dir / "log_details.json"
        trajectory_output = trajectories_dir / ablation.replace("+", "_")
        _run(
            [
                args.python_bin,
                str(SCRIPT_DIR / "collect_trajectories.py"),
                "--result-dir", str(result_dir),
                "--shared-log-details", str(shared_log_details),
                "--retrieval-log", str(retrieval_log),
                "--eval-file", str(eval_file),
                "--output-dir", str(trajectory_output),
                "--ablation-levels", ablation,
            ],
            cwd=PROJECT_ROOT,
        )

        manifest["runs"][ablation] = {
            "config": str(config_path),
            "result_dir": str(result_dir),
            "retrieval_log": str(retrieval_log),
            "trajectory_dir": str(trajectory_output),
        }

    manifest_path = args.output_dir / "subset_memory_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
