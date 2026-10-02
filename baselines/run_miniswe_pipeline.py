#!/usr/bin/env python3
"""
MiniSWE-Agent Pipeline with Result Reuse

Features:
- Reuses existing log_details.json and fault_localization.json from baselines/results/
- Generates missing steps only
- No data leakage between models
- Supports: gpt-5-mini, deepseek-v4-flash
- Supports: llm or bm25 log analyzer
- Tracks all LLM usage with cost/timing

Directory Structure:
- Log analysis & Fault localization: /baselines/results/<model>_<analyzer>/
- Patches (miniswe-agent mode): /miniswe-agent/baselines/miniswe-agent_<model>_<analyzer>/
- Patches (reference mode): /baselines/results/<model>_<analyzer>/

Usage:
    python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe
    python run_miniswe_pipeline.py --model deepseek-v4-flash --analyzer llm --mode miniswe
    python run_miniswe_pipeline.py --model gpt-5-mini --analyzer bm25 --mode reference
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Add paths. MiniSWE's repository root contains its CI helper packages, while
# ``src`` contains the actual ``minisweagent`` package.
_MINISWE_ROOT = Path(__file__).resolve().parent.parent / "miniswe-agent"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_MINISWE_ROOT))
sys.path.insert(0, str(_MINISWE_ROOT / "src"))

from utilities.llm_provider import get_llm
from utilities.llm_tracker import tracker
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.fault_localization import FaultLocalization
from utilities.ensure_repo import ensure_repo_at_commit
from utilities.fetch_failed_commit_changed_files import collect_changed_files_for_fail_and_parent

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MiniSWEPipeline:
    """
    Pipeline that reuses existing results and runs miniswe-agent
    """

    def __init__(self, model_name, config, log_analyzer_type="llm", mode="miniswe", num_instances=None):
        """
        Args:
            model_name: 'gpt-5-mini' or 'deepseek-v4-flash'
            config: OmegaConf config object
            log_analyzer_type: 'llm' or 'bm25'
            mode: 'miniswe' (save patches to miniswe-agent/) or 'reference' (save to baselines/results/)
            num_instances: Number of instances to process (None = all)
        """
        self.model_name = model_name
        self.config = config
        self.log_analyzer_type = log_analyzer_type
        self.mode = mode
        self.num_instances = num_instances

        # Get LLM
        self.llm = get_llm(model_name)

        # ALWAYS load/save log analysis and fault localization from baselines/results
        configured_result_dir = Path(config.project_result_dir)
        if not configured_result_dir.is_absolute():
            configured_result_dir = PROJECT_ROOT / configured_result_dir
        baselines_result_dir = configured_result_dir / f"{model_name}_{log_analyzer_type}"
        baselines_result_dir.mkdir(parents=True, exist_ok=True)

        self.log_details_file = str(baselines_result_dir / "log_details.json")
        self.fault_loc_file = str(baselines_result_dir / "fault_localization.json")

        configured_repo_dir = Path(config.baseline_repo_folder)
        if not configured_repo_dir.is_absolute():
            configured_repo_dir = PROJECT_ROOT / configured_repo_dir
        self.baseline_repo_folder = configured_repo_dir

        # Patches location depends on mode
        if mode == "miniswe":
            # MiniSWE-Agent mode: save to miniswe-agent/baselines/
            miniswe_base = PROJECT_ROOT / "miniswe-agent" / "baselines" / "results" / f"miniswe-agent_{model_name}_{log_analyzer_type}"
            miniswe_base.mkdir(parents=True, exist_ok=True)
            self.patches_dir = str(miniswe_base)
            self.patches_file = str(miniswe_base / "preds.json")
            self.metrics_file = str(miniswe_base / "run_metrics.json")
            self.cost_file = str(miniswe_base / "costs.json")
        else:
            # Reference mode: save to baselines/results/
            self.patches_dir = str(baselines_result_dir)
            self.patches_file = str(baselines_result_dir / "generated_patches.json")
            self.metrics_file = str(baselines_result_dir / "run_metrics.json")
            self.cost_file = str(baselines_result_dir / "costs.json")

        print(f"\n{'='*80}")
        print(f"MiniSWE-Agent Pipeline")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Log Analyzer: {log_analyzer_type}")
        print(f"Mode: {mode}")
        print(f"\nInput (reuse if exists):")
        print(f"  Log details:        {self.log_details_file}")
        print(f"  Fault localization: {self.fault_loc_file}")
        print(f"\nOutput (patches):")
        print(f"  Directory: {self.patches_dir}")
        print(f"  Patches:   {os.path.basename(self.patches_file)}")
        print(f"{'='*80}\n")

    def load_existing_results(self):
        """Load existing log details, fault localization, and patches if available"""

        log_details = {}
        fault_localization = {}
        patches = {}

        # Load log details
        if os.path.exists(self.log_details_file):
            with open(self.log_details_file, 'r') as f:
                log_list = json.load(f)
                # Convert list to dict keyed by sha_fail
                log_details = {item.get('sha_fail'): item for item in log_list if isinstance(item, dict)}
            print(f"[LOAD] {len(log_details)} existing log details from {self.model_name}_{self.log_analyzer_type}")
        else:
            print(f"[INFO] No existing log details found for {self.model_name}_{self.log_analyzer_type}")

        # Load fault localization
        if os.path.exists(self.fault_loc_file):
            with open(self.fault_loc_file, 'r') as f:
                fault_list = json.load(f)
                # Convert list to dict keyed by sha_fail
                fault_localization = {item.get('sha_fail'): item for item in fault_list if isinstance(item, dict)}
            print(f"[LOAD] {len(fault_localization)} existing fault localizations from {self.model_name}_{self.log_analyzer_type}")
        else:
            print(f"[INFO] No existing fault localization found for {self.model_name}_{self.log_analyzer_type}")

        # Load existing patches. Older pipeline runs wrote a list, while the
        # native CIBench writer uses a dictionary keyed by instance id.
        if os.path.exists(self.patches_file):
            with open(self.patches_file, 'r') as f:
                patch_data = json.load(f)
                if isinstance(patch_data, list):
                    patch_items = patch_data
                elif isinstance(patch_data, dict):
                    patch_items = [
                        {**item, "id": item.get("id", instance_id)}
                        for instance_id, item in patch_data.items()
                        if isinstance(item, dict)
                    ]
                else:
                    patch_items = []
                patches = {
                    str(item.get("id")): item
                    for item in patch_items
                    if item.get("id") is not None
                }
            print(f"[LOAD] {len(patches)} existing patches from {self.model_name}_{self.log_analyzer_type}")
        else:
            print(f"[INFO] No existing patches found")

        return log_details, fault_localization, patches

    def save_results(self, log_details, fault_localization, patches):
        """Save all results"""

        # Save log details
        with open(self.log_details_file, 'w') as f:
            json.dump(list(log_details.values()), f, indent=4)
        print(f"[SAVE] {len(log_details)} log details")

        # Save fault localization
        with open(self.fault_loc_file, 'w') as f:
            json.dump(list(fault_localization.values()), f, indent=4)
        print(f"[SAVE] {len(fault_localization)} fault localizations")

        # Save patches
        with open(self.patches_file, 'w') as f:
            json.dump(patches, f, indent=4)
        print(f"[SAVE] {len(patches)} patches")

    def run_log_analysis(self, datapoint, repo_path):
        """Run CI log analysis for a single datapoint"""

        if self.log_analyzer_type == "llm":
            return CILogAnalyzerLLM(
                repo_path=repo_path,
                ci_log=datapoint["logs"],
                sha_fail=datapoint["sha_fail"],
                workflow=datapoint["workflow"],
                workflow_path=datapoint["workflow_path"],
                llm=self.llm,
                model_name=self.model_name,
                task_id=datapoint["id"]
            ).run()
        else:  # bm25
            return CILogAnalyzerBM25(
                repo_path=repo_path,
                ci_log=datapoint["logs"],
                sha_fail=datapoint["sha_fail"],
                workflow=datapoint["workflow"],
                workflow_path=datapoint["workflow_path"],
                llm=self.llm,
                model_name=self.model_name,
                task_id=datapoint["id"]
            ).run()

    def run_fault_localization(self, datapoint, repo_path, log_analysis_result, changed_files_info):
        """Run fault localization for a single datapoint"""

        return FaultLocalization(
            sha_fail=datapoint["sha_fail"],
            repo_path=repo_path,
            error_logs=log_analysis_result,
            workflow=datapoint["workflow"],
            llm=self.llm,
            model_name=self.model_name,
            changed_files_info=changed_files_info,
        ).run()

    def run_miniswe_agent(self, fault_localization_result, datapoint):
        """
        Run MiniSWE-Agent for patch generation using fault localization data
        """

        print(f"[MINISWE] Generating patch for instance {datapoint['id']}...")

        # Prepare fault localization input
        fault_data = fault_localization_result.get("fault_localization_data", [])

        if not fault_data:
            print(f"[MINISWE] No fault localization data available")
            return {
                "id": datapoint["id"],
                "sha_fail": datapoint["sha_fail"],
                "diff": "",
                "model": self.model_name,
                "method": "miniswe-agent",
                "status": "no_fault_data"
            }

        # Format problem statement from fault localization. This exact text is
        # passed through to CIBench and becomes the repair agent's task.
        problem_statement = self._format_problem_from_fl(fault_data, datapoint)

        # Try to use actual miniswe-agent if available
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "miniswe-agent" / "src"))

            # Attempt to import and use miniswe-agent
            # This will work if miniswe-agent is properly installed/available
            result = self._call_miniswe_agent(
                problem_statement=problem_statement,
                datapoint=datapoint
            )

            return {
                **result,
                "id": datapoint["id"],
                "sha_fail": datapoint["sha_fail"],
                "diff": result.get("patch", result.get("diff", "")),
                "model": self.model_name,
                "method": "miniswe-agent",
                "fault_files": len(fault_data),
                "status": "completed"
            }

        except Exception as e:
            print(f"[MINISWE] ERROR: MiniSWE-Agent failed - {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"MiniSWE-Agent failed: {e}")

    def _format_problem_from_fl(self, fault_data, datapoint):
        """Format fault localization data into problem statement"""
        problem = f"Fix the CI failure in commit {datapoint['sha_fail'][:8]}\n\n"
        problem += "Fault Localization identified the following suspicious files:\n\n"

        fault_number = 0
        for file_entry in fault_data:
            nested_faults = file_entry.get("faults")
            entries = nested_faults if isinstance(nested_faults, list) else [file_entry]
            for fault in entries:
                if not isinstance(fault, dict):
                    continue
                fault_number += 1
                file_path = fault.get("file_path") or file_entry.get("file_path", "unknown")
                reason = fault.get("reason", "No reason provided")
                line_range = fault.get("line_range", "unknown")
                level = fault.get("fault_localization_level", "unknown")
                issue_type = fault.get("issue_type", "unknown")
                snippet = fault.get("code_snippet", "")

                problem += f"{fault_number}. {file_path} (lines {line_range})\n"
                problem += f"   Localization level: {level}\n"
                problem += f"   Issue type: {issue_type}\n"
                problem += f"   Reason: {reason}\n"
                if snippet:
                    problem += f"   Relevant code:\n```\n{snippet}\n```\n"
                problem += "\n"

        return problem

    def _call_miniswe_agent(self, problem_statement, datapoint):
        """
        Call actual miniswe-agent using cibench module
        """
        from minisweagent.run.benchmarks import cibench
        from minisweagent.config import get_config_from_spec
        from pathlib import Path as MiniswePath

        # Prepare instance for miniswe-agent
        instance = {
            "instance_id": str(datapoint["id"]),
            "id": str(datapoint["id"]),
            "sha_fail": datapoint["sha_fail"],
            "repo_owner": datapoint.get("repo_owner", ""),
            "repo_name": datapoint.get("repo_name", ""),
            "problem_statement": problem_statement,
            "problem_source": "ci failure",
        }

        # Get config
        config = get_config_from_spec(str(cibench.DEFAULT_CONFIG_FILE))
        config["model"]["model_name"] = cibench.configure_model_environment(
            self.model_name
        )

        # Create progress manager (minimal)
        class SimpleProgressManager:
            def on_instance_start(self, iid): pass
            def update_instance_status(self, iid, status): pass
            def on_instance_end(self, iid, status): pass

        # Run miniswe-agent
        output_dir = MiniswePath(self.patches_dir)
        cibench.process_instance(
            instance=instance,
            output_dir=output_dir,
            config=config,
            progress_manager=SimpleProgressManager(),
            memory_root=None,
            memory_enabled=False,
            memory_top_k=3,
            memory_ablation_levels="",
            memory_plugin_path=None,
            context_model=self.model_name,
            save_memory=False,
        )

        # Read generated patch
        preds_file = output_dir / "preds.json"
        if preds_file.exists():
            with open(preds_file) as f:
                preds = json.load(f)
                instance_id = str(datapoint["id"])
                if isinstance(preds, dict) and instance_id in preds:
                    prediction = dict(preds[instance_id])
                    prediction["patch"] = prediction.get("diff", "")
                    return prediction
                if isinstance(preds, list):
                    for prediction in preds:
                        if str(prediction.get("id")) == instance_id:
                            result = dict(prediction)
                            result["patch"] = result.get("diff", "")
                            return result

        return {"patch": ""}

    def process_dataset(self, dataset):
        """
        Process dataset with result reuse
        """

        # Load existing results for THIS model/analyzer only
        log_details, fault_localization, existing_patches = self.load_existing_results()
        patches = dict(existing_patches)

        # Determine which instances to process
        if self.num_instances:
            dataset = dataset[:self.num_instances]

        print(f"\n[PROCESS] {len(dataset)} instances with {self.model_name}_{self.log_analyzer_type}\n")

        for idx, datapoint in enumerate(dataset, 1):
            task_id = datapoint["id"]
            repo_name = datapoint["repo_name"]
            repo_owner = datapoint["repo_owner"]
            sha_fail = datapoint["sha_fail"]
            benchmark_owner = self.config.benchmark_owner

            print(f"\n{'='*80}")
            print(f"[{idx}/{len(dataset)}] Instance {task_id}: {repo_name} @ {sha_fail[:8]}")
            print(f"{'='*80}")

            # Check what work needs to be done
            need_log_analysis = sha_fail not in log_details
            need_fault_localization = sha_fail not in fault_localization
            task_key = str(task_id)
            existing_patch = existing_patches.get(task_key)
            if existing_patch is None:
                existing_patch = next(
                    (
                        patch for patch in existing_patches.values()
                        if str(patch.get("sha_fail", "")) == str(sha_fail)
                    ),
                    None,
                )
            need_patch_generation = existing_patch is None

            # Show what exists and what's missing
            status = []
            if not need_log_analysis:
                status.append("log✓")
            else:
                status.append("log✗")
            if not need_fault_localization:
                status.append("fault✓")
            else:
                status.append("fault✗")
            if not need_patch_generation:
                status.append("patch✓")
            else:
                status.append("patch✗")
            print(f"[CHECK] {' | '.join(status)}")

            if not (need_log_analysis or need_fault_localization or need_patch_generation):
                print(f"[SKIP] All data exists (log + fault + patch), no repo access needed")
                continue

            # The baseline checkout is needed only by log analysis and fault
            # localization. MiniSWE-Agent manages its own reusable checkout.
            if need_log_analysis or need_fault_localization:
                repo_path = str(self.baseline_repo_folder / repo_name)
                repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
                ensure_repo_at_commit(repo_url, repo_path, sha_fail)
            else:
                print("[REPO] Baseline checkout not needed; MiniSWE-Agent will reuse its checkout cache")
                repo_path = None

            # STEP 1: CI Log Analysis (Reuse or Generate)
            if not need_log_analysis:
                print(f"[1/3] REUSE log analysis for {self.model_name}_{self.log_analyzer_type}")
                log_analysis_result = log_details[sha_fail]
            else:
                print(f"[1/3] RUN log analysis with {self.model_name}_{self.log_analyzer_type}...")
                try:
                    log_analysis_result = self.run_log_analysis(datapoint, repo_path)
                    log_analysis_result['sha_fail'] = sha_fail
                    log_details[sha_fail] = log_analysis_result
                    # Save incrementally
                    with open(self.log_details_file, 'w') as f:
                        json.dump(list(log_details.values()), f, indent=4)
                    print(f"[1/3] COMPLETE")
                except Exception as e:
                    print(f"[1/3] FAILED: {e}")
                    continue

            # STEP 2: Fault Localization (Reuse or Generate)
            if not need_fault_localization:
                print(f"[2/3] REUSE fault localization for {self.model_name}_{self.log_analyzer_type}")
                fault_loc_result = fault_localization[sha_fail]
            else:
                print(f"[2/3] RUN fault localization with {self.model_name}_{self.log_analyzer_type}...")
                try:
                    # Get changed files
                    changed_files_info = collect_changed_files_for_fail_and_parent(
                        owner=benchmark_owner,
                        repo=repo_name,
                        repo_path=repo_path,
                        sha_fail=sha_fail,
                        workflow_rel_path=datapoint["workflow_path"],
                        workflow_yaml_from_dataset=datapoint["workflow"]
                    )

                    fault_loc_result = self.run_fault_localization(
                        datapoint, repo_path, log_analysis_result, changed_files_info
                    )
                    fault_loc_result['sha_fail'] = sha_fail
                    fault_localization[sha_fail] = fault_loc_result
                    # Save incrementally
                    with open(self.fault_loc_file, 'w') as f:
                        json.dump(list(fault_localization.values()), f, indent=4)

                    num_files = len(fault_loc_result.get("fault_localization_data", []))
                    print(f"[2/3] COMPLETE: {num_files} suspicious files")

                except Exception as e:
                    print(f"[2/3] FAILED: {e}")
                    continue

            # Check if we have suspicious files
            if not fault_loc_result.get("fault_localization_data"):
                print(f"[WARN] No suspicious files found, skipping patch generation")
                continue

            # STEP 3: MiniSWE-Agent Patch Generation (Reuse or Generate)
            if not need_patch_generation:
                print(f"[3/3] REUSE patch for {self.model_name}_{self.log_analyzer_type}")
                patches[task_key] = existing_patch
            else:
                print(f"[3/3] RUN MiniSWE-Agent with {self.model_name}...")
                try:
                    # CIBench updates preds.json in place and requires its
                    # instance-keyed dictionary format. Normalize legacy list
                    # files only when a new patch is actually being generated.
                    with open(self.patches_file, 'w') as f:
                        json.dump(patches, f, indent=4)
                    patch_result = self.run_miniswe_agent(
                        fault_loc_result, datapoint
                    )
                    patches[task_key] = patch_result
                    # Save incrementally
                    with open(self.patches_file, 'w') as f:
                        json.dump(patches, f, indent=4)
                    print(f"[3/3] COMPLETE")
                except Exception as e:
                    print(f"[3/3] FAILED: {e}")
                    continue

        # Every generated artifact is saved immediately above. Avoid rewriting
        # existing files when the whole dataset was reused.
        print(f"\n{'='*80}")
        print("Processing complete; generated results were saved incrementally.")
        print(f"{'='*80}")

        return {
            "log_details": list(log_details.values()),
            "fault_localization": list(fault_localization.values()),
            "patches": list(patches.values())
        }


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description='Run MiniSWE-Agent pipeline')
    parser.add_argument(
        '--model',
        choices=['gpt-5-mini', 'deepseek-v4-flash'],
        required=True,
        help='Model to use'
    )
    parser.add_argument(
        '--analyzer',
        choices=['llm', 'bm25'],
        default='llm',
        help='Log analyzer type (default: llm)'
    )
    parser.add_argument(
        '--mode',
        choices=['miniswe', 'reference'],
        default='miniswe',
        help='Mode: miniswe (save to miniswe-agent/) or reference (save to baselines/) (default: miniswe)'
    )
    parser.add_argument(
        '--num-instances',
        type=int,
        default=None,
        help='Number of instances to process (default: all)'
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MiniSWE-Agent Pipeline                                     ║
║                                                                              ║
║  Model:    {args.model:<60} ║
║  Analyzer: {args.analyzer:<60} ║
║  Mode:     {args.mode:<60} ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)

    # Load dataset
    hf_token = os.getenv("HF_TOKEN") or config.get("HUGGINGFACE_TOKEN")

    print("Loading dataset...")
    dataset_path = hf_hub_download(
        repo_id="ci-benchmark-user/ci-repair-bench",
        filename="ci_repair_dataset.parquet",
        repo_type="dataset",
        token=hf_token,
    )

    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    print(f"Loaded {len(dataset)} instances from dataset\n")

    # Create and run pipeline
    pipeline = MiniSWEPipeline(
        model_name=args.model,
        config=config,
        log_analyzer_type=args.analyzer,
        mode=args.mode,
        num_instances=args.num_instances
    )

    results = pipeline.process_dataset(dataset)

    # Print summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Log details:          {len(results['log_details'])}")
    print(f"Fault localizations:  {len(results['fault_localization'])}")
    print(f"Patches generated:    {len(results['patches'])}")

    # Print tracking summary
    print(f"\n{'='*80}")
    tracker.print_summary()

    # Save tracking report
    usage_report_path = os.path.join(
        pipeline.patches_dir,
        f"llm_usage_{args.model}_{args.analyzer}.json"
    )
    tracker.save_report(usage_report_path, include_all_calls=True)
    print(f"\nLLM usage report saved: {usage_report_path}")

    print(f"\n{'='*80}")
    print("All results saved to:")
    print(f"  {pipeline.patches_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
