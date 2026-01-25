# ============================================================
#  run_benchmark.py — CI-Builds-Repair Benchmark Runner
# ============================================================
import os
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from benchmark import CIFixBenchmark
from load_config import load_config
from benchmark_functions import fix_apply_generated_patch

# ============================================================
#  Configuration
# ============================================================
model_name = "diff"
current_dir = os.getcwd()
config_path = os.path.join(current_dir, "config.yaml")

config = OmegaConf.load(config_path)

# Initialize benchmark object
CIBenchPython = CIFixBenchmark(model_name, config_path)

# ============================================================
#  CHOOSE ONE DATASET OPTION
# ============================================================

# ---------- OPTION 1: Local Dataset ----------
# Uncomment this block if you already have a dataset locally

# Locally can load dataset from base_dir/dataset
# dataset_info = os.path.join(config.get("base_dir"), "dataset", "lca_dataset.parquet")

# Can load dataset from online from huggingface hub
dataset_info = hf_hub_download(
    repo_id="ci-benchmark-user/ci-repair-bench",
    filename="ci_reoair_dataset.parquet",
    repo_type="dataset",
    token=config.get("HUGGINGFACE_TOKEN"),  # optional if you've done `huggingface-cli login`
)


# Load dataset once
all_ids = [row["id"] for row in CIBenchPython.get_dataset(dataset_info=dataset_info)]

# Select datapoints from 327 to end
# selected_ids = all_ids[68:]
# selected_ids = ['71', '72', '73', '74', '76', '77', '78', '79', '80', '82', '83', '84', '86', '87', '88', '89', '90', '91', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '173', '174', '175', '176', '177', '179', '181', '182', '183', '184', '186', '187', '189', '191', '192', '193', '194', '196', '197', '198', '199', '201', '202', '203', '204', '205', '207', '208', '209']

# ---------- OPTION 2: Online Dataset ----------
# Uncomment this block if you want to fetch dataset from an online source (e.g., Hugging Face)
# dataset_info = "JetBrains-Research/lca-ci-builds-repair"  # or any other dataset name/id

# ============================================================
#  Run the Benchmark
# ============================================================
print(" Starting benchmark evaluation...")

CIBenchPython.eval_dataset(
    fix_repo_function=fix_apply_generated_patch,
    dataset_info=dataset_info,
    num_dp=None,           # Limit number of datapoints (optional)
    ids_list=None,         # Provide specific IDs if needed
    force_download=False   # Set True to re-download from online
)

# ============================================================
#  Get and Analyze Results
# ============================================================
CIBenchPython.get_results()

# ============================================================
#  Evaluate Jobs (Optional)
# ============================================================
# job_ids_file = "examples/jobs_ids.jsonl"
# job_results = CIBenchPython.eval_jobs(
#     job_ids_file=job_ids_file,
#     result_filename="jobs_results_test.jsonl",
# )

# ============================================================
#  Analyze Existing Results (Optional)
# ============================================================
# job_results_file = "examples/jobs_results.jsonl"
# CIBenchPython.analyze_results(jobs_results_file=job_results_file)

# ============================================================
#  End of Script
# ============================================================
pass
