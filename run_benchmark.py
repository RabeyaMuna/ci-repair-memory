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
dataset_info = os.path.join(config.get("base_dir"), "dataset", "lca_dataset.parquet")

# Can load dataset from online from huggingface hub
# dataset_info = hf_hub_download(
#     repo_id="ci-benchmark-user/ci-repair-bench",
#     filename="ci_repair_dataset.parquet",
#     repo_type="dataset",
#     token=config.get("HUGGINGFACE_TOKEN"),
# )



# Load dataset once
all_ids = [row["id"] for row in CIBenchPython.get_dataset(dataset_info=dataset_info)]

selected_ids = ["378"]

# Select datapoints from 327 to end
# selected_ids = all_ids[0:]

# # taipy - avaiga/taipy (19 issues)
# selected_ids = ['132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '436', 
#                 '437', '438', '439', '440', '441', '442', '443']

# # sqlglot - tobymao/sqlglot (5 issues)
# selected_ids = ['275', '276', '277', '324', '325']

# selected_ids = [66, 67, 93, 96, 98, 99, 100, 101, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 167, 202, 212, 223, 224, 225, 228, 229, 230, 232, 234, 235, 238, 240, 241, 242, 245, 248, 250, 251, 252, 253, 256, 257, 258, 260, 261, 264, 265, 266, 267, 269, 270, 271, 272, 273, 274, 277, 281, 282, 283, 285, 286, 289, 297, 298, 299, 300, 301, 302, 303, 310, 311, 312, 313, 314, 316, 318, 319, 321, 322, 324, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 341, 343, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 370, 372, 373, 374, 378, 436, 437, 438, 439, 440, 441, 442, 443, 444, 475, 519, 526, 551, 568, 569, 571, 572]

selected_ids = [
    "409", "410", "411", "412", "413", "414", "415", "417", "418", "419",
    "420", "421", "560", "562", "563", "564", "565", "566", "567", "568",
    "572", "573", "574",
]


# Total IDs: 23


# selected_ids = ['102', '104', '105', '106', '107', '108', '109', '110', '111', '112','113', '116', '118', '119', '122', '123', '125', '127', '128', '129', '156', '157', '158', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '176', '177', '178', '179', '180', '181', '182', '185', '187', '188', '190', '192', '193', '194', '195', '196', '197', '198', '200', '201', '202', '204', '206','208', '209', '295', '296', '407', '408', '409', '410', '411', '413','415', '416', '417', '418', '421', '71', '72', '73', '75', '76','77', '81', '82', '84', '85', '86', '87', '89', '90']



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
    ids_list=selected_ids,         # Provide specific IDs if needed
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
