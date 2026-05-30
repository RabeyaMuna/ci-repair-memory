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

# Select datapoints from 327 to end
# selected_ids = all_ids[0:]
selected_ids = selected_ids = ['9', '10', '12', '14', '15', '19', '21', '25', '31', '32', '33', '42', '48', '49', '94', '96', '98',
      '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '122', '126', '135', '137', '140', '152',
      '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168',
      '169', '170', '171', '172', '173', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185',
      '186', '187', '188', '189', '191', '192', '193', '194', '195', '197', '198', '199', '200', '201', '202', '203',
      '204', '205', '206', '207', '208', '209', '215', '222', '234', '237', '238', '240', '255', '258', '266', '274',
      '288', '294', '295', '304', '307', '309', '314', '315', '332', '333', '336', '337', '338', '339', '346', '354',
      '356', '367', '373', '374', '385', '386', '387', '388', '391', '392', '393', '396', '398', '400', '406', '413',
      '417', '420', '429', '430', '432', '433', '444', '461', '463', '464', '465', '466', '469', '473', '474', '475',
      '478', '479', '480', '482', '483', '484', '486', '487', '488', '489', '490', '491', '493', '494', '495', '505',
      '506', '507', '508', '510', '511', '512', '513', '514', '515', '517', '518', '529', '531', '532', '533', '534',
      '535', '537', '539', '540', '541', '544', '545', '546', '565', '567', '569', '571']

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
