# ============================================================
#  run_benchmark.py — CI-Builds-Repair Benchmark Runner
# ============================================================
import os
import argparse
import json
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from benchmark import CIFixBenchmark
from benchmark.load_config import load_config
from benchmark.benchmark_functions import (
    fix_apply_generated_patch,
    fix_apply_ground_truth_patch,
    fix_none,
)

# ============================================================
#  Configuration
# ============================================================
parser = argparse.ArgumentParser(description="Run CI repair benchmark evaluation")
parser.add_argument(
    "--ground-truth",
    action="store_true",
    help=(
        "Reproduce the complete sha_fail-to-sha_success repository transition. "
        "Never use this mode for model predictions."
    ),
)
parser.add_argument(
    "--ids",
    help="Optional comma-separated dataset IDs; defaults to every dataset ID",
)
parser.add_argument(
    "--resume-missing",
    action="store_true",
    help=(
        "Ground-truth mode only: preserve the existing push manifest, run only "
        "dataset IDs not recorded there, and append successful pushes"
    ),
)
parser.add_argument(
    "--skip-upstream-sync",
    action="store_true",
    help=(
        "Skip upstream syncing for forked repositories. Use this if you've already "
        "run setup/sync_all_repos_with_upstream.py to pre-sync repos. "
        "This significantly speeds up benchmark execution."
    ),
)
parser.add_argument(
    "--resume",
    action="store_true",
    help=(
        "Resume benchmark by running only incomplete instances. "
        "Checks results/jobs_ids_{model}.jsonl to identify completed instances "
        "and runs only the remaining ones."
    ),
)
parser.add_argument(
    "--fetch-failed",
    action="store_true",
    help=(
        "Fetch failed instances: Push code WITHOUT applying any patches to collect "
        "baseline failure info, timing metrics (queue time, execution time), and see "
        "which instances actually fail without fixes."
    ),
)
args = parser.parse_args()

if args.resume_missing and not args.ground_truth:
    parser.error("--resume-missing requires --ground-truth")

# Determine model name based on mode
if args.fetch_failed:
    model_name = "failed-baseline"
elif args.ground_truth:
    model_name = "ground_truth"
else:
    model_name = "diff"
current_dir = os.getcwd()
config_path = os.path.join(current_dir, "config.yaml")

config = OmegaConf.load(config_path)

# Initialize benchmark object
CIBenchPython = CIFixBenchmark(model_name, config_path, skip_upstream_sync=args.skip_upstream_sync)

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
requested_ids = (args.ids or os.environ.get("CIBENCH_IDS", "")).strip()
selected_ids = (
    [item.strip() for item in requested_ids.split(",") if item.strip()]
    if requested_ids
    else all_ids
)

# Filter out completed instances if --resume flag is set
if args.resume:
    jobs_file = os.path.join(config.get("out_folder"), f"jobs_ids_{model_name}.jsonl")
    if os.path.exists(jobs_file):
        completed_ids = set()
        with open(jobs_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        job = json.loads(line)
                        # Try different possible ID field names
                        job_id = str(job.get('id') or job.get('datapoint_id') or job.get('issue_id', ''))
                        if job_id:
                            completed_ids.add(job_id)
                    except:
                        pass

        original_count = len(selected_ids)
        selected_ids = [id for id in selected_ids if str(id) not in completed_ids]

        print(f"[RESUME] Found {len(completed_ids)} completed instances in {jobs_file}")
        print(f"[RESUME] Filtered {original_count - len(selected_ids)} completed IDs")
        print(f"[RESUME] Running {len(selected_ids)} remaining instances")

        if len(selected_ids) == 0:
            print(f"[RESUME] All instances already completed! Exiting.")
            exit(0)
    else:
        print(f"[RESUME] No existing results file found ({jobs_file}). Running all {len(selected_ids)} instances.")


# selected_ids = all_ids[0:100]  # Default to all IDs if no specific IDs are provided
selected_ids = ["10"]

# selected_ids = [
#   "63",
#   "37",
#   "13",
#   "15",
#   "12",
#   "36",
#   "30",
#   "57",
#   "29",
#   "34",
#   "31",
#   "25",
#   "28",
#   "17",
#   "62",
#   "40",
#   "24",
#   "10",
#   "46",
#   "66",
#   "49",
#   "1",
#   "9",
#   "39",
#   "23",
#   "6",
#   "55",
#   "48",
#   "56",
#   "68",
#   "14",
#   "53",
#   "160",
#   "71",
#   "164",
#   "172",
#   "171",
#   "158",
#   "192",
#   "84",
#   "162",
#   "167",
#   "207",
#   "203",
#   "151",
#   "193",
#   "76",
#   "184",
#   "4",
#   "60",
#   "43",
#   "19",
#   "20",
#   "16",
#   "8",
#   "41",
#   "38",
#   "45",
#   "21",
#   "22",
#   "51",
#   "35",
#   "165",
#   "176",
#   "168",
#   "83",
#   "201",
#   "91",
#   "170",
#   "86",
#   "204",
#   "74",
#   "194",
#   "157",
#   "179",
#   "80",
#   "175",
#   "153",
#   "73",
#   "559",
#   "159",
#   "199",
#   "173",
#   "152",
#   "89",
#   "202",
#   "297",
#   "326",
#   "317",
#   "299",
#   "298",
#   "300",
#   "301",
#   "307",
#   "302",
#   "303",
#   "304",
#   "306",
#   "305",
#   "147",
#   "130",
#   "149",
#   "260",
#   "268",
#   "435",
#   "558",
#   "70",
#   "220",
#   "551",
#   "256",
#   "286",
#   "253",
#   "264",
#   "263",
#   "148",
#   "285",
#   "252",
#   "245",
#   "289",
#   "262",
#   "327",
#   "271",
#   "284",
#   "235",
#   "131",
#   "187"
# ]

if requested_ids:
    print(f" Running only requested dataset IDs: {', '.join(selected_ids)}")

if args.resume_missing:
    jobs_ids_file_path = os.path.join(
        config.get("out_folder"), "jobs_ids_ground_truth.jsonl"
    )
    recorded_ids = set()
    if os.path.exists(jobs_ids_file_path):
        with open(jobs_ids_file_path) as reader:
            for line_number, line in enumerate(reader, start=1):
                if not line.strip():
                    continue
                try:
                    recorded_ids.add(str(json.loads(line).get("id")))
                except (ValueError, TypeError) as exc:
                    parser.error(
                        f"Cannot safely resume: invalid JSON at "
                        f"{jobs_ids_file_path}:{line_number}: {exc}"
                    )

    selected_ids = [
        datapoint_id
        for datapoint_id in selected_ids
        if str(datapoint_id) not in recorded_ids
    ]
    print(
        f"[RESUME] Found {len(recorded_ids)} recorded pushes; "
        f"{len(selected_ids)} dataset IDs remain to push"
    )
    


# selected_ids = ["43","63","37","28","13","15","12","14","559","60","326","25","24","30","29","34","31","302","303","36","66","57","56","4","8","10","9","6","298","46","49","48","19","17","20","16","297","51","53","304","62","40","39","41","38","1","21","22","55","317","23","307","306","305","45","35","68","147","299","300","301","160","71","164","172","171","158","165","192","84","176","168","83","201","91","162","170","86","204","74","194","167","157","179","80","175","207","153","203","73","159","199","173","151","152","89","193","202","76","184","187","174","78","150","88","154","155","183","182","156","181","186","189","166","205","191","77","79","295","296","93","97","96","92","95","94","69","574","566","573","562","568","564","563","560","565","567","457","474","475","470","491","493","464","456","448","446","449","461","465","462","484","483","480","463","478","459","495","455","469","473","490","453","447","485","479","494","458","492","487","489","488","466","120","119","109","128","108","121","110","118","126","112","103","104","111","114","102","116","127","115","122","501","513","496","502","519","515","499","512"]

# selected_ids = ["43", "16", "19", "457", "297", "494", "560", "562”, “6]
# selected_ids = [
#   "43",
#   "63",
#   "37",
#   "28",
#   "13",
#   "15",
#   "12",
#   "14",
#   "559",
#   "60",
#   "326",
#   "25",
#   "24",
#   "30",
#   "29",
#   "34",
#   "31",
#   "302",
#   "303",
#   "36",
#   "66",
#   "57",
#   "56",
#   "4",
#   "8",
#   "10",
#   "9",
#   "6",
#   "298",
#   "46",
#   "49",
#   "48",
#   "19",
#   "17",
#   "20",
#   "16",
#   "297",
#   "51",
#   "53",
#   "304",
#   "62",
#   "40",
#   "39",
#   "41",
#   "38",
#   "1",
#   "21",
#   "22",
#   "55",
#   "317",
#   "23",
#   "307",
#   "306",
#   "305",
#   "45",
#   "35",
#   "68",
#   "147",
#   "299",
#   "300",
#   "301",
#   "160",
#   "71",
#   "164",
#   "172",
#   "171",
#   "158",
#   "165",
#   "192",
#   "84",
#   "176",
#   "168",
#   "83",
#   "201",
#   "91",
#   "162",
#   "170",
#   "86",
#   "204",
#   "74",
#   "194",
#   "167",
#   "157",
#   "179",
#   "80",
#   "175",
#   "207",
#   "153",
#   "203",
#   "73",
#   "159",
#   "199",
#   "173",
#   "151",
#   "152",
#   "89",
#   "193",
#   "202",
#   "76",
#   "184",
#   "187",
#   "174",
#   "78",
#   "150",
#   "88",
#   "154",
#   "155",
#   "183",
#   "182",
#   "156",
#   "181",
#   "186",
#   "189",
#   "166",
#   "205",
#   "191",
#   "77",
#   "79",
#   "295",
#   "296",
#   "93",
#   "97",
#   "96",
#   "92",
#   "95",
#   "94",
#   "69",
#   "574",
#   "566",
#   "573",
#   "562",
#   "568",
#   "564",
#   "563",
#   "560",
#   "565",
#   "567",
#   "457",
#   "474",
#   "475",
#   "470",
#   "491",
#   "493",
#   "464",
#   "456",
#   "448",
#   "446",
#   "449",
#   "461",
#   "465",
#   "462",
#   "484",
#   "483",
#   "480",
#   "463",
#   "478",
#   "459",
#   "495",
#   "455",
#   "469",
#   "473",
#   "490",
#   "453",
#   "447",
#   "485",
#   "479",
#   "494",
#   "458",
#   "492",
#   "487",
#   "489",
#   "488",
#   "466",
#   "120",
#   "119",
#   "109",
#   "128",
#   "108",
#   "121",
#   "110",
#   "118",
#   "126",
#   "112",
#   "103",
#   "104",
#   "111",
#   "114",
#   "102",
#   "116",
#   "127",
#   "115",
#   "122",
#   "501",
#   "513",
#   "496",
#   "502",
#   "519",
#   "515",
#   "499",
#   "512",
#   "504",
#   "517",
#   "511",
#   "518",
#   "503",
#   "505",
#   "508",
#   "520",
#   "507",
#   "500",
#   "516",
#   "101",
#   "421",
#   "409",
#   "420",
#   "419",
#   "417",
#   "415",
#   "413",
#   "412",
#   "418",
#   "410",
#   "414",
#   "406",
#   "394",
#   "388",
#   "392",
#   "390",
#   "396",
#   "386",
#   "398",
#   "378",
#   "381",
#   "387",
#   "377",
#   "400",
#   "391",
#   "384",
#   "385",
#   "402",
#   "393",
#   "389",
#   "395",
#   "379",
#   "137",
#   "138",
#   "142",
#   "141",
#   "134",
#   "136",
#   "436",
#   "440",
#   "437",
#   "439",
#   "443",
#   "441",
#   "442",
#   "438",
#   "543",
#   "526",
#   "536",
#   "540",
#   "532",
#   "531",
#   "545",
#   "544",
#   "539",
#   "548",
#   "538",
#   "527",
#   "523",
#   "547",
#   "541",
#   "528",
#   "537",
#   "529",
#   "533",
#   "535",
#   "149",
#   "143",
#   "145",
#   "146",
#   "144",
#   "70",
#   "131",
#   "130",
#   "551",
#   "227",
#   "323",
#   "239",
#   "308",
#   "148",
#   "235",
#   "224",
#   "225",
#   "313",
#   "240",
#   "229",
#   "230",
#   "241",
#   "264",
#   "279",
#   "280",
#   "309",
#   "286",
#   "276",
#   "277",
#   "324",
#   "325",
#   "256",
#   "237",
#   "238",
#   "294",
#   "292",
#   "282",
#   "283",
#   "246",
#   "272",
#   "273",
#   "367",
#   "365",
#   "291",
#   "311",
#   "312",
#   "232",
#   "233",
#   "234",
#   "558",
#   "260",
#   "427",
#   "376",
#   "426",
#   "433",
#   "429",
#   "430",
#   "422",
#   "375",
#   "431",
#   "423",
#   "434",
#   "341",
#   "342",
#   "345",
#   "340",
#   "343",
#   "252",
#   "253",
#   "370",
#   "371",
#   "369",
#   "373",
#   "372",
#   "245",
#   "288",
#   "287",
#   "318",
#   "319",
#   "251",
#   "266",
#   "267",
#   "322",
#   "222",
#   "258",
#   "259",
#   "255",
#   "315",
#   "289",
#   "220",
#   "331",
#   "339",
#   "332",
#   "336",
#   "328",
#   "330",
#   "333",
#   "334",
#   "335",
#   "214",
#   "215",
#   "213",
#   "211",
#   "314",
#   "248",
#   "327",
#   "285",
#   "270",
#   "360",
#   "356",
#   "350",
#   "351",
#   "352",
#   "355",
#   "357",
#   "353",
#   "354",
#   "348",
#   "321",
#   "435",
#   "217",
#   "263",
#   "242",
#   "261",
#   "219",
#   "268",
#   "271",
#   "262",
#   "284",
#   "364",
#   "363",
#   "362"
# ]




# ---------- OPTION 2: Online Dataset ----------
# Uncomment this block if you want to fetch dataset from an online source (e.g., Hugging Face)
# dataset_info = "JetBrains-Research/lca-ci-builds-repair"  # or any other dataset name/id

# ============================================================
#  Run the Benchmark
# ============================================================
print(" Starting benchmark evaluation...")

# Select the appropriate fix function based on mode
if args.fetch_failed:
    fix_function = fix_none
    print(" [FETCH-FAILED MODE] Pushing code WITHOUT patches to collect baseline metrics")
elif args.ground_truth:
    fix_function = fix_apply_ground_truth_patch
else:
    fix_function = fix_apply_generated_patch

CIBenchPython.eval_dataset(
    fix_repo_function=fix_function,
    dataset_info=dataset_info,
    num_dp=None,           # Limit number of datapoints (optional)
    ids_list=selected_ids,         # Provide specific IDs if needed
    force_download=False,  # Set True to re-download from online
    resume=args.resume_missing or args.resume,  # Support both ground-truth and normal resume
)

# ============================================================
#  Get and Analyze Results
# ============================================================
CIBenchPython.get_results()

# ============================================================
#  Generate Timing Report
# ============================================================
print("\n" + "="*80)
print(" GENERATING TIMING REPORT")
print("="*80)

try:
    exec(open("generate_timing_report.py").read())
except FileNotFoundError:
    print("⚠ generate_timing_report.py not found, skipping timing analysis")
except Exception as e:
    print(f"⚠ Could not generate timing report: {e}")

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
