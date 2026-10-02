#!/usr/bin/env python3
"""
Verification script to ensure everything is set up correctly
Run this after setup.sh to verify the installation
"""

import sys
import os
from pathlib import Path

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_success(msg):
    print(f"{GREEN}✓{RESET} {msg}")

def print_error(msg):
    print(f"{RED}✗{RESET} {msg}")

def print_warning(msg):
    print(f"{YELLOW}!{RESET} {msg}")

def main():
    print("=" * 60)
    print("CI-REPAIR-BENCH Setup Verification")
    print("=" * 60)
    print()

    errors = []
    warnings = []

    # Check 1: Python version
    print("[1] Checking Python version...")
    if sys.version_info >= (3, 10):
        print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    else:
        print_error(f"Python {sys.version_info.major}.{sys.version_info.minor} (requires 3.10+)")
        errors.append("Python version too old")
    print()

    # Check 2: Virtual environment
    print("[2] Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_success("Virtual environment active")
    else:
        print_warning("Virtual environment not active (run: source venv/bin/activate)")
        warnings.append("Virtual environment not active")
    print()

    # Check 3: Core imports
    print("[3] Checking core imports...")
    try:
        from dotenv import load_dotenv
        print_success("python-dotenv")
    except ImportError as e:
        print_error(f"python-dotenv: {e}")
        errors.append("python-dotenv not installed")

    try:
        from langchain_openai import ChatOpenAI
        print_success("langchain-openai")
    except ImportError as e:
        print_error(f"langchain-openai: {e}")
        errors.append("langchain-openai not installed")

    try:
        from omegaconf import OmegaConf
        print_success("omegaconf")
    except ImportError as e:
        print_error(f"omegaconf: {e}")
        errors.append("omegaconf not installed")

    try:
        import pandas as pd
        print_success("pandas")
    except ImportError as e:
        print_error(f"pandas: {e}")
        errors.append("pandas not installed")
    print()

    # Check 4: Project modules
    print("[4] Checking project modules...")
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root / "baselines"))

    try:
        from utilities.llm_provider import get_llm, LLM_REGISTRY
        print_success("llm_provider")
    except ImportError as e:
        print_error(f"llm_provider: {e}")
        errors.append("llm_provider not importable")

    try:
        from utilities.llm_tracker import tracker
        print_success("llm_tracker")
    except ImportError as e:
        print_error(f"llm_tracker: {e}")
        errors.append("llm_tracker not importable")

    try:
        from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
        print_success("ci_log_analyzer_llm")
    except ImportError as e:
        print_error(f"ci_log_analyzer_llm: {e}")
        errors.append("ci_log_analyzer_llm not importable")

    try:
        from ci_repair.fault_localization import FaultLocalization
        print_success("fault_localization")
    except ImportError as e:
        print_error(f"fault_localization: {e}")
        errors.append("fault_localization not importable")
    print()

    # Check 5: Environment variables
    print("[5] Checking environment variables...")
    load_dotenv()

    env_vars = {
        "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }

    for var_name, var_value in env_vars.items():
        if var_value:
            masked = var_value[:10] + "..." if len(var_value) > 10 else var_value
            print_success(f"{var_name}: {masked}")
        else:
            print_error(f"{var_name}: NOT SET")
            errors.append(f"{var_name} not set in .env")
    print()

    # Check 6: Model configuration
    print("[6] Checking model configuration...")
    if 'LLM_REGISTRY' in dir():
        for model_name in ["gpt-5-mini", "deepseek-v4-flash"]:
            if model_name in LLM_REGISTRY:
                info = LLM_REGISTRY[model_name]
                api_key_status = "SET" if info.api_key else "NOT SET"
                print_success(f"{model_name}: {info.model_name} (API Key: {api_key_status})")
                if not info.api_key:
                    warnings.append(f"{model_name} API key not set")
            else:
                print_error(f"{model_name}: NOT CONFIGURED")
                errors.append(f"{model_name} not in registry")
    print()

    # Check 7: Directory structure
    print("[7] Checking directory structure...")
    required_dirs = [
        "baselines/results",
        "baselines/repo_cloned",
        "baselines/changed_files",
        "baselines/exceptions",
        "miniswe-agent/baselines/results",
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print_success(dir_path)
        else:
            print_warning(f"{dir_path} (will be created on first run)")
    print()

    # Check 8: Configuration file
    print("[8] Checking configuration file...")
    config_path = project_root / "config.yaml"
    if config_path.exists():
        print_success("config.yaml exists")
        try:
            config = OmegaConf.load(config_path)
            print_success(f"  benchmark_owner: {config.benchmark_owner}")
            print_success(f"  baseline_repo_folder: {config.baseline_repo_folder}")
        except Exception as e:
            print_error(f"config.yaml parse error: {e}")
            errors.append("config.yaml parse error")
    else:
        print_error("config.yaml NOT FOUND")
        errors.append("config.yaml missing")
    print()

    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)

    if errors:
        print(f"\n{RED}ERRORS ({len(errors)}):{RESET}")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print(f"\n{YELLOW}WARNINGS ({len(warnings)}):{RESET}")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors and not warnings:
        print(f"\n{GREEN}✓ ALL CHECKS PASSED!{RESET}")
        print("\nYou're ready to run:")
        print("  cd baselines")
        print("  python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe --num-instances 1")
        return 0
    elif errors:
        print(f"\n{RED}✗ SETUP INCOMPLETE{RESET}")
        print("Please fix the errors above before running the pipelines.")
        return 1
    else:
        print(f"\n{YELLOW}⚠ SETUP COMPLETE WITH WARNINGS{RESET}")
        print("You can proceed but some features may not work.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
