#!/bin/bash
# CI-REPAIR-BENCH Complete Setup Script
# Sets up a single virtual environment for all pipelines

set -e  # Exit on error

PROJECT_ROOT="/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH"
VENV_PATH="$PROJECT_ROOT/venv"

echo "=========================================="
echo "CI-REPAIR-BENCH Complete Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/8] Checking Python version..."
PYTHON_VERSION=$(python3 --version)
echo "  Found: $PYTHON_VERSION"
echo ""

# Create virtual environment
if [ -d "$VENV_PATH" ]; then
    echo "[2/8] Virtual environment exists at: $VENV_PATH"
    read -p "  Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing venv..."
        rm -rf "$VENV_PATH"
        echo "  Creating new virtual environment..."
        python3 -m venv "$VENV_PATH"
    else
        echo "  Using existing venv"
    fi
else
    echo "[2/8] Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi
echo ""

# Activate virtual environment
echo "[3/8] Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "  Active: $(which python)"
echo ""

# Upgrade pip
echo "[4/8] Upgrading pip..."
pip install --upgrade pip --quiet
echo "  pip version: $(pip --version)"
echo ""

# Install core dependencies first
echo "[5/8] Installing core dependencies..."
pip install --upgrade typing-extensions pydantic pydantic-core --quiet
echo "  Core dependencies installed"
echo ""

# Install all requirements
echo "[6/8] Installing project requirements..."
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
echo "  All requirements installed"
echo ""

# Verify installation
echo "[7/8] Verifying installation..."
python3 << 'PYEOF'
import sys
import os

# Add baselines to path
sys.path.insert(0, '/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines')

print("  Checking imports...")
try:
    from dotenv import load_dotenv
    print("    ✓ python-dotenv")

    from langchain_openai import ChatOpenAI
    print("    ✓ langchain-openai")

    from omegaconf import OmegaConf
    print("    ✓ omegaconf")

    import pandas as pd
    print("    ✓ pandas")

    from utilities.llm_provider import get_llm, LLM_REGISTRY
    print("    ✓ llm_provider")

    from utilities.llm_tracker import tracker
    print("    ✓ llm_tracker")

    # Check model configuration
    load_dotenv()
    print("\n  Checking model configuration...")

    for model_name in ["gpt-5-mini", "deepseek-v4-flash"]:
        if model_name in LLM_REGISTRY:
            info = LLM_REGISTRY[model_name]
            api_key_set = "SET" if info.api_key else "NOT SET"
            print(f"    ✓ {model_name}: {info.model_name} (API Key: {api_key_set})")
        else:
            print(f"    ✗ {model_name}: NOT CONFIGURED")

    print("\n  All imports successful!")

except ImportError as e:
    print(f"    ✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"    ✗ Error: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
else
    echo ""
    echo "  Verification failed! Check errors above."
    exit 1
fi

# Create necessary directories
echo "[8/8] Creating directories..."
mkdir -p "$PROJECT_ROOT/baselines/results"
mkdir -p "$PROJECT_ROOT/baselines/repo_cloned"
mkdir -p "$PROJECT_ROOT/baselines/changed_files"
mkdir -p "$PROJECT_ROOT/baselines/exceptions"
mkdir -p "$PROJECT_ROOT/miniswe-agent/baselines/results"
echo "  Directories created"
echo ""

# Summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual Environment: $VENV_PATH"
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To run baseline pipeline:"
echo "  cd $PROJECT_ROOT/baselines"
echo "  python main.py"
echo ""
echo "To run MiniSWE pipeline:"
echo "  cd $PROJECT_ROOT/baselines"
echo "  python run_miniswe_pipeline.py --model gpt-5-mini --analyzer llm --mode miniswe --num-instances 5"
echo ""
echo "For full documentation, see: SETUP_GUIDE.md"
echo "=========================================="
