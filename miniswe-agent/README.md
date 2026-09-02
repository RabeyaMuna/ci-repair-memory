# Mini-SWE-Agent

Mini-SWE-Agent patch-generation baseline for CI-Repair-Bench. It consumes
predicted fault-localization documents and does not run a separate CI analyzer.

## Installation

```bash
cd miniswe-agent
pip install -e .
```

## Usage

### GPT-5-mini external-agent baseline with predicted fault localization

From this directory, install both `minisweagent` and its CI helper package:

```bash
python3 -m venv .venv-mini
source .venv-mini/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run all benchmark instances that have matching fault-localization data:

```bash
mini-extra cibench \
    --dataset ../dataset/ci_repair_dataset.parquet \
    --fault-localization ../baselines/results/gpt-5-mini_llm/fault_localization.json \
    --model gpt-5-mini \
    --model-class litellm \
    --workers 1 \
    --output ../baselines/results/miniswe-agent_gpt-5-mini_fl
```

Patches are written to `preds.json`; each instance trajectory is written to
`<output>/<sha_fail>/<sha_fail>.traj.json`. The exact matched FL inputs are
copied to `<output>/fault_localization_input.json`.

## Shared Resources

This agent uses shared resources from the parent directory:

- **Dataset**: `../dataset/ci_repair_dataset.parquet`
- **Fault localization**: `../baselines/results/gpt-5-mini_llm/fault_localization.json`
- **Results**: `../baselines/results/miniswe-agent_gpt-5-mini_fl/`
- **Testbed repos**: `../repo/` - Cloned test repositories

## Project Structure

```
miniswe-agent/
├── src/minisweagent/     # Source code
├── tests/                # Tests
├── .venv/                # Virtual environment
├── pyproject.toml        # Dependencies
└── README.md             # This file
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run linting
ruff check src/
```

## See Also

- [Project Root README](../README.md) - Multi-agent benchmark overview
- [Evaluation Scripts](../scripts/) - Analysis tools
 - Codex Runner (see project root README) - Codex CLI–based agent
