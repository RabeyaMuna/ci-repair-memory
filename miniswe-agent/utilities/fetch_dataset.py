#!/usr/bin/env python3
"""
fetch_dataset.py - CLI wrapper for utilities.dataset_fetcher
============================================================

This is a thin wrapper around utilities.dataset_fetcher for backward compatibility.
New code should import directly from utilities.dataset_fetcher.

Usage:
    python scripts/fetch_dataset.py --split train --out data/issues.jsonl
    python scripts/fetch_dataset.py --split train --repos "flower,agno" --out data/issues.jsonl
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utilities.dataset_fetcher import main

if __name__ == "__main__":
    main()
