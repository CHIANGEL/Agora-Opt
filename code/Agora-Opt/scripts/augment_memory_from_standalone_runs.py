#!/usr/bin/env python3
"""Wrapper for debate_memory.augment_memory_from_standalone_runs."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from debate_memory.augment_memory_from_standalone_runs import main


if __name__ == "__main__":
    main()
