#!/usr/bin/env python3
"""Wrapper to run debate_memory.execute with package imports resolved."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from debate_memory.execute import parse_args, main


if __name__ == "__main__":
    args = parse_args()
    main(args)

