"""Compatibility entry point for the paired epsilon sweep."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv.insert(1, "epsilon")
    runpy.run_path(str(Path(__file__).with_name("sweep.py")), run_name="__main__")
