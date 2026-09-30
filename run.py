#!/usr/bin/env python3
"""Entry point for the triage prototype: `python run.py --help`.

Stdlib only -- no pip install step, no model download, no API key required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
