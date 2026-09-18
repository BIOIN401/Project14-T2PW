"""``python -m t2pw.mining ...`` entry point."""

from __future__ import annotations

import sys

from t2pw.mining.cli import main

if __name__ == "__main__":
    sys.exit(main())
