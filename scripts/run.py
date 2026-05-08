from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t2pw.pipeline.pipeline import legacy_sbml_cli_main  # noqa: E402


if __name__ == "__main__":
    legacy_sbml_cli_main()
