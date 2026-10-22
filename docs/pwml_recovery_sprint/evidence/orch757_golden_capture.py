"""ORCH-757: measure ``GOLDEN`` for ``tests/test_orch757_llm_token_budgets.py``.

Runs every stage driver of that test through its faked transport and prints, per stage,
the sha256 of each request minus ``max_tokens``/``timeout``. Run it on the base tree to
pin the golden and on the tip to compare; no provider is called.

    PYTHONPATH=<tree>/src python orch757_golden_capture.py <tree> <scratch-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def main() -> int:
    tree, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    sys.path[:0] = [str(tree / "src"), str(tree / "tests")]
    import test_orch757_llm_token_budgets as t  # noqa: PLC0415

    scratch.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("T2PW_EXTRACTION_DRAWS", raising=False)  # what tests/conftest.py pins
        digests = t.request_digests(scratch, t.install_wire(mp))
    print(json.dumps(digests, indent=4, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
