"""ORCH-754c / D-104: the production Stage-1 prompt IS the measured control ``r2_01_control_v03``.

G9 label: DELIBERATE BEHAVIOUR CHANGE, not a correction and not new functionality.
Every scored extraction since ORCH-747 round 2 ran ``r2_01_control_v03``; production
``pwml_system.txt`` was still ``v02_working_tree_resolution``. These tests FAIL at base
``f9892cdf`` (production = v02) and PASS at the tip (production = the measured file).

Hashes are taken over the text the model actually receives -- ``read_text`` with
universal newlines, so LF -- because the blob is LF and the working tree is CRLF only
via ``core.autocrlf`` (``git ls-files --eol``: ``i/lf w/crlf``). On this machine the
CRLF bytes hash to ``bc716123774c3d8b...``; that value does not survive a clone with
``core.autocrlf=false``, the LF text hash does.

A future prompt edit that fails ``test_production_prompt_text_is_pinned`` must move the
pin deliberately, with the old -> new delta documented (merge rule 4), and should name
the measurement that justifies the new prompt.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.paths import PROMPTS_DIR  # noqa: E402

PRODUCTION = PROMPTS_DIR / "pwml_system.txt"
MEASURED = ROOT / "docs" / "evaluation_v2" / "prompt_variants_r2" / "r2_01_control_v03.txt"

#: sha256 of r2_01_control_v03 as the model receives it (LF, UTF-8, 580 lines).
MEASURED_TEXT_SHA256 = "27f5e8732d77083313cab590dd6259ffe3828816647b0795699a0da6a68b8156"


def test_production_prompt_is_the_measured_control_byte_for_byte() -> None:
    assert PRODUCTION.read_bytes() == MEASURED.read_bytes()


def test_production_prompt_text_is_pinned() -> None:
    text = PRODUCTION.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == MEASURED_TEXT_SHA256
