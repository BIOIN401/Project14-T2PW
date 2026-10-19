"""ORCH-754c / D-104 amendment: the curator prompt no longer tells the curator to ADD interactions.

G9 label: CORRECTION of pre-existing, unauthorized behaviour. ORCH-747 (commit
``70d05e03``) added § 5 to ``pathway_curator_system.txt``, telling the curator to propose
``op=add at /processes/interactions/-`` (and nucleic_acids entities) bound to a verbatim
payload quote, and carved an exception into "Do not add new reactions or transports".
No decision authorized it, the curator is never sent the evidence strings it is told to
quote, and its patches land in ``audited_json`` before the freeze. The product owner ruled
REVERT. These tests FAIL at ``f9892cdf`` (the § 5 text is present) and PASS at the tip
(the file is the ``c9265fb8`` text again).

The hash is over the text the curator receives (``read_text``, LF): the blob is LF and the
working tree is CRLF only via ``core.autocrlf``, so a CRLF hash would not survive a clone.
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

CURATOR = PROMPTS_DIR / "pathway_curator_system.txt"

#: sha256 of ``git show c9265fb8:src/t2pw/llm/prompts/pathway_curator_system.txt`` (LF).
BASE_TEXT_SHA256 = "6462c35ed8131f557f9c5cc1af7c75d7f10911529897a744bb33b57e20896531"


def test_curator_prompt_gives_no_instruction_to_add_interactions() -> None:
    text = CURATOR.read_text(encoding="utf-8")
    assert "/processes/interactions/-" not in text
    assert "Interactions are the one exception" not in text


def test_curator_prompt_is_the_c9265fb8_text() -> None:
    text = CURATOR.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == BASE_TEXT_SHA256
