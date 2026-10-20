"""ORCH-754b -- the exact delta this card puts into the two pinned IR digests.

Same pinned legs, same DB-down configuration and same base-vs-tip machinery as
``orch747_ir_digest_delta.py`` (imported, not copied), with two things changed:

* the base is ``f9892cdf`` (this card's base), so the delta is this card's alone; the
  cumulative ``c9265fb8`` -> tip move, which is what the re-pinned values encode, is
  ``orch754b_cumulative_repin.py``;
* the owned prefixes are what this card may move on a committed, already-mapped leg: the
  ``withheld`` record of an interaction (its ``reason``, now naming a missing species or
  identity; the new ``unwrappable_proteins`` list; the new ``evidence``) and the matching
  warning text -- plus, for the one committed ``competes_with`` row, its ``interaction_type``
  and anything ORCH-747 drew for it (measured: it was already withheld for a bare protein
  endpoint, so only the type word moved). The committed
  legs carry no interaction wrapper (Stage 6 never ran with this code), so no complex moves.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch754b_ir_digest_delta.py [--rev f9892cdf] [--out FILE]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import orch747_ir_digest_delta as delta  # noqa: E402

delta.OWNED_PREFIXES = (
    "/ir/processes/interactions/[]/withheld",
    "/ir/processes/interactions/[]/export_status",      # competes_with: exportable -> withheld
    "/ir/processes/interactions/[]/interaction_type",   # competes_with: Inhibition -> the word
    "/ir/locations",
    "/ir/edges",
    "/ir/process_visualizations",
    "/report/warnings",
    "/report/counts/locations",
    "/report/counts/edges",
    "/report/counts/process_visualizations",
    "/report/ir_validation/counts",
)
delta.DEFAULT_OUT = HERE / "orch754b_ir_digest_delta.json"

if __name__ == "__main__":
    if "--rev" not in sys.argv:
        sys.argv[1:1] = ["--rev", "f9892cdf"]
    sys.exit(delta.main())
