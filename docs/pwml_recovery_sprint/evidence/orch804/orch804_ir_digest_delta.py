"""ORCH-804: the exact delta the empty-state fix puts into the pinned IR digests
(GOLDEN in test_compound_resolution_extraction.py, R3 in test_pwml_ir_duplicate_row_refusal.py).

Reuses ``orch747_ir_digest_delta.py`` (same pinned legs, same DB-down configuration,
base tree from ``git archive``). The owned prefixes are the state keys ``resolve_state``
assigns, and what follows a moved process: its member locations, edges and
visualizations, plus the location/edge/visualization counts. Anything else moving,
or any change to errors, ok or stops, means exit 1.

Usage: python .../orch804/orch804_ir_digest_delta.py --rev 563d39db --out FILE
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import orch747_ir_digest_delta as delta  # noqa: E402

delta.OWNED_PREFIXES = (
    "/ir/processes/reactions/[]/biological_state_key",
    "/ir/processes/interactions/[]/biological_state_key",
    "/ir/processes/transports/[]/transporters/[]/biological_state_key",
    "/ir/processes/transports/[]/left_biological_state_key",
    "/ir/processes/transports/[]/right_biological_state_key",
    "/ir/locations",
    "/ir/edges",
    "/ir/process_visualizations",
    "/ir/protein_complex_visualizations",
    "/ir/bound_visualizations",
    "/report/counts/locations",
    "/report/counts/edges",
    "/report/counts/process_visualizations",
    "/report/counts/protein_complex_visualizations",
    "/report/counts/bound_visualizations",
    "/report/ir_validation/counts",
)
delta.DEFAULT_OUT = HERE / "orch804_ir_digest_delta.json"

if __name__ == "__main__":
    sys.exit(delta.main())
