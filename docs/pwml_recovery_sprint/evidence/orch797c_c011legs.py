"""ORCH-797c: the c011 seam document's per-leg digests under a chosen source tree (diff two runs).

Usage: python orch797c_c011legs.py --src <tree>/src --out <file.json>
Imports ``t2pw`` from --src BEFORE the c011 test module (which would otherwise put the worktree's src first), builds
``_document(None)`` over the 39 cohort legs, and writes per leg: canonical_payload_sha256, the gate report's
canonical_graph_sha256 (the exporters' hash: the biology) and connectivity.
"""

import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402

print("t2pw resolved:", Path(t2pw.__file__).resolve().parent)
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "tests"))
import test_c011_freeze_seam_golden_equivalence as C  # noqa: E402

doc = C._document(None)
out = {name: {"payload": leg.get("canonical_payload_sha256"),
              "graph": (leg.get("final_stage3_gate_report") or {}).get("canonical_graph_sha256"),
              "connectivity": (leg.get("final_stage3_gate_report") or {}).get("connectivity")}
       for name, leg in sorted(doc["legs"].items())}
Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
print("written", args.out, len(out))
