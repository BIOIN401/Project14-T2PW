"""C-121 G4 -- ``evaluate_reaction_support``'s verdict on every archived leg.

``reaction_support.py`` must end byte-identical (F-179), and C-121 must not alter
which reactions are supported, admitted or scored. Byte-identity of the module is
necessary and not sufficient: the verdict is a function of the module AND of what
the rest of the pipeline hands it. So this measures the verdict itself.

Run it once with ``--tree`` pointing at the candidate and once at a materialized
base tree, then compare the two digest files byte for byte. A single differing
line is a moved F-179 verdict and blocks the merge.

    <py> c121_f179_verdicts.py <digest-out.json> [--tree <import root>]
                              [--repo <archive>] [--run runs_smoke/2026-09-08_1528]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_TREE = HERE.parents[3]
DEFAULT_REPO = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tree", default=str(DEFAULT_TREE))
ap.add_argument("--repo", default=str(DEFAULT_REPO))
ap.add_argument("--run", default="runs_smoke/2026-09-08_1528")
args = ap.parse_args()

TREE = Path(args.tree).resolve()
REPO = Path(args.repo).resolve()
OUT = Path(args.out).resolve()
sys.path.insert(0, str(TREE / "src"))
import t2pw  # noqa: E402

print("EXPECTED tree :", TREE.as_posix())
print("RESOLVED t2pw :", Path(t2pw.__file__).as_posix())
if Path(t2pw.__file__).resolve().parent != (TREE / "src" / "t2pw").resolve():
    print("MEASUREMENT_TREE_REFUSED")
    raise SystemExit(98)

from t2pw.pipeline.reaction_support import (  # noqa: E402
    evaluate_reaction_support,
    lineage_carrier_active,
    reaction_support_issue,
)

try:  # absent on the base tree, by construction
    from t2pw.pipeline.process_normalizer import (  # noqa: E402
        restore_autostates_if_required,
    )
except ImportError:
    restore_autostates_if_required = None
print("restoration API present:", restore_autostates_if_required is not None)

module = (TREE / "src" / "t2pw" / "pipeline" / "reaction_support.py").read_bytes()
print("reaction_support.py sha256:", hashlib.sha256(module).hexdigest())

root = REPO / args.run
legs = sorted(p for p in root.glob("papers/*/strict/final_mapped.json"))
print("legs:", len(legs), "under", args.run)

#: The base-comparable record: only fields BOTH trees can produce, so the digest
#: file can be compared byte for byte between a base tree and the candidate. The
#: restoration-invariance measurement is reported separately below and is
#: deliberately NOT in this dict -- the base tree has no restoration to measure,
#: and putting it here would make the two files differ for a reason that is not a
#: moved verdict.
records = {}
invariance = {}
for fm in legs:
    payload = json.loads(fm.read_text(encoding="utf-8"))
    leg = fm.parts[-3]
    entry = {
        "carrier_active": lineage_carrier_active(payload),
        "verdict": evaluate_reaction_support(deepcopy(payload)),
        "issue": reaction_support_issue(deepcopy(payload)),
    }
    records[leg] = entry
    note = ""
    if restore_autostates_if_required is not None:
        # The same verdict, measured AFTER the C-121 restoration would have run. If
        # the placeholder changed anything about reaction support, it shows here.
        restored = deepcopy(payload)
        acted = restore_autostates_if_required(restored)
        after_verdict = evaluate_reaction_support(deepcopy(restored))
        after_issue = reaction_support_issue(deepcopy(restored))
        inv = {
            "restoration_acted": acted,
            "verdict_unmoved_by_restoration": (
                json.dumps(entry["verdict"], sort_keys=True, default=str)
                == json.dumps(after_verdict, sort_keys=True, default=str)
            ),
            "issue_unmoved_by_restoration": (
                json.dumps(entry["issue"], sort_keys=True, default=str)
                == json.dumps(after_issue, sort_keys=True, default=str)
            ),
        }
        invariance[leg] = inv
        note = (" acted=" + str(inv["restoration_acted"])
                + " verdict_unmoved=" + str(inv["verdict_unmoved_by_restoration"])
                + " issue_unmoved=" + str(inv["issue_unmoved_by_restoration"]))
    print("  " + leg.ljust(16) + " carrier=" + str(entry["carrier_active"]) + note)

# The module hash is deliberately NOT in the digest: this file is the VERDICT
# comparison, and including the hash would make two trees differ for a reason the
# git diff already proves or disproves.
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(records, sort_keys=True, indent=2, default=str), encoding="utf-8"
)
print("")
print("digest file  :", OUT.as_posix())
print("digest sha256:", hashlib.sha256(OUT.read_bytes()).hexdigest())
print("legs measured:", len(records))
if invariance:
    moved = [k for k, v in invariance.items()
             if not (v["verdict_unmoved_by_restoration"]
                     and v["issue_unmoved_by_restoration"])]
    fired = [k for k, v in invariance.items() if v["restoration_acted"]]
    print("restoration fired on:", len(fired), fired)
    print("legs whose F-179 verdict the restoration moved:", len(moved), moved)
else:
    print("restoration-invariance: NOT MEASURABLE on this tree (no restoration API)")
print("F179 VERDICTS DONE")
