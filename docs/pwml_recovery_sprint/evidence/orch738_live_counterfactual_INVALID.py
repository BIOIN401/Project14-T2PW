"""INVALID -- DOES NOT RECONSTRUCT THE LIVE LEG. KEPT AS A DISCARDED ATTEMPT.

Driving merged_payload.json through normalize -> quarantine yields 0 surviving
reactions and quarantine ok=False for PMC4471609, where the LIVE leg had 6 and
True. merged_payload.json is the post-audit payload; the live pipeline does RAG
admission, gap resolution and identity mapping between there and quarantine, and
this script performs none of it.

Its numbers are NOT evidence of anything and must never be quoted. Committed so
the discarded attempt is visible rather than silently deleted. The real proof of
the live restoration needs no replay -- see C-121-LIVE-VALIDATION-RESULT.md s2.
"""

"""Would the LIVE C-121 legs have lost their state on the base code?

Drives the merged (post-audit, pre-quarantine) payload of each live leg through
quarantine_and_close in whichever tree is passed, and reports the surviving states.
Base tree -> tip tree is the counterfactual the live run cannot show by itself,
because the live run only ever executed one of the two.
"""
import json, sys
from pathlib import Path

TREE = Path(sys.argv[1])
RUN = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_validation/c121/2026-09-09_0028/papers")
sys.path.insert(0, str(TREE / "src"))
import t2pw
print("TREE:", t2pw.__file__)
assert str(TREE).lower() in t2pw.__file__.lower()

from t2pw.pipeline.process_normalizer import normalize_process_payload
from t2pw.pipeline.strict_quarantine import quarantine_and_close

for paper in ("PMC4471609", "PMC9544450"):
    src = RUN / paper / "strict" / "merged_payload.json"
    if not src.exists():
        print(f"\n### {paper}: no merged_payload.json"); continue
    payload = json.loads(src.read_text(encoding="utf-8"))
    normalized, _ = normalize_process_payload(payload, mode="pathwhiz")
    res = quarantine_and_close(normalized, strict_db=True, mode="pathwhiz")
    p = res.payload if hasattr(res, "payload") else {}
    states = sorted(str(r.get("name")) for r in (p.get("biological_states") or []) if isinstance(r, dict))
    el = p.get("element_locations") or {}
    missing = sum(1 for b, rows in el.items() if isinstance(rows, list)
                  for r in rows if isinstance(r, dict)
                  and not str(r.get("biological_state") or "").strip())
    pr = p.get("processes") or {}
    print(f"\n### {paper}")
    print(f"   surviving biological_states : {states}")
    print(f"   rows missing a state        : {missing}")
    print(f"   surviving reactions         : {len(pr.get('reactions') or [])}")
    print(f"   quarantine ok               : {getattr(res, 'ok', None)}")
