"""The decisive question: with ok=True, does the run still carry a refusal
of biological relevance somewhere a consumer can see it?"""
from __future__ import annotations
import json, sys
from copy import deepcopy
from pathlib import Path
TREE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(TREE / "src"))
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
print("MEASURED_TREE t2pw =", sys.modules["t2pw"].__file__)
cases = json.loads((TREE / "tests/fixtures/strict_failures/cases.json").read_text(encoding="utf-8"))["cases"]
for cid in ("only_unrelated_reactions_survive", "every_reaction_unresolvable",
            "reaction_inflation_unresolvable_participant"):
    c = next(x for x in cases if x["id"] == cid)
    r = quarantine_and_close(deepcopy(c["payload"]), strict_db=True)
    qr = r.quarantine_report
    print("\n" + "=" * 70)
    print(cid, "| ok =", r.ok, "| fixture recovers =", c["expect"]["recovers"])
    print("=" * 70)
    print("  quarantine_report keys:", json.dumps(sorted(qr.keys())))
    for k in ("review_reasons", "refusal_reasons", "would_have_refused"):
        if k in qr:
            print(f"  {k} = {json.dumps(qr[k])}")
    rel = qr.get("release")
    print("  release =", json.dumps(rel, indent=2, default=str) if rel is not None else "ABSENT")
