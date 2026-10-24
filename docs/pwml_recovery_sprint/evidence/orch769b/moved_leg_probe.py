"""ORCH-769b: what the tip quarantines on the two runs_verify PMC12856317 legs the corpus delta names.

    python moved_leg_probe.py <tip tree>/src

Replays quarantine_and_close on each leg's own final_mapped.json with its own requested_context, at the
tree given, and prints every non-accepted admission, the surviving process counts, and the reaction rows
of what was quarantined. Read-only on the legs; imports t2pw from the tree given.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import t2pw  # noqa: E402
from t2pw.pipeline.strict_quarantine import ACCEPTED_STATES, quarantine_and_close  # noqa: E402

print("t2pw:", t2pw.__file__)
PRIMARY = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
for run in ("2026-08-22_2017", "2026-08-24_1203"):
    leg = PRIMARY / "runs_verify" / run / "papers" / "PMC12856317" / "strict"
    fm = json.loads((leg / "final_mapped.json").read_text(encoding="utf-8"))
    ctx = (json.loads((leg / "quarantine_report.json").read_text(encoding="utf-8")).get("coverage") or {}).get(
        "requested_context")
    res = quarantine_and_close(fm, strict_db=True, pathway_context=ctx, mode="pathwhiz")
    print(f"\n== runs_verify/{run}/PMC12856317  pathway: {(ctx or {}).get('pathway_name')!r}")
    print("  input processes:", {b: len(v) for b, v in (fm.get("processes") or {}).items() if isinstance(v, list)})
    for a in res.admissions:
        print(f"  {a['pointer']:32s} {a['state']:30s} {a['reason']:40s} {a['essential_participant']!r}  {a['name']!r}")
    print("  surviving processes:",
          {b: len(v) for b, v in (res.payload.get("processes") or {}).items() if isinstance(v, list)})
    for a in res.admissions:
        if a["state"] not in ACCEPTED_STATES:
            r = fm["processes"][a["bucket"]][a["index"]]
            keep = {k: r.get(k) for k in ("name", "inputs", "outputs", "enzymes", "modifiers")}
            print("  quarantined row:", json.dumps(keep, ensure_ascii=False)[:700])
    ents = fm.get("entities") or {}
    for b in ("compounds", "proteins", "protein_complexes"):
        hits = [row.get("name") for row in ents.get(b) or [] if isinstance(row, dict)
                and "pyridoxal" in json.dumps(row).lower()]
        print(f"  '{b}' rows mentioning pyridoxal: {hits}")
