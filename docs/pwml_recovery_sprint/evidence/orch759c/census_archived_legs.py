"""ORCH-759c blast radius: does the bucket-aware closure change any archived strict leg?

For every ``<root>/**/strict/final_mapped.json`` given on the command line, runs
``quarantine_and_close`` twice in one process -- the code on PYTHONPATH (tip), and a
BASE EMULATION that hands ``_prune_entities`` the old single union set -- with the
leg's own Stage-0 context (its ``quarantine_report.json``, when present) and
``strict_db=True``, ``mode="pathwhiz"``. The emulation is the base rule exactly
(base passed ``referenced | component_norms`` as one set; the union of the tip's
mapping is that set) and it reproduced the real base hash on the chlorophyll leg
(``g9_leg_replay.py``). Research legs are skipped: research mode applies nothing.

Prints one line per leg whose result differs, then a summary. Read-only.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import t2pw.pipeline.strict_quarantine as sq

real_prune = sq._prune_entities


def union_prune(p, keep):
    if isinstance(keep, dict):
        keep = set().union(*keep.values())
    return real_prune(p, keep)


def run(payload, ctx, emulate_base):
    sq._prune_entities = union_prune if emulate_base else real_prune
    try:
        return sq.quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=ctx, mode="pathwhiz")
    finally:
        sq._prune_entities = real_prune


legs = sorted({p for root in sys.argv[1:] for p in Path(root).rglob("final_mapped.json") if p.parent.name == "strict"})
same = differ = errors = admitted = 0
flips = []
for leg in legs:
    try:
        payload = json.loads(leg.read_text(encoding="utf-8"))
        q = leg.parent / "quarantine_report.json"
        ctx = None
        if q.is_file():
            ctx = (json.loads(q.read_text(encoding="utf-8")).get("coverage") or {}).get("requested_context")
        base, tip = run(payload, ctx, True), run(payload, ctx, False)
    except Exception as exc:  # report, never hide
        errors += 1
        print(f"ERROR {leg}: {type(exc).__name__}: {exc}")
        continue
    admitted += sum(1 for a in tip.admissions if a["state"] in sq.ACCEPTED_STATES)
    bh, th = base.quarantine_report["resulting_payload_hash"], tip.quarantine_report["resulting_payload_hash"]
    if bh == th and base.ok == tip.ok and base.refusal_reasons == tip.refusal_reasons:
        same += 1
        continue
    differ += 1
    rb = {(e["bucket"], e["name"]) for e in base.removed_entity_report["removed_entities"]}
    rt = {(e["bucket"], e["name"]) for e in tip.removed_entity_report["removed_entities"]}
    rel = lambda r: (r.quarantine_report.get("release") or {}).get("status")
    print(f"DIFF {leg}\n   base ok={base.ok} {base.refusal_reasons} {rel(base)} | tip ok={tip.ok} {tip.refusal_reasons} {rel(tip)}"
          f"\n   removed only at tip={sorted(rt - rb)} only at base={sorted(rb - rt)}"
          f"\n   admissions identical={sorted((a['pointer'], a['state']) for a in base.admissions) == sorted((a['pointer'], a['state']) for a in tip.admissions)}")
    if base.ok != tip.ok:
        flips.append((str(leg), base.ok, tip.ok))
print(f"SUMMARY legs={len(legs)} accepted_processes_at_tip={admitted} same={same} differ={differ} errors={errors} ok_flips={flips}")
