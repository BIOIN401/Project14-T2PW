"""Stored payloads (97 exported runs): transports whose from/to is a placeholder or empty while elements_with_states
declares one state for that side (the D-W20b rule the normalizer applies since 2026-11-12)."""
import json, re
runs = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x["q"]]
recs = [json.loads(l) for l in open("C:/t/pwml-fix-work/replay/base/results.jsonl", encoding="utf-8")]
PH = re.compile(r"^_*auto_?state", re.I)
n = 0
for rec in recs:
    if not rec.get("pwml_written"):
        continue
    p = json.load(open(runs[rec["idx"]]["dir"] + "/final_mapped.json", encoding="utf-8"))
    states = {str(s.get("name") or "").strip().casefold() for s in p.get("biological_states") or [] if isinstance(s, dict)}
    for t in (p.get("processes") or {}).get("transports") or []:
        for side, fld in (("left", "from_biological_state"), ("right", "to_biological_state")):
            v = str(t.get(fld) or "").strip()
            if v and not PH.match(v):
                continue
            decl = {str(e.get("biological_state") or "").strip() for e in t.get("elements_with_states") or []
                    if isinstance(e, dict) and str(e.get("side", "")).strip().lower() == side and e.get("biological_state")}
            if len(decl) == 1:
                d = next(iter(decl))
                n += 1
                print(rec["idx"], t.get("name"), side, repr(v), "->", repr(d), "declared_state_exists" if d.casefold() in states else "NOT_A_STATE")
print("total", n)
