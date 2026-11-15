"""A4: payload reactions whose evidence cites reversibility, and any direction-like keys anywhere in reactions."""
import json, re
from pathlib import Path

ROOT = Path("C:/t/corr-assess")
sample = json.loads((ROOT / "sample.json").read_text(encoding="utf-8"))
rows, keys = [], set()
for s in sample:
    d = json.loads((ROOT / "base" / str(s["idx"]) / "final_payload.json").read_text(encoding="utf-8"))
    for i, r in enumerate(d["processes"]["reactions"]):
        for k in r:
            if re.search(r"revers|direction", k, re.I):
                keys.add(k)
        blob = json.dumps([r.get("evidence"), [e.get("evidence") for e in r.get("enzymes") or []]], ensure_ascii=False)
        if re.search(r"revers|⇌|<->|<=>|both forward and reverse", blob, re.I):
            rows.append({"idx": s["idx"], "r": i, "rxn": f"{' + '.join(r['inputs'])} -> {' + '.join(r['outputs'])}",
                         "evidence": (r.get("evidence") or "")[:200]})
print(json.dumps({"direction_keys_in_payload_reactions": sorted(keys), "reversibility_cited": rows}, indent=1, ensure_ascii=False))
