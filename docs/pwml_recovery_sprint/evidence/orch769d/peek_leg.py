import json, sys
from pathlib import Path
LEG = Path(sys.argv[1])
for fn in ("stage1_payload.json", "merged_payload.json", "final_mapped.json"):
    d = json.loads((LEG / fn).read_text(encoding="utf-8"))
    ents = d.get("entities") or {}
    print("==", fn, "top keys:", sorted(d.keys())[:30])
    for b in ("proteins", "protein_complexes"):
        rows = ents.get(b) or []
        print("  --", b, len(rows))
        for i, r in enumerate(rows):
            if isinstance(r, dict):
                print("   ", i, ascii(r.get("name")), "syn=", ascii(r.get("synonyms")), "ids=", ascii(r.get("mapped_ids"))[:80])
