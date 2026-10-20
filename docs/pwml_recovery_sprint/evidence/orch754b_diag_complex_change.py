"""Diagnostic: which pre-existing complex row the tip's interaction-endpoint replay changes, and how."""
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

logging.disable(logging.CRITICAL)
from t2pw.mapping.map_ids import MappingCache, _extract_global_organism, _rewrite_reaction_protein_enzymes_to_complexes

leg = sys.argv[1]
payload = json.loads(Path(leg).read_text(encoding="utf-8"))
before = deepcopy(payload["entities"].get("protein_complexes") or [])
report = _rewrite_reaction_protein_enzymes_to_complexes(
    payload, db=None, cache=MappingCache(Path("unused.json"), enabled=False),
    global_organism=_extract_global_organism(payload))
after = payload["entities"]["protein_complexes"]
for i, (b, a) in enumerate(zip(before, after)):
    if a != b:
        print("ROW", i, b.get("name"))
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                print("  ", k, "\n     before:", json.dumps(b.get(k))[:400], "\n     after: ", json.dumps(a.get(k))[:400])
print("interaction actions:", [x for x in report["actions"] if x["type"].startswith("interaction_")])
print("interactions:", [(r.get("entity_1"), r.get("relationship"), r.get("entity_2")) for r in payload["processes"]["interactions"]])
