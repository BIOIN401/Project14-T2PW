"""Which proteins block the 25 replay rows: offline Stage-6 replay + prefreeze + IR, tip only."""
import json
import logging
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path

logging.disable(logging.CRITICAL)
from t2pw.mapping.map_ids import MappingCache, _extract_global_organism, _rewrite_reaction_protein_enzymes_to_complexes
from t2pw.pwml.ir import build_pwml_ir
from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution


class _DownDb:
    last_error = "harvest_db_down"

    def available(self):
        return False


legs = [p for p in subprocess.run(["git", "ls-files", "runs", "runs_verify"], capture_output=True, text=True).stdout.split("\n")
        if p.endswith("/final_mapped.json")]
names = Counter()
for leg in legs:
    payload = json.loads(Path(leg).read_text(encoding="utf-8"))
    if not (payload.get("processes") or {}).get("interactions"):
        continue
    m = deepcopy(payload)
    _rewrite_reaction_protein_enzymes_to_complexes(m, db=None, cache=MappingCache(Path("unused.json"), enabled=False),
                                                   global_organism=_extract_global_organism(m))
    try:
        run_prefreeze_resolution(m, strict_db=True, db_resolver=_DownDb(), name_index=None)
        ir, _ = build_pwml_ir(m, db_resolver=_DownDb(), strict_db=True, name_index=None)
    except Exception:
        continue
    for row in ir["processes"]["interactions"]:
        for u in (row.get("withheld") or {}).get("unwrappable_proteins") or []:
            names[(leg.split("/papers/")[1].split("/")[0][:11], u["name"], tuple(u["missing"]))] += 1
for k, v in sorted(names.items(), key=lambda kv: -kv[1]):
    print(v, k)
print("rows-with-unwrappable-protein-endpoints (endpoint count):", sum(names.values()))
