"""ORCH-766: which Chunk E cohort rows move, and why the residual pin moves 27 -> 19.

    python -u chunk_e_species_delta.py <out.json>     (bounded_run.py, PYTHONPATH=<tree>/src)

Replays every leg of the frozen Chunk E cohort (``tests/data/baseline_cohort_manifest.json``)
exactly as ``test_the_full_stack_baseline_is_exactly_what_was_reported`` does -- Stage 3
normalization, ``quarantine_and_close(strict_db=True)``, the required-field gate run
the exporter's way -- using the test module's own helpers. For each admitted leg it
records the species rows the quarantine removed, the required-field codes by row,
and, for every species row of the quarantined payload, the same independent
reference facts ``census_species.py`` records (name / id leaf hits, IR state refs,
writer protein refs, IR species[0]; the IR built after ``prefrozen_when_compounded``,
as ``_ir_codes`` does). Run on base and on tip; prints a summary.
"""
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[3]
sys.path.insert(0, str(TREE / "tests"))
sys.path.insert(0, str(HERE))

import census_species as C  # noqa: E402
from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
import test_strict_quarantine_real_artifact_replay as E  # noqa: E402
from t2pw.pipeline.strict_quarantine import admitted_payload_hash, quarantine_and_close  # noqa: E402

records = []
for leg in E._legs():
    _source, raw = E._payload_for(leg)
    normalized = E._normalized(raw)
    result = quarantine_and_close(deepcopy(normalized), strict_db=True)
    row = {"leg": leg.relative_to(TREE).as_posix(), "ok": result.ok}
    if result.ok:
        out = result.payload
        universe = {}
        for p, v in C.leaves(out, "", {"/entities/species"}):
            for k in C.keys_of(v):
                universe.setdefault(k, []).append(p)
        # The IR is built the way the Chunk E test builds it (``_ir_codes``): these
        # legs predate pre-freeze compound resolution, so they go through it first.
        try:
            facts = C.ir_facts(prefrozen_when_compounded(deepcopy(out)), True)
        except AssertionError as exc:
            facts = {"error": f"prefrozen: {exc}"[:300]}
        species = []
        for idx, sp in enumerate(r for r in (out.get("entities") or {}).get("species") or [] if isinstance(r, dict)):
            names, ids = C.row_keys(sp)
            id_hits = sorted({p for k in ids for p in universe.get(k, [])})
            i = C.ir_index_of(sp, facts) if "error" not in facts else None
            species.append({
                "index": idx, "name": sp.get("name"), "taxonomy_id": sp.get("taxonomy_id"),
                "name_hits": sorted({p for k in names for p in universe.get(k, [])})[:12],
                "id_hits_species_path": [p for p in id_hits if C.species_path(p)][:12],
                "id_hits_other_path": [p for p in id_hits if not C.species_path(p)][:12],
                "ir_index": i, "ir_is_species0": i == 0,
                "ir_state_refs": (facts.get("state_refs") or {}).get(i, []) if i is not None else [],
                "writer_protein_refs": (facts.get("protein_refs") or {}).get(i, []) if i is not None else [],
            })
        stage3 = E._gate_reasons(out)
        row.update({
            "result_hash": admitted_payload_hash(out),
            "result_hash_without_species": admitted_payload_hash(
                {**out, "entities": {k: v for k, v in (out.get("entities") or {}).items() if k != "species"}}),
            "species_removed": [r for r in result.removed_entity_report["removed_entities"] if r["bucket"] == "species"],
            "stage3_after": "fail" if stage3 else "pass",
            "required_codes": E._required_contract_codes(out) if not stage3 else [],
            "ir_error": facts.get("error"),
            "species": species,
        })
    records.append(row)

Path(sys.argv[1]).write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
admitted = [r for r in records if r["ok"]]
by_row = Counter(c for r in admitted for c in r["required_codes"])
by_leg = Counter(c for r in admitted for c in set(r["required_codes"]))
print(f"legs {len(records)}, admitted {len(admitted)}")
print(f"required_contract_pass {sum(1 for r in admitted if r['stage3_after'] == 'pass' and not r['required_codes'])}")
print(f"by_leg {dict(by_leg)}")
print(f"by_row {dict(by_row)}")
print(f"species rows removed {sum(len(r['species_removed']) for r in admitted)} "
      f"in {sum(1 for r in admitted if r['species_removed'])} legs")
print("DELTA DONE")
