"""ORCH-766: diff chunk_e_base.json and chunk_e_tip.json (stdlib only; prints only).

Every changed leg must differ ONLY in entities.species, and every removed row must be
unreferenced by every independent fact the base run recorded. Each removed row's
required-field errors at base are listed, so the 27 -> 19 row delta is accounted for
row by row.
"""
import json
import sys

base = {r["leg"]: r for r in json.load(open(sys.argv[1], encoding="utf-8"))}
tip = {r["leg"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))}
assert set(base) == set(tip)
problems, delta_rows = [], 0
for leg in sorted(tip):
    b, t = base[leg], tip[leg]
    if b["ok"] != t["ok"]:
        problems.append(f"{leg}: quarantine verdict moved")
    if not t["ok"] or t["result_hash"] == b["result_hash"]:
        if t.get("species_removed"):
            problems.append(f"{leg}: removal without a hash change")
        continue
    if t["result_hash_without_species"] != b["result_hash_without_species"]:
        problems.append(f"{leg}: something other than entities.species moved")
    removed = [r["name"] for r in t["species_removed"]]
    print(f"== {leg}")
    print(f"   stage3_after {b['stage3_after']} -> {t['stage3_after']}")
    print(f"   required codes: {len(b['required_codes'])} -> {len(t['required_codes'])}"
          f"  (base {sorted(set(b['required_codes']))}, tip {sorted(set(t['required_codes']))})")
    print(f"   base species rows: {[s['name'] for s in b['species']]}")
    delta_rows += len(b["required_codes"]) - len(t["required_codes"])
    for name in removed:
        facts = [s for s in b["species"] if s["name"] == name]
        for f in facts:
            print(f"   REMOVED {name!r} tax={f['taxonomy_id']} ir_index={f['ir_index']} species0={f['ir_is_species0']}"
                  f" states={f['ir_state_refs']} proteins={f['writer_protein_refs']} name_hits={f['name_hits']}"
                  f" id_hits_species_path={f['id_hits_species_path']} id_hits_other_path={f['id_hits_other_path']}")
            if (f["name_hits"] or f["id_hits_species_path"] or f["ir_state_refs"] or f["writer_protein_refs"]
                    or f["ir_is_species0"] or f["ir_index"] is None):
                problems.append(f"{leg}: {name!r} is not provably unreferenced")
        if len(facts) != 1:
            problems.append(f"{leg}: {len(facts)} base rows named {name!r}")
print(f"\nrequired-field error rows removed: {delta_rows}")
print("PROBLEMS:", problems if problems else "none")
print("COMPARE DONE")
