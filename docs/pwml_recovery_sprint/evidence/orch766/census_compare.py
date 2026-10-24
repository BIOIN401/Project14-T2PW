"""ORCH-766: diff the base and tip censuses (stdlib only; prints, writes nothing).

    python -I census_compare.py <census_base.json[.gz]> <census_tip.json[.gz]>

For every leg: did the quarantined payload move, and if so, is the ONLY change the
species rows the tip removed? For every removed row: the base-run reference facts,
which must all say "unreferenced" -- no name equal to any scalar of the quarantined
payload, no id equal to a scalar under a species/taxonomy/organism path, no IR state
resolved to it, no protein the writer resolves to it, not IR species[0].
"""
import gzip
import json
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return {r["leg"]: r for r in json.load(fh)["records"]}


base, tip = load(sys.argv[1]), load(sys.argv[2])
assert set(base) == set(tip), "the two runs replayed different legs"

problems = []
changed = []
for leg in sorted(tip):
    b, t = base[leg], tip[leg]
    removed = [r["name"] for r in t["species_removed"]]
    if b["species_removed"]:
        problems.append(f"{leg}: base removed species rows")
    if t["result_hash"] == b["result_hash"]:
        if removed:
            problems.append(f"{leg}: species removed but result hash unchanged")
        continue
    changed.append(leg)
    if not removed:
        problems.append(f"{leg}: result moved with no species removal")
    if t["result_hash_without_species"] != b["result_hash_without_species"]:
        problems.append(f"{leg}: something other than entities.species moved")
    if [n for n in b["species_out"] if n not in removed] != t["species_out"]:
        problems.append(f"{leg}: species_out is not base minus the removed rows")
    print(f"== {leg} [{t['export_mode']}]")
    for key in ("ok", "release_status", "refusal_reasons", "gate_ok", "gate_error_codes", "closure_iterations"):
        flag = "" if b[key] == t[key] else "   <-- moved"
        print(f"   {key}: {b[key]} -> {t[key]}{flag}")
    print(f"   base species rows: {b['species_out']}")
    for row in t["species_removed"]:
        facts = [f for f in b["species_facts"] if f["name"] == row["name"]]
        print(f"   REMOVED {row['name']!r} ({row['reason']}, iteration {row['iteration']})")
        for f in facts:
            print(f"     base facts: tax={f['taxonomy_id']} class={f['classification']} ir_index={f['ir_index']}"
                  f" species0={f['ir_is_species0']} ir_state_refs={f['ir_state_refs']}"
                  f" writer_protein_refs={f['writer_protein_refs']}")
            print(f"       name_hits={f['name_hits']} id_hits_species_path={f['id_hits_species_path']}"
                  f" id_hits_other_path={f['id_hits_other_path']}")
            if (f["name_hits"] or f["id_hits_species_path"] or f["ir_state_refs"]
                    or f["writer_protein_refs"] or f["ir_is_species0"] or f["ir_index"] is None):
                problems.append(f"{leg}: removed row {row['name']!r} is not provably unreferenced")
        if len(facts) != 1:
            problems.append(f"{leg}: {len(facts)} base rows named {row['name']!r}")

unchanged = len(tip) - len(changed)
print(f"\nlegs compared: {len(tip)}; changed: {len(changed)}; byte-identical quarantine output: {unchanged}")
print(f"species rows removed: {sum(len(tip[l]['species_removed']) for l in changed)}")
print("PROBLEMS:", problems if problems else "none")
print("COMPARE DONE")
