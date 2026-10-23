"""Stdlib-only: for one leg, show who references each species row and what
taxonomy ids the protein lookups carried (ORCH-762 FX2 query scope)."""
import json
import sys

leg = sys.argv[1]
for fname in ("stage1_payload.json", "merged_payload.json", "final_mapped.json"):
    d = json.load(open(f"{leg}/{fname}", encoding="utf-8"))
    ents = d.get("entities", {})
    names = [r.get("name") for r in ents.get("species", []) if isinstance(r, dict)]
    print("==", fname, "species rows:", names)
    refs = {n: [] for n in names}
    for sec in ("proteins", "protein_complexes", "compounds", "nucleic_acids"):
        for i, r in enumerate(ents.get(sec, []) or []):
            if not isinstance(r, dict):
                continue
            for key in ("species", "organism"):
                v = r.get(key)
                if isinstance(v, str) and v in refs:
                    refs[v].append(f"{sec}/{i}.{key}")
            sref = r.get("species_ref")
            if isinstance(sref, dict) and sref.get("name") in refs:
                refs[sref["name"]].append(f"{sec}/{i}.species_ref")
    for i, b in enumerate(d.get("biological_states", []) or []):
        if isinstance(b, dict) and b.get("species") in refs:
            refs[b["species"]].append(f"biological_states/{i}")
    for n, r in refs.items():
        print("   ", repr(n), "referenced by", len(r), r[:6])
    if fname == "final_mapped.json":
        for i, r in enumerate(ents.get("proteins", []) or []):
            mm = r.get("mapping_meta") or {}
            q = mm.get("query") or {}
            sref = r.get("species_ref") or {}
            print("   protein", i, repr(r.get("name")), "species=", repr(r.get("species")),
                  "species_ref.tax=", sref.get("taxonomy_id"), "query.taxonomy_id=", q.get("taxonomy_id"),
                  "uniprot=", r.get("uniprot") or (r.get("mapped_ids") or {}).get("uniprot"))
        pdr = d.get("prefreeze_db_resolution")
        if isinstance(pdr, dict):
            print("   prefreeze_db_resolution keys:", list(pdr.keys()))
            sp = pdr.get("species") or {}
            print("   prefreeze species summary:", json.dumps(sp)[:1500])
