"""Probe A — reconstruct the pinned run's `placeholder_backed_proteins` population.

CRITERIA, taken from production and fixed BEFORE counting:

  bench/semantic.py:_check_placeholder_identity (:1391-1403)
    for bucket in ("proteins", "protein_complexes"):
        for row in entities[bucket]:
            if identity_status(row) != IDENTITY_PLACEHOLDER: continue
            placeholder_backed += 1

  So the metric is EXACTLY: rows in entities.proteins UNION
  entities.protein_complexes whose pipeline.entity_identity.identity_status(row)
  == "placeholder".  Nothing about `generated`, nothing about wrappers.

PARTITION under test (the user's proposed 16/5), using production predicates only:
  P_SENTINEL  = placeholder AND is_pathbank_unknown_protein(row)
  P_FUNCTIONAL= placeholder AND NOT is_pathbank_unknown_protein(row)
These are exhaustive and mutually exclusive over the placeholder set by
construction (a boolean predicate partitions its domain); the probe asserts it
anyway rather than relying on the argument.
"""
import json, os, glob, sys

sys.path.insert(0, os.environ["T2PW_SRC"])
from t2pw.pipeline.entity_identity import (  # noqa: E402
    IDENTITY_PLACEHOLDER, IDENTITY_VERIFIED, IDENTITY_UNRESOLVED,
    identity_status, is_pathbank_unknown_protein, placeholder_claims_real_identity,
)

PINNED = "runs/2026-08-02_2130"

def rows_of(path):
    d = json.load(open(path, encoding="utf-8"))
    ents = d.get("entities") or {}
    for bucket in ("proteins", "protein_complexes"):
        for idx, row in enumerate(ents.get(bucket) or []):
            if isinstance(row, dict):
                yield bucket, idx, row

placeholders, census = [], {"verified": 0, "placeholder": 0, "unresolved": 0, "other": 0, "total": 0}
legs = sorted(glob.glob(f"{PINNED}/papers/*/*/final_mapped.json"))
print(f"PINNED RUN : {PINNED}")
print(f"legs with final_mapped.json : {len(legs)}")
print()

per_leg = {}
for path in legs:
    parts = path.replace("\\", "/").split("/")
    leg = f"{parts[-3]}/{parts[-2]}"
    n = 0
    for bucket, idx, row in rows_of(path):
        census["total"] += 1
        st = identity_status(row) or IDENTITY_UNRESOLVED
        census[st if st in census else "other"] = census.get(st, 0) + 1
        if st != IDENTITY_PLACEHOLDER:
            continue
        n += 1
        placeholders.append({
            "leg": leg, "path": path.replace("\\", "/"),
            "pointer": f"/entities/{bucket}/{idx}",
            "name": row.get("name"),
            "sentinel": bool(is_pathbank_unknown_protein(row)),
            "generated": bool(row.get("generated")),
            "generation_reason": row.get("generation_reason"),
            "claims_real": placeholder_claims_real_identity(row) or "",
            "species": row.get("species"),
            "species_id": row.get("species_id"),
            "uniprot": (row.get("mapped_ids") or {}).get("uniprot") or row.get("uniprot"),
            "pathbank_protein_id": row.get("pathbank_protein_id"),
            "chosen_rule": (row.get("mapping_meta") or {}).get("chosen_rule"),
        })
    if n:
        per_leg[leg] = n

print("=== IDENTITY CENSUS over both buckets, all pinned legs ===")
for k in ("total", "verified", "placeholder", "unresolved", "other"):
    print(f"  {k:12s} {census.get(k, 0)}")
print()
print(f"*** placeholder_backed_proteins (pinned run TOTAL) = {census['placeholder']} ***")
print()
print("=== PER LEG ===")
for leg, n in sorted(per_leg.items()):
    print(f"  {leg:42s} {n}")
print()

sent = [p for p in placeholders if p["sentinel"]]
func = [p for p in placeholders if not p["sentinel"]]
print("=== PROPOSED PARTITION ===")
print(f"  P_SENTINEL   (placeholder AND is_pathbank_unknown_protein) = {len(sent)}")
print(f"  P_FUNCTIONAL (placeholder AND NOT sentinel)                = {len(func)}")
print(f"  sum = {len(sent) + len(func)}   metric = {census['placeholder']}   "
      f"exhaustive={len(sent)+len(func) == census['placeholder']}")
ids_s = {(p["leg"], p["pointer"]) for p in sent}
ids_f = {(p["leg"], p["pointer"]) for p in func}
print(f"  mutually exclusive = {not (ids_s & ids_f)}  (overlap size {len(ids_s & ids_f)})")
print()

print("=== ROW-LEVEL MEMBERSHIP ===")
for label, group in (("SENTINEL", sent), ("FUNCTIONAL", func)):
    print(f"--- {label} ({len(group)}) ---")
    for p in group:
        print(f"  {p['leg']:38s} {p['pointer']:34s} {str(p['name'])[:34]:34s} "
              f"gen={str(p['generated']):5s} reason={str(p['generation_reason'])[:32]:32s} "
              f"uniprot={str(p['uniprot'])[:10]:10s} sp={str(p['species'])[:22]:22s} "
              f"claims_real={p['claims_real'] or '-'}")
print()

print("=== CROSSWALK to the current-corpus 31-wrapper census ===")
print("Current census population = generated Unknown-backed protein_complex rows over runs*/ (11 legs).")
print("Pinned population = identity_status==placeholder rows over the pinned run only.")
print("These are DIFFERENT DEFINITIONS. Overlap computed by (leg, name) where both apply:")
cur = set()
for path in sorted(glob.glob("runs*/**/final_mapped.json", recursive=True)):
    p = path.replace("\\", "/")
    parts = p.split("/")
    leg = f"{parts[-3]}/{parts[-2]}"
    run = parts[0] + "/" + parts[1]
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    for r in (d.get("entities") or {}).get("protein_complexes") or []:
        if not isinstance(r, dict) or not r.get("generated"):
            continue
        comps = r.get("components") or []
        if any(isinstance(c, dict) and (c.get("pathbank_protein_id") == 9659
               or str(c.get("name") or "").strip().lower() == "unknown") for c in comps):
            cur.add((run, leg, str(r.get("name"))))
print(f"  current-census wrapper rows (run, leg, name) : {len(cur)}")
pin_names = {(p["leg"], str(p["name"])) for p in placeholders}
cur_in_pinned_run = {(leg, name) for (run, leg, name) in cur if run == PINNED}
print(f"  of those, in the pinned run                  : {len(cur_in_pinned_run)}")
print(f"  intersection with pinned placeholder rows    : {len(cur_in_pinned_run & pin_names)}")
if cur_in_pinned_run & pin_names:
    for x in sorted(cur_in_pinned_run & pin_names):
        print(f"      {x[0]:38s} {x[1]}")
print()
json.dump({"census": census, "placeholders": placeholders,
           "sentinel": len(sent), "functional": len(func), "per_leg": per_leg},
          open(os.environ["OUT_JSON"], "w", encoding="utf-8"), indent=1)
print(f"detail written: {os.environ['OUT_JSON']}")
