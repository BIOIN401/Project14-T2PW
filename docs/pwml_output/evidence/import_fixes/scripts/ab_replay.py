"""Replay A/B: base vs after PWML for every exported idx. Counts per section (ab97.py's), names lost/gained per entity
kind, reaction signatures (left names, right names, direction) lost/gained, transports (cargo, left loc, right loc) and
nucleic-acid types. usage: ab_replay.py BASE AFTER OUT.json [--dir replay|show]"""
import collections, filecmp, json, os, sys
import xml.etree.ElementTree as ET

BASE, AFTER, OUT = sys.argv[1:4]
SECS = {"reactions": "reactions", "compounds": "compounds", "proteins": "proteins", "complexes": "protein-complexes",
        "nucleic_acids": "nucleic-acids", "element_collections": "element-collections", "transports": "transports",
        "biological_states": "biological-states", "locations": "subcellular-locations", "compound_locations": "compound-locations",
        "protein_locations": "protein-locations", "nucleic_acid_locations": "nucleic-acid-locations", "interactions": "interactions",
        "transport_visualizations": "transport-visualizations", "membranes": "membrane-visualizations", "labels": "label-locations"}
KIND = {"Compound": "compounds", "Protein": "proteins", "ProteinComplex": "protein-complexes", "NucleicAcid": "nucleic-acids",
        "ElementCollection": "element-collections"}


def t(e, k):
    return (e.findtext(k) or "").strip()


def facts(path):
    root = ET.parse(path).getroot()
    pv = root.find(".//pathway-visualization")
    out = {"counts": {}}
    for k, tag in SECS.items():
        s = pv.find(tag)
        out["counts"][k] = len(list(s)) if s is not None else 0
    names, byid = {}, {}
    for kind, tag in KIND.items():
        s = pv.find(tag)
        rows = list(s) if s is not None else []
        names[tag] = sorted({t(r, "name") for r in rows})
        byid[kind] = {t(r, "id"): t(r, "name") for r in rows}
    out["names"] = names
    loc = {t(e, "id"): t(e, "name") for e in pv.iter("subcellular-location")}
    st = {t(e, "id"): loc.get(t(e, "subcellular-location-id"), "?") for e in pv.iter("biological-state")}
    nm = lambda e: byid.get(t(e, "element-type"), {}).get(t(e, "element-id"), "?" + t(e, "element-id"))
    sigs = collections.Counter()
    s = pv.find("reactions")
    for r in (s if s is not None else []):
        left = tuple(sorted(nm(e) for e in r.iter("reaction-left-element")))
        right = tuple(sorted(nm(e) for e in r.iter("reaction-right-element")))
        sigs[json.dumps([left, right, t(r, "direction")])] += 1
    out["reaction_signatures"] = sigs
    trs = collections.Counter()
    s = pv.find("transports")
    for r in (s if s is not None else []):
        for e in r.iter("transport-element"):
            trs[json.dumps([nm(e), st.get(t(e, "left-biological-state-id")), st.get(t(e, "right-biological-state-id"))])] += 1
    out["transport_signatures"] = trs
    out["transporters"] = sum(1 for _ in pv.iter("transport-transporter"))
    out["na_types"] = {t(e, "name"): t(e, "nucleotide-type") for e in pv.iter("nucleic-acid")}
    return out


files = sorted(f for f in os.listdir(BASE) if f.endswith(".pwml"))
after_files = set(f for f in os.listdir(AFTER) if f.endswith(".pwml"))
res = {"exports_base": len(files), "exports_after": len(after_files), "only_base": sorted(set(files) - after_files),
       "only_after": sorted(after_files - set(files)), "identical": 0, "changed": [], "lost_names": [], "lost_reaction_signatures": [],
       "lost_transport_signatures": [], "count_changes": [], "per_file": {}}
tot = {"base": collections.Counter(), "after": collections.Counter()}
for f in files:
    if f not in after_files:
        continue
    a, b = os.path.join(BASE, f), os.path.join(AFTER, f)
    same = filecmp.cmp(a, b, shallow=False)
    res["identical"] += same
    fa, fb = facts(a), facts(b)
    tot["base"].update(fa["counts"])
    tot["after"].update(fb["counts"])
    tot["base"]["transporters"] += fa["transporters"]
    tot["after"]["transporters"] += fb["transporters"]
    tot["base"]["na_typed"] += sum(1 for v in fa["na_types"].values() if v)
    tot["after"]["na_typed"] += sum(1 for v in fb["na_types"].values() if v)
    if same:
        continue
    res["changed"].append(f)
    d = {}
    for k in fa["counts"]:
        if fa["counts"][k] != fb["counts"][k]:
            d[k] = [fa["counts"][k], fb["counts"][k]]
            res["count_changes"].append({"file": f, "count": k, "base": fa["counts"][k], "after": fb["counts"][k]})
    if fa["transporters"] != fb["transporters"]:
        d["transporters"] = [fa["transporters"], fb["transporters"]]
        res["count_changes"].append({"file": f, "count": "transporters", "base": fa["transporters"], "after": fb["transporters"]})
    for tag in fa["names"]:
        for n in sorted(set(fa["names"][tag]) - set(fb["names"][tag])):
            res["lost_names"].append({"file": f, "kind": tag, "name": n})
    for s in fa["reaction_signatures"]:
        if s not in fb["reaction_signatures"]:
            res["lost_reaction_signatures"].append({"file": f, "signature": json.loads(s)})
    for s in fa["transport_signatures"]:
        if s not in fb["transport_signatures"]:
            res["lost_transport_signatures"].append({"file": f, "signature": json.loads(s)})
    na = {n: [fa["na_types"].get(n), fb["na_types"].get(n)] for n in set(fa["na_types"]) | set(fb["na_types"])
          if fa["na_types"].get(n) != fb["na_types"].get(n)}
    res["per_file"][f] = {"count_changes": d, "nucleotide_type_changes": na,
                          "transport_signatures": [sorted(fa["transport_signatures"].elements()), sorted(fb["transport_signatures"].elements())]
                          if fa["transport_signatures"] != fb["transport_signatures"] else None}
res["totals"] = {k: dict(v) for k, v in tot.items()}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
print({k: res[k] for k in ("exports_base", "exports_after", "only_base", "only_after", "identical")}, "changed", len(res["changed"]))
print("lost_names", len(res["lost_names"]), "lost_reaction_signatures", len(res["lost_reaction_signatures"]),
      "lost_transport_signatures", len(res["lost_transport_signatures"]))
for c in res["count_changes"]:
    print("  ", c)
for f, v in res["per_file"].items():
    if v["nucleotide_type_changes"]:
        print("  na", f, v["nucleotide_type_changes"])
print("totals base ", res["totals"]["base"])
print("totals after", res["totals"]["after"])
