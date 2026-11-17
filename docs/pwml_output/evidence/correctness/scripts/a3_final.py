import json, collections, sys
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_compare as C, a3_ours_sbml as O
r = json.load(open("C:/t/corr-assess/a3/compare.json"))
# matrix: per pair, count pathways agreeing in each category (0 only_a & 0 only_b); rxn also E_count_diff==0
mat = {}
for pr in C.PAIRS:
    row = collections.OrderedDict()
    n = 0
    for cat in ("comps", "cplx", "rxn", "trn", "itn"):
        ag = dis = 0
        for i, v in r.items():
            p = v["pairs"].get(pr[0] + "|" + pr[1])
            if not p: continue
            x = p[cat]
            bad = x["only_a"] or x["only_b"]
            if cat == "rxn" and x.get("E_count_diff"): bad = True
            if cat == "trn" and not (x["n_a"] or x["n_b"]): continue
            if cat == "itn" and not (x["n_a"] or x["n_b"]): continue
            ag += 0 if bad else 1; dis += 1 if bad else 0
        row[cat] = dict(agree=ag, disagree=dis)
    mat["|".join(pr)] = row
r_out = dict(per_pathway=r, matrix=mat, notes="names normalised (casefold, ASCII, alnum); PW-side compound names aligned to PWML order by position when counts equal (cmap_aligned); enzyme keys are complex names w/o trailing 'complex'; SBGN has no complex glyphs so cplx always differs; proteins omitted from matrix (PathWhiz renames to UniProt names). itn/trn counted only for pathways where either side has >=1.")
json.dump(r_out, open("C:/t/corr-assess/a3/compare.json", "w"), indent=1, default=str)
for k, v in mat.items(): print(k, {c: f"{x['agree']}/{x['agree']+x['disagree']}" for c, x in v.items()})
# ours sbml facts
tot = collections.Counter(); per = {}
for s in C.sample:
    i = s["idx"]; m = O.from_ours_sbml(f"C:/t/corr-assess/a3/ours/{i}.sbml")
    noop = sum(1 for t in m["trn"] if t["noop"]); samecomp = sum(1 for t in m["trn"] if t["Lcomp"] == t["Rcomp"])
    und = len(m["notes"]["undeclared_species"])
    multi = sum(1 for v in m["notes"]["species_by_entity"].values() if len(v) > 1)
    nsp = m["notes"]["n_species"]
    roles = collections.Counter(sbo for r_ in m["rxn"] for _, sbo in r_["roles"])
    per[i] = dict(trn=len(m["trn"]), noop_same_species=noop, same_compartment=samecomp, undeclared=und, multi_location_entities=multi, species=nsp, roles=dict(roles), itn=len(m["itn"]))
    tot.update(trn=len(m["trn"]), noop=noop, samecomp=samecomp, undeclared=und, species=nsp, itn=len(m["itn"]))
json.dump(dict(per=per, totals=tot), open("C:/t/corr-assess/a3/ours_sbml_facts.json", "w"), indent=1)
print(dict(tot)); print({i: (p["trn"], p["same_compartment"], p["undeclared"]) for i, p in per.items() if p["trn"] or p["undeclared"]})
print(collections.Counter(k for p in per.values() for k in p["roles"].elements()))
