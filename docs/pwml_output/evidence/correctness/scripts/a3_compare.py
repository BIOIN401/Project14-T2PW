"""A3: cross-format comparison. Run with C:/t/fmt-venv python (-W ignore)."""
import json, os, sys, collections, itertools
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M

B = M.B
sample = json.load(open(f"{B}/sample.json", encoding="utf-8"))
SRC = ["payload", "ir", "pwml", "db", "pw_sbml", "pw_sbgn", "web", "ours_sbml"]
PAIRS = [("payload", "ir"), ("ir", "pwml"), ("pwml", "db"), ("db", "pw_sbml"), ("db", "pw_sbgn"), ("pw_sbml", "pw_sbgn"),
         ("ir", "web"), ("payload", "ours_sbml"), ("payload", "pwml"), ("payload", "db"), ("payload", "pw_sbml"),
         ("payload", "pw_sbgn"), ("payload", "web"), ("ours_sbml", "pw_sbml"), ("ours_sbml", "pwml"), ("ours_sbml", "db")]


def load(i):
    out = {}
    out["payload"] = M.from_payload(json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8")))
    ir = json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    out["ir"] = M.from_ir(ir)
    out["pwml"] = M.from_pwml(f"{B}/replay/on_v2/pwml/{i}.pwml")
    dbj = json.load(open(f"{B}/pathwhiz_exports/{i}.db.json", encoding="utf-8"))
    ids = sorted((x["id"] for x in dbj["compounds"]), key=int)
    order = out["pwml"]["notes"]["comp_order"]
    cmap = {}
    if len(ids) == len(order):
        cmap = dict(zip(ids, order))
    out["cmap_aligned"] = len(ids) == len(order)
    out["db"] = M.from_db(dbj, cmap)
    out["pw_sbml"] = M.from_pw_sbml(f"{B}/pathwhiz_exports/{i}.sbml", cmap)
    if os.path.exists(f"{B}/pathwhiz_exports/{i}.sbgn"):
        out["pw_sbgn"] = M.from_pw_sbgn(f"{B}/pathwhiz_exports/{i}.sbgn", cmap)
    out["web"] = M.from_web(ir)
    if os.path.exists(f"{B}/a3/ours/{i}.sbml"):
        import a3_ours_sbml as O
        out["ours_sbml"] = O.from_ours_sbml(f"{B}/a3/ours/{i}.sbml")
    return out


def setcmp(a, b):
    a, b = set(a), set(b)
    return dict(common=len(a & b), only_a=len(a - b), only_b=len(b - a), ex_a=sorted(a - b)[:4], ex_b=sorted(b - a)[:4])


def rkeys(m, withE=False):
    d = collections.defaultdict(set)
    for r in m["rxn"]:
        if r["L"] or r["R"]:
            d[(r["L"], r["R"])] |= set(r["E"])
    return d


def rxncmp(A, B_):
    ka, kb = rkeys(A), rkeys(B_)
    common = set(ka) & set(kb)
    Emis = [k for k in common if ka[k] != kb[k]]
    Ecnt = [k for k in common if len(ka[k]) != len(kb[k])]
    rd = lambda k: " + ".join(sorted(k[0])) + " -> " + " + ".join(sorted(k[1]))
    return dict(common=len(common), only_a=len(set(ka) - set(kb)), only_b=len(set(kb) - set(ka)),
                n_a=len(A["rxn"]), n_b=len(B_["rxn"]), E_name_diff=len(Emis), E_count_diff=len(Ecnt),
                ex_a=[rd(k) for k in sorted(set(ka) - set(kb), key=str)[:3]], ex_b=[rd(k) for k in sorted(set(kb) - set(ka), key=str)[:3]],
                ex_Ecnt=[(rd(k), sorted(ka[k]), sorted(kb[k])) for k in Ecnt[:3]])


def trncmp(A, B_):
    ca = collections.Counter(t["cargo"] for t in A["trn"])
    cb = collections.Counter(t["cargo"] for t in B_["trn"])
    return dict(n_a=sum(ca.values()), n_b=sum(cb.values()), common=sum((ca & cb).values()), only_a=sum((ca - cb).values()),
                only_b=sum((cb - ca).values()), ex_a=[sorted(k) for k in (ca - cb)][:3], ex_b=[sorted(k) for k in (cb - ca)][:3])


def itncmp(A, B_):
    ka = {(i["a"], i["b"]) for i in A["itn"]}
    kb = {(i["a"], i["b"]) for i in B_["itn"]}
    return dict(n_a=len(A["itn"]), n_b=len(B_["itn"]), common=len(ka & kb), only_a=len(ka - kb), only_b=len(kb - ka),
                ex_a=sorted(map(str, ka - kb))[:3], ex_b=sorted(map(str, kb - ka))[:3])


def compare(A, B_):
    return dict(comps=setcmp(A["comps"], B_["comps"]), prots=setcmp(A["prots"], B_["prots"]), cplx=setcmp(A["cplx"], B_["cplx"]),
                nuc=setcmp(A["nuc"], B_["nuc"]), locs=setcmp(A["locs"], B_["locs"]), rxn=rxncmp(A, B_), trn=trncmp(A, B_),
                itn=itncmp(A, B_))


if __name__ == "__main__":
    res = {}
    models = {}
    for s in sample:
        i = s["idx"]
        mdl = load(i)
        models[i] = mdl
        res[i] = {"sources": sorted(k for k in mdl if k!="cmap_aligned"), "cmap_aligned": mdl["cmap_aligned"], "counts": {k: dict(comps=len(v["comps"]), prots=len(v["prots"]), cplx=len(v["cplx"]), nuc=len(v["nuc"]),
                                                          rxn=len(v["rxn"]), rxn_keys=len(rkeys(v)), trn=len(v["trn"]), itn=len(v["itn"]))
                                                for k, v in mdl.items() if k!="cmap_aligned"}, "pairs": {}}
        for a, b in PAIRS:
            if a in mdl and b in mdl:
                res[i]["pairs"][f"{a}|{b}"] = compare(mdl[a], mdl[b])
    json.dump(res, open(f"{B}/a3/compare.json", "w", encoding="utf-8"), indent=1, default=str)
    print("wrote compare.json")
