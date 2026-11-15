"""Aggregate funnel tables + per-run tidy CSV. Output: out/tidy_runs.csv, out/tables.json, prints markdown tables."""
import json, collections, csv, os
OUT = "C:/t/ev-oa/funnel/out/"
rows = [json.loads(l) for l in open(OUT + "runs_funnel.jsonl")]
RE = {json.loads(l)["path"]: json.loads(l) for l in open(OUT + "re_reports.jsonl")}
K8 = ["compounds", "proteins", "protein_complexes", "nucleic_acids", "reactions", "transports", "interactions"]
PWK = {"compounds": "compounds", "proteins": "proteins", "protein_complexes": "protein-complexes", "nucleic_acids": "nucleic-acids",
       "reactions": "reactions", "transports": "transports", "interactions": "interactions"}

def group(r):
    if r["qok"] is not True:
        return "OTHER40"
    if r["mode"] == "research":
        return "RS_research"
    return "P1_pathwhiz_pwml" if r.get("pwml") else "P0_pathwhiz_nopwml"

PCLS = {}
if os.path.exists(OUT + "paper_classes.csv"):
    for pr_ in csv.DictReader(open(OUT + "paper_classes.csv", encoding="utf-8")):
        PCLS[pr_["paper"]] = pr_["class"]
for r in rows:
    if "e2e" in r["path"]:
        r["paper"] = "PMC7059786"  # PMID 32185015 == PMC7059786 (same review)
    r["paper_class"] = PCLS.get(r["paper"], "") if r["qok"] is True else ""
    r["group"] = group(r)
    rc = RE[r["path"]]
    r["re"] = rc

def tot(rs, st, k):
    return sum((r[st] or {}).get(k, 0) for r in rs)

G = collections.OrderedDict()
G["ALL152"] = [r for r in rows if r["qok"] is True]
for g in ("P1_pathwhiz_pwml", "P0_pathwhiz_nopwml", "RS_research", "OTHER40"):
    G[g] = [r for r in rows if r["group"] == g]

T = {}
for g, rs in G.items():
    t = {"n_runs": len(rs), "n_papers": len({r["paper"] for r in rs})}
    for st in ("s1", "m", "f"):
        t[st] = {k: tot(rs, st, k) for k in K8}
    pr = [r for r in rs if r.get("pwml")]
    t["pwml"] = {k: sum(r["pwml"].get(PWK[k], 0) for r in pr) for k in K8} if pr else None
    t["f_on_pwml_runs"] = {k: tot(pr, "f", k) for k in K8} if pr else None
    # mapping status of F
    ms = collections.Counter()
    for r in rs:
        for k, v in (r.get("f_map") or {}).items(): ms[k] += v
    t["f_map"] = dict(ms)
    ex = collections.Counter()
    for r in rs:
        for k, v in (r.get("f_ext") or {}).items(): ex[k] += v
    t["f_ext"] = dict(ex)
    gen = collections.Counter()
    for r in rs:
        for k, v in (r.get("generated") or {}).items(): gen[k] += v
    t["generated"] = dict(gen)
    # quarantine
    qq = collections.Counter(); qa = collections.Counter(); qr = collections.Counter()
    for r in rs:
        for k, v in (r.get("q_quar_kind") or {}).items(): qq[k] += v
        for k, v in (r.get("q_acc_kind") or {}).items(): qa[k] += v
        for k, v in (r.get("q_reasons") or {}).items(): qr[k] += v
    t["q_quar"] = dict(qq); t["q_acc"] = dict(qa); t["q_reasons"] = dict(qr.most_common())
    rem = collections.Counter()
    for r in rs:
        for k, v in (r.get("rem_reasons") or {}).items(): rem[k] += v
    t["removed_entities"] = dict(rem)
    ns = collections.Counter()
    for r in rs:
        for k, v in (r.get("norm_stats") or {}).items(): ns[k] += v
    t["norm_stats"] = dict(ns)
    ea = collections.Counter()
    for r in rs:
        ea["ea_removed"] += r["ea_removed"]; ea["ea_demoted"] += r["ea_demoted"]
        for k, v in r["ea_removed_rules"].items(): ea[k] += v
    t["entity_admission"] = dict(ea)
    # reaction_evidence coverage by kind
    bk = collections.defaultdict(collections.Counter); wh = collections.Counter(); oi = collections.Counter()
    for r in rs:
        c = r["re"].get("pwml_coverage") or {}
        for k, v in (c.get("by_kind") or {}).items():
            for kk, vv in v.items():
                if isinstance(vv, int): bk[k][kk] += vv
        wh["by_design_withheld"] += c.get("by_design_withheld") or 0
        wh["merged_dup"] += c.get("merged_duplicates") or 0
        for w in r["re"].get("withheld_reasons") or []: wh[w] += 1
        for o in r["re"].get("omitted") or []:
            oi[(o.get("kind"), o.get("reason") or "(no reason: " + str(o.get("group")) + ")", o.get("ki"))] += 1
    t["re_by_kind"] = {k: dict(v) for k, v in bk.items()}
    t["re_withheld"] = dict(wh)
    t["re_omitted"] = {"|".join(map(str, k)): v for k, v in oi.most_common()}
    T[g] = t

json.dump(T, open(OUT + "tables.json", "w"), indent=1)

# tidy csv
cols = ["path", "root", "date", "paper", "paper_class", "group", "mode", "qok", "pwml_file", "pathway_name", "context_type", "document_type", "organism"]
for st in ("s1", "m", "f"):
    for k in K8: cols.append(f"{st}_{k}")
for k in K8: cols.append(f"pwml_{k}")
cols += ["f_cmp_mapped", "f_cmp_novel", "f_cmp_ambiguous", "f_prot_mapped", "f_prot_noid", "f_cplx_generated", "q_quar_reaction", "q_quar_transport", "q_quar_interaction",
         "q_acc_reaction", "q_acc_transport", "q_acc_interaction", "ent_removed_by_closure", "ea_removed", "ea_demoted_currency", "re_drawn", "re_total", "re_withheld_design",
         "no_pwml_stage", "no_pwml_message"]
os.makedirs(OUT, exist_ok=True)
import re as _re
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
with open(OUT + "tidy_runs.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in rows:
        d = {c: "" for c in cols}
        for c in ("path", "root", "date", "paper", "paper_class", "group", "mode", "qok", "pwml_file", "pathway_name", "context_type", "document_type", "organism"):
            d[c] = r.get(c, "")
        for st in ("s1", "m", "f"):
            for k in K8: d[f"{st}_{k}"] = (r[st] or {}).get(k, "")
        if r.get("pwml"):
            for k in K8: d[f"pwml_{k}"] = r["pwml"].get(PWK[k], 0)
        fm_ = r.get("f_map") or {}
        d["f_cmp_mapped"] = fm_.get("compounds:matched", 0)
        d["f_cmp_novel"] = fm_.get("compounds:novel", 0) + fm_.get("compounds:unresolved", 0)
        d["f_cmp_ambiguous"] = fm_.get("compounds:ambiguous", 0)
        d["f_prot_mapped"] = (r.get("f_ext") or {}).get("proteins:id", 0)
        d["f_prot_noid"] = (r.get("f_ext") or {}).get("proteins:noid", 0)
        d["f_cplx_generated"] = sum(v for k, v in (r.get("generated") or {}).items() if k.startswith("f:protein_complexes"))
        for k in ("reaction", "transport", "interaction"):
            d["q_quar_" + k] = (r.get("q_quar_kind") or {}).get(k, 0)
            d["q_acc_" + k] = (r.get("q_acc_kind") or {}).get(k, 0)
        d["ent_removed_by_closure"] = sum((r.get("rem_counts") or {}).get("entities", 0) for _ in [0]) if r.get("rem_counts") else ""
        d["ea_removed"] = r["ea_removed"]
        d["ea_demoted_currency"] = r["ea_demoted"]
        c = r["re"].get("pwml_coverage") or {}
        d["re_drawn"] = c.get("drawn", ""); d["re_total"] = c.get("total", ""); d["re_withheld_design"] = c.get("by_design_withheld", "")
        if r["group"] in ("P0_pathwhiz_nopwml",) or (r["qok"] is not True and not r.get("pwml")):
            dd = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
            if os.path.exists(dd + "/RESULT.txt"):
                txt = open(dd + "/RESULT.txt", encoding="utf-8").read()
                m = _re.search(r"stage\s*:\s*(\S+)", txt); d["no_pwml_stage"] = m.group(1) if m else ""
                m = _re.search(r"message\s*:\s*(.*)", txt); d["no_pwml_message"] = (m.group(1)[:110] if m else "")
        w.writerow(d)
print("wrote", OUT + "tidy_runs.csv")
for g, t in T.items():
    print("\n##", g, t["n_runs"], "runs", t["n_papers"], "papers")
    for st in ("s1", "m", "f"):
        print(st, t[st])
    print("pwml", t["pwml"], "| F on pwml runs", t["f_on_pwml_runs"])
