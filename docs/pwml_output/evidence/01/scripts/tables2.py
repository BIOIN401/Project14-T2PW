import json, collections
OUT = "C:/t/ev-oa/funnel/out/"
rows = [json.loads(l) for l in open(OUT + "runs_funnel.jsonl")]
def group(r):
    if r["qok"] is not True: return "OTHER40"
    if r["mode"] == "research": return "RS_research"
    return "P1" if r.get("pwml") else "P0"
G = collections.defaultdict(list)
for r in rows:
    G[group(r)].append(r);
    if r["qok"] is True: G["ALL152"].append(r)
for g in ("ALL152", "P1", "P0", "RS_research", "OTHER40"):
    rs = G[g]
    po = collections.Counter(); nstat = collections.Counter(); qk = collections.Counter(); wh = collections.Counter()
    for r in rs:
        for k, v in r["proc_origin"].items(): po[k] += v
        for k, v in r.get("norm_stats", {}).items(): nstat[k] += v
        for k, v in (r.get("q_quar_kind") or {}).items():
            if r.get("research_applied") is not False and r.get("mode") != "research": qk[k] += v
    print("\n###", g, len(rs))
    for kind in ("reactions", "transports", "interactions"):
        s1 = sum(r["s1"][kind] for r in rs); m = sum(r["m"][kind] for r in rs); f = sum(r["f"][kind] for r in rs)
        m_ps = po[f"m:{kind}:paper_stated"]; m_rag = po[f"m:{kind}:rag_literature"]; m_sa = po[f"m:{kind}:SAME_AS"]
        m_none = po[f"m:{kind}:none"]
        f_ps = po[f"f:{kind}:paper_stated"]; f_none = po[f"f:{kind}:none"]; f_other = f - f_ps - f_none
        print(f"{kind}: S1={s1} | M={m} (paper_stated {m_ps}, SAME_AS {m_sa}, no-lineage-added {m_none}, rag {m_rag}, other {m-m_ps-m_sa-m_none-m_rag}) | F={f} (paper_stated {f_ps}, no-lineage {f_none}, other {f_other}) | quarantined(applied) {qk.get(kind[:-1],0)}")
    print("norm_stats:", {k: nstat[k] for k in ("dedupe_removed_reactions", "dedupe_removed_transports", "no_op_quarantined_count", "interaction_enzymes_promoted", "orphan_proteins_dropped", "unused_proteins_quarantined", "n_aliases_rewritten", "n_same_as_groups", "duplicate_locked_reactions_quarantined_count") if nstat[k]})
    # entity origin / generated
    gen = collections.Counter()
    for r in rs:
        for k, v in (r.get("generated") or {}).items(): gen[k] += v
    print("generated:", dict(gen))
    fo = collections.Counter()
    for r in rs:
        for k, v in (r.get("f_ent_origin") or {}).items(): fo[k] += v
    print("F entity first-lineage origin:", dict(fo))
