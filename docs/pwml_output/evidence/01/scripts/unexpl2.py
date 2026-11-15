import json, os, re, collections
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
def norm(s):
    s = str(s or "").casefold(); s = re.sub(r"\([^)]*\)", " ", s); return re.sub(r"[^a-z0-9]+", " ", s).strip()
class UF:
    def __init__(s): s.p = {}
    def f(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)
cats = collections.Counter(); shown = 0
for r in rows:
    if r["qok"] is not True or r["mode"] != "pathwhiz": continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    s1 = json.load(open(d + "/stage1_payload.json", encoding="utf-8")); fm = json.load(open(d + "/final_mapped.json", encoding="utf-8"))
    mg = json.load(open(d + "/merged_payload.json", encoding="utf-8")); qr = json.load(open(d + "/quarantine_report.json", encoding="utf-8"))
    uf = UF()
    for x in mg["processes"].get("interactions", []):
        if str(x.get("relationship")).upper() == "SAME_AS": uf.u(norm(x.get("entity_1")), norm(x.get("entity_2")))
    key = lambda x: frozenset({uf.f(norm(x.get("entity_1"))), uf.f(norm(x.get("entity_2")))})
    fk = {key(x) for x in fm["processes"].get("interactions", [])}
    fnames = {norm(x.get("name")) for x in fm["processes"].get("interactions", [])}
    fent = {uf.f(norm(e["name"])) for b in fm["entities"].values() if isinstance(b, list) for e in b if isinstance(e, dict) and e.get("name")}
    qk = {key(q["original_process"]) for q in qr.get("quarantined", []) if q.get("process_kind") == "interaction"}
    # promoted to enzyme: interaction relationship catalyzes etc.
    seen = set()
    for x in s1["processes"].get("interactions", []):
        k = key(x)
        if k in seen: continue
        seen.add(k)
        if k in fk or k in qk: continue
        e1, e2 = uf.f(norm(x.get("entity_1"))), uf.f(norm(x.get("entity_2")))
        rel = str(x.get("relationship")).casefold()
        if norm(x.get("name")) in fnames: c = "name-present-in-F (kept, endpoints rewritten)"
        elif re.search(r"catalyz|enzyme|promotes|facilitat", rel): c = "catalysis-type: promoted into reaction enzymes (promote_interaction_enzymes)"
        elif e1 not in fent or e2 not in fent: c = "an endpoint entity is absent from final_mapped (entity dropped/merged)"
        else: c = "both endpoints present, row absent (dedupe/other)"
        cats[c] += 1
        if c.startswith("both") and shown < 6: print(r["path"][-60:-16], "|", x.get("name"), "|", rel); shown += 1
for k, v in cats.most_common(): print(v, k)
