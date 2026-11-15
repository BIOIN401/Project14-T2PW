"""Fate of S1 interactions by relationship category (pathwhiz canonical runs)."""
import sys, json, os, re, collections
sys.path.insert(0, "C:/t/oa/src")
from t2pw.pwml.interaction_types import pwml_interaction_type, interaction_relationship_word
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
def cat(x):
    w = interaction_relationship_word(x)
    t = pwml_interaction_type(w)
    if t: return t.lower() + " (typed)"
    if re.search(r"bind|interact|associat|complex|co-?local|component|recruit", w.casefold()): return "binding/interaction (unsigned)"
    return "other unsigned (modifies, regulates, degrades, ...)"
res = collections.defaultdict(collections.Counter)
for r in rows:
    if r["qok"] is not True or r["mode"] != "pathwhiz": continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    s1 = json.load(open(d + "/stage1_payload.json", encoding="utf-8")); mg = json.load(open(d + "/merged_payload.json", encoding="utf-8"))
    fm = json.load(open(d + "/final_mapped.json", encoding="utf-8")); qr = json.load(open(d + "/quarantine_report.json", encoding="utf-8"))
    uf = UF()
    for x in mg["processes"].get("interactions", []):
        if str(x.get("relationship")).upper() == "SAME_AS": uf.u(norm(x.get("entity_1")), norm(x.get("entity_2")))
    key = lambda x: frozenset({uf.f(norm(x.get("entity_1"))), uf.f(norm(x.get("entity_2")))})
    fk = {key(x) for x in fm["processes"].get("interactions", [])}
    qk = {key(q["original_process"]): q.get("reason") for q in qr.get("quarantined", []) if q.get("process_kind") == "interaction"}
    seen = set()
    for x in s1["processes"].get("interactions", []):
        k = key(x); c = cat(x)
        if k in seen: res[c]["S1 duplicate"] += 1; continue
        seen.add(k)
        res[c]["unique S1 rows"] += 1
        if k in fk: res[c]["in F"] += 1
        elif k in qk: res[c]["quarantined"] += 1
        else: res[c]["other lost"] += 1
for c, v in res.items():
    u = v["unique S1 rows"]
    print(c, dict(v), f"kept {v['in F']/u:.0%} quarantined {v['quarantined']/u:.0%}")
