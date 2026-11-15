"""Fate of each Stage-1 (paper_stated) interaction in pathwhiz-mode canonical runs: kept / quarantined(reason) / collapsed-duplicate / unexplained."""
import json, os, re, collections
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
def norm(s):
    s = str(s or "").casefold()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s
class UF:
    def __init__(s): s.p = {}
    def f(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)
fate = collections.Counter(); fate_q = collections.Counter(); per_group = collections.defaultdict(collections.Counter)
dup_in_s1 = 0; tot_s1 = 0
rfate = collections.Counter(); rfate_q = collections.Counter()
for r in rows:
    if r["qok"] is not True or r["mode"] != "pathwhiz":
        continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    s1 = json.load(open(d + "/stage1_payload.json", encoding="utf-8"))
    mg = json.load(open(d + "/merged_payload.json", encoding="utf-8"))
    fm = json.load(open(d + "/final_mapped.json", encoding="utf-8"))
    qr = json.load(open(d + "/quarantine_report.json", encoding="utf-8"))
    uf = UF()
    for x in mg["processes"].get("interactions", []):
        if str(x.get("relationship")).upper() == "SAME_AS":
            uf.u(norm(x.get("entity_1")), norm(x.get("entity_2")))
    def key(x):
        return frozenset({uf.f(norm(x.get("entity_1"))), uf.f(norm(x.get("entity_2")))})
    fkeys = {key(x) for x in fm["processes"].get("interactions", [])}
    qkeys = {}
    for q in qr.get("quarantined") or []:
        if q.get("process_kind") == "interaction":
            o = q.get("original_process") or {}
            qkeys[key(o)] = q.get("reason")
    grp = "P1" if r.get("pwml") else "P0"
    seen = set()
    for x in s1["processes"].get("interactions", []):
        k = key(x); tot_s1 += 1
        if k in seen:
            fate["S1-internal duplicate (same alias-resolved endpoint pair)"] += 1; per_group[grp]["dup"] += 1; dup_in_s1 += 1
            continue
        seen.add(k)
        if k in fkeys:
            fate["kept (endpoint pair present in final_mapped)"] += 1; per_group[grp]["kept"] += 1
        elif k in qkeys:
            fate["quarantined"] += 1; fate_q[qkeys[k]] += 1; per_group[grp]["quar"] += 1
        else:
            fate["unexplained (not in F, not in quarantine list)"] += 1; per_group[grp]["unexpl"] += 1
    # reactions: S1 reactions by (input set, output set)
    def rk(x):
        def side(v):
            out = []
            for a in v or []:
                a = a if isinstance(a, str) else (a.get("entity") or a.get("name") or "") if isinstance(a, dict) else str(a)
                out.append(uf.f(norm(a)))
            return frozenset(out)
        return (side(x.get("inputs")), side(x.get("outputs")))
    frk = {rk(x) for x in fm["processes"].get("reactions", [])}
    qrk = {}
    for q in qr.get("quarantined") or []:
        if q.get("process_kind") == "reaction":
            qrk[rk(q.get("original_process") or {})] = q.get("reason")
    lockq = {}
    for rec in fm.get("quarantined_locked_reactions") or []:
        o = rec.get("original_reaction") or {}
        lockq[rk(o)] = rec.get("reason")
    seenr = set()
    for x in s1["processes"].get("reactions", []):
        k = rk(x)
        if k in seenr:
            rfate["S1-internal duplicate (same input/output sets)"] += 1; continue
        seenr.add(k)
        if k in frk: rfate["kept (input/output sets present in final_mapped)"] += 1
        elif k in qrk: rfate["quarantined"] += 1; rfate_q[qrk[k]] += 1
        elif k in lockq: rfate["locked-reaction record (dedupe/no-op/filter)"] += 1; rfate_q["lock:" + str(lockq[k])] += 1
        else: rfate["unexplained"] += 1
print("INTERACTIONS (pathwhiz canon, S1 rows =", tot_s1, ")")
for k, v in fate.most_common(): print(v, k)
print(fate_q.most_common(10))
print({g: dict(v) for g, v in per_group.items()})
print("REACTIONS")
for k, v in rfate.most_common(): print(v, k)
print(rfate_q.most_common(10))
