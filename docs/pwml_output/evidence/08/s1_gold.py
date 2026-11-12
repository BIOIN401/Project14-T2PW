"""Dump gold reactions (7 papers) with per-reaction miss rate in final payloads. Read-only."""
import sys, json, collections
sys.path.insert(0, "C:/t/cm/src")
from t2pw.bench.eval_v2 import gold as G
v2 = G.load_supplement()
cases = {pid: ("eval_v2", c.complete, list(c.reactions)) for pid, c in v2.items()}
pin = json.load(open("C:/t/cm/src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]
meta = {}
for c in pin:
    pid = c["paper_id"]
    meta[pid] = {k: c.get(k) for k in ("organism", "species", "title", "pathway", "name") if c.get(k)}
    if pid in cases or not c.get("supported_reactions"): continue
    cases[pid] = ("pinned_v1", False, [("raw", r) for r in c["supported_reactions"]])
rows = [json.loads(l) for l in open("C:/t/cm/docs/pwml_output/evidence/01/gold_recall.jsonl")]
miss = collections.defaultdict(lambda: None)
agg = {}
for r in rows:
    a = agg.setdefault(r["paper"], {"runs": 0, "f": None, "s1": None})
    a["runs"] += 1
    if a["f"] is None: a["f"] = [0]*len(r["f_hits"]); a["s1"] = [0]*len(r["s1_hits"])
    for i, h in enumerate(r["f_hits"]): a["f"][i] += int(h)
    for i, h in enumerate(r["s1_hits"]): a["s1"][i] += int(h)
out = {}
for pid, (src, comp, rxs) in cases.items():
    L = []
    for i, g in enumerate(rxs):
        if isinstance(g, tuple):
            r = g[1]; ins = [t["name"] for t in r["inputs"]]; outs = [t["name"] for t in r["outputs"]]; lab = r.get("label", "")
        else:
            ins = [t.name for t in g.inputs]; outs = [t.name for t in g.outputs]; lab = g.label
        a = agg.get(pid)
        L.append({"i": i, "label": lab, "in": ins, "out": outs,
                  "final_hit_runs": a["f"][i] if a else None, "runs": a["runs"] if a else 0})
    out[pid] = {"src": src, "complete": comp, "meta": meta.get(pid), "rx": L}
json.dump(out, open("C:/t/cm/docs/pwml_output/evidence/08/gold_reactions.json", "w"), indent=1)
for pid, v in out.items():
    print(pid, v["src"], v["complete"], v["meta"])
    for r in v["rx"]: print("  ", r["i"], r["final_hit_runs"], "/", r["runs"], r["in"], "->", r["out"])
