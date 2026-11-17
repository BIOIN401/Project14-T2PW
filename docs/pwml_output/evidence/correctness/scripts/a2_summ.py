"""A2: summarise checks/*.json into per_pathway.json and an aggregate table (printed)."""
import json
from collections import Counter

O = "C:/t/corr-assess/a2/checks"
cfgs = ["before", "reg", "enz", "merge", "all"]
R = {c: json.load(open(f"{O}/{c}.json", encoding="utf-8")) for c in cfgs}
keys = ["crossings", "long_edges_gt600", "edge_len_max", "bends", "enzyme_rxn_dist_median", "enzyme_rxn_dist_max",
        "enzyme_far_gt200", "reg_edges", "reg_only_molecules", "dup_tags", "n_entity_drawings", "W", "H", "edge_through_node", "node_overlaps", "route_fail"]
agg = {c: Counter() for c in cfgs}
for c in cfgs:
    for i, x in R[c].items():
        for k in keys:
            v = x["metrics"].get(k)
            if isinstance(v, (int, float)):
                agg[c][k] += v
        agg[c]["area_Mpx"] += x["metrics"]["W"] * x["metrics"]["H"] / 1e6
        for k, v in x["diagram_vs_ir"].items():
            agg[c]["ISSUE:" + k] += len(v)
print("metric".ljust(46), *[c.rjust(9) for c in cfgs])
for k in sorted(set().union(*[set(a) for a in agg.values()])):
    print(k.ljust(46), *[str(round(agg[c].get(k, 0), 1)).rjust(9) for c in cfgs])
per = {}
for i, x in R["before"].items():
    per[i] = dict(pathway=x["pathway"], style=x["metrics"]["style"],
                  diagram_vs_ir=x["diagram_vs_ir"], payload_vs_ir=x["payload_vs_ir"],
                  metrics_before=x["metrics"],
                  metrics_after_all={k: R["all"][i]["metrics"].get(k) for k in keys + ["hidden"]},
                  after_all_issues={k: len(v) for k, v in R["all"][i]["diagram_vs_ir"].items()},
                  screenshot=f"C:/t/corr-assess/a2/before/{i}.png")
mods = json.load(open("C:/t/corr-assess/a2/modifiers_as_enzymes.json", encoding="utf-8"))["regulators_drawn_as_catalysts"]
dups = json.load(open("C:/t/corr-assess/a2/duplicate_reactions.json", encoding="utf-8"))
for i in per:
    per[i]["upstream_regulator_as_catalyst"] = [m for m in mods if str(m["idx"]) == i]
    per[i]["upstream_duplicate_reactions"] = dups.get(i)
json.dump(dict(source="IR regenerated from C:/t/integ (bcc67409) via a2_ir_replay.py; PWML byte-identical to base for 26/26",
               aggregate={c: dict(agg[c]) for c in cfgs}, pathways=per),
          open("C:/t/corr-assess/a2/per_pathway.json", "w", encoding="utf-8"), indent=1, default=str)
