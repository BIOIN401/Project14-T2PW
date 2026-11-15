import json, re, sys, collections
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
from lxml import etree
B = M.B
sample = [s["idx"] for s in json.load(open(f"{B}/sample.json", encoding="utf-8"))]
out = {}
tot = collections.Counter()
for i in sample:
    t = open(f"{B}/pathwhiz_exports/{i}.sbml", encoding="utf-8").read()
    m = M.from_pw_sbml(f"{B}/pathwhiz_exports/{i}.sbml")
    o = dict(n_species=t.count("<species "), initialAmount1=len(re.findall(r'initialAmount="1"', t)), arrow_literal=len(re.findall(r"&amp;rarr;", t)),
             reversible_true=t.count('reversible="true"'), reversible_false=t.count('reversible="false"'))
    noop = [x["name"][:80] for x in m["trn"] if x["L"] and x["R"] and set(x["L"]) == set(x["R"])]
    o["transport_noop_same_species"] = len(noop); o["n_transports"] = len(m["trn"])
    # species with the same name in multiple states -> species per compound id only
    sbgn = f"{B}/pathwhiz_exports/{i}.sbgn"
    try:
        g = M.from_pw_sbgn(sbgn)
        o["sbgn_compartment_labels"] = g["notes"]["compartment_labels"]
        o["sbgn_classes"] = g["notes"]["glyph_classes"]
        root = etree.parse(sbgn).getroot(); ns = etree.QName(root).namespace
        proc = [x for x in root.iter("{%s}glyph" % ns) if x.get("class") == "process"]
        o["sbgn_process_unlabelled"] = sum(1 for x in proc if x.find("{%s}label" % ns) is None)
        o["sbgn_map_has_id"] = root.find("{%s}map" % ns).get("id") is not None
        o["sbgn_arcs_target_glyph_not_port"] = sum(1 for a in root.iter("{%s}arc" % ns) if "." in (a.get("target") or "") and not re.search(r"\.\d+$", a.get("target")) is None and False)
    except Exception as e:
        o["sbgn_err"] = str(e)[:100]
    out[i] = o
    for k in ("n_species", "initialAmount1", "arrow_literal", "reversible_true", "reversible_false", "transport_noop_same_species", "n_transports"): tot[k] += o[k]
json.dump(dict(per=out, totals=tot), open(f"{B}/a3/pw_format_facts.json", "w", encoding="utf-8"), indent=1, default=str)
print(dict(tot))
for i, o in out.items(): print(i, {k: v for k, v in o.items() if k in ("transport_noop_same_species", "n_transports", "sbgn_compartment_labels", "sbgn_process_unlabelled", "sbgn_map_has_id")})
