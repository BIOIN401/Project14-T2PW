"""Scan a replay arm's IRs + PWMLs for the defects of this lane. usage: scan97.py <arm> [--detail]"""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET

sys.path.insert(0, "C:/t/pwml-fix/src")
from t2pw.vision.checks.layout_metrics import pwml_layout_metrics  # noqa: E402

ARM = sys.argv[1]
DETAIL = "--detail" in sys.argv
R = f"C:/t/pwml-fix-work/replay/{ARM}"
recs = [json.loads(l) for l in open(f"{R}/results.jsonl", encoding="utf-8")]
tot = collections.Counter()
rows = {}
for rec in recs:
    i = rec["idx"]
    if not rec.get("pwml_written"):
        continue
    ir = json.load(open(f"{R}/ir/{i}.pwml_ir.json", encoding="utf-8"))
    rep = json.load(open(f"{R}/ir/{i}.pwml_ir_report.json", encoding="utf-8"))
    st = {s["key"]: s for s in ir["biological_states"]}
    locname = {x["key"]: x.get("name") for x in ir.get("subcellular_locations", [])}
    tup = {k: (s.get("species_key"), s.get("subcellular_location_key"), s.get("cell_type_key"), s.get("tissue_key")) for k, s in st.items()}
    c = collections.Counter()
    ex = collections.defaultdict(list)
    tc = collections.Counter(tup.values())
    c["states"] = len(st)
    c["state_tuple_dups"] = sum(n - 1 for n in tc.values())
    if c["state_tuple_dups"]:
        ex["state_tuple_dups"] = [(k, s["name"], locname.get(s.get("subcellular_location_key"))) for k, s in st.items() if tc[tup[k]] > 1]
    for t in ir["processes"]["transports"]:
        for e in t["transport_elements"]:
            c["transport_elements"] += 1
            l, r = e["left_biological_state_key"], e["right_biological_state_key"]
            if l == r:
                c["tr_same_key"] += 1
                ex["tr_same_key"].append((t["name"], st[l]["name"] if l in st else l))
            elif tup.get(l) == tup.get(r):
                c["tr_same_tuple"] += 1
                ex["tr_same_tuple"].append((t["name"], st[l]["name"], st[r]["name"]))
        tk = [(x["entity_key"], x["biological_state_key"]) for x in t.get("transporters") or []]
        if len(set(tk)) < len(tk):
            c["tr_dup_transporter"] += len(tk) - len(set(tk))
            ex["tr_dup_transporter"].append(t["name"])
    for iss in rep.get("errors", []) + rep.get("warnings", []):
        if iss["code"] in ("unresolved_biological_state", "unresolved_biological_state_component"):
            c[iss["code"]] += 1
            ex[iss["code"]].append(iss.get("pointer") + " " + str(iss.get("biological_state") or iss.get("name")))
    xb = open(f"{R}/pwml/{i}.pwml", "rb").read()
    root = ET.fromstring(xb)
    for na in root.iter("nucleic-acid"):
        nt = na.find("nucleotide-type")
        if nt is None or not (nt.text or "").strip():
            c["na_untyped"] += 1
            ex["na_untyped"].append(na.findtext("name"))
    m = pwml_layout_metrics(xb)
    c["dup_boxes"] = m["duplicate_compartment_boxes"]
    c["unlabeled_boxes"] = m["unlabeled_compartment_boxes"]
    c["boxes"] = m["compartment_boxes"]
    if c["dup_boxes"]:
        ex["dup_boxes"] = m["examples"]["duplicate_boxes"]
    tot.update(c)
    rows[i] = {"counts": dict(c), "examples": {k: v for k, v in ex.items()}}
json.dump({"totals": dict(tot), "runs": rows}, open(f"C:/t/pwml-fix-work/scan_{ARM}.json", "w", encoding="utf-8"), indent=1, default=str)
print("TOTALS", dict(sorted(tot.items())))
for k in ("state_tuple_dups", "tr_same_key", "tr_same_tuple", "tr_dup_transporter", "unresolved_biological_state", "na_untyped", "dup_boxes", "unlabeled_boxes"):
    hit = sorted(i for i, r in rows.items() if r["counts"].get(k))
    print(k, len(hit), hit)
    if DETAIL:
        for i in hit:
            print("    ", i, rows[i]["examples"].get(k))
