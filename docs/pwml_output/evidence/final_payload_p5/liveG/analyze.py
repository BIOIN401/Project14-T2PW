"""usage: analyze.py OUTDIR  (a run's out/ dir). Offline, no network.
Prints JSON: fill provenance counts, PWML title/reference, species-only and wildcard-bindable states,
and layout_debug_counts recomputed by rebuilding the PWML from the run's pwml_ir.json with the committed writer
(counters are not persisted by the pipeline). Also reports whether the rebuild equals the run's PWML in the
state-bearing sections."""
import glob, json, os, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ["T2PW_FINAL_PAYLOAD"] = "1"
sys.path.insert(0, "C:/t/fp/src")
from lxml import etree
from t2pw.pwml import writer as w
discover_structure_signature = w.discover_structure_signature

out = Path(sys.argv[1])
found = glob.glob(str(out / "pathway*.pwml"))
pwml = Path(found[0]) if found else None
res = {"pwml": str(pwml) if pwml else None, "pwml_bytes": pwml.stat().st_size if pwml else None}
fm = json.load(open(out / "final_mapped.json", encoding="utf-8"))
fill = fm.get("final_payload_fill") or {}
res["fill"] = {"present": bool(fill), "written": fill.get("written"), "by_field": fill.get("by_field"),
               "n_entries": len(fill.get("entries") or []),
               "entry_kinds": {f"{k[0]}.{k[1]}": v for k, v in Counter((e.get("kind"), e.get("field")) for e in fill.get("entries") or []).items()}}
res["final_payload_pathway"] = fm.get("final_payload_pathway")
if pwml is None:  # research / failed runs write no PWML: fill + pathway provenance only
    print(json.dumps(res, indent=1, default=str)); sys.exit(0)
if pwml is None:  # research / failed runs write no PWML: fill + pathway provenance only
    print(json.dumps(res, indent=1, default=str)); sys.exit(0)
root = etree.parse(str(pwml)).getroot()
res["pwml_name"] = (root.findtext(".//pathway/name") or root.findtext(".//name"))
res["pubmed_ids"] = [e.text for e in root.iter("pubmed-id")]
rows = {}
for sec in ("tissues", "cell-types", "subcellular-locations"):
    for n in root.findall(f".//{sec}/{sec[:-1]}"):
        rows[(sec, (n.findtext("id") or "").strip())] = ((n.findtext("name") or "").casefold(), (n.findtext("ontology-id") or "").strip() or None)
sts = []
for st in root.findall(".//biological-states/biological-state"):
    c = [(st.findtext("species-id") or "").strip() or None]
    for sec, tag in (("tissues", "tissue-id"), ("cell-types", "cell-type-id"), ("subcellular-locations", "subcellular-location-id")):
        v = (st.findtext(tag) or "").strip()
        c.append(rows.get((sec, v), ("?" + v, None)) if v else None)
    sts.append(tuple(c))
res["n_states"] = len(sts)
res["species_only_states"] = sum(1 for s in sts if all(x is None for x in s[1:]))
res["wildcard_bindable_states"] = sum(1 for k, s in enumerate(sts) if any(j != k and all(x is None or x == y for x, y in zip(s, t)) for j, t in enumerate(sts)))
res["states"] = [list(map(str, s)) for s in sts]
res["section_counts"] = {sec: len(root.findall(f".//{sec}/{sec[:-1]}")) for sec in ("tissues", "cell-types", "subcellular-locations", "biological-states")}
ir = json.load(open(out / "pwml_ir.json", encoding="utf-8"))
ref = Path("C:/t/fp/reference/PW000001.pwml")
sig = discover_structure_signature(ref)
args = SimpleNamespace(name="x", description="", subject="Metabolic", pw_id="PW000000", height=1400, width=3200, background_color="#FFFFFF", ref=str(ref))
b = w.DeterministicPwmlBuilder(extraction=ir, signature=sig, args=args)
b.build()
res["layout_debug_counts"] = dict(b.layout_debug_counts)
print(json.dumps(res, indent=1, default=str))
