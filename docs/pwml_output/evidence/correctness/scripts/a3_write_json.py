import json, collections
B = "C:/t/corr-assess"
raw = json.load(open(f"{B}/a3/validation_raw.json", encoding="utf-8"))
val = {"tools": {"python": "3.13.6 (C:/t/fmt-venv)", "python-libsbml": "5.21.2", "libsbgnpy": "0.7.2 (XSD SBGN-ML 0.3 + schematron pd/er/af + spec rules)", "lxml": "6.1.3",
                 "momapy": "0.16.1 (skia renderer; readers sbgnml-0.2/0.3, sbml, celldesigner)", "sbmlnetwork": "0.5.12 (libsbmlnetwork 0.5.12, networkinfotranslator 0.5.12; autolayout + draw)", "skia-python": "144.1", "ipython": "needed by sbmlnetwork.draw"},
       "sbml": {}, "sbgn": {}}
for kind in ("ours", "pathwhiz"):
    agg = collections.defaultdict(lambda: dict(count=0, files=set(), example=""))
    ne = nw = 0
    for i, v in raw["sbml"][kind].items():
        ne += v["n_error"]; nw += v["n_warning"]
        for s in v["summary"]:
            a = agg[(s["severity"], s["category"], s["id"], s["short"][:90])]
            a["count"] += s["count"]; a["files"].add(i); a["example"] = s["example"][:200]
    val["sbml"][kind] = {"files": len(raw["sbml"][kind]), "level_version": sorted({v["L"] for v in raw["sbml"][kind].values()}), "errors": ne, "warnings": nw,
                         "by_category": [dict(severity=k[0], category=k[1], id=k[2], message=k[3], count=a["count"], files=len(a["files"]), example=a["example"]) for k, a in sorted(agg.items(), key=lambda x: -x[1]["count"])]}
val["sbml"]["ours"]["note"] = "warnings are deliberate: no initial amount (80601), no units (20616, 20518, 99508); 0 errors in 26/26"
val["sbml"]["pathwhiz"]["note"] = "0 errors; 11 invalid sboTerm on speciesReference in 3 files (7,9,150); PathWhiz writes initialAmount=1 on all 529 species so 80601 does not fire"
c = collections.Counter(); n = collections.Counter(); ex = {}
xs = 0; xs_other = collections.Counter()
for i, v in raw["sbgn"].items():
    if "xsd_errors" not in v: continue
    xs += v["xsd_errors"]
    for k, cnt in v.get("sch_kinds", []): c[tuple(k)] += cnt; n[tuple(k)] += 1
    for e in v.get("xsd_examples", [])[:1]:
        if "'id' is required" not in e: xs_other[e[:200]] += 1
val["sbgn"] = {"files_validated": sum(1 for v in raw["sbgn"].values() if "xsd_errors" in v), "file_missing_or_failed": [i for i, v in raw["sbgn"].items() if "xsd_errors" not in v],
               "format_note": "PathWhiz writes SBGN-ML 0.2 (language process description); libsbgnpy upconverts to 0.3 before XSD validation",
               "xsd_errors_total": xs, "xsd_map_id_missing_files": 25, "xsd_other_examples": list(xs_other)[:3],
               "momapy_load": "25/25 load with reader sbgnml-0.2 and render to PNG",
               "schematron_rule_totals": [dict(rule=k[0], message=k[1], count=c[k], files=n[k]) for k, _ in c.most_common(12)],
               "schematron_issues_per_file": {i: v.get("sch_issues") for i, v in raw["sbgn"].items()}}
val["momapy_sbml_load"] = "52/52 SBML files (26 ours, 26 PathWhiz) load with momapy reader 'sbml'; none carries a layout so momapy cannot render SBML (layout=None); rendered with SBMLNetwork autolayout instead"
val["renders"] = "C:/t/corr-assess/a3/render/<idx>_{ours_sbml,pw_sbml,pw_sbgn}.png (77 of 78; 58 pw_sbgn does not exist), log a3/render_log.json"
json.dump(val, open(f"{B}/a3/validation.json", "w", indent=None) if False else open(f"{B}/a3/validation.json", "w", encoding="utf-8"), indent=1)
print("ok")
