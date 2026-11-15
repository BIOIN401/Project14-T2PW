"""libSBML consistency + SBGN XSD/momapy validation. Run with C:/t/fmt-venv python."""
import json, os, sys, collections, glob
import libsbml
B = "C:/t/corr-assess"
sample = [s["idx"] for s in json.load(open(f"{B}/sample.json", encoding="utf-8"))]
CAT = {libsbml.LIBSBML_CAT_GENERAL_CONSISTENCY: "general", libsbml.LIBSBML_CAT_IDENTIFIER_CONSISTENCY: "identifier",
       libsbml.LIBSBML_CAT_UNITS_CONSISTENCY: "units", libsbml.LIBSBML_CAT_MATHML_CONSISTENCY: "mathml",
       libsbml.LIBSBML_CAT_SBO_CONSISTENCY: "sbo", libsbml.LIBSBML_CAT_OVERDETERMINED_MODEL: "overdetermined",
       libsbml.LIBSBML_CAT_MODELING_PRACTICE: "modeling_practice", libsbml.LIBSBML_CAT_INTERNAL_CONSISTENCY: "internal",
       libsbml.LIBSBML_CAT_SBML: "sbml_syntax", libsbml.LIBSBML_CAT_XML: "xml",
       libsbml.LIBSBML_CAT_SBML_L1_COMPAT: "l1", libsbml.LIBSBML_CAT_SBML_L2V1_COMPAT: "l2v1"}
SEV = {libsbml.LIBSBML_SEV_INFO: "info", libsbml.LIBSBML_SEV_WARNING: "warning", libsbml.LIBSBML_SEV_ERROR: "error",
       libsbml.LIBSBML_SEV_FATAL: "fatal"}

def sbml_check(path, units=True):
    d = libsbml.readSBMLFromFile(path)
    out = {"read_errors": d.getNumErrors(), "level": d.getLevel(), "version": d.getVersion()}
    if not units:
        d.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)
    n = d.checkConsistency()
    msgs = []
    for i in range(d.getNumErrors()):
        e = d.getError(i)
        msgs.append((e.getErrorId(), SEV.get(e.getSeverity(), str(e.getSeverity())), CAT.get(e.getCategory(), str(e.getCategory())),
                     e.getShortMessage(), e.getMessage()[:240].replace("\n", " ")))
    out["messages"] = msgs
    return out

def summarise(msgs):
    agg = collections.OrderedDict()
    for eid, sev, cat, short, m in msgs:
        k = (sev, cat, eid)
        a = agg.setdefault(k, {"severity": sev, "category": cat, "id": eid, "short": short, "count": 0, "example": m})
        a["count"] += 1
    return sorted(agg.values(), key=lambda a: (a["severity"] != "error", -a["count"]))

res = {"sbml": {}, "sbgn": {}}
for kind, src in (("ours", f"{B}/a3/ours/%d.sbml"), ("pathwhiz", f"{B}/pathwhiz_exports/%d.sbml")):
    for i in sample:
        if not os.path.exists(src % i): continue
        r = sbml_check(src % i)
        res["sbml"].setdefault(kind, {})[i] = {"read_errors": r["read_errors"], "summary": summarise(r["messages"]),
            "n_error": sum(1 for m in r["messages"] if m[1] in ("error", "fatal")), "n_warning": sum(1 for m in r["messages"] if m[1] == "warning"),
            "n_info": sum(1 for m in r["messages"] if m[1] == "info"), "L": f'{r["level"]}v{r["version"]}'}

# SBGN: XSD + spec rules + schematron (libsbgnpy), then momapy load
from pathlib import Path
from libsbgnpy.validator import validate_xsd, validate
from libsbgnpy.schematron import validate_schematron
import re
for i in sample:
    p = Path(f"{B}/pathwhiz_exports/{i}.sbgn")
    if not p.exists():
        res["sbgn"][i] = {"missing": True}; continue
    ent = {}
    try:
        x = validate_xsd(p); ent["xsd_errors"] = len(x); ent["xsd_examples"] = [str(e)[:200] for e in x[:3]]
    except Exception as e:
        ent["xsd_exc"] = repr(e)[:200]
    try:
        v = validate(p); ent["spec_errors"] = len(v)
        c = collections.Counter(re.sub(r"[0-9.]+", "N", str(e))[:110] for e in v)
        ent["spec_kinds"] = c.most_common(6); ent["spec_examples"] = [str(e)[:220] for e in v[:2]]
    except Exception as e:
        ent["spec_exc"] = repr(e)[:200]
    try:
        s = validate_schematron(p); ent["sch_issues"] = len(s)
        c = collections.Counter((x.rule_id, x.message[:110]) for x in s)
        ent["sch_kinds"] = [(list(k), n) for k, n in c.most_common(8)]
    except Exception as e:
        ent["sch_exc"] = repr(e)[:200]
    res["sbgn"][i] = ent
json.dump(res, open(f"{B}/a3/validation_raw.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print("ok")
