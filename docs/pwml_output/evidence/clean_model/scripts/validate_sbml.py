"""Build -> export SBML -> read with libsbml -> checkConsistency, per canonical run.

Run with the SCRATCH venv (python-libsbml lives only there):
  C:/t/ev-cm/venv/Scripts/python.exe -I -X utf8 validate_sbml.py [--runs N | --all] [--out C:/t/ev-cm/sbml_out] [--runlist C:/t/ev-cm/runlist.json] [--pick a,b,c]
Writes validate_sbml.json and validate_sbml.md next to this script's parent folder (evidence/clean_model/).
"""

import argparse
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."] * 5, "src")))  # -I ignores PYTHONPATH

import libsbml  # noqa: E402
from t2pw.clean_model import build_from_run  # noqa: E402
from t2pw.clean_model.export.sbml import sbml_text  # noqa: E402

OUTDIR = os.path.abspath(os.path.join(HERE, ".."))
SEV = {libsbml.LIBSBML_SEV_INFO: "info", libsbml.LIBSBML_SEV_WARNING: "warning", libsbml.LIBSBML_SEV_ERROR: "error",
       libsbml.LIBSBML_SEV_FATAL: "fatal"}


def check(path):
    doc = libsbml.readSBMLFromFile(path)
    res = {"read": doc is not None, "read_errors": [], "errors": Counter(), "warnings": Counter(), "messages": []}
    if doc is None:
        return res
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        res["read_errors"].append("%s %s %s" % (SEV.get(e.getSeverity()), e.getErrorId(), e.getShortMessage()))
    doc.setConsistencyChecks(libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)  # no units are claimed: unit checks only add noise
    n = doc.checkConsistency()
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        sev = SEV.get(e.getSeverity(), "?")
        if sev in ("error", "fatal"):
            res["errors"][str(e.getErrorId())] += 1
        elif sev == "warning":
            res["warnings"][str(e.getErrorId())] += 1
        if len(res["messages"]) < 40 and sev != "info":
            res["messages"].append({"sev": sev, "id": e.getErrorId(), "line": e.getLine(), "msg": e.getMessage().strip()[:300]})
    res["errors"], res["warnings"] = dict(res["errors"]), dict(res["warnings"])
    res["model_ok"] = doc.getModel() is not None
    if doc.getModel():
        m = doc.getModel()
        res["counts"] = {"compartments": m.getNumCompartments(), "species": m.getNumSpecies(), "reactions": m.getNumReactions()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="C:/t/ev-cm/sbml_out")
    ap.add_argument("--runlist", default="C:/t/ev-cm/runlist.json")
    ap.add_argument("--pick", default="", help="comma-separated substrings of run dirs to take first")
    a = ap.parse_args()
    runs = [r for r in json.load(open(a.runlist, encoding="utf-8")) if r.get("q") is True]
    if not a.all:
        chosen = []
        for sub in [s for s in a.pick.split(",") if s]:
            chosen += [r for r in runs if sub in r["dir"] and r not in chosen][:1]
        # spread the rest over the list
        step = max(1, len(runs) // max(1, a.runs))
        chosen += [r for r in runs[::step] if r not in chosen]
        runs = chosen[:a.runs]
    os.makedirs(a.out, exist_ok=True)
    out = {"runs": []}
    for i, r in enumerate(runs):
        d = r["dir"]
        name = "%03d_%s.xml" % (i, os.path.basename(os.path.dirname(d)))
        rec = {"dir": d, "file": name, "ne": r.get("ne"), "nr": r.get("nr"), "nt": r.get("nt"), "ni": r.get("ni")}
        try:
            text = sbml_text(build_from_run(d))
            path = os.path.join(a.out, name)
            with open(path, "wb") as fh:
                fh.write(text.encode("utf-8"))
            rec["bytes"] = len(text.encode("utf-8"))
            rec.update(check(path))
        except Exception as ex:  # report, never hide
            import traceback
            rec["exception"] = traceback.format_exc()[-800:]
        out["runs"].append(rec)
        print(name, rec.get("errors"), rec.get("warnings"), rec.get("exception", "")[-200:], flush=True)
    tot_e, tot_w = Counter(), Counter()
    for r in out["runs"]:
        tot_e.update(r.get("errors", {}))
        tot_w.update(r.get("warnings", {}))
    out["libsbml"] = libsbml.getLibSBMLDottedVersion()
    out["total_errors"], out["total_warnings"] = dict(tot_e), dict(tot_w)
    out["read_ok"] = sum(1 for r in out["runs"] if r.get("read"))
    with open(os.path.join(OUTDIR, "validate_sbml.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    md = ["# SBML validation (libsbml %s)" % out["libsbml"], "",
          "Runs: %d, read by libsbml: %d. Total errors by id: %s. Total warnings by id: %s." % (
              len(out["runs"]), out["read_ok"], out["total_errors"] or "none", out["total_warnings"] or "none"), "",
          "| file | entities/rx/tr/ix | species | reactions | errors | warnings |", "|---|---|---|---|---|---|"]
    for r in out["runs"]:
        c = r.get("counts", {})
        md.append("| %s | %s/%s/%s/%s | %s | %s | %s | %s |" % (r["file"], r["ne"], r["nr"], r["nt"], r["ni"], c.get("species"), c.get("reactions"),
                                                             r.get("errors", r.get("exception", "")[-80:]), r.get("warnings")))
    with open(os.path.join(OUTDIR, "validate_sbml.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
