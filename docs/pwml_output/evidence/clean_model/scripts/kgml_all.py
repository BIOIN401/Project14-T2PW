"""Export all canonical runs to KGML (XML outside the repo) and write kgml_all.json / kgml_all.md.

  C:/.../.venv/Scripts/python.exe -I -X utf8 kgml_all.py [--runlist C:/t/ev-cm/runlist.json] [--out C:/t/ev-cm/kgml_out]
"""

import argparse
import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."] * 5, "src")))  # -I ignores PYTHONPATH

from t2pw.clean_model import build_from_run  # noqa: E402
from t2pw.clean_model.export.kgml import write_kgml  # noqa: E402


def structural_ok(path):
    """Required attributes, unique ids, references resolve (same rules as tests/test_clean_model_kgml.py)."""
    req = {"pathway": {"name", "org", "number"}, "entry": {"id", "name", "type"}, "reaction": {"id", "name", "type"},
           "relation": {"entry1", "entry2", "type"}, "subtype": {"name", "value"}, "substrate": {"id", "name"},
           "product": {"id", "name"}, "component": {"id"}}
    root = ET.parse(path).getroot()
    if any(not req.get(e.tag, set()) <= set(e.attrib) for e in root.iter()):
        return False
    ids = [e.get("id") for e in root.findall("entry")]
    allids = ids + [r.get("id") for r in root.findall("reaction")]
    return (len(ids) == len(set(ids)) and len(allids) == len(set(allids))
            and all(c.get("id") in ids for c in root.iter("component"))
            and all(r.get("entry1") in ids and r.get("entry2") in ids for r in root.findall("relation"))
            and all(s.get("id") in ids for s in root.iter() if s.tag in ("substrate", "product")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runlist", default="C:/t/ev-cm/runlist.json")
    ap.add_argument("--out", default="C:/t/ev-cm/kgml_out")
    ap.add_argument("--report", default=os.path.abspath(os.path.join(HERE, "..", "kgml_all")))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [r for r in json.load(open(a.runlist, encoding="utf-8")) if r.get("q") is True]
    res = {"runs": len(runs), "files_written": 0, "structural_ok": 0, "failures": []}
    tot, om_tot = Counter(), Counter()
    for i, r in enumerate(runs):
        d = r["dir"]
        try:
            m = build_from_run(d)
            path = os.path.join(a.out, "%03d_%s.kgml" % (i, os.path.basename(os.path.dirname(d.rstrip("/\\")))))
            om = write_kgml(m, path)
            res["files_written"] += 1
            res["structural_ok"] += bool(structural_ok(path))
        except Exception:
            res["failures"].append({"dir": d, "error": traceback.format_exc()[-500:]})
            continue
        tot.update(om["stats"])
        for k in ("unmapped_entities", "skipped_reactions", "skipped_interactions", "skipped_group_members"):
            om_tot[k] += len(om[k])
        om_tot["direction_placeholder"] += om["direction_placeholder"]["count"]
        nf = om["no_slot_in_format"]
        for k in ("quotes", "stoichiometry_participants", "catalysts_and_modifiers", "compartments"):
            om_tot["lost_" + k] += nf[k]
        om_tot["lost_transport_locations"] += len(nf["transport_locations"])
        for p in om["non_kegg_prefixes_used"]:
            om_tot["runs_using_prefix_" + p] += 1
    e = max(tot["entries"], 1)
    res["totals"] = dict(tot)
    res["entry_share"] = {k: round(tot["entries_" + k] / e, 4) for k in ("kegg", "non_kegg", "undefined")}
    res["omission_totals"] = dict(om_tot)
    json.dump(res, open(a.report + ".json", "w", encoding="utf-8"), indent=1, sort_keys=True)
    md = ["# KGML export of all canonical runs", "",
          "XML written to `%s` (outside the repo). Structural check = required attributes, unique ids, references resolve (no KEGG service is called)." % a.out, "",
          "- runs: %d, files written: %d, structural check passed: %d, failures: %d" % (res["runs"], res["files_written"], res["structural_ok"], len(res["failures"])),
          "- entries: %d; KEGG id %d (%.1f%%), non-KEGG prefix %d (%.1f%%), undefined %d (%.1f%%)" % (
              tot["entries"], tot["entries_kegg"], 100 * tot["entries_kegg"] / e, tot["entries_non_kegg"], 100 * tot["entries_non_kegg"] / e,
              tot["entries_undefined"], 100 * tot["entries_undefined"] / e),
          "- reactions (incl. transports): %d; rn: %d, rhea: %d, undefined %d" % (tot["reactions"], tot["reactions_rn"], tot["reactions_rhea"], tot["reactions_undefined"]),
          "- relations: %d; signed %d, unsigned %d" % (tot["relations"], tot["relations_signed"], tot["relations_unsigned"]),
          "", "## Omission totals", ""] + ["- %s: %d" % (k, v) for k, v in sorted(om_tot.items())]
    open(a.report + ".md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
