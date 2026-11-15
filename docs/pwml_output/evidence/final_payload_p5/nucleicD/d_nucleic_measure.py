"""Measure D-W21(d) over the replayed payloads: paper rules, then NCBI Gene / RNAcentral lookup, then the unknown flag.

Usage: python d_nucleic_measure.py OUT.json [CACHE_DIR]   (PYTHONPATH=C:/t/fp/src; live public APIs, polite rate, own cache dir)
Adapted from final_payload_p4/c_nucleic_measure.py (same run list, same 152 stored runs, same text files).
"""
import collections
import copy
import datetime
import json
import sys
from pathlib import Path

from t2pw.clean_model.fill.cache import Cache
from t2pw.clean_model.fill.http import Throttle, UrllibFetcher
from t2pw.pwml.writer import _PROTEIN_TOKENS, nucleotide_type_for
from t2pw.service import final_payload_nucleic as fn

OUT = sys.argv[1]
CACHE = sys.argv[2] if len(sys.argv) > 2 else "C:/t/p5scratch/nucD/cache"
TODAY = datetime.date.today().isoformat()
fetcher = UrllibFetcher(throttle=Throttle(interval=0.5))
cache = Cache(CACHE)

runs = [r for r in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if r["q"]]  # same filter as replay.py (idx = position)
out = {"date": TODAY, "runs": len(runs), "rows": [], "totals": {}}
tot = collections.Counter()


def summarize(rep, t):
    t["paper_resolved"] += rep["resolved"]
    for k, v in (rep.get("lookup_by_source") or {}).items():
        t["lookup_resolved_" + k] += v
    for k, v in (rep.get("lookup_codes") or {}).items():
        t["unknown_lookup_" + k] += v
    for k, v in (rep.get("unknown_by_reason") or {}).items():
        t["unknown_paper_" + k] += v
    t["unknown"] += rep["unknown"]
    t["lookup_requests"] += rep["lookup_requests"]


for idx, run in enumerate(runs):
    d = Path(run["dir"])
    p = json.load(open(d / "final_mapped.json", encoding="utf-8"))
    ents = p.get("entities") or {}
    nas = [r for r in ents.get("nucleic_acids") or [] if isinstance(r, dict)]
    if not nas:
        continue
    txt_file = next((c for c in (d / "01_source_text.txt", d.parent / "01_source_text.txt") if c.exists()), None)
    text = txt_file.read_text(encoding="utf-8", errors="replace") if txt_file else ""
    taxids = [(s.get("name"), s.get("taxonomy_id")) for s in ents.get("species") or []]
    tot["runs_with_na"] += 1
    for r in nas:
        tot["na_rows"] += 1
        if r.get("pathwhiz_id") or r.get("pathbank_nucleic_acid_id") or r.get("pw_nucleic_acid_id"):
            tot["has_id"] += 1
        elif nucleotide_type_for(r) is not None:
            tot["settled_by_name"] += 1
        else:
            tot["unsettled"] += 1
    pp = copy.deepcopy(p)
    rep = fn.apply_nucleic_types(pp, text, fetcher=fetcher, cache=cache, today=TODAY, source=str(txt_file))
    summarize(rep, tot)
    for r0, r in zip(nas, pp["entities"]["nucleic_acids"]):
        if "nucleotide_type_status" not in r:
            continue
        ms = fn.mentions(text, r0, None)
        out["rows"].append({
            "idx": idx, "run": str(d)[-90:], "species": taxids, "name": r.get("name"), "synonyms": r.get("synonyms"),
            "status": r["nucleotide_type_status"],
            "type": r.get("nucleotide_type") or r.get("nucleotide_type_paper"),
            "by": "paper" if r.get("nucleotide_type_paper") else ("lookup" if r.get("nucleotide_type") else None),
            "paper_reason": r.get("nucleotide_type_unknown_reason"), "lookup_code": r.get("nucleotide_type_unknown_lookup"),
            "source": r.get("nucleotide_type_source"),
            "first_paper_mention": ms[0]["sentence"] if ms else None, "n_mentions": len(ms),
        })
    out["totals"] = dict(tot)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)  # checkpoint after every run

print(json.dumps(out["totals"], indent=1))
