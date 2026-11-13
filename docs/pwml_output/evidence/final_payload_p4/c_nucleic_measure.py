"""Measure D-W20c over the replayed payloads. Usage: python c_nucleic_measure.py OUT.json  (PYTHONPATH=C:/t/fp/src)."""
import collections, json, sys
from pathlib import Path
from t2pw.pwml.writer import nucleotide_type_for, _PROTEIN_TOKENS
from t2pw.service import final_payload_nucleic as fn

runs = [r for r in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if r["q"]]  # same filter as replay.py (idx = position)
out = {"runs": len(runs), "rows": []}
tot = collections.Counter()
for idx, run in enumerate(runs):
    d = Path(run["dir"])
    p = json.load(open(d / "final_mapped.json", encoding="utf-8"))
    nas = [r for r in (p.get("entities") or {}).get("nucleic_acids") or [] if isinstance(r, dict)]
    if not nas:
        continue
    txt_file = next((c for c in (d / "01_source_text.txt", d.parent / "01_source_text.txt") if c.exists()), None)
    text = txt_file.read_text(encoding="utf-8", errors="replace") if txt_file else ""
    tot["runs_with_na"] += 1
    tot["runs_with_na_no_text"] += 0 if text else 1
    for r in nas:
        tot["na_rows"] += 1
        if r.get("pathwhiz_id") or r.get("pathbank_nucleic_acid_id") or r.get("pw_nucleic_acid_id"):
            tot["has_id"] += 1
            continue
        if nucleotide_type_for(r) is not None:
            tot["settled_by_name"] += 1
            continue
        tot["unsettled"] += 1
        prot = any(t in _PROTEIN_TOKENS for t in fn.re.split(r"[^a-z0-9]+", str(r.get("name") or "").casefold()))
        if prot:
            st, c, w = "protein_like", {}, {"status": "protein_like"}
        elif not text:
            st, c, w = "no_paper_text", {}, {"status": "no_paper_text"}
        else:
            c = fn.classify(text, r)
            st = c["status"]
            w = fn.classify(text, r, None)
        tot[st] += 1
        in_name = bool(c.get("cue")) and c["cue"] in [t for t in fn.re.split(r"[^a-z0-9]+", str(r.get("name")).casefold())]
        both = st == "resolved" and w["status"] == "resolved" and w["type"] == c["type"]
        tot["resolved_cue_in_row_name"] += in_name
        tot["resolved_and_sentence_wide_agrees"] += both
        tot["sentence_window_" + w["status"]] += 1
        out["rows"].append({"idx": idx, "run": str(d)[-90:], "name": r.get("name"), "status": st,
                            "type": c.get("type"), "cue": c.get("cue"), "quote": c.get("quote"), "offset": c.get("offset"),
                            "mentions": c.get("mentions"), "cued": c.get("cued"), "sentence_window_status": w["status"],
                            "sentence_window_type": w.get("type"), "cue_in_row_name": in_name, "agrees_sentence_wide": both, "source_file": str(txt_file) if txt_file else None})
out["totals"] = dict(tot)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps(out["totals"], indent=1))
