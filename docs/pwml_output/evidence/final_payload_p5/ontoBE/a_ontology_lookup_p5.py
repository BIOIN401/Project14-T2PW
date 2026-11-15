"""P5 re-run of the P4 ontology measurement (evidence/final_payload_p4/a_ontology_lookup.py), adapted.

Same 30 names as P4 (+ inner/outer membrane and plural/cell-line cases). Two passes:
  pass 1  identical input to P4 (names only: no species, no paper text), after the P5 code changes;
  pass 2  each name with the context of a stored run it occurs in (species rows, 01_source_text.txt, states using it).
Live OLS4 + Cellosaurus, 1 request/s per host, own scratch cache. Run through the bounded wrapper. Output a_ontology_lookup_p5.md/.json
"""
import copy
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import Throttle, UrllibFetcher  # noqa: E402
from t2pw.service.final_payload_ontology import fill_ontology_ids  # noqa: E402

HERE = Path(__file__).parent
NAMES = {  # as P4, plus the P5 cases
    "subcellular_locations": ["inner mitochondrial membrane", "nerve fiber", "apical", "cytoplasmic leaflet of the IM", "periplasmic leaflet of the IM",
                              "etioplast", "microsomal fraction", "saliva", "stroma", "synaptic cleft", "synaptic vesicle", "synaptic vesicles",
                              "thylakoid membrane"],
    "cell_types": ["epidermal cell", "Type II cell", "Type III cell", "Type III cells", "erythroid", "erythroid cells",
                   "geniculate ganglion neuron", "MG-63", "Saos-2"],
    "tissues": ["bone marrow", "central nervous system", "cotyledon", "flower", "hypocotyl", "petal", "root", "young leaf"],
}
EXTRA = {"subcellular_locations": ["inner membrane (IM)", "outer membrane (OM)", "outer membrane", "inner membrane"], "cell_types": ["granulosa cells"]}
RUNS = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x["q"]]
# name -> run index whose payload/text give the context (first run of the 152 where the name occurs)
CTX = {"MG-63": 23, "Saos-2": 23, "Type II cell": 2, "Type III cell": 2, "Type III cells": 3, "erythroid": 48, "erythroid cells": 33,
       "geniculate ganglion neuron": 151, "cytoplasmic leaflet of the IM": 75, "periplasmic leaflet of the IM": 75, "inner membrane (IM)": 76,
       "outer membrane (OM)": 76, "outer membrane": 1, "inner membrane": 94, "granulosa cells": 146, "synaptic vesicles": 3, "nerve fiber": 3,
       "apical": 10, "saliva": 10, "stroma": 7, "microsomal fraction": 125, "young leaf": 139}
today = datetime.date.today().isoformat()
fetcher = UrllibFetcher(throttle=Throttle(interval=1.0))
cache = Cache("C:/t/p5scratch/ontoBE/cache")


def run_one(key, name, with_context):
    payload = {"entities": {key: [{"name": name}]}, "biological_states": []}
    text = None
    if with_context and name in CTX:
        d = Path(RUNS[CTX[name]]["dir"])
        src = json.load(open(d / "final_mapped.json", encoding="utf-8"))
        payload = copy.deepcopy(src)
        for k in list(payload["entities"]):
            if k not in ("species", key):
                payload["entities"].pop(k)
        payload["entities"][key] = [r for r in payload["entities"].get(key, []) if isinstance(r, dict) and r.get("name") == name] or [{"name": name}]
        payload["biological_states"] = [s for s in src.get("biological_states", []) if isinstance(s, dict)]
        txt = next((c for c in (d / "01_source_text.txt", d.parent / "01_source_text.txt") if c.exists()), None)
        text = txt.read_text(encoding="utf-8", errors="replace") if txt else None
    rep = fill_ontology_ids(payload, fetcher, cache, today, paper_text=text, paper_source="01_source_text.txt")
    row = next((r for r in payload["entities"][key] if r.get("name") == name), {})
    e = next((x for x in rep["entries"] if x["entity"] == name), None)
    refused = next((x for x in rep["refused"] if x["name"] == name), None)
    return {"id": row.get("ontology_id"), "cellosaurus_id": row.get("cellosaurus_id"), "entry": e, "refused": refused, "extra": rep["extra"]}


res = {1: {}, 2: {}}
lines = ["# P5 re-run of the P4 ontology measurement", "",
         f"Run {today}, live OLS4 + Cellosaurus. Pass 1 = P4's input (names only). Pass 2 = the context of a stored run (species, paper text, states).", "",
         "| kind | name | P4 | pass 1 (P4 input) | pass 2 (run context) |", "|---|---|---|---|---|"]
p4 = {}
for ln in open(HERE.parent.parent / "final_payload_p4" / "a_ontology_lookup.md", encoding="utf-8"):
    c = [x.strip() for x in ln.split("|")]
    if len(c) > 6 and c[1] in NAMES:
        p4[c[2]] = c[4] + " " + c[6] if c[3] == "resolved" else "unresolved"


def show(r):
    if r["id"]:
        e = r["entry"] or {}
        return f"{r['id']} ({e.get('source')}, {e.get('match_type')}" + (f", form '{e['matched_form']}'" if "matched_form" in e else "") + \
               (f", via paper phrase '{e['context_phrase']}'" if "via" in e else "") + ")"
    if r["cellosaurus_id"]:
        return f"Cellosaurus {r['cellosaurus_id']} (no CL id)"
    ref = r["refused"] or {}
    if not ref:
        return "resolved by the vocabulary map (PathWhiz row, no lookup needed)"
    return "unresolved: " + str(ref["reason"]) + (f" [cellosaurus: {ref['cellosaurus']}]" if ref.get("cellosaurus") else "")


for key, names in NAMES.items():
    for n in names + EXTRA.get(key, []):
        r1, r2 = run_one(key, n, False), run_one(key, n, True)
        res[1][n], res[2][n] = r1, r2
        lines.append(f"| {key} | {n} | {p4.get(n, '(new)')} | {show(r1)} | {show(r2)} |")
n30 = [n for v in NAMES.values() for n in v]
for p in (1, 2):
    ids = sum(1 for n in n30 if res[p][n]["id"])
    cel = sum(1 for n in n30 if res[p][n]["cellosaurus_id"] and not res[p][n]["id"])
    lines.append(f"\nPass {p}: ontology id for {ids} of {len(n30)} names (P4: 13 of 30); a Cellosaurus accession but no CL id for {cel} more.")
lines.append("\nNotes: pass 2 'Type III cells' is refused (id_used_by_other_row) because run 3's states also name the exact row 'Type III cell', which claimed CL:0002285 first; "
             "'Type II cell' resolves only because the paper text of PMC7059786 is available (the stored e2e runs that hold the name have none). "
             "Cellosaurus ids are not counted as ontology ids: PathWhiz cell_types need CL:+7 digits.")
(HERE / "a_ontology_lookup_p5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "a_ontology_lookup_p5.json").write_text(json.dumps({str(p): {n: {k: v for k, v in r.items() if k != "extra"} for n, r in res[p].items()} for p in res}, indent=1, ensure_ascii=False), encoding="utf-8")
print("\n".join(lines[-3:]))
