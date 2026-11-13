"""Live OLS4 lookup of the 27 vocabulary names the P3 map left unmatched, plus 3 names the first version wrote (M2 re-run). Network: OLS only. Run: python a_ontology_lookup.py"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
from t2pw.clean_model.fill.cache import Cache
from t2pw.clean_model.fill.http import Throttle, UrllibFetcher
from t2pw.service.final_payload_ontology import fill_ontology_ids

NAMES = {
    "subcellular_locations": ["inner mitochondrial membrane", "nerve fiber", "apical", "cytoplasmic leaflet of the IM", "periplasmic leaflet of the IM", "etioplast",
                              "microsomal fraction", "saliva", "stroma", "synaptic cleft", "synaptic vesicle",
                              "synaptic vesicles", "thylakoid membrane"],
    "cell_types": ["epidermal cell", "Type II cell", "Type III cell", "Type III cells", "erythroid", "erythroid cells",
                   "geniculate ganglion neuron", "MG-63", "Saos-2"],
    "tissues": ["bone marrow", "central nervous system", "cotyledon", "flower", "hypocotyl", "petal", "root", "young leaf"],
}
payload = {"entities": {k: [{"name": n} for n in v] for k, v in NAMES.items()}, "biological_states": []}
today = datetime.date.today().isoformat()
rep = fill_ontology_ids(payload, UrllibFetcher(throttle=Throttle(interval=1.0)), Cache("C:/t/olscache2"), today)
res = {e["entity"]: e for e in rep["entries"]}
ref = {r["name"]: r for r in rep["refused"]}
lines = ["# A: OLS4 ontology lookup over the 27 unmatched names + 3 extra written names", "", f"Run {today}, live OLS4 (exact=true; exact label, or exact synonym only after the term is fetched and lists the name in hasExactSynonym; M2 fix). "
         f"Requests {rep['requests']}, cache hits {rep['cache_hits']}.", "",
         "| kind | name | result | id | label matched | match type / reason |", "|---|---|---|---|---|---|"]
for key, names in NAMES.items():
    for n in names:
        if n in res:
            e = res[n]
            lines.append(f"| {key} | {n} | resolved | {e['accession']} | {e['label']} | {e['match_type']} |")
        else:
            r = ref.get(n, {"reason": "?"})
            lines.append(f"| {key} | {n} | unresolved | | | {r['reason']} {r.get('candidates', '')} |")
lines += ["", f"Resolved {rep['counts']['resolved']} of {rep['counts']['looked_up']}; counts {rep['counts']}"]
Path(__file__).with_name("a_ontology_lookup.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(rep["counts"])
