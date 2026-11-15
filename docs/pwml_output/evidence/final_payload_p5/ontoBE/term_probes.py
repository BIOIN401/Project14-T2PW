"""Term-level facts quoted in why_no_id.md (live OLS4 v2 term fetch + Cellosaurus entry fetch, 1 request/s, no cache). Output term_probes.txt."""
import sys
import urllib.parse as u
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
from t2pw.clean_model.fill.http import Throttle, UrllibFetcher  # noqa: E402

f = UrllibFetcher(throttle=Throttle(interval=1.0))
K = "http://www.geneontology.org/formats/oboInOwl#has"


def val(v):
    return [x.get("value") if isinstance(x, dict) else x for x in (v if isinstance(v, list) else [v])] if v else []


out = ["OLS4 v2 term fetch: label | obsolete | exact synonyms | related synonyms"]
for onto, iri in [("go", "GO_0070258"), ("go", "GO_0043165"), ("go", "GO_0005886"), ("go", "GO_0009279"), ("go", "GO_0019867"), ("go", "GO_0005623"),
                  ("go", "GO_0043005"), ("go", "GO_0009570"), ("go", "GO_0009532"), ("go", "GO_0005797"), ("go", "GO_0005802"), ("go", "GO_0009898"),
                  ("go", "GO_0098567"), ("go", "GO_0042598"), ("bto", "BTO_0001596"), ("bto", "BTO_0000971")]:
    q = u.quote(u.quote("http://purl.obolibrary.org/obo/" + iri, safe=""), safe="")
    d = f.get_json(f"https://www.ebi.ac.uk/ols4/api/v2/ontologies/{onto}/classes/{q}") or {}
    out.append(f"{iri}: {val(d.get('label'))} | obsolete={d.get('isObsolete')} | exact={val(d.get(K + 'ExactSynonym'))} | related={val(d.get(K + 'RelatedSynonym'))}")
out.append("")
out.append("Cellosaurus entry fetch (api.cellosaurus.org/cell-line/<ac>): species, site, BTO/CLO cross-references")
for acc in ("CVCL_0426", "CVCL_0548"):
    c = ((f.get_json(f"https://api.cellosaurus.org/cell-line/{acc}", {"format": "json"}) or {}).get("Cellosaurus") or {}).get("cell-line-list", [{}])[0]
    out.append(f"{acc}: names={[(n['type'], n['value']) for n in c.get('name-list', [])][:6]} species={[s.get('label') for s in c.get('species-list', [])]} "
               f"category={c.get('category')} BTO/CLO xrefs={[(x['database'], x['accession']) for x in c.get('xref-list', []) if x['database'] in ('BTO', 'CLO')]}")
rel = (f.get_json("https://api.cellosaurus.org/release-info", {"format": "json"}) or {}).get("Cellosaurus", {}).get("header", {}).get("release", {})
out.append(f"Cellosaurus release: {rel}")
Path(__file__).with_suffix(".txt").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
