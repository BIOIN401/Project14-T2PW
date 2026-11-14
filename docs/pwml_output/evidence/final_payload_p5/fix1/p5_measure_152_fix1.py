"""P5 ontoBE: live OLS4 + Cellosaurus measurement over the 152 stored runs (final_mapped.json + 01_source_text.txt).

For every run the real payload is passed through ``fill_ontology_ids`` (organism context, species rows and the paper text of that run),
and every name the vocabulary map does not resolve is collected. Output: p5_measure_152.json (raw), p5_measure_152.md (tables).
Network: OLS4 and api.cellosaurus.org only, 1 request per second per host, own cache C:/t/p5scratch/ontoBE/cache (never data/*.json).
Run through the bounded wrapper:  python bounded_run.py --timeout 3000 -- python p5_measure_152.py
"""
import collections
import copy
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
from t2pw.clean_model.fill.cache import Cache  # noqa: E402
from t2pw.clean_model.fill.http import Throttle, UrllibFetcher  # noqa: E402
from t2pw.clean_model.fill.sources import OLS4  # noqa: E402
from t2pw.pwml.vocabulary_map import map_vocabulary_name, membrane_row  # noqa: E402
from t2pw.service.final_payload_ontology import _SPECS, fill_ontology_ids, location_organisms  # noqa: E402

HERE = Path(__file__).parent
TODAY = datetime.date.today().isoformat()
P = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
runs = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x["q"]]
fetcher = UrllibFetcher(throttle=Throttle(interval=1.0))
cache = Cache("C:/t/p5scratch/ontoBE/cache")
KIND = {"subcellular-location": "location", "cell-type": "cell_type", "tissue": "tissue"}

names = {}  # (kind, name) -> info
membrane_occurrences = []
for idx, r in enumerate(runs):
    d = Path(r["dir"])
    payload = json.load(open(d / "final_mapped.json", encoding="utf-8"))
    txt = next((c for c in (d / "01_source_text.txt", d.parent / "01_source_text.txt") if c.exists()), None)
    text = txt.read_text(encoding="utf-8", errors="replace") if txt else None
    work = copy.deepcopy(payload)
    ents = payload.get("entities") or {}
    species = [str(s.get("name")) for s in ents.get("species") or [] if isinstance(s, dict)]
    # names the fill will look at (same enumeration as fill_ontology_ids, before it runs)
    todo = []
    for key, kind, state_keys, onto, prefix, branch in _SPECS:
        seen = set()
        cand = [str(x.get("name")).strip() for x in ents.get(key) or [] if isinstance(x, dict) and x.get("name")]
        for s in payload.get("biological_states") or []:
            n = next((str(s[k]).strip() for k in state_keys if s.get(k) not in (None, "")), "")
            if n:
                cand.append(n)
        rows = {str(x.get("name")).strip(): x for x in ents.get(key) or [] if isinstance(x, dict) and x.get("name")}
        for n in cand:
            if n.casefold() in seen:
                continue
            seen.add(n.casefold())
            orgs = location_organisms(payload, n, state_keys) if kind == "subcellular-location" else None
            outcome = map_vocabulary_name(kind, n, None, orgs)
            if orgs is not None and membrane_row(n):
                membrane_occurrences.append({"run": idx, "dir": str(d).replace("\\", "/").replace(P, ""), "species": species, "name": n,
                                             "organisms": sorted(orgs), "outcome": outcome[2], "written_name": outcome[0],
                                             "had_ontology_id": bool((rows.get(n) or {}).get("ontology_id"))})
            has_id = bool((rows.get(n) or {}).get("ontology_id") or (rows.get(n) or {}).get("ontology-id"))
            if has_id or outcome[2] in ("exact", "mapped"):
                continue
            todo.append((kind, n, sorted(orgs) if orgs else None))
    rep = fill_ontology_ids(work, fetcher, cache, TODAY, paper_text=text, paper_source="01_source_text.txt")
    ent = {(e["kind"], e["entity"]): e for e in rep["entries"] if e["field"] == "ontology_id"}
    cel = {(e["kind"], e["entity"]): e for e in rep["entries"] if e["field"] == "cellosaurus_id"}
    ref = {(e["kind"], e["name"]): e for e in rep["refused"]}
    for kind, n, orgs in todo:
        k = (KIND[kind], n)
        info = names.setdefault((kind, n), {"kind": KIND[kind], "name": n, "runs": [], "species": set(), "organisms": set(), "results": collections.Counter()})
        info["runs"].append(idx)
        info["species"].update(species)
        info["organisms"].update(orgs or [])
        if k in ent:
            e = ent[k]
            res = f"resolved {e['accession']} ({e['source']}, {e['match_type']}{', matched form ' + repr(e['matched_form']) if 'matched_form' in e else ''}" \
                  f"{', via ' + e['via'] + ': ' + repr(e['context_phrase']) if 'via' in e else ''})"
        elif k in cel:
            res = f"cellosaurus {cel[k]['accession']} ({cel[k]['label']}, {cel[k]['match_type']}); no CL id"
        elif k in ref:
            e = ref[k]
            res = f"unresolved: {e['reason']}" + (f" [cellosaurus: {e['cellosaurus']}]" if "cellosaurus" in e else "") + (f" {e['candidates']}" if "candidates" in e else "")
        else:
            res = "unresolved: ?"
        info["results"][res] += 1
    print(idx, len(todo), rep["requests"], rep["cache_hits"], flush=True)

# informational only: closest fuzzy OLS4 hits for names never resolved in any run (never used as ids)
closest = {}
for (kind, n), info in sorted(names.items()):
    if any(x.startswith(("resolved", "cellosaurus")) for x in info["results"]) and not any(x.startswith("unresolved") for x in info["results"]):
        continue
    onto = {"location": "go", "cell_type": "cl", "tissue": "bto"}[info["kind"]]
    d = fetcher.get_json(OLS4, {"q": n, "ontology": onto, "queryFields": "label,synonym", "fieldList": "obo_id,label,synonym", "rows": 4})
    closest[f"{info['kind']}|{n}"] = [(r.get("obo_id"), r.get("label"), r.get("synonym")) for r in ((d or {}).get("response") or {}).get("docs") or []][:4]

out = {"date": TODAY, "runs": len(runs),
       "names": [dict(v, runs=v["runs"], species=sorted(v["species"]), organisms=sorted(v["organisms"]), results=dict(v["results"])) for v in names.values()],
       "closest_fuzzy": closest, "membrane_occurrences": membrane_occurrences}
(HERE / "p5_measure_152.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
md = [f"# P5 ontoBE: names without a vocabulary row or ontology id in the {len(runs)} stored runs ({TODAY}, live OLS4 + Cellosaurus)", "",
      "| kind | name | runs | species | result per run (count) |", "|---|---|---|---|---|"]
for v in sorted(out["names"], key=lambda x: (x["kind"], x["name"].casefold())):
    md.append(f"| {v['kind']} | {v['name']} | {len(v['runs'])} | {', '.join(v['species'][:3])} | " + "; ".join(f"{k} x{c}" for k, c in v["results"].items()) + " |")
md += ["", "## inner / outer membrane occurrences", "", "| run | species | name | organisms | outcome | written as |", "|---|---|---|---|---|---|"]
for o in membrane_occurrences:
    md.append(f"| {o['run']} {o['dir'][-48:]} | {', '.join(o['species'][:2])} | {o['name']} | {', '.join(o['organisms'])} | {o['outcome']} | {o['written_name'] if o['outcome'] != 'unmapped' else '(not written under a row name)'} |")
(HERE / "p5_measure_152.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("done")
