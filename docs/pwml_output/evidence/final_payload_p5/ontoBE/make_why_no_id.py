"""Builds why_no_id.md from p5_measure_152.json (measurement over the 152 stored runs) plus the reasons written below.
Every reason cites what was queried (OLS4 exact search = exact=true, label+synonym, synonym hits confirmed on the fetched term; 'fuzzy' = the
same search without exact=true, informational only, never used as an id; Cellosaurus idsy search; PathWhiz app source read-only)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/t/fp/src")
from t2pw.pwml.vocabulary_map import map_vocabulary_name  # noqa: E402

HERE = Path(__file__).parent
M = json.loads((HERE / "p5_measure_152.json").read_text(encoding="utf-8"))
by = {(n["kind"], n["name"]): n for n in M["names"]}
CT_RULE = "PathWhiz `cell_types.ontology_id` must match `CL:\\d{7}` and be unique (smpdb app/models/data/biological_states/cell_type.rb line 11)"

# (kind, name) -> reason. kind: cell_type | location | tissue
R = {
    ("cell_type", "Type II cell"): "ambiguous by name: CL has no exact label/synonym 'Type II cell' (OLS4 exact 0 hits; fuzzy closest CL:0002063 pulmonary alveolar type 2 cell, CL:0000922 type II NK T cell, and the taste cell CL:0002286). The paper context selects one term only when the paper text is available: PMC7059786 carries the full phrase 'type II taste cell(s)' 3 times, the exact label of CL:0002286 and the only full 'type II ... cell' phrase, so the name resolves with the text (pass 2 of a_ontology_lookup_p5.md). The two stored runs holding the name (150, 151, e2e output folders) have no stored paper text and no full phrase in the payload, so it stays unresolved there.",
    ("cell_type", "Type III cell"): "CL has no exact 'Type III cell' (OLS4 exact 0; fuzzy closest CL:0002137 type III cell of adrenal cortex and CL:0002285 type III taste bud cell: ambiguous by name). The context phrase 'type III taste cell' (payload row 'Type III taste cell', or PMC7059786 text 4 times) selects CL:0002285 (PathWhiz row 'type III taste bud cell'); that phrase is NOT a CL label or synonym (OLS4 exact 0), it is the reviewed synonym `type iii taste cell` of vocabulary_map.py (marked 'reviewer to confirm'). Outcome in the 4 runs: resolved in 1 (run 3, from the paper text); refused id_used_by_other_row in 2 (runs 2 and 151) because the payload also carries the row 'Type III taste cell', which the writer already maps to that same PathWhiz row, so a second row would double it; run 150 has neither text nor a full phrase: unresolved.",
    ("cell_type", "Type III cells"): "same as 'Type III cell' (singular tried: OLS4 exact 0). Outcome in the 3 runs: resolved to CL:0002285 in 1 (run 10, paper phrase 'type III taste cell', reviewed synonym, not a CL label); refused id_used_by_other_row in run 3 (the exact singular row 'Type III cell' of the same payload claimed CL:0002285 first); run 150 has no text and no full phrase: unresolved.",
    ("cell_type", "erythroid"): "no CL term has the exact label or synonym 'erythroid' (OLS4 exact 0; fuzzy closest CL:0000038 erythroid progenitor cell, CL:0001066, CL:0000050): a progenitor or lineage term is not the cell type the paper names, not accepted; Cellosaurus has no cell line of that name.",
    ("cell_type", "erythroid cells"): "no CL term has the exact label or synonym 'erythroid cells' or the singular 'erythroid cell' (OLS4 exact 0 both; fuzzy closest CL:0000038 erythroid progenitor cell, CL:0000764 erythroid lineage cell): a lineage or progenitor is not the same entity, not accepted; Cellosaurus none.",
    ("cell_type", "geniculate ganglion neuron"): "CL holds only subtype terms (CL:4033157 geniculate ganglion TRPV1 neuron, CL:4033158 ... SPARCL1 neuron, CL:4033159 ... VGLUT2 neuron) and no generic 'geniculate ganglion neuron' (OLS4 exact 0); a subtype would assert more than the paper, and PathWhiz row 'Neuron' is the parent type (deliberately not mapped); Cellosaurus none.",
    ("cell_type", "MG-63"): f"cell line, not in CL (OLS4 CL exact 0). Cellosaurus has exactly one exact match, CVCL_0426 (identifier MG-63, Homo sapiens 9606, human payload species agrees), found in all 6 runs and kept in the payload as `cellosaurus_id` with `db` provenance. PathWhiz cannot take it: {CT_RULE}. Cellosaurus cross-references BTO_0001596 'MG-63 cell' (not used: a cell line is not a tissue row).",
    ("cell_type", "Saos-2"): f"cell line, not in CL (OLS4 CL exact 0). Cellosaurus has exactly one exact match, CVCL_0548 (identifier SaOS-2, exact synonym Saos-2, Homo sapiens 9606), found in all 6 runs and kept in the payload as `cellosaurus_id` with `db` provenance. PathWhiz cannot take it: {CT_RULE}. Cellosaurus cross-references BTO_0000971 'SAOS-2 cell' (not used: a cell line is not a tissue row).",
    ("cell_type", "idioblast"): "plant cell type absent from CL: OLS4 CL exact 0 and fuzzy 0 results; Cellosaurus none.",
    ("cell_type", "laticifer"): "plant cell type absent from CL: OLS4 CL exact 0 and fuzzy 0 results; Cellosaurus none.",
    ("cell_type", "internal phloem associated parenchyma"): "descriptive plant cell-type phrase, no CL term: OLS4 CL exact 0 and fuzzy 0 results; Cellosaurus none.",
    ("tissue", "young leaf"): "BTO has no exact label or synonym 'young leaf' (OLS4 exact 0, fuzzy 0 results); BTO:0000713 'leaf' (PathWhiz row 'Leaf') is the organ at any age, a wider entity: not mapped (vocabulary_map.py note).",
    ("tissue", "nerve fibers"): "BTO has no exact 'nerve fibers' or singular 'nerve fiber' (OLS4 exact 0 both); the only fuzzy hit BTO:0002191 'fibre tract' is a different structure.",
    ("location", "nerve fiber"): "GO has no exact label/synonym: GO:0043005 neuron projection lists 'nerve fiber' only as a RELATED synonym (term fetched, refused). UBERON:0006134 'nerve fiber' exists but PathWhiz `subcellular_locations.ontology_id` must match `GO:\\d{7}`, and a nerve fiber is an anatomical structure, not a cellular component.",
    ("location", "apical"): "an orientation, not a compartment: no GO cellular-component term has the exact label or synonym 'apical' (OLS4 exact 0; fuzzy hits are regulation processes such as GO:1905141); PathWhiz rows 'Apical cell membrane' / 'apical membrane' exist but the paper does not say membrane, so not mapped (vocabulary_map.py note).",
    ("location", "apical membrane of Type III cells"): "sub-domain of a cell type's membrane, no GO term of that name (OLS4 exact 0, fuzzy 0); the generic PathWhiz 'Apical cell membrane' would drop the cell-type qualifier: not mapped. State-only name from the e2e run 150.",
    ("location", "cytoplasmic leaflet of the IM"): "sub-compartment (one leaflet) of the bacterial inner membrane: no GO cellular-component term has this exact label or synonym (OLS4 exact 0, fuzzy 0). Nearest concept GO:0009898 'cytoplasmic side of plasma membrane' (exact synonyms 'internal side of plasma membrane', 'internal leaflet of plasma membrane') names other strings and rests on IM = plasma membrane; mapping to it or to PathWhiz 'Inner Membrane' would merge a leaflet with the whole membrane: not accepted.",
    ("location", "periplasmic leaflet of the IM"): "sub-compartment (one leaflet) of the bacterial inner membrane: no GO term with this exact label or synonym (OLS4 exact 0, fuzzy 0). GO:0098567 'periplasmic side of plasma membrane' has no synonyms and names another string; PathWhiz 'Inner Membrane' / 'Periplasm' would merge leaflet and whole: not accepted.",
    ("location", "cytoplasmic leaflet of inner membrane"): "same as 'cytoplasmic leaflet of the IM': no exact GO term (OLS4 exact 0, fuzzy 0), a leaflet is not the whole inner membrane.",
    ("location", "periplasmic leaflet of inner membrane"): "same as 'periplasmic leaflet of the IM': no exact GO term (OLS4 exact 0, fuzzy 0), a leaflet is not the whole inner membrane.",
    ("location", "microsomal fraction"): "not an ontology compartment: a laboratory fraction (a sample). GO has no term of that name (OLS4 exact 0, fuzzy 0); the nearest, GO:0042598 'vesicular fraction', is obsolete.",
    ("location", "saliva"): "a body fluid, not a cellular component: GO has none (OLS4 exact 0; fuzzy hits are salivary-gland regulation processes). UBERON:0001836 'saliva' exists but PathWhiz requires `GO:\\d{7}` for locations.",
    ("location", "stroma"): "no GO term has the exact label or synonym 'stroma' (OLS4 exact 0). GO:0009570 chloroplast stroma and GO:0009532 plastid stroma carry no 'stroma' synonym at all (terms fetched: exact [], related []), so choosing one would be a guess about which stroma the paper means; UBERON:0003891 'stroma' is generic tissue stroma and not a GO id.",
    ("location", "synaptic vesicles"): "resolved in 1 of 3 runs (run 3): the exact name has no hit, the singular 'synaptic vesicle' is the exact label of GO:0008021 (matched form recorded). In runs 10 and 150 the payload also has the row 'synaptic vesicle', which holds GO:0008021, so the plural row is refused id_used_by_other_row (one id, one row: it is a second name of the same compartment) and stays id-less.",
    ("location", "bacterial cell"): "the whole organism cell, not a sub-cellular compartment: no GO cellular-component term 'bacterial cell' (OLS4 exact 0; fuzzy hits are processes); GO:0005623 'cell' is obsolete.",
    ("location", "bulk cytosol"): "no GO term has the exact label or synonym 'bulk cytosol' (OLS4 exact 0, fuzzy 0); GO:0005829 'cytosol' is the closest, but 'bulk' is a qualifier in no label or synonym, so the exact-only rule does not accept it.",
    ("location", "cell"): "GO:0005623 'cell' is obsolete in current GO (OLS4 term fetch: label 'obsolete cell', isObsolete true), so no GO cellular-component term is labelled 'cell' (exact 0 within GO:0005575); CL:0000000 'cell' is a cell type, not a compartment. Note for the owner: `pwml/ir.py` SUBCELLULAR_LOCATION_CREATE_DEFAULTS still writes GO:0005623 for 'cell', which passes PathWhiz's format check but is an obsolete id (ir.py is outside this task).",
    ("location", "extracellular environment"): "no GO term has this label or synonym (OLS4 exact 0, fuzzy 0); GO:0005615 'extracellular space' (PathWhiz row 'Extracellular Space') is close in meaning but 'environment' is in none of its names, and vocabulary_map.py maps only 'extracellular'.",
    ("location", "leaf surface"): "an anatomical surface, not a cellular component: no GO term (OLS4 exact 0; the one fuzzy hit GO:0009897 'external side of plasma membrane' is unrelated).",
    ("location", "medial Golgi"): "GO has 'Golgi medial cisterna' GO:0005797 (exact synonym 'Golgi apparatus medial cisterna'); 'medial Golgi' is neither its label nor an exact synonym (OLS4 exact 0), so mapping would be a closest match.",
    ("location", "trans-Golgi"): "GO has 'trans-Golgi network' GO:0005802 (exact synonyms 'TGN', 'trans Golgi network'; 'Golgi trans face' only related); 'trans-Golgi' is neither label nor exact synonym (OLS4 exact 0), so mapping would be a closest match.",
    ("location", "Inner Membrane"): "organism-aware (P5): PathWhiz row 'Inner Membrane' is the bacterial inner membrane (98 % of its 1285 production states are prokaryote; e_pathwhiz_row_usage.txt), and these two runs are Homo sapiens, so the name is not written under that row. OLS4 GO exact for 'Inner Membrane' is 0 (GO has 'mitochondrial inner membrane' GO:0005743 and 'organelle inner membrane', but the paper does not say which organelle): unresolved, the PWML drops the location.",
    ("location", "inner membrane"): "organism undetermined (P5): the state using this name has no species and the payload also lists Arabidopsis thaliana next to E. coli, so the organism kinds are {bacterium, unknown}; not written under the bacterial row 'Inner Membrane'. OLS4 GO exact for 'inner membrane' is 0.",
}
MAPPED = [  # names the vocabulary map now resolves, not looked up
    ("cell_type", "granulosa cells", 1, "mapped to PathWhiz row 'Granulosa Cell' (CL:0000501): the singular is a row name, plural of the same cell type (new in P5)."),
    ("location", "plasma_membrane", 1, "mapped to PathWhiz row 'Cell Membrane' (GO:0005886, GO label 'plasma membrane'): underscore spelling of the same term (new synonym in P5)."),
]


def after(v):
    res = v["results"]
    if all(k.startswith("resolved") for k in res):
        return "resolved: " + "; ".join(sorted({k.split(" (")[0].replace("resolved ", "") + " (" + k.split(" (", 1)[1] for k in res}))
    if all(k.startswith("cellosaurus") for k in res):
        return "unresolved for PathWhiz; cellosaurus_id " + sorted(res)[0].split(" ")[1] + " kept in payload"
    if all(k.startswith("unresolved") for k in res):
        return "unresolved"
    return "; ".join(f"{k} x{c}" for k, c in res.items())


rows = []
for (kind, name), reason in R.items():
    v = by.get((kind, name))
    if v is None:
        raise SystemExit(f"{kind} {name} not in measurement")
    rows.append((kind, name, len(v["runs"]), after(v), reason))
for kind, name, n, reason in MAPPED:
    rows.append((kind, name, n, "resolved by the vocabulary map", reason))
seen = {(r[0], r[1]) for r in rows}
rest = [v for k, v in by.items() if k[1] and (v["kind"], v["name"]) not in seen]
md = ["# Why a name has no ontology id: per-name table (P5, ontoBE)", "",
      f"Measured {M['date']} over the {M['runs']} stored runs (`p5_measure_152.py`: every run's real payload and paper text through `fill_ontology_ids`, live OLS4 + Cellosaurus). "
      "Meaning of columns: *runs* = runs of the 152 in which the name occurs without a PathWhiz row or id; *after P5* = outcome of the new code. "
      "An id-less cell type, tissue or location is kept in the final payload and omitted from the PWML only (writer commit 0c320e27, wildcard guard keeps a row when dropping it would let a state bind to another).", "",
      "## A. Names without an id", "", "| kind | name | runs | after P5 | reason (what was queried) |", "|---|---|---|---|---|"]
for kind, name, n, aft, reason in sorted(rows, key=lambda r: (r[0], r[1].casefold())):
    md.append(f"| {kind} | {name} | {n} | {aft} | {reason} |")
md += ["", "Names that did resolve in the measurement (no reason needed, listed for completeness):", ""]
for v in sorted(rest, key=lambda x: (x["kind"], x["name"].casefold())):
    md.append(f"- {v['kind']} `{v['name']}` ({len(v['runs'])} runs): {after(v)}")
md += ["", "## B. What PathWhiz accepts as a cell-type id (read-only look at the app)", "",
       "- `app/models/data/biological_states/cell_type.rb`: `validates :ontology_id, uniqueness: true, presence: true, format: { with: /\\ACL:\\d{7}\\z/ }` (importer message 'Ontology can't be blank').",
       "- The same file defines `CELL_TYPE_DB = \"CELL ONTOLOGY\"`; `tissue.rb` takes `BTO:\\d{7}`, `subcellular_location.rb` takes `GO:\\d{7}`.",
       "- All 43 production `cell_types` rows (harness seed `vocab_inserts.sql`) carry CL ids; none carries a Cellosaurus (CVCL_) or any other prefix.",
       "- Therefore a Cellosaurus accession cannot be written to `cell_types`. MG-63 and Saos-2 are kept in the final payload (field `cellosaurus_id`, `db` provenance in `final_payload_fill`) and the PWML writer drops them because they carry no CL id.",
       "", "## C. Inner and outer membrane per occurrence (E)", "",
       "PathWhiz rows `Inner Membrane` (id 32, GO:0070258) and `Outer Membrane` (id 33, GO:0043165) are the bacterial membranes: 1271 of 1285 and 1232 of 1243 production states using them have a prokaryote species, mitochondria have their own rows 12 and 14 (`e_pathwhiz_row_usage.txt`). "
       "Their stored GO ids are other terms (OLS4: GO:0070258 'inner membrane pellicle complex', GO:0043165 'Gram-negative-bacterium-type cell outer membrane assembly'), so the rows are bound by name only and the ids are never written. "
       "Correct GO terms for reference: GO:0005886 plasma membrane (exact synonym 'cytoplasmic membrane'), GO:0009279 cell outer membrane, GO:0019867 outer membrane (generic).", "",
       "| run | species rows | name in the data | organisms of the states using it | before P5 (no organism) | after P5 (written as) |", "|---|---|---|---|---|---|"]
for o in M["membrane_occurrences"]:
    legacy = map_vocabulary_name("subcellular-location", o["name"], None)
    before = f"{legacy[2]}" + (f" '{legacy[0]}'" if legacy[2] in ("mapped", "exact") else " (kept by name or dropped, no row)")
    aft = f"{o['outcome']}: '{o['written_name']}'" if o["outcome"] == "mapped" else "unmapped: not written under a row name (location dropped from the PWML unless a state would collide)"
    md.append(f"| {o['run']} `{o['dir'][-52:]}` | {', '.join(o['species'][:3])} | {o['name']} | {', '.join(o['organisms'])} | {before} | {aft} |")
md += ["", "Leaflets (`cytoplasmic/periplasmic leaflet of the IM` and `... of inner membrane`, 8 occurrences in runs 68, 75, 76, 94) are never mapped to the whole membrane: see section A.", ""]
md += ["## D. Limits of this record (not established)", "",
       "- The two numbered-cell-type resolutions rest on one reviewed synonym (`type iii taste cell` -> `type III taste bud cell`, marked 'reviewer to confirm' in vocabulary_map.py); only `Type II cell` -> CL:0002286 rests on an exact CL label.",
       "- The paper text reaches the ontology fill only through `state['final_payload_paper_text']` (run_pathway with the flag on); the Streamlit path has none, so numbered names resolve there only from the payload's own names.",
       "- Organism kind comes from the species rows of the states that use the location (PathWhiz `classification` plus an exact genus table); a state without a species counts every species row of the payload, so a mixed-species payload leaves the name undetermined (run 68 `inner membrane`).",
       "- Two payload rows can still reach one PathWhiz row through the vocabulary map alone (e.g. `neurons` and `Neuron`); the ontology fill refuses to add to that (`id_used_by_other_row`) but does not merge existing duplicates.",
       "- An id-less row that the writer's wildcard guard keeps (a state would otherwise collide) is still written by name; for a name equal to a PathWhiz row name (here only the non-bacterial `Inner Membrane` / `Outer Membrane`) that binds the bacterial row at import. Not seen in the 14 replayed runs; no test covers it.",
       "- Cellosaurus: one release (57.0, 2026-10-22) and two cell lines were checked; the exact-match rule was not measured on other cell lines (HeLa, HEK293 and similar return many derivatives; a full page of 100 is refused).",
       ""]
(HERE / "why_no_id.md").write_text("\n".join(md), encoding="utf-8")
print(len(rows), "rows;", len(rest), "other resolved names")
