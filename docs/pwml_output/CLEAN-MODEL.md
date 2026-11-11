# CLEAN MODEL: what was built (2026-11-10, branch `pwml/clean-model`)

Built unattended while the PO was away. Decisions are P-14 to P-28 in `DECISIONS-PROPOSED.md`. Nothing here edits an existing file under `src/`; PWML output is unchanged (additive only, P-14).

## 0. NOT ESTABLISHED (read first)

1. **PWML is not rewired.** The model is proven lossless against the frozen payload (exact round trip), but no PWML is produced from it and no golden-seam byte comparison of a model-built PWML exists. Requirements section 6 line 2 ("build PWML from the clean model, byte-identical") is **not** met; only its precondition is (P-14, P-23).
2. **Titles are not in any local run file** (0 of 152). They come only from the online fill (NCBI), which was run on 10 runs, not all 152. PMID is present locally in 152 of 152, PMCID 150, DOI 150, organism 152.
3. **Online fill was measured on 10 runs only** (seed 20261110). Its precision rests on a hand check of 31 accepted fills in the FIRST pass (24 correct, 2 wrong, 2 questionable, 3 unverifiable); the check was not repeated on the refreshed pass. The 4 bad ones were all reaction-level EC/Rhea ids inferred from a catalyst; that inference is now off by default (P-27) and the refreshed numbers below exclude it. Rhea ids therefore are not filled by default at all.
4. **KGML**: all 152 files now parse with Biopython's `KGML_parser` (0 of 152 before the review fix: reaction ids pointed at no entry, and the pathway name needs a `path:` prefix). Cytoscape KEGGscape, KEGGgraph and pathview were NOT tried. Whether strict consumers accept `chebi:` / `uniprot:` / `undefined` entry names is untested (07b item 6). No KEGG service was called.
5. **SBML validity** is libsbml consistency checking (L3V2, unit checks off): 0 errors. Whether any SBML tool renders these files usefully (they have no initial amounts, no kinetics, interactions only in an annotation) was not tested.
6. Coverage is bounded by the frozen payload: 990 of 2293 KGML entity entries are `undefined` (no KEGG, ChEBI or UniProt id), 724 of 724 KGML reactions have no reaction id, and direction/stoichiometry are unstated for every reaction.
7. No human review of biology. One independent code review plus a re-review of the fixes (section 7).
8. The typed model does not yet give every payload key an exporter-readable home; keys without one are carried in `extras` (round trip stays exact, exporters ignore them). The review found DrugBank ids and compound class flags in this state; they are now typed. Other extras keys may remain.

## 1. What exists

Package `src/t2pw/clean_model/`:

| file | role |
|---|---|
| `model.py` | dataclasses; every value is `Field(value, Prov)` with label `paper / db / inferred / default / unknown`, source, accession, retrieval date, match type, quote, location, `frozen`. Objects: Pathway, Location, Compound, Protein, Complex, NucleicAcid, Reaction, Transport, Interaction, removed records, mapping records, `extras` (any payload key without a typed home) and `db_layer` (items, suggestions, conflicts, refused) |
| `builder.py` | `build_from_run(run_dir)`, `build_from_payload(payload, meta)`. Local files only, no network |
| `roundtrip.py` | `to_payload(model)`; acceptance `to_payload(build(p)) == p` |
| `vocab.py` | controlled interaction vocabulary, explicit `unsigned` class (signs only where the word states one) |
| `fill/` | online layer: `http.py` (generic User-Agent, no email, per-host spacing, retries), `cache.py` (gitignored `clean_model_cache/`), `sources.py` (UniProt, ChEBI current API + OLS4 for exact names, QuickGO, NCBI Taxonomy/esummary, Rhea), `fill.py` (`fill_model`) |
| `export/sbml.py`, `export/sbo.py` | SBML L3V2 exporter, static SBO table (ids verified on OLS4) |
| `export/kgml.py` | KGML exporter plus omissions report |

Tests: `tests/test_clean_model_*.py` (102 tests, no network) and `tests/clean_model_fill_fakes.py` (fake fetcher). Scripts and evidence: `docs/pwml_output/evidence/clean_model/` (`build_all`, `validate_sbml`, `fill_sample`, `kgml_all`, `MEASUREMENT.md`). One `.gitignore` line (`clean_model_cache/`).

## 2. How to run it

```
PYTHONPATH=<repo>/src  python -X utf8 -c ...   # (scripts below do the path insert themselves)
from t2pw.clean_model.builder import build_from_run
from t2pw.clean_model.fill.fill import fill_model            # network; cached
from t2pw.clean_model.export.sbml import sbml_for_run, write_sbml
from t2pw.clean_model.export.kgml import kgml_for_run, write_kgml
m = build_from_run(run_dir)            # run_dir holds final_mapped.json
xml = sbml_for_run(run_dir)          # unfilled; for a filled model: fill_model(m) then sbml_text(m) (exporters never call the network)
```
Scripts (run with `python -I -X utf8`): `build_all.py`, `validate_sbml.py` (needs python-libsbml; the package itself never imports it; evidence used a scratch venv), `fill_sample.py --runs N --seed S --out DIR`, `kgml_all.py`. Tests: through the bounded wrapper (TEST_MATRIX section 0).

## 3. Field and provenance scheme

- Label on every value: `paper` (stated by the paper; quote and location kept where the pipeline has them), `db` (database lookup: source, accession, retrieved, match type in accession / exact_label / exact_synonym / cross_reference / inchikey / gene_symbol_taxon), `inferred` (derived: suggestions, step order, location refs from a state), `default` (convention value: PWML-style stoichiometry 1 etc.; written as `unstated` outside PWML), `unknown` (absent).
- Frozen payload values are `frozen=True` and never overwritten. A conflicting database value is recorded in `db_layer.conflicts`, not applied (measured: 8 conflicts, 7 of them species-vs-strain taxon ids, e.g. 562 vs 83333).
- Database fills go to slots with no payload key, or to `db_layer`, so `to_payload(model)` still equals the frozen payload after a fill (checked on 10 of 10 filled runs).
- Name lookups for compounds: exact label or exact synonym, accepted only when one distinct ChEBI id matches; several matches become a suggestion; an id already used by another entity is refused (narrowing guard, D-109). Protein candidates for id-less proteins are `inferred` suggestions only, never in an id field (0 made in the sample). No id is given to complexes or classes.
- Unknown-sentinel complex members are recorded as `unknown` members, never as entities.

## 4. Measured numbers

All from `evidence/clean_model/`. Canonical set: 152 runs (P-01).

| measure | value | requirement-6 target |
|---|---|---|
| runs built | 152 / 152 | |
| exact round trip `to_payload(build(p)) == p` | 152 / 152, 0 diffs | zero paper-stated field lost: met |
| PMID / PMCID / DOI / organism present | 152 / 150 / 150 / 152 | "PMID where the run has it": met; title 0 locally, 10 / 10 after fill |
| reaction / transport / interaction quotes retained | 621/621, 106/106, 351/352 | 621 retained: met |
| interaction spellings -> controlled kinds | 48 exact (39 normalised) -> 13 kinds; 117 signed, 235 explicitly unsigned | controlled vocabulary with unsigned class: met |
| complexes with Unknown members | 138 / 415 (all-Unknown), kept as unknown members | |
| interactions with undeclared endpoint | 15 | |
| SBML (no fill), all 152 | 152/152 read by libsbml, 0 errors, 2718 warnings all id 80601 (no initial amount; deliberate), 2718 species, 727 reactions | |
| SBML with fill, 10 runs | 0 errors; `bqbiol:is` resources 255 -> 354; 85 Ensembl transcript/gene and 32 KEGG gene ids as `isEncodedBy` (not `is`) | |
| KGML, all 152 | 152/152 pass own checker and parse with Biopython; 2293 entity entries (+193 reaction placeholder entries): 556 KEGG id (24.2%), 747 `chebi:`/`uniprot:` (32.6%), 990 `undefined` (43.2%); 724 reactions, 0 with an id; 325 relations, 106 signed, 219 unsigned | |
| live fill, 10 runs (refreshed, after review fixes) | first pass 118 live requests + 26 cache hits, 0 failures; refreshed pass 4 live + 133 cache; 493 fields and 252 ids filled: UniProt 318, ChEBI 135, NCBI 30, GO 10; 8 ambiguous compound names, 8 conflicts; titles, authors, year 10 of 10 | |
| compound ChEBI by name (10 runs) | accepted 4, ambiguous 8, no match 11 | |
| hand check of 31 accepted fills (first pass) | 24 correct, 2 wrong, 2 questionable, 3 unverifiable (all 4 bad = reaction EC/Rhea, now off) | |

## 5. Design choices worth knowing (details P-26 to P-28)

- SBML: one compartment per location name; one species per (entity, location); no initialAmount; `reversible="false"` only as the L3-required placeholder with `t2pw:directionStated=false`; stoichiometry omitted unless paper-stated; transport reaction SBO:0000655; interactions kept in a `t2pw` annotation (SBML core has no interaction object), unsigned marked unsigned; omissions list (removed/quarantined records, refused fills) and licence lines inside the file.
- KGML: documented placeholders for the required pathway attributes (`name="path:undefined"`, P-29), entries named with a KEGG id only if the model already holds one, otherwise `chebi:`/`uniprot:`, otherwise `undefined`; unsigned relations have no subtype; omissions sidecar JSON lists what KGML cannot hold (compartments, stoichiometry, quotes).
- No live KEGG call exists in code (P-17).

## 6. Failed attempts and dead ends

- ChEBI's current API has no exact label/synonym search (its `es_search` is fuzzy), so exact name lookup goes through OLS4; QuickGO search is fuzzy, so GO matching fetches term synonyms and re-checks exactness; the old ChEBI `webservices` service is not used (HTTP 500).
- libsbml error 10402 on the first SBML pass (several elements per namespace in an annotation); fixed by one `t2pw:info` wrapper.
- Reaction EC/Rhea inferred from a catalyst alone gave wrong ids (2 wrong, 2 questionable of 4 hand-checked). Disabled by default; also `uniprot_entry` reads EC only from recommended/alternative names and misses EC on included domains.
- `python -I` ignores `PYTHONPATH`, so scripts insert `src` on `sys.path` themselves.
- The first KGML design wrote fresh reaction ids; Biopython refused all 152 files. Fixed by reusing the catalyst entry id (P-29).
- Independent review (opus) also found: an "Unknown" name could still become a species via reactant/product/transporter references; Ensembl transcript ids were written as `bqbiol:is` on a protein; cache hits stamped today's date; GO `is_a` ancestors were written as `isPartOf`; exporters could call the network via `fill=True`. All fixed (P-30).
- The first pytest run of a new test folder errors if the parent of `--basetemp` does not exist.

## 7. Review and gates

@@REVIEW@@

## 8. Open questions for the PO

1. **D-023 / SBML (P-16).** Confirm the SBML build in `docs/webserver/DECISIONS.md` (this run did not edit it). Cost of a no: delete `export/sbml.py`; the model stays.
2. **Rewire PWML to the clean model?** Gains: one source of truth, gene names and title/references into PWML (needs your yes, B6). Cost: golden-seam byte proof on 152 runs; a separate task. Option: keep PWML as is and only use the model for other formats.
3. **Unsigned relations in non-PWML formats (Q-02, P-18).** Kept and marked unsigned. If you prefer them dropped from KGML, one switch in the exporter.
4. **Compound name lookup in the database layer (Q-04, P-19).** Allowed here, labelled `db`. It does not touch the PWML. Moving it into the pipeline before the freeze would raise compound ids (up to 39% of the 28% without), but it is a logic change needing a D-entry.
5. **KGML unmapped entities (P-24, P-28).** Choice made: placeholders and prefix names. Alternatives: omit the entry (drops biology), refuse the file. Also unknown: whether strict KGML consumers accept non-KEGG prefixes. Test with Cytoscape KEGGscape / KEGGgraph (owner time) before promising a KGML export.
6. **KEGG licence (P-17).** Option A holds. A written reply from Pathway Solutions is still pending; 556 KEGG ids in KGML come from the frozen payload, which already contains them.
7. **Online fill at scale.** Live calls ran on 10 runs. Running all 152 costs roughly 1,800 requests (about 12 per run), a few minutes; say if you want it as a committed measurement. Species-vs-strain taxon conflicts (562 vs 83333) are flagged as conflicts; a parent-taxon check would silence the noise.
8. **Other formats.** BioPAX is the best next target (it holds quotes, unsigned relations and needs no layout); not built this round.
9. **SBML compartment granularity and Unknown-backed complexes (06 Q4, Q5)** were decided provisionally (location name; complex with unknown members).
