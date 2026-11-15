# Independent verification of three headline numbers

Corpus (M): 192 final_mapped.json (runs_papers 13, runs_validation 11, runs_verify 129, runs_smoke 23, runs 14,
e2e4 1, e2e5 1). Quarantine status is `ok` in `quarantine_report.json` next to final_mapped.json: ok=true in 152,
ok=false in 8, no report file in 32. Export mode is `export_mode` in the same report: 101 pathwhiz + 51 research
among the 152. Scripts: docs/pwml_output/evidence/verify/ (survey.py, c1.py, c2.py, c2c.py, c3.py).

## Claim 1: 40 of 101 PathWhiz-mode runs wrote no .pwml, holding 276 processes
Earlier: 40/101, 276. Mine: 40/101, 276. **AGREE.**
Method: 152 ok runs, filtered to export_mode=pathwhiz (101); "no .pwml" = no `*.pwml` in the run folder (a
recursive walk gives the same; none in the parent either); 61 have one, 40 do not. Processes = len(reactions)
+ len(transports) + len(interactions) in `final_mapped.json["processes"]` = 183 + 32 + 61 = 276 (reactions,
transports, interactions; no other process key exists).
Difference: none.

## Claim 2: 197 of 464 protein/enzyme units failed; compounds 147 of 517
Compounds: earlier 147/517. Mine: 148 of 520 units with no id (152 if a unit counts as failed when any run
of it failed). **AGREE** (within 1%).
Proteins/enzymes: earlier 197/464 (42%: 98 unmapped + 99 placeholder). Mine: **DISAGREE**.
- Proteins only (`entities.proteins`, 787 rows, 308 distinct (paper, normalized name) units, 52 papers): 88
  fail in every run of the unit (90 in any run) = 29%. Failure = none of uniprot, uniprot_id,
  pathbank_protein_id, pw_protein_id, mapped_ids set (unmapped, 53 units), or placeholder = identity_status
  "placeholder" / pathbank_protein_id 9659 / name "Unknown" (35 units). Row level: 75 unmapped + 43 placeholder of 787.
- Adding `protein_complexes` (415 rows, 264 units; my cruder rule: placeholder if identity_status placeholder,
  mapped if pathbank_complex_id or mapped_ids) gives 572 units, 298 failed (52%). Neither matches 464/197.
- Compounds: id = any of pathbank_compound_id, db_id, pwc_id, pw_compound_id, hmdb/kegg/chebi/pubchem/cas/
  chemspider/biocyc/drugbank id, mapped_ids. Units 520 (148 fail, 28%).
Cause of the protein gap (G, from reading 02-mapping-failures-and-homology.md after computing): the earlier table
excludes 39 "Unknown" sentinel rows, merges proteins with complexes into 464 units, uses the complex's wrapper
status, and adds 29 unmapped units seen only in the quarantine reports (entities removed from final_mapped.json).
My count uses final_mapped.json only, so quarantine-dropped proteins are absent by construction. The 21%/21%
split cannot be reproduced from final_mapped alone; the 42% headline is not supported on a proteins-only basis
(29%) and depends on the complex and quarantine definitions. Compound failure rate is the robust number.

## Claim 3: qa.py on the 61 canonical runs that have a .pwml: 36 pass, 25 fail, all on interaction issues
Earlier: 36/25, interaction only. Mine: 36 pass, 25 fail, 25 of 25 fail only on interaction errors (350 error
strings, all interaction class). **AGREE.**
Method: `from t2pw.pwml.qa import run_pwml_qa` with PYTHONPATH=C:/t/oa/src (entry point
`run_pwml_qa(pwml_bytes) -> {ok, errors, warnings, stats}`), applied to the single .pwml in each of the 61 runs
(each run has 0 or 1). Errors classed by text: interaction-type not Activation/Inhibition, `<name>` child on
interaction, element-count, missing visualization, sides. Zero spontaneous+enzyme, stoichiometry, orphan-location
or XML parse errors.
Difference: none.

## Reconciliation result (added by orchestrator-dispatched check)
Claim 2 disagreement resolved: definitions differ, not errors. Frame rules on the proteins bucket with Unknown kept reproduce 308 units / 88 fail exactly; 464/197 = 435 final-output units (168 fail, 38.6%) + 29 quarantine-only unmapped. See the RECONCILIATION section at the end of assessment/02-mapping-failures-and-homology.md.
