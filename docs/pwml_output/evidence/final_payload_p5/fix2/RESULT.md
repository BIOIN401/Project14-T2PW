# Final payload P5, fix round 2 (review 2: R2-1, R2-2, R2-3)

Code commit: `d44c835f` on `pwml/final-payload` (after the flip commit `5d1ac2ce`, so the tree was coherent). Files: `src/t2pw/pwml/writer.py`,
`src/t2pw/service/final_payload_nucleic.py`, `src/t2pw/service/final_payload.py`, new `tests/test_final_payload_fix2.py`, edited
`tests/test_final_payload_fix1.py` and `tests/test_final_payload_nucleic_lookup.py`.

## R2-1 (major) membrane suffix: FIXED
Verified: the rename fired for every membrane the wildcard guard kept, including an undetermined organism (unlisted genus, mixed
`{bacterium, unknown}`, `{bacterium, eukaryote}`), so "(non-bacterial, ...)" was a claim the code could not make.
Change: the mechanism stays (the written name still cannot match the bacterial PathWhiz row); the suffix is now
` (unresolved, no ontology id)`. The paper's own name is recorded in `builder.vocabulary_outcomes` as
`{kind: subcellular-location, name: <paper name>, outcome: membrane_kept_unresolved, written: <renamed>}` (the existing detail list of
non-exact vocabulary names; `BuildResult` is not changed). Counter `nonbacterial_membranes_kept_by_name` is now `membranes_kept_unresolved`.
Tests (`test_final_payload_fix2.py`): unlisted genus, `{bacterium, unknown}`, `{bacterium, eukaryote}`, all-eukaryote (each asserts the
suffix, no "non-bacterial" anywhere, counter, and the vocabulary_outcomes record); all-bacterial still maps to the PathWhiz row; flag off untouched.
Edited existing tests (legitimate consequence of the review, only the suffix string and the counter name): `fix1`
`test_f1_kept_nonbacterial_membrane_is_renamed_...` (2 asserts) plus the two other asserts naming the old counter.

## R2-2 (major) protein-coding -> DNA without paper evidence: FIXED
Verified: `_protein_context` returned False (pass) with no text, so the Streamlit path (no `final_payload_paper_text`) typed a protein-coding
gene as DNA unguarded (the CHL27 case). Now `_dna_context_code(text, row)` returns None only when at least one paper sentence names the row
(name or synonym) and none of those sentences has a protein word; otherwise `protein_context` (a protein word) or the new
`no_protein_context_evidence` (no text, or the row is never named). RNA gene types do not pass through this rule. New code is in `LOOKUP_CODES`.
Tests: no text (`""` and `None`) -> unknown + `no_protein_context_evidence` + nothing written; text that never names the row -> same; naming
sentence without protein word -> DNA (also via a synonym); one protein sentence among naming sentences -> `protein_context`; ncRNA with no
text still RNA. Edited existing tests (text fixture only, assertions unchanged): `nucleic_lookup`
`test_protein_coding_alias_is_dna_and_symbol_beats_alias` and `test_gene_type_map[protein-coding-DNA]` (the gene-type map test now passes
`text="Foo1 was measured in leaves."` for every type).

### 152-run re-measure (live public APIs were not needed: every lookup was answered from the local cache C:/t/p5scratch/nucD/cache, `lookup_requests` 0)
Script `nucleicD/d_nucleic_measure.py` unchanged, run through the bounded wrapper: `d_nucleic_measure_r22.json` / `.stdout.txt`.

| | resolved by lookup | unknown after lookup |
|---|---|---|
| 6 runs with nucleic acids, 39 rows (4 settled by name, 10 by paper, 35 unsettled by name) | **1** (miR171c, RNA) | **24** (13 `no_symbol_name`, 7 `not_attempted`, 3 `not_found`, 1 `protein_context` = CHL27) |

`no_protein_context_evidence` count: 0, because all 152 stored runs have paper text and the one protein-coding row that reaches the lookup (CHL27)
is named with a protein word. Counterfactual with the previous rule on the same code and cache (`d_nucleic_measure_old_rule.py`): identical
rows and totals, so **R2-2 itself changes 0 of the 152-run rows**. The drop from the P5 round-1 measure (4 resolved: RyhB, miR171c, LHCB1.1, CHL27)
to 1 comes from fix round 1 (F2: `conflicting` rows are not looked up: RyhB run 105 and LHCB1.1; F3: CHL27 `protein_context`), not from this round.
Where R2-2 matters is the Streamlit path (no text), which this measure does not exercise; that path is covered by the unit tests only.

## R2-3 (minor) shared 300 s budget: FIXED (title own budget)
Verified: `maybe_apply_final_id_fill` gave one fetcher (one 300 s budget) to fill, nucleic lookup and title, so a slow fill silently stopped the title
and reference (before phase 5 the title had its own budget).
Change: the title/reference step builds its own `_live_fetcher(TITLE_BUDGET_SECONDS)` (60 s) in `apply_final_pathway_meta` when the caller gave no
fetcher. Order is unchanged (fill, nucleic, title). Independence checked: the title reads only `meta` (`_fill_pathway` in the clean-model builder);
nothing the fill writes. A caller-given fetcher is still used by all three steps. `_Budget.exhausted` is set when a request is refused; `apply_final_id_fill`,
`maybe_apply_paper_nucleic_types` and `apply_final_pathway_meta` each report `budget_exhausted` (bool; a fake fetcher has none: False). Docstring "known
limit (2)" updated (worst case about 360 s of live requests, was 600).
Tests: with a fake clock (only `final_payload.time` is faked) every non-title request takes 400 s; the fill spends its budget (`budget_exhausted` True)
while the title is written from NCBI with its reference and `budget_exhausted` False; and a spent budget on a given fetcher is reported and the title
falls back to the given title. Edited existing test (consequence of the design): `fix1` F9, renamed
`test_f9_fill_and_nucleic_share_one_fetcher_the_title_has_its_own`: the title now gets no shared fetcher (`meta: None`); a caller-given fetcher is
still passed to all three.

## Tests
`tests/test_final_payload_*.py`, `tests/test_pwml_writer*.py`, `tests/test_pwml_ir*.py` (15 files), bounded wrapper with the heavy-lock retry
loop (`tests/focus2.stdout.txt`): **304 passed, 1 failed, FINAL SURVIVING COUNT 0**. The one failure is the known environmental
`test_pwml_writer.py::test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance` (`db_resolution.available` is True because `C:/t/fp/.env`
exists; recorded in fix1/RESULT.md and writerA/RESULT.md; not touched by this change). `tests/focus1.stdout.txt` is the first run (2 failed: the same one
plus a bug in my own new test, fixed). Not run: SMOKE, the golden-seam tests (orchestrator).

## Not established / remaining risks
* No live-API re-measure: the cache answered everything, so no new request was made against NCBI / RNAcentral.
* The Streamlit path (no paper text) for nucleic acids is tested with fakes only; R2-2 will leave more rows unknown there (by design).
* `membrane_kept_unresolved` is recorded on the builder (`vocabulary_outcomes`) and counted in `layout_debug_counts`; neither is written to a file by the
  writer today, so the text version must read them from the builder object.
* The title budget (60 s) is a constant, not an env setting. If NCBI esummary is slow for more than 60 s the title still falls back to the name.
* `apply_final_id_fill` and `apply_final_pathway_meta` now build their live fetcher before their `try`; construction cannot realistically raise, and the
  `maybe_*` wrappers still catch everything.
