# Final payload P5 fix round 1 (review 1, F1-F11)

Code and tests: commit `f4b0be9a` on `pwml/final-payload`. Review: `../review1/REVIEW.md`. Every finding was checked against the code first;
all of F1-F9 were real. Test runs: `../tests/fix1a|fix1b|fix1c.stdout.txt` (fix1a and fix1b contain a test mistake of mine in the new F6 test, fixed in fix1c).

## Per finding

| F | status | what was done |
|---|---|---|
| F1 | fixed | `writer.idless_unmapped` decides from the WRITTEN `ontology-id` (the vocabulary map drops the id of a non-bacterial membrane, so a human "outer membrane" + GO:0019867 is now droppable like any id-less row). When the wildcard guard forces such a row to stay, its name is rewritten to `<name> (non-bacterial, no ontology id)` so it cannot bind to the bacterial row, and `nonbacterial_membranes_kept_by_name` is counted. Tests: dropped case, kept case, flag off. See "Importer reading" below for why the record's own id is NOT written. |
| F2 | fixed (Option B) | reason `conflicting` -> lookup code `not_attempted`, no request (like `protein_like`). |
| F3 | fixed, proposed rule | comment/docstring no longer say "PO ruling". protein-coding -> DNA accepted only if no paper sentence naming the row or a synonym has a `_PROTEIN_TOKENS` word (`_protein_context`, sentence-level, whole sentence); else unknown with lookup code `protein_context` (new). With no paper text the check is vacuous (the earlier tests and the Streamlit path need that). |
| F4 | fixed | `Lookup.resolve`: a source that refused with anything but `not_found` (ambiguous, gene_type_not_decisive, too_many_records) blocks the other source's answer: lookup code `refused_by_a_source` (new). `not_found` from one source still lets the other resolve. |
| F5 | fixed | `numbered_phrases` no longer cuts; `numbered_context` refuses `too_many_phrases` when more than `MAX_PHRASES` (6). Counted in `numbered_context_refused`. |
| F6 | fixed | numbered cell-type names (`Type <numeral> cell`, singular or plural) sort after all other rows (then plural-looking names), so a row with its own exact CL label claims the id first. Note: "Type II taste cell" is itself a PathWhiz row, so it already claims its id through `claimed`; the test uses a name that needs the lookup. |
| F7 | fixed | `cellosaurus_for` wrapped: a FetchError counts `cellosaurus_refused["fetch_error"]` and also still `counts.fetch_error` (the existing test asserts that total), but only an OLS failure sets the flag that skips later lookups. |
| F8 | fixed | `organism_kind`: classification Prokaryote with an unlisted genus returns None. Stored-run check: the 152-run measurement is byte-identical to the ontoBE one (`p5_measure_152_fix1.md` vs `ontoBE/p5_measure_152.md`, empty diff); all 28 bacterial membrane occurrences (E. coli, Pseudomonas) still map; none relied on the old default. (The brief said 20; the measurement table has 28 `bacterium | mapped` rows.) |
| F9 | fixed | `maybe_apply_final_id_fill` builds `_live_fetcher()` once and passes it to the id fill, the nucleic lookup and the pathway title. Actual behaviour, now documented in the code: ONE budget (`T2PW_FINAL_PAYLOAD_BUDGET_S`, counted from the fetcher's construction) covers all three; a fill that spends it leaves the nucleic rows `unknown` (`lookup_error`) and the title on its fallback. A caller-given fetcher is used as is. `maybe_apply_paper_nucleic_types` called alone still builds its own. |
| F10 | fixed | docstring of `_apply_wildcard_guard` rewritten (the fixpoint loop re-checks; scope gap against a populated production DB stated). |
| F11 | fixed | comment only, `tests/test_final_payload_ontology.py` (the states differ by tissue, not by a kept location). |
| F12 | not done | outside this brief (needs a re-run of `replay.py` against an export of `0727df29`). |
| F13 | open | see proposed decisions. |
| F14 | see below | |

## Existing assertions that changed (reviewer: please check)

1. `tests/test_final_payload_nucleic_lookup.py::test_unknown_flag_reasons_and_never_a_type`: `rows[0]["nucleotide_type_unknown_lookup"]` was
   `"not_found"`, now `"not_attempted"` (that row is the `conflicting` one; F2). Not an env pin: a decided behaviour change.
2. `tests/test_final_payload_ontology_p5.py::test_organism_kind_tables_...`: `organism_kind("Unknownus bacter", "Prokaryote")` was `"bacterium"`,
   now `None` (F8). Same reason.
3. `tests/test_final_payload_ontology.py` comment only (F11).
No other assertion touched. No assertion was weakened: both changes assert the opposite outcome because the rule was reversed on purpose.

## Importer reading (F1), read-only, `smpdb/lib/pwml_parser.rb`, `app/models/data/biological_states/subcellular_location.rb`

`store_object` finds an existing row with `klass.find_by(attribute_hash)`, the hash being the attributes present in the PWML node (name and
ontology-id; an omitted ontology-id is not in the hash). A row is therefore found by NAME ALONE when no id is written, and MySQL compares
names case-insensitively under its default collation (not verified on the production schema). `SubcellularLocation` validates a unique name and a unique
`GO:\d{7}` ontology id. So (a) a bare "outer membrane" binds to the bacterial "Outer Membrane" row (the review's finding, confirmed);
(b) writing the record's own id (GO:0019867) would miss that row (different id) and then try to create a row whose name collides with it
(Rails' uniqueness check is case-sensitive by default; whether the DB has a unique index on `name` was not checked): a failure or a second
row of an existing name, not a clean result. Hence: drop it when the guard allows, rename it when the guard forces it to stay.
NOT established: the collation, the DB unique index, and the importer's behaviour on the renamed row (it cannot be created without an id, so
the import reports it, which is the same visible outcome as any other unresolved name that must stay).

## Known failure, `tests/test_pwml_writer.py::test_cli_export_emits_the_canonical_organism_and_keeps_its_provenance`

Environmental. The last assertion is `report["db_resolution"]["available"] is False`, whose comment says "no worktree carries a .env, so
`PathBankDbResolver.from_env()` answers None". `from_env` (src/t2pw/mapping/map_ids.py:894) calls `ensure_dotenv_loaded()` and reads
`PATHBANK_DB_HOST` / `PATHBANK_DB_USER`; with `C:/t/fp/.env` present the DB is reachable and `available` is True (the pytest message is `assert True is False`). Nothing in the diff
touches it. I did not edit it (the file is outside my edit list). Narrow pin proposed: in that test, make `PathBankDbResolver.from_env` return None
(`monkeypatch.setattr("t2pw.mapping.map_ids.PathBankDbResolver.from_env", classmethod(lambda cls, overrides=None: None))`); an env-only pin
(`monkeypatch.setenv("PATHBANK_DB_HOST", "")`) may not survive `ensure_dotenv_loaded` depending on its override setting, which I did not check. Not run against `3e8a66f3` (no stash allowed).
The `.env` copy in `C:/t/fp` must be removed after live runs (F14); I did not touch it.

## Tests (through the bounded wrapper, retry-the-lock pattern, `run_fix1.sh`)

* fix1c (final set): `tests/test_final_payload_*.py` (all 9 files incl. the new `test_final_payload_fix1.py`, 13 new tests),
  `tests/test_pwml_writer.py`, `tests/test_pwml_ir.py`, `tests/test_pwml_ir_duplicate_row_refusal.py`: **273 passed, 1 failed** (the environmental
  canonical-organism test above), 10.9 s, FINAL SURVIVING COUNT 0.
* fix1a (before the test fix): 272 passed, 2 failed (the same one plus my F6 test, which used a name that is itself a PathWhiz row), surviving 0.
* The SMOKE suite was not run (orchestrator's).

## Re-measure

Nucleic (`d_nucleic_measure_fix1.json`, same 152 stored runs, 6 runs with nucleic acids, 39 rows, 35 unsettled by name; own cache; `cmp_nuc_fix1.py` lists the changed rows):

| | before (D) | after |
|---|---|---|
| resolved by paper | 10 | 10 |
| resolved by lookup | 4 (CHL27 DNA, LHCB1.1 DNA, RyhB RNA idx 105, miR171c RNA) | 1 (miR171c RNA) |
| unknown | 21 | 24 |
| unknown lookup codes | not_found 5, not_attempted 1, no_symbol_name 14, species_ambiguous 1 | not_attempted 7, not_found 3, no_symbol_name 13, protein_context 1, species_ambiguous 1 |

Rows that lost their resolution: LHCB1.1 and RyhB (conflicting in the paper -> F2), CHL27 (protein subunit wording -> F3). Four more `conflicting`
rows (araO2, araI2, CHLH promoter, RyhB idx 84) were already unknown and are now not sent to the lookup (`not_attempted`).
All of that is the cautious direction by design; net 14 -> 11 resolved of 35 unsettled.

Ontology (`p5_measure_152_fix1.md/.json/.py`, the ontoBE script, 152 stored runs, live OLS4/Cellosaurus through its cache): output identical to
`ontoBE/p5_measure_152.md` (empty diff): F5/F6/F7/F8 change no number on these runs; 28 bacterial membrane rows mapped, 3 unmapped as before.

## Proposed decisions (the product owner has not ruled; text for DECISIONS-PROPOSED.md, not appended by me)

* **P-F2 (conflicting rows).** A nucleic-acid row for which the paper itself uses both DNA and RNA cues (`conflicting`) stays `unknown` and is
  not sent to the online lookup. Reason: a lookup would silently choose one reading against the paper's own text (LHCB1.1 became DNA although the
  paper also calls it mRNA); an import message the curator can see is the lesser harm. Cost: 2 of the 4 lookup resolutions on the 152 runs. Alternative: let
  the lookup decide (Option A of nucleicD Q1), at the risk above.
* **P-F3 (protein-coding -> DNA).** Mapping an NCBI protein-coding gene to type DNA is a proposal of this track, not a ruling (D-W21(d) only says
  "flag unknown; try an online lookup"). It is accepted only when no sentence of the paper naming the row or a synonym contains a protein word
  (protein, factor, complex, enzyme, receptor, binding, product, subunit, polymerase, kinase, ...); otherwise the row stays unknown with
  `protein_context`. The word list is the writer's `_PROTEIN_TOKENS`, so common words (product, binding, factor, complex) make it conservative. Cost: CHL27 on the 152
  runs. Alternatives: no protein-coding mapping at all (all such rows unknown), or the mapping without the sentence check (the earlier behaviour).
* **P-F13 (reviewed synonym `type iii taste cell` -> CL:0002285).** `type iii taste cell` is not a CL label nor an exact synonym (OLS exact 0), so it
  sits outside the exact-only rule. It is kept as a reviewed vocabulary synonym of `type III taste bud cell` (CL:0002285): the type I-III scheme
  applies only to taste bud cells and the sibling `type II taste cell` is CL:0002286. Provenance is recorded as `reviewed_synonym` on every write. Product owner
  to confirm; suggest proposing the synonym to the Cell Ontology. If rejected, drop the `SYNONYMS["cell-type"]` entry and the numbered-context fallback for it.

## Remaining risks / not established

* F1: collation and unique-index behaviour of the production PathWhiz DB; behaviour of the renamed row at import (not run). With an unpatched importer and a populated DB an omitted component can still bind to a row of another pathway (F10 scope gap, unchanged).
* F3: the whole-sentence protein-word check is crude (cautious); no negative example set beyond CHL27.
* F9: sharing the budget means a slow id fill starves the nucleic lookup and title; the default budget (`BUDGET_SECONDS`) was not re-tuned.
* Flag-off byte identity: not re-replayed here (reading only: every change is behind `final_payload_enabled()` or in flag-on-only modules; the F8 change is in `vocabulary_map`, used only under the flag).
* The `.env` copy in `C:/t/fp` is still there.
