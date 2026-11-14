# Final payload P5 (D): unknown flag and NCBI Gene / RNAcentral lookup for the DNA/RNA type of a nucleic acid

Ruling: D-W21(d), "DNA/RNA type the paper does not settle is flagged unknown; also try an online lookup to settle it."
Everything is behind `T2PW_FINAL_PAYLOAD`. Flag off, nothing here runs and the PWML writer is untouched.

## Order per nucleic-acid row (no database id)

1. Name rule (`writer.nucleotide_type_for`): unchanged. Rows it settles are never touched.
2. Paper rule (P-42, unchanged: 2-token window AND whole-sentence intersection). Sets `nucleotide_type_paper`.
3. Online lookup for the rows still unsettled (not for protein-like names): NCBI Gene, then RNAcentral, both always asked, in the
   species of the pathway.
4. Still unsettled: flagged unknown. A type is never guessed.

Existing paper or frozen values are never overwritten by a lookup value (a row that already holds a non-empty `nucleotide_type`
is left and flagged `slot_filled`).

## Keys (on the row, kept in the final payload)

| key | values |
|---|---|
| `nucleotide_type_status` | `resolved` (paper or lookup) or `unknown` |
| `nucleotide_type_unknown_reason` | why the PAPER did not settle it: `conflicting`, `no_cue`, `not_mentioned`, `protein_like`, `no_paper_text` |
| `nucleotide_type_unknown_lookup` | why the LOOKUP did not: `not_attempted` (protein-like), `no_fetcher`, `species_unknown`, `species_ambiguous`, `no_symbol_name`, `not_found`, `ambiguous`, `gene_type_not_decisive`, `sources_disagree`, `too_many_records`, `lookup_error`, `lookup_skipped_after_error`, `slot_filled`, `writer_ignores` |
| `nucleotide_type` | `DNA` or `RNA`, lookup-resolved rows only. This is the key the writer already reads (PathBank column name), so no writer change; it needed one line in the IR key whitelist (`ir.py`) |
| `nucleotide_type_source` | lookup-resolved rows: `{source, accession, match_type, retrieved, quote, species_taxid, candidates, confirmed_by}` |

`final_payload_fill` gets one entry per row: `field: "nucleotide_type"` (lookup-resolved: value, accession, source, match_type, quote,
species_taxid, candidates, retrieved, paper_reason) or `field: "nucleotide_type_status"` with `value: "unknown"`, `match_type` =
the paper reason, `lookup` = the lookup code. `by_field` counts `nucleic_acid.nucleotide_type` and `nucleic_acid.nucleotide_type_status`.
Paper-resolved rows keep their P4 entry (`nucleotide_type_paper`) and get only `nucleotide_type_status: "resolved"`.

The writer does what it did for an unsettled row today: no `nucleotide-type`, an import message. The unknown flag keys are not in
the IR whitelist, so they never reach the writer (test: `test_unknown_flag_does_not_change_what_the_writer_does`).

## Lookup rules

* Species: the one `taxonomy_id` of the payload's `entities.species` rows. None or not numeric: `species_unknown`, no request. More
  than one distinct id: `species_ambiguous`, no request (the nucleic-acid rows carry no species of their own).
* Names tried: the row name and its synonyms that are one gene-symbol-shaped token (3-40 characters of letters, digits, `_ . -`, first
  a letter). `PORA promoter`, `ALAD1/HEMB1`, `-35 box`, `operator at -280` are not symbols: `no_symbol_name`, no request.
* **NCBI Gene** (E-utilities): `esearch` `"NAME"[Gene Name] AND txid<T>[Organism:exp]` (the taxon and its descendants; K-12 records
  sit under 511145, below 562), `esummary` to verify the record's own symbol (`exact_symbol`) or an entry of `otheraliases`
  (`exact_alias`) equals the name case-insensitively (the search itself is looser), `efetch` XML for the gene type
  (`Entrezgene_type`) and the status. Records whose `Gene-track_status` is not live (discontinued, replaced) are dropped. This is
  needed: for RyhB in E. coli, 14 records match, 13 are discontinued strain-level duplicates. Exactly one live record must remain:
  none is `not_found`, more than one is `ambiguous` (even if they agree), more than 20 is `too_many_records`.
  The `esummary` JSON has no gene-type field, hence the `efetch`.
* Gene type to nucleic-acid type, decided from the record:

  | NCBI gene type | type | why |
  |---|---|---|
  | protein-coding | DNA | PO ruling: a protein-coding gene is DNA as a genetic element. The record says nothing about whether the paper means the gene, its mRNA or its protein, so this is the weakest mapping (see CHL27 below) |
  | ncRNA, tRNA, rRNA, snRNA, snoRNA, scRNA, miscRNA | RNA | the gene's only product is an RNA, and a symbol such as RyhB, MIR171c, tRNA-Leu conventionally names that RNA, not its locus |
  | pseudo, other, unknown, biological-region, anything else | unresolved (`gene_type_not_decisive`) | the record does not decide DNA vs RNA |

* **RNAcentral**: the search at rnacentral.org is served by the EBI Search REST API (`ebi.ac.uk/ebisearch/ws/rest/rnacentral`), which
  is what is used. The rnacentral.org `/api/v1/rna` endpoint was tried first and has no usable name filter (`gene=`, `external_id=`,
  `description=` returned unrelated or empty pages, 2026-11-13). Query `gene:"NAME" AND TAXONOMY:<T>`; `TAXONOMY` matches the entry's
  own taxid EXACTLY (no descendants: 511145 entries do not answer for 562), so the entry id must end in `_<T>`, its `gene` list must
  contain the name (case-insensitive, whole entry, not a substring) and it must be active. Any such entry means an RNA molecule.
  Several entries (sequences) of one gene name are not an ambiguity: they are all RNA.
* If both sources answer and the types differ: `sources_disagree`, unresolved. If both agree, NCBI Gene is the recorded source and
  the RNAcentral accession is in `confirmed_by`.
* First fetch error ends the lookups of the run (`lookup_error`, then `lookup_skipped_after_error`); the flag is still written.
* Fetcher, throttle, cache: `clean_model.fill.http.UrllibFetcher` (User-Agent `t2pw-clean-model/0.1 (research tool)`, 1 request/s/host
  in production through `final_payload._live_fetcher()` with its 300 s budget) and `clean_model.fill.cache` (the gitignored
  `clean_model_cache/`). Negative answers are cached. The measurement used `C:/t/p5scratch/nucD/cache`, not committed.

## Code touched

* `src/t2pw/service/final_payload_nucleic.py`: `apply_nucleic_types` (new entry), `_ncbi_gene`, `_rnacentral`, `_Lookup`; `apply_paper_nucleic_types`
  only gained the optional `reasons` out-dict (its behaviour and every existing assertion are unchanged).
* `src/t2pw/service/final_payload.py`: the `maybe_apply_paper_nucleic_types` hook now calls `apply_nucleic_types` with the live budgeted
  fetcher (or the `fetcher`/`cache`/`today` the caller passes); one line at the call site forwards those.
* `src/t2pw/pwml/ir.py`: `"nucleotide_type"` added to the common entity key whitelist.
* Not touched: `writer.py`.

## Tests

* New `tests/test_final_payload_nucleic_lookup.py` (fakes only, no network): RNA gene, protein-coding alias, exact (not prefix) name,
  two live records ambiguous, every gene type mapped, undecisive types, RNAcentral exact taxid and gene field, sources disagreeing and
  agreeing, species unknown / ambiguous / two species, non-symbol names not queried, fetch error stops the lookups, every unknown
  reason, no paper text, ids and name-settled rows untouched, slot filled, writer-ignores guard, cache reuse, no fetcher, rollback of
  the lookup step only, the hook (flag off / on), the writer writes a looked-up type only with the flag and an unknown row stays
  untyped exactly as an unflagged row. `tests/test_final_payload_nucleic_type.py` is unchanged and passes.
* `tests/test_final_payload_*.py` + `tests/test_pwml_ir*.py` with the new file, through the bounded wrapper (`tests/nucD3.*`):
  232 selected, 231 passed, 1 failed, surviving processes 0. The failure is
  `test_final_payload_ontology_p5.py::test_writer_keeps_inner_and_outer_separate_for_bacteria_and_drops_them_for_a_human`
  (`KeyError: 'organisms'`): another agent's uncommitted work in `vocabulary_map.py` / `writer.py` / the ontology fill, not touched by
  this task. `tests/nucD2.*` is the earlier run, 2 failures of mine (they still expected the "agreeing records collapse" rule that was
  then removed in favour of the brief's one-candidate rule); fixed in the code commit.

## Measurement on the 152 stored runs (live, 2026-11-13)

Script `d_nucleic_measure.py`, data `d_nucleic_measure.json`, same run list and text files as P4 `c_nucleic_measure.py`.

| | rows |
|---|---|
| nucleic-acid rows | 39 |
| settled by name (writer) | 4 |
| unsettled | 35 |
| resolved from the paper (all DNA, as P4) | 10 |
| resolved by NCBI Gene | 4 (miR171c RNA, RyhB RNA, CHL27 DNA, LHCB1.1 DNA) |
| resolved by RNAcentral alone | 0 (it confirmed miR171c and RyhB) |
| ambiguous | 0 |
| not found | 5 |
| not attempted (no symbol-shaped name 14, protein-like 1, species ambiguous 1) | 16 |
| **flagged unknown** | **21** |

Unknown by paper reason (and lookup outcome): conflicting 4 (not_found 2, no_symbol_name 1, species_ambiguous 1), no_cue 9
(no_symbol_name 6, not_found 3), not_mentioned 7 (no_symbol_name 7), protein_like 1 (not_attempted). 4 + 9 + 7 + 1 = 21; 10 + 4 + 21 = 35.

### The four online resolutions, checked against the record text

| row | species | record | type | paper sentence | verdict |
|---|---|---|---|---|---|
| RyhB (run 105) | E. coli 562 | GeneID 2847761 `ryhB`, K-12 MG1655, gene type ncRNA, matched as exact symbol. RNAcentral URS000012CA7F also (gene RyhB/ryhB) | RNA | "Post-transcriptional repression of enterobactin synthesis by the RyhB sRNA" | right |
| miR171c | A. thaliana 3702 | GeneID 5007828 `MIR171c`, ncRNA, aliases MIR171C / microRNA171C. RNAcentral URS000021475C also | RNA | "Overexpression of miR171c enhanced degradation of the target SCL transcripts" | right (a miRNA) |
| LHCB1.1 | A. thaliana | GeneID 839870 `CAB2`, protein-coding, matched as exact alias LHCB1.1 | DNA | "the CAB locus consisting of LHCB1.1, LHCB1.2, and LHCB1.3" | right for that sentence; the paper step called it `conflicting` (it also mentions the mRNA), so the type follows the PO rule for a protein-coding gene |
| CHL27 | A. thaliana | GeneID 824861 `CRD1`, protein-coding, matched as exact alias CHL27 | DNA | "the membrane-bound subunits CHL27 (Rzeznicka et al., 2005) and LCAA" | **doubtful**: the paper uses CHL27 for the protein subunit. DNA is the gene-as-genetic-element reading of a row the pipeline put in `nucleic_acids`; if that row is really the protein, it should not be a nucleic acid at all |

RyhB in run 84 stays unknown (`species_ambiguous`): that payload has two species rows (E. coli and Arabidopsis thaliana). The same
paper's run 105 resolves it. Nothing came out wrong in the sense of a contradicted type; CHL27 is the one to look at.

## Not established

* Only 6 of the 152 runs have nucleic acids, and only 4 rows resolve online, so the lookup precision is shown by 4 hand-checked
  cases, not a rate. Gene-type to DNA/RNA for `protein-coding` is a product rule, not something the record can confirm.
* RNAcentral alone never resolved a row here, so that path is tested with fakes only on its decisions, and live on 2 confirmations.
* Species-level taxids over strain-level NCBI records: the `exp` search and the drop of discontinued records worked for E. coli,
  but a gene with several LIVE strain records in the species would be `ambiguous` and stay unknown.
* The `final_payload_fill` block is written without a nucleic lookup budget of its own beyond the shared 300 s `_Budget`; the
  worst case adds one more 300 s to the post-pipeline (the id fill has its own).
* Flag-on full pipeline run with a real unsettled nucleic acid (the live path through `run_pathway`) was not run; tests use fakes.

## Open questions (for the product owner)

1. Conflicting paper rows go to the lookup (LHCB1.1 became DNA although the paper also says mRNA). Option A (as built): the lookup
   may settle them. Option B: only `no_cue` and `not_mentioned` rows go to the lookup; conflicting stay unknown. Cost of B: LHCB1.1
   goes back to unknown (1 row of 4).
2. `protein-coding` -> DNA (CHL27). Option A (as built, the PO ruling). Option B: a protein-coding record settles DNA only when the
   paper has no protein-like mention, or never (protein-coding stays unknown): 2 of 4 online resolutions are lost, the PWML import
   message returns for them. 
3. Two or more species rows (run 84): option A (as built) skip. Option B: look the name up in every species and accept only if
   exactly one species has a hit. Cost: a name found in the wrong species of a multi-species payload would be typed on a guess.
