# Measurement D (final payload P4), code HEAD 3c7d02f8

Scripts: `scripts/` (copies of the P3 scripts, paths p3 -> p4; `replay.py` additionally passes `01_source_text.txt` as
`state["final_payload_paper_text"]` as `run_pathway` does, and records `layout_debug_counts`, the ontology and nucleic reports;
bounded timeout raised 540 -> 2400 s for the one 152-run chunk). Raw: `replay/{off,on}`, `import/{controls,on}`,
`cmp_p4.out.txt`, `onto_p4.out.txt`, `on_vs_p3_diff_summary.txt`, `import_*.txt`. Heavy lock held by `fp`/`pwi`, 0 survivors on all 7 import batches.

## 1. Flag-off identity (152 runs, HEAD vs P3 off at 3e782d76)
97 write PWML in both (same set), **97 of 97 sha256-identical, 0 different**; 55 write none. Replay starts from stored
`final_mapped.json`, so task B's normalizer change cannot show up here (B is proven separately in `b_before_after.md`).

## 2. Flag-on replay (152 runs)
| | P3 on | P4 on |
|---|---:|---:|
| files written | 97 | 97 (same set) |
| byte-identical to P3 on | | 80 of 97; **17 changed** (idx 3 4 7 10 17 27 68 70 76 84 105 116 119 125 126 150 151) |
| diff size (P3 -> P4, lines) | | +42 / -147 (`on_vs_p3_diff_summary.txt`) |
| paper text found (01_source_text.txt) | | 150 of 152 runs |
| live network requests / errors | | 48 / 0 (fill cache otherwise) |
| fill errors | | 0 |

Diff kinds (line level, `on_vs_p3_diff_summary.txt`): subcellular-location blocks removed in 12 files (3 4 10 27 68 70 76 116 119 125 150 151);
`+ontology-id` lines in 3 7 17 119 126 150 151; `+nucleotide-type` lines in 7 (6), 84 (1), 105 (1).

Counters (the 97 written runs unless noted):
| counter | value |
|---|---:|
| ontology_id entries written (all 152 runs) | 17 (14 in written runs), 14 distinct (kind, name) |
| by kind | location 9, tissue 7, cell_type 1 |
| of the 27 unmatched names from A | **11 filled** (= all 11 that A's lookup resolved: etioplast, synaptic cleft, synaptic vesicle, thylakoid membrane, bone marrow, central nervous system, cotyledon, flower, hypocotyl, petal, root); 16 stay without id (lookup no_exact_match) |
| other ids filled outside the 27 | 3 (epidermal cell CL:0000362, inner mitochondrial membrane GO:0005743, nerve fiber GO:0043005) |
| ontology lookups (all runs, repeats counted) | looked_up 227, resolved 17, no_exact_match 210, fetch_error 0 |
| `locations_dropped_no_id` | **21** |
| nucleic types from paper text | 35 candidate rows (unsettled by name) in 5 runs; **16 resolved** (8 in written runs); not_mentioned 6, no_cue 12, conflicting 0, protein_like 1 |
| `nucleic_acids_type_unset` (written runs) | **6** |

Per-name list in `onto_p4.out.txt` (its "A resolved" tally line is mis-parsed, ignore it; A resolved 11).

## 3. Harness import (Docker t2pw-pwi; 3-patch importer, vocabulary seed; 59 files = same index list as P3; 5 controls)
Definitions recomputed identically for both columns by `scripts/cat2_p4.py` from the stored `results.jsonl` (P3 column re-derived from
`final_payload_p3/import/on/results.jsonl`; its vocabulary, transport, left=right, tv-blank and total match the P3 doc, its
nucleic/interactions/other split there used different grouping, so those three rows differ from FINAL-PAYLOAD-P3.md s4:
nucleic 33 / interactions 28 / other 33 there vs 38 / 21 / 5 here, same 147 total).

| | P3 on | **P4 on** |
|---|---:|---:|
| files success true | 59 | 59 |
| clean (0 messages) | 43 | **46** |
| messages | 147 | **81** |
| vocabulary | 62 (12 files) | **24 (7 files)** |
| transport family (all `transport*`) | 21 (5 files) | 28 (5 files) |
|   of which left=right "unique biological states" | 6 (5 files) | 8 (5 files) |
|   of which transporter visualization blank | 3 (3 files) | 4 (3 files) |
| nucleic acid | 38 | **18** (1 file) |
| interactions | 21 | **6** |
| other | 5 | 5 |
| transports created / in file | 31 / 37 | **29** / 37 |
| interactions created / in file | 9 / 17 | **14** / 17 |

Controls (5 files, same for P3 and P4, identical): PW741182 0 messages (clean), PW012926 1, PW176888 2, PW687775 3, PW1 stub 176;
all `success: true`; clean 1 of 5; 182 messages. Control reports (`import/controls/b_00/summary.*`, `results.jsonl`) contain no
quarantine report files; none exist under this evidence dir.

### Vocabulary messages left (24)
| kind | messages | files |
|---|---:|---:|
| cell types | 22 | 6 |
| tissues | 2 | 1 (idx 139) |
| subcellular locations | **0** (was 26) | 0 |
So none of the remaining vocabulary messages are locations; locations without an id are now dropped (21 times), not imported blank. Cell types (22, same as P3) are untouched by A (only 1 cell-type name got an id); tissues 14 -> 2.

### Remaining messages by category (81)
- vocabulary 24: cell_type 22 (6 files: idx 10, 23, 37, 48, 98, 151), tissue 2 (idx 139).
- transport family 28: transport_element left=right 8, transport "elements too short" 8, transport_visualization blank 8, transporter_visualization blank 4.
- nucleic acid 18 (one file, idx 7): nucleotide type blank 6, type not in list 6, nucleic_acid_location 4, interaction_nucleic_acid_visualization 2.
- interactions 6 (idx 7): interaction_visualization 2, left/right element blank 2, wrong-length 2.
- other 5: protein_complex too short 1, protein_complex_visualization 1, compound "Pwc already taken" 1, compound_location 1, reaction_compound_visualization 1.

### Regression to flag: run 10 transports
Run 10: transports created **3 of 7 (P3: 6 of 7)**, left=right 4 (P3: 1). Its P4 PWML dropped 3 locations that have no ontology id
(`-subcellular-location` x3), so some transports now have the same biological state on both sides (identical left and right) and fail. Net across the 59 files: transports created 29 vs 31 (run 151 improved 1 -> 2 of 3, run 10 worsened 6 -> 3 of 7, other runs unchanged; per file in `import_per_file.txt`). Not traced further; whether dropping the location is the right choice for a transport between two compartments that both lack ids is a product question (alternative: keep the location row blank, which costs the vocabulary messages again).

## Not established
- Production behaviour (empty DB plus seed; messages are candidates, not proofs).
- Biological correctness of the 17 changed flag-on files was not inspected, only line-diffed by element type.
- Why exactly run 10 loses transports (inferred from the diff and the l=r count, not traced in the importer).
- 55 non-written runs and the flag-on 152 -> 97 set are unchanged from P3; the live (LLM) path was not run.
- Flag-on replay reads the paper text from `01_source_text.txt` beside the run; 2 runs have none (no effect on written files unless they carry nucleic acids).
