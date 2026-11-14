# P5 final measurement (replay identity, flag-on replay, harness import)

Code under test: `pwml/final-payload` HEAD `28278264792e6e19f262feb14be66a70c3199789`, clean `git archive HEAD` export at
`C:/t/p5scratch/measure/code` (no `.env`, no dirty caches), `PYTHONPATH` = its `src`. Procedure = P4 d3 (`replay.py` of writerA, only the output dir changed;
152 stored runs from stored `final_mapped.json`; `T2PW_OFFLINE_CURATOR=1` set inside the script as in P4). Flag set explicitly by the script: off = `T2PW_FINAL_PAYLOAD=0`, on = `1`.
Heavy lock: `fp-p5m` (replay) and `pwi-m03/m04/c03/c04` (import), retry-the-lock loop, 0 lock failures beyond the retries. Docker project `t2pw-pwi` stopped afterwards.
Files here: `cmp_p5.out.txt` (replay), `import_table.md`, `import_per_file.md`, `cat_p5.out.txt`, `extra.out.txt`, `voc.out.txt`, `scripts/`. PWML output and raw import results are not committed (`C:/t/p5scratch/measure/`).

## (1) Flag-off identity: 97 of 97 identical
97 written, same set as P4 d3 off and P3 off; **sha256 identical to both, 97 of 97, 0 different**. 55 write no PWML (35 stage3_revalidation, 20 required_field_gate; same stage per run as d3 off). The clean export (no `.env`) did not change anything.

## (2) Flag-on replay (T2PW_FINAL_PAYLOAD=1)
* Written 97, same set as P4 d3 on. Byte-different from d3 on: **22 of 97** (idx 3 7 8 10 23 29 31 35 37 46 50 68 70 76 84 98 105 111 119 146 150 151); from P3 on: 25. Not traced file by file, see Not established.
* No non-stage3 error classes other than the 20 required-field-gate runs (same unwritten set as off). fill_error 0, paper text found 150 of 152, live requests 31, fetch errors 0.
* Counters, written runs (d3 in brackets): `locations_dropped_no_id` 4 (4, 3 files); `cell_types_dropped_no_id` 13 (0, 6 files); `tissues_dropped_no_id` 1 (0, 1 file); `components_kept_wildcard_guard` 21 (0, 13 files); `locations_kept_state_collision` 16 (18, 11 files); `membranes_kept_unresolved` 0 (0); `nucleic_acids_type_unset` 9 (10, 3 files).
* Ontology lookups: looked_up 226, resolved 19, no_exact_match 202, fetch_error 0; ontology_id entries location 9, tissue 7, cell_type 3.
* Nucleic types (all 152 runs): candidates 35, resolved 10, unknown 24 (conflicting 6, no_cue 10, not_mentioned 7, protein_like 1), lookup_resolved 1 (NCBI Gene), lookup_requests 14, budget_exhausted 0.
* Checker (writerA `cmp_wa.py` logic, `scripts/cmp_p5.py`): species-only states in written PWML **0** (d3 0, off 0). Wildcard-bindable states **63 in 20 files** (d3 on 65 in 20, flag off 59 in 19): not 0; 59 exist with the flag off (stored states that are subsets of siblings), same finding as writerA.

## (3) Harness import, 59 flag-on files (same idx list as P4 d3), importers (a) = branch 03 tip + owner change (`C:/t/pwpr/h03`), (b) = + exact-match fix 04 (`C:/t/pwpr/h04`)
Each with its own work dir (`C:/t/pwi/p5w/m03-work`, `m04-work`, `c03-work`, `c04-work`), out dir outside it, chunks of 12 with `--resume`, all 59 `success: true`.
Category definitions as P4 (`cat2_p4.py`, re-implemented in `scripts/cat_p5.py`; P4 d3 column recomputed from `d3/import/on59_results.jsonl` and equals the P4 table).

| | P4 d3 | P5 (a) 03 | P5 (b) 04 |
|---|---:|---:|---:|
| clean files of 59 | 45 | **48** | **48** |
| messages | 94 | **67** | **63** |
| vocabulary | 38 | 18 | 18 |
| transport family | 18 | 18 | **14** |
| of which left=right | 5 | 5 | 4 |
| nucleic acid | 24 | 20 | 20 |
| interactions | 9 | 6 | 6 |
| other | 5 | 5 | 5 |
| transports created / expected | 32/37 | 32/37 | **33/37** |
| interactions created / expected | 13/17 | 14/17 | 14/17 |

Controls (messages): PW741182 0, PW012926 1, PW176888 2, PW687775 3, PW1 stub 176, **identical for P4 d3, (a) and (b)** (all 5 run with both importers).

### Files worse than P4 d3: none
No file has more messages, a new message key type, or a new message text (checked at text level, `extra.out.txt`; no NEWTEXT line). Nothing is worse in either column.
Files with fewer messages (same in a and b): idx 7 (33 to 26: nucleotide types now resolved, three nucleic-acid/interaction messages gone), 10 (12 to 10), 23, 37, 98 (4 to 0 each), 151 (12 to 6 in a, to 2 in b). Cause of 10/23/37/98/151: id-less cell types now dropped from the PWML (the "cell_type: Ontology can't be blank / invalid" pairs are gone; cell_types_dropped_no_id 13). Only idx 151 differs between (a) and (b): transports 2/3 in (a), 3/3 in (b) (exact state matching), messages 6 vs 2.
Remaining vocabulary (18, both importers): subcellular_location 14, cell_type 2, tissue 2, in idx 10, 48, 68, 125, 139, 151; probably id-less locations kept by the wildcard guard (not verified per file), which fire only because the harness DB is empty.
Per-file table: `import_per_file.md`.

## Not established
* Production behaviour: the harness DB is empty (rows with find_by never match; "Ontology can't be blank" fires), so a pass or fail here is not a production result.
* Why 22 flag-on files differ from d3 on was not traced per file. Known contributors: the P5 writer guard (cell types, tissues, locations; counters above), nucleic type resolution; writerA earlier saw 19 files differ only in compound names with identical sources (suspected cache or DB state), not investigated here either.
* The fill cache `C:/t/ev-fp/fillcache` is shared and mutable (31 live requests were made this run); a repeat run could differ in the ontology lookups if the remote answers differ.
* Single replay and single import per column; no repeat run to measure variance (Stage 1 is not involved, replay is deterministic apart from the live lookups).
* The remaining nucleic-acid messages (20) and 5 "other" were not re-triaged.
