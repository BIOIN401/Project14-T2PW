# FINAL PAYLOAD, PHASE 3 (2026-11-12, branch `pwml/final-payload`, from `bf20c050`, code HEAD `d6914468`)

Evidence: `evidence/final_payload_p3/` (measure.md, b2_vocabulary_map.md, b4_left_equals_right.md, importer patch p3, tests_final/, import/, replay/). Decisions P-37 to P-40 in `DECISIONS-PROPOSED.md`.

## 0. NOT ESTABLISHED (read first)
1. **No live run was made. The default was NOT flipped.** The worktree `C:/t/fp` has no `.env`; copying the main checkout's `.env` was refused by a deny rule and I did not work around it. So step D (PathWhiz, research and name-only modes; `final_payload_fill` provenance in a live run; title and PMID reference; harness import of a live PWML) is untested. The live seam has never run with the flag on. Driver and commands are ready: `evidence/final_payload_p3/live/` (`LIVE.md`, `drive.py`, `run1.sh`; input PMC12782028 / PMID 41522513). Needs either a `.env` in `C:/t/fp` or variables exported.
2. Nothing was imported into production PathWhiz; harness = empty DB plus vocabulary seed. "Clean" means zero messages.
3. The 31 flag-on files that changed against P2 (B1/B2/B3) were not inspected for biological correctness; only the vocabulary table and typing rule were reviewed.
4. 3 `transport_transporter_visualization` messages remain (runs 151, 10, 68); cause not traced (probably the failing left=right transports).
5. B4: the replay shows 4 equal-state transports in 4 files, not the triage's 6 in 5; the other 2 were not located. The harness still counts 6 messages in 5 files.
6. The import step ran with trailing `&` in its script (rule slip); the wrapper and lock were used and survivors were 0 on the on-batches; the controls report was not grepped.
7. Importer patch 3 covers `transport-transporter-id` only. Reaction-coupled transports probably have the same gap (not checked). The `get_enzyme` lookup ignores the biological state.
8. Open minor from re-review: `pw_nucleic_acid_id` alone does not stop nucleotide typing (writer.py ~1221). Not fixed.
9. First SMOKE run at `3e782d76` had 5 failures, all subprocess exit 3221225794 (0xC0000142); not reproduced on rerun at `d6914468` (511 passed). Cause not established (machine load suspected).

## 1. Changes (all behind `T2PW_FINAL_PAYLOAD`)
- **A. Importer patch 3** (Ruby, `lib/pwml_parser.rb`, four addition-only hunks, +19 lines): `@transport_transporter_list`, `transport-transporter-id` branch in `add_content`, `store_transport_transporter_ids` (via `Transport#get_enzyme`). Applied to `C:/t/pwi/smpdb`, the harness work copy (gitignored) and the owner's copy (backup `C:/t/pwi/backup/pwml_parser.rb.owner-2026-11-12`, diff = exactly the hunks, nothing staged or committed). `ruby -c` Syntax OK in the `smpdb-app` image. Patch: `evidence/final_payload_p3/pathwhiz_importer_fixes_p3.patch`, notes `importer_bugs_p3.md`. A failed remap now yields "can't be blank" instead of silently attaching to another transporter.
- **B1** (`98e50b3c`, writer.py ~1478-1500): nameless complex name via `_wrapper_complex_name` with entity-key fallback; synthesised complex ids skip all explicit ids.
- **B3** (`98e50b3c`, writer.py `nucleotide_type_for` ~724, used ~1221): DNA/RNA only for unambiguous whole tokens; protein-like names untyped; unset counted in `nucleic_acids_type_unset`. Review minors fixed in `d6914468` (more protein tokens; skip when `pathwhiz_id` / `pathbank_nucleic_acid_id` present; label fallbacks).
- **B2** (`3e782d76`, new `src/t2pw/pwml/vocabulary_map.py`, writer.py `_vocabulary_fields` ~797): reviewed synonym table; 13 of the 40 unmatched names mapped (24 of 57 occurrences), 27 unchanged and counted, 0 ontology ids written (none in data).
- **B4**: no code change. Cause: autostate filler in `process_normalizer.py:3198-3207` (forbidden directory), the model naming one state twice (idx 102, which also has a correct twin transport), and `ir.resolve_state` default. See b4 md.

## 2. Tests (bounded wrapper, surviving 0)
SMOKE 511 passed (`d6914468`; first run at `3e782d76` 506 + 5 transient subprocess failures, see 0.9); golden seam 38 passed; focused (`test_final_payload_*`, `test_pwml_writer`) 96 passed. One existing flag-on test was edited (undo the "Extracellular Space" rename, now on the flag-on dump only; reviewer judged legitimate). No default flip, so no test needed `T2PW_FINAL_PAYLOAD=0`.

## 3. Flag-off identity
152-run replay at `3e782d76`: 97 PWML files, 97 sha256-identical to the stored flag-off outputs (`git diff bf20c050 e9c3e197 -- src` is empty); 55 write none in both. Not re-replayed at `d6914468`; its diff touches only flag-on branches (reviewed).

## 4. Harness (59 files, three-patch importer)
| | P2 on6 | P3 on |
|---|---:|---:|
| clean | 29 | 43 |
| messages | 211 | 147 |
| vocabulary | 110 (27 files) | 62 (12 files) |
| transport family | 37 | 21 |
| transporter visualization "blank" | 19 (12 files) | 3 (3 files) |
| nucleic acid | 33 | 33 |
| left=right | 6 | 6 |
| interactions | 28 | 28 |
Transports created 31 of 37; transporter visualizations 0 to 16. Controls (5): PW741182 clean, PW012926 1, PW176888 2, PW687775 3, PW1 stub 176. The nucleic-acid count is unchanged because the one affected file's names are not unambiguous or carry ids (not traced).

## 5. Live runs: not run (section 0.1)
## 6. Default: not flipped (P-40)

## 7. Reviews
| review | model / effort | scope | verdict |
|---|---|---|---|
| 4 | opus / high | `bf20c050..3e782d76` | APPROVE WITH FIXES, no majors; minors fixed in `d6914468` except the one in 0.8 |
| 5 | sonnet / high | `3e782d76..d6914468` | APPROVE (one nit, 0.8) |

## 8. Open questions for the product owner
1. **Live runs:** (A) put a `.env` in `C:/t/fp` yourself and the orchestrator runs three modes (about 1 to 2 h each at 3000 s cap), then flips the default; (B) flip now on measurement alone (risk: live seam unproven).
2. **27 unmapped location/cell/tissue names:** create rows with owner-supplied ontology ids, or drop the location.
3. **Left=right transports:** fix upstream (autostate in `process_normalizer.py`, forbidden for this track) or drop degenerate transports in the writer (changes content).
4. **Nucleic acids (33 messages):** default the type per kind, or withhold with their interactions.
5. Spontaneous (still nil), gap filling, membrane drawing: unchanged from P2.

## 9. Dispatches (9 of 12)
| # | task | model / effort |
|---|---|---|
| 1 | A importer patch | sonnet / medium |
| 2 | B1, B3, B4 | sonnet / medium |
| 3 | B2 vocabulary | sonnet / high |
| 4 | C replay and harness (about 2 h) | sonnet / medium |
| 5 | D live runs (blocked, prepared driver) | sonnet / medium |
| 6 | tests (plus 2 resumes) | haiku |
| 7 | review | opus / high |
| 8 | review fixes | sonnet / medium |
| 9 | re-review | sonnet / high |
Dead ends: `python -` heredocs run by mistake twice (no effect); `&` on one script; `.env` copy denied.
