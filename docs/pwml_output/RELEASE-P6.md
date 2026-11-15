# RELEASE P6: PWML output work into `webserver/refactor` and `main` (2026-11-14 to 2026-11-15)

Answers `docs/webserver/DECISIONS.md` D-W23. Branch `pwml/final-payload` (worktree `C:/t/fp`), release worktree `C:/t/rel` (branch `release/pwml`, from `origin/webserver/refactor`). Evidence: `evidence/release_p6/` (`gates_e228611f/` = independent gate run on the first merge result, `harness/`, `impl/` = implementer runs incl. the two proofs below, `scripts/`). No live LLM run was made; no `.env` was used.

## 0. NOT ESTABLISHED (read first)
1. Everything in `FINAL-PAYLOAD-P5.md` §0 still holds (thin live evidence for the final-payload default, empty harness database, single replay per column, `T2PW_TRANSPORT_SIDES` off, disclosure of drops only in the builder).
2. The offline equivalence tests (`test_equivalence_run_pathway.py`, `test_equivalence_app_vs_service.py`) now run with the final payload OFF and layout v1 (harness pins, D-W23 a/b). The default path (final payload on, layout v2) is covered by the new `tests/test_equivalence_default_path.py`, offline only, with a network that always fails: a SUCCESSFUL live fill is still not exercised end to end by any test.
3. Layout v2 is the default from this release. Its quality numbers come from the 97 replayed files and 59 harness imports into an empty PathWhiz database; no curator has judged v2 pictures beyond the orchestrator's screenshots (`PWML-LAYOUT.md`).
4. With v2 the Streamlit Width/Height inputs (`streamlit_app.py` ~2664) and `vis_width` no longer set the canvas: v2 recomputes it (`writer.py` ~3001). Not a correctness problem; the inputs are not relabelled (review 1 F6).
5. The PathWhiz harness and replay gates were run on the first merge result `e228611f`. The final merge adds only test-file changes and two docstrings (`24fdbe73`) plus docs; the pytest gates were re-run on the exact pushed SHA, replay and harness were not.

## 1. What was done
| item | commit | result |
|---|---|---|
| merge side fix + layout into the final payload | `4c6e8dfb` | clean merge of `pwml/side-fix` `2971851e` (contains `pwml/layout` `07d23e9c`); review 1 confirmed no extra hunks |
| (c) reaction-member `side` from role | in `2971851e` | applied to the default output (D-W23 c); side mismatches 0 of 2072 members on the 97 default files |
| per-host circuit breaker (P5 §7 Q6, R4-2) | `854872d9` | `_Budget` counts consecutive FetchErrors per host (`urlsplit(url).netloc`); N = 5; the time budget stays global; a success resets only its host |
| (b) layout default v2 | `7d973ce0` | `layout_version()`: only `v1` (any case, stripped) selects v1; unset, empty, `v2` and every other value give v2. 8 tests in `test_pwml_writer.py` and `test_final_payload_writer_fixes.py::test_no_processes_flag_on_equals_flag_off_minus_bound` pinned to v1 by env only (they assert v1 geometry / whole-file bytes); no central pin, the suite runs v2 by default |
| (a) offline equivalence | `7e258805` | `REPLAY_ONLY_ENV` pins `T2PW_FINAL_PAYLOAD=0` and `T2PW_PWML_LAYOUT=v1`; new `tests/test_equivalence_default_path.py` (own process) |
| review fixes | `24fdbe73` | v2 assertion parses the XML; payload compared with the recorded live payload; per-host cost cap asserted; docstrings |

**Why the harness also pins layout v1.** The equivalence fixture was recorded on 2026-11-05/08, before the layout code, so its `pwml_sha256` is v1 geometry. Without the v1 pin `test_equivalence_app_vs_service.py` fails 3 of 14 on that hash (`impl/eq_app_nov1.log`; `test_equivalence_run_pathway.py` passes 8 of 8 either way, `impl/eq_run_nov1.log`). This is D-W23 (b) "pinned to v1"; the orchestrator accepted it.

**The default-path test** (6 tests; final payload and layout left unset, read inside the module fixture before the conftest pin): both arms (app and `run_pathway`) run through export with status pass; no request leaves the machine (`refused == []`, `misses == []`); the final payload ran and failed open on both arms; app and `run_pathway` give identical PWML bytes and payload; the PWML has membrane visualizations and label locations (v2 only); the app payload equals the recorded live payload except the `retrieved_at` stamps and the `final_payload_pathway` record. The v2 assertion was shown to bite: with the v2 hook disabled the test fails (`impl/eq_bite.log`, 1 failed 5 passed, `test_the_pwml_is_layout_v2`).

## 2. Open phase-5 questions decided (D-W23 e; `FINAL-PAYLOAD-P5.md` §7)
| # | question | decision | why |
|---|---|---|---|
| 1 | enable `T2PW_TRANSPORT_SIDES` | **stays OFF** | D-W23 exception: no-loss is unproven (two replay repairs wrote states the quote does not support) and the standing rule is never to worsen text-to-PWML output |
| 2 | nucleic acids: cautious rules or option A | **keep the cautious rules** (conflicting rows not looked up; protein-coding to DNA only with positive paper evidence) | no change to output; option A writes LHCB1.1 as DNA against its own paper and CHL27 as DNA in a protein context. Cost kept: RyhB (a correct RNA) stays unknown |
| 3 | MG-63 / Saos-2 | **keep them in the payload only** (no PWML row) | the alternatives make a biological claim (a cell line written as a tissue) or need a PathWhiz change (branch 05, PATHWHIZ-FIXES track). **Returned to the PO as an open question** if either alternative is wanted |
| 4 | reviewed synonym `type iii taste cell` to CL:0002285 | **not decided: open question for the PO** | confirming or rejecting a term mapping is a biology call; the synonym stays as built in P5 until the PO rules |
| 5 | where to disclose dropped components, renamed membranes, unknown nucleic types (R3-4) | **the export report** (not the PWML `description`, which would put process notes into PathWhiz) | implementation deferred to a follow-up task (carry `BuildResult` counters and `vocabulary_outcomes` into the export report); not in this release |
| 6 | per-host circuit breaker (R4-2) | **done** (`854872d9`) | a few lines; one dead host no longer skips the other hosts' lookups |
| 7 | push the PathWhiz branches | **push `t2pw/01..04` as branches, no PRs, no merge** | D-W23 f; result in §5 |
| 8 | internal rendering of the paper pathways | **handed to the correctness programme** | D-W23 g |

## 3. Gates
Independent `ws-test-runner` on the first merge result `e228611f` (`origin/webserver/refactor` `c757814b` + `pwml/final-payload` `7e258805`), bounded wrapper, 0 surviving processes (`gates_e228611f/`):
| gate | result |
|---|---|
| SMOKE | 511 passed (52.9 s) |
| golden seam (4 files, unedited vs `c757814b`) | 38 passed |
| `test_equivalence_run_pathway.py` (alone) | 8 passed |
| `test_equivalence_app_vs_service.py` (alone) | 14 passed |
| `test_equivalence_default_path.py` (alone) | 5 passed (6 after `24fdbe73`) |
| focused, default env (26 files, `focused_files.txt`) | 532 passed |
| focused, `T2PW_PWML_LAYOUT=v1` | 532 passed |
| replay, 152 stored runs, default output (final payload on, v2) | 97 PWML written; every v2 quality count 0 (node overlaps, edges through nodes, out of canvas, nodes not in box, box overlaps, empty boxes, internal labels, label overlaps, edges under labels); side mismatches 0 of 2072 members; identical to the integration check `bcc67409` |
| PathWhiz harness, 59 files, importer (b) `C:/t/pwpr/h04` | 48/59 clean, 63 messages, 33/37 transports, 14/17 interactions: identical to P5 (b) per file (no file better or worse); controls PW741182 0, PW012926 1, PW176888 2, PW687775 3, PW1 stub 176 (unchanged). Stack `t2pw-pwi` stopped |

Implementer runs (not independent, `impl/`): SMOKE 511, golden 38, focused 532 (v2) and 532 (v1), equivalence 14 / 8 / second fixture 8 / default path 5, then after the review fixes default path 6, focused 532, SMOKE 511.

Final gates on the exact SHA pushed to `main`: see §6.

## 4. Reviews
| review | model / effort | scope | verdict |
|---|---|---|---|
| 1 | `ws-reviewer` opus / high | `49551d19..7e258805` (merge checked clean; delta `4c6e8dfb..7e258805`) | CHANGES REQUESTED: F1 blocking (v2 check passed on v1 output: the empty `<membrane-visualizations/>` tag); notes F2 tautological assertion, F3 no check against the recorded payload, F4 and F5 docstrings, F6 Streamlit width inputs (§0.4), F7 no release doc or committed evidence (this doc, `evidence/release_p6/`) |
| 2 | `ws-reviewer` opus / high | `7e258805..24fdbe73` | APPROVE WITH NOTES: F1 to F5 fixed, bite proof confirmed; notes N1 (stamps not checked against the frozen clock), N2 (host cap 3, could be 2), N3 (how the hook was disabled: `impl/bite_toggle.py`, `impl/bite_run.sh`). N1 and N2 not fixed in this release. Both reviews: `evidence/release_p6/review/REVIEWS.md` |

## 5. PathWhiz branches (D-W23 f)
Pushed 2026-11-15 from the worktree `C:/t/pwpr/04` to the PathWhiz remote `origin` as NEW branches only; no pull request opened, nothing merged, no existing branch touched. Before: `git ls-remote --heads origin` listed 48 heads, none under `t2pw/`. After: `t2pw/01-vcv-node-typo` `17af3272`, `t2pw/02-transport-state-remap` `48ab7499`, `t2pw/03-transport-transporter-remap` `c5c12d69`, `t2pw/04-biological-state-exact-match` `80bdaca7` (stacked: 01 < 02 < 03 < 04). The owner's checkout was not touched. Branches `t2pw/05+` belong to the PATHWHIZ-FIXES track. Note: the remote warned that the installed Git Credential Manager will be blocked by Bitbucket Cloud on 2026-11-30 (upgrade to a version newer than 3.0.0).

## 6. Merge and final gates
Procedure: this docs commit on `pwml/final-payload`, merged into `release/pwml` (from `origin/webserver/refactor`, fetched again first); SMOKE, golden, the three equivalence files (each alone) and the focused set (default and v1) re-run by an independent `ws-test-runner` on that exact merge SHA; then fast-forward pushes of `webserver/refactor` and `main` (`origin/main` `06a0b8c0` is an ancestor). The SHAs and the counts of that run are recorded in a docs-only follow-up commit on `webserver/refactor` (evidence `evidence/release_p6/gates_final/`), because a doc cannot hold the counts of a run on its own commit.

**Result (2026-11-15).** Merge `bc63f504` = `origin/webserver/refactor` `c757814b` + `pwml/final-payload` `2d99a683` (code identical to `24fdbe73`). Independent `ws-test-runner` (sonnet / medium) on exactly `bc63f504`, bounded wrapper, 0 surviving processes, exit 0 each (`evidence/release_p6/gates_final/`): SMOKE 511 passed; golden seam 38 passed (the four files and both equivalence files unedited vs `c757814b`); `test_equivalence_run_pathway.py` 8, `test_equivalence_app_vs_service.py` 14, `test_equivalence_default_path.py` 6 (each alone); focused 532 (default) and 532 (`T2PW_PWML_LAYOUT=v1`).
`webserver/refactor` pushed `c757814b..bc63f504` (fast-forward). **`main` NOT pushed: the push command was denied by the permission system**; `origin/main` stays at `06a0b8c0`, which is an ancestor of `bc63f504`, so `git push origin bc63f504:main` is a fast-forward of a fully tested SHA whenever it is allowed. This record lives on `pwml/final-payload` only, so that `bc63f504` stays the tested tip for that push.

## 7. Open questions for the product owner
1. Synonym `type iii taste cell` to CL:0002285: confirm or reject (§2 #4).
2. MG-63 / Saos-2: payload only (now), BTO tissue rows, or a PathWhiz `cell_type.rb` change accepting `CVCL_` (§2 #3).
3. Streamlit Width/Height inputs no longer set the canvas under v2: relabel, hide, or keep (§0.4).
4. Push `bc63f504` to `main` (D-W23 d): the push was denied by the permission system and needs to be allowed or done by the PO.

## 8. Dispatches
| # | task | agent | model / effort |
|---|---|---|---|
| 1 | implement (a), (b), per-host breaker; resumed once after an early "waiting on the lock" return | general-purpose | sonnet / high |
| 2 | gates on `e228611f` (SMOKE, golden, 3 equivalence files, focused x2, replay metrics, harness 59 + controls) | ws-test-runner | sonnet / medium |
| 3 | review 1 | ws-reviewer | opus / high |
| 4 | fix round 1 (resume of 1) | general-purpose | sonnet / high |
| 5 | review 2 (resume of 3) | ws-reviewer | opus / high |
| 6 | final gates on the pushed SHA | ws-test-runner | sonnet / medium |
