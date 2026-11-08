TASK: R3 series audit (R3a-R3d)  RANGE: 480b2384..2e3d6a28  READ AT: C:/t/ws-R3d-post @ 2e3d6a28 (clean afterwards); docs that postdate the merge via `git show 71cfe7d3:`
VERDICT: CLEAN WITH NOTES

No blocking findings. The composed block preserves behaviour and the merge gates held. There are two should-fix-before-R5 items, and both are tracking gaps rather than code defects. I ran no pytest and took no heavy lock.

## Check 1: composed behaviour (no loss found)

- **Normalised diff.** I diffed base block 2084-2527 against the concatenated app call sites and `extraction.py` bodies (`compose.py`). The only differences are:
  - `st.stop()` replaced by a returned failure;
  - error text bound to `message` first;
  - function-local imports;
  - `paths.PROJECT_ROOT`;
  - the `preprocess_fn` pass-through;
  - the `user_task_context` strip, which stays in the app and is outside the compared service bodies.

  All writes, early exits, messages and their order match.
- **Moved helpers.** An AST compare of the 10 moved functions (`fnc.py`) shows only the same mechanical changes.
- **Dead globals.** Of the names the base block bound, every one read later is re-bound from session state first (`names.py`). The folded `user_task_context` is never read after the block.
- **Cross-step defaults.** Every widget now passed unconditionally is bound on all paths: PDF options at app:1636-1639, `rag_incomplete_flag` at 1721/1737, the `infer_*` widgets in the form.

## Findings

**Should-fix before R5**

- **F1. `run_extraction` never clears quarantine artifacts, and the LEDGER does not name it.**
  - Where: `extraction.py:1033-1034` (docstring only), app:1924, `strict_quarantine.py:4306`.
  - That docstring warns that a stale report lets a careless caller "export under" the previous run's decision about a different pathway.
  - R5's worker runs in a shared checkout ("one pipeline run per checkout"), so `outputs/quarantine_report.json` from one job survives into the next.
  - The equivalence gate cannot see this, because the service arm starts in a fresh root (`harness.py:305`).
  - Fix: add it to the R5 carry-forward, with a per-run clear in `run_pathway` or `run_extraction`.
- **F2. `ExtractionRequest` has no defaults and no source for its settings, and this is untracked.**
  - Where: `extraction.py:996-1019`.
  - The harness reads the real form through AppTest (`harness.py:315`), which a worker cannot do.
  - The driver sets only export mode, input mode, text and focus (`driver.py:843, 2791-2803`), so benchmark behaviour depends on the form defaults.
  - A hand-copied set in R5 can drift silently.
  - Fix: name it for R5, for example constants shared by the form and the worker.

**Notes**

- **F3. Gaps in the LEDGER current-state block** (checked at `71cfe7d3`).
  - Missing:
    - R3c Q3, the `InputFailure` rename (carried in R3d.md section 8.5);
    - R3b section 7.2, the headless `reset_refinement_state` decision;
    - Q-R3d-4 / task R26 (Addendum 6), which has no LEDGER mention.
  - Stale:
    - "drop the `render_attempts` shim": R3d section 1 shows it cannot go, because two display-panel callers remain;
    - R3a N3: R3d section 8.5 says it is already covered;
    - N4 lists 5 Reporter methods, but the R5 worker needs 7 (`markdown` and `code` were added in R3c).
  - The docstring at `extraction.py:1036` still says "before any LLM call", though Stage 0 has already called the LLM (R3d F4). It is not in the F5 wording list.
- **F4. `test_c` now compares two arms that run the same `extraction.py`.** A regression in the moved code changes both arms alike. Only `test_b` anchors to the recording, on a fixture with no additions, no out-of-scope hits, RAG off and the default mode. This is tracked (R3d F1 / R26). Per Addendum 6, RAG coverage may still not arrive.
- **F5. App-to-service keyword seam has no in-suite check for the attempts settings.** No test asserts that `extract_attempts` or `infer_attempts` reach the pipeline from the real app (`streamlit_app.py:1888, 1909`). The suite tests only call the service directly, and the app mutations cover stop, reset, token stats and `state`. The `ivm` proofs that did compare callee arguments are one-time scripts. Base had the same gap inline.
- **F6. Test edits that go past the letter of a ruling.** All were reviewer-confirmed and none weakens a test.
  - `harness.py`: the service-arm rewrite (about 85 lines) has no PO ruling. It rests on `TODO(R3)` and the R3d dispatch. Q-R3d-3 covers only the `unused` lines.
  - Q-R3c-1 said about 15 lines; actual +22/-7.
  - Q-R3d-1 said about 20 lines; actual +24/-11.
  - c060a also moved `OWNED_FLOOR` and added a constant, under an orchestrator note, with R1/R2 precedent and identical selection proven.
- **F7. Evidence limits.** `R3a/postmerge` holds JSON only (exit 0, survivors 0), so the LEDGER's post-merge "511" cannot be recovered there. `R3a/independent/smoke.log` shows 511 on the identical code tree. No post-merge JSON records a commit SHA.

## Check 2: `run_extraction` versus the app sequence

| difference | matters headless? | named? |
|---|---|---|
| No ambiguous gate. Stage 1 refuses as `stage_one_failed`, after Stage 0's LLM spend. `pathway_context` and `candidate_examples` are dropped on failure | Yes: the retry and error mapping must not treat it as a retryable LLM failure | Yes, Q-R3d-2 / F4 |
| `pipeline_ready` not set False, `pipeline_error` absent | Only if `state` is reused | Yes, F4 (ambiguous case only) |
| No quarantine reset | Yes (F1) | No |
| No `reset_refinement_state` | No: fresh state, and equivalence passes with `state={}` | R3b doc only, not LEDGER |
| No `token_stats` write | Observability only; the counter is process-cumulative, so per-job only with one child per job | Only generally ("token observability") |
| `state["export_mode"]` written raw, no label mapping | No: Stage 1 coerces, and an unknown value stays strict | No need |
| `pwml_strict_db` absent, reads True | Low: only RAG rounds read it here, and the checkbox default is also True | R3d section 1 only |
| Settings source | Yes (F2) | No |
| `preprocess_fn=None` | No | R3b section 7.3 |
| `install_app_defaults` not installed | No on this path: `reporter` and `state` are passed explicitly at `extraction.py:714, 950-951`. The harness loads AppTest first (`harness.py:315`), so equivalence would not catch a forgotten pass | R2 F1 carried |
| No deadline parameter (the app has none either) | Medium: `PathwayRequest.deadline` has nowhere to go, so only the child kill bounds a run | Not named; W4b F6 covers RAG only |

## Check 3: gate audit (from git and committed evidence)

| task | biology diff | golden files | existing tests edited, and cover | RESULT + LEDGER | survivors (reports) | new `*quarantine_report.json` | cache / secret | SMOKE 511 on merge |
|---|---|---|---|---|---|---|---|---|
| R3a `220b1621` | none | unchanged | none (1 new, labelled) | yes | 0 (115) | none | none | pre-merge log; post-merge exit 0 only |
| R3b `4ea3524a` | none | unchanged | `test_preprocessor`, 4 path lines, Q-R0-1 | yes | 0 (193) | none | none | yes, log |
| R3c `51db01a6` | none | unchanged | s2orch, one function, Q-R3c-1 | yes | 0 (192) | none | none | yes, log |
| R3d `2e3d6a28` | none | unchanged | c060a (Q-R0-1 note), s2orch (Q-R3d-1), equivalence (Q-R3d-3), harness (see F6) | yes | 0 (209) | none | none | yes, log |

- Docs-only commits are truly docs-only: only the 5 merges touch `src` or `tests`.
- Reports without a survivor field are all `.pin.json` manifests.
- The secret check was a shape scan of added lines (0 hits). I did not read `.env`, so it is not a comparison against real values.
- E2E-2, not an R3 task, added a `quarantine_report.json` at `55f6d9d4`; it was renamed at `1062ce0d`. The census is 83 at base, tip and `71cfe7d3`.
- W4b's 3 test edits are authorised behaviour changes under Q-E2E-1, not R-series.

## Check 4: accumulated test edits

Five existing test files changed for R3:
- `test_preprocessor`: 4 paths;
- c060a: +7/-3;
- s2orch: one function, rewritten twice, net one hunk;
- `test_equivalence`: +2;
- `harness.py`: service arm rewritten, now stricter because it starts from the text.

The golden-seam files are untouched. Non-vacuity holds:
- The source-reading tests I grepped for moved names are all either repointed or package-wide (`test_app_still_joins_text_parts_that_way` scans `app/` and `service/`).
- No positive source assertion on a moved name is left passing on an unused app import.

Can a real regression pass unnoticed? Only in narrow places. Ordinary moved-code regressions are caught by the new stage suites and by the cassette keys, but these remain exposed:
- paths the fixture cannot see (additions, the filter, RAG, research mode) rest on mock-based unit tests, c055 and c060a (F4);
- the attempts keyword seam at the app call site (F5);
- headless-only omissions such as F1, which no arm exercises.

## Check 5: carry-forward list

See F3. R3d F2, F4 and F5, R3c Q5, R2 F1 and the R3 lesson are tracked. R3d F3 needs no tracking, because the stage suites kill those mutations.

## Files

- `C:/t/rev-R3-series/VERDICT.txt`
- Scripts in `C:/t/rev-R3-series/`: `compose.py`, `fnc.py`, `names.py`, `gate.sh`, `evid.py`, `evid2.py`, `evid3.py`
- Reviewed: `C:/t/ws-R3d-post/src/t2pw/service/extraction.py`, `C:/t/ws-R3d-post/src/t2pw/app/streamlit_app.py`, `C:/t/ws-R3d-post/tests/equivalence/harness.py`
