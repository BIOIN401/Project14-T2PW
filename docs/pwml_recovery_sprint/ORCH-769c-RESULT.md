# ORCH-769c: the WNT PCP crash, an interactions-only graph has nothing to draw (D-112 § 3c). RESULT

**Lane:** `orch769c/empty-graph-render`, base `e3d5b65f` (= `main`). The implementer was a general-purpose agent
running unattended. Not merged, not pushed, not self-approved: an independent reviewer judges the diff. No LLM call,
no pipeline run, no network. Nothing in `src/t2pw/app/streamlit_app.py`. No gate change (D-106). No prompt change.

| commit | what |
|---|---|
| `7ba7836f` | tests first, on base code: render, driver zero counts, interactions-only AppTest |
| `43bf388b` | the AppTest reordered so that at base it fails on the button lookup (§ 5, item 1) |
| `1a832f4d` | test first: the driver counts the canonical payload (finding F-2) |
| `814d6da8` | **fix A**: nothing to draw returns `b""`; the driver keeps final zero counts |
| `31c64c6f` | **fix B**: the driver counts the payload it writes as `final_mapped.json` (F-2) |
| `9360976c` | the seam-golden pin moved deliberately (merge rule 4), with its derivation |
| docs commit | this file, `evidence/orch769c/`, `evidence/g11/ORCH-769c/02-31`, `evidence/g11/pin/ORCH-769c/` |

**Outcome.**

* **Corpus (D-112 § 2): ACCEPT.** The pass rate goes from 51/91 to **52/91**. The fixed leg is
  `ORCH-767/wnt_pcp-strict/2026-10-24_0131`. No passing leg turns failing and no gold recall is lost
  (`evidence/orch769c/19-corpus-compare.log`).
* The other 92 legs are byte-identical in every recorded field, file digests included (`20-corpus-legdiff.log`).
* The target leg now writes `pathway.review_required.pwml`: release `review_required`, required gate ok, no IR error.
  Its PWML has the same bytes as the export ORCH-768 recorded "for information" at base (§ 6).

---

## 1. The defect

Measured on `ORCH-767/wnt_pcp-strict/2026-10-24_0131` (PMC10425814, strict):

1. The quarantine boundary passed. It admitted 18 processes (15 `core_accepted`, 3 `auxiliary_accepted`), all of them
   interactions, and quarantined 48. The quarantine input held 11 reactions, 2 transports and 53 interactions
   (`quarantine_report.json` admissions by pointer). Release: `review_required`, completeness 0.6.
2. The app then calls `initialize_refinement_review_state` (`streamlit_app.py:6861`). At `:1877` it draws the review
   diagram, and `render_draft_graph_to_png_bytes` (`draft_graph_render.py:261-262` at base) raised
   `RuntimeError("DraftGraph has no reaction nodes — nothing to render.")`. The diagram lays out only `reaction`,
   `transport` and `reaction_coupled_transport` nodes (`_FLOW_RXN_KINDS`); `interaction` nodes are never drawn.
3. The raise came before `refinement_working_json` was set (`:1880`). So `_render_review_refine_section` returned
   early (`:2686`), the `refinement_generate_pwml` button (`:2948`) was never rendered, and the driver failed
   (`driver.py:3140-3160`). The app's catch-all shows the error with `st.error` and `st.exception` (`:6869-6876`).
4. The report was wrong twice:
   * the driver filed the leg as `crash`, because "no reaction nodes" matches no `_NO_REACTION_MARKERS` entry;
   * it reported `reactions = 11, transports = 2` for a graph with none (F-2).

A second, nodeless raise at base (`:256-257`, "DraftGraph has no nodes") had the same effect on an empty graph.

## 2. Every caller of `render_draft_graph_to_png_bytes`, and how each handles `b""`

Grepped over `src/`, `scripts/` and `tests/` at the tip:

| caller | what it does with the bytes | `b""` |
|---|---|---|
| `streamlit_app.py:1877` `initialize_refinement_review_state` | stores them as `refinement_graph_bytes` | stored. Review state now completes; QA (`:1878`) runs. |
| `refinement_graph_bytes` reader `:2702-2711` (review page) | `if isinstance(..., (bytes, bytearray)) and graph_bytes:` then `st.image` and the download button | skipped: no image, no download button |
| reader `:2819-2825` ("Submit Changes to AI") | `if not ... graph_bytes: st.error("No refinement graph image is available."); st.stop()` | **refuses**. This is the known cost, § 7, item 2. |
| reader `:2868` (checkpoint) and `:2939` (undo) | copies the value | a plain value |
| `interactive_curator.py:610` `apply_patch_and_rerender` | initialised to `b""` (`:605`). The render and QA sit in one `try`, and an exception sets `qa_report = {"error": ...}` | returned as is. **Changed:** a patch that leaves nothing to draw now gets a real QA report instead of `{"error": "Graph render or QA failed: ..."}`. The bytes were `b""` at base too. |
| result of `apply_patch_and_rerender`, at `streamlit_app.py:2895` | stored as `refinement_graph_bytes` | as above |
| `interactive_curator.py:261` `build_interactive_curator_messages` | `base64.b64encode(graph_png_bytes or b"")` | tolerated. Only reached through "Submit Changes to AI", which refuses `b""` first. |
| `src/draft_graph_render.py` | re-export shim (`import *`) | n/a |
| `tests/test_interactive_curator.py:348, :432` | monkeypatched to return `b"png"` | n/a |

No other reader exists. `draft_graph_png_bytes` (`streamlit_app.py:5697`) is popped only, never written. The ORCH-768
harness calls `initialize_refinement_review_state`, not the renderer.

## 3. The diff (`git diff e3d5b65f 9360976c -- src`: 2 files, +15 −6)

**`src/t2pw/pipeline/draft_graph_render.py`** (fix A). Both raises are replaced by one guard. A nodeless graph also
reaches it, because `_parse_graph([], [])` returns no reactions:

```python
    ordered, node_by_id, rxn_reactants, rxn_products, rxn_enzymes = _parse_graph(nodes, edges)

    # Only reactions and transports are drawn. With none (no nodes at all, or an
    # interactions-only pathway, ORCH-769c) there is no diagram: b"" is the value every
    # caller already treats as "no image", where raising stopped refinement review.
    if not ordered:
        return b""
```

**`src/t2pw/batch/driver.py`**, the counts line in `_drive` (base `:2998-2999`):

```python
    mapped = (
        artifacts.get(CANONICAL_PAYLOAD_KEY)          # fix B
        or artifacts.get("final_mapped_db")
        or artifacts.get("final_mapped")
    )
    if isinstance(mapped, dict) and mapped:          # fix A: zeros kept; absent payload = not recorded
        outcome.counts.update(_payload_counts(mapped))
```

The base code was `_payload_counts(final_mapped_db or final_mapped)`, merged with `if value`. Fix B uses the precedence
`_add_identity_artifacts` (`:1861-1865`) already uses for `final_mapped.json`, so the counts and the file describe one
object. The precedence is inlined, not extracted into a shared helper, so `_add_identity_artifacts` is not touched;
a shared helper is the obvious follow-up. `_NO_REACTION_MARKERS` is **not** changed, as the brief says. Fix A removes the crash, and the
decline-or-export question is open (§ 7, item 1).

## 4. Tests (`git diff e3d5b65f 9360976c -- tests`: 4 files)

| test | kind | base | tip |
|---|---|---|---|
| `tests/test_orch769c_empty_graph_render.py::test_g9_an_interactions_only_graph_renders_to_empty_bytes` | G9 | RuntimeError | pass |
| `...::test_g9_a_graph_with_no_nodes_renders_to_empty_bytes` | G9 | RuntimeError | pass |
| `...::test_preservation_a_graph_with_a_reaction_still_renders_a_png` | preservation | pass | pass |
| `tests/test_batch_driver.py::test_g9_final_zero_counts_replace_the_stale_stage2_counts` | G9 (fix A) | `assert 1 == 0` | pass |
| `...::test_g9_counts_come_from_the_canonical_payload_not_the_pre_quarantine_one` | G9 (fix B) | `assert 1 == 0`, also at fix A alone | pass |
| `...::test_preservation_an_absent_final_payload_keeps_the_stage2_counts` | preservation | pass | pass |
| `tests/test_streamlit_quarantine_boundary.py::test_g9_an_interactions_only_pathway_opens_review_and_exports` | G9, AppTest, run alone | `KeyError: 'refinement_generate_pwml'` | pass: button found, `pwml_export_result.ok`, the protein in the XML, `refinement_graph_bytes == b""` |
| `tests/test_batch_driver_seam_golden.py` `GOLDEN` | pin moved (merge rule 4) | n/a | § 5.3 |

The AppTest payload: two UniProt-identified proteins (`P48506`, `P48637`) and one `binds` interaction. There is no
reaction, transport or compound. The Stage-0 context names both proteins as the requested core. Quarantine admits it
at base too; the test asserts that before the lookup that fails.

## 5. Proofs

### 5.1 G9: RED at the test-first commits, GREEN at the tip

| G11 | commit | run | result |
|---|---|---|---|
| `02` | `7ba7836f` (base code) | render tests and two driver tests | **3 failed, 2 passed**: both render G9 raise `RuntimeError`, zero-count G9 `assert 1 == 0` |
| `03` | `7ba7836f` | the AppTest alone, first order | failed at `_app_exceptions(at) == []` on `'DraftGraph has no reaction nodes — nothing to render.'` (the `st.exception` element). This is the right cause at the wrong line; see item 1 below. |
| `05` | `43bf388b` | the AppTest alone, reordered | **failed: `KeyError: 'refinement_generate_pwml'`** |
| `06` | `1a832f4d` | three driver tests | **2 failed, 1 passed**: both G9 `assert 1 == 0` |
| `07` | `814d6da8` (fix A only) | all six unit and driver tests | 5 passed, **1 failed**: the canonical-count G9. Fix A alone does not fix the reported counts. |
| `13` | `9360976c` | the six, plus `test_batch_driver_seam_golden.py` | **8 passed** |
| `14` | `9360976c` | the AppTest alone | **1 passed** |

Item 1: the brief named the base failure as the button lookup, so the precondition and the lookup now come first.
The exception check now comes after the click. G11 `03` stays on the record as the run that showed the cause.

### 5.2 G9 on the real legs (`evidence/orch769c/g9_render_legs.py`; G11 `15` base export, `16` tip)

This is the diagram exactly as the review step draws it: `build_draft_graph`, then `render(..., dpi=100)`.

| leg (`final_mapped.json` sha256) | processes | base `9b1943a8` (src tree) | tip |
|---|---|---|---|
| ORCH-767 WNT PCP `0131` (`5658510e…`) | 0 R, 0 T, 18 I | **raised** `RuntimeError: DraftGraph has no reaction nodes` | returned `b""` (0 bytes) |
| ORCH-759 WNT PCP `1432` (`02e4e6fc…`) | 3 R, 2 T, 11 I | PNG 32,619 B, sha256 `045608d2…` | **byte-identical**: 32,619 B, `045608d2…` |

Outputs: `g9_render_legs_base.json`, `g9_render_legs_tip.json`.

### 5.3 The seam-golden pin move (merge rule 4), derived

Fix B moves `tests/test_batch_driver_seam_golden.py`. Every fixture there carries `canonical_export_payload = _CANON`
(one reaction, no entities) beside `final_mapped_db = _MERGED`. Captures from `evidence/orch769c/golden_capture.py`,
which reuses the test module's own `_legs` and `_observe`:

| capture | tree | reproduces the committed `GOLDEN` |
|---|---|---|
| `golden_base.json` (G11 `09`) | export of `e3d5b65f` | **7/7** byte-identical |
| `golden_fixa.json` (`10`) | export of `814d6da8` (fix A only) | **7/7**: fix A moves nothing |
| `golden_tip.json` (`11`) | `31c64c6f` | 1/7 |

Delta (`golden_delta.json`, `golden_delta.py`, G11 `12`):

* **slot 0** (status|stage|kind) and **slot 1** (message) move on **0 of 7** legs;
* **slot 2** (digest) moves on **6 of 7**. `input_timeout` never reaches the counts line and is unchanged;
* on every moving leg the only differing observable field is `row`, and inside it only `counts`:
  `compounds 2 → 0`, `entities 3 → 0`, `proteins 1 → 0`. `reactions 1` and `transports 0` are unchanged. Zero keys are
  added and zero dropped (`only_counts_moved: true`).

The new digests and this derivation are written in the test's `GOLDEN` comment block, in the same form as C-053 to C-083.

### 5.4 Non-vacuity

The stepwise commits are the mutations:

* reverting the render guard is the base code (G11 `02`, `05`);
* reinstating the `if value` filter is the base code for the zero test (`02`, `06`);
* dropping the canonical precedence is fix A alone, which leaves the canonical test red (`07`).

## 6. The corpus measurement (D-112 § 2), and the harness pin delta

**Run.** ORCH-768 harness `orch768_replay_corpus.py` from `.claude/worktrees/orch768` (`4baa9bd7`), read-only, with
`PYTHONDONTWRITEBYTECODE=1` so nothing is written beside it:

* base: `run --src C:/t/bt-orch769c/base/src`, an export of `e3d5b65f` (src tree `9b1943a8`). G11 `17`, 26.9 s;
* tip: `run --src <worktree>/src` (`9360976c`). G11 `18`.

Network attempts: 0 on every leg. `git status` is clean after the run.

* `corpus_base.json` is **byte-identical** to ORCH-768's committed `orch768_corpus_baseline.json` (sha256 `0e17b69a…`).
* `corpus_base.txt` differs from ORCH-768's `.txt` on line 1 only: the resolved `t2pw` path.

**Compare** (`compare base.json tip.json --target ORCH-767/wnt_pcp-strict/2026-10-24_0131`, G11 `19`, exit 0):

```
pass rate base 51/91 -> tip 52/91
fixed: ['ORCH-767/wnt_pcp-strict/2026-10-24_0131']
passing legs turned failing: none
gold recall lost: none
still failing, at a different step (information): none
target ORCH-767/wnt_pcp-strict/2026-10-24_0131: FIXED
D-112 § 2: ACCEPT
```

**Beyond pass and fail** (`corpus_leg_diff.py`, G11 `20`): 1 of 93 leg records changed in any field, the target. On
it, at base: `failure_stage = refinement_review_crash` and `refinement_error = RuntimeError: DraftGraph has no
reaction nodes`. At the tip:

* `failure_stage = null` and `pwml_written = true`;
* `export.ok`, `release = review_required`, `pwml_name = pathway.review_required.pwml`;
* `required_gate_ok = true`; `ir_blocking`, `ir_validation` and `stage3_contract_codes` all `[]`;
* `quarantine_reused = true`.

The written PWML (`e68f008b…`) and the required-gate and pre-freeze reports are **byte-identical** to the ORCH-768
"export had review opened" record at base. The fix unblocks the exporter and changes nothing it produces.

**Harness pin delta.** The orchestrator applies it in whichever of ORCH-768 and ORCH-769c merges second; I did not edit
the harness or its test. Measured by running ORCH-768's own `tests/test_orch768_replay_corpus.py` (from `4baa9bd7`)
in scratch trees, one with base `src` and one with tip `src`:

* G11 `30`, base: **9 passed**;
* G11 `31`, tip: **8 passed, 1 failed**.

The failure is `test_replay_reproduces_the_recorded_verdict[ORCH-767/wnt_pcp-strict/2026-10-24_0131]`:
`H.fidelity(...) == []` gets `['outcome recorded=refinement_review_crash replay=pwml']`. The exact delta:

1. `EXPECTED[WNT_PCP]`: `{"pwml_written": False, "failure_stage": "refinement_review_crash"}` becomes
   `{"pwml_written": True, "failure_stage": None}`. Do not pin `pwml_matches_leg`: the leg wrote no PWML, so it is
   `False`.
2. The fidelity assertion for this one leg expects exactly `['outcome recorded=refinement_review_crash replay=pwml']`
   instead of `[]`. That is the one deliberate outcome change. The other two legs keep `[]`.
3. Nothing else moves. `test_the_fidelity_check_bites` (it reads WNT PCP's release, `review_required` on both sides)
   and `test_two_replays_are_identical` pass at the tip.

## 7. Not settled (for the product owner)

1. **Should an interactions-only pathway be DECLINED (`no_reactions`) or EXPORTED (`review_required`)?** Today the two
   seams disagree:
   * the batch driver declines a payload that has no reaction and no transport **at Stage 2**
     (`driver.py:2888-2896`, `KIND_NO_REACTIONS`);
   * the quarantine boundary admits one, because with an undeclared core only "not empty" counts and with a declared
     core interactions can be `core_accepted` (`strict_quarantine.py:1214-1226`).

   So a pathway that *starts* interactions-only is declined, and one that *becomes* interactions-only after quarantine
   reaches export.

   **What this lane does by default: the pathway exports.** On WNT PCP that gives `pathway.review_required.pwml`,
   which is not `release_ready` and not strict benchmark success. It also has thin content: 9 proteins, 6 complexes,
   0 reactions, and **1 drawn interaction of 18**. The IR withholds the other 17 as `untyped_interaction`; their raw
   types are `binds`, `binding`, `part_of`, `recruits` and `dissociates`, and the one kept is `Wnt5a activates YAP`,
   type Activation. The withholding is pre-existing writer behaviour, identical at base.
   * **(a) Export as `review_required`** (the current default). It follows merge rule 7 and contract §§ 4 and 7:
     incomplete but correct, preserved, never `release_ready`. The cost: a PWML whose drawn content is one interaction
     may not meet § 4's "useful". The Stage-2 decline stays inconsistent with it.
   * **(b) Decline as `no_reactions` / `diagnostic_only`.** Add the marker, or a post-quarantine "no reaction and no
     transport survived" check in the driver. The cost: WNT PCP goes back to 0 PWMLs (the corpus loses this leg's
     pass). Interaction-centred signalling pathways, which PathWhiz does represent, could never export in strict mode.
   * **(c) Align Stage 2 with quarantine** (stop declining interactions-only at Stage 2). The cost: more thin exports
     like this one; a driver change and a pinned test (`test_empty_pathway_is_no_reactions_not_unknown`) move.
   * Separately, whether `untyped_interaction` withholding should map `binds`/`part_of` to a PathWhiz type is a
     writer question (D-106 territory: not a gate to relax here).
2. **The known cost of `b""`.** In the interactive app the review page shows no diagram and no download button for
   such a graph, with no caption saying why, and **"Submit Changes to AI" refuses** with "No refinement graph image is
   available." (`streamlit_app.py:2823-2825`). AI refinement is therefore off for interactions-only graphs. Batch is
   unaffected. Either fix needs a change in the protected `streamlit_app.py`:
   * a caption, or
   * letting the curator round run without an image (`build_interactive_curator_messages` already tolerates `b""`).

   It needs a D-xxx.
3. **Fix B goes beyond the brief's literal wording.** The brief said "stop dropping zero counts, so the reported
   counts are the final ones". Measured (F-2), that alone leaves WNT PCP reporting `reactions = 11, transports = 2`,
   because the counts line read the pre-quarantine `final_mapped_db`. Fix B is in the same statement and adopts the
   precedence `_add_identity_artifacts` already uses. Its cost is the seam-golden move in § 5.3. It is a separate
   commit (`31c64c6f` plus `9360976c`), so the orchestrator can drop it and keep fix A if the reviewer rules it out of
   scope. Fix A alone still fixes the crash and the corpus leg, because the counts are not in the corpus verdict.
4. **The reported counts are now the canonical payload's.** That means the post-quarantine, exported graph. A reader
   who wants "what mapping produced before quarantine" no longer finds it in `counts`. The quarantine report and
   `merged_payload.json` still carry it.

## 8. Findings, failures and dead ends

1. **F-1. The crash is the only blocker on WNT PCP.** ORCH-768 predicted it (`export_if_refinement_opened.ok`), and
   it is confirmed byte for byte: the tip's PWML is the base's counterfactual PWML (§ 6).
2. **F-2. The diagnosis's "stale Stage-2 count" was wrong for this leg. Believed, then disproved.** The brief said the
   `if value` filter kept the Stage-2 `reactions = 11`. Two facts disprove it.
   * (a) The leg's reported `proteins = 32, entities = 52, compounds = 1` are not the Stage-2 numbers
     (`merged_payload.json`: 45, 55, 0). They are not the canonical numbers either (9, 21, 0). So the counts line read
     a third payload, `final_mapped_db`.
   * (b) The quarantine input held 11 reactions and 2 transports (66 admissions: 11 + 2 + 53). Between
     `final_mapped_db` and the quarantine only enrichment and pre-freeze resolution run, and neither adds or removes a
     process row (ORCH-768-RESULT § 1).

   So `final_mapped_db` itself held 11 and 2, and the `if value` filter never fired on this leg. **Not directly
   observed:** `final_mapped_db` is not persisted in the leg, so this is an inference from (a) and (b). G11 `07` shows
   the consequence on the synthetic copy of the shape.
3. **F-3. The AppTest's first RED was the right cause at the wrong line** (G11 `03`). At base the app shows the render
   exception with `st.exception`, so `_app_exceptions` catches it before the button lookup. The test was reordered
   and committed separately (`43bf388b`), and RED was recorded again (`05`).
4. **F-4. A dev probe of the tip before the fix was committed.** G11 `04` ran the AppTest with fix A applied but not
   committed (`worktree dirty=True`), to confirm the payload exports before the test was frozen. The fix was then
   reverted and the test committed. RED (`05`) and GREEN (`14`) are both at clean SHAs. Likewise G11 `08`: fix B
   uncommitted, the canonical test and the golden. It showed the golden move, which § 5.3 then derived at clean trees.
5. **F-5. Exporting a base tree for `pinned_pytest --expect-tree` needs `pyproject.toml`.** G11 `28` was refused
   `EXPECT_TREE_NOT_A_CHECKOUT` (exit 98, no test run): `tree_pin.is_checkout` requires `pyproject.toml` and
   `src/t2pw`. Rerun as `29`.
6. **F-6. G11 `01` is a reservation only.** I allocated it to test the allocator's output format and never ran a job
   with it. It stays in the git-ignored `.staging/`, and `g11_evidence.py check` does not count it.

## 9. Gates (all at `9360976c`, each through `bounded_run.py` and `pinned_pytest.py`, 0 survivors on every report)

| gate | G11 | result |
|---|---|---|
| 1. G9 RED/GREEN | `02 03 05 06 07` / `13 14`, legs `15 16` | § 5.1, § 5.2 |
| 1. Corpus compare | `17 18 19 20` | **ACCEPT**, 51/91 → 52/91 |
| 2. Affected: render tests, `test_draft_graph_entity_kinds.py`, `test_batch_driver.py`, `test_batch_driver_quarantine_artifacts.py`, `test_batch_driver_seam_golden.py`, `test_batch_fetch.py`, `test_batch_pwml_artifact_naming.py`, `test_batch_report.py`, `test_batch_research_gate_fail_open.py`, `test_batch_run.py`, `test_orch759d_app_timeout.py`, `test_interactive_curator.py` | `21` | **250 passed** |
| 3. SMOKE (the 22 files of `TEST_MATRIX.md` `## Commands`) | `22` | **511 passed** = 508 + the 3 new tests in `test_batch_driver.py` |
| 4. `tests/test_eval_v2_*.py` (6 files) | `23` | **89 passed, 1 xfailed** |
| 5. `test_streamlit_quarantine_boundary.py` alone | `24` | **24 passed** (23 + 1 new) |
| 5. `test_c052_prefreeze_report_at_the_streamlit_seams.py` alone | `25` | **13 passed** |
| 5. `test_c055_rag_loop_wiring.py` alone | `26` | **14 passed** |
| 5. `test_batch_preflight.py` alone | `27` | 32 passed, 4 skipped, **1 failed** (see below) |
| ORCH-768 acceptance test, base / tip `src` (information, § 6) | `30` / `31` | 9 passed / 8 passed, 1 failed (the pin delta) |
| `g11_evidence.py check --task ORCH-769c` | n/a | exit 0, 30 artifacts, 0 non-compliant. 1 refused pin, which is `28` (F-5). |

The `test_batch_preflight.py` failure is
`test_the_message_names_the_problem_the_interpreter_and_the_cure` ("this project ships a .venv": a worktree has none).
It is on the known list (`TEST_MATRIX.md` § ORCH-757, and the ORCH-759 known-failures row: worktree only).

It fails identically at base: G11 `29`, an export of `e3d5b65f` run with `--expect-tree`. That run has 2 more
failures, which are artifacts of the partial export: `docs/batch_runner.md` and `run_overnight.bat` were not exported.
They pass in the worktree (`27`).

Not run: Chunk D (excluded by the brief), and the live pipeline (no LLM, no pipeline).

## 10. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

* **SMOKE 508 → 511.** `tests/test_batch_driver.py` gains 3 tests:
  * `test_g9_final_zero_counts_replace_the_stale_stage2_counts`
  * `test_g9_counts_come_from_the_canonical_payload_not_the_pre_quarantine_one`
  * `test_preservation_an_absent_final_payload_keeps_the_stage2_counts`

  Measured `511 passed` (G11 `22`). Merge rule 10's number moves with it.
* **`tests/test_streamlit_quarantine_boundary.py` 23 → 24** (alone). It gains
  `test_g9_an_interactions_only_pathway_opens_review_and_exports`.
* **New file `tests/test_orch769c_empty_graph_render.py`** (3 tests). It is in no chunk and not in SMOKE. Its focused
  selection is the file.
* **`tests/test_batch_driver_seam_golden.py` `GOLDEN` re-baselined**, 6 of 7 digests (§ 5.3).
* **ORCH-768 acceptance test**: `EXPECTED[WNT_PCP]` and its fidelity expectation move as in § 6, in whichever lane
  merges second.

## 11. Files

* `src/t2pw/pipeline/draft_graph_render.py`, `src/t2pw/batch/driver.py`
* `tests/test_orch769c_empty_graph_render.py` (new), `tests/test_batch_driver.py`,
  `tests/test_streamlit_quarantine_boundary.py`, `tests/test_batch_driver_seam_golden.py`
* `docs/pwml_recovery_sprint/evidence/orch769c/`:
  * scripts: `g9_render_legs.py`, `golden_capture.py`, `golden_delta.py`, `corpus_leg_diff.py`;
  * outputs: `g9_render_legs_{base,tip}.json`, `golden_{base,fixa,tip,delta}.json`, `corpus_{base,tip}.{json,txt}`;
  * one log per G11 job: `NN-<label>.log`.
* `docs/pwml_recovery_sprint/evidence/g11/ORCH-769c/02-31*.json`, `evidence/g11/pin/ORCH-769c/*.pin.json`

## 12. Integration (after review round 0, APPROVE WITH NOTES)

The orchestrator asked for an integration step before the merge. Everything here ran at `64d400f9`, each job
through `bounded_run.py` (and `pinned_pytest.py` for pytest), with 0 survivors.

| commit | what |
|---|---|
| `bd4ac872` | `git merge --no-ff sprint/pwml-recovery` @ `509fe185`: ORCH-766, ORCH-768, ORCH-769b. Parents `5d15c2f2` and `509fe185`. No conflicts. |
| `cdbf46c9` | **pin move (merge rule 4)** in `tests/test_orch768_replay_corpus.py`, exactly as § 6 documents. `EXPECTED[WNT_PCP]` becomes `{"pwml_written": True, "failure_stage": None}`, and that leg's fidelity expects `['outcome recorded=refinement_review_crash replay=pwml']`. Nothing else in the file changes. |
| `64d400f9` | docstring only: `test_g9_final_zero_counts_...` no longer blames WNT PCP's `reactions = 11` on the `if value` filter (F-2) |

After the merge, `git diff 509fe185 64d400f9 -- src` is the same 2 files, +15 −6, as § 3. The harness and the manifest
are unchanged from `4baa9bd7`.

| rerun | G11 | result |
|---|---|---|
| `tests/test_orch768_replay_corpus.py` | `32` | **9 passed** |
| G9 file, `test_batch_driver.py`, `test_batch_driver_seam_golden.py` | `33` | **66 passed** |
| SMOKE (22 files) | `34` | **511 passed** |
| `test_streamlit_quarantine_boundary.py` alone | `35` | **24 passed** |
| corpus, sprint tip `509fe185` (export, src tree `f64fca5e`) | `36` | **54/91**, 0 network attempts |
| corpus, merged tip `64d400f9` (src tree `322d398d`) | `37` | **56/91**, 0 network attempts |
| `compare` sprint → merged, `--target` WNT PCP | `38` | **ACCEPT**: 2 fixed, 0 turned failing, 0 gold recall lost |
| every leg record, sprint → merged (`corpus_leg_diff.py`) | `39` | 2 of 93 legs changed, the 2 fixed |

`PATHBANK_DB_*` was unset in the shell, every corpus job ran under `env -u PATHBANK_DB_{HOST,PORT,USER,PASSWORD,NAME}`,
and the worktree has no `.env`.

**The expected count was 54; the measurement is 56.** 54/91 is the *sprint tip itself* (with ORCH-769b), not 53.
Over that, ORCH-769c fixes two legs:

* `ORCH-767/wnt_pcp-strict/2026-10-24_0131`, as in § 6. Its PWML is the same, `e68f008b…`.
* **`runs_verify/2026-08-22_2017/PMC12856317`, new on the merged tree.**
  * At `e3d5b65f` it failed the final gate on `unknown_protein_modifier_reference`.
  * ORCH-769b now quarantines that reaction (`quarantined_broken_reference: 1`). That leaves 2 interactions and no
    reaction, so **at the sprint tip it crashes in refinement review with this lane's defect**
    (`refinement_review_crash`, "DraftGraph has no reaction nodes").
  * With ORCH-769c it writes `pathway.review_required.pwml` (`8e7d4ed8…`): release `review_required`, required gate
    ok. Those bytes equal the sprint-tip "export had review opened" record.

**Finding F-7, for § 7 item 1. The new PWML draws no process at all.** The PMC12856317 PWML has 0 reactions and
0 interactions, and only 2 compounds and 1 protein (ALAS2). The IR withholds both interactions:

* `heme inhibition of ALAS2`: `bare_protein_endpoint`;
* `ALAS2 cofactor binding`: `untyped_interaction`.

WNT PCP draws 1 of 18. So on the merged tree this lane's default ("export as `review_required`") already ships one
PWML with **no drawn process**. Contract § 4 calls `review_required` a "valid, useful PWML". The review crash had been
acting, by accident, as a "nothing drawable" guard. I did not decide it.

The product owner's options are § 7 item 1's, now sharper:

* (a) keep exporting;
* (b) decline when no reaction or transport survives;
* (c) a narrower rule: decline when the IR draws no process, and export otherwise. Under (c) WNT PCP (1 drawn
  interaction) exports and PMC12856317 (0) does not. The cost is a new check after the IR, in the exporter or the
  driver.

Until that is ruled, both legs count as corpus passes, because the harness counts any written PWML.

`g11_evidence.py check --task ORCH-769c`: exit 0, 38 artifacts (`02`-`39`), 0 non-compliant.

**TEST_MATRIX delta, unchanged by the merge:** SMOKE 511, boundary file 24, plus the ORCH-768 acceptance test's pin
move, applied here in `cdbf46c9`.
