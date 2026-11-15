# C live test (D-W21c): T2PW_TRANSPORT_SIDES, equal / missing-side transports

Branch `pwml/final-payload`. Code under test: commit `e86cc420` (`src/t2pw/service/transport_sides.py`, hook in
`service/extraction.py`, `tests/test_transport_sides.py`). `transport_sides.py` has not changed since that commit.
This document analyses the artefacts of the earlier implementer agent (it hung before writing a result) and adds
`compare_output.md/.txt/.json`, `transports_table.md`, `noise_floor.md` (scripts `transports_table.py`, `noise.py`, both plain
python over the stored runs, no model call). No pipeline run, no LLM call, no heavy job was started by this analysis.

## 0. Conclusion

* **Fixed? Partly, and not reliably.** On the four stored defective payloads the step repaired all four rows in one call
  each (replay, section 3). On fresh live draws the defect was reproduced only for TubI (PMC6112128: 3 of 5 draws defective);
  there the step repaired 1 of 2 defective draws (the other got an empty reply `{}`; re-asking the same payload 3 times gave
  `{}`, a valid repair, `{}`). The three enterobactin cases did **not** occur in 12 fresh PMC12096016 draws, so they were fixed
  on replay only.
* **Loss shown? None shown, but very little was tested.** Only one live run (PMC6112128 on_1) was ever changed by the step.
  In it the only Stage-1 difference is the `to` side and one new `extracellular` state (exact paired diff). Everything after
  Stage 2 is a fresh sample and differs from the off arm by about as much as two off draws differ from each other. The five
  controls with transports (3 papers, 6 on-draws) triggered zero repair calls and zero changes. That is a handful of draws,
  not a proof of no-loss.
* **Two quality problems found by reading the quotes** (section 4): the quote check proves the quote is in the paper, not that it
  supports the state written. Replay case 1 wrote an inside state `intracellular` (with `subcellular_location: cytoplasm`) on the
  strength of "out of the cell", and replay case 2 wrote `cytoplasmic state` as the destination of ferric-enterobactin import on the
  strength of "imported via TonB-dependent uptake", which names no compartment. The second is a biology claim the paper does not
  make (and TonB-dependent outer-membrane uptake delivers to the periplasm).
* **Recommendation: enable after more evidence (do not enable by default now).** Reasons and the list of what to add are in
  section 9. The step is cheap, fail-open and acts on no row that already has two distinct sides, so its risk is bounded, but
  D-W21c says adopted only if it fixes AND keeps everything, and neither half is established beyond a few draws.

## 1. What the step does

Applied once to the Stage-1 payload, after the Stage-1 boundary, in `service/extraction.run_stage_one`, only when
`T2PW_TRANSPORT_SIDES` is 1/true/yes (default off; read at call time).

1. Detect (pure Python): a transport is deficient when its effective from or to state is empty, or both are equal after
   `process_normalizer._normalize`. Effective side = row-level `from_biological_state`/`to_biological_state`, else the single
   `biological_state` declared on the matching `elements_with_states` left/right row.
2. No deficient row: zero model calls, the very same payload object is returned.
3. Otherwise one model call (max 12 rows; same model as Stage 1, temperature 0, JSON), with, per row, the cargo, the row
   evidence, a window of the evidence sentence plus one sentence either side (found with punctuation/case folding so U+2010
   and ASCII hyphens agree), the declared state names, and the paper text. The system prompt tells the model that "secreted",
   "out of the cell", "efflux" state an extracellular destination and "imported", "uptake", "into the cell" an extracellular
   origin, never to guess a more specific compartment, to reuse a declared state name if one fits, and to return null for a side
   it cannot quote.
4. Accept a side only if its quote (>= 2 words) is found in the paper text; keep every already-stated side (except when the
   two sides were equal, then both are re-asked); reject a result that is an equal pair; skip a repair that would make the row an
   exact copy (same cargo, same two sides) of another transport; any exception leaves the payload unchanged and is recorded.
5. Writes only `from_biological_state`/`to_biological_state` (not `elements_with_states`) and, if a chosen state is not yet
   declared, one `biological_states` row (species only if all existing states share one; `subcellular_location` = the model's
   compartment string up to 40 characters; evidence = the quote). The record goes to `state["transport_sides"]` and
   `transport_sides_report.json`.

## 2. Unit tests

| run (tests/ dir) | selection | result |
|---|---|---|
| `cimplA` | `test_transport_sides.py` + `test_service_extraction_stage1.py` + `test_service_extraction.py` + `test_process_normalizer.py` + `test_process_normalizer_transport_states.py` | 1 failed, 121 passed (the failure was `test_repairs_both_missing_sides_from_quotes_with_dash_folding`: expected `Cytoplasm`, got `cytoplasm`) |
| `cimplA2` | same selection after the test was corrected | **122 passed**, 0 surviving processes |
| `cimplB` | golden seams `test_batch_driver_seam_golden.py` + `test_c011_freeze_seam_golden_equivalence.py` | **10 passed**, 0 survivors |
| `cimplBase`, `cimplBase2` | `test_streamlit_quarantine_boundary.py` on isolated base trees (e86cc420 and f4b0be9a) | **24 passed** each, 0 survivors |
| `cimplC`, `cimplC2` | `test_streamlit_stage8_export_contract.py` + quarantine boundary (C), quarantine boundary alone (C2), in the shared worktree | C: 1 failed, 27 passed; C2: 1 failed, 23 passed. The one failure is `test_research_mode_keeps_the_unmapped_candidate_and_does_not_block` (reactions moved pre-freeze, OPDA reaction; the test has no transports). Flag off, and the step is not on that path |

`tests/test_transport_sides.py` has 13 tests: repair of both missing sides with dash folding, U+2010/ASCII quote match, quote
not in text rejected and row unchanged, one-word/missing quote rejected, never writes an equal pair, one named side leaves the
other empty, stated side kept and a new side may not equal it, duplicate-twin skip, clean payload gives zero calls and the same
object, failure fails open, flag default off, and the `run_stage_one` hook (flag off no call, flag on records and repairs).
No test reaches the network (fake `chat_fn`).

Not established: the C/C2 failure is not shown to be unrelated by a bisect. What the evidence shows: it passes on the two isolated
base trees (24/24), it concerns reactions in a test with no transports, and the shared worktree had other agents' uncommitted edits
at the time (`wrapper vs HEAD ... dirty=True`). A re-run of that test on the merge result is the orchestrator's SMOKE gate.

## 3. The four stored cases (replay: the stored Stage-1 payload, the live model, one call each)

`replay_stored.py`, output `replay/replay_*.json`, `replay_stdout.txt` (4 calls, 0 errors, 0 rejected, input never mutated).

| # | case | Stage-1 sides before | sides after | quotes | new states |
|---|---|---|---|---|---|
| 1 | TubI tubercidin export (PMC6112128) | none / none | `intracellular` / `extracellular` | both "out of the cell" | `intracellular` (cytoplasm), `extracellular` |
| 2a | enterobactin secretion (PMC12096016, 08-21) | `cytoplasmic state` / none | `cytoplasmic state` / `extracellular` | "secreted to the extracellular environment" | `extracellular` |
| 2b | ferric-enterobactin import (same run) | none / none | `extracellular` / `cytoplasmic state` | from "extracellular ferric iron"; to "imported via TonB-dependent uptake" | none (reused) |
| 3 | enterobactin secretion (08-27) | none / none | `cytoplasm` / `extracellular` | "produced in the cytoplasm", "secreted to the extracellular environment" | both |
| 4 | enterobactin secretion (09-02) | `E. coli cytoplasmic state` on both | `E. coli cytoplasmic state` / `extracellular` | "produced in the cytoplasm", "secreted to the extracellular environment" | `extracellular` (species E. coli, `subcellular_location: extracellular environment`) |

Status of the four, taking live and replay together:

1. **TubI export**: reproduced live (off_1 neither side, on_2 neither side, on_1 one side). Replay fixed it. Live fixed 1 of 2
   defective draws; the same-payload debug re-ask fixed 1 of 3. So about 3 repairs in 6 calls; fix is real but unreliable.
2. **ferric-enterobactin import**: not reproduced in 12 fresh draws (the 5 draws that had the import all had `extracellular` ->
   `cytoplasmic state`). Replay "fixed" it, with the unsupported destination of section 4.
3. **enterobactin secretion 08-27**: not reproduced fresh. Replay fixed it with fully literal quotes (the cleanest case).
4. **enterobactin secretion 09-02**: not reproduced fresh. Replay fixed it. In the stored run Stage 2 appended a correct twin
   `E. coli cytoplasmic state` -> `extracellular space`; after the fix the row would be `... -> extracellular`, a near duplicate
   under a different state name, which the exact-copy skip does not catch. Stage 2 and the writer on that were not run.

## 4. Are the repairs supported by the paper text (reviewer check)

I grepped each quote in the stored paper text (`01_source_text.txt`; hyphens in the PMC12096016 text are U+2010).

| repair | quote present verbatim | supports the state written? |
|---|---|---|
| on_1 TubI: `to` = `extracellular` (live) | "out of the cell" in "TubI is responsible for transporting TBN out of the cell once synthesized": yes | Yes only if "out of the cell" entails extracellular (PO question 1). The step's prompt answers yes. `from` was already `cytoplasmic state` (the model's own, not touched by the step) |
| replay 1: `from` = `intracellular` | same quote | Weak. The text implies the cargo starts inside; it names no compartment, yet the state row carries `subcellular_location: cytoplasm`. TRACE advised leaving this side empty (normalizer then places generic `cell`) |
| replay 1: `to` = `extracellular` | same quote | As on_1 |
| replay 2a: `to` = `extracellular` | "secreted to the extracellular environment": yes (sentence "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment by a TolC-dependent process") | Yes, literal |
| replay 2b: `from` = `extracellular` | "extracellular ferric iron": yes (the sentence before the import sentence: "Following secretion, enterobactin chelates extracellular ferric iron, and the ferric-enterobactin complex is then imported via TonB-dependent uptake") | Yes, from the neighbouring sentence |
| replay 2b: `to` = `cytoplasmic state` | "imported via TonB-dependent uptake": yes | **No.** The quote names no compartment; the state is the one Stage 1 had declared. Not stated by the paper, and for TonB-dependent uptake the first destination is the periplasm. TRACE class (d): should stay empty |
| replay 3 both | "produced in the cytoplasm", "secreted to the extracellular environment": yes | Yes, literal |
| replay 4 both | same quotes | Yes, literal. `from` reuses the existing `E. coli cytoplasmic state` |

Cause: `_quote_in_text` only tests that the quote occurs in the paper; nothing ties the quote to the state name. This is a
detector weakness of the step, not a transcription slip.

## 5. Live arms: Stage-1 transports before and after

Driver `run_case.py` (real Stage 0, Stage 1, boundary, hook, Stage 2 with feedback loop, merge, offline normalizer; RAG off;
`T2PW_EXTRACTION_DRAWS=1`; stored paper text; flag on for the on arm). Full per-run rows: `transports_table.md`; the table
`compare_output.md` carries counts. ok / one-sided / equal(or neither) at Stage 1 (after the step for on runs):

| paper | run | before the step | after the step | normalized |
|---|---|---|---|---|
| PMC6112128 | off_1 | 0/0/1 (neither side) | same | 0/0/1 (`__auto_state__` both sides) |
| | off_2 | no transport | | |
| | on_1 | 0/1/0 (`cytoplasmic state` / empty) | **1/0/0** | 1/0/0 |
| | on_2 | 0/0/1 (neither side) | 0/0/1 (not repaired: reply `{}`) | 0/0/1 |
| | on_s1 | no transport (Stage 1 only) | | |
| PMC12096016 | off_1, off_2, on_1, on_2 | all 1/0/0 | unchanged | 1/0/0 |
| | on_s1..s8 (Stage 1 only) | s1 and s5 no transport; s2, s4, s6, s7, s8 two transports; s3 one; all ok | unchanged | not run |

PMC12096016 totals: 12 draws, 14 transports, all with two distinct sides; 0 deficient. The stored rate was 4 of 24 runs (about
17 percent), so 0 of 12 has probability about 0.11 under that rate: weak evidence, not a different rate. PMC6112128: 5 draws, 3
defective (off_1, on_1, on_2), 2 without any transport.

Repairs made in live runs: exactly one.

| run | transport | old sides | new sides | quote | supported |
|---|---|---|---|---|---|
| PMC6112128 on_1 | TubI-mediated TBN export (cargo tubercidin) | `cytoplasmic state` / empty | `cytoplasmic state` / `extracellular` | "out of the cell" | by entailment only (section 4) |

New state row: `extracellular`, species `Streptomyces tubercidicus NBRC 13090`, `subcellular_location: extracellular`,
evidence "out of the cell". After Stage 2 and the normalizer the state names are `cytoplasmic state`, `extracellular`,
`__auto_state__` (the last was also present without the repair, in every arm): no duplicate `extracellular`.

Rejected in live runs: PMC6112128 on_2, reason recorded as "no side with a quote found in the paper text". The recorded reason
is misleading (see section 6).

## 6. What the debugging agent found (`debug_reply.py`, `debug_on2_replies.json`, `logs/debug.*`)

It re-asked the on_2 Stage-1 payload (both sides empty, evidence "TubI is responsible for transporting TBN out of the cell once
synthesized") three times and kept the raw replies:

| try | raw reply | result |
|---|---|---|
| 1 | `{}` | rejected |
| 2 | `{"rows": [{"id": 0, "from": null, "to": {"state": "extracellular", "compartment": "extracellular", "quote": "out of the cell"}}]}` | accepted: `` / `extracellular`, `from` left empty |
| 3 | `{}` | rejected |

So the on_2 failure is not a quote-matching bug: the model (temperature 0, same input) sometimes returns an empty object, and
the code reports that as "no side with a quote found". There is no retry. A second observation: when the model does answer for
this paper it names only the `to` side and leaves `from` null, which is the PO's preferred behaviour for question 2 (the
normalizer then supplies the generic `cell`). These 3 calls are outside the arms and are not counted in the run table.

## 7. Arm-paired differences and noise floor

**Paired (same Stage-1 draw, Stage 1 before vs after the step; `compare_output.md`):** 19 on runs. 18 are "no fact differs
(byte-identical)" (all 17 runs with no deficient row, plus on_2 where the step rejected). The one that differs is PMC6112128 on_1:
the facts that differ are exactly the transport `(tubercidin, cytoplasmic state, '')` -> `(tubercidin, cytoplasmic state,
extracellular)` and the new state `(extracellular, extracellular)`. Nothing else at Stage 1 changed (expected: none else).
Caveat: compare.py compares facts (entities by type, reactions, transports, interactions, states, locations), not every JSON key;
the repair code only touches the two side fields and `biological_states`.

**Unpaired (normalized payload after Stage 2 and the normalizer; `noise_floor.md`).** Mean symmetric difference summed over
reaction, interaction, state, location and entity facts; one off-off pair and 4 off-on pairs per paper (2 off + 2 on draws):

| paper | off-off (n=1) | off-on (n=4, range) | on-on (n=1) |
|---|---|---|---|
| PMC11487621 | 59 | 61 (57-64) | 41 |
| PMC12096016 | 30 | 33 (28-38) | 48 |
| PMC12421875 | 28 | 55 (48-64) | 40 |
| PMC12444477 | 157 | 161 (150-173) | 186 |
| PMC6112128 | 41 | 40 (28-51) | 11 |

The on arm differs from the off arm by about as much as two off draws differ from each other, except PMC12421875 (55 vs 28),
where the on-on pair (40) lies between and a single off-off pair is a poor floor. Note that for 17 of 19 on runs the step did
nothing, so those on-vs-off differences are pure sampling noise by construction. With n=1 the floor has no spread estimate, and
a real loss smaller than a draw-to-draw difference would not show.

## 8. Controls, call counts, losses

* Controls PMC12421875, PMC12444477, PMC11487621, 2 on draws each: **0 deficient rows, 0 repair calls, Stage-1 pre == post
  (byte-identical) in all 6**. Their transports in every on and off draw are two distinct sides (including the hard cases:
  inner-membrane leaflet to periplasmic leaflet in PMC12444477, `cell` -> `extracellular` diffusion in PMC12421875). No false
  positive detected. The controls cannot show that a *deficient* row in a control paper is handled well, because none occurred.
  PMC12452463 was not run.
* Draw-to-draw variation already changes the transport count (PMC11487621 off_2 has no transport; PMC12444477 on_1 has 1, off_1
  has 2). Not a step effect.
* **LLM calls by the step** (`summary.json` `llm_by_stage`, stage "Stage 1 transport sides"): 1 call in PMC6112128 on_1 (14.2 s)
  and 1 in on_2; 0 in the other 17 on runs. Every other stage's request count equals its off counterpart (7 / 7 / 7 / 7 for the
  controls, 4 / 4 for PMC12096016); the exceptions are explained by Stage-1/Stage-2 retries (PMC12444477 on_2: 8 requests,
  5 Stage 2; PMC6112128 on_2: 2 Stage 1). No LLM errors, no failures, no exceptions in any run.
* **Anything lost by the on arm beyond the noise floor: none seen.** The one repaired run (on_1) has more reactions (9) and
  entities (33) than off_1 (8, 31) and off_2 (9, 26) and loses nothing relative to them beyond the off-off spread. One
  repaired run cannot support a no-loss claim.
* **`on_s*` samples.** `run_case.py --stage1-only` (Stage 0 + Stage 1 + the hook, stop before Stage 2): 8 extra draws of
  PMC12096016 (`on_s1`..`on_s8`) and 1 of PMC6112128 (`on_s1`). Their purpose (inferred from the driver; the agent left no note) was
  to collect more Stage-1 draws cheaply (2 requests each, no Stage 2) in the hope of drawing a defective transport for the step to
  act on. None did, so they exercised only the detector (14 healthy transports, zero false positives). `on_s2` of PMC6112128 never
  started: the bounded wrapper refused it because another job held the heavy lock (`logs/PMC6112128-on-s2.stdout.txt`, "the child
  was NOT started"); no run output exists for it.

## 9. The PO's three open questions from TRACE.md and how the step treated them

1. *Does "out of the cell" entail extracellular?* The step answers **yes by construction**: its system prompt states that
   "secreted", "out of the cell", "efflux" give an extracellular destination (and "imported", "uptake", "into the cell" an
   extracellular origin). This is a small biology rule that nobody has blessed, embedded in a prompt. It is what fixed TubI
   (live on_1, debug try 2). Without it TubI stays unfixable (no literal compartment word).
2. *Unnamed side: empty or `cell`?* The step leaves a side empty when the model returns null (live debug try 2, on_1 where `from`
   was already the model's own). But it does not enforce this: replay case 1 invented `intracellular`/cytoplasm for `from`, and
   replay case 2b filled `to` with an existing declared state from a quote that names nothing.
3. *Duplicate twin:* the step **skips** a repair that would give the row the same cargo and the same two states as another
   transport (unit-tested; recommended option). It does not catch a near duplicate under a differently named state (case 4 after
   Stage 2), and no live run exercised the skip.

## 10. Recommendation: enable after more evidence

Do not enable now. Reasons: the step fixes what it is shown, but (a) it fired on 2 of 19 live runs and repaired 1; the three
enterobactin defects were demonstrated on replay only; (b) a repair is accepted on a quote that need not support the state
(sections 4 and 9), which can write a biology claim the paper does not make (case 2b); (c) "keeps everything" rests on 19 on runs
of which 17 were no-ops by construction, one noise floor pair per paper and one real repaired run.

What would turn this into an enable: (1) the PO rules on questions 1 and 2 and a reviewer then tightens acceptance so that a quote
must contain the compartment word, or one of the blessed phrases, for the state it justifies, and so that an unnamed side is never
filled from the declared-state list (code change, not made here); (2) one retry on an empty or rowless reply; (3) more draws of
PMC6112128 (the only paper where the defect reproduces) with repaired rows followed through Stage 2, and at least 3 more repaired
runs; (4) PMC12452463 as the fourth control; (5) a flag-on `run_pathway` (audit, curator, mapping, export) run on one affected
paper. Cost of waiting: TubI-type exports keep coming out as `cell` -> `cell`; the three enterobactin forms are rare (about 1
in 6 single-draw runs, none seen in 12 fresh draws, and masked by the 3-draw union merge in paper-run configuration).

## 11. Open questions for the product owner

| question | options | cost |
|---|---|---|
| Q1: may "out of the cell", "secreted", "efflux" (and "imported", "uptake", "into the cell") entail an extracellular side? | (a) yes, bless a short phrase list in code; (b) literal only | (a) a biology rule to bless, fixes TubI; (b) TubI stays `cell` -> `cell`, 1 of 4 stored cases unfixed |
| Q2: an unnamed side, empty or `cell`? | (a) always empty (normalizer places generic `cell`); (b) name `cell` | (a) needs the acceptance code tightened (replay cases 1, 2b broke it); (b) adds a state the paper does not state |
| Q3: duplicate twin? | (a) skip (current, exact-copy only); (b) also skip near duplicates by state compartment; (c) keep both | (b) a few lines more, untested against the writer; (c) two transports for one event |
| Q4 (new): retry on an empty reply? | (a) none (current); (b) one retry | (b) one more call, only on deficient rows; TubI repair rate was about 1 in 2 |
| Q5 (new): is enable-after-more-evidence the bar, or a stricter measure such as the stored-gold scoring protocol? | | each extra paper-run costs about 6 min and paid calls |

## 12. NOT ESTABLISHED

* PMC12452463 (control with a same-topic transport) was not run.
* The flag-on full `run_pathway` path (audit, curator, mapping, PWML export, writer) was never run; this test stops at the
  normalizer. Stage-2 effect beyond one repaired run, and the curator effect, are unknown.
* Whether the writer / PathWhiz import tolerates the near-duplicate transports of case 4; the duplicate-skip was only
  unit-tested.
* Whether the three enterobactin defects recur at the stored 17 percent rate (0 of 12 fresh draws; the trace attributes part of
  the apparent rarity to the 3-draw merge, untested here).
* A statistical no-loss claim: 1 repaired live run, 1 off-off pair per paper, 2 draws per arm for controls.
* `T2PW_EXTRACTION_DRAWS=3` (the paper-run configuration): all live runs used 1 draw.
* Why the C/C2 quarantine-boundary test failed in the shared worktree (passes on isolated trees).
* The model returned `{}` for reasons unknown; the raw replies of the live arms were not stored (only the 3 debug replies).
* The `on_s*` purpose is inferred from the driver, not from a note.
* Run scratch directories (`runs/*/*/scratch`, 20 of them, pipeline tmp copies) are not committed; everything the tables use is at
  run level.
