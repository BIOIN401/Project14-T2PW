# ORCH-758b — an empty chunk no longer fails the paper (defect D3)

**Status:** implemented and gated on the lane branch; **not reviewed, not merged.** An independent
reviewer judges the actual diff next.

| | |
|---|---|
| branch | `orch758b/empty-chunk` (worktree `.claude/worktrees/orch758b`) |
| base | `3582b0d7` (the approved ORCH-757 tip) |
| test-first commit (base code) | `11ace4fc` |
| code commit | `a9179eb8` |
| docs/evidence commit | the commit that adds this file |
| charter | `ORCH-758-RESULT.md` § 4 D3 (and D7), § 6 item 3 option (a); `PRODUCT_CONTRACT.md` § 1 ("an initial empty `{}` extraction" may not end a run); D-104 AMENDMENT 2 |

No LLM call and no pipeline run was made. Every number below names its G11 report under
`evidence/g11/ORCH-758b/` and its log under `evidence/orch758b/`.

---

## 1. The defect

PMC10425814 (199,748 chars) is chunked into 4; chunk 3 is 90 % reference list. On every draw both
calls (`normal`, then `structurally_empty_repair`) returned valid JSON with 0 entities and 0
processes, `finish_reason=stop`. `_run_json_stage` raised "structurally empty payload … on all 2
attempts", `run_ensemble` re-raised when every draw had failed (before it writes its side file), and
`run_stage_one_with_chunking` turned that one chunk into a fatal Stage-1 failure for the paper:
"Chunk 3 failed to produce valid JSON." Two of four WNT legs died this way (`ORCH-758-RESULT.md` § 4
D3, evidence `orch758_stage1_boundaries.txt`).

## 2. The diff

Two files, 138 insertions, 11 deletions. Line numbers are the tip `a9179eb8`.

**`src/t2pw/pipeline/stage_one_ensemble.py`**

- `run_ensemble` (`:1033-1039`): when every draw failed it still raises **the same object** it
  raised before (`failures[-1][1]`), now carrying `ensemble_failures = [(draw, exception), …]` on the
  instance, as `_run_json_stage` already adds `terminal_reason`/`ladder`. Without it a caller sees only
  the last draw's attempts and cannot tell "every draw said nothing" from "draw 1 was cut off at
  `length` and draw 3 said nothing".
- `_failed_draws` (`:1055`): the existing `failed_draws` comprehension, moved out so two callers
  share it. The success path writes the same list as before.
- `record_empty_chunk(failure, chunk_id)` (`:1062`): writes the side-file entry the ensemble skipped
  (`chunk_id`, `empty_chunk: true`, `draws_succeeded: 0`, `failed_draws`, empty `buckets`), so
  `stage1_ensemble_report.json` keeps one entry per chunk, in chunk order. With one draw (no
  ensemble) or no artifact dir it writes nothing. Only the chunked path calls it, and only after it
  has accepted the chunk as empty, so a failure that ends a run writes nothing, as before.

**`src/t2pw/pipeline/pipeline.py`**

- `_said_nothing(entry)` (`:2656`): one attempt-log entry with no `error`, a
  `boundary_diagnostics.finish_reason` other than `length`, and a `raw` that parses to a dict
  `_payload_is_structurally_empty` calls empty.
- `_empty_chunk_attempts(failure)` (`:2671`): for each draw (`ensemble_failures`, or the failure
  itself with one draw), it requires at least one attempt and every attempt `_said_nothing`. It
  returns the attempts with draw tags, the way `run_ensemble` tags them, or `None`. `None` means the
  failure stays fatal.
- `_record_empty_chunk(chunk, failure, attempts)` (`:2685`): builds the record (chunk id, section,
  draws, per attempt `draw`/`attempt`/`note`/`finish_reason`/`raw_chars`, the failure text) and files
  it three ways: a `stage1_empty_chunk` boundary record (`BOUNDARY_STAGE1_EMPTY_CHUNK`, `:2653`;
  outcome `valid_json_zero_processes`), the ensemble side-file entry, and `chunk_entry["empty_chunk"]`.
  It gets its own boundary name, so `bench/metrics.py`'s `stage1_extraction` counting and
  `_pooled_draws`' `stage1_ensemble` reading do not see it.
- `run_stage_one_with_chunking` (`:2787-2825`): on `PipelineFailure` it asks
  `_empty_chunk_attempts`. `None` raises the **unchanged** wrapper (same stage, message, `attempts`
  object, `from failure`). Otherwise the chunk's payload is `{}` and it goes down the normal path:
  raw lock entry, cleaning pass, chunk record, `outputs`. If **every** chunk was empty it raises
  `PipelineFailure(stage="extraction", "Extraction stage returned a structurally empty payload (no
  entities and no processes) on every attempt of every one of the N chunks.")` before any lock
  artifact is written, chained `from` the last chunk's failure, with every chunk's attempts.

Not changed: any prompt or prompt builder, `_run_json_stage`, the ladder, retry counts, budgets,
`chunk_text`, `merge_stage_one_outputs`, cleaning, the unchunked branch, `streamlit_app.py`.

## 3. Every fatal path, kept

| chunk outcome | base | tip | pinned by |
|---|---|---|---|
| invalid JSON on every attempt (with the localized repair also failing) | "Chunk N failed…" | same | `test_still_fatal_when_any_attempt_broke[invalid-json]` |
| `finish_reason=length`, empty text (the D9 empty-length shape) | same | same | `[empty-length]` |
| `finish_reason=length` on valid, empty JSON | same | same, by the explicit `length` guard | `[empty-json-at-length]` |
| attempt 1 empty, attempt 2 invalid. `_run_json_stage` reports this as "structurally empty payload" too (F1) | same | same | `[empty-then-invalid]` |
| N=3: draws 1-2 empty, draw 3 broken in any of the 4 ways | same | same | the same four cases at `3-draws` |
| any non-`PipelineFailure` exception (operation timeout, leg deadline) | propagates | propagates: never caught here | `run_ensemble`'s existing `test_an_unrelated_exception_propagates_immediately` |
| a draw that issued no attempt (budget refusal before attempt 1) | fatal | fatal (`not attempts`) | by construction (no production caller activates a `LegDeadline`, P2-04) |
| every chunk empty | fatal ("Chunk 1 failed…", after chunk 1) | fatal as an empty extraction, after every chunk (§ 7 Q3) | `test_still_fails_when_every_chunk_is_empty`, `test_g9_every_chunk_empty_…` |
| unchunked, empty on every attempt | "Extraction stage returned a structurally empty payload … on all 2 attempts." | same | `test_still_unchanged_unchunked` |

In every "same" row the test asserts `stage == "extraction chunk 2"`, the exact message, that the
cause is the inner `PipelineFailure`, that `failure.attempts is failure.__cause__.attempts`, and
that chunk 3 was never called.

## 4. Proofs

`tests/test_orch758b_empty_chunk.py`, 24 tests. Each one drives the real
`run_stage_one_with_chunking` → `run_extraction_pipeline` → (`run_ensemble`) → `_run_json_stage`
path. Only `chat_detailed` is faked, in `pipeline` and in the client (the localized JSON repair
imports it from there lazily). Each sentence is one chunk (`chunk_word_limit=5`, no overlap), and
the fake answers by the chunk's marker word. The two empty replies differ in bytes, as production's
did. Draw counts 1 and 3 are set inside the tests (`T2PW_EXTRACTION_DRAWS`).

### 4.1 G9: RED at the test-first commit, GREEN at the tip

| run | tree | report | result |
|---|---|---|---|
| RED | `11ace4fc` (base code) | `g11/ORCH-758b/01-g9-red.json`, log `orch758b/01-g9-red.log` | **11 failed, 13 passed** |
| GREEN | `a9179eb8` | `/03-g9-green.json` | **24 passed** |

The 11 RED runs fail for the expected behavioural reasons (log `01`): 9 raise
`PipelineFailure: Chunk {1,2,3} failed to produce valid JSON.` (the empty chunk is placed first,
in the middle and last). The 2 `test_g9_every_chunk_empty_…` cases fail
`assert 'extraction chunk 1' == 'extraction'`. The 13 that pass at base are the `test_still_*`
preservation cases.

- `test_g9_an_empty_chunk_contributes_nothing_and_the_paper_goes_on[1-draw|3-draws]`. Chunk 2 is
  empty on all attempts. At the tip the run returns chunks 1 and 3's reactions. Chunk 2 has
  `output` with no reactions, the notes `retry`, `exhausted` × draws, the draw tags `[1,1,2,2,3,3]`,
  and `empty_chunk` with `chunk_id 2`. There is one `stage1_empty_chunk` boundary. With 3 draws the
  side file has entries `[3, 0, 3]` succeeded, the middle one `chunk_id 2, empty_chunk: true`, and the
  `.txt` reads "0 of 3 succeeded". Provider calls are `(draws, 2·draws, draws)`, exactly the calls
  base would make: no extra retry.
- `test_g9_every_chunk_empty_fails_as_an_empty_extraction_after_every_chunk_was_read`: the tip
  behaviour of the all-empty case (§ 7 Q3).

### 4.2 LLM inputs unchanged (D-104 AMENDMENT 2): new acceptance test, labelled as such

`test_llm_inputs_are_those_of_the_paper_without_the_empty_chunk` runs 6 cases: the empty chunk
first, in the middle or last, × 1 and 3 draws. It compares a paper containing the empty chunk with
the same paper without that chunk's sentence:

- the merged Stage-1 payload is **byte-identical** (`json.dumps`). The Stage-2 prompt, the RAG seed
  and `reaction_summary.txt` (curator, enrichment agent) are all built from this payload. It also
  equals `clean_stage_one(merge_stage_one_outputs(...))` of the reference run's chunk outputs;
- every Stage-1 prompt (system + user) of the other chunks is byte-identical;
- every Stage-2 round-1 prompt of the other chunks, built by `run_stage_two_with_chunking` from the
  chunk records themselves, is byte-identical;
- no Stage-1 or Stage-2 prompt in the run with the empty chunk contains `empty_chunk`,
  `stage1_empty_chunk`, `structurally empty` or `valid_json_zero_processes`.

`test_llm_inputs_the_lock_manifest_gains_nothing_from_the_empty_chunk`: the locked-reaction
manifest, which `audit_json_llm` and `gap_resolver` read into prompts, is byte-identical to one
written from chunks 1 and 3 alone.

These cannot pass on base, where the run dies at the empty chunk (RED above). The base functions
they compare against (`merge_stage_one_outputs`, `clean_stage_one`, `write_stage1_lock_artifacts`,
the prompt builders) are untouched by the diff.

## 5. Gates

| gate | report | result |
|---|---|---|
| 1. G9 RED at test-first `11ace4fc` | `/01` | 11 failed, 13 passed |
| 1. G9 GREEN at tip `a9179eb8` | `/03` | 24 passed |
| 1. G9 on the uncommitted fix (same bytes as `a9179eb8`) | `/02` | 24 passed |
| 2. affected A: 32 non-RAG files (list `orch758b/affected_a.txt`) | `/04` | 685 passed, 4 skipped, **1 failed**: `test_measurement_tree_pin.py::test_missing_or_wrong_pythonpath_is_refused_before_collection` (pre-existing, F5) |
| 2. affected B: 18 RAG files (`affected_b.txt`) | `/05` | 357 passed, 1 skipped |
| 2. `test_streamlit_quarantine_boundary.py` alone (matched the grep; AppTest) | `/11` | 23 passed |
| 2/5. the 2 failing nodes on base `src/` | `/10` | 2 failed, identical messages |
| 3. SMOKE (22 files, `smoke.txt`) | `/06` | **508 passed** |
| 4. `tests/test_eval_v2_*.py` (6 files) | `/07` | 89 passed, 1 xfailed |
| 5. `test_batch_preflight.py` alone | `/08` | 32 passed, 4 skipped, **1 failed**: `test_the_message_names_the_problem_the_interpreter_and_the_cure` ("this project ships a .venv": a worktree has none), the known worktree failure in TEST_MATRIX § ORCH-757; same on base `/10` |
| 5. `test_c055_rag_loop_wiring.py` alone | `/09` | 14 passed |

Every report: `FINAL SURVIVING COUNT : 0`. No `--heavy-lock`, one job at a time.
`g11_evidence.py check --task ORCH-758b`: 11 artifacts, 0 non-compliant, 11 invoke pytest directly,
all 11 with a committed pin verdict, 0 refused, 0 foreign `src`.

The affected set is `grep -l -E "chunk|ensemble|empty_payload|extraction_ladder" tests/*.py` (53
files), minus `conftest.py`, minus `test_c055_rag_loop_wiring.py` (gate 5), minus
`test_streamlit_quarantine_boundary.py` (run alone, `/11`). It includes
`test_stage_one_ensemble.py`, `test_stage_one_paper_dictionary.py`, `test_early_failure_replay.py`,
`test_empty_extraction_payload.py` and `test_orch757_llm_token_budgets.py`. `test_stage_one_ensemble.py::test_every_draw_failing_raises_the_last_failure`
(`excinfo.value is failures[-1]`) passes, so the re-raised object is unchanged. "On base `src/`" means
`git checkout 3582b0d7 -- src/t2pw/pipeline/pipeline.py src/t2pw/pipeline/stage_one_ensemble.py` in
this worktree with the tip's tests, then `git checkout HEAD -- …` (no stash). The batch-A skips are
the path-sensitive corpus skips under a `worktrees` path. Chunk D was not run (the orchestrator runs it).

## 6. Findings, failures and dead ends

- **F1. The failure message is not a discriminator.** A draw whose attempt 1 is valid-empty and
  whose attempt 2 is invalid JSON ends with `saw_empty_payload=True`, so `_run_json_stage` raises
  "returned a structurally empty payload … on all 2 attempts" for a draw that *also* broke. The salvage
  and the repair paths set the same flag. So the judgement reads the attempt log (`error`, `raw`,
  `finish_reason`), not the message. `[empty-then-invalid]` pins this: base and tip both fatal.
- **F2. `run_ensemble` exposes only the last draw.** The re-raised failure's `.attempts` are draw N's
  alone. A chunk judged on them would be accepted as empty when draws 1 to N-1 were cut off at `length`.
  Hence `ensemble_failures`. The 3-draw cases of `test_still_fatal_when_any_attempt_broke` put the
  broken draw *last*, so the empty draws come first and only the per-draw check can reject the chunk.
- **F3. Stage 2 reads the chunk records.** `run_stage_two_with_chunking` round 1 iterates the Stage-1
  `chunk_details` (when there are more than 1) and sends each chunk's `text` with its own `output` as
  the Stage-1 context. An accepted empty chunk therefore reaches Stage 2 round 1 with an empty Stage-1
  slice, the same as any chunk whose Stage-1 output is empty, unless its section relevance is below
  0.3 (`references` 0.1, `acknowledgements` 0.05). Rounds 2 and later already re-chunk the whole paper
  with the full payload, at base too. That Stage-2 call is chunk text, not the record (§ 4.2 proves
  no record string reaches it), but base never made it, because base died. See Q2.
- **F4. Byte identity of the merge was checked, not assumed.** I worried that a `{}` chunk *first*
  would seed the merge's key order. `test_llm_inputs_…[first]` shows the merged payload is
  byte-identical anyway: the final `clean_stage_one` rebuilds the container in fixed order.
- **F5. Two gate failures are environmental, identical on base.**
  `test_measurement_tree_pin.py::test_missing_or_wrong_pythonpath_is_refused_before_collection`
  (`T2PW_FROM_WRONG_TREE not in ['T2PW_UNIMPORTABLE']`) tests the measurement launcher, not pipeline
  code. It fails identically on base `src/` (`/10`). It is not in TEST_MATRIX's known-failures list;
  I did not investigate its environment dependence further. `test_batch_preflight.py`'s `.venv` node is
  the known worktree failure.
- **Dead end, rejected before writing it.** Writing the side-file entry inside `run_ensemble`'s
  all-failed branch would have been one site. But it would also add an entry to *unchunked* 3-draw
  runs that fail, and those must stay unchanged. So `run_ensemble` only attaches the failures, and the
  chunked path writes the entry once it has accepted the chunk.
- **Run `/02` was on the uncommitted fix.** It is labelled `g9-green-wip`. `/03` is the GREEN run on
  the committed tip.

## 7. Not settled: for the product owner

- **Q1. Should a release with an empty chunk be forced to `review_required`?** Not implemented, by
  charter.
  - (a) No (as built). A human sees the empty chunk in `extraction_boundary_report.json`
    (`stage1_empty_chunk`), the chunk record and the side file. Cost: no release-level signal, and a
    chunk holding body text plus references (WNT chunk 3 had 7,355 chars of body: end of the disease
    section, Conclusion, Acknowledgements) is passed over silently if the model wrongly read nothing
    in its body part.
  - (b) Always. Cost: likely every long paper whose reference list fills a chunk becomes
    `review_required`, which dilutes the flag. About 1 h plus review; touches the release-status code.
  - (c) Only when the empty chunk's section is not `references`/`bibliography`/`acknowledgements`.
    Cost: depends on section detection, which did not split WNT chunk 3's body from its reference
    list (one chunk holds both; D3 option (b)). Needs a measurement.
- **Q2. Should Stage 2 skip an empty chunk in round 1?** As built, it reads the chunk against an
  empty Stage-1 slice (F3). Skipping it saves one Stage-2 call per empty chunk per leg and removes the
  chance of Stage-2 additions mined from reference titles. But it changes Stage-2 behaviour (out of
  this lane), and round 2 reads the chunk anyway. About 30 min plus review, in
  `run_stage_two_with_chunking`.
- **Q3. The all-empty chunked case now differs from base in three ways.** (i) Every chunk is read
  before the run fails, so N× base's calls. Base stopped after chunk 1, because one empty chunk could
  not be judged before the others were read. (ii) The stage is `extraction`, not
  `extraction chunk 1`. (iii) The message names the empty extraction, not "failed to produce valid
  JSON". `grep` finds no consumer of the `extraction chunk` stage string or of either message in
  `src/`/`scripts/`. If the product owner wants base's message kept, it is a one-line change.
- **Q4. `budget_exhausted` (latent).** *Settled by the round-0 review and fixed in § 10.* With a live `LegDeadline` (none in production: P2-04), a chunk
  whose rungs 1-2 were valid-empty and whose rung 3 was refused for budget would be accepted as empty.
  § 9's `budget_exhausted` means "another step might have helped". Option: also require
  `failure.terminal_reason != "budget_exhausted"` for each draw. Cost: 2 lines, plus a test. Today it
  is unreachable.
- **Q5. The remaining fatal paths still say "failed to produce valid JSON"** even when the cause was a
  `length` stop. Kept, by charter ("stays fatal exactly as now").

## 8. TEST_MATRIX delta (for the orchestrator to apply; I did not edit it)

Append end-of-file:

```markdown
## ORCH-758b (D3): test obligations — appended end-of-file by rule · 2026-10-21

Lane `orch758b/empty-chunk` (`11ace4fc` test-first, `a9179eb8` code). Reports `evidence/g11/ORCH-758b/01`–`11`.

**SMOKE is unchanged at 508** (`g11/ORCH-758b/06`). No pin moved.

### New focused selection

`tests/test_orch758b_empty_chunk.py` (24, ~2 s): a chunk that returns valid, structurally empty JSON on
every attempt of every draw contributes `{}` and is recorded (`stage1_empty_chunk` boundary,
`chunk_entry["empty_chunk"]`, an ensemble side-file entry when draws > 1); every other chunk failure stays
fatal unchanged; all-empty still fails; unchunked unchanged; LLM inputs byte-identical to the paper
without the empty chunk (D-104 amendment 2). Run it for any change to `run_stage_one_with_chunking`,
`run_ensemble`'s failure path, `_payload_is_structurally_empty`, or the attempt-log keys `raw`, `error`
and `boundary_diagnostics.finish_reason` written by `_run_json_stage`.

### Known failures

Unchanged from ORCH-754's list. In a worktree, `test_measurement_tree_pin.py::test_missing_or_wrong_pythonpath_is_refused_before_collection`
also fails (`T2PW_FROM_WRONG_TREE not in ['T2PW_UNIMPORTABLE']`), identically on base `src/` (`g11/ORCH-758b/10`).
```

## 9. Files changed

Test (`11ace4fc`): `tests/test_orch758b_empty_chunk.py` (new).
Code (`a9179eb8`): `src/t2pw/pipeline/pipeline.py`, `src/t2pw/pipeline/stage_one_ensemble.py`.
Docs/evidence (this commit): this file; `evidence/g11/ORCH-758b/01`–`11` and their pin verdicts under
`evidence/g11/pin/ORCH-758b/`; `evidence/orch758b/` (11 logs, the five file lists, `run.sh`, the bounded
wrapper invocation used for every run).

---

## 10. Repair round 1

The round-0 review verdict was **REJECT**, with one blocking finding (B1) and a small note (note 1).
Same worktree, branch and rules. Only the following changed.

**B1. Two failures that are fatal at base were accepted as empty.** "Everything else stays fatal"
was not met.

- (a) **Budget refusal.** With a live `LegDeadline`, attempt 1 returned valid empty JSON and the
  ladder refused attempt 2 as `budget_exhausted`. The chunk never got its retry, but
  `_empty_chunk_attempts` accepted it. This settles § 7 **Q4** as in scope: the fix is the option
  listed there.
- (b) **`content_filter`.** A valid empty reply with `finish_reason="content_filter"` was accepted.
  The client returns such a reply as a partial answer (`client.chat_detailed` docstring).
- **What I got wrong.** In round 0 I treated `length` as the only finish that could disqualify an
  attempt. I also treated a budget refusal as unreachable because production activates no deadline
  (P2-04). Unreachable today does not mean "stays fatal": the code accepted it.

**Note 1.** In the all-chunks-empty case, round 0 wrote N `stage1_empty_chunk` "accepted" records
and N side-file entries, then failed. That contradicted `record_empty_chunk`'s own docstring.

### The fix (`1c0262e7`; tests first in `bc5579a5`)

- `_said_nothing` (`pipeline.py:2656`) now accepts only `finish_reason == "stop"` (`:2663`). Round 0
  accepted anything except `"length"`.
- `_empty_chunk_attempts` (`:2672`) returns `None` for any draw whose `terminal_reason` is
  `BUDGET_EXHAUSTED` (`:2681`).
- `run_stage_one_with_chunking` now only collects each empty chunk as `(chunk record, failure)`
  inside the loop. It records them (boundary record, chunk record, side-file entry) **after** the
  all-empty check (`:2819-2831`), so a run that fails there records nothing. Consequence: in the
  3-draw side file, the empty chunk's entry now comes after the other chunks' entries, not in chunk
  order. It names its `chunk_id`, and the entries without one are the other chunks in order. The
  `record_empty_chunk` docstring and the G9 test (`[3, 3, 0]`) say so.

### New tests, in `tests/test_orch758b_empty_chunk.py` (now 30)

| test | role | round-0 code `a9179eb8` | base `src/` `3582b0d7` | round-1 tip |
|---|---|---|---|---|
| `test_still_fatal_when_any_attempt_broke[{1,3}-draw(s)-empty-json-content-filter]` | preservation (B1b) | **fail** ("Chunk 3 failed": chunk 2 was accepted) | pass | pass |
| `test_still_fatal_when_the_budget_refused_the_retry[1-draw, 3-draws]` | preservation (B1a). A real `LegDeadline` with a fake clock, priced by `price_llm_call()`. Chunk 2's last draw is admitted for its first call, then the clock moves 20 s and the retry is refused. The test asserts `__cause__.terminal_reason == "budget_exhausted"` | **fail** | pass | pass |
| `test_still_no_accepted_record_when_every_chunk_is_empty[1-draw, 3-draws]` | preservation (note 1): no `stage1_empty_chunk` boundary, no side file | **fail** | pass | pass |
| `test_g9_an_empty_chunk_…[3-draws]` | G9, side-file order updated | **fail** (`[3, 0, 3]`) | fail (raises) | pass |

Each B1 case asserts the base failure exactly: stage `extraction chunk 2`, message "Chunk 2 failed
to produce valid JSON.", `attempts is __cause__.attempts`, cause a `PipelineFailure`, and chunk 3
never called.

### Gates (round 1)

| gate | report | result |
|---|---|---|
| new tests on round-0 code (`bc5579a5` tree) | `/12` | **7 failed**, 23 passed: the 6 new cases above + `test_g9_…[3-draws]` |
| same file on base `src/` `3582b0d7` | `/13` | 11 failed (the same 11 G9/acceptance nodes as `/01`), **19 passed**: every `test_still_*`, including the 6 new |
| G9 file, uncommitted fix (same bytes as `1c0262e7`) | `/14` | 30 passed |
| G9 file at `1c0262e7` | `/15` | **30 passed** |
| SMOKE | `/16` | **508 passed** |
| `test_stage_one_ensemble.py`, `test_empty_extraction_payload.py`, `test_early_failure_replay.py` (`r1focus.txt`) | `/17` | **66 passed** |
| `tests/test_eval_v2_*.py` | `/18` | **89 passed, 1 xfailed** |

Every report shows `FINAL SURVIVING COUNT : 0`. I re-ran only the gates the reviewer named.
The round-0 affected batches (`/04`, `/05`, `/11`) were not repeated.

### TEST_MATRIX delta: supersedes § 8's numbers

In § 8's text, read `(30, ~2 s)` for `(24, ~2 s)` and `Reports evidence/g11/ORCH-758b/01`–`18` for
`01`–`11`. Add one sentence to its description: "only a `stop` finish counts as 'said nothing', a
draw whose retry a leg deadline refused (`budget_exhausted`) stays fatal, and empty chunks are
recorded only once the run is known not to be all-empty." Everything else in § 8 stands. SMOKE is
unchanged at 508 (`/16`).

### Still not settled

Q1, Q2, Q3 and Q5 of § 7 stand as written. Q4 is settled by the review (above).

## 11. Orchestrator record: reviews and merge (appended by the sub-orchestrator, 2026-10-22)

Chartered by the ORCH-756-759 sub-orchestrator from `ORCH-758-RESULT.md` § 4 D3 (blocks PWML production for long
papers; a defect, not a product choice), base `3582b0d7`. Severity standard fixed in the review prompt before round
0: a byte outside scope; **a failure fatal at base no longer fatal** other than the solely-structurally-empty chunk; a
changed stage/message/attempts/cause on a fatal path; other chunks' LLM-bound content changed or chunk metadata
reaching an LLM; a non-behavioural G9; a lane-caused gate failure; a weakened gate or dropped `review_required`.

**Round 0 (`7e8f8f60`): REJECT, one blocking finding.** B1 (item 2): `_said_nothing` rejected only `length`, so a
valid empty reply stopped by `content_filter` was accepted; and `_empty_chunk_attempts` accepted a chunk whose retry
was refused `budget_exhausted` under a live `LegDeadline` (latent today: no production caller activates one).
Reviewer probes: base `ORCH-758b/02-rev-probes-base-unpinned`, tip `14-rev-probes-tip-unpinned` (P3 = B1a, P9 = B1b, P8
= note 1). Note 1: an all-empty failed run left N "accepted" records behind.

**Repair round 1 (`10f0dd27`):** accept only `finish_reason == "stop"`; a `budget_exhausted` draw keeps the chunk
fatal; empty chunks recorded only after the all-empty check (§ 10).

**Round 1: APPROVE WITH NOTES.** Probes `19` 12 passed; G9 tip `20` 30 passed; SMOKE `21` 508; affected `22` 162
passed + 1 xfailed; the new test file on base `src/` (`rev758b-base` `03-rev-r1-tests-on-base-unpinned`) 19
preservation tests pass, the same 11 G9/acceptance tests fail as at round 0. Notes: (1) in 3-draw side files the
empty chunk's entry now comes after the others' (successful entries carry no `chunk_id`); (2) Q1 and Q2 remain for
the product owner. On Q2 the reviewer found Stage 2 round 1 reading the empty chunk is the same exposure as
unchunked runs and rounds 2+; no gate would catch a Stage-2 process inferred from a reference title (its quote is
real paper text); minimal fix if wanted: a guard beside `pipeline.py:481`.

Reviewer probe runs used the bounded wrapper without `pinned_pytest.py` (it refuses test files outside the tree);
probe P0 printed the imported `pipeline.py` path to prove the measured tree.

**Merge:** `--no-ff` into `sprint/pwml-recovery` = **`6b7766ec`**; merged-tip gates `ORCH-757i/34` (Chunk D
187/187), `67` (SMOKE 508), `68`-`74`; pushed with `main` at `f392d447`.
