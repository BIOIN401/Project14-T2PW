# C-106 — implementer predictions, recorded BEFORE any measurement in this worktree

Written by the C-106 implementer in `C:/t/c106` @ `c7fb5c5`, before running the census
probe, the c102 focused suite, the mutation harness or the F-152 scenarios. Sprint
practice: predictions first, measurement second, and **the failed prediction stays beside
the correction** rather than being edited out.

The Lead's own predictions are in `orch717_census_predictions.md` and its measurement in
`orch717_census_probe.log`. I have read both. **I am re-deriving every number in my own
worktree anyway** — C-106 § 8 and REV-106 A4 both require the author's numbers, not the
Lead's, and if mine differ, mine win and it is a finding.

---

## 1. Census pins (C-106 § 3.1)

| # | Quantity | Current pin | My prediction | Confidence |
|---|---|---|---|---|
| CP1 | `len(paths)` tracked `quarantine_report.json` (line 375) | `>= 62` | **72** | high — `git ls-files` is deterministic and the worktree shares the branch's commit |
| CP2 | test 10 `legs` (line 397) | `== 62` | **72** | high |
| CP3 | test 13 `checked` (line 511) | `== 62` | **72** | high |
| CP4 | test 10 `withheld` (line 405) | `== 92` | **97** | medium — I am accepting the Lead's probe as a *hypothesis to falsify*, not as a source. Independent risk: my worktree could resolve `pinned_gold_set_path()` differently and change which legs have a gold case at all |
| CP5 | test 13 `with_matched_forbidden` (line 515) | `== 23` | **26** | medium, same caveat |
| CP6 | test 10 `set(affected) - F132_PAPERS` (line 404) | `{"PMC13231680"}` | **unchanged** | medium — this is the one that, if it fires, makes the card larger than a census re-pin |
| CP7 | test 10 `cleared` (line 407) | `[]` | **unchanged, `[]`** | high |

**CP4/CP5 are the two the handoff, F-151 and REV-104 all omit.** If I move only CP2/CP3
the file stays red, and that is the card's headline trap.

**No prediction here licenses relaxing a pin.** All seven stay `==` except CP1, which is a
floor and stays `>=` at a raised value.

## 2. Base behaviour (G9-a)

| # | Claim | My prediction |
|---|---|---|
| B1 | focused `test_c102_coverage_denominator.py` at `c7fb5c5` | **2 failed, 12 passed** — tests 10 and 13, both on `assert 72 == 62` |
| B2 | the same file at my tip | **14 passed** |
| B3 | `c102_mutation_attack.py` at `c7fb5c5` | aborts on `assert code == 0` at line 145 before applying M1. **Zero mutations exercised** |

## 3. Mutation results (C-106 § 3.2)

| # | Claim | My prediction |
|---|---|---|
| M-a | M1–M7 go RED | high confidence — M1..M7 were exercised through the driver at C-102 |
| M-b | **R5 goes RED** | **medium.** R5 has *never* run through this driver. REV-102 ran it by hand and it went **GREEN**; test 4 was then written to bite it. If R5 comes back GREEN it is a finding I report, not a thing I fix |
| M-c | every restore is byte-exact once the saved-bytes restore lands | high |
| M-d | the harness at base corrupts line endings | **79745 bytes / 1673 CRLF on disk → 78077 bytes / 0 CRLF after `write_text(newline="")`**, i.e. I expect to reproduce REV-104's numbers exactly. If the byte count differs the file changed since REV-104 measured and I say so |

## 4. F-152 parse (C-106 § 3.3)

| # | Scenario | Base prediction | Tip prediction |
|---|---|---|---|
| F-a | a **green** file whose output contains the prose `"3 errors"` | base records `errors=3` and **aborts** (exit 1, 1 file reported) | tip records `errors=0`, no abort |
| F-b | a file with a **genuine** `1 failed` | base records `failed=1` | tip **still** records `failed=1` and folds it into the totals — a parse that fixes F-a by counting nothing is a worse defect (REV-106 A8) |
| F-c | the real 22-file selection | `errors=0` on every file, 22 files reported | **unchanged** |

## 5. Gates

| Gate | Prediction |
|---|---|
| focused c102 | 14 passed |
| new `test_c106_mutation_harness_executable.py` | N passed — **I do not yet know N**; it is whatever the file ends up containing and I will not pad it to hit a round SMOKE number |
| SMOKE, 22 files | `473 + 14 + N`. If the measured total differs by any amount I stop and report rather than adjusting |
| gold-readers, 22 files | **456 passed / 8 skipped / exit 0**, unchanged. C-106 is test-and-evidence-only; any move here means I touched something I should not have |

## 6. Things I expect to get wrong

Recorded so that being wrong is visible rather than retrofitted:

1. **N.** I have no basis for predicting my own new file's test count before writing it.
2. **CP4/CP5.** I am reproducing someone else's measurement; the honest confidence is
   "medium", and the reason to run it myself is precisely that the three documents which
   agree with each other on "re-pin to 72" are all wrong.
3. **R5.** See M-b. A GREEN R5 is the single most likely genuine surprise in this card.

---

# OUTCOMES — appended after every measurement, predictions above left untouched

Nothing above this line was edited after the fact. Where I was wrong, the wrong
prediction stays and the correction sits beside it.

## 1. Census pins — all seven correct

| # | Predicted | Measured | |
|---|---|---|---|
| CP1 | 72 | **72** | correct |
| CP2 | 72 | **72** | correct |
| CP3 | 72 | **72** | correct |
| CP4 | 97 | **97** | correct |
| CP5 | 26 | **26** | correct |
| CP6 | unchanged | `{'PMC13231680'}` | correct |
| CP7 | `[]` | `[]` | correct |

Re-derived in my own worktree: `evidence/c106_census_probe.log`,
G11 `C-106/02-census-probe.json`. The attribution is exact — the other thirteen
runs sum to **62 legs / 92 withheld / 23 matched**, and
`runs_verify/2026-08-28_1816` contributes **10 / 5 / 3**, giving 72 / 97 / 26.
So all three moved pins are one named run, and nothing else moved.

## 2. Base behaviour — all three correct

* **B1** base `c7fb5c5` focused c102: **2 failed, 12 passed in 5.22s**, both on
  `assert 72 == 62` (`c106_c102_base_red.log`, G11 `01`).
* **B2** tip: **14 passed** (`c106_c102_tip_green.log`, G11 `03`).
* **B3** the harness aborted on its baseline precondition before any mutation —
  confirmed twice: once implicitly at base, and once deliberately in the
  preservation case (`c106_preservation_red_baseline.log`, G11 `13`).

## 3. Mutations

* **M-a** M1–M7 all RED. Correct.
* **M-b** I gave R5 only **medium** confidence and said a GREEN R5 was the most
  likely genuine surprise in this card. **R5 went RED**, killed by
  `test_4_withheld_terms_remain_in_the_diagnostics` — exactly the test D-083
  follow-on 1 added for it. My caution was unnecessary but it was the right
  caution to record: this was R5's **first ever pass through the driver**.
* **M-c** every restore byte-exact. Correct.
* **M-d — WRONG, AND IT IS A FINDING.** I predicted
  `bytes=78077 crlf=0 bare_lf=1673` after the text-mode write, quoting REV-104
  and F-151, which both record 78077. **Measured: `bytes=78072`.**

  `79745 - 1673 = 78072`. Removing 1673 CR bytes from a 79745-byte file can only
  give 78072, so **REV-104's 78077 is arithmetically impossible** and is
  reproduced verbatim in F-151. Everything else in that row is confirmed exactly:
  on disk `bytes=79745 crlf=1673 bare_lf=0`, and after the write `crlf=0
  bare_lf=1673`. Measured in `c106_d084_probe.log`, G11 `04`, with the mutation
  content held to the IDENTITY so every byte of the delta is line endings alone.

  **The defect REV-104 described is completely real and is unaffected**; only the
  transcribed byte count is off by five. Reported rather than silently corrected,
  per C-106 section 8.

## 4. F-152 — all three correct, and one extra scenario worth having

* **F-a** correct: base records `errors=3` on a GREEN file and **aborts**
  (exit 1, 0 files reported); tip records `errors=0` and does not abort.
* **F-b** correct: a genuine `1 failed` is still `failed=1` at the tip and still
  folds into the totals. The parse did not fix the false positive by counting
  nothing.
* **F-c** correct: the real 22-file selection is still `errors=0` on every file,
  22 files reported, 456 passed / 8 skipped.
* **Not predicted, and worth recording:** I added a third scenario,
  `red_with_errors_prose` — a GENUINE red whose failure message contains
  "3 errors". At base that red is converted into an INFRASTRUCTURE FAILURE and
  the gate stops early instead of counting it; at the tip it folds in correctly.
  F-152 names this case in prose but nobody had measured it.

## 5. Gates

| Gate | Predicted | Measured |
|---|---|---|
| focused c102 | 14 passed | **14 passed** |
| new file | N, unknown in advance | **N = 16 passed in 0.17 s** |
| SMOKE, 22 files | `473 + 14 + N` | **503 passed in 38.46 s** = `473 + 14 + 16` exactly |
| gold-readers, 22 files | 456 / 8 / exit 0 | **456 passed, 8 skipped, exit 0** |
| gold-readers split | 22 files, errors=0 | **22 files, 456/0/8/0 errors**, identical to combined |

I said in section 6 that I had no basis for predicting N. I did not pad the file
to reach a round SMOKE number: 16 is what the guards needed.

## 6. My own process failures, preserved

Three, all mine, none of them in the shipped diff:

1. **The CRLF fixture repeated its marker 40 times.** `apply_mutation` refused it
   with `ValueError: the substitution matched 40 times, not 1` and the focused run
   came back **1 failed, 15 passed** (G11 `09`, superseded by `10`). **The refusal
   was the harness working** — a substitution that matches anything other than
   once is exactly what it must reject — so the fixture was fixed, not the rule.
   The reason is now recorded in `_crlf_fixture`'s docstring.
2. **The own-guard driver checked `git status --porcelain` against HEAD.** All six
   guards had gone RED correctly and the driver still reported FAILED, because
   three of the four files it inspects are C-106's own uncommitted changes, so a
   clean porcelain there was never achievable. The bug was in the check, not the
   guards. Log preserved as
   `c106_own_guard_mutations.attempt1-porcelain-vs-head.log` (G11 `11`), corrected
   in `12`. The right question is byte-identity against a pre-run snapshot, which
   is D-084's own question.
3. **An f-string syntax error** in the first F-152 scenario driver (G11 `07`,
   superseded by `08`). Caught by the wrapper as a real nonzero exit, which is the
   process working.

## 7. One thing I could not do as the card described

C-106 section 3.5 orders `TEST_MATRIX.md` edits at the chunk table and the SMOKE
command block. Both are **above `:477`**, and C-054's own end-of-file note says
nothing may be INSERTED above that line because the file is cited by line address.
Those citations are live: `.claude/agents/pwml-test-runner.md:59`,
`FINDINGS.md:1120-1148`, `evidence/c056c_gate_counts.json` and
`DECISION-BUNDLE-PACK11.md:79` all cite `:213-218`, `:242-252`, `:265`,
`:268-273` or `:477`.

Resolved by making every edit above `:477` **line-neutral** — each rewrites or
extends a line in place, none adds or removes one — and putting the full C-106
record at end-of-file, which is where C-054 put its own for the same reason.
Verified: line 477 still reads what it read before, and the file grew only at the
end.

**Separately, and pre-existing:** `:213-218` currently addresses the bounded-runner
function table and `:242-252` the Chunk-D/Chunk-E paragraphs, **not** the chunk
table and SMOKE command block those citations claim. That drift predates C-106 —
I preserved the line numbers exactly and did not introduce it — but the
`pwml-test-runner` agent is instructed to do stem-exact chunk membership against
`:213-218`, so somebody should route it.
