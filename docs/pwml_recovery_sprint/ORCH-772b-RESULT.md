# ORCH-772b — a no-PWML leg names its recorded cause; coverage scored by D-114 AMENDMENT 2

**Status:** implemented and measured; **not merged, not self-approved**. Branch `orch772b/no-pwml-cause`, worktree
`.claude/worktrees/orch772b`, base **`00eb12d3`** (sprint tip with ORCH-772 merged). Brief: the orchestrator's ORCH-772b message
(ORCH-772 review r0 notes 1 and 2, `evidence/orch772/REVIEW-r0.md`), then its scope addition for **D-114 AMENDMENT 2**. No live
LLM call, no pipeline run, no network.

| commit | kind | what |
|---|---|---|
| `8bf8c07c` | evidence scripts | `orch772b_no_pwml_cause.py` (new reader), `orch756_review_pack.py`, `orch770_morning_summary.py`, `orch770_night.sh`, `orch768_replay_corpus.py` |
| `31eed6e6` | code + tests | `src/t2pw/app/reaction_evidence.py` (2 hunks: one constant, `pwml_coverage_lines`); `tests/test_orch772b_no_pwml_cause.py` (new); pins moved (§ 6) |
| catalogue commit | docs | `PWML_KNOWN_ISSUES.md`: KI-065 (new, section H), KI-021 status and planned fix |
| record commit | docs | this file, `evidence/orch772b_corpus_coverage.{json,txt}`, `evidence/orch772b/`, `evidence/orch772b_compare_coverage.py`, `evidence/g11/ORCH-772b/` |

`src/` diff base..tip: **one file**, `reaction_evidence.py`. The changes are the new constant `HEADLINE_EXCLUDES_BY_DESIGN` beside
`BY_DESIGN_KI`, and the body of `pwml_coverage_lines`. The OMITTED-section rendering (`_omissions`, `omitted_lines`), which ORCH-773
is extending, is **not touched**. Protected `streamlit_app.py` and `reaction_evidence_view.py`: unchanged.

---

## 1. Coverage at `00eb12d3` under D-114 AMENDMENT 2 (`evidence/orch772b_corpus_coverage.{json,txt}`, G11 `09`/`10`)

**Target line (every strict leg, no PWML = 0 %): 29/91 legs ≥ 75 % headline (KI-008 not counted), 22/91 full count; PWML pass
rate 56/91.**

| view | definition | legs | mean | median | ≥ 75 % |
|---|---|---|---|---|---|
| **91 strict legs** (the target) | **headline, KI-008 not counted** | 91 | **42.8 %** | **44.4 %** | **29/91** |
| 91 strict legs | full count | 91 | 38.1 % | 28.6 % | 22/91 |
| 81 replayed (secondary) | headline | 81 | 48.1 % | 50.0 % | 29/81 |
| 81 replayed | full count | 81 | 42.7 % | 43.8 % | 22/81 |
| 56 that write a PWML (secondary) | headline | 56 | 69.6 % | 76.0 % | 29/56 |
| 56 that write a PWML | full count | 56 | 61.8 % | 61.8 % | 22/56 |

The full-count figures are ORCH-772's, unchanged. Amendment 2 changes which figure is the headline, not the arithmetic.

**Where the missing processes go** (replayed legs). The no-PWML legs' 275 processes now carry their cause's id:

| KI | omissions |
|---|---|
| KI-008 | 120 |
| KI-001 | 118 |
| KI-011 | 106 |
| **KI-006 (whole leg, no PWML)** | **89** |
| **KI-016 (whole leg, no PWML)** | **81** |
| KI-015 | 80 |
| **KI-022 (whole leg)** | **26** |
| **KI-012 (whole leg)** | **24** |
| KI-009 | 20 |
| **KI-015 (whole leg)** | **19** |
| **KI-013 (whole leg)** | **17** |
| **KI-023 (whole leg)** | **16** |
| unclassified | 12 (the F-3 dedupe reasons of ORCH-772) |
| KI-004 | 6 |
| **KI-038 (whole leg)** | **3** |

**Lowest legs, headline.** All no-PWML legs are at 0 %; the first five are:
* ORCH-755 enterobactin 0/13 (KI-016 `unexportable_entity:1`);
* ORCH-755 sour taste 0/14 (KI-023 `actor_schema_not_canonical`; 7 × KI-001);
* ORCH-759 chlorophyll 0/97 (KI-006 `reaction_enzyme_must_be_protein_complex`; 27 × KI-011);
* ORCH-759 enterobactin 0/11 (KI-013 degree-0 protein);
* ORCH-767 enterobactin 0/21 (KI-012 invented wrapper; 4 × KI-001).

Among legs that write a PWML:
* `2026-08-22_2017/PMC12856317` 0/2 (KI-065);
* ORCH-767 WNT PCP 1/49;
* `PMC11172790` 1/7;
* ORCH-759 WNT PCP 6/34;
* `2026-08-24_1428/PMC12444477` 5/20.

## 2. What changed

**Note 1, the recorded cause** (callers only, as instructed):
* `evidence/orch772b_no_pwml_cause.py` (new) reads, from a leg directory, the first record in pipeline order that says the leg
  stopped:
  1. `quarantine_report.json` `ok: false` → its `refusal_reasons`;
  2. `final_stage3_gate_report.json` `ok: false` → each error's `reason`;
  3. `RESULT.txt` `issue codes`;
  4. `RESULT.txt` `message`.

  Report items are joined with `"; "` verbatim and a `RESULT.txt` line is taken as written; nothing is reworded. If none of these
  is recorded it returns `""`, and the module falls back to its own sentence.
* `orch756_review_pack.py` passes it as `build_report(..., no_pwml_cause=...)`, so every live leg's `reaction_evidence.txt` now
  names the cause.
* `orch770_morning_summary.py` passes it as well, and the coverage cell adds `no PWML: <KI>`.
* `orch768_replay_corpus.py` uses `replay_no_pwml_cause()`, the replay's own record of the step that stopped it:
  * the quarantine refusal reasons;
  * the final-gate error reasons;
  * the upstream contract codes;
  * the refinement error;
  * the release of a withheld PWML;
  * for an export failure, the required-gate / IR / Stage-8 codes, else the export error.
* Survey before choosing the join (G11 `01`, `evidence/orch772b/causes.{py,out.txt}`): on all 25 no-PWML replayed legs the joined
  items label the same KI as the first item. The worry was a joined list of exact codes labelling `unclassified`, and none occurs
  on the corpus (§ 8 F-2).

**Note 2:** `orch770_night.sh` calls the morning summary with `--src "$WT/src"`.

**D-114 AMENDMENT 2:**
* § 1, the module: the headline is `PWML coverage: <drawn> of <total − KI-008> processes (…) -- target 75% (D-114 AMENDMENT 2:
  the <n> interactions PathWhiz cannot carry, KI-008, are not counted)`. The next line is always the full count, then the per-kind
  line.
* § 1-2, the corpus: per leg it prints the headline with the full count beside it. The target line is the share ≥ 75 % on all 91
  (both definitions) with the PWML pass rate; the 81 and 56 views are secondary.
* § 1, the morning summary: shows both figures.
* § 3: KI-065 added for the zero-process PWML; KI-021's status is now "product decision D-114 AMENDMENT 2: export" and its planned
  fix "none for a decline" (it read "ORCH-773 (with D-113 § 3)"). The appendix is **not edited**.

## 3. Proof: verdicts, PWML bytes and coverage numbers unchanged; only the no-PWML reason/KI text moved

* `evidence/orch772b_compare_coverage.py`, base `00eb12d3` (its own harness on its own `src`, G11 `08`) → tip (G11 `09`). Output:
  `evidence/orch772b/coverage_compare_base_to_tip.txt`, G11 `11`, exit 0. On **93 legs** the following are identical:
  * every replay record apart from `pwml_coverage` (every verdict, every PWML sha256, every output-file digest);
  * every coverage number (drawn, total, per kind, KI-008 count, merged duplicates, not counted);
  * every omission row except the no-PWML rows.

  On **25 no-PWML legs** the lost processes keep their count (275) and move only from `unclassified` to their cause's id
  (KI-006 4 legs, KI-016 4, KI-022 8, KI-012 2, KI-013 2, KI-015 2, KI-023 2, KI-038 1; 25 legs). "Other differences: none."
* `compare` (G11 `12`): pass rate 56/91 → 56/91; nothing fixed or broken; no gold recall lost; **91/91 replay records identical
  apart from `pwml_coverage`**. It prints `REJECT` only because D-112 § 2 asks for a pass-rate gain.
* The base run reproduces the merged ORCH-772 corpus record byte for byte (sha256 `e288332d…` =
  `evidence/orch772_corpus_coverage.json`).
* Tip under `PYTHONHASHSEED` 1 and 2: identical JSON (`6a893e06…`) and text (`fdb6ec8e…`). 0 network attempts.
* **LLM inputs:** the module change is in the human-only renderer (ORCH-772 isolation unchanged). Goldens at tip:
  `test_orch759b_recorder_reattach.py` ALONE 7 passed (G11 `19`); the paper-dictionary base-bytes test and
  `test_orch757_llm_token_budgets.py` passed inside `20`.

## 4. Before and after, one no-PWML leg (ORCH-767 enterobactin, its own leg directory)

`evidence/orch772b/reaction_evidence_ORCH-767_enterobactin_{BASE_00eb12d3,TIP}.txt` and `.diff`: base and tip review packs on
the live leg (G11 `13`, `14`). The whole diff:

```
< PWML coverage: 0 of 21 processes (0.0%) -- target 75%
< ...Excluding the 0 interactions PathWhiz cannot type (KI-008, withheld by design): 0 of 21 (0.0%)
<   No PWML written: neither pathway.pwml nor pathway.review_required.pwml is in this run
> PWML coverage: 0 of 21 processes (0.0%) -- target 75% (D-114 AMENDMENT 2: the 0 interactions PathWhiz cannot carry, KI-008, are not counted)
>   Full count, the 0 KI-008 interactions included: 0 of 21 (0.0%)
>   No PWML written: Generated protein complex 'enterobactin synthase complex' component 'enterobactin synthase' is missing positive stoichiometry.
< No PWML written: neither pathway.pwml nor pathway.review_required.pwml is in this run (known issue: unclassified). Nothing is drawn; ...
> No PWML written: Generated protein complex '...' component 'enterobactin synthase' is missing positive stoichiometry. (known issue: KI-012). Nothing is drawn; ...
```

The zero-process PWML of KI-065 was checked on the replay output: its `pathway.review_required.pwml` has `<reactions/>`,
`<transports/>`, `<interactions/>`. Its payload has `protein_complexes: []` and the interaction endpoint `ALAS2`, as in the leg's
own `final_mapped.json`.

**ORCH-767 morning-summary sample, regenerated** with the tip scripts into scratch (G11 `15`,
`evidence/orch772b/morning_summary_ORCH-767_sample.md`):

| paper | PWML | PWML coverage | release |
|---|---|---|---|
| enterobactin | no | 0/21 (0%); full count 0/21 (0%); no PWML: KI-012 | review_required |
| chlorophyll | no | 0/61 (0%); full count 0/61 (0%); no PWML: KI-004 | review_required |
| sour_taste_v2 | yes | 7/12 (58%); full count 7/12 (58%) | review_required |
| wnt_pcp | no | 0/66 (0%); full count 0/66 (0%); no PWML: unclassified | review_required |
| wnt_ca | no | 0/48 (0%); full count 0/48 (0%); no PWML: KI-016 | diagnostic_only |
| menaquinone | yes | 14/14 (100%); full count 14/14 (100%) | review_required |

WNT PCP is `unclassified`: its recorded `message` is "the app never rendered the 'refinement_generate_pwml' button …", which no
catalogue entry matches. KI-020's text ("DraftGraph has no reaction nodes …") is only in its DETAIL block (§ 9 Q2).

## 5. Tests (bounded, heavy lock `ORCH-772b`, pinned, fresh `--basetemp`; every report 0 survivors)

| run | G11 | result |
|---|---|---|
| focused at tip (new 772b file, 772 file, `test_reaction_evidence.py`, `test_orch768_replay_corpus.py`, `test_orch762_protein_identity.py`, `test_pwml_interaction_protein_wrappers.py`, the two LLM-input goldens) | `20` | **182 passed** |
| SMOKE, 22 files | `16` | **511 passed** |
| ALONE view / app hook / 759b | `17` / `18` / `19` | 11 / 1 / 7 passed |
| **G9 at base** `00eb12d3`: the base tree's own launcher on a base export plus the new test file only | `06` | **7 failed**, as intended |

**G9:** the five `test_g9_*` tests are regression proofs, and each fails on a **value** at base
(`evidence/orch772b/g9_base_00eb12d3.out.txt`):
* the review-pack text lacks the cause;
* the morning-summary cell is `'0/21 (0%)'`;
* `night.sh` lacks `--src "$WT/src"`;
* the replay's lost rows are `unclassified`, not `KI-012`;
* the headline is `1 of 2`, not `1 of 1`.

The two `test_new_*` tests are new acceptance tests for the reader; at base they fail only because the file is absent. That is
expected for new capability, not claimed as proof.

No known failure was hit. The review pack had no test before; it now has one, run as a subprocess on the committed fixture leg.
`g11_evidence.py check --task ORCH-772b`: **19 artifacts, 0 non-compliant**, exit 0.

## 6. Pins moved deliberately (merge rule 4)

1. `tests/test_orch772_pwml_coverage.py` test 3. Old:
   * line 2: `PWML coverage: 3 of 6 processes (50.0%) -- target 75%`
   * line 3: `By kind, drawn of total …`
   * line 4: `Excluding the 1 interactions …: 3 of 5 (60.0%)`

   New:
   * line 2: `PWML coverage: 3 of 5 processes (60.0%) -- target 75% (D-114 AMENDMENT 2: the 1 interactions PathWhiz cannot carry, KI-008, are not counted)`
   * line 3: `Full count, the 1 KI-008 interactions included: 3 of 6 (50.0%)`
   * line 4: `By kind, drawn of full count …`
2. Same file, test 6: the same two lines for the no-PWML set (`0 of 6`).
3. Same file, test 11: the printed corpus layout. Before, the stats came first. Now the TARGET line comes first, then 6 stats lines
   (91, 81 and 56 legs, each with headline and full count), then the by-KI line.
4. `docs/reaction_evidence_sample.txt` (test 2's golden), regenerated with `orch754b_regen_reaction_evidence_sample.py` (G11 `03`).
   Only its three coverage lines move (`evidence/orch772b/reaction_evidence_sample.diff`).

No other pin moved. `tests/test_orch768_replay_corpus.py` is unchanged and green.

## 7. Failures and dead ends

1. **Base G9 run refused (G11 `05`, exit 98, no test ran).** The worktree's `pinned_pytest.py` refuses any tree but its own. Re-run
   with the base export's own launcher (`06`).
2. **G11 `02` hit the heavy lock** (exit 95, child not started, no report). Its reservation stays in the git-ignored `.staging/`.
   The retry took a fresh path (`03`); the ORCH-772 reuse was not repeated.
3. **The tip of `8bf8c07c` alone does not pass the 772 tests** (by reading the code; no test was run at that commit). The evidence scripts were committed before the module change, so at
   that commit the 772 test pins and the harness print layout disagree. The branch tip `31eed6e6` is consistent; review the two
   commits together.

## 8. Findings

* **F-1.** With the recorded cause, the 275 processes lost with no-PWML legs label as KI-006 (89), KI-016 (81), KI-022 (26),
  KI-012 (24), KI-015 (19), KI-013 (17), KI-023 (16) and KI-038 (3). The largest whole-leg blockers on the corpus are the
  bare-enzyme required gate and quarantine refusals.
* **F-2. Limitation of a joined cause.** The module labels the whole cause string. A cause made of several **exact** codes, e.g.
  the required-gate pair `species_missing_taxonomy, species_missing_classification` (ORCH-759 / 759p menaquinone live legs), labels
  `unclassified`, though each code is KI-019. No corpus leg has such a cause at `main` (survey, G11 `01`), but those two live legs
  do. Fix: label a `"; "`-joined cause by its first item, inside `reaction_evidence.py`, outside this lane's brief (§ 9 Q1).
* **F-3. The no-PWML processes show no per-row hand workaround.** The compact list under "Not drawn because no PWML was written" gives
  no `Add by hand` line; the section intro names the cause and KI once. Adding the cause's workaround there is OMITTED-section
  rendering, which is ORCH-773's (§ 9 Q1).
* **F-4.** The zero-process PWML (KI-065) is on an August Stage-6 payload: its interaction endpoint was never wrapped. Current Stage 6
  would wrap `ALAS2`, so a live run would likely draw the inhibition. This is not measurable offline (KI-058).

## 9. Not settled

* **Q1 (for ORCH-773 or the orchestrator):** two small module changes in OMITTED rendering would close F-2 and F-3:
  1. label a joined cause by its first item;
  2. print the cause's workaround once in the no-PWML intro.

  Cost: a few lines in `omitted_lines` / the cause label, and the sample pin. Not done, because the brief forbids that region of
  the file.
* **Q2: the crash leg's cause.** Read `RESULT.txt`'s DETAIL last line as a fifth source, so WNT PCP's live leg labels KI-020? Cost:
  DETAIL can be a traceback, so pick the last line only, which is one rule more. Today it stays `unclassified` with its verbatim
  message.
* **Q3: `orch755_review_pack.py`** (the ORCH-755 runner's copy) still renders without the cause. It is not on the ORCH-774 path
  (`orch756_paper_runs.sh` calls `orch756_review_pack.py`). Leave it, or align it.
* **Q4:** the morning summary imports `t2pw` and still runs outside `bounded_run` in `orch770_night.sh` (review note 2;
  pre-existing). Wrap it?

## 10. Process

* All Python jobs ran through `bounded_run.py --heavy-lock ORCH-772b` (19 reports, plus the unused `02`). `g11_evidence.py
  next/check` ran directly as prescribed. Free RAM was checked 3 times: 10.8, 11.1 and 11.4 GB, each ≥ 6 GB. It was checked
  before the first job, the corpus runs and SMOKE, but not before every job.
* One wait on the heavy lock (`02`). Orphan sweep at the end: VS Code language servers and a live ORCH-773 wrapper tree, whose
  parent was alive, were present. None was mine, and **no process was killed**.

## 11. LEDGER row

> **`ORCH-772b` · 2026-10-25 — IMPLEMENTED, awaiting review, branch `orch772b/no-pwml-cause` @ `31eed6e6` (+ docs).** ORCH-772
> review notes 1-2 and D-114 AMENDMENT 2:
> * A leg with no PWML now names its recorded cause in the evidence text, review pack, morning summary and corpus. The cause is the
>   quarantine refusal, the final-gate errors, the issue codes, or the message, taken verbatim. The 275 processes lost with
>   no-PWML legs move from `unclassified` to their KI id.
> * `night.sh` passes `--src "$WT/src"` to the morning summary.
> * The headline coverage leaves KI-008 out, with the full count beside it.
>
> At `00eb12d3`: on 91 strict legs the headline mean is 42.8 %, median 44.4 %, 29/91 ≥ 75 %; the full count is 38.1 %, 28.6 %,
> 22/91; PWML pass rate 56/91. Verdicts, PWML hashes and every coverage number are identical on 93 legs (only no-PWML reason/KI
> text moved). SMOKE 511. KI-065 added (the zero-process heme PWML); KI-021 is now "product decision D-114 AMENDMENT 2: export".
> Record: `ORCH-772b-RESULT.md`.

## 12. TEST_MATRIX delta

* **New focused selection:** `tests/test_orch772b_no_pwml_cause.py` (7, ~3 s). It runs the review pack in a subprocess and replays
  one fixture leg. Run it for any change to `orch756_review_pack.py`, `orch770_morning_summary.py`, `orch770_night.sh`,
  `orch768_replay_corpus.py`, `orch772b_no_pwml_cause.py` or `reaction_evidence.pwml_coverage_lines`.
* **Pins moved:** § 6. **SMOKE unchanged at 511** (G11 `16`).
* **A G9 run at a base export must use the base tree's own `pinned_pytest.py`**; the worktree's launcher refuses with exit 98.
