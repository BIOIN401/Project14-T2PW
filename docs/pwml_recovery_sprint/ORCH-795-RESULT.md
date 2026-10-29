# ORCH-795 RESULT: the coverage headline leaves out every interaction PathWhiz cannot carry (D-114 AMENDMENT 2 § 1, ORCH-789 F-1, F-2)

Branch `orch794/fix` off `f3a80e09`. Code commit `483f4098` (shared with ORCH-794). Not merged, not pushed.
**Merge rule 5 is waived by the product owner (D-119 § 4: no reviewer round for ORCH-790..795).**

## 1. Cause (verified)

`app/reaction_evidence.py` `_omissions` (base l.812) counted only `e["ki"] == "KI-008" and e["group"] == "ir"`. An unsigned
interaction quarantined before the IR for its entity keeps its quarantine label (KI-001, KI-015 …), so it stayed in the headline
denominator. The base replay reproduces ORCH-789 § 1 exactly: chlorophyll 43/82, wnt_pcp 5/64 (§ 3 table).

## 2. Change

| file | function | change |
|---|---|---|
| `src/t2pw/app/reaction_evidence.py` | `_entry` | new field `uncarried`: an interaction whose relationship word (`interaction_relationship_word`) has no PWML type (`pwml_interaction_type == ""`). The row is passed in from the quarantine record (`original_process`, else `detail.row`) or from the final payload |
| same | `_omissions` | `by_design_withheld` = counted entries that are `uncarried`, or KI-008 withheld by the IR (as before). The full count beside the headline is unchanged |
| same + `PWML_KNOWN_ISSUES.md` appendix | `KNOWN_ISSUES` | F-2: row `coarse_grained_same_entity_transformation` -> KI-015 (still counted: no metric change, as the brief says) |
| `tests/test_orch795_headline_unsigned.py` | new | 1 G9 + 1 preservation |
| `docs/reaction_evidence_sample.txt` | re-rendered | the `test_reaction_evidence` pin (2 lines, § 4) |

The headline label still reads "… the N interactions PathWhiz cannot carry, KI-008 …". N now includes rows the list labels with
their own omission KI. The string is test-pinned, so it was left as is (§ 6 item 2).

## 3. Before / after on the six ORCH-784/785 legs

From `evidence/orch795_coverage.txt` (G11 `06-coverage-base.json`, `07-coverage-tip.json`). Each leg's own artifact set (`record`)
and the harness `replay_leg` (canonical) agree on every leg.

| leg | base headline | tip headline | full count (both) | excluded: base -> tip |
|---|---|---|---|---|
| chlorophyll (ORCH-785) | 43/82 (52.4%) | **43/68 (63.2%)** | 43/108 (39.8%) | 26 -> 40 |
| wnt_ca (ORCH-785) | 10/39 (25.6%) | **10/31 (32.3%)** | 10/46 (21.7%) | 7 -> 15 |
| sour_taste_v2 (ORCH-785) | 6/10 (60.0%) | 6/10 (60.0%) | 6/10 | 0 -> 0 |
| enterobactin (ORCH-784) | 8/12 (66.7%) | **8/10 (80.0%)** | 8/21 (38.1%) | 9 -> 11 |
| menaquinone (ORCH-784) | 22/25 (88.0%) | **22/24 (91.7%)** | 22/25 | 0 -> 1 |
| wnt_pcp (ORCH-784) | 5/64 (7.8%) | **5/23 (21.7%)** | 5/71 (7.0%) | 7 -> 48 |

The moves are +14, +8, 0, +2, +1, +41: the 66 rows of ORCH-789 § 3, leg for leg. The expected values, wnt_pcp 8% -> 22% and
chlorophyll 52% -> 63%, are met. The orch768 corpus reads the measured tree's `by_design_withheld` (`leg_percent`), so its
headline moves with this change. `test_orch768_replay_corpus.py` and the corpus pins in `test_orch772_pwml_coverage.py` pass at tip.

## 4. Proof and pins

| what | result | G11 report |
|---|---|---|
| G9 `test_orch795_headline_unsigned.py`, base `C:/t/orch794/base` | **1 failed, 1 passed** (G9 red on values; preservation green) | `evidence/g11/ORCH-795/05-g9-base-final.json` |
| same, tip `483f4098` | **2 passed** | `04-g9-tip-final.json` |
| affected tests (shared run, 39 files + 3 AppTests) | see `ORCH-794-RESULT.md` § 3: 967 passed, 1 pre-existing failure | `evidence/g11/ORCH-794/27-`..`30-` |
| `g11_evidence.py check --task ORCH-795` | exit 0, 4 pinned | `08-g11-check.json` |

**Pins moved (merge rule 4):**
- `test_orch772_pwml_coverage.py::test_each_replayed_leg_carries_the_measured_trees_coverage` (the D-112 § 2 corpus check): `by_design_withheld`
  enterobactin 7 -> 9, wnt_pcp 17 -> 51; `drawn` and `total` unchanged; sour taste (7, 12, 0) unchanged.
- same file, `test_no_pwml_written_…`: headline "0 of 6 … the 0" -> "0 of 5 … the 1"; full count "the 0 KI-008 … 0 of 6" -> "the 1 … 0 of 6".
- same file: label `coarse_grained_same_entity_transformation` "unclassified" -> KI-015; appendix 84 -> 86 codes.
- `test_orch772b_no_pwml_cause.py::test_g9_the_morning_summary_…`: "0/21 (0%)" -> "0/12 (0%)"; full count 0/21 unchanged.
- `docs/reaction_evidence_sample.txt` (pin of `test_reaction_evidence.py`): "10 processes … the 0 interactions" -> "8 … the 2"; "the 0
  KI-008 … of 10" -> "the 2 … of 10". Re-rendered by `C:/t/orch794/regen_sample.py` (`03-regen-sample.json`).

## 5. Failures and dead ends

- None in the metric code. The first affected run (`ORCH-794/09-affected-tip-r1`) failed exactly the 4 pins above, as predicted.
  They were moved after that run.

## 6. KI delta, LLM-side, NOT settled

- **KI delta:** `coarse_grained_same_entity_transformation` -> KI-015 (appendix row added). No new code from this task.
- **LLM-side:** none.
- **NOT settled (product owner):**
  1. **F-2 as a merged duplicate.** On sour taste, both same-entity "reactions" are also drawn as transports. Labelling them KI-014
     (listed, not counted) would raise sour taste's headline from 6/10 to 6/8. Cost: a cargo-plus-actor match rule in the
     metric. This lane only adds the row (brief).
  2. **The headline label** still names KI-008, although the excluded rows now include ones listed under KI-001 or KI-015. Options:
     (a) keep; (b) reword it to "unsigned interactions (KI-008 class)". Cost of (b): 4 pinned strings.
  3. **Legs with no PWML now exclude unsigned rows from their headline denominator too.** They score 0% either way (D-114 AMENDMENT
     2 § 2). Only the denominator printed changes.

## 7. Proposed LEDGER row

`| ORCH-795 | D-114 AM2 § 1 headline excludes every unsigned interaction; F-2 catalogue row | orch794/fix @ 483f4098 | built, tested; rule 5 waived (PO) | G9 05 red / 04 green; six legs: chl 52.4->63.2%, wnt_ca 25.6->32.3%, entero 66.7->80.0%, mena 88.0->91.7%, wnt_pcp 7.8->21.7%, sour 60.0% | ORCH-795-RESULT.md |`
