# Test-process evidence for the pathway-mining experiment

Every test and benchmark in this repository runs through the bounded foreground
wrapper (`docs/pwml_recovery_sprint/evidence/bounded_run.py`) with guaranteed
cleanup, and the wrapper's structured report is committed rather than pasted.
This directory holds those reports for the experimental pathway-mining work.

They are **not** filed under `docs/pwml_recovery_sprint/evidence/g11/` on
purpose: that tree is keyed to sprint task ids (`C-xxx`, `D-xxx`, `ORCH-xxx`),
and this is an experimental research feature, not a sprint card. Inventing a task
id to file under would misrepresent what this work is. The process discipline is
the same — bounded, owned, cleanup verified, zero survivors, measured tree
verified, report committed.

| File | |
|---|---|
| `001-smoke-feature-disabled.json` | the production SMOKE gate run on this branch with the feature OFF |
| `001-smoke-feature-disabled.pin.json` | the measured-tree verdict for that run |
| `002-mining-suite.json` | the pathway-mining suite |
| `002-mining-suite.pin.json` | its measured-tree verdict |

## What each report must show, and does

| Field | 001 | 002 |
|---|---|---|
| exit reason | `completed` | `completed` |
| exit code (real) | 0 | 0 |
| **final surviving count** | **0** | **0** |
| cleanup success | `true` | `true` |
| measured tree `violations` | `[]` | `[]` |
| heavy lock | acquired and released (`EXP-MINING`) | acquired and released |
| result | **508 passed** in 44.45 s | **222 passed** in 4.54 s |

508 is the count merge gate 10 expects for SMOKE, unchanged by this branch — the
feature-disabled regression evidence.

Pre-existing processes are reported by the wrapper and were **never** killed.
Cleanup targeted only PIDs the job created; no global `taskkill /IM python.exe`
or `pkill python` was used anywhere.

## Reproducing

Both commands are in [`../testing.md`](../testing.md). `--basetemp` is mandatory
and its parent must already exist.

## Tip measurements

`001` and `002` were taken mid-branch, before the documentation commit's two
report-wording fixes. `003` and `004` re-measure the same two gates at the
branch tip and are the ones to cite.

| File | Gate | Result |
|---|---|---|
| `003-smoke-feature-disabled-tip.json` | production SMOKE, feature OFF | **508 passed** in 45.67 s |
| `004-mining-suite-tip.json` | the pathway-mining suite | **222 passed** in 4.09 s |

Both: exit `completed` / code 0, **final surviving count 0**, cleanup success,
heavy lock `EXP-MINING` acquired and released, measured-tree `violations: []`.

## ORCH-742a: mining stability fixes

Two G9 corrections, committed in `ad4245f5`: the preflight probe budget
(hardcoded `8` → `PROBE_MAX_TOKENS = 64`, override
`PATHWAY_MINING_PROBE_MAX_TOKENS`) and the enforcement of
`max_new_reactions_per_round`. No live call in any run below. Every run: final
surviving count **0**, cleanup success, measured-tree `violations: []`.

| File | Tree / SHA | Selection | Result |
|---|---|---|---|
| `005-mining-suite-untouched-tip` | `C:/t/mining` @ `9c085967` | `tests/test_mining_*.py` | **222 passed** (baseline before any change) |
| `006-g9-base-arm` | `C:/t/mining-base` @ `9c085967` | `test_mining_stability.py` (earlier assertion order) | **3 failed** (behavioural) |
| `007-g9-tip-arm` | `C:/t/mining` working tree, before the commit | `test_mining_stability.py` + `test_mining_budget.py` | **23 passed** |
| `008-g9-base-arm-final` | `C:/t/mining-base` @ `9c085967` | `test_mining_stability.py` exactly as committed | **3 failed** (behavioural) |
| `009-mining-suite-fix-tip` | `C:/t/mining` @ `ad4245f5` | `tests/test_mining_*.py` | **226 passed** |
| `010-smoke-feature-disabled-fix-tip` | `C:/t/mining` @ `ad4245f5` | production SMOKE, `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** |

`008` is the G9 base arm to cite. Its three failures are assertions, not
import errors: the preflight **failed** a scripted reasoning model that needs
more than 8 tokens (`'fail' == 'ok'`); the failure detail named neither the
budget nor `PATHWAY_MINING_PROBE_MAX_TOKENS`; and a run with
`max_new_reactions_per_round=1` went on past round 1 and stopped with
`frontier_exhausted`. `006` is the same file with test 1's assertions in a
different order: it failed 3/3 as well, and is kept because it was run. For
both base arms the test file was copied into the base worktree, run with that
tree's own launcher (`--expect-tree C:/t/mining-base`), then deleted.

`007` measured the uncommitted working tree. `009` and `010` measured the
commit and are the ones to cite for the tip. `005`, `009` and `010` held the
heavy lock `ORCH-742a` (acquired and released). The G9 arms did not take it.

## ORCH-743: pathway-mining iteration 1 (the four pilot-001 blockers)

Base `23e1ae83`. Item commits: `638c8f48` (1, raw completion lineage), `5fd3acad` (2, citation
recovery), `f5e44ec7` (3, search), `6f910144` (4, screen rescue). Every run below ended with
exit reason `completed`, **final surviving count 0**, cleanup success and, where pinned,
measured-tree `violations: []`.

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `011-mining-suite-untouched-tip` | `C:/t/mining` @ `23e1ae83` | `tests/test_mining_*.py`, before any change | **226 passed** (heavy lock `ORCH-743`) |
| `012-g9-base-arm-lineage` | `C:/t/mining-base2` @ `23e1ae83` | item 1 G9 base arm, `test_mining_lineage.py`, run by the orchestrator | **3 failed** (behavioural) |
| `013-item2-replay/` | `C:/t/mining` @ `638c8f48` (+ the item-2 working tree for call 3 and the offline fallback) | item 2 diagnosis: pilot-001's extraction input replayed, 3 live calls, est. $0.018879 | see its [`README.md`](013-item2-replay/README.md) |
| `014-mining-suite-orch743-tip` | `C:/t/mining` @ `6f910144` | `tests/test_mining_*.py` (16 files) | **252 passed** in 6.59 s (heavy lock `ORCH-743`) |
| `015-smoke-feature-disabled-orch743-tip` | `C:/t/mining` @ `6f910144` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** in 52.58 s (heavy lock `ORCH-743`) |
| `016-item3-search-probe/` | `C:/t/mining` @ `6f910144` | live, read-only EuropePMC + NCBI search. No model call and no cache read or write | see below |
| `017-item4-screen-fixture/` | `C:/t/mining` @ `6f910144` | offline: the mining screen over pilot-001's 21 screened papers | see below |

The G9 base arms for items 2, 3 and 4 are run by the orchestrator. This worker cannot
write into the base worktree.

**016, the search probe.** Pilot-001's three round-1 frontier items (12-OPDA, AOS, the
topic) ran through the new formulation and the real `PaperSource.search`. Attempt 1 gave 3
queries, 75 result slots and **66 unique papers**; pilot-001's round 1 gave 30 slots and 12
unique. NCBI contributed 10, 12 and 12 of the 25 results per query. Before, it contributed
none. Query strings: `"12-OPDA" AND "Arabidopsis thaliana"`, and
`"12-OPDA" AND "jasmonate biosynthesis" AND "Arabidopsis thaliana"` for attempt 2.

**017, the screen fixture.** Each paper is screened with the vocabulary its round had.
Included: PMC9947334 (required; rescued, score 4.0), PMID 35404431 (desired; rescued, score
3.0) and PMC13364333 (the production screen's own inclusion). Excluded: PMC12280011
(desired, not reached), both borderline papers, and **all 15 must-exclude papers**.

**Final tip.** Commit `db3a46de` is an item-3 follow-up: an item with no usable term is no
longer searched on the species alone. It came after `014` and `015`, so both gates were
re-measured at the final tip. Cite `018` and `019`.

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `018-mining-suite-orch743-final` | `C:/t/mining` @ `db3a46de` | `tests/test_mining_*.py` (16 files) | **252 passed** in 6.53 s (heavy lock `ORCH-743`) |
| `019-smoke-feature-disabled-orch743-final` | `C:/t/mining` @ `db3a46de` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** in 38.93 s (heavy lock `ORCH-743`) |

**G9 base arms for items 2-4, run by the orchestrator.** Workers could not write into the base
worktree. Each new test file (plus item 4's fixture) was copied into `C:/t/mining-base2` @ `23e1ae83`,
run pinned with that tree's own launcher (`--expect-tree C:/t/mining-base2`), then deleted again. The
base tree's status is empty afterwards. Every failure is an assertion on behaviour, not an import error.

| File | Tree / SHA | What | Result at base |
|---|---|---|---|
| `020-g9-base-item2` | `C:/t/mining-base2` @ `23e1ae83` | the two `test_g9_*` nodes of `test_mining_citation_recovery.py` | **2 failed**: `('', '', '') == ('LOX2', …)`; the LOX2 reaction is never admitted |
| `021-g9-base-item3` | `C:/t/mining-base2` @ `23e1ae83` | the four `test_g9_*` nodes of `test_mining_search.py` | **4 failed**: the decoy is returned for an OR query; `10 == 25`; `{'europepmc'}` only; `1 == 2` attempts |
| `022-g9-base-item4` | `C:/t/mining-base2` @ `23e1ae83` | `test_mining_screen_rescue.py::test_g9_the_mining_screen_includes_primary_atlox2_enzymology` | **1 failed**: `[] == ['PMC:PMC9947334']` |

All three: FINAL SURVIVING COUNT 0, pin `violations: []`. Item 1's base arm is `012`.

## ORCH-743 repair round 1 (six blocking review findings)

Pre-repair tip `16fd8a4b`. Fix commits: `4ec77a7f` (bio B-1, a recovered species is never
species evidence), `d220e6a9` (bio B-2 + review B1, rescue route 1), `e58b0d45` (bio B-3,
rescue route 2 vocabulary and matching), `305ce27c` (review B2, the processed cap),
`7e1925fb` (review B3, a wider window instead of a missing second formulation). No live
model call in this round. Both runs: exit reason `completed`, **final surviving count 0**,
cleanup success, heavy lock `ORCH-743` acquired and released, pin `violations: []`,
wrapper clean at `7e1925fb`.

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `023-mining-suite-orch743-repair1` | `C:/t/mining` @ `7e1925fb` | `tests/test_mining_*.py` (16 files) | **266 passed** in 7.37 s (252 before this round) |
| `024-smoke-feature-disabled-orch743-repair1` | `C:/t/mining` @ `7e1925fb` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** in 51.87 s |

`tests/fixtures/mining/orch743_rescue_negatives.json` is new. It holds the EuropePMC title
and abstract of PMC13181208, PMC13153360 and PMC12977409, the three papers the bio-audit
found rescued live at `16fd8a4b`. The records came from one read-only EuropePMC REST
call per paper. There was no model call and no cache read or write.

The G9 base arms for this round are run by the orchestrator at `16fd8a4b`. The nodes and
their expected failures are in `C:/t/mining-iter/g9_manifest.json`.

**G9 base arms for repair round 1, run by the orchestrator.** Each test file (plus
`orch743_rescue_negatives.json`) was copied into a detached base worktree and run pinned with that
tree's own launcher, then restored. The base trees' status is empty afterwards. Every failure is an
assertion on behaviour.

| File | Tree / SHA | What | Result at base |
|---|---|---|---|
| `025-g9-rep1-bio-b1` | `C:/t/mining-rep-base` @ `16fd8a4b` (pre-repair tip) | 4 `test_mining_citation_recovery.py` nodes (B-1, including C1/C3) | **4 failed**: a species recovered only by the fallback gives `target_species_supported` |
| `026-g9-rep1-rescue` | `C:/t/mining-rep-base` @ `16fd8a4b` | 5 `test_mining_screen_rescue.py` nodes (B-2, review B1, B-3) | **5 failed**: PMC13181208, the title-only paper, PMC12977409 and PMC13468208 are rescued; a quarantined lead's name enters the vocabulary |
| `027-g9-rep1-search` | `C:/t/mining-rep-base` @ `16fd8a4b` | 2 `test_mining_search.py` nodes (review B2, B3) | **2 failed**: `25 == 5` papers processed; `[1] == [1, 2]` attempts |
| `028-g9-rep1-item2-restated` | `C:/t/mining-base2` @ `23e1ae83` | the 2 item-2 G9 nodes as restated after B-1 | **2 failed**: `('', '', '') == ('LOX2', …)`; `'LOX2' in ['', '']` |

All four: FINAL SURVIVING COUNT 0, pin `violations: []`.

## ORCH-743 repair round 2 (bio B-4)

Pre-repair tip `59eb761d`. Fix commit `e21e73f2`: rescue route 1 no longer rescues a paper
whose title names an organism other than the target. That covers a paper where the target
is only the source of a transgene expressed in another plant, e.g. PMC10742046, emmer wheat
overexpressing AtAOS / AtOPR3. There is a new fixture,
`tests/fixtures/mining/orch743_transgene_source.json`, holding the bio re-audit's saved title
and abstract. No live call of any kind in this round. Both runs: exit reason `completed`,
**final surviving count 0**, cleanup success, heavy lock `ORCH-743` acquired and released,
pin `violations: []`, wrapper clean at `e21e73f2`.

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `029-mining-suite-orch743-repair2` | `C:/t/mining` @ `e21e73f2` | `tests/test_mining_*.py` (16 files) | **269 passed** in 7.90 s (266 before this round) |
| `030-smoke-feature-disabled-orch743-repair2` | `C:/t/mining` @ `e21e73f2` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** in 39.71 s |

The G9 base arm is run by the orchestrator at `59eb761d`. The nodes are in
`C:/t/mining-iter/g9_manifest.json` (item `r2-bio-B-4`).

**G9 base arm for repair round 2, run by the orchestrator.**

| File | Tree / SHA | What | Result at base |
|---|---|---|---|
| `031-g9-rep2-bio-b4` | `C:/t/mining-rep-base` @ `59eb761d` (pre-repair-2 tip) | the two `test_g9_b4_*` nodes of `test_mining_screen_rescue.py` (plus `orch743_transgene_source.json`) | **2 failed**: route 1 rescues the emmer-wheat AtAOS/AtOPR3 paper (PMC10742046, score 3.5) and the strengthened PMC11478715 (3.25) |

FINAL SURVIVING COUNT 0, pin `violations: []`; the base tree was restored afterwards and its status is empty.

## ORCH-743 iteration 2 (the pilot-002 audit's five defects)

Base `ecf6b9a0` (the tip pilot-002 ran on). Commits:
- `89b35443`: IT2-1 diagnosis, evidence `032`;
- `0fd12c12`: IT2-1, extraction in passage batches;
- `f2dd1215`: IT2-5, every ladder rung in the lineage;
- `6ca31150`: IT2-4, the grape host;
- `6a879818`: IT2-3, a budget-cut window never exhausts an item;
- `1b86f4af`: IT2-2, NCBI identity from the PMID only.

**`032-it2-extraction-diagnosis/`**: 4 live calls, one provider request each, est.
**$0.013588**. It replays pilot-002's `_0002` input, with identity proven by the request hash.
The hidden reasoning surveys every passage and drafts every claim before it answers. Call 3
spent 16,001 reasoning tokens of a 16,000 budget and produced 0 characters. The `reasoning`
knob is never sent, and `effort: low` did not help at 4,000. Two of the four requests went to a
provider that did not reason and answered nearly empty. See its README.

Both gate runs: exit reason `completed`, **final surviving count 0**, cleanup success, heavy lock
`ORCH-743` acquired and released, pin `violations: []`, wrapper clean at `1b86f4af`.

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `033-mining-suite-orch743-it2` | `C:/t/mining` @ `1b86f4af` | `tests/test_mining_*.py` (17 files) | **281 passed** in 8.49 s (269 before this iteration) |
| `034-smoke-feature-disabled-orch743-it2` | `C:/t/mining` @ `1b86f4af` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset | **508 passed** in 39.04 s |

New fixtures: `tests/fixtures/mining/pilot002_grape_host.json` (IT2-4) and
`tests/fixtures/mining/pubmed_pmid34638573.xml` (IT2-2). The G9 base arms are the
orchestrator's, at `ecf6b9a0`. The nodes are in `C:/t/mining-iter/g9_manifest.json`
(items `it2-1` … `it2-5`).

**G9 base arm for iteration 2, run by the orchestrator.**

| File | Tree / SHA | What | Result at base |
|---|---|---|---|
| `035-g9-it2` | `C:/t/mining-rep-base` @ `ecf6b9a0` (the pilot-002 code) | the 7 `test_g9_it2_*` nodes of `test_mining_iteration2.py` (plus `pubmed_pmid34638573.xml`, `pilot002_grape_host.json`) | **7 failed**: `[12] == [5, 5, 2]`; the reference PMCID PMC411572 is kept; the same article is 2 papers; the foreign full text is used; a budget-cut window exhausts the item; the grape paper is rescued; rung fractions are `[None, None, None]` |

FINAL SURVIVING COUNT 0, pin `violations: []`; the base tree status is empty afterwards.

## ORCH-745 (make pathway mining work)

Four change sets, each implemented by a `general-purpose` agent in its own worktree and reviewed on its actual diff
by `pwml-reviewer` and `pwml-bio-auditor` (change sets B and D also by a red-team `pwml-bio-auditor`):
- A evidence hygiene: F-2, F-4, A4, A7, IT2-N3; repair 1 (A1, A2 …); repair 2 (B1, B2 …);
- C run control: B1/F-7, A5, F-3, B2, F9; repair 1 (F3-ORDER …);
- B species reading: F-5, F-1, F-6, B3, N-5; repair 1 (red-team B1–B9 …); repair 2 (rule 5(a) removed);
- D loss-of-function polarity/binding/extracts; repair 1 (comparative no-effect, polarity attribution).

Every G9 base arm was run by the orchestrator, pinned (`--expect-tree C:/t/mining-rep-base`), in
`C:/t/mining-rep-base` detached at the arm's base SHA; the test file (and any new fixture) was copied in and removed
afterwards, and the base tree's status was empty again after every arm. Every arm: exit reason `nonzero`, every
failure an `AssertionError` (no import/attribute/name errors), **FINAL SURVIVING COUNT 0**, pin `violations: []`.

| File | Base SHA | What | Result |
|---|---|---|---|
| `036-g9-745c` | `f9556288` | CS-C: 12 `test_g9_745c_*` nodes of `test_mining_run_control.py` (+ `pilot003_thin_answers.json`) | **12 failed** |
| `037-g9-745a` | `f9556288` | CS-A: 12 `test_g9_745a_*` nodes (16 cases) of `test_mining_evidence_hygiene.py` (+ `orch745a_pilot_texts.json`) | **16 failed** |
| `038-g9-745b` | `f9556288` | CS-B: 8 `test_g9_745b_*` nodes of `test_mining_species_reading.py` (+ two fixtures) | **8 failed** |
| `039-g9-745c-r1-at-d01e0825` | `d01e0825` | CS-C repair 1: F3-ORDER and B2-RECORD nodes | **2 failed** |
| `040-g9-745c-r1-at-f9556288` | `f9556288` | the same two nodes at the original base | F3-ORDER **passed** (base priority order preserved), B2-RECORD **failed** |
| `041-g9-745a-r1-at-6ceef736` | `6ceef736` | CS-A repair 1: 13 nodes (21 cases) | **21 failed** |
| `042-g9-745b-r1-at-4f8d33a3` | `4f8d33a3` | CS-B repair 1: 71 `test_repair1_745b_*` nodes of `test_mining_species_repair1.py` | **71 failed** |
| `043-g9-745a-r2-at-92ab5be9` | `92ab5be9` | CS-A repair 2: 5 nodes (12 cases) | **12 failed** |
| `044-g9-745b-r2-at-fd53a301` | `fd53a301` | CS-B repair 2: 118 `test_repair2_745b_*` rows of `test_mining_species_repair2.py` | **118 failed** |
| `045-g9-745d-at-fd7c737c` | `fd7c737c` | CS-D: 51 `test_cs_d_745_*` rows of `test_mining_species_cs_d.py` | **51 failed** |
| `046-g9-745d-r1-at-6454ef98` | `6454ef98` | CS-D repair 1: 75 `test_cs_d_r1_745_*` rows of `test_mining_species_cs_d_repair1.py` | **75 failed** |

**Gate runs (committed), both trees pinned, `wrapper vs HEAD: clean`, exit reason `completed`, FINAL SURVIVING COUNT 0.**

| File | Tree / SHA | What | Result |
|---|---|---|---|
| `048-mining-suite-orch745` | `C:/t/mining` @ `3ce64f28` (`experiment/pathway-mining`: change sets A + C merged) | `tests/test_mining_*.py` (19 files) | **377 passed** in 20.39 s (281 before ORCH-745) |
| `049-smoke-feature-disabled-orch745` | `C:/t/mining` @ `3ce64f28` | production SMOKE (22 files), `T2PW_PATHWAY_MINING_ENABLED` unset, heavy lock `ORCH-745` | **508 passed** in 65.30 s |
| `050-mining-suite-orch745-species-branch` | `C:/t/m745-b` @ `686ae134` (`orch745/b-species-reading`: A + C + B + D, **not merged**) | `tests/test_mining_*.py` (25 files) | **856 passed** in 24.40 s |
| `051-smoke-feature-disabled-orch745-species-branch` | `C:/t/m745-b` @ `686ae134` | production SMOKE (22 files), mining off, heavy lock `ORCH-745` | **508 passed** in 43.61 s |

`orch745/` holds the change-set briefs and repair briefs (`briefs/`), the binding F-1 bio adjudication
(`briefs/F1_BIO_ADJUDICATION.md`), every review's findings as returned (`reviews/`), each implementer's G9 manifest,
and the live EuropePMC `core` response shape the orchestrator checked on 2026-10-07 (no LLM call). Change sets B and
D (`orch745/b-species-reading` @ `686ae134`) are **not merged**: their final review left two blocking findings
(RT-D-R2-1 / BIO-D-R2-1 and BIO-D-R2-2, `reviews/csd_r2_final_full.txt`) after the repair-round cap. See
`docs/pwml_recovery_sprint/ORCH-745-RESULT.md` on `sprint/pwml-recovery`.
