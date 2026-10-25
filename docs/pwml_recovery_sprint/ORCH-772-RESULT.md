# ORCH-772 — PWML coverage and "Omitted from the PWML" (D-114 §§ 1, 3)

**Status:** implemented and measured; **not merged, not self-approved** (an independent reviewer judges the diff).
Branch `orch772/coverage-omissions`, worktree `.claude/worktrees/orch772`, base **`3ef3fa9c`** (sprint tip; `src/` =
`main` `b16a45e7`). Implementer: a general-purpose agent, unattended. No live LLM call, no pipeline run, no network.

| commit | kind | what |
|---|---|---|
| `d1006c10` | code + tests | `src/t2pw/app/reaction_evidence.py` (coverage, OMITTED section, `KNOWN_ISSUES`, `known_issue_id`); `tests/test_orch772_pwml_coverage.py` (new); pins moved (§ 6) |
| `2d561020` | evidence scripts | `orch768_replay_corpus.py` coverage per leg + aggregates; `orch770_night.sh` task id; `orch770_morning_summary.py` coverage column; `orch772_retrim_reaction_evidence_bundles.py` |
| `a238df69` | evidence script | corpus coverage carries a canonical leg's recorded pre-freeze removals (finding F-1) |
| `fba739cd` | evidence script | corpus report: failure class per low leg, PWML-writing legs apart, omissions by KI |
| `c1216df3` | evidence script | fix: the by-KI tally broke the stats print loop (F-6) |
| `43445214` | tests | pins the three fixture legs' corpus coverage; exercises `print_coverage` |
| docs commit | docs | this file, `evidence/orch772_corpus_coverage.{json,txt}`, `evidence/orch772/`, `evidence/orch772_reaction_evidence_sample.diff`, `evidence/g11/ORCH-772/01-40` |

`src/` diff base..tip: **one file**, `src/t2pw/app/reaction_evidence.py` (+523 / −11). The protected
`src/t2pw/app/streamlit_app.py` is byte-identical (blob `ed5d5536` at both; `evidence/orch772/llm_isolation_grep.txt` § 2).
`reaction_evidence_view.py` is **not touched** (§ 9 Q5). `strict_quarantine.py` and `apply_audit_patch.py` not touched.

---

## 1. Coverage at current `main` (the corpus, `evidence/orch772_corpus_coverage.{json,txt}`, G11 `30`/`31`)

PWML coverage = processes drawn in the PWML ÷ processes in the text record (§ 2). Target 75 % per leg.

| denominator | legs | mean | median | legs ≥ 75 % |
|---|---|---|---|---|
| **91 strict legs with an outcome** (LLM-originating and no-PWML legs = 0 %) — headline | 91 | **38.1 %** | **28.6 %** | **22/91** |
| 91, excluding interactions withheld by design (KI-008) | 91 | 42.8 % | 44.4 % | 29/91 |
| **81 replayed** — headline | 81 | **42.7 %** | **43.8 %** | **22/81** |
| 81, excluding by-design withholdings | 81 | 48.1 % | 50.0 % | 29/81 |
| only the 56 legs that write a PWML — headline (`evidence/orch772/pwml_leg_stats.out.txt`, G11 `40`) | 56 | 61.8 % | 61.8 % | 22/56 (14 at 100 %) |
| 56 PWML legs, excluding by-design | 56 | 69.6 % | 76.0 % | 29/56 |

Pooled over the 81 replayed legs: **344 of 1,081 processes drawn (31.8 %)**. 26 of the 81 replayed legs are at 0 %: the 25 that write
no PWML and one that writes a PWML drawing nothing (F-2).

**Where the 737 missing processes go** (replayed legs, `omitted processes by known issue` line of the `.txt`): lost with a whole leg
that writes no PWML **275**; KI-008 untyped interaction **120**; KI-001 protein without UniProt/DrugBank **118**; KI-011 gene/protein
symbol clash **106**; KI-015 other structural **80**; KI-009 unwrappable interaction endpoint **20**; unclassified **12** (F-3); KI-004
non-protein actor **6**. Listed but not counted: **22 merged duplicates** (KI-014).

**The five lowest legs** (all at 0 %; ties sorted by id; the main cause is the whole-leg failure, then the pre-freeze removals):

| leg | coverage | why no PWML (replay class) | main omission reasons |
|---|---|---|---|
| ORCH-755 enterobactin 2201 | 0/13 | quarantine refusal `unexportable_entity:1` (KI-016; cause KI-017) | 13 processes lost with the leg |
| ORCH-755 sour taste 0139 | 0/14 | Stage-8 `actor_schema_not_canonical` (KI-023) | 7 × KI-001 pre-freeze; 7 lost with the leg |
| ORCH-759 chlorophyll 1231 | 0/97 | required gate `reaction_enzyme_must_be_protein_complex` (KI-006, Stage 6) | 55 lost with the leg; 27 × KI-011 |
| ORCH-759 enterobactin 1037 | 0/11 | final gate, degree-0 protein (KI-013) | 11 lost with the leg |
| ORCH-767 enterobactin 2224 | 0/21 | final gate, invented wrapper stoichiometry (KI-012) | 17 lost with the leg; 4 × KI-001 |

(ORCH-767 Wnt/Ca2+ is the sixth, 0/48, quarantine refusal KI-016/KI-018.) **The five lowest legs that do write a PWML:**
`runs_verify/2026-08-22_2017/PMC12856317` **0/3** (KI-004 cofactor in the modifier slot, KI-005; KI-008; KI-009); ORCH-767 WNT PCP
**1/66** (21 × KI-015 `undeclared_entity_in_entity_1`, 17 × KI-008); `runs_verify/2026-09-06_1425/PMC11172790` **1/9** (5 × KI-001,
2 × KI-008); ORCH-759 WNT PCP **6/42** (17 × KI-001, 8 × KI-008); `runs_verify/2026-08-24_1428/PMC12444477` **5/33** (13 × KI-008,
11 × KI-001).

Base (`3ef3fa9c`, G11 `29`, `evidence/orch772/corpus_base_3ef3fa9c.{json,txt}`): coverage **n/a** (the base tree's
`reaction_evidence` carries no `pwml_coverage`), PWML **56/91**.

## 2. Design

**Denominator** (orchestrator ruling, one constant to change: `COVERAGE_KINDS`, `MERGED_DUPLICATE_KI`): reactions + transports +
interactions of the final payload, **plus** every pre-freeze removal of those kinds:
* the strict quarantine's `quarantined[]` rows, **only when it was applied**: `research_mode.applied = false` and `ok = false` leave
  the rows in the payload read (the driver writes `final_mapped_db`, the pre-quarantine payload, when the boundary refused), so
  counting them again would double-count;
* the payload's `quarantined_locked_reactions` entries strict quarantine does not already list (by lock id, the module's rule);
* a row whose reason labels KI-014 (`duplicate_locked_reaction`) is **listed as "merged duplicate, not counted"**;
* a removed row of another kind (reaction-coupled transport, sub-pathway) is listed, not counted (§ 9 Q2).

**Numerator:** read from the IR, only when a PWML was written (`PWML_FILENAMES`, or the app's export result). The writer draws every
IR reaction and transport row (`writer.py` 1320-1444) and skips each interaction the IR marks `withheld` (1463-1477), so: reactions
and transports are drawn when the IR's rows line up with the payload's (count and names, the same check the module already applies
to interactions); an interaction is drawn when the module's existing per-row verdict says "drawn as". **No PWML → 0 drawn.** An IR
that does not line up, an IR older than per-row verdicts, or no IR beside a written PWML → coverage `Not recorded`, and the rows
concerned are listed as "whether the PWML draws these is not recorded", never as omitted.

**By design:** interactions the IR withholds with KI-008 (`untyped_interaction`) count in the headline; the secondary line leaves them
out. Without a PWML no IR verdict is read, so that line equals the headline.

**Text layout.** Right under the title: `PWML coverage: <drawn> of <total> processes (<pct>) -- target 75%`, a per-kind line, the
excluding-by-design line, the no-PWML cause, the not-counted count, and a pointer to the section. The old `Coverage:` (core-term)
line is relabelled `Core-term coverage:`. The section **`OMITTED FROM THE PWML (n)`** (n = total − drawn) sits after REMOVED
ENTITIES, before CORE COVERAGE (§ 7 dead end 1): each counted omission as a block (`Kind`, `Name`, `Equation`/`Endpoints`/`Cargo`,
`Reason` verbatim, the IR's verdict text as `Verdict` when withheld, ORCH-773's `detail.omission` as `Omission` when recorded,
`Known issue`, `Add by hand`), then compact lists: not drawn because no PWML was written; verdict not recorded; listed, not counted;
entities left out (`removed_entity_report.json`, with KI id, ORCH-773's `detail.gate` and D-113 § 1 actor removals when recorded).

**Known issues.** `KNOWN_ISSUES` is one constant: the appendix's 82 codes (code, match, patterns, KI, "then label by"), its 19
messages, and the verbatim "User workaround" of each of the 33 KI ids the appendix labels with. `known_issue_id`: exact codes
first, then prefix/suffix by longest pattern, then messages by most fixed text (so `PWML export stopped by the quarantine
boundary: …` is KI-016, not KI-028's broad `PWML export ...`); `protein_complex_component_unrepresentable:<name>:<reason>` is
labelled by its inner reason when that maps ("then label by"). Driver `gate.<slug>` codes match by the appendix's prefix/suffix rows;
the full gate messages (what ORCH-773 writes as a row reason) match the message table. `unclassified` otherwise.

**Corpus (b).** `orch768_replay_corpus.py` calls the measured tree's `reaction_evidence.build_report` on each replay's artifacts:
the replayed quarantine set, the canonical payload (or, on a refusal, the payload the quarantine was given), the IR and the PWML name
only when a PWML was written. On a canonical checkpoint the leg's recorded `quarantined[]` rows are carried before the replay's own
(F-1). A tree without the field reports n/a. Per leg: drawn/total/% and the top omission reasons; aggregates on both denominators
and both variants; the lowest legs. Network-free (probe refused 3, every leg 0 attempts) and deterministic (§ 4).

## 3. LLM-bound text is unchanged (D-104 AMENDMENT 2)

* **By construction:** the only `src/` change is `reaction_evidence.py`; it is imported only by `reaction_evidence_view.py`, which only
  the protected app's last line imports; no LLM call site (`curation/`, `llm/`, `rag/`, `pipeline/reaction_summary.py`,
  `pipeline/draft_graph.py`) imports it; the new names occur only in that file (`evidence/orch772/llm_isolation_grep.txt`).
* **By the goldens at tip:** `tests/test_orch759b_recorder_reattach.py` ALONE (15-request LLM-input golden) **7 passed** (G11 `10`,
  again at the final tip `38`); `test_stage_one_paper_dictionary.py::test_what_a_model_reads_is_the_bases_bytes` +
  `tests/test_orch757_llm_token_budgets.py` (every request) **57 passed** (`11`, again in `35`).

## 4. No pipeline verdict or PWML byte moved; the corpus run is deterministic

* `compare` base `3ef3fa9c` → tip (G11 `32`, `evidence/orch772/compare_base_to_tip.txt`): pass rate **56/91 → 56/91**, fixed none,
  passing legs turned failing none, gold recall lost none, moved none, and **replay records differing apart from `pwml_coverage`:
  none (91/91 identical)**. Every verdict, every PWML sha256 and every output-file digest is in those records. `compare` prints
  `D-112 § 2: REJECT` (exit 1) only because that rule asks a fix for a higher pass rate; this lane changes no verdict by design.
* Determinism: tip under `PYTHONHASHSEED=1` and `=2` (G11 `30`, `31`) gives byte-identical JSON (sha256 `e288332d…`) and text
  (`678e31b5…`); base twice (`20`, `29`) identical (`22131022…`). 0 network attempts on all 81 legs.

## 5. Tests (all through `bounded_run.py --heavy-lock ORCH-772`, pinned, fresh `--basetemp`; every report 0 survivors)

| run | G11 | result |
|---|---|---|
| focused at the final tip: `test_orch772_pwml_coverage.py` (new, 30), `test_reaction_evidence.py`, `test_orch768_replay_corpus.py`, `test_orch762_protein_identity.py`, `test_pwml_interaction_protein_wrappers.py`, the two LLM-input goldens | `35` | **175 passed** |
| SMOKE, 22 files | `34` | **511 passed** (expected 511) |
| ALONE `test_reaction_evidence_view.py` | `08`, `36` | 11 passed |
| ALONE `test_reaction_evidence_app_hook.py` | `09`, `37` | 1 passed |
| ALONE `test_orch759b_recorder_reattach.py` | `10`, `38` | 7 passed |

No red anywhere, so no base comparison of a failure was needed; none of the known failures of § ORCH-754/§ ORCH-759 is in these
selections. **G9:** `tests/test_orch772_pwml_coverage.py` is labelled NEW CAPABILITY (no base failure claimed); no correction of
existing behaviour is claimed. Its test 1 is the drift alarm: it parses `PWML_KNOWN_ISSUES.md`'s appendix (codes, messages) and the
workaround cells and fails on any difference with `KNOWN_ISSUES`. `g11_evidence.py check --task ORCH-772`: **40 artifacts, 0
non-compliant**, exit 0; 16 pytest reports, all with a committed pin verdict.

## 6. Pins moved deliberately (merge rule 4)

1. `tests/test_reaction_evidence.py` test 3: `"Coverage: Not recorded"` → `"Core-term coverage: Not recorded"` (the relabel).
2. `docs/reaction_evidence_sample.txt` (test 2's golden), regenerated with the existing
   `evidence/orch754b_regen_reaction_evidence_sample.py` (G11 `04`, `05`). Delta, exactly (`evidence/orch772_reaction_evidence_sample.diff`):
   4 lines added under the title (the coverage block: `Not recorded …; 10 processes`, `reactions 5 of 6; transports 1 of 1;
   interactions Not recorded of 3`, the excluding line, the pointer); `Coverage: 0.75` → `Core-term coverage: 0.75`; the new section
   (1 omitted process, KI-001, and 3 interactions whose verdict the leg's old IR does not record) inserted before CORE COVERAGE.
   Nothing else in the sample moved.
3. The four trimmed bundles `tests/fixtures/reaction_evidence/*.bundle.json` gain, **verbatim from their committed legs, only the keys
   the new section reads** (IR reaction/transport names; transport cargo and states; quarantined interactions' endpoints): 5, 35, 6
   and 15 keys; every other line is byte-identical apart from 19 lines that gained a trailing comma. The script proves each still
   renders exactly like its committed leg (`evidence/orch772_retrim_reaction_evidence_bundles.py`, G11 `03`). Without it test 2 and
   test 13 fail: the trim had dropped what the new section reads (F-4).

No other pin moved. `test_orch768_replay_corpus.py` is unchanged and green.

## 7. Failures, dead ends, and what I believed that was wrong

1. **The section at the top broke the existing first-occurrence helpers.** I first placed OMITTED right after the header.
   `test_reaction_evidence.py::test_payload_quarantined_locked_reactions_are_listed_…` finds a row's block by its first `Name:` line
   and walked back to `QUARANTINED`; the OMITTED block came first (G11 `02`, 6 failed). Moved the section after REMOVED ENTITIES and
   added a pointer line under the coverage headline, rather than moving that test.
2. **Rows whose verdict is not recorded were first called "omitted".** On the sample leg (IR older than per-row verdicts) three
   interactions appeared as `OMITTED 2-4`; that claims they are missing when they may be drawn. They now have their own list.
3. **I believed the replay's quarantine report was the leg's record of removals. Wrong on canonical legs** (F-1): first corpus run
   gave ORCH-767 enterobactin 0/17 against 0/21 in its own leg (G11 `16`, superseded). Fixed in `a238df69`; all corpus figures above
   come from the fixed harness (`29`-`32`). Superseded runs: `15`-`17`, `20`-`23` (correct verdicts, wrong denominators on 32 legs),
   `26`/`27` (F-6).
4. **G11 `01`** first hit the heavy lock held by `ORCH-773-rev` (exit 95, child not started, no report written); the same reserved
   path was used for the retry once the lock was free. The rule says never reuse one; recorded here.
5. An f-string written with a `chr()` workaround in the first draft was replaced before any commit.

## 8. Findings

* **F-1. Re-quarantining a canonical checkpoint erases the record of what was removed.** The replay's quarantine sees a payload the
  leg already quarantined, so its `quarantined[]` holds only rows the measured tree removes on top: 32 of 73 canonical legs had recorded
  rows the replay does not repeat; 2 canonical legs get new rows at `main` (ORCH-767 chlorophyll 5, `2026-08-22_2017/PMC12856317` 1);
  0 canonical replays refuse (`evidence/orch772/qcheck.{py,out.txt}`, G11 `18`). Any future corpus metric built on the replayed
  quarantine report must carry the recorded rows the same way.
* **F-2. At `main` one leg writes a PWML that draws no process**: `runs_verify/2026-08-22_2017/PMC12856317`, 0/3 (its one reaction
  removed, KI-004/KI-005; its two interactions withheld, KI-008 and KI-009). D-113 § 3's reading ("a pathway whose IR draws no process
  is declined, `no_drawable_process`") is not built (KI-021).
* **F-3. Two `dedupe_processes` reasons are not in the catalogue**: `coarse_grained_same_entity_transformation` (11 corpus rows) and
  `output_subset_of_input_without_distinct_product` (1) — no-op reactions removed before Stage 6 (`process_normalizer.py` ~4160). They
  print `unclassified` (`evidence/orch772/unclass.{py,out.txt}`, G11 `33`). Also uncatalogued: `filter_unresolvable_reactions`'
  `unresolved_*_entity` reasons and the removed-entity reasons `unreferenced_after_quarantine` / `degree_zero_after_quarantine`.
* **F-4.** The trimmed Reaction Evidence fixtures are "every rendered value kept"; any new rendered field must re-trim them from the
  committed legs, or tests 2 and 13 fail. The re-trim script is now committed for that.
* **F-5.** The committed sample leg (`runs_verify/2026-09-01_1612/PMC12096016/strict`) has an IR from before per-row interaction
  verdicts; its coverage is honestly `Not recorded`. Older legs like it cannot give a coverage figure without guessing.
* **F-6.** The by-KI tally (a dict) entered the stats print loop and raised `KeyError` after the stats printed (G11 `26`, `27`); fixed in
  `c1216df3` and now exercised by test 11.
* **F-7.** A live leg and the corpus can differ for a legitimate reason: ORCH-767 WNT PCP is 0/66 in its own leg (it crashed, KI-020)
  and 1/66 at `main`, where ORCH-769c lets it export.

## 9. Not settled (for the product owner)

* **Q1. The denominator.** Built: duplicates out (listed), unsigned interactions in (with a secondary line), every other pre-freeze
  removal in. Options: (a) keep; (b) take KI-008 out of the headline too (headline becomes today's secondary: 42.8 % / 29 of 91 legs
  ≥ 75 % instead of 38.1 % / 22); (c) also take out the no-op removals of F-3 (11 + 1 rows) as "not biology". Cost of (b)/(c): one
  constant each (`BY_DESIGN_KI` use in `_omissions`, `MERGED_DUPLICATE_KI`), plus the pins of test 3.
* **Q2. Reaction-coupled transports and sub-pathways** removed before the freeze are listed, not counted (the definition names
  reactions, interactions, transports). Option: count an RCT as a transport. Cost: one constant (`COVERAGE_KINDS`); no corpus leg
  has one.
* **Q3. Catalogue the F-3 reasons** (ORCH-771 follow-up). Until then they print `unclassified` with no hand-workaround line.
* **Q4. The zero-draw PWML (F-2):** build D-113 § 3's `no_drawable_process` decline, or accept a written PWML with 0 % coverage
  (`review_required`). The coverage line now makes such a file visible.
* **Q5. The app view** (`reaction_evidence_view.py`) is untouched: the download carries the new text, but the on-screen metric is still
  labelled "Coverage" (the core-term ratio) and the screen shows neither PWML coverage nor the OMITTED section. Options: (a) leave;
  (b) relabel the metric "Core-term coverage" and add a PWML coverage metric (a small view change and an AppTest re-run). Not done:
  the brief limits view edits to breakage or misreporting, and the metric reports the same ratio as before.
* **Q6. Which number is the 75 % target judged on?** Per leg, on the 91, 81, or only the legs that write a PWML (61.8 % mean,
  22/56 ≥ 75 %)? All are printed.
* **Q7. `compare` for a lane that must not move verdicts** prints REJECT by the D-112 § 2 rule. The new "records differing apart from
  pwml_coverage" line is the measurement such a lane needs; whether `compare` should accept "no verdict moved" for a no-op lane is open.

## 10. Part (c): the ORCH-774 runner

* `evidence/orch770_night.sh <worktree> [task id]`: `TASK=${2:-ORCH-770}` names `runs_papers/<task>`, the G11 task and the gold
  file `evidence/<task lowercased, no dash>_gold_accuracy.txt`; without the argument every path is what it was (`orch770`).
  `bash -n` passes. Not run (no live run in this lane).
* `evidence/orch770_morning_summary.py`: per-paper column **PWML coverage**, from each leg directory through `reaction_evidence`
  (`--src`, default the tree the script lives in; `n/a` on any failure; the summary is still written). Run on the ORCH-767 legs into
  scratch (`C:/t/orch772/ms767`, never the primary's run directory), G11 `12`; output `evidence/orch772/morning_summary_ORCH-767_sample.md`:

| paper | PWML | PWML coverage | release | reactions / interactions | min | code |
|---|---|---|---|---|---|---|
| enterobactin | **no** | 0/21 (0%) | review_required | 5 / 8 | 64.0 | `d6849c9e` |
| chlorophyll | **no** | 0/61 (0%) | review_required | 27 / 5 | 74.2 | `d6849c9e` |
| sour_taste_v2 | **yes** | 7/12 (58%) | review_required | 0 / 0 | 49.0 | `d6849c9e` |
| wnt_pcp | **no** | 0/66 (0%) | review_required | 0 / 18 | 103.0 | `d6849c9e` |
| wnt_ca | **no** | 0/48 (0%) | diagnostic_only | 4 / 33 | 109.2 | `d6849c9e` |
| menaquinone | **yes** | 14/14 (100%) | review_required | 11 / 1 | 55.5 | `d6849c9e` |

(The `reactions / interactions` column is the review index's own count: sour taste's 7 drawn processes are transports.)
A reviewer sample of the full new text on a live leg that writes a PWML with withheld interactions (ORCH-759 WNT PCP, 6/42):
`evidence/orch772/reaction_evidence_ORCH-759_wnt_pcp_sample.txt` (G11 `39`).

## 11. Process

* Every Python job ran through `bounded_run.py --heavy-lock ORCH-772` (40 reports). Free RAM was checked four times (twice before the first job, before the
  first corpus run and before SMOKE): 14.7-15.3 GB free each time, not before every one of the 40 jobs. The `g11_evidence.py next` / `check` calls ran directly, as the brief prescribes. One wait on the lock (held by `ORCH-773-rev`), by a background wait loop.
* **Orphan sweeps** three times (after G11 `06`, after `34`, and after the last job `40`; `Get-CimInstance Win32_Process`, python.exe), and every wrapper report shows `FINAL SURVIVING COUNT 0`: only the two VS Code isort language-server processes
  were present each time; **no process was killed**.
* No `python -` heredoc; helpers are files under `C:/t/orch772/`, copies of the cited ones in `evidence/orch772/`.

## 12. LEDGER row

> **`ORCH-772` · 2026-10-25 — IMPLEMENTED, awaiting review, branch `orch772/coverage-omissions` @ `43445214` (+ docs).** The Reaction
> Evidence `.txt` prints PWML coverage (drawn in the PWML ÷ processes in the text record; drawn read from the IR only when a PWML was
> written) under its title and a section OMITTED FROM THE PWML with each left-out process's verbatim reason, KI id
> (`KNOWN_ISSUES` = the PWML_KNOWN_ISSUES appendix, drift-tested) and hand workaround. Coverage at `main` on the corpus: 91 legs mean
> 38.1 %, median 28.6 %, 22/91 ≥ 75 % (excluding KI-008: 42.8 %, 29/91); 81 replayed 42.7 % / 43.8 % / 22/81; PWML-writing legs 61.8 %.
> Biggest losses: whole legs with no PWML (275 processes), KI-008 (120), KI-001 (118), KI-011 (106). No verdict or PWML byte moved
> (91/91 replay records identical apart from coverage); LLM inputs unchanged (759b golden 7 passed). SMOKE 511. Findings: canonical
> re-quarantine loses recorded removals (handled), one PWML draws 0 processes, two dedupe reasons uncatalogued. Record: `ORCH-772-RESULT.md`.

## 13. TEST_MATRIX delta

* **New focused selection:** `tests/test_orch772_pwml_coverage.py` (30, ~3 s; it replays the three ORCH-768 fixture legs once). Run it
  for any change to `src/t2pw/app/reaction_evidence.py`, to `PWML_KNOWN_ISSUES.md`'s appendix or workaround cells (test 1 fails until
  `KNOWN_ISSUES` is updated with them), or to `evidence/orch768_replay_corpus.py`.
* Add `tests/test_orch772_pwml_coverage.py` to the `reaction_evidence.py` obligation beside `test_orch762_protein_identity.py` and
  `test_pwml_interaction_protein_wrappers.py`.
* **Pins moved:** § 6 (test 3 line; the sample; four fixture bundles gain keys). **SMOKE unchanged at 511** (G11 `34`).
* A new rendered field in the Reaction Evidence text must re-trim the fixture bundles
  (`evidence/orch772_retrim_reaction_evidence_bundles.py` pattern) or tests 2 and 13 of `test_reaction_evidence.py` fail.
