# ORCH-773 review, round 1 + phase 2 (independent reviewer)

Branch `orch773/omit-per-row`, tip `ab9a157c` (round-1 code `1a0fc159`, merge `fa26fbc2` of sprint `00eb12d3`, phase-2 code `52313f0c`).
Reviewer worktrees (detached, left in place): `.claude/worktrees/rev773r1-tip` @ ab9a157c, `.claude/worktrees/rev773r1-base` @ 00eb12d3.
Scratch: `C:/t/rev773r1/` (scripts `scripts/`, logs `logs/`, replay outputs `out-base`, `out-p1`, `out-tip`, rendered text `txt-tip/`).
Exported trees: `C:/t/rev773r1/p1tree` = `git archive 0971487b src reference/PW000001.pwml data/pathwhiz_id_db.json` (phase-1 tip);
`C:/t/rev773r1/basetree` = `git archive 00eb12d3 src tests pyproject.toml pytest.ini reference/PW000001.pwml data/pathwhiz_id_db.json`
plus the tip `tests/test_orch773_omit_per_row.py` and `tests/fixtures/orch773/` only.
Every Python job ran under `bounded_run.py --heavy-lock ORCH-773-rev1`; every report shows `FINAL SURVIVING COUNT : 0`; RAM was checked
(>= 6 GB free) before each job; the orphan sweep after each job found only the two VS Code isort language servers (PIDs 284424, 275448)
and, while they were live, the wrapped jobs of the ORCH-772b reviewer (owned by its own wrapper); nothing was killed.

## VERDICT: APPROVE WITH NOTES

0 BLOCKING, 15 NOTES. N1 needs a product-owner ruling, but it is not something this lane created (reasons in N1).

## Boundary check

* `git diff --stat 00eb12d3 ab9a157c -- src`: `strict_quarantine.py`, `process_normalizer.py` (+18), `reaction_evidence.py` (+54). Nothing else in `src`.
* `streamlit_app.py`: `git diff --quiet 3ef3fa9c ab9a157c -- src/t2pw/app/streamlit_app.py` exit 0; also exit 0 against `00eb12d3`.
* `process_normalizer.py`: the lane diff against `00eb12d3` is byte-identical to the round-1 diff `0971487b..1a0fc159` (cmp). It adds three
  constants and uses them: `f"{ISOLATED_LOCATED_PROTEIN_MESSAGE} {protein_name}"` and `f"{DEGREE_ZERO_PROTEIN_MESSAGE} {protein_name}"`
  equal the old literals character for character (constant + one space + name); `COMPLEX_NAMED_SOURCE_WRAPPER_REASON` equals the old
  `generation_reason` literal. Gate messages are byte-identical; the corpus final-gate errors are unchanged on every base-passing and
  still-failing leg (122).
* `reaction_evidence.py`: the lane diff against `00eb12d3` is byte-identical to the phase-2 diff `fa26fbc2..52313f0c` (cmp). Hunks:
  two `KNOWN_ISSUES` rows (catalogue), the `actor_removed_from` sub-line prints "catalysts kept" only for D-113 rows, `omitted_actors`
  in `_omissions`, the key carried by `build_report`, `_actor_hand_fix`, and the title plus `OMITTED ACTOR` blocks in `omitted_lines`.
  All serve the granted actor-block rendering (N12 on the additive report key).
* `strict_quarantine.py`: no change after `1a0fc159` (`git diff 1a0fc159 ab9a157c -- src/t2pw/pipeline/` empty).
* Non-code: `ORCH-773-RESULT.md`, two appendix rows in `PWML_KNOWN_ISSUES.md`, evidence. Tests: `test_orch773_omit_per_row.py`,
  `test_orch772_pwml_coverage.py` (2 pins, P2.6), `tests/fixtures/strict_failures/cases.json` (5 cases, R1.6).

## What I ran (my own output; G11 reports under `rev773r1-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-773/`, pins under `g11/pin/ORCH-773/`)

| # | report | what | result |
|---|---|---|---|
| 108 | `108-rev1-corpus-base.json` | ORCH-768 corpus, `--src rev773r1-base/src` (00eb12d3), PATHBANK_DB_* unset | `base.json` sha256 `e288332d...8c2b` = the implementer `evidence/orch773/corpus-p2-base.json` |
| 109 | `109-rev1-corpus-tip.json` | same, `--src rev773r1-tip/src` | `tip.json` sha256 `24b033fd...b8a6` = the implementer `corpus-p2-tip.json` |
| 110 | `110-rev1-corpus-compare.json` | `compare base.json tip.json --target ORCH-767/enterobactin-strict/2026-10-23_2224` | **56/91 -> 72/91**, 16 fixed, passing legs turned failing: none, gold recall lost: none, target FIXED, **D-112 s2: ACCEPT**, exit 0 |
| 111 | `111-rev1-corpus-phase1.json` | same corpus on the phase-1 tip `0971487b` (p1tree) | 72 PWMLs (used for the phase-1 -> tip PWML diff) |
| 112, 123 | `112-rev1-pwml-analysis.json`, `123-rev1-pwml-analysis2.json` | `scripts/analyze.py`: full sha256 of every `*.pwml` each replay wrote (144 tip, 112 base, 144 p1), release status per leg, XML scan of every PWML (123 = same with the transporter check fixed to accept `protein-id`) | **base -> tip: 56/56 base-passing legs identical PWML set (names and bytes)**, 0 stopped writing, 16 newly writing. **p1 -> tip: 69/72 identical; the 3 that differ are exactly 755 sour taste, 767 Wnt/Ca2+, rv 09-02_2052 PMC12452463.** Reactions: 734 at tip, **spontaneous=true 0** (also 0 at base and p1); reaction multiset (spontaneous, #enzymes, left, right) **unchanged on 56/56 base->tip and 72/72 p1->tip**; dangling enzyme refs 0, dangling transporter refs 0 |
| 113 | `113-rev1-noenzyme-probe.json` | `scripts/noenzyme.py`: where the tip enzyme-less reactions come from | N1 |
| 114 | `114-rev1-actor-text-check.json` | `scripts/check_text.py`: tip `reaction_evidence.build_report` + `render_text` on every tip leg replay artifacts; per dropped actor: kept process in the frozen canonical, no remaining actor resolving to the dropped one, remaining transporters declared, block content | **4 legs, 7 actor removals, 7/7 OMITTED ACTOR blocks, problems 0** (texts in `C:/t/rev773r1/txt-tip/`) |
| 115 | `115-rev1-live-cap-price.json` | `scripts/live_cap.py`: the legs own recorded (live) quarantine reports | N2 |
| 116 | `116-rev1-g9-base.json` (+pin) | `test_orch773_omit_per_row.py` on basetree (00eb12d3 + new test/fixtures), `--expect-tree`, pin `t2pw_file` = basetree, 0 violations | **13 failed, 6 passed**, every failure on a value or labelled NEW (below) |
| 117 | `117-rev1-focused-tip.json` (+pin) | `test_orch773_omit_per_row.py test_orch772_pwml_coverage.py test_reaction_evidence.py test_strict_failure_replay.py` at tip | **111 passed, 8 skipped** (the 8 known data-driven skips of `test_strict_failure_replay.py`) |
| 118 | `118-rev1-smoke-tip.json` (+pin) | SMOKE, the 22 files of the TEST_MATRIX Commands block | **511 passed** in 44.7 s |
| 119 | `119-rev1-alone-streamlit-qb.json` (+pin) | `test_streamlit_quarantine_boundary.py` ALONE, `T2PW_OFFLINE_CURATOR=1` | **24 passed** |
| 120 | `120-rev1-alone-re-view.json` (+pin) | `test_reaction_evidence_view.py` ALONE | **11 passed** |
| 121 | `121-rev1-base-on-tip-frozen.json` | `scripts/base_on_tip_frozen.py`: BASE (00eb12d3, policy 2026-07-31.1) `quarantine_and_close` and BASE gates (final, Stage 8, required-field, prefreeze + IR) on each of the 16 TIP frozen payloads, context from the leg own record, mode pathwhiz | **16/16: base quarantine ok, nothing quarantined or removed, payload idempotent, base gates ALL PASS** (includes the 3 transporter legs with the kept, transporter-less transports) |
| 122 | `122-rev1-record-diff.json` | `scripts/record_diff.py`: leaf diff of every corpus record base -> tip, ignoring file digests and `pwml_coverage` | **56 base-passing legs and 9 replayed still-failing legs: the ONLY differing leaf is `decision_input_hash_matches_leg` True -> False** (the policy-version bump). Gate verdicts, export codes, release and PWML names identical |

Logs: `C:/t/rev773r1/logs/<seq>-*.log`. `g11next` (the `g11_evidence.py next` reservations) also ran under the wrapper with the heavy lock.

### G9 (merge rule 9), reproduced

Base `116` failures, read from the log: `test_g9_unexportable_component_*` and `*_kept_only_by_a_synonym_*` on `['quarantine:unexportable_entity:1'] == []`;
`*_degree_zero_protein_*` on `['final:Protein has degree 0 ...'] == []`; the two wrapper tests on `['final:Generated ... stoichiometry.'] == []`;
`*_stage8_non_canonical_transporter_is_dropped_*` on `['stage8:actor_schema_not_canonical'] == []`; `*_bare_protein_enzyme_*` on
`['required:reaction_enzyme_must_be_protein_complex']`; `*_ir_complex_without_components_*` on `['ir:protein_complex_missing_components']`;
the new round-1 `test_g9_a_process_quarantined_at_admission_caps_a_release_ready_pathway` on `'release_ready' == 'review_required'`.
Labelled NEW and failing at base by construction: `test_new_release_cap` (in-test import of `OMITTED_FROM_PWML_REASON`) and the 3
`test_new_text_record_*` cases (title `OMITTED FROM THE PWML (1)` / `(7)` / `(7)`, no actor count). 6 preservation tests pass at base.
Tip `117`: all pass. Behavioural, not symbol absence (the one import failure is the test labelled NEW).

## The charge, item by item

**1. D-114 AMENDMENT 1.**
* Transport with a failing transporter keeps the process (`strict_quarantine.py:3010-3040`, gate-error path; `:2978-2984`, entity path).
  The group rule takes the transporter branch only when EVERY error of the first refusing gate naming that transport is
  `transporters/<j>` with an index (`:3010-3012`); anything else falls to `omit(...)` whole (`:3041-3042`), and a later round that
  refuses the transport for another reason omits it whole by the ordinary rule. Measured: 6 transports kept in 3 legs, each lost exactly
  its one failing transporter (114: kept in the frozen canonical 6/6, remaining transporters `[]`, the dropped row `detail.actor` names
  the same actor as the row 6/6). In the PWML the 3 legs gain exactly 3 / 2 / 1 transports, all with `<transport-transporters/>`
  (112 C3), and no other structural change (reaction multiset identical p1 -> tip). Recorded: a `removed_entities[]` row per dropped
  transporter (`bucket transporters`, the gate exact code, `detail.actor_removed_from` with `decision: D-114 AMENDMENT 1 s2`).
  Release capped: all 3 legs `review_required` with `omitted_from_pwml:0_processes+3_actors`, `26_processes+2_actors`,
  `0_processes+1_actors` (112 B).
* Reaction with a failing enzyme omitted whole: in `drop_entity` the only reaction-keeping branch is D-113 (`:2967-2977`), which needs
  an extracted enzyme kept; any other enzyme hit goes to `omit` (`:2985-2986`); the gate-error path never keeps a reaction (`:3010` is
  transports only). Measured: no exported reaction lost an enzyme (reaction multiset unchanged, 112 C2); 755 enterobactin
  `EntB isochorismatase reaction` still omitted (gold 4/5, PO-accepted).
* Interaction with a bad endpoint omitted whole: no branch keeps an interaction; R1.5 counts (6 interactions via an endpoint) match
  the tip records.
* Transport failing for another reason: `:3041` / `:2985`; none in the corpus (R1.3 case iii), shown by code only.

**2. Release cap** (`:3446-3455`, applied through `_cap_release_for_omissions` at `:3709`, read by `batch.driver._frozen_release_record`
for the PWML name). Counts every `QUARANTINE_STATES` admission (admission-time included) plus this pass actor drops; skipped when the
graph is refused and in research mode. Admission never drops an actor while keeping the process (`_actor_verdict:1355-1405`
quarantines the process), so the actor count is complete. Measured (112 B): the only base `release_ready` record (ORCH-759p
menaquinone, entity-only removal, F-8) stays `release_ready`; 4 records move `diagnostic_only -> review_required`, all fixed legs;
17 records carry `omitted_from_pwml:` (15 fixed + 2 base-passing already `review_required`, PWML bytes and names unchanged).
Entity-only removal does not cap: `test_preservation_an_entity_only_removal_does_not_cap_the_release` passes and 759p menaquinone
holds in the corpus. `cases.json` moves exactly as RESULT R1.6 states (diff `0971487b..1a0fc159`: 4 cases `release_ready -> review_required`,
eligible true -> false, reason `omitted_from_pwml:1_processes+0_actors`; `only_unrelated_reactions_survive` gains
`omitted_from_pwml:2_processes+0_actors`), and `test_strict_failure_replay.py` passes at tip (117). Live effect: N2.

**3. Round-0 notes N1, N4, N6.** N1: `_COMPONENT_OMITTED_PREFIX` (`:2605`, used `:2950`). N4: the three constants live in the gate
module and the quarantine imports them (`:81-84`, `:2631`, `:2855`). N6: the rollback comment no longer cites D-113 s3 (`:2582-2585`).

**4. Every dropped transporter and the D-113 wrapper in OMITTED FROM THE PWML.** 114: 7/7 actor removals have an `OMITTED ACTOR`
block with Process, Actor, Slot, the exact Reason, Gate, Decision, Known issue (KI-023 x6, KI-012 x1, none `unclassified`) and an
Add-by-hand line; D-113 shows `Catalysts kept: EntF complex` and the reaction enzymes are now `[EntF complex, EntE complex]`. Title
carries "; k actors left out of processes still drawn"; none of the 7 processes is in `report.omitted` (coverage counts them drawn).

**5. KI rows.** `protein_complex_component_omitted:` (prefix, label by inner reason, else KI-015) and "Located protein is isolated in
connectivity graph" (KI-013), in the appendix and in `KNOWN_ISSUES`, pinned by the 83/20 count (passes at tip, 117).

## Severity standard, item by item

1. Gate relaxed / something exported a gate would reject: **NO.** No gate predicate changed (constants only, messages byte-identical);
   122: every base-passing and still-failing record has identical gate verdicts; 121: the base quarantine and base gates accept all 16
   tip frozen payloads unchanged. PathWhiz contract: the kept transports carry no transporter, which `transport.rb:29-33` allows. Two
   pre-existing PathWhiz exposures ride along in the newly exported legs and are NOT created by this lane: N1 (enzyme-less reactions)
   and N6 (transporters written as `protein-id`).
2. Omission not recorded / not visible: **NO** (114 for actors; round 0 for processes and entities; N4 is a theoretical stale case).
3. Passing leg turned failing: **NO** (110).
4. Gold recall lost: **NO** (110; 767 enterobactin PWML 5/5; 755 enterobactin 4/5 accepted).
5. Whole-pathway refusal silently replaced: **NO** (122: the 9 replayed still-failing legs are record-identical but for the version bit).
6. Pre-freeze LLM input / omission text in a prompt: **NO.** The pass is unchanged in where it runs (round 0); `reaction_evidence` is
   imported only by `reaction_evidence_view.py` (grep), which renders and downloads text; no prompt builder reads it.
7. Unexplained byte change in a base-passing PWML: **NO** (112: 56/56 names and bytes).
8. Omission rule wider than the gate predicate on a realistic payload: **NO** (the new branch is narrower than phase 1: it keeps a process
   phase 1 omitted).
9. Post-freeze repair / out of boundary / protected file: **NO** (boundary above).
10. G9 without base-red proof: **NO** (116 vs 117).
11. Reaction exported with no enzyme and `spontaneous=false`: **not caused by this lane** (N1). The lane removes no enzyme from any
    exported reaction (112 C2); the 6 enzyme-less reactions in 2 fixed legs had no enzyme in the leg own input (113).
12. Reaction newly marked spontaneous: **NO** (0 `spontaneous=true` at base, p1 and tip; the IR forces false, `ir.py:2064-2065`).
13. Transport exported with an invalid transporter: **NO** for the kept transports (0 transporters left, 114); all other transporter
    references resolve to a declared protein (123); see N6 for the pre-existing `protein-id` form.
14. Dropped transporter missing from the text: **NO** (114: 6/6 transporters, plus the D-113 wrapper).

## Findings

**N1. NOTE (needs a PO ruling; read against severity items 1 and 11). Enzyme-less reactions are exported with `spontaneous=false`, and
two of the fixed legs add six of them.** 112/113: the tip PWMLs hold 20 such reactions (40 counting both PWML copies), PathWhiz
`reaction.rb:93-94` `check_spontaneous` rejects each one, and the upload still succeeds without it (ORCH-761 table, line 168).
14 of the 20 are on 7 base-passing legs and are byte-identical at base (e.g. ORCH-767 chlorophyll 6, ORCH-756 enterobactin
`LDH-catalyzed reduction of pyruvate to lactate`). The other 6 are on 2 fixed legs that write a PWML only at tip:
rv 08-18_1328 PMC12452463 `enterobactin iron binding`, and rv 09-06_1425 PMC12376012 `Ceramide synthase reaction`,
`Sphingomyelin synthase reaction`, `Glucosylceramide synthase reaction`, `Sphingomyelin hydrolysis`, `Glucosylceramide hydrolysis`.
Each of the 6 has `enzymes: []` in the leg own `final_mapped.json` input, is admitted `core_accepted` by the base rules, and sits
unchanged in the base frozen canonical (113). This lane strips no enzyme: the reaction multiset is unchanged base -> tip (56/56) and
p1 -> tip (72/72), and the D-113 branch keeps a reaction only while an extracted enzyme stays. So AMENDMENT 1 s1 holds, and so does
item 11 as the PO charged it (a reaction kept without its failing enzyme). The cause is pre-existing: no T2PW gate requires a reaction
enzyme, and the IR forces `spontaneous=false` (`ir.py:2064-2065`, `process_normalizer.force_reactions_non_spontaneous`). Read
literally and apart from the diff, items 1 (second clause) and 11 would make these 6 blocking; I do not, because base already exports
the same kind of reaction on base-passing legs and the lane did not create any of them.
Cost: coverage overstates what PathWhiz keeps. rv 09-06_1425 PMC12376012 is the one fixed leg at or above 75% (88.2%, 15/17); after
PathWhiz drops the 5 reactions it would import 10/17 (59%).
Options for the PO: (a) accept and catalogue it as a KI, with the coverage caveat; (b) a new per-row predicate "reaction with no enzyme"
that omits such reactions and lists them (a new rule, so a PO decision; it would also change 7 base-passing legs if applied to all of
them); (c) model spontaneity upstream. My view: (a) now, plus a ticket for (b) or (c) before coverage is reported as import-true.

**N2. NOTE. The corpus cannot see the live effect of the release cap on admission-time quarantines.** 115: in the legs own
recorded (live) quarantine reports, 2 `release_ready` records carry admission-time quarantined processes: ORCH-759p menaquinone
(2 x `quarantined_unmapped_entity`; its PWML is already `pathway.review_required.pwml` through the driver prefreeze cap) and
rv 08-21_2239 PMC12452463 (2 x `quarantined_broken_reference`; the live leg wrote a bare `strict/pathway.pwml`). Both are canonical
checkpoints, so the replay starts from the reduced payload, its own report shows no quarantined process, and the replay classifies
rv 08-21_2239 PMC12452463 `review_required` at base already. A live re-run of that payload at tip would be capped and renamed
`pathway.review_required.pwml`. This is the intended effect of the orchestrator reading (charge item 2), so it is not a finding
against the code. But RESULT R1.4 ("No release_ready record moved") holds for the corpus only, and section 12 item 10 ("none in the
corpus") is contradicted by the live records. Worth a sentence in RESULT and the LEDGER.

**N3. NOTE (R1.7 item 1). The same biology is treated two ways, depending on where the failure is caught.** A transporter with no
identity caught at admission (`_actor_verdict`, `QUARANTINED_UNMAPPED_ENTITY`) still costs the whole transport (4 transports in 3
legs, 76). The same transporter caught by the per-row pass (`_unexportable_entities`, entity path `:2978-2984`) costs only the
transporter. My view: option (b), applying AMENDMENT 1 s2 at admission, is the consistent reading. It changes a base-passing PWML
(ORCH-767 sour taste v2) and can only be measured live, so it needs the PO ruling the implementer asked for. Not resolving it here is
correct.

**N4. NOTE (theoretical). An OMITTED ACTOR block can say "still drawn" about a process that is not.** `_omissions` builds
`omitted_actors` from every `removed_entities[]` row with `actor_removed_from` (`reaction_evidence.py`, phase-2 hunk) and never checks
that the process survived. Two cases would give the same process an OMITTED process block and a "still drawn" actor block:
(i) a transporter dropped in round k, then the transport omitted whole in a later round for another reason; (ii) in one round, the
`by_record` transporter drop (`:3010-3040`) followed by `drop_entity` omitting the same transport through an element. No corpus leg has
either case (114: 7/7 processes kept, none in `report.omitted`). Suggested guard: render the block only when the process is in the
frozen payload, or have `_apply_per_row_failures` withdraw the actor row when it later omits that process.

**N5. NOTE (theoretical). The transporter drop is by position, with no name check.** `position` and `name` come from the freeze
candidate row (`_row_failures:2775-2783`), and the drop removes `actors[position]` of the `working` row (`:3015-3038`). These line up
because the candidate is a deep copy of `working` with quarantined rows dropped (`_freeze_candidate:2646-2649`) and nothing reorders
transporters. Measured: `detail.actor` matches the row name 6/6 (114). A one-line guard (`_actor_name(actors[position]) == f["name"]`)
would make it fail closed.

**N6. NOTE (pre-existing, outside the lane). Transporters are written as `<protein-id>`.** PathWhiz `transport_transporter.rb:14-16`
validates `protein_complex_id` presence, and `pwml_parser.rb:532` maps `protein-id` to a protein. Base-passing legs carry 48 such
references; the fixed legs add more (e.g. rv 08-18_1328 PMC12452463 transports 24 and 28). KI line 113 records "the contract accepts a
bare transporter". This lane does not touch the writer, and the 6 kept transports have no transporter. Flagged so that ORCH-761
confirms how PathWhiz imports a bare-protein transporter.

**N7. NOTE. Canonical-checkpoint replays undercount the process half of the cap reason.** ORCH-755 sour taste and rv 09-02_2052
PMC12452463 record `omitted_from_pwml:0_processes+...`, because the leg recorded admission removals are not in the replay own
report. This is a corpus artifact; a live record would count them.

**N8. NOTE (P2.7 item 1).** The 0/6 leg (rv 08-24_1203 PMC12856317) is policy under D-114 AMENDMENT 2 s3, not a defect. Agree with option (a).
Option (b) would contradict AMENDMENT 2 s3. Option (c) is overruled by AMENDMENT 2 s3.

**N9. NOTE (P2.7 item 2).** The closure prune codes print `(unclassified)` directly under the OTOP1/FepA actor blocks
(e.g. `- proteins OTOP1: degree_zero_after_quarantine (unclassified)`). That reads like an unexplained loss, although nothing drawn is
lost. My preference is (a), the ORCH-772 F-3 follow-up, using a "not an omission" catalogue row. Wording only.

**N10. NOTE (P2.7 item 3).** KI-023 workaround "None for a user (no file)." is now wrong for a kept transport. The actor block gives the
right fix, but the catalogue entry should be updated in a docs commit.

**N11. NOTE (P2.7 item 4).** ORCH-772b also edits around `reaction_evidence.py` and the ORCH-772 test. Whichever lane merges second
must re-run `test_orch772_pwml_coverage.py` (the 83/20 appendix pin), `test_orch773_omit_per_row.py` (`test_new_text_record_*`)
and `test_reaction_evidence_view.py` alone.

**N12. NOTE (boundary, wording).** `build_report` gains the key `omitted_actors`, and the "Entities left out" sub-line drops
"catalysts kept" for non-D-113 rows. Both are additive and serve the granted actor-block rendering. The only reader of the report is
`reaction_evidence_view.py` (grep), and `test_reaction_evidence_view.py` passes alone (120). In boundary, in my reading.

**N13. NOTE (wording).** The title reads "1 actors". The `removed_entities` bucket `transporters` names a process slot, not an entity
bucket (R1.7 item 3); agree it is wording.

**N14. NOTE (R1.7 item 4).** Round-0 F-2 (the cascade on a degree-0 protein), F-3 (the export re-quarantine branch, which now also
reaches the transporter edit on a refined payload) and F-5 (two calls outside the try) are unchanged and still open. They are
theoretical on the corpus (`quarantine_reused=True` on every fixed leg).

**N15. NOTE (process, this review).** (a) I ran `python.exe --version` once outside the wrapper, to choose f-string syntax. No stdin
was used, it exited at once, and the sweep after it was clean. (b) My G11 reports 108-123 and their pins are uncommitted in
`rev773r1-tip` (no commits, by rule). (c) The `g11next` reservations were all promoted, so no `.staging` reservation of mine is left
unused. (d) Heavy-lock contention with ORCH-772b-rev: two of my preflights saw the lock held, and I waited each time. I got no exit 95.

## Decision compliance

| decision | status |
|---|---|
| D-106 (identity requirement not relaxed) | satisfied: no predicate changed; unidentified proteins and transporters leave, none exported |
| D-112 s2 (corpus acceptance) | satisfied: ACCEPT reproduced (110); base and tip corpus records byte-identical to the implementer records (108, 109) |
| D-113 s1 / s2 | satisfied: 767 enterobactin wrapper leaves its reaction, EntF complex stays, gold 5/5; count gate untouched |
| D-113 s3 orchestrator reading | superseded by D-114 AMENDMENT 2 s3; the 0/6 leg exports `review_required` (N8) |
| D-114 s2 (per-row omission pre-freeze, recorded, no gate relaxed, review_required) | satisfied (110, 114, 121, 122); N1 for the PathWhiz-side reaction gap |
| D-114 s3 (text record lists everything left out) | satisfied for processes and actors (114) |
| D-114 AMENDMENT 1 s1 / s2 / s3 | satisfied (charge item 1); admission-time scope open (N3) |
| D-114 AMENDMENT 2 s3 | respected (N8) |
| D-104 AMENDMENT 2 (LLM inputs) | satisfied (item 6) |
| merge rule 8 | satisfied in the freeze path; N14 (F-3) unchanged |
| protected `streamlit_app.py` | byte-identical to 3ef3fa9c and 00eb12d3 |

## What I could not verify

* The round-1 affected sweep (30 files), Chunk E alone, `tests/test_eval_v2_*.py`, and the alone runs of `test_c052_*`,
  `test_orch769c_*`, `test_c055_*`, `test_c030a_*` and `test_reaction_evidence_app_hook.py`: I did not re-run them (time). The implementer
  reports 94, 92, 95 and 102-106 are not reproduced by me.
* `ki_census.py` and `coverage_table.py` (P2.3, P2.4 coverage numbers): not re-run. I checked only the KI ids on the 7 actor blocks.
* `test_strict_failure_replay.py` at base with the old `cases.json` values: not re-run. The old values come from the `0971487b` diff.
* An actual PathWhiz import of any PWML. N1 and N6 are read from the PathWhiz source, not from a measurement.
* Live runs with the PathBank DB reachable, and the live effect of the cap (N2), which I priced from recorded reports only.
* The `run_pwml_export` re-quarantine branch on a refined payload (N14): reasoned from the code, not run.
