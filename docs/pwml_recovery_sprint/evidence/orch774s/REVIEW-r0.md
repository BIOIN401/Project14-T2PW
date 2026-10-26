# ORCH-774s review, round 0 (independent reviewer)

Branch `orch774s/summary-import-check`, base `1f223061`, code `b12086f9`, tip `2249c01c`. I read the full diff of
`evidence/orch770_morning_summary.py`, `tests/test_orch774s_morning_summary.py` and `evidence/orch774s/validate.py`, and the RESULT.

## VERDICT: APPROVE WITH NOTES

No blocking finding. Every number I checked agrees with the PWML XML and with PathWhiz `check_spontaneous`. The 772/772b cells
are byte-identical on every leg with an absolute leg path. No leg shape crashes the summary. One borderline item (N-1) changes the
n/a text of a missing leg directory; see the note.

## Checks

| check | result | evidence |
|---|---|---|
| Boundary | Only `orch770_morning_summary.py`, the new test file, and docs/evidence (RESULT, `orch774s/`, G11). No `src/` change. | `git diff 1f223061 2249c01c --stat` |
| PMC12376012, re-derived myself | Corpus replay at `1f223061` (my run, ORCH-772b G11 `39`): `cov=15/17 (88.2%)`. Its PWML has 14 `<reaction>`, every one `<spontaneous>false</spontaneous>`, 9 with a `<reaction-enzyme>` and 5 with `<reaction-enzymes/>`. Under PathWhiz (enzymes < 1 and spontaneous == false) 5 are rejected, so would import (15 - 5)/17 = 10/17 = 58.8%. The summary script on that replay directory also gives dropped 5. | grep on `C:/t/rev772b/r1/out-tip/runs_verify_2026-09-06_1425_PMC12376012/pathway.review_required.pwml`; `C:/t/rev774s/shapes/fake/MORNING_SUMMARY.md` |
| Drop rule against PathWhiz | `reaction.rb:90-96` rejects (enzymes > 0 and spontaneous == true) or (enzymes < 1 and spontaneous == false); a missing spontaneous is nil and passes. `pwml_parser.rb` attaches the nested `reaction-enzymes` children before `save`, so the validation sees them. The summary counts (spontaneous text != true and no `reaction-enzyme`). | smpdb, read-only |
| The two rules agree on every real PWML | All 46 leg PWML files (runs_papers, runs_verify, the orch767/orch756 worktrees): 184 reactions, all `spontaneous=false` (the exporter hard-codes it, `ir.py:2064-2065`), 0 true. 9 empty `<reaction-enzymes/>`, 175 non-empty lists, no long-form empty list. Every `<reaction>` has exactly one enzymes list. No leg directory holds both PWML names. | `C:/t/rev774s/pwml_counts.txt` (grep) |
| Dropped column against the XML | Every leg in the summaries matches its own XML: ORCH-756 enterobactin 1 (would import 6/9), ORCH-756 menaquinone 0, ORCH-767 sour taste and menaquinone 0, ORCH-759 WNT PCP 0, ORCH-759p enterobactin, sour taste and chlorophyll 0. | `C:/t/rev774s/ms/tip-*/MORNING_SUMMARY.md` vs `pwml_counts.txt` |
| 772/772b cells unchanged | I ran the base (`1f223061`) and tip summary on ORCH-767, ORCH-759, ORCH-759p, ORCH-756 and ORCH-755 (from the repo root). Every no-PWML and not-recorded cell is byte-identical. Every PWML cell is the base text plus only `; would import x/head (p%)`. The only other difference is N-1. | `C:/t/rev774s/ms/{base,tip}-*/MORNING_SUMMARY.md` |
| Omitted list complete | Process rows equal full count minus drawn on every leg: ORCH-767 21/61/5/66/48/0, ORCH-759 11/20/97/0/0/36/0, ORCH-759p 9/24/4/79, ORCH-756 9/3/0/0, ORCH-755 13/14/17/27/7. KI ids come from `build_report` (`_entry` -> `known_issue_id`), not reimplemented. The actor rows read `omitted_actors` (keys `actor`, `kind`, `process`, `ki`, always set at `reaction_evidence.py:809-819`). | the summaries; code |
| Leg shapes | No crash on any of the following: no PWML, a PWML, crash/timeout legs (ORCH-755 menaquinone, chlorophyll, ara_operon, WNT PCP: `not recorded of 0`, `-`), relative leg paths from a non-root cwd, 81 replay directories, an empty directory, a malformed PWML (dropped `-`), a directory with only `RESULT.txt`, a missing directory, and an index row with no `leg_dir`. | `C:/t/rev774s/summaries.log`, `C:/t/rev774s/shapes/fake/MORNING_SUMMARY.md` |
| Tests | `test_orch774s_morning_summary` + `test_orch772b_no_pwml_cause` + `test_orch772_pwml_coverage`: 45 passed at tip. The new tests are labelled NEW ACCEPTANCE (G9). | G11 `06` |

## Notes

1. **N-1 (borderline): the n/a text of a missing leg directory changed.**
   * Base shows `n/a`; tip shows `n/a (no leg directory)` (`leg_report`). Seen on ORCH-755 when its relative leg paths are read
     from another cwd.
   * No number or KI label changes, and the new text keeps the existing `n/a (<why>)` form. Every ORCH-767/759/759p/756 cell
     (absolute paths) is identical apart from the charged suffix.
   * If "any change to the 772/772b cells" is meant literally, this is the only violation, and returning plain `n/a` for that case
     is a one-line fix.
2. **N-2: the drop rule is one-sided against PathWhiz.**
   * It does not count a reaction with an enzyme and `spontaneous=true` (also rejected by `check_spontaneous`).
   * It counts a reaction with no `<spontaneous>` element, which PathWhiz keeps (nil).
   * Neither case occurs: the exporter always writes `false` (`ir.py:2064`), and 46 of 46 files agree. Theoretical.
3. **N-3: an unreadable PWML would crash the run.**
   * `no_enzyme_reactions` catches only `ET.ParseError`. An `OSError` while reading the PWML, such as a locked file, would escape
     `coverage()` and `main()` and stop the summary from being written. `leg_report` swallows everything, but this call does not.
   * Theoretical; catching `(ET.ParseError, OSError)` would close it.
4. **N-4: the actor rows are untested on a real leg.**
   * Every leg available offline predates ORCH-773, so all show 0 actors. The 81 replay directories hold no leg-shaped artifacts.
   * The key names match the module by code reading, and no test covers an actor line. The ORCH-774 legs running tonight are the
     first real input.
5. **N-5: merged duplicates and non-process removals are left out.** The summary lists only `counted` rows, so merged duplicates
   (KI-014) and entity-only removals (kinds outside reaction/transport/interaction) are not listed. Those are not lost processes, so
   I agree, but the RESULT could say so.
6. **N-6: "would import" is an upper bound.** It subtracts only the reactions without an enzyme. A dropped reaction can also take
   its interactions with it, and `find_existing` reactions bypass validation. RESULT section 5 already says this.
7. **N-7 (pre-existing): an index row with no `leg_dir`.** Such a row becomes `Path("")`, i.e. the cwd, so the cell reads whatever
   is in the directory the runner was started from. This is unchanged from base; the new column inherits it.

## Lock, pauses and process hygiene

* The ORCH-774 overnight run took the heavy lock at 22:30, not 22:55.
* Lock-held exits, where the child never ran:
  * G11 `05` and `07`: held by ORCH-774i (smoke, then Chunk D);
  * G11 `10`: held by ORCH-774, so my standalone probe script never ran.
* **No PAUSE file was created.** PAUSE only acts between legs, and the runner was mid-leg.
* As the orchestrator allowed, I ran **the summary script itself without the heavy lock**, still under `bounded_run`, on finished
  leg directories only, offline:
  * G11 `09`: 12 runs, 5.7 s;
  * G11 `11`: 0.55 s.
* RAM was at least 8 GB free before each run. G11 `08` was a WSL-bash launch failure, so no Python ran.
* The orphan sweep found nothing of mine. The ORCH-774 runner processes were not touched.

## G11 reports

All under `.claude/worktrees/rev772b-tip/docs/pwml_recovery_sprint/evidence/g11/ORCH-774s/`, made at tip `2249c01c`, each with 0
survivors:
* `06-rev-tests-retry`: heavy lock, 45 passed;
* `09-rev-summaries-nolock2`: no lock, summary only;
* `11-rev-summary-replays-nolock`: no lock, summary only.

Not results: `05`, `07`, `10` (exit 95) and `08` (WSL bash).

Also relied on: ORCH-772b G11 `39-rev1-corpus-tip-seed5`, the replay at `1f223061` that is the source of PMC12376012.

## Not verified

* The actor rows on a real leg (N-4).
* A standalone Python check of the PWML files. It was blocked by the lock, so it was replaced by the grep survey
  (`pwml_counts.txt`) and by the summary script itself.
