READY TO LAUNCH: the species gate is approved, merged and green at `experiment/pathway-mining` @ `cc9dbc0d` (mining suite 1145, SMOKE 508). Pilot-004 is pre-registered (`9217349a`) and staged in `C:/t/mining-pilot4`. Per the product owner's instruction, nothing has run: no preflight, no pilot, no model call.

# ORCH-748 — pathway mining back to launch-ready: result

2026-10-15 · Lead Orchestrator. Follow-up to `ORCH-746-RESULT.md` (BLOCKED on CSE-R2-C1).
- Every subagent ran on Opus 5.5 at extra-high effort, as the product owner asked.
- **Nothing merged into `sprint/pwml-recovery` or `main`. Nothing pushed. $0 spent.**
- Production is untouched: every change is in `src/t2pw/mining/species.py`, its tests and `docs/pathway_mining/`.
- The product owner's ORCH-747 work in the primary checkout was not touched. The only commit there is the dispatch entry
  `c9265fb8`, made while `LEDGER.md` was clean.

## Product-owner decisions (2026-10-15)

1. Unblock with ORCH-746's options (a) + (c): one subtract-only micro-round applying exactly the final code reviewer's
   measured fix, then ONE re-review with no repair round.
2. A new review bar. Blocking = a weakening, a contract violation, or a realistic end-to-end route in a class
   `known_limitations.md` does not document. A new paraphrase of a documented class is a pilot watch item. A docs
   inaccuracy is a required docs-only fix.
3. Offline tests only, in the `C:/t` worktrees. Pilot-004 staged, not run.
4. Do not interfere with ORCH-747.

## Final state

| item | value |
|---|---|
| `experiment/pathway-mining` | **`cc9dbc0d`** code (merge of `orch745/b-species-reading` @ `8668d18a`); tip `9217349a` with the records and pre-registration |
| species gate | change sets B, D and E (rounds 0-2) plus ORCH-748 repair 3: `c4a1468f` (five `_NO_EFFECT` entries F1-F5 + 18 tests), `8a766c3d` / `42bf7caa` / `8668d18a` (docs) |
| mining suite | 1127 at `f3df072f` → **1145** at `cc9dbc0d` (`056`) |
| SMOKE | **508** with mining off at `cc9dbc0d` (`057`, `--heavy-lock ORCH-748`) |
| G9 | `055`: 16 / 16 `test_g9_748e3_*` fail with `AssertionError` at `f3df072f` (orchestrator, pinned). Reproduced independently by the code reviewer and the red team |
| survivors | 0 on every bounded run (implementer, orchestrator, 3 reviewers, test runner) |
| pilot-004 | pre-registered `9217349a`; worktree `C:/t/mining-pilot4` detached at `cc9dbc0d`, clean, `.env` copied; runs dir `C:/t/mining-pilot-runs4`; launch commands `docs/pathway_mining/evidence/orch748/LAUNCH-PILOT-004.md` (also `C:/t/mining-pilot-runs4/LAUNCH.md`) |
| services at staging | LM Studio serving `text-embedding-nomic-embed-text-v1.5`; heavy lock free |
| spend | $0 |

## What repair 3 changed

Five subtract-only `_NO_EFFECT` entries, verbatim from the ORCH-746 code reviewer's prototype `rev746e_r2_fix.py`:
- F1 `<n> min / h ... but`
- F2 a later rise after "but" / ";"
- F3 `no statistically`
- F4 Columbia / Wassilewskija / Landsberg as control words
- F5 `initially` / `at first`

The compiled pattern is byte-identical to the prototype applied to `f3df072f`.

**Measured:**
- 721-probe matrix: 163 credited rows at both trees, 0 rows changed.
- Widening: 0 in about 166,000 fuzz pairs and 544 end-to-end rows at three trees.
- Real-corpus precision unchanged (22 / 17 / 15 / 2); no correct credit lost.
- Recall cost: 6 of 566 real loss-of-function effect sentences, and 0 of the jasmonate slice.
- Known correct-biology losses:
  - "aos mutants (Columbia background) accumulated no 12-OPDA after wounding.";
  - "(Fig. 2H) but ..." sentences;
  - a substrate rise after "but" read as the product's.

## Review (one round, under decision 2's bar)

| reviewer | verdict | fit for pilot | required docs fixes |
|---|---|---|---|
| `pwml-reviewer` | APPROVE_WITH_FINDINGS; docs re-check REQUEST_CHANGES at `42bf7caa`, **APPROVE** at `8668d18a` | YES | CSE-R3-C1, -C2, then the sentence-scope heading |
| `pwml-bio-auditor` | SOUND_WITH_FINDINGS | YES | BIO-748E3-1, -2 |
| red team | SOUND_WITH_FINDINGS | YES | RT-748-1 |

No blocking finding.

**Orchestrator ruling on RT-748-1.** It found three undocumented classes, each demonstrated end to end:
- a computational knockout read as an observation;
- a genotype modifier the gate does not read;
- vetoes scoped to their own sentence.

I upheld the red team's own classification as a docs fix. All three use constructed prose with 0 corpus instances, all are
pre-existing at `686ae134`, and none is a weakening. Each is now documented and on the watch list.

Reviews as returned: `docs/pathway_mining/evidence/orch748/reviews/`.

## Not fixed (registered)

- **Optional docs notes not applied:**
  - the photoperiod false refusal ("16 h light, ..., but"; CSE-R3-C3);
  - "transiently" refused as a host marker (RT-748-2).
- **Performance:** F2 doubles a documented quadratic worst case. Pathological 60 KB "sentences" with 10,000 "but"s take
  12.6 s (RT-748-3, CSE-R3-C3); real text takes ≤ 0.2 ms.
- **Pre-existing, unchanged:** every ORCH-746 open item:
  - allele-numbered genotypes;
  - the collapsed `13-HPOT -> 12-OPDA [AOS]` node (W23);
  - organism-screen dependence (W30);
  - unverified participant citations (F-a);
  - `unknown` papers keeping primary credit (F-b);
  - fed-precursor / occurrence evidence held back;
  - "only marginally less" credited.
- **Change set F:** not started. It follows pilot-004.
- **Production cards `PF-746-01..05`:** still registered, not dispatched.

## For the product owner

1. **Launch pilot-004** when your other work is done. Follow `LAUNCH-PILOT-004.md`: pre-launch checks, preflight, then ONE
   bounded run of about 20-40 minutes under the heavy lock, at most about $0.32. Then the two audits and the record branch.
2. **Sprint ledger.** The ORCH-748 progress, merge, pre-registration and close entries live in
   `docs/pathway_mining/evidence/orch748/PROGRESS.md`. They are not yet in the sprint `LEDGER.md`, because that file carries
   your uncommitted ORCH-747 entry and a commit there would sweep it in. Say when ORCH-747 is committed, and I will append
   them and add this result to `docs/pwml_recovery_sprint/`.
3. **Expect a thin pilot.** All three reviewers predict few in-planta admissions and a review-route majority. Watch W14
   (basal / mock absence) above all.

## Addendum, 2026-10-20: pilot-004 was run

Launched on the product owner's go-ahead, exactly as pre-registered. **PLAUSIBLE_BUT_THIN, definition of done MET, no
blocking finding**: 1 admitted target reaction (OPDA -> 4,5-didehydro-JA, review grade, `CORRECT_TARGET_SPECIES`),
0 admissions against policy, 0 wrong identifiers, 0 survivors, est. $0.066. Full record:
`experiment/pathway-mining-pilot-004` @ `9128aa8c`, `docs/pathway_mining/pilot/004/PILOT-004-RESULT.md`.

The "For the product owner" items above are now settled as follows: item 1 is done; item 2 is done in the sprint
`LEDGER.md` close entry. Item 3's prediction held: the pilot was thin, and W14 did not fire. The new constraint is
extraction yield: the 12,000-token budget and the shrink ladder's prefix truncation lost reactions. The
2026-10-19 re-check's claim that the budget could not reach mining was wrong; the correction is in
`PROGRESS.md` (`c6570797`).
