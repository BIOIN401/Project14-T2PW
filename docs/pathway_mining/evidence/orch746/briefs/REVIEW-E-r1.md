# ORCH-746 change set E — repair round 1 review (shared brief for all three reviewers)

**What you review.** T2PW pathway-mining species gate, worktree `C:/t/m745-b`, branch `orch745/b-species-reading`.
- Repair-1 delta: `git -C C:/t/m745-b diff 5eca087b..01d436cf` (13 commits; `species.py`, `tests/test_mining_species_cs_e.py`
  (appended), `species_policy.md`, `known_limitations.md`, `testing.md`).
- Whole change set E: `686ae134..01d436cf`. `686ae134` is the species gate ORCH-745 could not approve; `5eca087b` is E's
  round-0 tip.

**History.** ORCH-745's final review left two blockers (`C:/t/mining/docs/pathway_mining/evidence/orch745/reviews/csd_r2_final_full.txt`):
RT-D-R2-1 / BIO-D-R2-1 (an inner comma fooling `_excluded`) and BIO-D-R2-2 (the negated-comparison window stopping at an
abbreviation). E round 0 (`5eca087b`; brief `C:/t/bt/orch746/briefs/CS-E.md`) made an effect count only when three readings
of the clause agree (the `686ae134` `_excluded`, the bio-auditor's and the red team's prototype readings) and widened the
window. Its round-0 review (`C:/t/bt/orch746/reviews/cse_r0.txt`) confirmed the subset property and the two blockers closed,
and found three blocking routes: CSE-R0-1 (an appositive / inner comma pair after an adjacent head), RT-E2-1 (a negated delay
with absence forms other than "did not / failed to") and RT-E2-2 (no-difference prose: non-significance, mutant "and"
wild type). Repair 1 (brief `C:/t/bt/orch746/briefs/CS-E-repair1.md`, plus an addendum, items 9-10, sent to the implementer)
does, all as subtract-only changes:
1. a fourth reading (the code reviewer's `rev746e2_r4.py`): two or more ", " after the opener do not close the clause;
2. delay vetoes: `until` anywhere, `before / prior to` + a number;
3. no-difference vetoes (the red team's `rt746e_veto.py`, verbatim);
4. a concessive veto (`although / though / despite / in spite of`);
5. the "accumulated ... less" form earns nothing when a participant sits inside its span;
6. a fifth reading: a bare " with " (no comma) opens a clause;
7. `nowhere` in the gap-negator lookahead;
8. `_MARKER_OF` gap tokens may not be `regulat* / upstream / downstream / gene(s) / factor(s) / activator(s) / repressor(s) / defect(s) / target(s)`;
9. a sixth reading: a comma followed by a participle or consequence word (`\w+ing`, accompanied, followed, coupled, paralleled,
   associated, yet, thus, thereby, hence, so) opens a clause;
10. a `within (experimental / measurement / technical) error / variation` veto.

**Evidence so far.**
- Implementer: full mining suite at `01d436cf` = **1005 passed** (856 at `686ae134`, 915 at `5eca087b`), 0 survivors. Probe
  matrix `C:/t/bt/m746-e/probe_matrix_r1.txt` (423 probes = the 223 earlier ones, every round-0 reviewer probe, 13 + 6 of its
  own; trees `686ae134`, both ORCH-745 prototypes, `5eca087b`, `5eca087b` + this round's reviewer prototypes, tip; scripts
  `matrix_r1.py` / `combine_r1.py`): TSS 265 / 209 / 205 / 189 / 123 / 88; **0 rows credited at the tip that `5eca087b`
  refuses**; 101 rows lose credit, of which the correct-biology losses are O1c (concessive), Q2 (delay) and two
  participle-clause rows (documented).
- Orchestrator's G9 base arms, pinned in `C:/t/mining-rep-base`: `052` — all 44 round-0 rows fail at `686ae134`; `053` — all
  **82** `test_cs_e_r1_746_*` rows fail with `AssertionError` at `5eca087b`, 0 survivors, pin violations `[]`
  (`C:/t/mining/docs/pathway_mining/evidence/053-g9-746e-r1-at-5eca087b.json` / `.pin.json`).
- The implementer's own list of shapes it still suspects (credited at the tip; documented in `known_limitations.md`): the true
  subject AFTER the effect word with no opener between ("JA-Ile accumulated after wounding at the expense of / alongside /
  plus reduced 12-OH-JA-Ile", reversed claim); RT-D-R2-2 ("the precursor of JA-Ile, which is jasmonic acid, accumulated");
  a regulator gap word outside the list ("a modulator of AOS"); a comparison > 60 characters after "did not" (CW1), after a
  semicolon inside a reference, or an unlisted comparison word ("different"); "only marginally / slightly less" (a
  product-owner policy item).

**Binding.** Biology spec `C:/t/mining/docs/pathway_mining/evidence/orch745/briefs/F1_BIO_ADJUDICATION.md` (T1, T1b, T2-T11;
the full pilot-003 replay admits exactly #3, #14, #34: #14 in_planta on ev_7d2787cbb14b4905[7], #3/#34 review /
review_article_only on ev_c98d84c7d84855c1[12]). Product-owner rules for E: close the two blockers and nothing else; repairs
only subtract credit, never add any; keep the fail-closed policy (in-planta credit only from loss-of-function genetics; no
discussion / conclusion credit; the effect-polarity rule; reviews never independent); fed-precursor and occurrence evidence
stays held back.

**SEVERITY STANDARD.** 'blocking' = (a) a gate weakened relative to `686ae134` (or relative to `5eca087b` for this round's
delta), (b) a contract violation, or (c) a route to an admission against policy DEMONSTRATED end to end (build_claims ->
ClaimStore -> admission.judge / MiningController._admit gives accepted) with realistic plant-biochemistry prose. A contrived
paraphrase inside the residual class that `known_limitations.md` documents = non_blocking. For every blocking finding name
the clause and give the end-to-end evidence path; for every non_blocking one say why. **This is repair round 1 of at most 2;
a required fix must only remove credit.** Also answer **fit_for_pilot YES/NO with a reason**: the next step is ONE
pre-registered live pilot (A. thaliana, "jasmonate biosynthesis", seeds 12-OPDA + AOS) whose every admitted reaction is
adjudicated by pwml-bio-auditor; its definition of done requires zero admissions against policy.

**Process rules.** Read-only on every repository file. Write only under your fresh scratch dir `C:/t/bt/rev746e-r1-<role>/`
(never into any worktree, never into another `C:/t/bt` dir; earlier scripts under `C:/t/bt/rev746e2-*/`, `C:/t/bt/rev745d-r2-*/`
and `C:/t/bt/m746-e/` are READ ONLY: copy what you need). No recursive deletes. Every python/pytest via the bounded wrapper:
`cd C:/t/m745-b && PYTHONPATH=C:/t/m745-b/src C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py --label <l> --timeout <=400 --json C:/t/bt/rev746e-r1-<role>/runs/<unique>/report.json -- <py> -u <script, or docs/pwml_recovery_sprint/evidence/pinned_pytest.py --pin-verdict <path> -q -p no:cacheprovider --basetemp=<fresh dir whose parent exists> <tests>>`
(pinned_pytest takes NO `--`); Bash tool `timeout` 480000; never `&`; never `pytest -n`; check `final_surviving_count` 0. For
probes at `5eca087b` or `686ae134`, load a scratch copy of `git -C C:/t/m745-b show <sha>:src/t2pw/mining/species.py` or a git
archive of that SHA in your scratch dir; never check anything out in a worktree. No network, no LLM calls, no SMOKE.

**Your final message is your review, in this exact shape** (it is all the orchestrator sees):
```
VERDICT: <APPROVE | APPROVE_WITH_FINDINGS | REQUEST_CHANGES | REJECT | SOUND | SOUND_WITH_FINDINGS | UNSOUND>
FIT_FOR_PILOT: <YES | NO> — <reason, and what the pilot auditor must watch>
SUMMARY: <what you checked and ran, with run paths, and what holds>
FINDINGS:
- <ID> [blocking | non_blocking] (<clause, or why not blocking>) <file:location>
  PROBLEM: ...
  EVIDENCE: <run paths; tip vs 5eca087b vs 686ae134 states>
  FIX: <subtract-only, or "none">
```
