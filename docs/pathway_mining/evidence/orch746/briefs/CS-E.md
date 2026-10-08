# ORCH-746 change set E — the two open species-gate blockers, nothing else

You implement change set E of ORCH-746 for T2PW pathway mining. Read these first, in order:

1. `C:/t/bt/orch746/briefs/COMMON_RULES.md` — the implementer rules. They bind you, with three changes:
   the base SHA is **`686ae134`** (not `f9556288`); the mining-suite baseline is **856 passed**; your scratch
   directory is **`C:/t/bt/m746-e/`** and your G9 manifest is `C:/t/bt/m746-e/g9_manifest_cs_e.json`.
2. `C:/t/mining/docs/pathway_mining/evidence/orch745/reviews/csd_r2_final_full.txt` — the final review of the
   species gate. It names the two blockers, every probe that demonstrated them, and the prototypes.
3. The three prototypes in `C:/t/mining/docs/pathway_mining/evidence/orch745/prototypes/`:
   `rev745d_r2_proto.py` (bio-auditor, `_excluded`), `rt745dr2_fix.py` (red team, `_excluded`),
   `rev745d_r2_proto_win.py` (bio-auditor, the `_NO_EFFECT` window).
4. In your worktree: `src/t2pw/mining/species.py` (`_NEW_SUBJECT` :479, `_NO_EFFECT` :488-507, `_excluded`
   :803-816, `_effect` :819-854, its one caller :947), `docs/pathway_mining/species_policy.md` (rule 5(b) item 4,
   lines 141-219) and `docs/pathway_mining/known_limitations.md`, and `tests/test_mining_species_cs_d_repair2.py`
   (its `probe()` helper is the reviewers' fixture).

**Worktree** `C:/t/m745-b`, branch `orch745/b-species-reading`, commits on top of `686ae134`. It is clean.
**Allowed files, and only these:** `src/t2pw/mining/species.py`, a NEW `tests/test_mining_species_cs_e.py`,
`docs/pathway_mining/species_policy.md`, `docs/pathway_mining/known_limitations.md`, `docs/pathway_mining/testing.md`.
Commit subject style: `experiment: ORCH-746 CS-E <id> - <what>`.

## The binding property: E only subtracts credit

The product owner's rule for this round: **repairs may only subtract unsafe credit; no recall gain of any kind.**
Every sentence that earns `in_planta` after your change must also earn it at `686ae134`. Design for that property
and prove it; do not argue it.

## E1 — RT-D-R2-1 / BIO-D-R2-1: an inner comma fools `_excluded`

At `686ae134`, `_excluded` treats a comma-led clause as closed at ANY comma after its FIRST opener, and treats a
participant as inside a closed clause whenever its gap holds any comma. A parenthetical comma ("(Fig. 3A, B)",
"(9S,13S)", "(3R,7S)"), a second comma-led clause, or ", which, after wounding, ..." fools it, so the head
participant gets the opposite predicate: product-up / substrate-down and reversed reactions are admitted.

The two reviewer prototypes fix this differently, and **neither dominates the other** (the orchestrator checked by
hand; you must confirm by running them):
- the red team's `rt745dr2_fix.py` still admits the bio-auditor's E1r ("In cyp94b3 mutants, JA-Ile, which is the
  substrate of CYP94B3, accumulated after wounding, with reduced 12-OH-JA-Ile.", reversed claim) because it checks
  only the FIRST opener in the gap;
- the bio-auditor's `rev745d_r2_proto.py` credits correct-direction rows that `686ae134` and the red team refuse
  (P1c, F1c), which is a recall gain this round forbids.

The handoff says: use the bio-auditor's prototype and, where it and the red team's differ, take the STRICTER
behaviour. Combined with the subtract-only rule, the orchestrator's design decision is:

> **A form earns its effect only when all three readings credit it**: the `686ae134` `_excluded` (unchanged), the
> bio-auditor's `_excluded` (blank parentheticals, only ", " closes a clause, the LAST opener in the gap decides
> whether it is open, a head kept only in the shape "X, <opener> ..., <form>" with one opener adjacent to X), and the
> red team's `_excluded` (a head kept across a closed clause only when the clause opens right after it).

The simplest faithful construction: move `_effect`'s body into a helper that takes the exclusion rule as a
parameter, keep the three rules as three small functions, and have `_effect` return the first reading's reason only
when every reading returns a non-empty reason. That makes "credit at E ⊆ credit under each reading" true by
construction. If you find a smaller construction with the same guarantee, you may use it, but you must state the
guarantee and show why it holds. Do not invent a fourth reading or any other rule.

Docstrings must explain WHY there are three readings (each prototype misreads a shape the other reads correctly;
agreement is the fail-closed choice; the `686ae134` reading is kept so the change cannot add credit).

## E2 — BIO-D-R2-2: the negated-comparison / delay window stops at an abbreviation

In `_NO_EFFECT`, the CS-D repair-1 R1 entry's window `(?:[^.;:]|[.:](?=\d)){0,60}?` stops at a full stop not
followed by a digit, so "The opr3 mutant did not accumulate OPC-8:0 (Fig. 3A) until 6 h after wounding." is credited
as an absence. Replace the window with `[^;]{0,60}?` (`rev745d_r2_proto_win.py`). The text is already one split
sentence, and a wider veto only refuses more. Do NOT add a bare `than` veto: "JA-Ile was 20-fold lower in jar1
mutants than in wild-type leaves." must keep its credit. `_NO_EFFECT` is used only as a veto (species.py:923), so
this change can only subtract credit.

## Tests — `tests/test_mining_species_cs_e.py` (new)

Write the tests FIRST and run them at `686ae134` (before touching `species.py`), so you see every regression row
fail there; then implement.

**Regression rows, named `test_cs_e_746_*`, each asserting the claim is NOT `target_species_supported`, and each of
which MUST FAIL at `686ae134`.** Every probe the final review demonstrated as admitted at `686ae134`:
- BIO-D-R2-1: P1r, E1r, F1r, E2r, P3r, F4r, P2, F2, E9, F3 and the six regression-fixture rows in the review
  (lines 146-151). The bio-auditor's scripts are `C:/t/bt/rev745d-r2-bio/scripts/{c2,c3,e1,e3}.py`.
- RT-D-R2-1: T1, T1b, T2r-T5r, T6-T9, W1r-W7r, W8, W9, H4r and the five regression-fixture rows (lines 490-494).
  Scripts: `C:/t/bt/rev745d-r2-redteam/scripts/{e2e_r2,e2e_r2b,r2probe,r2probe2}.py`.
- CSD-R2-N1 (code reviewer): A1, A2, A4, A5, B1. Script: `C:/t/bt/rev745d-r2-code/scripts/pm_r2.py`.
- BIO-D-R2-2: CF1, CF2, CF4 (review lines 211-213), and the code reviewer's CSD-R2-N2 CF1-CF4.

Those scripts are READ ONLY: copy what you need into `C:/t/bt/m746-e/`, never write into another directory.
Reproduce each row with the claim the reviewer used (substrate, product, enzyme/gene, observed organisms). **If a
listed probe is NOT admitted at `686ae134`** (it was a control, or it needs a path your fixture does not take), do
not put it in the regression set: list it in your report with its observed state. Every row in the regression set
must fail at `686ae134` with an `AssertionError`; that is the G9 proof, and the orchestrator re-runs it.

Also add ONE end-to-end row on the admission path for each blocker (P1r and CF1): build the claim, register it
in a `ClaimStore`, judge it with `admission.judge` (or the `MiningController._admit` path the existing species tests
use), and assert it is not accepted. Use only functions that exist at `686ae134`.

**Controls, named `test_new_acceptance_cs_e_746_*`, that pass at BOTH `686ae134` and your tip:**
- "No JA-Ile was detected in gh3.10 jar1 knockout lines." (GH3.10 claim) → TSS / `in_planta`;
- CF5, the "(Figure 3A)" sentence of the review → stays NOT TSS;
- "JA-Ile was 20-fold lower in jar1 mutants than in wild-type leaves." → TSS;
- the bio-auditor's A1-A10 (`C:/t/bt/rev745b-r2-bio/scripts/probes.py`, results `.../runs/r6/out.txt`) → TSS /
  `in_planta`, where not already pinned by an existing test (say which).

Existing tests that must still pass untouched: the T-rows (T1, T1b, T2-T11) and the full pilot-003 replay (exactly
#3, #14, #34), the CS-D repair-2 recall rows (`test_cs_d_r2_745_1_the_head_before_a_closed_relative_clause_keeps_its_credit`,
"OPC-8:0, which is formed from OPDA, was undetectable", and the rest), and the whole 856. Do not edit any existing
test. If one fails, stop and report it; do not "fix" the expectation.

**Recall cost, recorded not tested as credit:** the correct-direction partners (P1c, E1c, F1c, P3c, E2c, E7c, T2c-T5c,
A4c) — give each a `test_new_acceptance_cs_e_746_*` row pinning its state at your tip, with a comment saying
whether it is credited at `686ae134` (it may only lose credit, never gain it).

## Prove the subtract-only property on every probe

Run every probe from the three reviewers' scripts above (all rows, regression rows and controls alike) under four
trees: your worktree at `686ae134` before your change; `686ae134` + `rev745d_r2_proto.py` + `rev745d_r2_proto_win.py`;
`686ae134` + `rt745dr2_fix.py` + `rev745d_r2_proto_win.py`; and your tip. (The prototypes are monkeypatch plugins:
import them in a copy of the script; they print `PLUGIN ACTIVE`.) Produce one table
`C:/t/bt/m746-e/probe_matrix.txt`: probe id, claim, and the species state under each of the four trees. The tip
column may hold TSS only where ALL three other columns hold TSS. Report any row that violates that.

## Docs (describe what ships; plain statements, real names)

- `species_policy.md` rule 5(b) item 4: replace the `_excluded` description with the three-reading rule (what each
  reading does, credit only on agreement, and that a parenthetical's commas never close a clause in the bio reading);
  replace the window description ("a full stop or colon followed by a digit does not end the window") with "the
  window runs to the end of the sentence or a semicolon, 60 characters at most".
- `known_limitations.md`:
  - correct lines 78-81 ("nothing but plain words between them") and the CS-D repair-2 bullet (lines 149-162) so they
    are true of the shipped rule; list figure callouts, stereo descriptors and inner comma pairs;
  - add the recall cost: the correct-direction rows above that earn nothing;
  - correct lines 139-148: the window now crosses abbreviation full stops;
  - the final review's doc-only asks, verbatim in substance: RT-D-R2-2 (a participant that modifies a non-participant
    head, "the precursor of JA-Ile, which is jasmonic acid, accumulated", is read as the subject — check whether the
    three-reading rule still admits it and state what it does); BIO-D-R2-3 / RT-D-R2-3 (a regulator's or upstream
    gene's mutant written as "<marker> lacking a regulator of <GENE>" / "with defects upstream of <GENE>" within three
    tokens still binds); RT-D-R2-4 (replace the 6454ef98 precision figures with the 686ae134 measurement: 30
    surviving, 22 credited with the screen empty of which 1 correct, 20 held by the screen, 2 credited with it of
    which 1 correct (T1 correct, dapat wrong); name the CWH8 "produced no farnesol" sentence (PMC12505894) as the
    measured recall cost); CSD-R2-N3 (a gap token starting "in-"/"of-", "CRISPR in-frame deletion lines of AOS", no
    longer binds: a recall cost).
- `testing.md`: the new file and its row count, and the new suite total.

## Final report (your last message; it is all the orchestrator sees)

As COMMON_RULES § Final report, plus: the probe matrix path and its summary (rows per tree, any violation); every
listed probe you left out of the regression set and why; the A1-A10 status; the full mining-suite count with
`final_surviving_count`; and your own honest list of shapes the three-reading rule still admits that you suspect are
wrong (do not fix them: report them).
