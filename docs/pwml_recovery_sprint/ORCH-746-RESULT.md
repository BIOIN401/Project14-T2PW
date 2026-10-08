BLOCKED: change set E (the species gate) ended its last allowed repair round with one blocking finding (CSE-R2-C1: a temporary absence such as "JA-Ile was initially undetectable but increased after 1 h" is still credited as an absence; `pwml-reviewer` REQUEST_CHANGES), so the gate was not merged, pilot-004 was never run, and change set F, which builds on the merged gate and follows pilot-004, was not started. Definition of done items 2–4 are not met.

# ORCH-746 — species-gate fix round E, pilot-004, change set F: result

2026-10-08 · Lead Orchestrator. Brief: `C:/t/mining-handoff/ORCH-746-HANDOFF.md`. **Nothing merged into `sprint/pwml-recovery`
or `main`, nothing merged into `experiment/pathway-mining` except records, nothing pushed. No live model call ($0).**
Production is untouched: every code change is in `src/t2pw/mining/species.py` on the unmerged species branch.

**Timeline.** Started 2026-10-07 22:43. The machine slept from about 00:00 to 11:55, then crashed at about 12:06 and killed
the first E review after 5 minutes; after the restart no sprint process had survived, the heavy lock was free and every
worktree was clean, and the review was relaunched. Work ended at 14:45 on 2026-10-08.

## Bottom line

- **Both ORCH-745 blockers are closed**, and every reviewer at every round confirmed it: the inner-comma misread
  (RT-D-R2-1 / BIO-D-R2-1) and the negated-delay window stopping at an abbreviation (BIO-D-R2-2).
- **Every round only removed credit, by construction and by measurement.** Credit at each tip is a subset of credit at the
  previous tip and at `686ae134`. Up to 84,000 fuzzed sentences per reviewer, 721 probes and the real corpus found no
  exception. The pilot-003 replay still admits exactly #3, #14 and #34 with their binding grades.
- **The review did not converge, for the reason ORCH-745 recorded.** Every round, at least one reviewer found an ordinary
  paraphrase of the same two classes (a delay or temporary absence, a no-difference result) that the word lists missed:
  - round 0: "not detected until 30 min" (other absence forms);
  - round 1: "not detected at 15 min but accumulated at later time points", "did not reach statistical significance",
    "in jar1 mutants and control plants";
  - round 2: "initially undetectable but increased after 1 h".
  Repair 2 closed these by shape rather than words (two time points in a sentence; a later presence after "but"; a control
  genotype named without "than"; any significance or trend word). Two of the three final reviewers called the gate sound and
  fit for pilot. The merge-gate code reviewer found one more temporary-absence route and blocked.
- **Why BLOCKED rather than a ruling or a third round.** You ruled in advance: "If it still has a blocking finding after
  that: STOP and report BLOCKED. Do not open another species-gate change set." I upheld CSE-R2-C1 as blocking:
  - it is the class all three reviewers ruled blocking in rounds 0 and 1 ("…not detected 15 min after wounding, but
    increased strongly." is a near-twin of round 1's blocking DL8);
  - `known_limitations.md` says a later presence after "but" is caught, so the route is not inside the documented residual
    and the contrived-paraphrase carve-out does not apply;
  - merge rule 5 needs the reviewer's approval, and `pwml-reviewer` did not give it.
  The reviewer's subtract-only fix is measured at zero cost and committed as a record. Applying it would have been a third
  repair round.
- **Change set F and pilot-005 were not started** (F follows pilot-004 and tightens the merged gate). Its design was scouted
  read-only and recorded. **The five production follow-up cards are done** (docs only).

## Final state

| item | value |
|---|---|
| `experiment/pathway-mining` | **`4c2aa0b0`**. Code unchanged at `3ce64f28` (change sets A + C, ORCH-745); 9 ORCH-746 commits of evidence and records (`a9ed963b`…`4c2aa0b0`) |
| species gate, unmerged | `orch745/b-species-reading` @ **`f3df072f`** (`C:/t/m745-b`, clean): ORCH-745's B + D plus change set E's 24 commits (`686ae134..f3df072f`; `species.py` +244 / −16, `tests/test_mining_species_cs_e.py` 271 tests, three docs) |
| mining suite | `686ae134` 856 · `5eca087b` 915 · `01d436cf` 1005 · **`f3df072f` 1127 passed**, 0 survivors (implementer, re-run by every reviewer) |
| SMOKE | not re-run (no merge happened); last measured 508 with mining off at `686ae134` (ORCH-745 evidence `051`) and at `3ce64f28` (`049`) |
| G9 base arms (orchestrator, pinned, `C:/t/mining-rep-base`) | `052` 44/44 rows fail at `686ae134` · `053` 82/82 at `5eca087b` · `054` 103/103 at `01d436cf`; every failure an `AssertionError`, 0 survivors, pin `violations: []` |
| pilot branches | `experiment/pathway-mining-pilot-001` `dcb64fc4`, `-002` `d6cb9868`, `-003` `b7e26d8d`; **no pilot-004 or -005** |
| live spend | **$0** |

## Pilots

| | pilot-001 | pilot-002 | pilot-003 | pilot-004 |
|---|---|---|---|---|
| code | `23e1ae83` | `ecf6b9a0` | `f9556288` | **not run** |
| papers found / read | 21 / 1 | 150 / 10 | 64 / 4 | — |
| target (A. thaliana) reactions | 0 | 1 (in vitro: should not have been admitted) | 0 | — |
| bio verdict | NOT_PLAUSIBLE | PLAUSIBLE_BUT_THIN | NOT_PLAUSIBLE | — |
| why | | | | the species gate was not approved, so no approved SHA existed to pre-register |

**Admitted reactions: none in any pilot that ran under a correct gate.** What the unmerged gate admits on pilot-003's 16
recorded answers (offline replay at `f3df072f`; not a pilot, does not satisfy the definition of done):

| claim | reaction | grade, level | credited sentence | bio verdict (F1 adjudication) |
|---|---|---|---|---|
| #14 `cl_b89be46f92d4c4e1` | jasmonic acid → JA-Ile (GH3.10), PMC12096059 | in planta, single_paper | "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was below the detection limit." | CORRECT_TARGET_SPECIES |
| #34 `cl_0bc93d248b7f31db` | OPDA → OPC-8:0 (OPR3), PMC13364333 (review) | review, review_article_only | "In Arabidopsis, OPDA and dn-cis-OPDA are reduced by OPR3 in the peroxisomes to produce OPC-8:0 and OPC-6:0, respectively." | CORRECT_TARGET_SPECIES at review grade |
| #3 `cl_5338b4fc07a47cea` | dn-cis-OPDA → OPC-6:0 (OPR3), PMC13364333 (review) | review, review_article_only | same sentence | CORRECT_TARGET_SPECIES at review grade |

## Change set E

| round | tip | suite | review: code / bio / red team | blocking | fit for pilot (code / bio / red team) |
|---|---|---|---|---|---|
| 0 | `5eca087b` | 915 | REQUEST_CHANGES / SOUND_WITH_FINDINGS / UNSOUND | CSE-R0-1 appositive comma pair; RT-E2-1 delay with other absence forms; RT-E2-2 no-difference | YES under audit / YES / YES after its vetoes |
| repair 1 | `01d436cf` | 1005 | REQUEST_CHANGES / UNSOUND / UNSOUND | the same two classes in other words: CSE-R1-C1–C3, RT-R1-1–2, BIO-E746R1-1–2 | YES once each reviewer's vetoes are in |
| repair 2 (last) | `f3df072f` | 1127 | **REQUEST_CHANGES / SOUND_WITH_FINDINGS / SOUND_WITH_FINDINGS** | **CSE-R2-C1** temporary absence ("initially undetectable but increased") | NO as shipped, YES with its zero-cost fix / **YES** / **YES** |

Every review is committed as returned under `docs/pathway_mining/evidence/orch746/reviews/` on `experiment/pathway-mining`,
with every brief (`briefs/`), G9 manifest and reviewer prototype (`prototypes/`).

**What E changed** (all in `species.py`, all subtract-only):
- A loss-of-function effect counts only when **six readings** of which participant a comma-led clause cuts off all agree.
  These are the `686ae134` reading, the bio-auditor's (parentheticals blanked), the red team's (head adjacent to the opener),
  an ambiguous-close reading, a bare-"with" reading and a participle / consequence reading.
- Vetoes in `_NO_EFFECT`, written as shapes:
  - a time course: two time or stage points; a later presence after "but" or ";"; delay words;
  - non-significance: any `statistical`, `significant` or `significance`, trends, `n.s.`, p-value phrases, variability;
  - a result shared with a control: a wild-type or control genotype word in a sentence with no "than"; "all genotypes";
  - concessive openers, "nowhere near", "within experimental error", and a position read as a change ("lower leaves").
- Background wording ("known to", "shown previously", "well documented") in `_CITATION`.
- Regulator wording in the marker tail and after the gene (`_MARKER_OF`, `_CONTAINER`).
- A participant directly after the effect word joins the subject candidates (attributive "reduced X").
- The negated-comparison window runs to the sentence end.

**Measured effect** (implementer's matrix, re-run by the reviewers on git archives):
- **Probes credited:** 721 probes; credited at `686ae134` 495, `5eca087b` 419, `01d436cf` 283, `f3df072f` 163. Nothing
  credited at a later tree is refused at an earlier one.
- **Real-corpus precision** (red team; surviving / credited with the screen empty / held by the screen / credited with it):
  - `5eca087b` 30 / 22 / 20 / 2;
  - `01d436cf` 29 / 21 / 19 / 2;
  - `f3df072f` 22 / 17 / 15 / 2.
  At every tree the only correct credit is T1, and the screened set is T1 (correct) plus a pre-existing `dapat` credit
  (wrong). No correct corpus credit was lost.
- **Recall cost, documented in `known_limitations.md` on the species branch.** Repair 2 alone loses 11 correct-biology probe
  rows. On real text, 93 of 566 loss-of-function sentences carrying an effect form (16 %) are newly refused. The costs that
  matter most for this pilot:
  - the concessive veto refuses "Although JA accumulated after wounding, JA-Ile was strongly reduced in jar1 mutants.";
  - the later-presence shape also matches a *negated* presence, so "In aos mutants, 12-OPDA was not detected, but 13-HPOT
    accumulated" and "OPDA accumulated to high levels, but OPC-8:0 was not detected." earn nothing. These are strong evidence
    for the pilot's own seed enzyme. BIO-E746R2-4: the docs misdescribe this as "a later presence";
  - "undetectable in jar1 mutants but abundant in Col-0" (control word without "than") and "reduced at 30 min and 1 h" (two
    time points) earn nothing.

**What still admits against policy at `f3df072f`** (every route is also admitted at `686ae134`, so nothing was weakened):
- the blocking CSE-R2-C1 temporary-absence family ("initially / at first undetectable but increased", "…after wounding,
  but rose / reached");
- documented residuals, which the two biology reviewers rated constructed or rare:
  - a later presence written with an unlisted verb, or across a semicolon;
  - non-significance with no listed word ("not reproducible");
  - a comparator outside the control list ("the complemented line", "Columbia");
  - the true subject more than two tokens after the effect word, and "the precursor of X, which is Y" (both need a
    direction-reversed extraction);
  - a role word outside the regulator lists ("a partner of AOS");
  - "an established JA-Ile-deficient line";
  - "reduced turnover";
  - a comparison more than 60 characters after "did not".

## Change set F

Not started. F (participant citations verified, `unknown` papers read as reviews, a per-paper principal organism) tightens the
merged species gate and is sequenced after pilot-004. The species gate was not merged and pilot-004 did not run, and you ruled
out another species-gate change set. A read-only design scout ran while E was implemented. Its map is
`docs/pathway_mining/evidence/orch746/cs_f_design_scout.json`:
- **F-a:** keeps #14 only through the paper's own "jasmonic acid (JA)" alias. About 70–85 existing test rows cite a
  participant to a span that never names it, so their fixtures or expectations move (merge rule 4).
- **F-b:** needs no `species.py` change (`models.reads_as_review` plus fixtures). The replay fixtures already declare paper
  types.
- **F-c:** needs the EuropePMC metadata and a new paper field.

## Production follow-up cards (docs only, D-102; production untouched)

Drafted from production code at `b4bb0b74`, each adversarially checked line by line against the code (every draft needed
11–18 corrections). Text in `C:/t/mining-handoff/cards/` and committed under `docs/pathway_mining/evidence/orch746/cards/`.
Each says "REGISTERED, not dispatched".

| card | defect |
|---|---|
| `PF-746-01` | `rag/eligibility.py`: the organism lexicon holds 45 entries, only 2 of them plants (Arabidopsis thaliana, Quercus robur), so the paper screen cannot see most plant hosts (wheat, Marchantia, pepper, grape papers screened with no organism); the species gate leans on that screen |
| `PF-746-02` | `llm/client.py`: provider routing unpinned and unreported; the reasoning setting never sent; `response_hash` only for replies with text; usage only process-cumulative |
| `PF-746-03` | `rag/acquire.py:536-543`: `.//ArticleIdList/ArticleId` also matches cited references, so the last cited PMCID / DOI wins (pilot-002's 7 wrong ids and 2 wrong full texts) |
| `PF-746-04` | `rag/embed`: the lexical-fallback embedding is cached under the real model key, so retrieval stays poisoned after the endpoint returns |
| `PF-746-05` | `rag/ingest._split_sections`: first-occurrence header matching and the `body` fallback mislabel sections, which the species gate's section rule depends on |

## Remaining known limitations

- The species gate is lexical. It is now fail-closed by shape for the delay and no-difference classes, and a paraphrase
  outside the shapes still earns credit (listed above).
- Allele-numbered genotypes ("jar1-1", "opr3-1", "dde2-2") never bind the claim's gene. Every reviewer named this as the
  main reason a live pilot would be thin; it is pre-existing.
- A coordinated subject loses its first participant at the "and" split.
- A collapsed `13-HPOT → 12-OPDA [AOS]` node (AOS forms the allene oxide; AOC cyclises it) can earn in-planta credit from
  canonical "aos mutants accumulated no 12-OPDA" prose. The bio-auditor named it the likeliest adjudication issue for an
  AOS-seeded pilot (cascade B4, outside E).
- Everything ORCH-745 registered as not fixed still stands, apart from the two blockers E closed: B4, A9/A17, B5, the screen
  missing At enzymology papers, N-9/N-10/N-11, N-1, N1, per-batch lineage, the `report.md` count misstatements, the CS-A
  non-blocking items, and BIO-745B-2 (a participant's citation is never checked against its span: F-a).

## Decisions for the product owner to revisit

1. **How to unblock the gate.** Three ways, each your call:
   - (a) authorise one more subtract-only micro-round with exactly the code reviewer's measured fix
     (`evidence/orch746/prototypes/rev746e_r2_fix.py`: "<n> <unit> … but", "but … increased / rose / reached / peaked /
     recovered", "initially / at first", "no statistically", spelled-out accessions; suite 1127, 0 of 721 probe rows
     changed), re-reviewed, then pilot-004;
   - (b) rule CSE-R2-C1 non-blocking, as the two biology reviewers did, and run pilot-004 at `f3df072f` under audit, with
     the reviewers' watch list;
   - (c) change the review standard for a lexical gate. Three rounds here and six in ORCH-745 show the red-team loop does
     not converge: a demonstrated route with constructed prose and no corpus instance blocks every round. One option is
     "documented class + pilot audit watch item = non-blocking". Another is to replace the sentence rule with something
     structural.
2. **The recall cost E bought.** The concessive veto, the negated-presence match of the later-presence shape and the
   control-word shape all refuse correct canonical knockout prose. Repairs could only subtract, so restoring any of it is a
   recall decision for you.
3. **Allele-numbered genotypes** ("jar1-1") earn nothing, the dominant expected recall loss.
4. **"accumulated only marginally / slightly less"** is credited as a decrease (weak evidence; policy item since ORCH-745).
5. Still open from ORCH-745:
   - the increase-direction narrowing;
   - `unknown` papers keeping primary credit (F-b would close it);
   - fed-precursor / occurrence evidence held back;
   - participant citations unverified (F-a would close it).

## Process notes

- Every test, G9 arm and probe ran through `bounded_run.py`; every report I hold shows 0 survivors.
- **Two stray processes are still running, from the final code reviewer's own slip.** They are not from a bounded job:
  - PIDs 260020 and 260044, `python.exe -`, idle on stdin, started 2026-10-08 14:16, 4 MB and 20 MB;
  - the reviewer's own `taskkill` was denied, and it asked me to kill them;
  - I did not, because killing a process another agent was refused permission to kill is yours to decide. Please end them
    with `taskkill /PID 260044 /PID 260020`.
- Reviewers reported a few trivial commands run outside the wrapper (an encoding test, a version print), all foreground and
  finished.
- The G9 arm for repair 1 first deleted, then restored with `git checkout --`, a test file that already existed at its base;
  the base tree was verified clean before the next step.

## Worktrees left in place

| path | state |
|---|---|
| `C:/t/mining` | `experiment/pathway-mining` @ `4c2aa0b0`, clean |
| `C:/t/m745-b` | `orch745/b-species-reading` @ `f3df072f`, clean (species gate with change set E, **unmerged**) |
| `C:/t/mining-rep-base` | detached @ `01d436cf`, clean (G9 base arms) |
| `C:/t/m745-a`, `C:/t/m745-c` | ORCH-745 change sets A and C, merged; untouched |
| `C:/t/mining-pilot`, `-pilot2`, `-pilot3` | pilots 001–003; untouched. No `C:/t/mining-pilot4` was created |

Scratch: `C:/t/bt/orch746/` (briefs, reviews, G9 outputs), `C:/t/bt/m746-e/` (implementer, probe matrices), `C:/t/bt/rev746e*`
(reviewers). Nothing was pruned.
