# ORCH-796 RESULT: all six papers re-run on the ORCH-790..795 / 794b / 780 tip

**Runs.** Same runner as before (`evidence/orch770_night.sh`): strict mode, 3 draws, pinned worktree `.claude/worktrees/orch796run`, `.env` copied in for the run and deleted after it.
- **Pass 1**, 11:53-15:21, code `55fd55ed` (ORCH-790..795): all six papers.
- **Pass 2**, 16:32-21:45, code `774d0087` (plus ORCH-794b and ORCH-780): chlorophyll, wnt_pcp and wnt_ca again (`C:/t/orch79x/rerun796.sh`).
- Every leg exited with `FINAL SURVIVING COUNT : 0`. G11 reports are in `evidence/g11/ORCH-796/07`-`17`, and `g11_evidence.py check --task ORCH-796` is clean: 19 artifacts, 0 non-compliant (`18`).
- Two report numbers were used twice: 07 and 08 belong both to an integration-gate report and to a leg report. This happened because the integration worktree and the primary checkout allocated report numbers independently. The labels differ and the check accepts both.

Per-paper figures come from `runs_papers/ORCH-796/MORNING_SUMMARY.md` (written by the runner; latest leg per paper) and `REVIEW/<paper>/strict/reaction_evidence.txt`. The "before" figures are the ORCH-784/785 legs as read in `ORCH-789-RESULT.md` § 1.

Merge rule 5 was waived: no independent review round was run (D-119 §§ 4-5, PO instruction).

## 1. Coverage, before → after (headline, D-114 AMENDMENT 2)

| paper | ORCH-784/785 | ORCH-796 | full count | code | note |
|---|---|---|---|---|---|
| chlorophyll | 43/82 (52%) | **52/58 (90%)** | 52/62 (84%) | `774d0087` | pass 1 wrote no PWML (the ORCH-794 regression, fixed by 794b) |
| wnt_ca | 10/39 (26%) | **22/41 (54%)** | 22/61 (36%) | `774d0087` | pass 1 failed at Stage 1 (chunk 4 invalid JSON, LLM-side); no EIF3G in the PWML |
| wnt_pcp | 5/64 (8%) | **9/26 (35%)** | 9/58 (16%) | `774d0087` | pass 1 wrote no PWML: the KI-022 whole-pathway refusal, fixed by ORCH-780 |
| enterobactin | 8/12 (67%) | **7/8 (88%)** | 7/9 (78%) | `55fd55ed` | |
| menaquinone | 22/25 (88%) | **14/14 (100%)** | 14/14 | `55fd55ed` | smaller draw (10 reactions) |
| sour_taste_v2 | 6/10 (60%) | **8/12 (67%)** | 8/13 (62%) | `55fd55ed` | |

- All six papers write a PWML, and every PWML imports in full: "would import" equals the drawn count, and 0 processes are dropped on import.
- Four of the six papers are at or above 75%.
- Draws differ between runs (CLAUDE.md), so each row is one draw, not a controlled A/B.
- Gold (eval-v2): menaquinone PWML scores R 1.000 (9/9). Enterobactin PWML scores R 0.600 (3/5), down from 5/5 in earlier runs. The loss is in Stage-1 recall (2/5 at Stage 1), so it is a draw effect and LLM-side.

## 2. What is left, by cause (from `MORNING_SUMMARY.md` § Omitted)

### 2a. Deterministic or ruling-bound (not LLM-side; still open)

| cause | chl | wnt_ca | wnt_pcp | sour | what it needs |
|---|---|---|---|---|---|
| Alias identity UniProt's gene fields cannot decide (PIF1, TAK1, NARF, c-Jun, Kir2.1) | 2 | 8 | 4 | 1 | PO ruling: `ORCH-793-RESULT.md` § 7 |
| Family names still outside the term list (InsP3R/IP3R, SERCA, calmodulin, Smurf, Par6, Dvl, MRLC, DIA1, Profilin, Wnt) | 1 (GBP) | ≈12 | ≈18 | - | PO list: ORCH-789 § 8 Q3; ORCH-794 Q2 (InsP3R) |
| KI-011 gene/protein role still ambiguous (PIF1/4/5 → CHLH, FLU, FHY3) | 5 | - | - | - | `_GENE_ROLE_CUE` wording (ORCH-789 C8-iii, PO) |
| KI-009 bare endpoint (DAAM/Rac1/Celsr1; STIM1/2 → Orai1/2/3) | - | 3 | 5 | - | a diagnosis lane (the ORCH-791 order did not cover these shapes) |
| CaMKII holoenzyme with no components (KI-015) | - | 3 | - | - | PO choice: ORCH-790 Q1 |

The counts with ≈ are hand counts from the omitted list.

### 2b. LLM-side causes, ranked (only better extraction can fix these)

| # | cause | chl | wnt_ca | wnt_pcp | entero | mena | sour | total |
|---|---|---|---|---|---|---|---|---|
| 1 | The relationship word is unsigned ("binds", "interacts", "transports", "associates"), so PathWhiz cannot carry the row (KI-008; outside the headline) | 2 | 8 | 19 | 1 | 0 | 0 | 30 |
| 2 | Paraphrase twins: the same fact written 2-3 times ("Wnt binds Frizzled" ×2, Dvl/Dishevelled ×2 with each of Par6, Smurf, Inversin and DAAM, "Wnt5a binds Fzd2" ×3, "IP3-induced Ca2+ release" ×3, JNK/ROCK activates *and* phosphorylates). This inflates both numerator and denominator (KI-039) | - | ≈8 | ≈15 | - | - | - | ≈23 |
| 3 | Transports or channel conduction extracted as same-entity "reactions" ("H+ influx via OTOP1", "K+ efflux via Kir2.1", "Ca2+ influx via VGCC", "5-HT release via SNARE"; F-2, KI-015) | - | - | - | - | - | 4 | 4 |
| 4 | Stage-1 failures and recall: wnt_ca pass 1 lost the whole leg to invalid JSON in chunk 4; enterobactin Stage-1 recall was 2/5 against gold this draw | - | (leg) | - | 2 gold steps | - | - | - |
| 5 | Lumped steps: "Combined enterobactin assembly from 2,3-DHB, ATP and L-serine" (KI-012), and TOC1 → "PIFs" left undeclared at Stage 1 | 1 | - | - | 1 | - | - | 2 |
| 6 | An identity invented by the audit (MenA "based on function", ORCH-792 § 9); correctly refused downstream | - | - | - | - | (0 rows this draw) | - | - |

Ranking by rows recoverable: causes 1 and 2 dominate wnt_pcp and wnt_ca. Prompting for signed verbs and for one row per fact would move these two papers the most. Cause 3 caps sour taste.

## 3. Failures and dead ends this task

- **Pass-1 chlorophyll regression (ORCH-794).** The kept row was spelled `tRNA^Glu`, while the references said `tRNAGlu`. The offline replay missed it because it rebuilt the rows instead of using the leg's own spelling. Fixed in ORCH-794b (`a2762049`), with a G9 test and a leg replay. Lesson: a replay must use the leg's own rows, never rebuilt ones.
- **D-116 had never been merged.** ORCH-780 had been paused at the ORCH-781 usage limit and not picked up since, which cost the pass-1 wnt_pcp PWML. It was ported and merged (D-119 § 5).
- The ORCH-768 corpus cannot show Stage-6 fixes, because Stage 6 is not replayed. ORCH-790..795 left the PWML pass rate unchanged at 72/91. Only ORCH-780 moved it (72 → 79/91).

## 4. Not settled (for the PO)

1. The ORCH-793 alias rows: options in `ORCH-793-RESULT.md` § 7. Cost of doing nothing: ≈15 rows across 4 papers.
2. Extend the family list again (§ 2a row 2), or accept these rows as KI-001.
3. Charged tRNA: nucleic acid (the current rule) or compound (as PathBank models it)? ORCH-794 Q1.
4. Quarantine and IR still compare names by different keys wherever two names differ only by punctuation (ORCH-794-RESULT § 11). One small deterministic lane; it can turn a whole pathway into a refusal, as it did in pass 1.
5. F-2: count same-entity duplicates as merged duplicates? This would take sour taste from 8/12 to 8/8 (ORCH-795 Q3).
