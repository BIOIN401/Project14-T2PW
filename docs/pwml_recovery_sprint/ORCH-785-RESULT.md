# ORCH-785 RESULT — live re-run of the 3 legs that wrote no PWML in ORCH-784

**Code:** `9a4dc617` (ORCH-786/787/788 on top of ORCH-782/783, D-117 and D-118).
**Runner:** `C:/t/orch785/night3.sh` is `evidence/orch770_night.sh` with the paper loop cut to `chlorophyll wnt_ca sour_taste_v2`, and no gold scoring because those papers have no gold. Pinned worktree `.claude/worktrees/orch785run` @ `9a4dc617`. Strict, 3 draws, 10,800 s per leg. Ran 2026-10-28 17:43-21:45.
**Lifecycle:** G11 `evidence/g11/ORCH-785/03`-`05`. Every leg exited 0 with 0 survivors. The `.env` copy was removed when the runner exited, and the heavy lock was released.
**Summary (untracked):** `runs_papers/ORCH-785/MORNING_SUMMARY.md`.

## Result: 3/3 PWMLs, no whole-pathway refusal

| paper | ORCH-785 PWML coverage | ORCH-784 | ORCH-781 |
|---|---|---|---|
| chlorophyll | **43/82 (52%)**; full count 43/108 | no PWML | 24/82 (29%) |
| wnt_ca | **10/39 (26%)**; full count 10/46 | no PWML | 15/53 (28%) |
| sour_taste_v2 | **6/10 (60%)** | no PWML | 6/11 (55%) |

All three are `review_required`, and none drops a reaction on import. Omissions by KI:
- chlorophyll: KI-008 26, KI-001 15, KI-009 11, **KI-011 7** (41 in ORCH-781), KI-015 6;
- wnt_ca: KI-015 17, **KI-001 7** (24 in ORCH-781), KI-008 7, KI-011 4, KI-006 1, KI-009 1;
- sour taste: KI-001 2, unclassified 2.

## Findings

1. **D-117 § 1 (KI-011) works live.** Chlorophyll coverage went from 29% to 52%, and its KI-011 omissions from 41 to 7.
2. **D-117 § 2 (KI-001 families) cut Wnt/Ca2+'s KI-001 omissions from 24 to 7, but coverage did not rise (26% vs 28%).** The draw has fewer text-record processes (39 vs 53), and KI-015 (other structural quarantine reasons) went up to 17. Not triaged. Draws vary at temperature 0 (CLAUDE.md), so one leg does not settle this.
3. **ORCH-786/787/788 removed every whole-pathway stop that ORCH-784 hit.**

## Not settled

- Wnt/Ca2+ is still at 26%. Its 17 KI-015 rows are the next largest gap and are not triaged.
- Chlorophyll's 26 KI-008 rows are interactions PathWhiz cannot carry: by design, and left out of the headline.
- Open product-owner questions carried over from the RESULT docs:
  - ORCH-786 § 5: running text "STIM1 2" also counts as a slash list;
  - ORCH-787 § 7: the new reason codes still carry stand-in KI labels;
  - ORCH-788: a dropped state with no process omitted does not cap the release at `review_required`.
