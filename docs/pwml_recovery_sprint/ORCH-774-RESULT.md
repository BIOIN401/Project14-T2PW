# ORCH-774 — the D-114 confirmation run (6 papers, strict, 3 draws). RESULT

**Sub-orchestrator record.** One live run, as chartered: enterobactin, chlorophyll, `sour_taste_v2` (*Mus musculus*), `wnt_pcp`,
`wnt_ca`, menaquinone; strict, 3 draws, 10,800 s per leg. Runner: the committed ORCH-770 runner (`evidence/orch770_night.sh`, task id
argument from ORCH-772, `--src` from ORCH-772b), started from worktree `.claude/worktrees/orch774` pinned at **`329d81b2`** (= `main`:
ORCH-771, 772, 772b, 773, 769d merged; final integration Chunk D 188/188, SMOKE 511, `ORCH-774i`). `.env` copied from the primary
(draws 3; budgets 128000, extraction 64000). 2026-10-25 22:30 → 2026-10-26 05:22. Every leg report `FINAL SURVIVING COUNT : 0`
(`evidence/g11/ORCH-774/01-08`; `g11_evidence.py check --task ORCH-774`: 8 artifacts, 0 non-compliant).

**The morning summary** `runs_papers/ORCH-774/MORNING_SUMMARY.md` was written by the runner itself at 05:22 (`summary_exit=0`). The
runner calls the summary script from the PRIMARY checkout (`EV="$P/..."`), so it ran the merged ORCH-774s script (sprint `4286854f`,
script from `orch774s` @ `66061e06`) against the pinned tree's `reaction_evidence` (`--src` = the worktree's `src` at `329d81b2`). The
pipeline under test stayed pinned; only the reporting script was newer. No regeneration was needed.

## Per paper (code `329d81b2` for every leg)

| paper | PWML | coverage, headline (KI-008 excluded) | full count | would import | dropped on import | min | omitted, by KI |
|---|---|---|---|---|---|---|---|
| enterobactin | **yes** | **8/10 (80%)** | 8/17 (47%) | 8/10 | 0 | 38 | KI-008 7, KI-001 2 |
| chlorophyll | **yes** (none in ORCH-767) | 27/86 (31%) | 27/87 (31%) | 26/86 (30%) | 1 | 129 | KI-011 42, KI-001 16, KI-008 1, KI-015 1 |
| sour taste (*Mus musculus*) | **yes** | 10/15 (67%) | 10/15 (67%) | 10/15 | 0 | 22 | KI-001 5 |
| WNT PCP | **no** | 0/65 (0%) | 0/65 | — | — | 68 | whole pathway: `no_defensible_reaction_support` (KI-022); rows KI-001 40, KI-022 16, unclassified 9 |
| Wnt/Ca2+ | **yes** (none in ORCH-767) | 8/49 (16%) | 8/53 (15%) | 8/49 | 0 | 105 | KI-001 31, KI-015 8, KI-008 4, unclassified 2 |
| menaquinone | **yes** | **13/13 (100%)** | 13/13 (100%) | 13/13 | 0 | 49 | none |

**5 of 6 papers write a PWML** (ORCH-767: 2 of 6), all `review_required`. **2 of 6 reach the 75% target** (enterobactin, menaquinone).

## Gold (eval-v2 `score_leg`, `evidence/orch774_gold_accuracy.txt`)

| paper / level | ORCH-774 | ORCH-767 |
|---|---|---|
| enterobactin Stage 1 / final / PWML | P 1.000 R 1.000 F1 1.000 (5/5) at all three | 5/5, 5/5, **not written** |
| menaquinone Stage 1 / final / PWML | P 1.000 R 1.000 F1 1.000 (9/9) at all three | 9/9, P 0.909 R 1.000, P 0.909 R 1.000 |

## LLM health (summary § LLM health)

241 model attempts; empty `finish_reason=length` only in Stage 1 (2/61); 1 provider `finish_reason=error` (Stage 1, 143 s); no call
of 1,200 s or longer. Down from ORCH-767 (Stage 2 6/34 empty, three 20+ minute calls).

## Process record

- ORCH-774s (summary additions) merged during the run; its merged-tip gates were taken in a PAUSE gap between legs (01:18–01:19,
  focused 64, SMOKE 511). A first gap at 23:09 ran no test (gate worktree checkout clash, malformed task id) and cost the run ~30 s.
- Orphan sweeps after the run: no sprint `python.exe` alive; heavy lock and PAUSE absent. No process killed.
- ORCH-775 (main's read-only investigator) ran small probes without the lock during the run, as announced.

## Findings

1. **F-1. The whole D-114 chain works live:** per-row omission (ORCH-773) turned two whole-pathway failures of ORCH-767 into PWMLs
   (chlorophyll, Wnt/Ca2+), and the text and summary list every omission with a KI id.
2. **F-2. KI-001 (protein with no UniProt/DrugBank id) is the largest omission cause tonight** (2 + 16 + 5 + 40 + 31 = 94 rows), then
   KI-011 (a gene and a protein share one bare symbol, 42, chlorophyll). Both originate upstream (Stage 3 / Stage 6) and cannot be
   measured on the replay corpus (ORCH-768 F-6, KI-058).
3. **F-3. WNT PCP fails on a pathway-level requirement** (`no_defensible_reaction_support`, KI-022: none of its 3 reactions carries
   reaction-level support), which ORCH-773 classifies as whole-pathway by design.
4. **F-4. Coverage is low on the regulatory papers** (chlorophyll 31%, Wnt/Ca2+ 16%), and those are the papers the omissions hit.

## Not settled (for the product owner)

1. **WNT PCP / KI-022.** (a) Keep: no PWML when no reaction has reaction-level support (PRODUCT_CONTRACT 2–3). (b) Treat the
   unsupported reactions as per-row omissions and export the interactions only (D-113 § 3). That is a gate-classification change and
   needs a D-entry; it would also move the 8 corpus legs that fail this way.
2. **Enzyme-less `spontaneous=false` reactions** (ORCH-773 r1 N1, ORCH-775 investigating): 1 tonight (chlorophyll), shown as
   "would import" in the summary.
3. **The KI-001 and KI-011 losses** are the levers for the 75% target on chlorophyll, WNT PCP and Wnt/Ca2+. Both are upstream, so a
   fix is measurable only live, unless per-leg recording of the UniProt/PathBank answers is built (ORCH-768 § 11 item 1 (a)).
4. Carried open items: ORCH-773 r1 N2 (release cap on admission-time quarantines, live), N3 (AMENDMENT 1 § 2 at admission time);
   ORCH-769d `/-` defect and N1; ORCH-772b F-3, the "audit and DB mapping ..." row and `(budget_exhausted)`; KI-065's second
   zero-process leg; KI-023's workaround text; the closure-prune codes uncatalogued (ORCH-772 F-3).
