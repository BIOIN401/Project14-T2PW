# ORCH-751 — the scored cohort, and a correction to the ensembling claim

**Status:** delivered, uncommitted. Evidence `evidence/g11/ORCH-751/02`–`08`, outputs
`C:/t/cohort/`. Base `c9265fb8`. 21 draws, 7 conditions, 3 draws each, control prompt
(`r2_01_control_v03`), 64,000 output tokens, temperature 0.

**Headline:** ensembling is **not** the reliable win ORCH-747 § 12 reported. It recovers
reactions that are missed *randomly* and does nothing for reactions missed *systematically*
— and on one paper the pooled row count inflated 2.3x without the chemistry to justify it.

---

## 1. The cohort, and what it can support

| condition | paper | gold | why it is here |
|---|---|---|---|
| enterobactin | PMC12096016 | **certified, 5 reactions** | scored |
| menaquinone | PMC12421875 | **certified, 9 reactions** | scored |
| Ara operon (PW701720) | PMC11983282 | none | replaces the unfetchable 1981 scan |
| sour taste (PW739181) | PMC7059786 | none | the only 4/4 coverage in the cohort |
| WNT PCP (PW495535) | PMC10425814 | none | 19/19 term coverage |
| Wnt/Ca2+ (PW488575) | PMC10425814 | none | **same paper as above — not independent** |
| chlorophyll (PW766996) | PMC5130987 | none | 12/14 coverage |

Three curated pathways have **no usable paper at all** and are absent: *M. tuberculosis*
and olfactory OR1A1 were coverage-limited (6/18 and 2/10), and calcium signalling had no
honest candidate. See `ORCH-751` cohort notes and § 5.

**Why so few primaries.** Only **13 of 36** papers the curators cited can be fetched:
14 return HTTP 500 on the Europe PMC full-text endpoint, 6 have no PMCID, 3 are not
indexed. Verified independently: `isOpenAccess=Y` → 200 and `=N` → 500, with no exception
in 40+ fetches, including `inEPMC=Y` cases. Six of seven picks are therefore reviews. **A
review that restates another paper cannot corroborate it**, which caps what this cohort can
establish.

---

## 2. Results

| condition | reactions per draw | union | interactions per draw | rows without evidence |
|---|---|---|---|---|
| enterobactin | 4, 5, 5 | 6 | 1, 3, 2 | 0 |
| menaquinone | 2, 9, 7 | 9 | 1, 1, 1 | 0 |
| chlorophyll | 16, 19, 18 | **41** | 0, 7, 16 | 0 |
| WNT PCP | 1, 0, 0 | 1 | 9, 11, **0** | 0 |
| Wnt/Ca2+ | 0, 0, 1 | 1 | 0, 4, 5 | 0 |
| Ara operon | 0, 0, 0 | 0 | 12, 12, 11 | 0 |
| sour taste | 0, 0, 0 | 0 | 6, 1, 9 | 0 |

### 2.1 Recall against certified gold, per draw and pooled

| | per draw | union of 3 | gold |
|---|---|---|---|
| enterobactin | 4, 4, 4 | **4** (0.800) | 5 |
| menaquinone | 2, 9, 7 | **9** (1.000) | 9 |

**Ensembling worked on one and not the other, and the difference is the failure mode.**
On menaquinone the draws missed *different* reactions, so pooling recovered everything, as
ORCH-747 § 12 found. On enterobactin **all three draws matched the same four** and missed
the same fifth: a systematic miss, which no amount of pooling fixes. The per-draw counts
4/5/5 conceal this — the extra row in draws 2 and 3 is not the missing gold reaction.

ORCH-747 § 12 reported union 5/5 on enterobactin from 5 draws. Both are true; which you get
depends on whether the miss is random on that run. **The claim "pooling reaches recall
1.000" is therefore withdrawn and replaced with: pooling recovers randomly-missed
reactions only.**

### 2.2 The pooled row count inflates, and the dedup key is why

Chlorophyll: draws of 16, 19 and 18 reactions pool to **41 distinct keys**. Overlap between
draws, measured:

| pair | shared keys |
|---|---|
| draw 1 vs draw 2 | 1 |
| draw 1 vs draw 3 | 11 |
| draw 2 vs draw 3 | 1 |
| all three | 1 |

Cause: the dedup key is the exact set of input and output names, so
`A + NADPH -> B + NADP+` and `A -> B` are **different keys for the same reaction**. Draw 2
listed cofactors where draws 1 and 3 did not. The union is largely the same chemistry keyed
three ways, not 41 reactions.

This is a defect in the **ORCH-750 ensembling implementation**, which keys reactions the
same way (`matcher.strict_key` over both sides). The evaluator does not have this problem:
it scores cofactors separately and never fails a match on them (`docs/evaluation_v2/README.md`
§ 3), which is why the gold-scored menaquinone union is right at 9/9 while the raw row count
inflates. **Recommended fix: dedup on defining participants only, with cofactors merged
onto the surviving row.** Until then, N>1 will inflate reaction rows on any cofactor-rich
pathway, and chlorophyll biosynthesis is the worst case in this cohort.

### 2.3 Regulatory papers still yield no reactions, and one draw collapsed

Ara operon and sour taste produced 0 reactions in all 6 draws, consistent with ORCH-747
§ 10.3. Their content is interactions: 11–12 per draw for araC, 1–9 for sour taste.

**WNT PCP draw 3 returned 0 interactions and 0 reactions** where draws 1 and 2 returned 9
and 11. An empty extraction, not an error — the leg completed. On a 199,748-character paper
this is the § 6.2 budget failure mode in a milder form and should be watched.

### 2.4 The evidence result holds

**Zero rows without an evidence quote, across all 21 draws.** Now 125 draws across every
round of this work, every prompt, every paper type, with no exception.

---

## 3. What this changes

1. **ORCH-747 § 12.2 and § 12.4 item 2 are amended, not withdrawn.** Pooling still gives
   the largest measured gain available, but it is conditional: it fixes random misses, not
   systematic ones. Expected benefit should be stated per paper, not as a constant.
2. **ORCH-750 needs a dedup fix** before N>1 is used in production (§ 2.2).
3. **The cohort cannot carry reaction-level claims beyond the two certified papers.** Five
   of seven conditions have no gold; two of those five share one paper.
4. **A 2.3x row inflation is invisible without gold.** On chlorophyll there is no way to
   tell recovery from duplication. Any pooled run on an ungolded paper should report the
   union-to-mean ratio so the inflation is at least visible.

## 4. Open for the product owner

Carried from ORCH-747 § 7, still unresolved, plus:

5. **Bare-protein endpoints** (ORCH-749): spelling is fixed and exportable interactions went
   from 0 to single digits, but most typed rows are still withheld because PathWhiz cannot
   draw a bare protein. Wrapping them as single-protein complexes, as enzymes and
   transporters already are, is an export-representation decision.
6. **Three curated pathways have no usable paper.** Accept a coverage-limited paper, pay for
   access to a closed primary, or drop them from the cohort.

## 5. Not done

- No gold exists for the five new cohort papers; only the two inherited certified cases are
  scored. Building paper gold for the cohort is the obvious next investment.
- The ensembling code path itself was not exercised end to end here: the union was computed
  offline from 3 independent draws, which is equivalent in result and keeps the measurement
  independent of the new code being correct.
- Re-pinning the 2 moved IR digests is still owed (ORCH-747 § 6, ORCH-749).
