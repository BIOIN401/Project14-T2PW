# ORCH-733 addendum — are the H / I / J Stage-1 refusals legitimate?

**2026-09-08.** Read-only. No production change, no re-run, no LLM draw. Production frozen at
`045447c8`; `git diff 045447c8 HEAD -- src/` empty. Tool: `evidence/hij_adjudication.py`.
Reports `evidence/g11/ORCH-733/04`–`05`, both `FINAL SURVIVING COUNT : 0`.

**Method.** Two discriminators, in this order of authority:

1. **Gold outranks the runs.** For a pinned gold paper, `mechanistic_relevance` and
   `export_rationale` state what the correct outcome *is*. A run that disagrees is the defect.
2. **Cross-leg, for everything else.** If the same paper produced a canonical pathway — or a PWML —
   in another leg, the refusal cost recoverable content rather than describing the paper.

---

# Answer

> ## **I** is legitimate and should not be touched. **H** is mostly legitimate, and its real signal is an *acquisition* problem, not a schema one. **J** is the only genuinely over-conservative mechanism — and it is structurally wrong while being numerically small.

| class | legs | correct refusal | bounded by gold | defensible loss | unadjudicable |
|---|---:|---:|---:|---:|---:|
| **H** container missing | 19 | 14 | 2 | **1** | 2 |
| **I** zero processes | 14 | **14** | 0 | **0** | 0 |
| **J** guard skipped | 7 | 3 | 0 | **1** | 3 |

---

# I — 14 legs. Legitimate, and this is the product working as designed.

Three papers, and **13 of the 14 legs are the two `context_only` gold cases**:

| paper | legs | gold `mechanistic_relevance` | gold `export_rationale` |
|---|---:|---|---|
| `PMC13231680` | 10 | `context_only` | *"Nothing lipid-A-related is exportable at any level of partiality. **The correct pipeline outcome is an empty pathway plus a rejection reason.**"* |
| `PMC12180156` | 3 | `context_only` | *"With zero heme-biosynthesis reactions recoverable, nothing about heme biosynthesis is exportable."* |
| `PMC12624714` | 1 | not in gold | never yielded a pathway in any leg of any run |

**A pathway with no reactions is the specified answer for these papers, not a failure.** `F-100`
already established that `PMC13231680/strict`'s empty pathway is correct and that the T-105 leg
which *did* produce something was the false positive. `PMC12180156` is the paper at the centre of
the `glycine → heme` fabrication census.

> **This is where my first pass got it backwards and had to be corrected.** The adjudicator
> initially scored "the same paper produced a PWML elsewhere" as evidence that a refusal lost
> biology. For these two papers that is exactly inverted: the PWML elsewhere is the **known
> false positive**. The bug was mechanical — `expected_export` is a *string* (`partial_only`,
> `strict_exportable`), never boolean `False`, so the gold branch never fired at all and all 13
> legs were reported as "recoverable". Gold now outranks the runs in the code.

**Nothing to fix. Any change that made class I produce output would be re-introducing a known
fabrication.**

---

# H — 19 legs. Mostly legitimate; the finding is upstream of extraction.

**11 legs never yielded a pathway in any run of any configuration**, and their titles say why:

```
PMC11946230  a high-throughput screening strategy for …
PMC12898691  cladeoscope-gsa: revealing evolutionary associations …
PMC12971581  Ochrobactrum anthropi causing Fournier's gangrene …     (a case report)
PMC13139079  environmental resistome of culturable gram-negative …   (a survey)
PMC13233763  glycine, the missing link between carbohydrate …
PMC13264790  comprehensive bioinformatics and experimental analysis …
```

These are not pathway papers. Stage 1 returning a payload with no `processes` container is the
correct description of a paper that describes no reactions. **The defect they expose is
acquisition, not extraction** — and it is already on the record: `ORCH-732` § 2 found that *"the
eligibility screen does not select for 'mechanistic pathway paper'. It screens for pathway
**terms**, so an inhibitor paper and an omics paper score well."*

A further **3** legs are gold `context_only` (correct), and **2** are gold `partial_only`
(`PMC12312563`, `PMC12452463`) where a strict non-export is what gold expects.

**One defensible loss:** `PMC12782028/research` in `runs/2026-08-02_2130` — gold
`expected_export: strict_exportable`, failed with *"Extraction boundary failed and could not be
recovered: 2 post-extraction contract error(s)"*. **But it is not a live defect:** that same paper
has since exported `pathway.review_required.pwml` in **eight** consecutive later runs
(2026-08-21 through 2026-09-02), with 3–4 canonical processes every time. Whatever caused the
2026-08-02 boundary failure is not reproducing.

Two `PMC13278307/research` legs are counted unadjudicable — see J.

---

# J — 7 legs. The one over-conservative mechanism, and it is structurally wrong.

```
multi_example_review detected with no selected_example.
Extraction skipped to prevent mixed-pathway output.
```

Two properties, both worth recording independently of how often they bite:

### 1. The guard is all-or-nothing

When it fires, **extraction is skipped entirely** — no scoped retry, no single-example fallback,
no partial output. A paper that could yield one clean example yields nothing. Compare rung 3 of
the extraction ladder, which exists precisely to offer *"narrower section-based extraction"* as an
escalation; the multi-example guard does not reach for it.

### 2. The guard is non-deterministic on the same paper

| paper | guard fired | same paper elsewhere |
|---|---|---|
| `PMC12444477` | 1 leg (`2026-07-27_1623/strict`) | `2026-07-28_0919/strict` → **29 canonical processes**, `pathway.pwml` |
| `PMC13278307` | 3 legs (`2026-07-27_1623/strict`, `2026-07-28_2122/research` **and** `/strict`) | `2026-07-28_0919/strict` → **14 canonical processes**, `pathway.pwml` |

Same paper, same era, same configuration: fires in some runs, not others.

### The defensible loss is one leg, not four

**`PMC12444477` — yes.** Gold: `mechanistic_relevance: core`, *"The enzyme roster and reaction
order are **fully extractable**"*. Skipping extraction entirely on a paper gold calls core and
fully extractable is over-conservative on gold's own authority, independent of any run.

**`PMC13278307` — NOT claimed as a loss, and I will not count it as one.** It is *"an overview of
mobile colistin resistance (mcr) genes"* — a review of resistance genes, not a pathway paper, and
it is **not in gold**, so there is no authority to appeal to. The leg that "succeeded" produced a
**bare `pathway.pwml`** under the pre-`review_required`, pre-`F-179` naming regime — the exact
class `HANDOFF` flags as possibly carrying gold-forbidden content. **A 14-reaction pathway
extracted from an antibiotic-resistance review is plausibly the very mixed-pathway output the
guard exists to prevent.** Whether the guard was right or the extraction was, artifacts cannot
settle; it needs biological review. Recorded as unadjudicable.

The remaining 3 legs (`PMC12935629` ×2, `PMC12326985` ×1) are papers that never yielded a pathway
anywhere — correct refusals.

---

# Recommendation

**No production card is justified by this.** One defensible loss in H (not reproducing for a
month) and one in J. That is below the bar this sprint has repeatedly set, production is re-frozen
under `D-098`, and merge rule 6 forbids weakening a biological gate to increase PWML production —
which is precisely what relaxing the multi-example guard would risk.

Two things worth recording instead, in order of value:

1. **The highest-yield target is acquisition, not extraction.** 11 of 19 H legs are papers that are
   not pathway papers at all. Fixing the eligibility screen so it selects for *mechanistic pathway
   papers* rather than pathway *terms* removes those legs before any LLM budget is spent on them —
   and `ORCH-732` reached the same conclusion independently from the staging side. **This is a
   screening-criteria question for the product owner, not an engineering card.**

2. **`F-194` (registered below): the multi-example guard is all-or-nothing and non-deterministic.**
   Not chartered. If it is ever chartered, the shape is *"a multi-example review with no selected
   example should fall back to scoped single-example extraction, not to no extraction"* — and it
   must be measured against the `PMC13278307` question first, because the guard may be preventing
   exactly the fabrication class `F-179` was written for.
