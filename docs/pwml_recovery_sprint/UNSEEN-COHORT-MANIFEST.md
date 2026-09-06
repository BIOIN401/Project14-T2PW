# The unseen-paper pilot — FROZEN MANIFEST

**ORCH-724, 2026-09-06.** Ten papers, two modes each, **20 legs**. Topics file:
`topics_unseen_pilot.txt`. Evidence: `evidence/orch724_exclusion_set.json`,
`orch724_cohort_candidates.json`, `orch724_cohort_verification.json`,
`orch724_cohort_citations.json`; bounded-run reports `evidence/g11/ORCH-724/05`–`09`.

## How "unseen" was established

Every candidate id was checked against a **195-id exclusion set** built from five
independent sources, so this is a measurement rather than a claim:

| source | ids |
|---|---|
| RAG acquire cache — ever used as a retrieval SOURCE in development | 167 |
| ever run as a pipeline leg | 38 |
| named in any `topics_*.txt` | 10 |
| curation corpus | 10 |
| gold cases | 10 |
| **union** | **195** |

**None of the ten below appears in that set.** The five *development pathways* — lipid A,
menaquinone, heme, enterobactin and cholesterol biosynthesis — are also all avoided, so
the cohort is unseen at the **pathway** level and not merely at the paper level.

**Selection was not optimized for easy wins.** Candidates came from category-shaped
EuropePMC queries and were chosen for pathway-chemistry content, not for expected score.
Two picks (`PMC12071552`, `PMC11172790`) are deliberately *thin* on explicit conversion
statements, and one (`PMC12326985`) states no pathway chemistry at all by design.

Every paper was verified fetchable **through the project's own `t2pw.rag.acquire.fetch_full_text`**
— the same code path the pilot run uses — so none can fail for an acquisition reason that
selection could have caught. All ten returned substantial text (51 k–133 k chars).

## The ten

| # | category | PMCID | requested pathway | organism | chars | reaction cues |
|---|---|---|---|---|---|---|
| 1 | bacterial | `PMC12071552` | wall teichoic acid D-alanylation | *S. aureus* | 53,363 | 27 |
| 2 | bacterial | `PMC11172790` | pyoverdine biosynthesis | *P. aeruginosa* | 52,127 | 18 |
| 3 | eukaryotic | `PMC12376012` | sphingolipid biosynthesis | *H. sapiens* | 69,320 | 9 |
| 4 | eukaryotic | `PMC7232280` | molybdenum cofactor biosynthesis | *N. crassa* | 51,253 | 18 |
| 5 | long/branched | `PMC11405693` | phenylpropanoid biosynthesis | *A. thaliana* | 82,748 | 9 |
| 6 | long/branched | `PMC8510960` | terpenoid indole alkaloid biosynthesis | *C. roseus* | 87,179 | 5 |
| 7 | cofactor/regulator | `PMC3480714` | cobalamin biosynthesis | *S. enterica* | 51,672 | 12 |
| 8 | cofactor/regulator | `PMC12542839` | riboflavin biosynthesis | *B. subtilis* | 133,074 | 38 |
| 9 | mixed/ambiguous | `PMC13017326` | anthocyanin biosynthesis | *V. vinifera* | 68,059 | 12 |
| 10 | negative/context | `PMC12326985` | siderophore biosynthesis | *A. baumannii* | 110,584 | **0** |

## Citations and inclusion reasons

**1. `PMC12071552`** — Wang L, Xie J, Wang Q, *et al.* *D-Serine Can Modify the Wall
Teichoic Acid of MRSA via the dlt Pathway.* Int J Mol Sci, 2025. PMID 40362350.
doi:10.3390/ijms26094110
*Bacterial cell-envelope modification, entirely unrelated to any development pathway. The
chemistry is real but sparse — "the adenylation reaction catalyzed by SaN315 DltA", DltA
accepting D-Ala or D-Ser as substrate — inside a transcriptomics-heavy paper. A fair test
of whether a small number of genuine steps survives a noisy document.*

**2. `PMC11172790`** — Manko H, Steffan T, Gasser V, Mély Y, Schalk I, Godet J. *PvdL
Orchestrates the Assembly of the Nonribosomal Peptide Synthetases Involved in Pyoverdine
Biosynthesis in Pseudomonas aeruginosa.* Int J Mol Sci, 2024. PMID 38892200.
doi:10.3390/ijms25116013
*NRPS assembly-line chemistry: "PvdL catalyzes the attachment of a fatty acid to a
glutamate residue, followed by the addition of L-tyrosine and L-2,4-diaminobutyrate."
**Disclosed adjacency:** pyoverdine is a siderophore, as enterobactin is, and enterobactin
is a development pathway. The organism, enzymes, chemistry and assembly logic are all
different, and the paper is retained because a modular NRPS is a genuinely hard case — but
a reader comparing to enterobactin results should know the class is shared.*

**3. `PMC12376012`** — Mahawar U, Wattenberg B. *Intricate Regulation of Sphingolipid
Biosynthesis: An In-Depth Look Into ORMDL-Mediated Regulation of Serine
Palmitoyltransferase.* BioEssays, 2025. PMID 40548458. doi:10.1002/bies.70036
*Eukaryotic, human, and regulator-dense: the pathway is inseparable from its ORMDL
regulation. Tests whether regulators are represented without being mistaken for reaction
participants.*

**4. `PMC7232280`** — Wajmann S, Hercher TW, Buchmeier S, Hänsch R, Mendel RR, *et al.*
*The First Step of Neurospora crassa Molybdenum Cofactor Biosynthesis: Regulatory Aspects
under N-Derepressing and Nitrate-Inducing Conditions.* Microorganisms, 2020. PMID 32272807.
doi:10.3390/microorganisms8040534
*Eukaryotic (filamentous fungus) — a clade absent from the development corpus, which is
all bacteria and human. Highest reaction-cue density of the eukaryotic picks.*

**5. `PMC11405693`** — Knosp S, Kriegshauser L, Tatsumi K, *et al.* *An ancient role for
CYP73 monooxygenases in phenylpropanoid biosynthesis and embryophyte development.* EMBO J,
2024. PMID 39090438. doi:10.1038/s44318-024-00181-7
*Plant — a third kingdom, absent from development. Phenylpropanoid biosynthesis is long and
branches into lignin, flavonoid and ester arms, so it exercises branch retention.*

**6. `PMC8510960`** — Liu Y, Patra B, Singh SK, *et al.* *Terpenoid indole alkaloid
biosynthesis in Catharanthus roseus: effects and prospects of environmental factors in
metabolic engineering.* Biotechnol Lett, 2021. PMID 34564757. doi:10.1007/s10529-021-03179-x
*The canonical long, branched, multi-compartment specialized-metabolite pathway (~30 steps
across several cell types). The hardest completeness case in the cohort.*

**7. `PMC3480714`** — Deery E, Schroeder S, Lawrence AD, *et al.* *An enzyme-trap approach
allows isolation of intermediates in cobalamin biosynthesis.* Nat Chem Biol, 2012.
PMID 23042036. doi:10.1038/nchembio.1086
*Cofactor biosynthesis stated as explicit intermediate-by-intermediate chemistry — the
paper's whole method is isolating pathway intermediates. Dense, unambiguous reaction text.*

**8. `PMC12542839`** — Ruchala J, Najdecka A, Wojdyla D, Liu W, Sibirny A. *Regulation of
Riboflavin Biosynthesis in Microorganisms and Construction of the Advanced Overproducers of
This Vitamin.* Int J Mol Sci, 2025. PMID 40650021. doi:10.3390/ijms26136243
*The most reaction-dense paper in the cohort (38 cues, 133 k chars) and heavily
regulator-focused. Tests scale and the regulator/participant boundary together.*

**9. `PMC13017326`** — Wang Y, Duan X, Xu R, Huang H. *Molecular mechanisms of phytohormone
ABA-regulated anthocyanin biosynthesis in grape berry: an epigenetic dual-gating hypothesis
within a five-layer regulatory framework.* Front Mol Biosci, 2026. PMID 41907139.
doi:10.3389/fmolb.2026.1761085
*Mixed/ambiguous by construction: a hormone-signalling and epigenetics paper wrapped around
a biosynthetic pathway, and explicitly a **hypothesis** paper. Scope resolution, not
chemistry, is what is being tested.*

**10. `PMC12326985`** — Kubin CJ, Garzia C, Uhlemann A-C. *Acinetobacter baumannii treatment
strategies: a review of therapeutic challenges and considerations.* Antimicrob Agents
Chemother, 2025. PMID 40631987. doi:10.1128/aac.01063-24
***The negative control.*** *A clinical therapeutics review: **0** reaction cues over 110,584
characters. It names "lipid A biosynthesis" exactly once, as a colistin-resistance
mechanism — a deliberate trap, since lipid A is a development pathway and the words are
present without any chemistry behind them. **The correct outcome is little or nothing.** A
rich pathway here is a fabrication finding, not a success.*

## Run protocol

- **Strict once and research once per paper. 20 legs. No favourable-draw reruns.** A leg is
  re-run only for an objectively invalid infrastructure execution (wrapper timeout, crash
  before any stage, survivor-process contamination) — never because the biology
  disappointed.
- Runs on the **frozen production SHA**, under the bounded runner, one heavy job at a time.
- Existing Phoenix / evaluation artifacts are captured where already supported. **No new
  observability infrastructure is built for this run.**

## Staging result, and one expected refusal

`scripts/batch_run.py --stage-only` against `topics_unseen_pilot.txt`:

```
acquisition funnel: requested 10, examined 10, eligible 10, ineligible 0,
                    no_full_text 0, accepted 10
planned 10 paper(s) x 2 mode(s) = 20 run(s); 0 candidate(s) skipped
```

**All ten papers stage. Twenty legs planned. Nothing skipped.** Evidence:
`evidence/g11/ORCH-724/10-pilot-stage.json`.

### `bench_acceptance.py --verify-plan` REFUSES this plan, and that is correct

```
verdict: REFUSED
  MISSING  PMC12444477 … (the ten gold papers)
  EXTRA    PMC11172790 … (the ten pilot papers)
```

`--verify-plan` exists to stop a **pinned acceptance run** from executing against the
wrong papers: it compares the plan to the **gold set**, which is by construction the ten
*development* papers. The unseen pilot is deliberately none of them, so a refusal is the
guard doing its job — **not a defect, and not a reason to change the topics file.**

The consequence is stated plainly so nobody looks for it later: **this pilot cannot be
scored by `bench_acceptance.py`, because no gold exists for these ten papers and none is
being invented.** Its outputs are judged on (a) the structural/product table that
`evidence/orch724_pilot_summary.py` produces — PWML generated, release state, reaction
count, graph validity, referential integrity, species, unprovenanced rows — and (b)
**human biological review**, which is the point of the exercise. A pilot leg is never
recorded as a benchmark pass or fail.

---

## THE PWML DENOMINATOR IS 10, NOT 20 — read this before quoting any pilot number

**Measured over the project's entire committed history, not assumed:**

| mode | legs | produced a `.pwml` |
|---|---|---|
| research | **153** | **0** |
| strict | 156 | 34 |

**Research mode has never produced a PWML, in 153 legs.** It is not a failure mode and not
a regression: `bench/acceptance.py:125` defines `_STRICT_DELIVERABLES` as
`("pathway.pwml", "pathway.review_required.pwml")`, and only a strict leg calls the
exporter (`batch/driver.py:68`). **Research mode is diagnostic** — it contributes reaction
extraction and RAG behaviour, not a PWML artifact.

### Consequence for the pilot's tables

The pilot runs **20 legs over 10 papers**, but only the **10 strict legs** can produce a
PWML. Therefore:

- **"PWML generated" is `n / 10`, never `n / 20`.** Quoting `n / 20` halves the reported
  rate through a category error, and it is exactly the kind of figure that survives into an
  abstract uncorrected.
- **Research legs must never be counted as PWML failures.** A research leg with no `.pwml`
  is behaving exactly as designed.
- The two modes answer different questions and their denominators are **not summed** — the
  same discipline `F-177` established for canonical vs fallback payloads and
  `rd093_rag_metrics.py` for truncated vs untruncated legs.

### The outcome classes are also not interchangeable

A leg can end without a PWML for reasons that mean completely different things, and the
report keeps them apart:

| class | what it means | is it a product failure? |
|---|---|---|
| `scope_conflict` | Stage 0 read a pathway string that disagrees with the requested one — often the *same* pathway phrased differently | **No** — an instrument/agreement artifact, not a failure to recover chemistry |
| contract `FAIL` | a gate blocked export, e.g. a referential-integrity violation | **The gate working.** No broken graph shipped. Counts against completeness, not against correctness |
| `review_required` PWML | a PWML was produced and needs human review | **A SUCCESS** if the biology is useful — the charter is explicit |
| research leg, no PWML | by design | **No** |

**`release_ready` is not the success metric and `review_required` is not a failure.**
