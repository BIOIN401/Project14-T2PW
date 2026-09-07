# ORCH-724 — FINAL REPORT. The last bounded product pass.

**Lead Orchestrator, 2026-09-06/07.** Frozen production `c4a97f60`; report tip later.
`local = origin/ = git ls-remote` verified throughout. `main` never written.

---

# 1. `F-183` — CLASSIFIED. No change was needed.

**Not a live production export bypass.** The call graph is closed and was read, not inferred.

`validate_pre_export` has exactly ONE production call site — `streamlit_app.py:4788` — and it
stands ahead of `DeterministicPwmlBuilder` (`:5003`) **inside the same function**,
`run_pwml_export` (`:4610`). The second path, `writer.run_pwml_pipeline_export`
(`writer.py:2642`), is reachable only from `scripts/run_pwml.py` (`README.md:40`) and from
tests; no batch run, benchmark, app export or script invokes it, over `src/`, `scripts/`,
`tests/` and every `.ps1`/`.bat`/`.sh`/`.yml`/`Makefile`. `batch/driver.py` drives *the real
Streamlit app* through `AppTest`, so every product PWML — all twenty pilot legs included — is
gated.

The CLI is **not protected by the F-179 rule**; what makes it not a bypass is that **nothing
in the product reaches it.** It stays OPEN as a hygiene precondition only.

**Whether any change was needed: NO.** F-183 did not delay the project.

---

# 2. The admission audit

**Credible curated positives examined: all 41.** Every curated core reaction's quote was
located verbatim in that paper's stored full text using the production folding. **None was
excluded**, so the decision set is credible.

**Rejection reasons sampled:** every reason code in the committed archive, stratified, plus
all curated matches. **2,439 curated-matching rejections over 88 untruncated legs.**

**Correct vs overstrict.** The headline overstates the problem, and the correction is the
finding: at the product-relevant unit — *(run, leg, curated reaction)* — only **95 pairs were
never admitted**, and the blocking reasons are **spread across five codes**:

| minimal blocking reason | count | verdict |
|---|---|---|
| `evidence_relation_disagrees_with_claim` | 27 | MIXED — real disagreements plus two distinct parsing artifacts |
| `no_local_evidence_span` | 20 | **CORRECT** — no span, no admission. `F-179` working |
| `evidence_relation_roles_unassignable` | 16 | **OVERSTRICT** |
| `candidate_type_cannot_fill_gap` | 11 | **CORRECT** — a reaction cannot fill an enzyme-identity or compartment gap |
| `evidence_states_no_reaction_relation` | 8 | CORRECT on inspection |

**Most of the rejecting is correct.** One code is a demonstrated over-rejection with a single
general cause: as sole blocker, `evidence_relation_roles_unassignable` accounts for **294
rejections across only 12 distinct spans, 71.8% of them one construction** — a nominalized
`conversion of X to Y` whose catalyst follows in an attached `catalyzed by …`. No template
read it; `parse_span_relation` returned `None` on all of them. The reactions lost include
**the first step of the enterobactin pathway**.

**Was a production change justified? YES — exactly one**, and it is the only one taken.

---

# 3. Production

**The change.** `C-118`, authorized by `D-095`, merged `a5ffdbeb --no-ff`. ONE template
appended to `_EXTRA_PROSE_PATTERNS` in `src/t2pw/rag/admission.py`, following the `C-061`
precedent. `_PROSE_PATTERNS` **byte-identical**, sha256 `fea7cc2dd2393224`, derived
independently by both the orchestrator and the reviewer. `D-095a` additionally permitted a
**test-only** re-pin of the `C-061` preservation golden.

**`F-179` regression.** Byte-identical to the committed baseline: 80 supported / 24
indeterminate / 11 no-defensible-core, the same 5 previously-exported-now-blocked legs,
**zero newly-blocked legs on `PMC12096016` or `PMC12782028`.** *Recorded honestly:* that
instrument replays payloads through `reaction_support.py`, which `C-118` does not touch, so
it is **insensitive to this change by construction** and is not the evidence that carries the
merge.

**What does carry it.** The reviewer regenerated the 2,000-span golden at both revisions
(115/1885/5 → 249/1751/10, byte-equal to the committed JSONs) and showed by direct
entry-level diff that all **134 moved entries** read exactly `chorismate → isochorismate`
with `EntC`, **none from the accepted bucket, none `ok → refused`, none displacing an earlier
template.** `F-179` was proved by **reordering**: with the new template forced to position
one, the glycine/succinyl-CoA condensation and eleven sibling nominalizations still return
`None`.

**The sharpest question of the wave, and its answer.** The reviewer noted the block is on the
*head noun*, not the chemistry. The corpus **does** contain
*"the conversion of glycine and succinyl-CoA into aminolaevulinic acid"*. At the tip it still
parses to `None` (no attached catalyst) and `glycine → heme` against it stays refused; it
appears in **0 of 14,919** persisted evidence spans. And on a synthetic sentence that *does*
carry an attached catalyst, the template reads the one-step ALA reaction the sentence states
and **still refuses** `glycine → heme`. **The F-179 guarantee rests on the gate comparing the
claim to the relation, not on the template being unable to reach glycine.**

**Reviewer verdict: `APPROVE`**, with five findings. Post-merge SMOKE **508 passed**, exit 0,
zero survivors, pin verdict committed.

**Final frozen SHA: `c4a97f60`** — plus, and this matters, `src/t2pw/app/streamlit_app.py`
working-tree sha256 **`47e4fafa…`**, which differs from its HEAD blob by ~8 KB. It is tracked,
modified and never committed; every historical benchmark ran against it too. **The
reproducible identity of this pilot is (SHA + that file hash), not the SHA alone.** Committing
it is a production change and is not authorized under `D-090`. **This must be resolved before
publication.**

---

# 4. The unseen pilot

**Manifest.** Ten papers, frozen in `UNSEEN-COHORT-MANIFEST.md` / `topics_unseen_pilot.txt`.
"Unseen" was **measured**, not asserted: every id checked against a **195-id exclusion set**
from five independent sources (RAG acquire cache 167, ever-run-as-a-leg 38, topics files 10,
curation 10, gold 10). None appears. The five development *pathways* are avoided too, and the
cohort adds two kingdoms absent from development (fungi, plants).

**20-leg completion: 20 of 20 executed.** 9 h 43 m (34,978 s), `FINAL SURVIVING COUNT : 0`,
`cleanup : success`, 46 descendants observed and 46 terminated.

| leg outcome | strict | research | total |
|---|---|---|---|
| pass | 1 | 4 | **5** |
| fail | 6 | 1 | **7** |
| scope_conflict | 1 | 4 | **5** |
| timeout | 2 | 1 | **3** |

**PWML count: 1** (`PMC12071552/strict`, `pathway.review_required.pwml`, graph valid,
completeness 0.4). **Reactions extracted: 72.** **Rows carrying no provenance at all: 1 of
72.**

**Automated summary:** `evidence/orch724_pilot_summary.json`.
**Manual-review package:** `PILOT-MANUAL-REVIEW.md` — every leg with paper, mode, PWML path,
source-text path, final payload path, reaction list, enzymes, organisms, RAG-added rows,
warnings, and the four human labels.

---

# 5. Two corrections to the numbers, made before they could propagate

**(a) The PWML denominator is 10, not 20.** Measured across the project's entire history:
research legs have produced a PWML **0 times in 153 legs**; `acceptance.py:125` defines the
deliverables as strict-only. Research mode is diagnostic **by design**. Quoting `1/20` halves
the rate through a category error.

**(b) `PMC3480714`'s two legs are MY error, not the product's.** I assigned
*Salmonella enterica* from the pathway's reputation; the Deery enzyme-trap work is in a
**recombinant *E. coli* host**. Stage 0 read *E. coli* correctly and **consistently across two
independent draws** while my manifest was wrong. Both legs ended `scope_conflict` and **the
gate was right both times** — evidence *for* the organism check. Not re-run: a mis-specified
request is a curation error, not an infrastructure failure, and re-running it after seeing the
result would be outcome-driven tuning.

**Consequently the honest strict-PWML denominator is 6**, not 10 and not 20:

| strict legs | 10 |
|---|---|
| − manifest error (`PMC3480714`) | −1 |
| − negative control, where NO PWML is the correct outcome (`PMC12326985`) | −1 |
| − timed out at the 60-min ceiling (`PMC12542839`, `PMC13017326`) | −2 |
| **evaluable strict legs** | **6** |
| **produced a PWML** | **1** |

---

# 6. The one repeated major defect

Per-entity issue codes were **aggregated to mechanism families**, because
`failures_by_code.txt` ranks by distinct papers and every unresolved protein mints its own
code — scattering one mechanism across twenty codes and ranking it below a one-off.

| mechanism | papers | legs |
|---|---|---|
| **identity: mapping metadata incomplete** | **7** | 10 |
| **identity: protein has no UniProt/DrugBank id** | **6** | 7 |
| scope: Stage-0 scope string disagrees with request | 4 (3 genuine + 1 my error) | 5 |
| referential integrity: reference to unregistered entity | 1 | 1 |
| scope: review paper, extraction correctly skipped | 1 | 1 |
| identity: protein has no species/organism | 1 | 1 |

The top two are **one underlying mechanism**. Stated as the single repeated major defect:

> ## `F-185` — Entity identity resolution is incomplete on unseen papers, and the required-field gate then blocks strict PWML export even though the chemistry was extracted correctly. **7 of 10 papers.**

**It correlates with organism.** The development corpus was *E. coli*, *B. subtilis* and human
— densely annotated. The unseen cohort deliberately added fungi and plants, and the identity
layer does not cover them: `NIT-9G`/`NIT-9E` (*Neurospora*), `G10H`, `SGD`, `DAT`, `CrNPF2.9`,
`CrTPT2`, `CrGATA1`, `CrPIF1` (*Catharanthus*). Two sub-causes present identically at the
gate: **real proteins the mapper cannot resolve**, and **strain/construct labels extracted as
proteins** (`NIT-9G` — the `strain_or_construct` forbidden class).

**This is not a fabrication and not a recovery failure.** In every instance the reactions were
extracted and export was refused for missing identifiers. The failure direction is the safe
one.

Also registered, **not fixed**: `F-184` (negation/hypothetical scoping across
`_ALL_PROSE_PATTERNS` — pre-existing; 8 of 15 probes were already admitted at the base SHA,
and `C-118` is the first of fourteen templates to carry any negation guard at all).

---

# 7. What the pilot shows that the numbers alone do not

**Extraction is not the bottleneck.** `PMC12542839/research` recovered the **canonical
riboflavin biosynthesis pathway, essentially complete and correct** — `GTP → DARPP → ArPP →
ArP`, `Ru5P → DHBP`, `ArP + DHBP → DRL`, `DRL → riboflavin`, `riboflavin + ATP → FMN + ADP`,
`FMN + ATP → FAD` — 11 reactions with correct intermediates and correct branch convergence,
from a paper the system has never seen. **And it produced no PWML**, because research never
exports and that paper's strict leg hit the ceiling.

**The negative control behaved exactly as designed.** `PMC12326985` — a clinical review with
**0 reaction cues over 110,584 chars** — was refused in **20.9 s**:

> `multi_example_review detected with no selected_example. Extraction skipped to prevent
> mixed-pathway output.`

It did not invent a pathway, and it did not take the planted bait — the single passing mention
of "lipid A biosynthesis", a development-corpus pathway named with no chemistry behind it.

**Recovery and delivery are separate axes and must be reported separately.** 72 reactions
recovered against 1 PWML delivered is not a comprehension failure; it is a delivery failure
with four non-biological causes: identity resolution, a compute ceiling, scope-string
mismatches, and one referential-integrity block.

---

# 8. Publication source tables

## End-to-end

| | |
|---|---|
| papers attempted | 10 |
| legs completed | **20 / 20** |
| legs passing | 5 (strict 1, research 4) |
| PWML generated | **1** |
| usable PWML (graph valid, importable structure) | 1 |
| invalid outputs | **0** |
| reactions extracted | 72 |
| rows with no provenance ("invention floor") | **1 of 72 = 1.4%** |
| negative control correctly refused | **yes**, in 20.9 s |

## Reaction quality

**Not computed, and deliberately so.** There is no gold set for these ten papers and none was
invented. `bench_acceptance.py --verify-plan` **refuses** this plan — correctly, since it
compares against the pinned gold set, which is by construction the ten *development* papers.
Precision / recall / F1 and substrate/product/enzyme/species accuracy are therefore **pending
human review**, which is the point of the exercise.

## RAG

Committed metrics stand unchanged (`Recall@5 = 93.0%`; `unsupported_candidate_admitted = 0`
on untruncated legs). **RAG added 0 rows in the pilot** — every pilot reaction is
paper-derived.

## Human review

**Pending.** `PILOT-MANUAL-REVIEW.md` is the package; labels are PASS / PASS WITH MINOR
OMISSIONS / MAJOR BIOLOGICAL ERROR / INVALID OR UNUSABLE PWML.

Two items to look at first:
1. the **1 row carrying no provenance** (`PMC12376012/research`) — the invention class;
2. `PMC11172790/research`'s product named **`"three-amino-acid product"`** — a descriptor, not
   a molecule, adjacent to the `placeholder_product` forbidden class.

---

# 9. FINAL RECOMMENDATION

> **Is there any repeated major defect that genuinely justifies delaying manuscript work?**

**No.**

`F-185` is repeated (7 of 10 papers) and it is real, but it does **not** justify delaying the
manuscript, for three reasons:

1. **It is a completeness limitation, not a correctness failure.** Nothing it causes is a
   fabricated reaction, a wrong substrate/product, a wrong organism or an unusable PWML. Every
   instance is the system *declining to deliver*, which is the failure direction this project
   chose deliberately and defends in print.
2. **It is characterizable now.** "Entity identity resolution does not cover non-model
   organisms, and strict export requires it" is a clean, honest limitation section. It does not
   need to be fixed to be reported truthfully.
3. **The charter's stopping rule governs.** Occasional missing reactions, imperfect recall and
   conservative rejection are explicitly listed as *not* grounds to reopen. `F-185` is a
   sharper version of exactly that.

The pilot also produced the two results a manuscript most needs: a **complete, correct
pathway recovered from an unseen paper**, and a **negative control correctly and cheaply
refused**.

## `STOP ENGINEERING — PROCEED TO HUMAN PWML REVIEW AND MANUSCRIPT ANALYSIS.`

**Two items are owed before submission, and neither is engineering:**

- **resolve the `streamlit_app.py` uncommitted-modification caveat**, or the reproducibility
  claim is false as written;
- **complete the human review** in `PILOT-MANUAL-REVIEW.md` — every biological quality number
  in the manuscript depends on it, and none of them can be produced by this pass.
