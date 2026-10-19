# Evaluating an LLM text-to-pathway pipeline: methods, results and limitations

**Scope.** Everything established by the ORCH-747 evaluation work (2026-10-13 to
2026-10-16) that belongs in a write-up: how the gold standards were built, how the
extraction was measured, what the measurements show, what they do not show, and what is
reproducible from the repository. Process history and open engineering decisions live in
`docs/pwml_recovery_sprint/ORCH-747-RESULT.md`; this document is the scientific record.

**One-line summary of the result.** Prompt wording made no measurable difference to
extraction quality (Δ mean F1 = 0.029, 0.21 pooled SD, n = 10 per arm), while pooling
three independent draws of the same prompt raised reaction recall from 0.689 to 1.000 on
the weakest condition at unchanged precision — so the dominant controllable factor is
sampling variance, not instruction design.

---

## 1. What was evaluated

T2PW converts a paper's text into a structured biological pathway and exports it as PWML
for PathWhiz/PathBank import. The evaluation here targets **Stage 1 extraction** only: the
single LLM call that turns paper text into a structured payload of entities and processes
(reactions, transports, reaction-coupled transports, interactions). Downstream retrieval,
merging, curation, identifier mapping and export were held fixed.

Model: `deepseek/deepseek-v4-flash` via OpenRouter, temperature 0, 64,000 max output
tokens (§ 6.2 explains why that number). One call per draw, through the production stage
runner so that retry and JSON-repair behaviour matched production exactly.

---

## 2. Gold standard design

### 2.1 Two golds per pathway, and why

The reference pathways are human-curated PathBank entries built by a laboratory from
sources of the curator's choosing. The paper fed to the pipeline is therefore **not
guaranteed to be the source the curator used**, so a disagreement between pipeline output
and curated record can mean either a pipeline error or a legitimate difference of source.
Scoring against the curated record alone cannot distinguish them.

Each pathway therefore carries two references:

1. **the curated PathBank record**, read read-only from the live database, giving the
   lab's own view of the pathway; and
2. **a paper gold**, an exhaustive inventory of what one specific pinned full text states,
   every row carrying a verbatim quote from that text.

Agreement between them is evidence; disagreement is a finding to classify, not an error to
assign. § 5.1 reports what this design caught.

### 2.2 Certification status matters and is tracked

A gold drafted by a model is not a gold. Three papers carry **certified exhaustive** gold
from earlier work: a human read the whole text and listed every transformation it states
with both sides named, with quote verification against a hash-pinned text. The araC paper
gold produced here is **model-drafted and NOT certified**; its `supported_reactions_complete`
flag is `false`, which makes the evaluator skip it for precision, recall and F1 by
construction. The assembler writes that flag `false` regardless of what a draft claims, so
a model cannot certify its own exhaustiveness.

### 2.3 Blind and hint-given drafting, with a contamination check

The paper gold was drafted twice: once blind, and once with the curated record supplied as
a scope hint. The hint is the obvious contamination route — a drafter handed a list of
expected processes may begin "finding" them. Measured: 15 interactions against 11, 8
matching, 7 quotes verbatim-identical between two passes that never saw each other's work,
and no in-scope reaction in the hinted pass unrelated to anything the blind pass found.

### 2.4 Quote verification and its blind spot

Every gold row cites a verbatim quote, verified against a text whose SHA-256 is recorded;
a text whose hash differs is refused rather than trusted. Verification folds out case,
whitespace and punctuation.

**That folding is also a blind spot, and it bit.** Because whitespace is removed, a text
whose words have been scrambled by a faulty PDF reader still "verifies" quotes that are
themselves scrambled. Three extraction revisions were needed before the text was correct
(§ 5.2). The decisive check was not the hash but a cross-revision one: **every quote in all
four drafts verifies against the final text, including the two early drafts whose quotes
failed against the intermediate revisions** — the drafters had been quoting real sentences
throughout, and the extraction was wrong.

*Methodological recommendation for any similar study: hash-pinning a derived text proves
you read the same bytes, not that those bytes represent the document.*

---

## 3. Measurement instrument

Reaction scoring uses the evaluator's own strict matcher (`eval_v2`), unchanged:

- A row is a true positive when **every defining participant** of an in-scope gold reaction
  is present on the correct side; reversible gold may match backwards.
- Equality of a normalised name, with no containment in either direction, so **congeners are
  never credited** (`menaquinone` does not match `menaquinone-7`; `CoA` does not match
  `acetyl-CoA`). Stereo descriptors are never stripped.
- Cofactors are scored separately and never fail an otherwise correct reaction.
- Precision counts rows claimed as paper extraction; recall counts in-scope gold reactions
  matched; both tables are reported separately and never summed.

**Interactions are not scored by this instrument**, which measures two-sided conversions
only. They were measured directly instead: count, values outside the permitted vocabulary,
rows lacking an evidence quote, and restatements of one claim at two resolutions.

---

## 4. Papers, conditions and sample sizes

| paper | pathway | organism | gold | in-scope reactions |
|---|---|---|---|---|
| PMC12096016 | enterobactin biosynthesis | *Escherichia coli* | certified exhaustive | 5 |
| PMC12421875 | menaquinone biosynthesis | *Lactococcus lactis* | certified exhaustive | 9 |
| PMC319880 | araC autoregulation (DNA footprinting) | *Escherichia coli* | model-drafted, not certified | 1 (+15 interactions) |

The third paper is deliberately unlike the first two: its measured result is protein–DNA
occupancy, not chemistry. It was included to test generalisation and it exposed a real
limitation (§ 6.4).

| round | design | draws |
|---|---|---|
| 1 | 10 prompt variants x araC | 10 |
| 2 | 16 variants x 3 papers x 1 draw | 48 |
| 3 | 6 variants x 2 certified papers x 3 draws | 36 |
| 4 | 2 variants x 2 certified papers x 5 draws | 20 |
| | **total** | **114** |

Draws scored against certified gold: **88**. All 26 araC-only cases of rounds 1 and 2 are
excluded from F1 for the reason in § 6.4.

---

## 5. Results

### 5.1 Evidence discipline held, without exception

**Across all 114 draws, not one extracted row lacked a verbatim evidence quote.** This held
for every prompt variant, including those that tripled the number of rows, raised
extraction to DNA-site resolution, added classification routers, added worked examples and
added pre-output self-checks.

This is the strongest result of the study. The usual worry about pushing an extraction
model for more coverage is that coverage is bought by invention; it was not, in any
condition tested.

### 5.2 Precision is high; recall is the bottleneck

Over the 88 certified-gold draws, precision was 1.000 in all but 2 (minimum 0.800). False
reactions occur but are rare. Single-draw recall, by contrast, ranged from 0.22 to 1.000.

An earlier round-2 analysis reported precision as exactly 1.000 everywhere. **That was an
overstatement from 32 draws and is withdrawn**; the direction holds, the absolute claim
does not.

### 5.3 Sampling variance at temperature 0 dominates the prompt effect

The same prompt, on the same paper, at temperature 0, gave materially different draws:

| condition | matched reactions, per draw | gold |
|---|---|---|
| control, menaquinone (round 3) | 9, 7, 4 | 9 |
| control, menaquinone (round 4) | 4, 7, 4, 9, 7 | 9 |
| self-check variant, menaquinone | 9, 2, 9 | 9 |
| mixed-case variant, menaquinone | 2, 9, 9 | 9 |

Measured noise floor: **±0.25 F1 on a single draw**; pooled SD across draws 0.135.
The within-variant spread equals the entire between-variant spread, which means
**single-draw prompt comparisons are uninterpretable**. Round 2's one-draw ranking did not
survive replication: its joint winner fell to third in round 3 and the variant that led
round 3 failed to separate from the control in round 4.

### 5.4 Prompt engineering produced no measurable gain

Twenty-six variants were authored across two generations, each a complete drop-in system
prompt, all verified to declare an identical JSON schema (86 keys) so that downstream
export behaviour was unaffected. Designs included resolution-precedence rules, bucket
grammar gates, canonical-naming rules, indirect-conversion sweeps, worked regulatory
examples, pre-output self-checks, and four distinct classification/routing designs.

Head-to-head, 5 draws per arm, 2 papers:

| | enterobactin | menaquinone | overall mean F1 |
|---|---|---|---|
| control | 0.978 | 0.796 | 0.887 |
| best challenger | 0.933 | 0.898 | 0.916 |

Δ = 0.029 = **0.21 pooled SD**. Not a real effect, and a per-paper trade rather than a win:
the control is better on one paper and worse on the other.

Two variants were, however, clearly **harmful**, losing 7 of 9 reactions on the menaquinone
paper — far outside the noise floor. Both were suppression-shaped rules, consistent with
§ 5.2: in a regime where precision is already near-ceiling, rules that remove rows can only
cost recall.

One prediction is worth recording because it was confident, well-argued and wrong. The
variant author predicted that classification routing would not help and that two
shape-based rules would be the largest winners. Those two rules were the worst in the grid,
collapsing on the paper they were predicted to help, while a routing variant tied for
first. *An argument from how a rule ought to read is not evidence about how a model
behaves.*

### 5.5 Pooling independent draws reaches recall 1.000 at unchanged precision

Because precision is near-ceiling and each draw misses *different* reactions, the misses
are recoverable by union. Expected recall against number of pooled draws, averaged over
every combination of that size (n = 5 draws per arm):

| condition | N=1 | N=2 | N=3 | N=4 | N=5 |
|---|---|---|---|---|---|
| control, enterobactin | 0.960 | 1.000 | 1.000 | 1.000 | 1.000 |
| control, menaquinone | 0.689 | 0.922 | **1.000** | 1.000 | 1.000 |
| challenger, enterobactin | 0.880 | 0.940 | 0.980 | 1.000 | 1.000 |
| challenger, menaquinone | 0.844 | 1.000 | 1.000 | 1.000 | 1.000 |

Pooling cost nothing in precision here: over the 115 pooled rows across all four arms,
115 were true positives, **precision 1.000, no false reaction entered any pool.**

**Effect size comparison: +0.31 recall (ensembling, weakest condition) against +0.03 F1
(best prompt change).** An order of magnitude, for a change in sampling strategy rather
than instruction design.

### 5.6 The two-gold design caught defects in the curated reference

Comparing the curated PathBank record for the araC pathway against the paper gold produced
three findings about the **reference data**, not the pipeline:

- **A sign disagreement.** The record states that the araO1 operator *activates* the araC
  promoter; the paper's central result is that occupancy of that site *represses* it by
  competing with RNA polymerase. The sign is a deliberate claim rather than a drawing
  convention: the same curated pathway uses `Inhibition` on another edge.
- **An edge encoding a superseded model.** A second edge encodes the model the paper's own
  Discussion argues against.
- **An entity-type error.** CAP is stored as a nucleic acid although it is a protein, and
  the same database holds the correct protein record with a UniProt accession.

Classification of these as gold-data defects, policy disagreements, or legitimate
source differences is pending; neither of the two reviews the curator cited has retrievable
full text, so the question cannot be closed from the literature available.

*Generalisable point: an evaluation whose reference is a single curated database inherits
that database's errors as if they were ground truth. The second, paper-anchored gold is
what makes them visible.*

---

## 6. Limitations and threats to validity

### 6.1 Sample size and scope
Two papers with certified gold, both metabolic, both bacterial, 14 gold reactions between
them. Reaction-level conclusions rest on that. The ensembling result in particular is
measured on 2 conditions x 2 papers x 5 draws and should be replicated more widely before
being treated as general.

### 6.2 A configuration artifact nearly produced a false result
Three of four grid attempts failed for reasons unrelated to the prompts. The decisive one:
the model spends output tokens on reasoning before emitting content, so at the configured
16,000-token budget it frequently exhausted the allowance and returned an empty body with
`finish_reason=length`. At 12,000 and 16,000 a given variant failed deterministically; at
64,000 the same variant succeeded. Had this not been chased, six of ten variants would have
been recorded as producing unparseable output — a prompt conclusion from a budget cause.

*Any study of this kind should verify that the measured configuration matches the
production configuration before attributing a failure to a prompt.*

### 6.3 Tooling defects masqueraded as model failures
Four separate defects in surrounding code produced symptoms that looked like extraction
failures:

| defect | symptom it produced |
|---|---|
| PDF reader interleaves two-column pages | sentences shredded; quote verification cannot detect it (§ 2.4) |
| graph builder mis-typed DNA entities as compounds | every DNA entity duplicated, correctly typed copies orphaned |
| text report rendered reactions and transports only | a run with 1 reaction and 12 interactions displayed the reaction alone, reading as a recall failure that had not occurred |
| process-row cleaner deletes 5 prompt-mandated fields | a review-required gate cannot fire on process rows |

*Three of the four were invisible in the output being examined. An evaluation that reads a
rendered report rather than the payload can measure the report.*

### 6.4 One paper type could not be scored at all
No variant, in any round, extracted the araC paper's single gold reaction: recall 0 in all
26 draws. F1 is undefined rather than zero, because no variant claimed any reaction, so
precision has an empty denominator. The pipeline represents that paper's content as
interactions, and whether a transcription step should be recorded as a reaction is an
unresolved scope question. **Until it is resolved, regulatory papers cannot contribute to
reaction-level scoring**, and no claim here extends to them.

### 6.5 Interaction quality is largely unmeasured
The instrument scores two-sided conversions only. Interaction counts (11–23 per draw on
araC) were compared against a model-drafted, uncertified gold, so no precision or recall is
claimed for them. The alias-row and split-naming defects observed in full pipeline runs
were traced to stages *after* extraction, so two prompt variants aimed at them could not
have worked.

### 6.6 Inherited evaluator limitations
From the evaluator's own readiness assessment, unchanged here: retrieval negative-query
rejection is structurally 0%; 6 of 11 per-call reproducibility fields are unrecorded;
the wrong-identity mapping flag has a measured precision of 5/21; and the identifier
reference tables are read from a committed snapshot, so an identifier absent from it is
scored `no_reference` rather than wrong.

### 6.7 Pooling was tested in only its most permissive form
Union pooling was measured. A majority rule, which would trade some recall gain for
protection against the rare false row (§ 5.2), was not tested. Pooling also multiplies cost
and latency per extraction linearly in N.

---

## 7. Reproducibility

| item | where |
|---|---|
| base commit | `e4348aaf` (all work in the working tree, uncommitted at time of writing) |
| prompt variants, generation 1 and 2 | `docs/evaluation_v2/prompt_variants/`, `prompt_variants_r2/` with audits and schema checks |
| gold: curated extract, paper gold, blind/hinted drafts, comparison | `docs/evaluation_v2/pathbank_gold/` + its README |
| paper source manifests: citation, PDF and text hashes, extraction revisions and why each was superseded | `docs/evaluation_v2/pathbank_gold/PMC319880__source.json` |
| certified gold for the two metabolic papers | `src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json` |
| pinned paper texts (outside the repository by design) | `C:/t/eval-fulltext/<PMCID>.txt`, SHA-256 recorded per case |
| column-aware PDF extraction | `docs/pwml_recovery_sprint/evidence/orch747_pdf_columns.py` |
| the A/B harness and the scorer | `scripts/prompt_ab.py`, `scripts/score_prompt_grid.py` |
| per-draw payloads and rendered summaries | `C:/t/prompt-grid/{final,r2,r3,r4}/` |
| per-run process reports (every run bounded, zero surviving processes) | `docs/pwml_recovery_sprint/evidence/g11/ORCH-747/` |

Every draw records the prompt's SHA-256, the paper text's SHA-256, the model, temperature
and token budget. Identical draws differ (§ 5.3), so a single draw is not reproducible by
construction; the distribution is.

---

## 8. What to claim, and what not to

**Supportable:**
- A quote-anchored extraction prompt can be pushed to higher coverage and finer resolution
  without producing unsupported rows (114/114 draws).
- In this configuration, extraction precision is near-ceiling and recall is the limiting
  factor.
- Sampling variance at temperature 0 is large enough to invalidate single-draw comparisons
  (±0.25 F1).
- Pooling three independent draws recovered all gold reactions in 3 of 4 conditions with no
  precision loss, a far larger gain than any prompt variation tested.
- A paper-anchored second gold exposes errors in a curated reference database that a
  single-reference evaluation would have scored as pipeline failures.

**Not supportable from this work:**
- that any particular prompt wording is better than another;
- that ensembling helps on regulatory or non-metabolic papers;
- any reaction-level claim about papers whose content is not two-sided chemistry;
- any precision or recall figure for interactions;
- that the curated reference errors found are defects rather than source differences.
