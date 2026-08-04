# The pinned scientific benchmark

## 1. What problem it solves

Before this, a batch run reported one number per mode: passes over attempted
pairs. That single number conflates five unrelated questions —

1. did search find a paper that *could* describe the pathway at all?
2. did extraction produce a structurally valid payload?
3. is the payload **biologically about the requested pathway**?
4. does it satisfy PathWhiz's importer-shape rules?
5. did research mode produce a reviewable candidate?

— so it moves when *retrieval* changes and gets read as if the *exporter*
changed. Worse, it has no notion of scientific error at all: in
`runs/2026-07-28_2122` a payload whose `entities.proteins` contained coenzyme A,
whose reaction outputs included names the paper never states, and whose sigma
factor σ32 was filed as a compound scored exactly the same as a correct one,
because the only question asked was whether the gate was happy.

This package answers those five questions separately, against a **pinned gold
set** whose expectations were read out of the papers by hand with a verbatim
quote behind each one.

Code: `src/t2pw/bench/{goldset,semantic,metrics,acceptance,render}.py`, data
`src/t2pw/bench/gold/pinned_v1.json`, CLI `scripts/bench_acceptance.py`,
work list `bench/pinned_topics.txt`.

---

## 2. The gold set

Ten papers already fetched by the 2026-07-27/28 batches, covering the five
pathways in `topics.txt`. Eight are mechanistically relevant; two deliberately
are not.

| paper | pathway | relevance | export | min core |
|---|---|---|---|---|
| PMC12444477 | lipid A | core | partial_only | 3 |
| PMC13231680 | lipid A | context_only **(negative control)** | partial_only | 0, ceiling 0 |
| PMC12657337 | menaquinone | core | strict_exportable | 3 |
| PMC12421875 | menaquinone | core | strict_exportable | 7 |
| PMC12312563 | menaquinone | partial | partial_only | 1 |
| PMC12856317 | heme | partial | partial_only | 1 |
| PMC12180156 | heme | context_only | partial_only | 0, ceiling 2 |
| PMC12096016 | enterobactin | core | strict_exportable | 4 |
| PMC12452463 | enterobactin | partial | partial_only | 2 |
| PMC12782028 | cholesterol | partial | strict_exportable | 2 |

### Three rules the schema enforces

**Expectations are what the paper supports, not what the pathway is.**
`min_connected_reactions` for PMC12444477 is 3, not the nine steps of the Raetz
pathway, because the review names all nine enzymes but leaves six of the nine
intermediates chemically unnamed in its body text. Scoring against the textbook
would mark a faithful extraction as a failure and reward a pipeline that
hallucinated the missing names. `expected_export` records *separately* whether a
complete importable pathway is obtainable from the paper at all, which is what
keeps a 0% strict rate from being blamed on the exporter.

**Enzymes come in two tiers.** `expected_enzymes` are described as catalysing a
step; missing one is a coverage failure. `acceptable_enzymes` are regulators and
accessory proteins; their presence is neither required nor an error. Without the
second tier, every correct extraction looks over-broad.

**Forbidden identifiers are the load-bearing half.** These are strings that are
*not real distinct chemical or protein identities*: placeholder product names
(`LpxA product`), cofactors that keep getting filed as proteins (`coenzyme A`,
`thiamine diphosphate`), strain and construct labels (`MenFD`, `BsHepPPS`),
regulators filed as metabolites (σ32), domain names (`ArCP`, `TH3`). Emitting one
carrying a real accession is the worst outcome the pipeline can produce, because
every structural gate passes and the result is silently wrong. Acceptance
priority 1 is defined on this list.

### Negative controls

`max_retained_reactions: 0` marks a paper that must yield nothing. PMC13231680
is an NDM-1 inhibitor study retrieved because its discussion mentions LpxC once;
full-text counts are `lipid A` 1, `UDP` 0, `GlcNAc` 0, `Kdo` 0, `acyl` 0. A floor
rewards recall and can always be satisfied by inventing chemistry; a ceiling of
zero can only be satisfied by declining to invent it. PMC12180156 carries a
ceiling of 2 rather than 0 because it does fully state two reactions — neither of
which belongs to heme biosynthesis.

---

## 3. The eight semantic checks

`t2pw.bench.semantic.validate_semantic_coverage` is deterministic and offline.

| check | what it asks |
|---|---|
| `requested_pathway_anchors_present` | is this the pathway that was asked for? |
| `reaction_source_carrier_present` | does every reaction carry a citation *field*? (hygiene) |
| `retained_reactions_match_supported_signatures` | is the chemistry the paper actually states? |
| `organism_compatible` | is any reaction attributed to a forbidden organism? |
| `no_real_id_or_name_conflict` | does anything carry an accession it has not earned? |
| `no_rejected_rag_reaction_reintroduced` | did a claim the admission gate refused come back? |
| `minimum_connected_core` | do enough reactions form ONE chemically connected chain? |
| `placeholder_identities_distinguished` | do Unknown-backed rows admit what they are? |

**A source carrier is not scientific support.** The second check was previously
called `evidence_source_for_every_retained_reaction`, which claimed far more than
it measured: a bare `evidence: "the paper says"` satisfies it, as does a
`source_ref` pointing at a passage about something else, so it could report a
hallucinated reaction as fully sourced. Its error count was renamed to match
(`reactions_missing_source_carrier`).

The third check is the one that measures support. Each gold case carries
`supported_reactions` — reaction signatures (inputs, outputs, optional enzyme,
directionality, permitted aliases) read out of the paper by hand, each with a
quote that is **verified against the stored `01_source_text.txt` at scoring
time**. All 19 signatures in the shipped set verify; a quote that does not is
reported as a *gold-set defect* and excluded from scoring rather than blamed on
the pipeline. It reports reaction-level precision and recall: a retained claim
matching a signature is a true positive, a signature with no match is a false
negative, and a retained claim matching nothing is unattributed.

`supported_reactions_complete` (default `false`) is what makes "unsupported" a
defensible word. With a *subset* signature list, an unmatched row may be a
genuine invention, a step stated in a form no signature was written for, or a
legitimate cross-paper RAG addition — indistinguishable from the list alone. So
unmatched rows are reported as **unattributed** and are not counted as errors,
and `precision` is reported as `attribution_rate`. Counting them as fabrications
reported 227 hallucinated reactions in a run that produced far fewer. Recall is
unaffected either way: a declared signature the payload lacks is a real miss.

Negative-control **ceilings** are kept alongside signature matching — they are
the only rule that works for a paper with no supportable chemistry at all.

**An inapplicable check is not a pass.** If the RAG admission report or the paper
text is missing, the affected check reports `applicable: False` and the paper is
excluded from `confirmed`. Semantic success is scored on `confirmed`, not `ok`:
`ok` considers only *applicable* checks, so scoring on it would count a leg whose
reintroduction check never ran as a success.

### Connectivity model

Two reactions are adjacent when they share a **non-cofactor** metabolite. Both
halves matter. Enzymes are excluded because two unrelated steps sharing a
promiscuous enzyme are not a pathway — the same rule `rag.admission` enforces via
`participant_names()`, and deliberately *not* what `pipeline.qa_graph.build_graph`
computes (it adds enzyme edges, so its component counts run optimistic).
Cofactors are excluded because a graph linked through ATP reports any bag of
unrelated reactions as one pathway. The cofactor list is imported from
`t2pw.rag.synthesize.COFACTOR_NAMES`, not restated.

---

## 4. Separated denominators, over what actually ran

Every rate carries its own population and its exclusions, and an empty
population renders as `n/a` rather than `0%`. The governing rule: **an
unattempted or unscorable paper is missing coverage, not a pipeline failure.**

| key | question | population |
|---|---|---|
| `gold_relevance_prevalence` | how much of the gold set is mechanistically relevant? | all gold cases |
| `extraction_success` | payload with ≥1 reaction? | relevant cases **with an attempted leg** |
| `semantic_pathway_success` | **confirmed** — all checks passed *and* evaluable? | relevant cases **with a scorable payload** |
| `strict_pwml_success` | did papers that *can* export, export? | `strict_exportable` cases **with an attempted strict leg** |
| `research_deliverable_produced` | did research mode PRODUCE a report? (output only) | relevant cases **with an attempted research leg** |
| `research_semantically_confirmed` | is the science in it right? | relevant cases **with an attempted research leg** |

The two research rates are deliberately separate. On `runs/2026-07-28_2122` they
are **5/5 produced** and **0/5 confirmed**: every attempted research leg emitted a
citation report, and not one of them is scientifically confirmed. Quoting the
first as a success rate is the specific misreading the split prevents.

`AcceptanceReport.completion()` reports coverage on its own — planned gold cases,
papers attempted, strict/research legs attempted, payloads available,
semantically scorable legs, fully completed cases, and the papers with no
attempted leg.

**An incomplete run can never be accepted.** `scripts/bench_acceptance.py` exits
`1` whenever `report.is_complete` is false, *before* the priority checks are even
consulted. Error counts are computed over the legs that ran, so fewer legs means
fewer chances to be wrong: 0/20 clears priorities 1-3 perfectly, and 19/20 is the
same failure in miniature — the twentieth leg is exactly where the missing error
would be. Text and JSON reports are still written for a partial run (it is worth
reading, just not worth quoting) and are labelled `PARTIAL -- not a quotable
benchmark result`.

---

## 5. Scientific errors, counted separately

`false_real_identifiers`, `placeholder_backed_proteins`, `unsupported_reactions`,
`reactions_missing_source_carrier`, `cross_organism_reactions`,
`orphaned_references`, `missing_pathway_anchors`, `missing_supported_reactions`,
`quarantined_processes`.

Never summed. One fabricated accession is worse than forty honest Unknown-backed
proteins, and the acceptance priorities order them explicitly rather than
weighting them into a score.

Expected-enzyme and expected-metabolite recall are computed per leg and are
**reported** in the per-paper block, not silently calculated.

## 5b. Blocker rankings are three, not one

"Papers released if fixed" only means something inside one population, so
`AcceptanceReport.blockers` is keyed by scope:

- `strict_export` — strict legs of `strict_exportable` cases only
- `research_deliverable` — research legs of relevant cases
- `extraction` — legs that produced no payload, in either mode

PMC12312563 is `partial_only`: fixing the Stage-3 gate that stops its strict leg
is real work, but it cannot raise the strict-PWML rate by even one paper, because
the paper is not in that denominator. A merged ranking presented it as the top
strict blocker.

**A paper whose correct output is empty is never an extraction blocker.** A
negative control — and any `context_only` case with no minimum core — is
*supposed* to yield nothing, so an empty extraction there is the pipeline behaving
correctly. PMC13231680 previously topped the `extraction_empty` ranking for
declining to invent a lipid A pathway from a paper that contains none. Over-
retention on such a paper is still reported, as unsupported retention via the
ceiling rule, and never as a blocker.

---

## 6. Corrected research-mode reporting

`batch/report.py` doctrine was *"research mode is fail-open by construction, so a
research failure can only mean the code itself is wrong."* That was true when
research mode relaxed everything; it is false now. The structural guards
(`entities_required`, `processes_required`, `invalid_payload`, …) abort in **both**
modes by design, and a paper that yields no parseable payload fails research mode
for a reason that has nothing to do with research mode.

`metrics.classify_research_failure` reports, most specific first:

- `timeout`
- `provider_failure` — LLM or network
- `deliberate_ambiguity_stop` — the run refused on purpose; a correct outcome
- `structural_extraction_failure` — no parseable payload; aborts in both modes
- `relaxed_export_gate_failure` — the relaxation did not cover the rule that fired
- `research_mode_implementation_defect` — nothing above explains it

`metrics.classify_strict_boundary` does the same for strict mode, replacing
`failure_kind="contract"` — which reported the Stage-3 gate, the PWML
required-field gate and a structural guard as the same thing — with the boundary
the run actually died at.

---

## 7. Running it

```bash
# validate the gold set
python scripts/bench_acceptance.py --validate-gold

# regenerate the scoped pinned topics file from the gold set
python scripts/bench_acceptance.py --write-topics bench/pinned_topics.txt

# check the topics file BEFORE any fetch
python scripts/bench_acceptance.py --verify-topics bench/pinned_topics.txt

# stage the pinned batch: fetches by id, no search, and executes ZERO legs
python scripts/batch_run.py --topics bench/pinned_topics.txt --modes strict,research \
                            --stage-only --fresh

# ALWAYS verify the staged plan before running a leg
python scripts/bench_acceptance.py --verify-plan runs/TIMESTAMP

# run it (resumes the staged directory; finished pairs are skipped)
python scripts/batch_run.py --topics bench/pinned_topics.txt --modes strict,research

# score it
python scripts/bench_acceptance.py --run-dir runs/TIMESTAMP --json out.json --out out.txt
```

### Scope must be pinned too

Each topics line is `PAPERID | requested pathway | requested organism`.
`fetch.parse_topics` reads a first field that looks like a paper id as a **pinned
spec** and takes the remaining fields as the requested scope: fetched by id, zero
search calls, score bypassed via `OUTCOME_PINNED_OVERRIDE`, request preserved.

A **bare** pinned id also parses, and that is the trap.
`TopicSpec(pinned_id=..., topic="", organism="")` flows through
`_as_batch_paper` into `requested_pathway: ""` / `requested_organism: ""`, so
`driver._extraction_focus` hands extraction no scope, `_reconcile_stage0_scope`
has no request to contradict, and the screening record is computed against an
empty pathway. `runs/2026-08-01_1724` was staged that way: ten papers fetched,
zero skipped, a clean funnel, and every row scored `off_topic` /
`no_mechanistic_pathway_terms` while being admitted by `pinned_override`. It is
kept as `runs/INVALID-scopeless-2026-08-01_1724/` — renamed out of the
`RUN_STAMP_FMT` pattern so `runner.find_resumable` cannot pick it up.

`t2pw.bench.preflight` exists to make that silence impossible. It compares plan
rows (or parsed specs) against the gold set and refuses on any drift:

- a lost or changed `requested_pathway` / `requested_organism`;
- a recorded search query (a pinned paper is fetched by id and records none);
- an admission outcome other than `pinned_override` — **including a missing one**,
  since "we cannot tell how this paper got in" is not a property an acceptance run
  may have;
- a **duplicate** paper id, which used to be silently deduped to the last row —
  two rows for one paper make the run ambiguous, and because resume state is keyed
  on `(slug, mode)` a duplicate can also consume a pair slot twice;
- a missing or extra paper — the plan must be *exactly* the gold set;
- plan `modes` other than exactly `["strict", "research"]`, which would silently
  halve the benchmark while still producing a directory that scores;
- a pair count other than papers × modes (10 × 2 = 20 for this gold version).

`verify_specs` runs the same checks on parsed `TopicSpec`s *before* any fetch,
with `require_outcome=False` — no acquisition outcome exists yet at that point.

### Staging

`--stage-only` acquires the papers, writes `plan.json` / `skipped.json` /
`SUMMARY.txt`, and returns **before the run loop**. Zero legs is therefore a
property of control flow: no child command is built, no manifest row and no leg
directory can be created however fast the paper cache is.

**Never stage with a deadline.** `--stage-only` returns *before the run loop*,
so zero legs is a property of control flow: no child command is built, no
manifest row and no leg directory can be created however fast the paper cache
is. A small `--deadline` cannot promise that — `run_batch` compares
`spent >= deadline_seconds` **inside** the loop, after acquisition, so a
1.08-second budget is simply not exhausted when the fetch comes from cache and
the first leg starts anyway (this happened on 2026-08-01 with `--deadline
0.0003`). A zero or negative value is worse: `deadline_seconds > 0` is then
false and the entire batch runs.

### Why it is not called "eligibility yield"

`gold_relevance_prevalence` is 8/10 for this set. That is the composition of the
gold set, not a measurement of `rag.eligibility`: pinned acquisition bypasses the
screener, and a pinned `CandidatePaper` carries neither title nor abstract, so
every recorded score is -1.5 / `off_topic` regardless of the paper. To measure
the screener, run it in non-vetoing shadow mode over a derived lead window and
report precision / recall / specificity against the gold relevance labels —
which is what `scripts/eligibility_dry_run.py` already does, marking such
decisions `provisional`.

### Exit codes

`0` scored and acceptance priorities 1-3 hold · `1` a priority is violated, **or
no leg was attempted** · `2` usage error or unreadable gold set / run directory.

The "no leg attempted" case is exit 1 on purpose: a run with no output has no
scientific errors, so priorities 1-3 hold vacuously and an empty run directory
would otherwise report as acceptable.

---

## 8. Artifacts the benchmark needs

Two were added to `driver._add_identity_artifacts`, on the **common** path that
every failure branch already calls:

- **`final_mapped.json`** was previously written only by `_add_strict_artifacts`,
  i.e. only when a strict run reached a PWML export. Every strict run in
  `runs/2026-07-28_2122` died at the Stage-3 gate, so not one stored a mapped
  payload — and `merged_payload.json` is *pre-mapping*, carrying no accession of
  any kind. Auditing identity against it reports "zero false identifiers" for
  every run, which reads as a clean bill of health and is really "the file cannot
  answer the question". The scorer records `payload_source` on every leg for
  exactly this reason.
- **`rag_admission_report.json`** — `AdmissionReport.rejected` never reached disk
  at all, so "was a rejected reaction reintroduced?" was not a check that failed,
  it was a check that could not run.

  It must be read from **`rag_result.synthesis.admission`**, the path
  `streamlit_app.render_rag_panels` uses. The report hangs off `SynthesisResult`,
  not off the orchestration result, so `getattr(rag_result, "admission")` is
  always `None` against a real run — the artifact would never appear while the
  wiring looked correct. `driver._rag_admission` reads the production path and
  keeps the flat attribute only as a fallback. It is written on the common path,
  so it survives a strict Stage-3 failure, a research success, and a failure
  after RAG synthesis but before PWML export.
