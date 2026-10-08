# Testing

**No test in this feature spends an API credit.** The model is a scripted
function, the literature service is a scripted source, embeddings run in the
offline lexical mode with a temp-directory cache, and the vector backend is
in-memory.

## Running the suite

```bash
# focused, fast (~4 s)
PYTHONPATH=<tree>/src python -m pytest -q --basetemp=<short-tmp> tests/test_mining_*.py
```

Under the sprint's bounded wrapper, as every test in this repository must run:

```bash
PYTHONPATH=<tree>/src python docs/pwml_recovery_sprint/evidence/bounded_run.py \
  --label mining-suite --timeout 600 --heavy-lock EXP-MINING \
  --json docs/pathway_mining/evidence/002-mining-suite.json -- \
  python -u docs/pwml_recovery_sprint/evidence/pinned_pytest.py \
    --pin-verdict docs/pathway_mining/evidence/002-mining-suite.pin.json \
    -q --basetemp=<short-tmp> tests/test_mining_*.py
```

`--basetemp` is mandatory and its **parent must already exist** — pytest does not
create intermediate directories, and a missing parent errors every test in setup,
which looks exactly like a large regression.

## The files

| File | Tests |
|---|---|
| `test_mining_feature_flag.py` | default-off, three activation paths, override in both directions, controller refusal, cheap import (asserted in a subprocess), no writes when disabled |
| `test_mining_species.py` | local normalization, genus≠binomial, all seven states, the ortholog barrier, every verdict carries a reason |
| `test_mining_graph.py` | the three connection kinds; enzyme / pathway-name / paper / vocabulary / cofactor **never** connect; multi-substrate representation; quarantine; bridge promotion; reconsideration; deterministic primary-component selection; JSON round trip |
| `test_mining_starting_inputs.py` | every accepted paper-reference form and refusal of the rest, composability, mandatory species, upload determinism, evidence-type labelling, config precedence |
| `test_mining_extraction.py` | per-field evidence attribution, uncited fields dropped, fabricated citations refused, schema validation, empty / truncated / parse failures, the shrink-and-retry ladder, explicit fallback, per-call pricing |
| `test_mining_admission.py` | the ordered gates, dedup preserving independent evidence, conflict retention, reconsideration chains, provenance-chain completeness |
| `test_mining_budget.py` | the closed stop-reason set, every limit, convergence, cancellation, cumulative-on-resume budgets, fresh wall clock |
| `test_mining_artifacts.py` | every required file, manifest completeness, no secrets anywhere, schema versioning, per-round self-containment, cancellation, resume, regeneration |
| `test_mining_cli.py` | gating, exit codes, usage errors, `status` / `report` / `list`, config-file overrides, one-shared-controller assertion |
| `test_mining_ui.py` | the entry hook's no-op behaviour, the main-app hook's shape, both view payloads, the round slider, DOT exports, the offline renderer, large-graph trimming |
| `test_mining_stability.py` | two G9 corrections: the preflight probe budget (a reasoning model that needs more than 8 tokens, the `PATHWAY_MINING_PROBE_MAX_TOKENS` override, the named exhaustion) and `max_new_reactions_per_round` stopping a real run while keeping the capped round (since ORCH-745 B1 the cap counts newly admitted reactions only) |
| `test_mining_citation_recovery.py` | ORCH-743 item 2 (G9): an uncited enzyme / gene / species value that occurs verbatim in the claim's own cited participant span is kept and recorded as `verbatim_cited_span`, but only when the field's own citation list is missing; never from an unshown span, an uncited span or paper metadata; an explicit empty list is respected; the request names every per-field citation key. Repair round 1 (bio B-1): a species kept only that way never makes a claim `target_species_supported` (pilot-001's transgenic-host span, a contrast sentence); the target fails closed to `species_ambiguous` |
| `test_mining_search.py` | ORCH-743 item 3 (G9): the frontier item's own terms are required in its query; the result window is `max_results_per_query` (default 25) and NCBI is reached; a second formulation before exhaustion, and immediate exhaustion on zero results; the rendered query strings are pinned. Repair round 1 (G9 at 16fd8a4b): `max_papers_processed` holds inside one result window (review B2); a topic item whose first window was all already-seen papers is asked again over a wider window before it is exhausted (review B3) |
| `test_mining_screen_rescue.py` | ORCH-743 item 4 (G9): the mining-only rescue of production-screen `context_only` / `omics_only` / `off_topic` exclusions; the 21 pilot-001 screened papers in `tests/fixtures/mining/pilot001_screened_papers.json` against the bio-audit (AtLOX2 enzymology in, every must-exclude paper out, even with every name the round-2 frontier held); vetoes are never rescued. Repair round 1 (G9 at 16fd8a4b): route 1 never rescues `off_topic` (bio B-2) and raises the title-only bar too (review B1); route 2's vocabulary is the seeds and `target_species_supported` reactions only, ignores names under 5 characters, matches acronyms in case, and neither route rescues a transgenic-host title (bio B-3); three live false rescues are in `tests/fixtures/mining/orch743_rescue_negatives.json`. Repair round 2 (G9 at 59eb761d, bio B-4): route 1 never rescues a paper whose title names another organism, i.e. the target is only a transgene source (PMC10742046 in `tests/fixtures/mining/orch743_transgene_source.json`; pilot-001's Triticum paper lifted above the bar) |
| `test_mining_iteration2.py` | ORCH-743 iteration 2 (G9 at ecf6b9a0), the pilot-002 audit's five defects: extraction in passage batches with an output budget that fits one (IT2-1); NCBI identity taken from the PMID only, cross-source dedupe and the full-text title check (IT2-2); no exhaustion on a budget-cut result window (IT2-3); the grape host (IT2-4); every ladder rung in `model_raw.jsonl` (IT2-5) |
| `test_mining_evidence_hygiene.py` | ORCH-745 change set A (G9 at f9556288), on pilot-003's four included papers as the run held them and pilot-002's abstract-only records (`tests/fixtures/mining/orch745a_pilot_texts.json`): a review is graded a review from its declared NCBI type, its title, its abstract or its own full text, never from a citation of someone else's review, and a reaction only it supports is `review_article_only` on the decision, the reaction node, the view and the summary (F-2); reference lists, front matter and publisher notes never become spans and never take a passage slot, while every other pilot-003 passage stays retrievable (F-4); one passage, one span (A4); per-paper retrieval ranks the paper's own chunks (A7); a failed EuropePMC lookup is asked again (IT2-N3). Repair round 1 (G9 at 6ceef736, on texts committed under `runs/2026-07-27_*` in `tests/fixtures/mining/orch745a_repair1_texts.json`): a review named only in its JATS subject heading, "Journal Article" alone is not primary, a failed lookup asked twice then marked unverified (A1); a reference list with no heading and ACS front-matter flags never become spans, legends resume at a "FIG." / "Fig" label (A2, A7); EuropePMC-first then PMID is one paper, also after resume (A4); an abstractless record keeps its front-matter abstract (A5); plain-text abstracts (R745A-4); the evidence-level report column (A6); more self-referential phrases (R745A-1); "reviewed-preprint" and "Peer Review" are not review types (R745A-5). Repair round 2 (G9 at 92ab5be9, excerpts of cached EuropePMC texts in `tests/fixtures/mining/orch745a_repair2_excerpts.json`): a table's "References" column no longer swallows the body prose after it, and legends opening "(A)", "|", "clcB", "13-LOX" or "ΔclcB" after a headingless list stay retrievable, because a mention ends a list again (B1, which reverts R745A-6); a headingless list after a short false one is still found; "P < 0.05" survives abstract cleaning (B2); "Review Essay", "REVIEW ARTICLE", "Minireviews" (N2); a short front-matter passage is not an abstract (N4); an explicit PMCID then its PMID is one paper (N1) |
| `test_mining_run_control.py` | ORCH-745 change set C (G9 at f9556288), the pilot-003 audit's run-control defects: `max_new_reactions_per_round` counts newly admitted reactions only, so 26 quarantined leads do not stop a run (B1 / F-7); no round starts after a recorded stop or a spent discovery budget (A5); the frontier cap keeps a quarter of its slots for `missing_species_evidence` items, the most on-route first, and records the cut in `frontier_cut.jsonl` (F-3); a thin answer, by structure only (no field cited, or a schema placeholder echoed; pilot-003 had 7 of 16, raw texts in `tests/fixtures/mining/pilot003_thin_answers.json`), is re-asked once and marked, the record describes the answer kept, and the provider is never an empty string (B2); repair round 1: a reserved species item takes the earlier of its priority position or the next fourth place, so an under-cap frontier keeps the priority order (F3-ORDER); the CLI loads `.env` before `run` / `resume` / `preflight` (F9) |
| `test_mining_species_reading.py` | ORCH-745 change set B (G9 at f9556288), replaying pilot-003's 16 recorded extraction answers and pilot-002's 13-LOX claim (`tests/fixtures/mining/orch745b_pilot_payloads.json`, span texts from `orch745a_pilot_texts.json`): a participant needs a citation or its name verbatim in a cited span, a placeholder echo is no value, tissue needs a cited span (F-5); the species state is read on the claim's own supporting sentences, so GH3.10's knockout evidence admits jasmonic acid -> JA-Ile `in_planta` and the bryophyte review's OPR3 steps are `review`, while every guard row of the bio-auditor's adjudication (T1b, T4-T11, #33) stays out and the full replay admits exactly those three (F-1); the paper's screen organisms are never the claim's species (B3); a same-paper merge decides on the combined evidence, in either batch order (F-6); the prompt pre-fills no boolean (N-5) |
| `test_mining_species_repair1.py` | ORCH-745 change set B repair round 1 (G9 at 467a9faa): every probe sentence the adversarial review found admitted, one parametrised row each, held back: cell-free preparations (B1), negated conversions (B2), hedges (B3), homology without "ortholog" (B4), hosts without the old host words (B5), supplied precursors (B6), the review route's negations, open questions and co-occurrence (B7), foreign gene symbols (B8), narrative citations, interpretive and numbered methods sections (B9, N3); recall the colon split, the single-capital split and the spaced dash cost (N1, F5); a comparison species span (N2); bare X for the negative markers (BIO-745B-1); the legacy floor over three batches (F6); support is the target-species claims only, a review never independence, and a later lead never downgrades (A3); a paper whose `unknown` grade could not be verified reads as a review for the species gate and the evidence level (A1-R1, G9 at 4f8d33a3); plus the controls the reviews marked correct |
| `test_mining_species_repair2.py` | ORCH-745 change set B repair round 2 (G9 at fd53a301): positive species credit narrowed structurally. Every probe sentence the round-1 reviewers saw admitted (red team negation, hedge, application, cell-free / protein-mutant, host, comparison and gene-prefix families; bio N/H/V/T/F/C/RV blocks; code R1-B1..B5), one parametrised `test_repair2_745b_*` row each, is not `target_species_supported`; the `5a_removed` rows are the deliberate removal of rule 5(a) (recall cost); NFKC and the widened allele prefixes (N-1); a lifted node shows its support's gene, tissue, conditions and species text (R1-B6); two batches of one paper are registered before either is judged, so a later inference batch never leaves a stale admission and summary / report agree (R1-N1); per-claim non-supporting rows (R1-N5b); plus the controls the reviews marked correct (loss-of-function recall, the T2/T3 review sentence, the held-back probes) |
| `test_mining_species_cs_d.py` | ORCH-745 change set D (G9 at fd7c737c): every probe the final CS-B reviewers demonstrated, one parametrised `test_cs_d_745_*` row each, is not `target_species_supported`: a product that still forms or rises, a substrate that falls, a no-effect phrase ("comparable to", "indistinguishable", "normally", "still accumulated"), a negated absence form ("was not abolished", "never abolished", "not deficient in") (D1); a participant named only inside the enzyme's own name, another gene's mutant, a protein or transcript measure, and the verbatim arc sentence of PMC13358708 read from `tests/fixtures/mining/pilot001_screened_papers.json` (D2); a loss of function measured in an extract (D3); Brachypodium, Camelina and a Bd-prefixed allele (D4). The negated-verb absence forms ("did not accumulate", "failed to accumulate", "no longer detected") now earn credit (D1's recall gain; bio-r-2 moved here from the repair-2 `5a_removed` rows). Controls: T1, the absence controls, a substrate that accumulates, the bio-auditor's canonical A1-A10, and each in-planta veto in loss-of-function form (code review N-1) |
| `test_mining_species_cs_d_repair1.py` | ORCH-745 CS-D repair round 1 (G9 at 6454ef98): a negated comparison or delay after a negated-verb or "X no" form ("did not accumulate significantly less JA-Ile than", "did not accumulate until 6 h", "produced no less", "accumulated no more") is not an absence (R1); "accumulated less / to lower levels" is a decrease, never a substrate increase, both sides among the subject candidates is ambiguous, a comma-led clause starts a new subject, a whole-plant phenotype is the subject, an "-ase" noun is the enzyme, and a negation after the effect counts (R2); a marker in the JAR1 pathway / background is not of the gene, and the SALK insertion line of GH3.10 binds again (RT-D-4); recall: "accumulated less <product>". Controls: the brief's list. The `both_sides_before_the_effect_is_ambiguous` rows are the documented recall cost |
| `test_mining_integration.py` | the eight required behaviours, end to end |

## The integration fixture

`tests/helpers_mining.py` holds a **synthetic**, jasmonate-flavoured corpus of
nine papers, built to exercise the policy rather than to be biochemically
authoritative. Since ORCH-745 CS-B repair 2 each paper meant to support a
target-species reaction states its result by loss-of-function genetics ("In
Arabidopsis thaliana lox2 knockout lines, 13-HPOT was undetectable in wounded
leaves."), and every scripted claim it supports names that enzyme: neither a
sentence that only names the species nor a plain stated conversion is species
support. It deliberately contains:

| | |
|---|---|
| P1 + P2 | two paper-supported reactions that chemically connect (`α-linolenic acid → 13-HPOT → 12-OPDA`) |
| P3 | a relevant but **disconnected** reaction (`JA-Ile → 12OH-JA-Ile`) |
| P4 | a later paper (round 3) supplying a legitimate biochemical **bridge** |
| P5 | a **wrong-species** reaction (*Zea mays*) |
| P6 | a claim that names the target but cites nothing for it: held back (since ORCH-745 B3 `species_not_stated`, the paper's screen organism is no longer the claim's species) |
| P7 | a paper **conflicting** with P2 about the enzyme |
| P8 | **independent support** for P1's reaction from a second laboratory |
| P9 | a paper the model **cannot finish reading** (truncated completion) |

`ScriptedPaperSource` serves papers on a per-round schedule — the test advances
its round from the controller's own `round_start` progress event, which is how a
later paper can supply the bridge. `ScriptedModel` cites only evidence ids that
are actually present in the prompt, so the citation check in
`extraction.build_claims` is exercised rather than bypassed.

## What the integration test asserts

1. connected reactions enter the primary graph, via the shared metabolite;
2. the disconnected reaction starts in a candidate subgraph;
3. wrong-species evidence is retained but never enters the target-species graph
   — and the claim with no species evidence (P6) is held back too;
4. the conflict is visible, both positions kept, the disputed attribute cleared
   on the node;
5. the round-3 bridge promotes the component, and the original discovery round
   survives;
6. every admitted reaction has an intact `reaction → claim → span → paper → id`
   chain, and independent papers are counted rather than collapsed;
7. every round directory is complete and the growth-by-round record matches;
8. no PWML, SBML or XML exists anywhere, both artifacts say `produces_pwml:
   false`, and no mining module references the exporter.

Plus: external validation is `not_evaluated` everywhere, the run stopped for a
named reason, the frontier asked new questions each round, and every evidence row
is quotable with a resolvable source identifier.

## The feature-disabled regression

The production regression gate, run on this branch with the feature off:

```
SMOKE (22 files)   508 passed in 44.45s
```

exactly the expected count. Report:
`docs/pathway_mining/evidence/001-smoke-feature-disabled.json`
(exit 0, 0 surviving processes, cleanup success, measured tree verified,
`violations: []`).

`T-107`, `T-108` and `T-109` were **not** rerun and are untouched.

## Live testing

None was performed. A tiny bounded live smoke test would be acceptable if
credentials, services, locks and repository policy allowed, but the fixture-based
integration is sufficient for verification and spends nothing. Evaluating output
*quality* against real literature is separate work needing explicit cost
authorization — see [`known_limitations.md`](known_limitations.md).
