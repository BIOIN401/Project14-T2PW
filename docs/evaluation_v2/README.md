# Evaluation v2 (ORCH-742c) — definitions, gold labels, and how to reproduce

**A NEW, separately versioned evaluator (`eval-v2.0`).** It re-runs nothing and edits nothing: T-107,
T-108, T-109 and every committed benchmark report keep their meaning. These numbers come from a different
instrument and are **not** a re-score, replacement or correction of any earlier result. Production is
frozen (D-102); no production module imports this code (`tests/test_eval_v2_metrics.py::
test_no_production_module_imports_evaluation_v2`). A produced PWML is not treated as evidence of
biological correctness anywhere here.

| file | what it is |
|---|---|
| `src/t2pw/bench/eval_v2/gold_v2_supported_reactions.json` | the new supported-reaction labels (a supplement keyed by case id) |
| `src/t2pw/bench/eval_v2/{matcher,gold,reactions,rag,repro,report,fetch}.py` | the evaluator |
| `docs/evaluation_v2/sample_report.json` / `SAMPLE_REPORT.md` | the versioned sample report (offline, from git-tracked artifacts) |
| `docs/evaluation_v2/retrieval_live.json` | live retrieval measurement (production retriever, real embeddings) |
| `docs/evaluation_v2/READINESS.md` | verdict for the upcoming results run |
| `tests/test_eval_v2_*.py` | new acceptance tests (+ one strict-xfail finding) |

## 1. Gold: four newly complete positive cases

**Where, and why there.** `pinned_v1.json` is byte-identical (product-owner decision 1). The new labels are
a **supplement** in `src/t2pw/bench/eval_v2/`, not in `src/t2pw/bench/gold/`: `goldset.pinned_gold_set_path()`
only ever returns `pinned_v1.json`, but `docs/pwml_recovery_sprint/evidence/c122_cohort_candidates.py` globs
`gold/*.json`, so a file there would be picked up. A supplement keyed by case id also keeps every pinned_v1
field (organism, forbidden identifiers, notes) single-sourced. `test_supplement_is_invisible_to_every_existing_gold_reader`
pins that `gold/` still holds only `pinned_v1.json`.

**Source texts.** Each case was read against the EuropePMC full-text XML (read-only fetch, no key, no LLM,
`python -m t2pw.bench.eval_v2.fetch`, run under the bounded wrapper, G11 `ORCH-742c/01`). The texts live
OUTSIDE the repo (`C:/t/eval-fulltext/`); the supplement commits only PMCID, URL, fetch date and the sha256
of the XML and of the flattened text, and `gold.load_fulltext` refuses a text whose hash differs. Every quote
(reactions, off-pathway statements, summaries, exclusions) is verified in the fetched text, and every
**in-scope** quote is also verified in the git-tracked `01_source_text.txt` of the latest committed run, so
scoring is reproducible from the repository alone. (PMC12421875's committed text lacks two
off-pathway/exclusion quotes, matching pinned_v1's recorded Results discontinuity; no scored reaction depends
on them.)

**What "complete" means (v2).** The whole text was read; every transformation it states with both sides
named is listed — in `reactions` (requested pathway, declared scope), `off_pathway_reactions` (another
pathway, an assay reporter, transport, binding) or `summary_statements` (an overall multi-step conversion).
Statements naming a catalyst without both sides, covalent enzyme/carrier-bound states, protein
modification and regulation go to `excluded_statements` with the reason. **Per D-093 this is a claim about
the target paper only; it never by itself makes an unmatched row "invented"** (see § 3).

| case | species | in-scope | off-pathway | summaries | scope in one line |
|---|---|---|---|---|---|
| PMC12096016 | *E. coli* | 5 (EntC, EntB, EntA, EntE, EntF) | 4 (MenD, LDH assay, TolC secretion, Fe chelation) | 2 | DHB + assembly arms; ArCP loading and EntD phosphopantetheinylation excluded as protein-bound states (the D-091 audit's rule for ThDP adducts) |
| PMC12657337 | *E. coli* | 3 (HepPPS, MenA, UbiE) | 2 (Idi IPP→FPP as stated, acs) | 0 | the paper's own "MK-7 pathway" module (Fig. 1); MVA/precursor modules stated only by enzyme names |
| PMC12312563 | *L. monocytogenes* | 1 (MenD) | 0 | 0 | adopts the D-091 audit biology (upheld by D-092), re-verified on the fetched text: MenD 133, other Men symbols 0, free chorismate 0 |
| PMC12421875 | *L. lactis* | 9 (MenF, D, H, C, E, B, I, A, G) | 3 (biosensor Fe reduction, ferrozine, DHNA efflux) | 1 | MenH–MenI stated through the steady-state model's letter-assigned equations (5)–(11); MenG's wrong verb noted, its DMK→MK pair scored |

**Not completed, and why.** PMC12782028: MSMO1's catalysis is stated on a compound *class* ("4,4-dimethyl and
4alpha-methylsterols") with no named product, which a two-sided signature cannot represent honestly.
PMC12856317: its protease statements are degradation of ALAS proteins, not metabolite conversions; a scope
ruling on those belongs to the product owner. PMC12444477 (lipid A, a long regulation paper) and
PMC12452463 (a broad review whose route PRODUCT_CONTRACT § 13 records as chemically broken) were not attempted in this lane. The two
negative controls already carry `max_retained_reactions` ceilings. **So: 4 newly complete cases.**

## 2. Strict matcher (`matcher.py`)

Equality of a normalized key; no containment in either direction. Normalization, in full: reuse of
`goldset.normalize_name` (NFKC, unicode dashes, Greek letters **spelled out**, casefold, a trailing charge sign
dropped, punctuation → one space; digits and locants kept) + `-ic acid` → `-ate` (protonation form) +
`pyrophosphate` → `diphosphate`. Candidate-side reading rules: a trailing parenthetical is an alternative
name (`oxidoreductase (entA)`); for **enzymes only**, a trailing `complex` is dropped and a complex also
stands for its own `components` (names and gene names). Stereo descriptors are never stripped; `serine` meets
`L-serine` only through a declared, justified alias.

**Congeners are not credited** (`test_strict_matcher_never_credits_a_congener`): `menaquinone`/`menaquinone-7`,
`lipid A`/`Kdo2-lipid A`, `coenzyme A`/`acetyl coenzyme A`, `DHNA`/`DHNA-CoA` — all credited by the old
lenient matcher — and `CoA`/`acetyl-CoA`, `PGE2`/`PGE1`, `2,3-DHB`/`2,5-DHB`, alpha/beta-ketoglutarate,
`DMK-7`/`DMK-8`. The old matcher's result is reported beside v2 on the same rows in every leg.

**Known false-credit risk (measured, 1 row in the corpus):** the parenthetical rule credits
`heptanoyl pyrophosphate (HepPP)` (`runs/2026-07-28_2122`, PMC12657337) through its correct abbreviation
although the spelled-out name is a different molecule.

## 3. Reaction metrics (`reactions.score_leg`)

Scored bucket: `processes.reactions` (transports are not reactions and are not scored). Rows with identical
input/output key sets are collapsed first (R-D092-2: a duplicated row is not a second true positive).

Row classes, in precedence order: `target_paper_supported` (all **defining** participants of an in-scope gold
reaction on the right sides; reversible gold may match backwards) · `paper_stated_off_pathway` ·
`paper_stated_summary` · `partial_match` (≥ 1 defining substrate **and** ≥ 1 defining product of an in-scope
reaction) · `external_rag_supported` / `indeterminate_external` (no gold match; the committed R-D092-1
classifier `evidence/rd092_1_reaction_lineage.py`, reused by path, finds an external source) · `unsupported`
(everything else: the exhaustively read paper does not state it and nothing external defends it).

**Two tables, never summed (D-093).**

| metric | definition |
|---|---|
| reaction precision | TP rows / rows claimed as paper extraction (every row except those whose lineage names an external source and that match no gold statement) |
| reaction recall | in-scope gold reactions matched / in-scope gold reactions |
| reaction F1 | harmonic mean of the two |
| unsupported rate | `unsupported` rows / all retained (deduplicated) rows; the **upper bound** adds `indeterminate_external` |
| off-pathway / summary / partial | counted apart: stated by the paper, not invented, not correct |
| substrate / product accuracy | over in-scope gold reactions: defining substrates (products) present on the aligned row (the full match, else the best partial), / all defining substrates (products) |
| enzyme accuracy | over matched gold reactions with a gold enzyme: correct / (correct + wrong + missing). **Scored apart**: a wrong or missing enzyme never fails an otherwise correct reaction |
| species accuracy | binomial (genus + epithet) of the row's biological state vs the case's actual organism; `unknown` (no species recorded) is excluded from the denominator and reported |
| evidence-support rate | rows whose evidence carries a quote-sized span (≤ 2,000 chars) found verbatim (folded) in the committed paper text / all rows. Longer strings are `dump_only` (R-D092-1 fact 4) |
| minor cofactors | gold `cofactor`-role participants missing from a matched row, listed apart from defining chemistry |

Aggregates are micro-averaged, per population (canonical `final_mapped.json` vs fallback
`merged_payload.json`, F-177) and never across populations.

## 4. Mapped-ID correctness (`reactions.id_correctness`)

Per entity with a mapped identifier: the **claim** is the entity's own name (never its `synonyms`, which may
be written from the record under test); the **reference** is every name that committed tables and ID-keyed
records give for that identifier — `data/id_mapping_cache.json`, `data/enrichment_cache.json` (UniProt and
KEGG records fetched by identifier), `data/pathwhiz_id_db.json`, and the payload's own mapped-record candidate
and ID-keyed enrichment block. Verdicts: `correct_identity` (a reference name equals the claim; spacing-only
differences tolerated) · `wrong_identity` (reference names exist, none agrees) · `unresolved_or_novel` (no
identifier, novel/generated) · `not_evaluable_no_reference`. Nothing is fixed.

**Calibration, by hand, on the 21 T-107/T-108 `wrong_identity` flags:** 5 are genuine — `enterobactin
synthase` (the EntB/D/E/F assembly, or EntF in PMC12096016's own usage) mapped to EntE / P10378 in 4 legs, and
`cardiolipin` (a class) mapped to one molecular species; 16 are synonym gaps or policy-allowed states
(`2,3-dihydro-2,3-dihydroxybenzoate` vs its stereo-named record ×6, `(δ-)aminolevulinic acid` vs
5-aminolevulinate ×6, `UDP-GlcNAc`, `laurate`, `aminolevulinic acid synthase 2`, and `holo-EntB`, which D-035
lets carry the parent's accession). **Measured precision of the flag: 5/21 ≈ 24%.** It is a candidate list for
adjudication, not a verdict; ChEBI synonyms (all 265 cached ChEBI lookups are errors) would be the obvious
missing reference.

## 5. RAG

**Live (`rag.live_retrieval`, `retrieval_live.json`).** The production chain — `ingest.chunk_paper`,
`embed.Embedder`, `store.MemoryVectorStore`, `ingest.build_hybrid_scorer`, `retrieve.query_for_gap` — over the
latest committed `01_source_text.txt` of all ten gold papers (232 chunks); one query per in-scope gold
reaction, built by the production query builder from a gap describing that reaction as missing. The embedding
cache is written OUTSIDE the repo; no chat/completion call is made. **Measured with `embedding_mode: api`
(LM Studio, `text-embedding-nomic-embed-text-v1.5`)**; the mode is recorded on every run. Two relevance labels,
reported side by side: `target_quote` (a chunk of the target paper holding the gold quote — strict: the same
chemistry in another corpus paper is *not* relevant) and `participants_named` (a chunk of any paper naming every
defining participant). **Negative queries**: each positive query restricted to the chunks of another complete
case whose exhaustive gold states no such reaction. The production retriever has **no score floor** and its
hybrid score is min-max normalized per query (top hit ≈ 1.0), so retrieval-level negative rejection is 0 by
construction; rejection happens only at admission.

**Archived (`rag.archived_failure_split`).** Over committed `rag_admission_report.json` of the complete cases'
legs, for each in-scope gold reaction **not already in that leg's strict Stage-1 payload**: `A` no persisted
retrieved span names every defining participant and no candidate matches · `B` a span names them all but no
candidate matches · `C` a candidate matches and every match was rejected · admitted. Per **accepted** candidate:
`D` = its own span does not name all its inputs and outputs; accepted candidates that are simply not the target
paper's chemistry are counted apart as `accepted_external_biology` (not an error, D-093). Limits: only candidate
records (with clipped spans) are persisted, not full chunk lists, so `A` means "not visible in what was
persisted"; legs whose report hit `max_report_entries` are a separate population.

**Per chunk** (live hits and persisted candidates): provenance present, `rag.provenance.validate_provenance` on
the reaction a transcription from that chunk would become, and a valid source reference (`PMC…`, PMID, DOI,
`seed_paper`, or an http(s) URI).

## 6. Un-gated RAG influence audit (`reactions.leakage_audit`) — report only

A reaction row of the Stage-2/merge output (`merged_payload.json`; the canonical `final_mapped.json` is audited
separately) is an **un-gated RAG-influenced candidate** when (1) its key sets match no row of the same leg's
strict `stage1_payload.json`, (2) it corresponds to retrieved evidence — `admission_match` (a persisted
candidate of either verdict with the same key sets), `span_in_row_evidence` (a retrieved span ≥ 40 folded chars
inside the row's evidence) or `participant_only_in_rag` (a participant absent from the seed paper text but named
in a retrieved span) — and (3) no **accepted** admission record has its key sets. Each candidate records whose
retrieved text it matches; `external_evidence` separates other papers from retrieval of the seed paper itself.
Nothing is changed.

## 7. Reproducibility metadata (`repro.completeness`)

Field-by-field presence over every committed `model_attempt` record, with the exact file/function where a
future hook would go for each missing field (no hook is added). Notable: `response_hash` is present as a key and
**empty on all 663 committed records** — `CompletionDiagnostics.note()` publishes the attempt row before
`chat_detailed` assigns the hash; `request_hash` hashes the messages only (so it is the prompt hash; nothing
hashes model + temperature + max_tokens); temperature is recorded only per audit round.

## 8. Fallback embedding cache (`tests/test_eval_v2_embedding_cache.py`)

**Confirmed.** `Embedder.embed_one` caches the lexical fallback vector under `_cache_key(real model, text)`, so
a run made while the endpoint was down poisons the cache and every later online run is served lexical vectors as
cache hits. With `RAG_EMBEDDING_DIM` unset (this `.env`) the fallback is 256-wide against the model's 768, so
`MemoryVectorStore.query` silently degrades those pairs to lexical overlap; with a matching dimension it cannot
tell at all. Pinned by a characterisation test and a `xfail(strict=True)` test of the correct behaviour, which
flips loudly when fixed. Production cache behaviour is unchanged.

## 9. Reproduce

Every command runs under `docs/pwml_recovery_sprint/evidence/bounded_run.py` (G11).

```bash
PYTHONPATH=<tree>/src python -m t2pw.bench.eval_v2.report evaluate --repo . --out docs/evaluation_v2 \
    --live-json docs/evaluation_v2/retrieval_live.json          # offline, deterministic
PYTHONPATH=<tree>/src python -m t2pw.bench.eval_v2.report live-retrieval --repo . \
    --cache <dir outside the repo>/emb.json --out docs/evaluation_v2/retrieval_live.json   # embeddings only
PYTHONPATH=<tree>/src python -m t2pw.bench.eval_v2.fetch --out <dir outside the repo> PMC12096016 ...
```

`tests/test_eval_v2_report.py` re-runs `evaluate` and requires byte-for-byte agreement with the committed
report, so every number in `SAMPLE_REPORT.md` is reproducible from git-tracked artifacts.

## 10. Known unevaluated areas

- Reaction P/R/F1 exists only for the four complete cases; any other paper needs its own exhaustive gold first.
- Stoichiometry, directionality beyond the reversible flag, compartments and transports are not scored.
- The strict matcher has no synonym source: a correct row written with an unlisted synonym is `unsupported`
  or `partial_match` until a gold alias is added (the per-row detail shows every such row).
- Retrieval rank on archived runs is not re-derived here (see `evidence/rd093_rag_metrics.py`, which exists and
  was not rebuilt); v2's archived view is the A/B/C/D outcome split.
- Only T-108 (`runs_verify/2026-09-01_1612`) has committed `LEG_TRACE.jsonl`; older runs record no per-call
  metadata at all. T-109's run tree is not committed and is therefore not evaluated.
