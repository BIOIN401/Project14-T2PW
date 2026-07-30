# Paper eligibility screening

Which papers are allowed into the expensive pipeline, and why.

Implementation: `src/t2pw/rag/eligibility.py`.
Wired at: `src/t2pw/batch/fetch.py` (before `fetch_full_text`).
Stage-0 boundary: `src/t2pw/batch/driver.py` (`_reconcile_stage0_scope`).
Tests: `tests/test_paper_eligibility.py`,
`tests/test_paper_eligibility_corrections.py`, and the Stage-0 integration tests in
`tests/test_batch_driver.py`.
Dry-run tool: `scripts/eligibility_dry_run.py`.

---

## The problem this solves

A topic query is a blunt instrument. From `runs/2026-07-28_2122`, searching
`"lipid A biosynthesis" AND "Escherichia coli"` returned a Fournier's-gangrene
case report and a river-resistome surveillance study; `"enterobactin
biosynthesis"` returned two poultry/turkey ESBL virulence surveys; `"heme
biosynthesis"` returned a COVID-19 lncRNA comorbidity study and a gene-set
evolution tool. Every one of them had retrievable full text, so every one of them
was downloaded and run through the app **twice** (strict + research), and then
appeared in the morning triage as an extraction failure — sending debugging into
the extractor for papers that were never about the requested pathway.

Two separate defects were behind it:

1. **No gate.** Retrievable full text was the only admission criterion.
2. **Requested metadata was stamped as observed metadata.** Every acquire fetcher
   passed `organism=organism` — the organism the *search asked for* — onto each
   `CandidatePaper`. Downstream that is indistinguishable from a paper that
   actually reported that organism, and it pinned `select._organism_score` at
   1.0, so the one organism check that existed could never fire.

## Requested vs observed

These are different facts and now live in different fields:

| | field | meaning |
|---|---|---|
| request | `requested_pathway`, `requested_organism` | what the batch asked for; **frozen** |
| observation | `observed_pathways`, `observed_organisms` | what the paper's own title/abstract reports |
| comparison | `organism_match` | `match` / `genus_level` / `mismatch` / `unknown` |

A paper that names no organism keeps `observed_organisms == []` and
`organism_match == "unknown"`. The requested organism is **never** copied in to
fill that gap: "we asked for E. coli" and "this paper is about E. coli" must not
look the same downstream.

`RequestedScope` is a frozen dataclass. That is the enforcement mechanism, not a
convention — an assignment raises rather than silently re-pointing the batch.

Three places had to be closed for the split to actually hold:

* **`_as_batch_paper` reads observed fields only from the decision.** An earlier
  version promoted `candidate.organism` into `observed_organisms`, which
  re-imported the stamped value from any legacy cache row. Reading from one source
  also makes `BatchPaper.observed_organisms`, `organism_match` and
  `eligibility["observed_organisms"]` unable to disagree —
  `BatchPaper.scope_disagreements()` asserts that rather than assuming it, and
  `eligibility_summary` surfaces any violation in the plan.
* **The acquisition cache is versioned** (next section).
* **Stage 0 gets a production caller** (see "Stage 0 may observe, not overwrite").

## Legacy acquisition caches (schema 1 → 2)

Files in `data/rag_index/acquire_cache/` written before 2026-07-29 carry no
`schema_version`, and every candidate row's `organism` is the organism the *search*
asked for. Reading one as observed evidence would reintroduce exactly the false
claim this change removes — the cached Fournier's-gangrene row asserts
`"Escherichia coli"`.

`migrate_cached_payload` runs **on read**, in memory, and demotes only the organism
claim: `organism` → `requested_organism`, observed fields cleared, everything else
(ids, titles, abstracts, URIs, years) kept. A legacy cache therefore stays usable
offline instead of being thrown away. `search_candidates` reports
`cache_schema_migrated` / `cache_schema_version` in its `status` dict, and
`CandidatePaper.from_dict` applies the same demotion for any other reader.

Detection is on key *presence* (`is_legacy_candidate_row`), not value: the current
serializer always writes all five scope fields via `asdict`, so "none of them
present" is a reliable legacy signal, and a modern row's genuinely-observed
`organism` survives untouched.

## Species matching

`match` means the same **species**. Everything else is represented separately:

| requested | observed | verdict | why |
|---|---|---|---|
| `Escherichia coli` | `Escherichia coli` | `match` | exact binomial |
| `Escherichia coli` | `Escherichia coli K-12 MG1655` | `match` | strain-qualified form of the same species |
| `Escherichia coli` | `Escherichia fergusonii` | `genus_level` | same genus, different species |
| `Escherichia coli` | `Escherichia colicinogenes` | `genus_level` | word boundaries: `coli` is not a prefix match |
| `Bacillus subtilis` | `Bacillus cereus` | `genus_level` | same genus, different species |
| `Escherichia coli` | `Escherichia` | `genus_level` | **a bare genus never infers a species** |
| `Bacillus subtilis` | `Listeria monocytogenes` | `mismatch` | different genus |
| anything | (nothing named) | `unknown` | not a mismatch |

`genus_level` is *permitted* evidence — a related taxon often carries the same
mechanism — but deliberately not a match: it scores 0.25 instead of the 1.0 species
bonus, raises an explicit warning, and sets `needs_manual_review`. Only `mismatch`
triggers the organism veto.

The organism lexicon holds **species-level aliases only**; bare genus names live
separately in `_KNOWN_GENERA` and drive a binomial scan, so a species the lexicon
has never seen (`Escherichia fergusonii`) is observed as itself rather than
collapsing onto a lexicon neighbour. A bare genus mention is dropped when a species
of that genus is also observed. Across several observations the strongest verdict
wins (`match` > `genus_level` > `mismatch`), because a paper that works in E. coli
and validates in mice reports both.

## Selection rules

Screening reads **title + abstract only**, deliberately: the whole point is to
decide before paying for the full text. `score = positives - negatives`.

### Positive evidence

| category | weight (each / cap) | what matches |
|---|---|---|
| `pathway_alias` | 2.0 / 2.0 | the requested pathway name and its aliases (`heme biosynthesis`, `heme synthesis`, `haem biosynthesis`, …) |
| `pathway_term` | 1.5 / 3.0 | expected enzyme/metabolite terms for that pathway (`lpxc`, `mend`, `ppox`, `entb`, `squalene`, …) |
| `reconstruction_term` | 1.0 / 2.0 | pathway-reconstruction / enzyme-characterization language (`reconstitution`, `biosynthetic gene cluster`, `kinetic characterization`, `crystal structure`, `heterologous expression`, …) |
| `organism_match` | 1.0 / 1.0 | the requested organism or one of its taxonomy aliases is present |
| `mechanism_term` | 0.75 / 1.5 | biochemical reaction/mechanism language (`catalyzes`, `substrate`, `kinetics`, `intermediate`, `decarboxylation`, `inhibitor`, …) |
| `enzyme_term` | 0.5 / 1.0 | any `-ase` word (stop-listed against `increase`/`disease`/…) or an EC number |
| `pathway_head` | 0.5 / 0.5 | the bare head compound (`cholesterol`, `enterobactin`) — see below |

### Negative evidence

| category | weight | decisive? |
|---|---|---|
| `incompatible_organism` | 3.0 | yes (`organism_veto`) |
| `clinical_case_report` | 3.0 | yes (`negative_veto`) |
| `epidemiology_survey` | 3.0 | yes (`negative_veto`) |
| `animal_virulence_survey` | 3.0 | yes (`negative_veto`) |
| `software_only` | 3.0 | yes (`negative_veto`) |
| `pathway_only_in_background` | 1.5 | no |
| `pathway_context_only` | 1.5 | fails the anchor requirement |
| `no_mechanistic_pathway_terms` | 1.5 | fails the anchor requirement |

Two of the negatives are conditional, because the unconditional form was wrong:

* `animal_virulence_survey` requires an animal host **and** virulence/typing
  language **and** no pathway anchor. A genuine lipid A paper done in chickens is
  not this shape; the requirement is specifically the *unrelated*-pathway survey.
* `software_only` requires a software term **and** no wet-lab term. A web server
  paper with in-vitro validation is not software-only.

### The anchor requirement: evidence must be *local*

A pathway alias occurring somewhere and a generic word like "mechanism" occurring
elsewhere is not evidence of mechanistic content. Eligibility requires a **pathway
anchor**, judged in the text immediately around each pathway mention — the sentence
containing it, or `local_window_tokens` either side:

* **(a)** a pathway-specific enzyme/metabolite term (`lpxc`, `mend`, `ppox`,
  `entb`, `squalene`), mentioned outside screening context; or
* **(b)** any pathway mention **with strong reaction/enzyme evidence local to it**.

Three further restrictions, each earned from a real false positive:

* **`pathway_head` never anchors.** Naming the molecule is not evidence the paper is
  about its biosynthesis; treating it as an anchor admitted four
  cholesterol-signalling and cholesterol-in-cancer papers whose only link to
  "cholesterol biosynthesis" was the word "cholesterol".
* **"Strong" excludes the framing vocabulary.** `mechanism`, `mechanistic`,
  `inhibition`, `inhibitor`, `feedback`, `flux`, `reduction`, `oxidation` and any
  bare `-ase` word are not strong: they appear in the framing sentence of almost any
  molecular-biology abstract, and a gene symbol in a hit list ends in "-ase" too.
  `substrate`, `catalyzes`, `intermediate`, `kinetics`, `active site`, `purified`,
  `crystal structure` and `reconstitution` are. Also excluded: `biosynthesis` /
  `biosynthetic`, which sit *inside* the aliases themselves and so would let every
  alias certify itself.
* **A screen's gene list is not an anchor.** `SQLE`, `CYP51A1` and `DHCR24` named
  among differentially expressed proteins or transcripts satisfied (a) literally
  while saying nothing about the paper's subject. When a document announces itself
  as a screen (`_OMICS_TERMS` anywhere in it) path (a) survives only if the pathway
  is named in the **title** — the line between PMC13264790 ("...analysis of
  PPOX...", so PPOX is its subject) and PMC12113831 (titled about oak-leaf extract,
  with SQLE only among its proteomic hits). Screening vocabulary is checked at
  document level precisely because both of those papers put "proteomic analysis" /
  "transcriptome sequencing" in the methods framing sentence and the gene names many
  words later, so no bounded window around the gene contained it.

Configurable via `eligibility_require_pathway_anchor` and
`eligibility_local_window_tokens`.

### Classification

Every decision carries one:

| value | meaning |
|---|---|
| `mechanistic` | anchored — admitted, subject to the score and the negatives |
| `context_only` | the pathway is named; nothing local to that mention is a reaction or an enzyme |
| `omics_only` | the pathway appears only as screening output, or there is no anchor and the paper reads as a screen |
| `off_topic` | no pathway mention at all |

### Outcome resolution, most decisive first

1. screening disabled → `eligible`
2. `scope.pinned` → `pinned_override` (score bypassed; mismatches still recorded as warnings)
3. title shorter than `min_title_chars` and no abstract → `insufficient_metadata`
4. `organism_match == mismatch` → `ineligible_organism`
5. any decisive negative → `ineligible_pathway`
6. no pathway anchor → `ineligible_pathway`
7. `score < threshold` → `ineligible_pathway`
8. otherwise → `eligible`

`duplicate` and `no_full_text` come from the fetcher, and `scope_conflict` from
`apply_stage0_observation`; all eight share one vocabulary
(`ELIGIBILITY_OUTCOMES`) so a skip record never has to be parsed out of prose.

Step 4 before step 5 is deliberate: `ineligible_organism` means "re-run this topic
against the other organism and it is a good paper", which is a different and more
useful statement than "off topic".

## Thresholds

Every tunable lives in `RAG_DEFAULTS` (`src/t2pw/config.py`) and is readable from
the environment, so a run's screening behavior is reproducible from its config.

| key | env var | default |
|---|---|---|
| `eligibility_enabled` | `RAG_ELIGIBILITY_ENABLED` | `True` |
| `eligibility_min_score` | `RAG_ELIGIBILITY_MIN_SCORE` | `2.0` |
| `eligibility_title_only_min_score` | `RAG_ELIGIBILITY_TITLE_ONLY_MIN_SCORE` | `1.5` |
| `eligibility_min_title_chars` | `RAG_ELIGIBILITY_MIN_TITLE_CHARS` | `20` |
| `eligibility_require_pathway_anchor` | `RAG_ELIGIBILITY_REQUIRE_ANCHOR` | `True` |
| `eligibility_organism_veto` | `RAG_ELIGIBILITY_ORGANISM_VETO` | `True` |
| `eligibility_negative_veto` | `RAG_ELIGIBILITY_NEGATIVE_VETO` | `True` |
| `eligibility_review_margin` | `RAG_ELIGIBILITY_REVIEW_MARGIN` | `0.5` |
| `eligibility_local_window_tokens` | `RAG_ELIGIBILITY_LOCAL_WINDOW_TOKENS` | `12` |
| `eligibility_candidate_ceiling` | `RAG_ELIGIBILITY_CANDIDATE_CEILING` | `60` |
| `eligibility_stage0_conflict_aborts` | `RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS` | `True` |

`title_only_min_score` is the lower bar for a screen with no abstract. A title
carries a fraction of the evidence, so scoring it against the full bar would
reject nearly everything; such decisions are marked `provisional`.

**No LLM.** The scorer is fixed lexicons, word-boundary regexes and arithmetic —
no network, no model, no clock, no randomness. A model here would make the gate
non-reproducible and would cost exactly what the gate exists to save. There is
deliberately no LLM-based paper selector.

## Stage 0 may observe, not overwrite

`apply_stage0_observation(scope, stage0_context)` folds Stage 0's reading of a
paper into `ObservedContext` — `observed_pathways`, `observed_organisms`,
`aliases`, `ambiguities` — and returns the requested scope **unchanged**. It has
no code path that writes `requested_pathway` or `requested_organism`.

When Stage 0 reads a pathway or organism that strongly contradicts the request it
returns a non-empty `conflicts` list, and the caller marks the paper ineligible or
`scope_conflict`. What it must never do is adopt the paper's apparent scope as the
batch's request: that quietly changes what the whole batch is about. A *related
taxon* (`genus_level`) is not a conflict.

### The production caller

`driver._reconcile_stage0_scope`, in `batch/driver.py`, immediately after stage 1
succeeds. That is the first and only point in a run where both halves exist: the app
has stored its Stage-0 context in `st.session_state["pathway_context"]` and the plan
record next to it says what the batch asked for. It:

1. builds the `RequestedScope` from the plan record (`requested_pathway` /
   `requested_organism`, falling back to the legacy `topic` / `organism` spellings);
2. calls `apply_stage0_observation`, and asserts the request came back unchanged;
3. records the observation on the run (`RunOutcome.observed_context`, plus
   `stage0_observed_*` counts) whether or not there is a conflict;
4. on conflict: writes a `scope_conflict.json` artifact naming requested vs
   observed, adds the `scope_conflict` issue code, and stops the run with
   `status="scope_conflict"` — which `report._norm_status` folds to
   `STATUS_INELIGIBLE`, so it is not a failure and never enters the ranked
   fix-list. `RAG_ELIGIBILITY_STAGE0_CONFLICT_ABORTS=false` downgrades the stop to
   a warning.

Stopping (rather than only annotating) is the point: the screening gate decided from
a title and abstract, Stage 0 has read the whole paper, and continuing would spend
the audit, the DB mapping and the export on a paper that is not the one requested.

A pinned paper is exempt — a human chose it, so there is no request to contradict —
and an app that stores no readable Stage-0 context changes nothing.

## Ineligible is not a failure

Nothing was attempted, so nothing failed. Concretely:

* A screened paper becomes a `skipped.json` record and gets **no** paper folder
  and **no** manifest row, so it cannot reach the triage at all.
* Belt and braces in `batch/report.py`: `_norm_status` folds `ineligible_*` /
  `insufficient_metadata` / `scope_conflict` to `STATUS_INELIGIBLE`, which
  `_is_failure` excludes and `triage_class` files as `incomplete`. Before this,
  those spellings fell through to `STATUS_UNKNOWN`, which *was* counted as a
  failure.

## Plan artifacts

`plan.json` carries, per accepted paper, an `eligibility` object with `score`,
`threshold`, `classification`, `matched_positive`, `matched_negative`,
`requested_pathway`, `requested_organism`, `observed_pathways`,
`observed_organisms`, `organism_match`, `outcome`, `reason`, `provisional`,
`needs_manual_review`, `warnings` and `screening_input`. `skipped.json` carries the
same object for every rejected paper.

A top-level `plan["eligibility"]` block records the thresholds the run was screened
with, the per-outcome and per-classification tallies, the organism-match
distribution, the ids flagged for manual inspection, any `scope_disagreements`, and
the acquisition funnel — so a plan explains the papers that are missing from it
rather than leaving their absence to be inferred.

## Persisted screening input

`screening_input` holds the exact text a decision was taken on: `title`, the
`abstract` bounded to `MAX_PERSISTED_ABSTRACT` (4000 chars), `abstract_chars`,
`abstract_truncated`, `abstract_source`, `abstract_authoritative`, and
`abstract_sha256` **of the full supplied abstract**. The decision is deliberately
taken on the persisted slice, not on any unpersisted tail, so replay uses exactly
the same evidence. The length and digest retain an audit link to the original
input when it was longer than the bound.

That makes a rejection reproducible offline from `skipped.json` alone: re-screen the
stored title and abstract at the recorded thresholds and the same outcome, score and
observed organisms come back.

`scripts/eligibility_dry_run.py` picks its abstract in this order, and records which
it used:

1. `--abstracts` — an explicit `{paper_id: abstract}` map.
2. `plan_screening_input` — the persisted abstract from an accepted `plan.json`
   record or a screened rejection in `skipped.json`. **This is the normal path**
   for any run written on or after 2026-07-29, and needs no extra input.
3. `acquisition_cache` — a locally cached publisher abstract for a legacy plan.
4. `derived_from_stored_full_text` — a bounded lead window recovered from
   `papers/<slug>/01_source_text.txt`, for legacy plans that persisted no abstract.
5. `title_only`.

Sources 1–3 are authoritative. A derived abstract keeps the verdict marked
`provisional` and subject to the provisional `needs_manual_review` rules, because
`01_source_text.txt` is a flat XML-to-text dump with no abstract markup:
`derive_abstract` jumps past an `Abstract`/`Summary` heading when there is one, else
finds where running prose starts, then drops licence and contributor-role sentences.
For `runs/2026-07-28_2122`, 27 publisher abstracts are recovered from the local
acquisition cache and the remaining paper uses the derived full-text proxy. The
dry-run remains offline: it never refreshes or queries that cache over the network.

## Filling the requested count

The gate is selective — it rejected 21 of the 28 papers in the 2026-07-28_2122
titles — so the old fixed `want * 3` over-fetch, calibrated for a fetcher that took
every hit with full text, would silently under-deliver every topic. Instead
`fetch_papers` escalates per topic: it requests `3x`, then doubles, examining only
candidates it has not seen, and stops when

* the requested count is filled (`stop_reason: "filled"`),
* the search stops yielding new records (`"source_exhausted"` / `"no_candidates"`),
* `eligibility_candidate_ceiling` candidates have been examined
  (`"candidate_ceiling"`), or
* the run-wide `--limit` is reached (`"run_limit"`).

Pass a dict as `stats` to receive the funnel: `requested`, `examined`, `eligible`,
`ineligible`, `duplicate`, `no_full_text`, `fetch_failed`, `accepted`, per topic and
overall, with `topics_short` listing every topic that came up short and why. The
runner logs it and stores it at `plan["eligibility"]["acquisition"]`. Nothing is
capped silently.

## Adding a pathway

Add an entry to `_PATHWAY_LEXICON` with its name `aliases` and its expected
enzyme/metabolite `terms`. Nothing else changes. An unlisted pathway is **not**
rejected: aliases are generated from the requested name (`<head> biosynthesis`,
`<head> synthesis`, `<head> pathway`, …), so the gate degrades in discriminating
power rather than in correctness.

## Known limits

* **Title-only screening is provisional and has low recall.** With the contextual
  anchor rule, a title that names the pathway but no reaction is `context_only`, so
  on the 2026-07-28_2122 titles recall against the labelled set is 0.33 (precision
  1.00). Every miss is flagged `needs_manual_review`. With abstracts, the same set
  scores precision 1.00 / recall 1.00. Screen on titles only to triage; screen on
  abstracts to decide.
* **A derived abstract is a proxy**, not parsed markup, and stays `provisional`.
* **Genus-level relations pass with a reduced bonus and a review flag.**
  `Bacillus subtilis` requested and `Bacillus cereus` observed is `genus_level`, not
  a match: admitted, scored at a quarter of the species bonus, and sent to a human.
* **Cross-species work is rejected by the organism veto.** A cholesterol
  biosynthesis paper done only in mice is `ineligible_organism` against a
  `Homo sapiens` request. Set `RAG_ELIGIBILITY_ORGANISM_VETO=false` to demote
  that to a score penalty. Papers that name *both* organisms already match.
* **The in-app RAG path is unchanged.** `t2pw.rag.select` remains the gate there.
  It does, however, benefit from the stamping fix: `_organism_score` now sees a
  genuinely empty candidate organism (0.5, neutral) instead of a fabricated
  match (1.0).
