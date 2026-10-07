# Evidence policy

## The version-1 admission basis

**Proper paper evidence is sufficient to admit a research claim, and nothing else
is required.**

In particular, an admitted reaction does **not** need — and version 1 will not
seek — an identifier or mapping from PathBank, PWC, HMDB, KEGG, ChEBI, PubChem,
UniProt or any other biochemical identifier service. Those services are not
queried during mining rounds at all.

> Literature acquisition and paper-metadata services (EuropePMC, NCBI, Crossref,
> Semantic Scholar, bioRxiv) **are** used, freely. The restriction is on
> biochemical **fact validation**, not on paper **discovery**.

This also means the standing `placeholder_backed_proteins` disagreement is out of
scope here and is not addressed, worked around, or affected. The research graph
has no use for PathWhiz placeholder proteins.

### The deferred resolver

`t2pw.mining.resolver` is the seam a later, separately authorized
database-validation layer attaches to. Today:

```python
ResolutionResult(status="not_evaluated", refs={}, resolver="noop")
```

Every `ResearchEntity.external_refs` and every
`ResearchReaction.external_validation` carries exactly that. The shape is
explicit rather than absent on purpose: an absent field could be read as
"checked and clean", and a normalized *name* must never be mistaken for a
resolved *identifier*.

`summary.json` and `report.md` both state
`external_database_validation: deferred_not_evaluated` in so many words.

---

## The provenance chain

Every admitted reaction and every material attribute is traceable through:

```
graph reaction -> claim -> evidence span -> paper -> source identifier / file hash
```

`t2pw.mining.models.provenance_chain_complete` is the machine check. The
controller runs it over every admitted reaction before a run reports, and any
break is recorded in `summary.json` under `provenance_problems` and surfaced as a
warning in `report.md`.

A link is broken if: the reaction has no claim; a claim id does not resolve; a
claim has no evidence span; a span has no text; a span's paper is unknown to the
run; or the paper has **no resolvable source identifier** (PMID, PMCID, DOI, URL
or uploaded-file sha256). A "chain" ending at a title is not auditable.

### What is preserved per paper

`pmid` · `pmcid` · `doi` · `title` · `authors` · `journal` · `year` ·
`article_type` · `url` · `acquisition_source` · `cache_ref` · `file_sha256` ·
`file_name` · `retrieved_at` · `screening_outcome` · `screening_score` ·
`inclusion_reasons` · `exclusion_reasons` · `evidence_type` ·
`discovered_round` · `processed_round` · `discovered_via_query_id` ·
`discovered_via_frontier_id` · `observed_organisms` · `observed_pathways` ·
`abstract` · `full_text_chars` · `notes`

### What is preserved per evidence span

`evidence_id` · `paper_id` · the **exact** `text` · `section` · `page` ·
`paragraph` · `locator` · `chunk_id` · `char_start` / `char_end` ·
`context_ref` · `extraction_round` · `retrieval_query_id` ·
`frontier_item_id` · `retrieval_score` · `retrieved_at`

The query and frontier item are on the span so provenance runs backwards all the
way to the *question that found it*.

### What is preserved per claim

`claim_id` · `claim_identity` · `claim_type` · structured `subject` /
`relationship` / `object` · `species_text` and `species_state` ·
`substrates` and `products` (each participant with its own evidence ids) ·
`enzyme` / `protein` / `gene` · `direction` · `reversible` · `compartment` ·
`conditions` · `tissue` · `pathway_names` · **`field_evidence`** (field name ->
the evidence ids supporting *that field*) · `field_attribution` (field name ->
`model_citation` or `verbatim_cited_span`) · `evidence_ids` · `model` ·
`actual_model` · `prompt_version` · `schema_version` · `model_call_id` ·
`extraction_status` · `hypothesis` · `conflicts_with` · `duplicate_of` ·
`equivalent_claim_ids` · `first_seen_round` · `last_updated_round` ·
`confidence`

---

## No claim may be stronger than its evidence

This is enforced twice, mechanically, not by prompt instruction.

**1. At extraction.** The model is shown numbered passages, each labelled with an
`evidence_id` this run minted, and must cite ids for every field it fills. In
`extraction.build_claims`, a field whose citation list is empty — or cites an id
that was never shown — is **dropped**, and the drop is recorded in the round's
`errors` list as e.g.

```
claim[0].enzyme='a plausible-sounding synthase' (no valid citation)
```

The request names each field's own citation list (`enzyme_evidence_ids`,
`gene_evidence_ids`, `species_evidence_ids`, `compartment_evidence_ids`,
`direction_evidence_ids`, `reversible_evidence_ids`) and shows the expected JSON
shape. The claim-level `evidence_ids` never counts as a citation for another field.

**The verbatim fallback (enzyme, gene, species only).** A model can fill a name and
cite only the reaction's participants. Pilot-001's model did exactly that for all
59 of its enzyme, gene and species values
([`evidence/013-item2-replay/`](evidence/013-item2-replay/README.md)). The
fallback applies only when the field's own list is **missing**. An explicit empty
list means the model declined to cite, and a list citing only unshown ids is a
failed citation. Both are respected, and the value is dropped. A value whose
list is missing is kept **only** if it occurs verbatim — ignoring case and whitespace, and
never inside a longer word — in one of **that claim's own cited participant
spans**. It is then attributed to the first such span. Otherwise it is dropped
exactly as above. The fallback never looks at:

* a span the model was not shown (an unshown id is discarded before anything else);
* a shown span this claim's participants do not cite, including one cited only in
  the claim-level `evidence_ids`;
* paper-level metadata such as the screen's observed organisms.

Every kept field records how its evidence was attached, in
`ExtractedClaim.field_attribution`: `model_citation` (the model cited the span) or
`verbatim_cited_span` (the fallback). The span itself is in `field_evidence`.

**A recovered species is never species evidence.** Finding a species name in a
cited passage does not show that the passage ties that species to this
reaction. Pilot-001's own spans name *Arabidopsis* only as a transgenic host
("expression of bryophyte LCPUFA genes in transgenic Arabidopsis") or as a
comparison ("equivalent to Arabidopsis AtFAD3"). So a species kept **only** by
the fallback goes to `species.classify_species` as the observed value, but with
no species evidence ids:

* the target species -> `species_ambiguous`. The claim is kept as a lead, and the
  next frontier gets a `missing_species_evidence` item for it. It never reaches
  `target_species_supported` this way;
* another species, or a second one -> `non_target_species` / `multiple_species`,
  which quarantines the claim as before.

Only a species the model cited itself (`model_citation`) can make a claim
`target_species_supported`. `species.py` is unchanged. Enzyme and gene recovery
are unchanged too.

**2. At admission.** `admission._strip_unsupported` re-checks the material fields
(`enzyme`, `protein`, `gene`, `compartment`, `direction`, `reversible`) and blanks
any that still carry a value with no evidence, recording the strip on the
decision. Substrates and products are *not* strippable — they are the reaction;
a claim missing either is `incomplete`.

### The worked cases from the brief

| Situation | What happens |
|---|---|
| Passage supports substrate and product but not the enzyme | The reaction is admitted **with no enzyme**. Nothing is invented. |
| A separate passage supports the enzyme association | It attaches as a separately evidenced attribute: `field_evidence["enzyme"]` cites that span. |
| Species support comes from another part of the same paper | Recorded separately in `field_evidence["species"]`, with its own span. |
| The paper only *proposes* the reaction | `hypothesis: true` -> decision `hypothesis_only`, quarantined and labelled, never an established reaction. |
| The evidence concerns another species | `non_target_species` -> quarantined lead. Never silently transferred. See [`species_policy.md`](species_policy.md). |
| Two papers disagree about direction, enzyme, compartment or products | Both claims retained; a `ConflictRecord` is created; the disputed attribute is **cleared on the graph node** rather than silently chosen. |

---

## Which papers are read: the screen and the mining rescue

Every discovered paper goes through the production eligibility screen,
`t2pw.rag.eligibility.screen_candidate`, unchanged. Its pathway-anchor rule
excludes a paper as `context_only`, `omics_only` or `off_topic` when nothing
next to the pathway mention is a reaction or an enzyme. It does this before the
score is consulted. In pilot-001 that rule excluded PMC9947334 (primary
recombinant AtLOX2 enzymology, score 4.0) as `context_only`.

For mining only, `acquisition.Acquirer.rescue` reconsiders **that exclusion
only**. An organism veto or a negative-evidence veto is never reconsidered. A
paper is re-admitted by either of two routes:

1. `context_only` or `omics_only` only. An `off_topic` paper names no topic
   term at all, so this route never brings one back. The production screen
   reads the target species in it (`organism_match == "match"`), and
   `screen_candidate`'s own `thresholds` accept it with the anchor rule off and
   both bars raised by 1.0: `min_score` (2.0 -> 3.0) and, for a paper with no
   abstract, `title_only_min_score` (1.5 -> 2.5). The recorded reason names the
   threshold that was actually applied. The **title** must also name no organism
   other than the target, because the screen reads "match" even when the target
   is only the source of a transgene expressed in another plant (PMC10742046:
   emmer wheat "overexpressing two genes ... from *Arabidopsis thaliana*", score
   3.5). The check reuses the existing detectors read-only: the production
   screen's `observe_metadata` on the title alone (what its `compare_organism`
   calls a `mismatch`) and species.py's common-name table. Neither knows wheat
   or grape, so `_EXTRA_HOSTS` adds `wheat`, `triticum`, `grape` and `vitis`, each
   from a live false rescue (pilot-002's grape VaPAT1 paper, PMC5104190, is the
   latest). That list is a documented
   ceiling: a host that no detector knows, such as barley, soybean or
   *Nicotiana tabacum* written as a binomial, is still not caught.
2. The **title** names the target species, and one sentence of the title or
   abstract names it together with an enzyme or metabolite from **this run's**
   seeds or its `target_species_supported` reactions (`queries.rescue_terms`).
   Nothing comes from the frontier, so a non-target or quarantined lead never
   supplies a name. In pilot-001 every round-2 frontier item came from
   quarantined Marchantia claims. No pathway vocabulary is hard-coded. The
   topic, the species and cofactors are never terms. A name keeps its
   disambiguating note (`ALA (18:3n3)`), a name shorter than 5 characters is
   ignored, and an acronym (no lower-case letter, e.g. `12-OPDA`) must match in
   its own case. A bare `ALA` rescued a paper where it means 5-aminolevulinic
   acid, and `Ala` one where it means alanine. Both species guards exist because
   of the pilot-001 wheat transgene paper, which names *A. thaliana* only as a
   gene source, in its abstract.

Neither route rescues a paper whose **title** makes the target species a host
for a gene: *transgenic*, *heterologous(ly)*, *ectopic(ally)* or
*overexpress…*. This covers the must-exclude maize ZmLOX2 paper "… in Transgenic
Arabidopsis", which the name `lipoxygenase` would otherwise rescue. The rule is
a title word list, so it also turns away a native gene overexpressed in the
target. That is fail-closed: the production exclusion stands.

The paper's `screening_outcome` becomes `mining_rescue`. Its inclusion reason
is `mining_rescue: <what matched>`, and the production exclusion reason is kept
in `notes`. The claim-level species guardrail is unaffected: a rescued paper's
claims face exactly the same gates.

The rule is pinned by `tests/fixtures/mining/pilot001_screened_papers.json`: the
21 papers pilot-001 screened, with the bio-audit's adjudication.

| | pilot-001 papers |
|---|---|
| required, now included | PMC9947334 (route 1, score 4.0) |
| desired, now included | PMID 35404431 (DAD1; route 1, score 3.0) |
| desired, still excluded | PMC12280011 (`off_topic`, score 0.0; it names none of the run's names verbatim) |
| must stay excluded, and do | all 15, with the run's vocabulary (the seeds: pilot-001 admitted no target-species reaction), and even with every name its round-2 frontier held |
| borderline, excluded | PMC11300881 (no species reading), PMC13096701 (score 2.25) |

`tests/fixtures/mining/orch743_rescue_negatives.json` holds three papers the
16fd8a4b rescue let in, live: PMC13181208 and PMC13153360 (`off_topic`, route 1)
and PMC12977409 (`ALA` = 5-aminolevulinic acid, route 2). All three stay out.

---

## Evidence type and independent support

Each paper is labelled `primary`, `review`, `preprint` or `unknown` —
determined from the declared article type where available, else from source and
title markers, and honestly `unknown` where it cannot be determined. A review is
usable evidence; a reader must simply be able to see that it is one. A reaction
supported only by a review gets `evidence_level: review_article_only`.

**Independent support is never collapsed.** Two papers reporting the same
chemistry produce one *reaction* with two `paper_ids`, two claim records and two
evidence spans — which is what `multi_paper_reactions` in the summary counts.
Deduplication merges normalized *claims*, never *citations*.

---

## Model-output failures never become evidence

An empty completion, a truncated one (`finish_reason=length` or unbalanced
brackets), an unparseable one, a schema violation, or a payload kept only for
diagnosis are all in `NON_ADMISSIBLE_EXTRACTION_STATUSES`. Gate 1 of admission
rejects them **before any biological test**, and a partially parsed payload is
labelled `partial_repair_diagnostic_only` and can never become a reaction.

Each failure is a row in `model_calls.jsonl` with its real `finish_reason`,
token counts, latency and retry ladder — so "the literature had nothing to say"
and "the model returned nothing" are never confused.

### A thin answer is re-asked once, and stays marked

A payload can parse and validate and still be empty of reading. Pilot-003's calls
`_0001`, `_0003` and `_0014` answered in 150-210 output tokens and 4-8 s, where every
other call spent hundreds to thousands: alphabetical keys, no enzyme or species field,
no participant `evidence_ids`, and schema placeholders as values (`"stoichiometry":
"has_stoichiometry"`, `"products_evidence_ids_placeholder"`). Most likely a provider route
that did no reasoning; the shared client neither pins nor reports the provider.

`modelclient.thin_answer` names two shapes: any value that echoes a schema placeholder
(`has_*`, `*_placeholder`, `...`, `ev_...`), or records (`claims`) of which not one cites
evidence for a single field (no participant `evidence_ids`, no non-empty `*_evidence_ids`;
the claim-level list does not count). `{"claims": []}` is a legitimate answer and is never
thin. `MiningModelClient.call_json` re-asks a thin answer **once**, with the same passage and
schema: a fresh request, which on this stack can reach another provider, and a counted call.
A re-ask that is not thin is used. Otherwise the thin answer is kept and marked: the record's
`answer_quality` is `thin_after_retry` (or `thin_not_retried` when the call budget forbade the
re-ask). A thin answer is never upgraded silently. It is not dropped either: its claims go
through the same citation check and admission gates as any other, which is what keeps an
uncited participant out of the graph.
