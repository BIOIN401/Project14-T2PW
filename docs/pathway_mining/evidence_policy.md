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
`verbatim_cited_span` (the fallback). The span itself is in `field_evidence`. A
recovered species is passed to `species.classify_species` like any cited species.
The guardrail is unchanged.

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
