# Protein export policy

One authoritative answer to "may this identifier ship as this protein's
identity, and if not, what ships instead?". Written 2026-07-29 on branch
`research-mode`.

The policy exists because the two obvious answers are both wrong. Requiring a
verified UniProt accession for every enzyme deletes real biology from novel
pathways — run 2026-07-28_2122 failed all 16 strict legs, 7 of them at the
post-pipeline gates, and the single largest cause was enzymes a paper states
clearly and no database has ever heard of. Accepting whatever the resolver
returns produces the opposite failure, and it is worse because it passes every
gate: in run 2026-07-28_0919 `PhoP` shipped as NAD+, `pmrHFIJKLM operon` as the
lactose operon repressor, and `mcr genes` as the human mineralocorticoid
receptor, all with `resolution.status == "matched"` and zero gate errors.

## The three outcomes

Exactly one applies to any protein-shaped participant.

### A. Verified real mapping

**The ladder fails closed.** Every rung must pass *affirmatively*; missing
evidence is never a pass. A real identifier is accepted only when it clears all
six rungs of `map_ids.verify_real_protein_identity`, in this order:

| # | Rung | Requirement | Failure reason |
|---|------|-------------|----------------|
| 1 | **identifier resolution** | a non-sentinel UniProt / DrugBank / PathBank id | `no_real_identifier` |
| 2 | **entity type** | the entity name is not one the compound rules own, **a candidate row describing the shipped identifier exists**, and it is protein-shaped | `entity_type_incompatible`, `identity_evidence_missing` |
| 3 | **species / taxon** | `ok` or `genus_level` (below) | `species_mismatch`, `identity_evidence_missing` |
| 4 | **name** | `_name_gate_verdict` returns **`keep`**: canonical name, exact gene/locus symbol, or an audited alias | `implausible_name_match`, `identity_evidence_missing` |
| 5 | **minimum score** | ≥ `_REAL_PROTEIN_MIN_SCORE` (0.5) | `score_below_minimum`, `identity_evidence_missing` |
| 6 | **margin** | ≥ `_REAL_PROTEIN_MIN_MARGIN` (0.1) over the best rival naming a different accession | `ambiguous_insufficient_margin` |

`identity_evidence_missing` is the fail-closed reason. It is **not** a
refutation, and the caller must route it to outcome B rather than dropping the
actor: "we cannot confirm this" and "this is wrong" have the same effect on the
identifier and opposite effects on the biology. Four situations produce it —
candidate absent, species `unknown`, name-gate `skip`, and an unscored
non-PathBank candidate. All four were passes before.

#### Species matching (rung 3)

`ok` requires affirmative agreement: the same NCBI taxonomy id, the same
normalized binomial, or a strain/subspecies qualification of it at a word
boundary (`Escherichia coli` ↔ `Escherichia coli K-12`).

`mismatch` covers same-genus-different-species — `Escherichia coli` ↔
`Escherichia fergusonii`, `Bacillus subtilis` ↔ `Bacillus cereus`. Blanket genus
agreement used to pass, which made those two organisms interchangeable.

`genus_level` is returned **only** when the request was explicitly genus-level (a
single token, `"Escherichia"`) and the candidate sits in that genus. It passes,
and it is recorded on the verdict as `species_compatibility: "genus_level"` — it
is never reported as an exact species match.

`unknown` (one side silent) is missing evidence.

#### PathBank rows (rungs 4 and 5)

There is **no blanket name exemption**. It used to return `skip` before any other
evidence was weighed, which under a fail-closed ladder is indistinguishable from
a pass — so curated provenance alone was shipping accessions. A PathBank
candidate now proves itself the same way as any other: shared name token, exact
gene symbol, or an audited alias.

What survives is the narrow, honest part: PathBank supplies no score, so a
PathBank row may pass rung 5 unscored (`skipped_unscored_pathbank_row`) *after*
clearing name and species on its own evidence. A non-PathBank candidate with no
score is missing evidence.

The cost is real and accepted: three matches of the `LpxL` → P0ACV0 shape (entity
is the modern gene symbol, PathBank stores the legacy one — `htrB`, `msbB`,
`pgpA`) were correct and now route to outcome B instead of shipping unverified.

#### Score attribution (rung 5)

Candidate-local score first, always. Result-level `confidence` is consulted
**only** when `result.mapped_ids` identifies the same candidate
(`_candidate_identifies_result_choice`). On the second mapping pass the row
routinely ships an accession the resolver did *not* pick this time —
`_merge_mapped_ids` keeps whatever arrived first — and reading the result's
confidence there let a weak identity borrow a strong one's score.

A verified row carries `mapping_meta.identity_status = "verified"` and its full
`identity_verdict`.

### B. Evidence-backed unresolved protein actor

A reaction enzyme, catalytic modifier or transporter with a usable functional
name and **direct role evidence** but no verified identity keeps its biology:

* the functional name is preserved on a generated `protein_complex`;
* the complex's only component is PathBank's Unknown protein, id 9659, UniProt
  `Unknown`;
* the reaction / transport actor points at the functional complex name — never
  at the literal `Unknown`;
* `identity_status = "placeholder"`, plus `fallback_used`, `target_organism`,
  `cross_species_placeholder`, the actor's `evidence` / `provenance` /
  `source_refs`, and `real_mapping_failure_reason` (taken from the row's own
  audit trail, not a fixed string);
* it is **never** counted as a real UniProt mapping. `verified_real_proteins`
  and `unknown_backed_functional_complexes` are separate numbers.

**What authorizes the placeholder**, stated for what it is. Canonical membership
in `reactions[].enzymes` / `transports[].transporters` (or a `catalyst`/`enzyme`
modifier) is enough to authorize it — the payload asserts the role there and
nowhere else does it mean anything. But membership is a *structural* assertion,
not a sourced one, so it is recorded as:

* `role_basis: "canonical_actor_membership"` — never called direct evidence;
* `direct_evidence_present: true|false` — separately, whether the actor or its
  protein row actually carries evidence / provenance / source_refs;
* `provenance` and `source_refs` verbatim, **only when present**.

The authorization is withdrawn, and the process claim quarantined instead, when
the declared role is not one the collection means or the actor names a compound.

**Evidence is cited, not copied.** `rag/conform.py` flattens retrieved evidence
into a single string, and one reference payload's reaction #14 is 139,576
characters made of the same 4,812-character passage 29 times. `mapping_meta`
therefore carries `evidence_digest` — a 200-character excerpt, a sha256 of the
full text, its length, and a `truncated` flag — not the text. 200 clears the
longest genuine evidence sentence measured in the reference runs (157) and
truncates every blob (next value up: 4,636).

### C. Unsupported or unused

* **Not referenced by a surviving process** → quarantined into
  `payload["quarantined_proteins"]` before the strict gates, with the reason and
  whether it had been verified. Research mode keeps it and flags it instead.
* **Referenced but with no evidence for the role it claims** (an inhibitor
  parked in `enzymes`, a cofactor sitting in the actor list) → the *process
  claim* is quarantined into `payload["quarantined_process_claims"]`. It does
  not get the Unknown fallback: wrapping an unsupported claim in a functional
  complex manufactures an enzyme the paper never stated, which is worse than a
  missing identifier.

## The prune/gate contradiction

`prune_disconnected_proteins` and `drop_process_orphan_proteins` used to spare a
degree-0 protein that carried an external identifier, and the connectivity gate
three calls later then failed the payload for exactly those rows — four of
PMC12444477's strict-leg errors existed for no other reason. Sparing them never
saved them; it only moved the failure downstream.

Strict mode now quarantines an unused protein whether or not it is mapped.
Research mode retains and flags it, and both passes now run in both modes so the
research census (`unused_proteins_flagged_for_review`) exists at all. A
component of a surviving protein_complex is exempt in both modes — its edge runs
through the complex, which is what keeps the Unknown-backed complex's sentinel
alive.

## False protein promotion

`PROTEIN_LIKE_RE` matches the substring `enzyme`, which is inside `coenzyme`.
Every "coenzyme A" name in run 2026-07-28_2122 was therefore filed in
`entities.proteins`, where it can never acquire a UniProt ID; 8 of that night's
27 distinct post-gate issue codes are one of two cofactors misfiled this way.

`entity_identity.compound_name_block_rule` answers this with two sources, in
order: the payload's own compound registry (authoritative), then the
carrier-moiety name shape (`coa_thioester`, `acp_thioester`). Both defer to an
enzyme head-noun check, so `succinyl-CoA` and `beta-hydroxyacyl-ACP` are
metabolites while `succinyl-CoA synthetase` and `beta-hydroxyacyl-ACP
dehydratase` stay proteins — the distinction the regex could not make. Bare
`ACP` and `holo-ACP` are the carrier protein; `trans-2-acyl-ACP` is not.

The guard is applied at the regex-driven verdicts only (`_is_protein_like`,
`_protein_like_norms`, `route_entity_for_mapping`) — an explicit type hint and
an explicit protein declaration both still outrank it, so nothing declared a
protein changes route.

## Reporting

Six numbers, in `report["protein_export_policy"]["summary"]` (Stage 6) and
mirrored into the mapping summary. Stage 3 publishes the two it alone can count
onto `payload["protein_export_policy"]`, which Stage 6 carries forward:

| Key | Meaning |
|-----|---------|
| `verified_real_proteins` | passed all six rungs |
| `unknown_backed_functional_complexes` | outcome B, well formed |
| `ambiguous_real_candidates_rejected` | candidate lists refused rather than resolved by order |
| `unused_proteins_quarantined` | outcome C, unused |
| `false_protein_promotions_blocked` | participants the regex would have promoted |
| `bare_unresolved_proteins_remaining` | proteins with no identity and no placeholder — the number that says whether strict export can succeed |

## Strict gate

`run_strict_post_normalization_gates` accepts a verified real protein and a
correctly formed Unknown-backed functional complex, and rejects:

* a bare unresolved protein (pre-existing rule);
* a placeholder posing as a real mapping — one that ships a plausible accession,
  reports a `matched` resolution, or omits `fallback_used`
  (`placeholder_claims_real_identity`);
* a generated wrapper whose `Unknown`-named component is not the real PathBank
  sentinel.

## Tests

`tests/test_protein_export_policy.py` covers the twelve required scenarios;
`tests/test_pathbank_unknown_fallback.py` and `tests/test_map_ids_name_gate.py`
keep the pre-existing fallback and name-gate contracts.
