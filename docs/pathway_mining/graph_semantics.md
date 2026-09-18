# Primary graph vs candidate subgraphs

## Reactions are first-class nodes

A reaction is its own node with substrate and product edges — never a pairwise
metabolite edge. A reaction with three substrates and two products serializes as
**one** reaction node with five incident edges, so no viewer can read it as a
chain of pairs. Enzymes are annotations (`catalyzes` edges, `chemical: false`)
or explicitly typed nodes; they are never chemical bridges.

## What creates biochemical connectivity

A shared **chemical participant** in compatible roles. Three kinds, all labelled:

| Kind | Meaning |
|---|---|
| `product_to_substrate` | a product of one reaction is a substrate of another — the canonical pathway step |
| `shared_substrate` | two reactions consume the same metabolite — a branch point |
| `shared_product` | two reactions produce it — a convergence |

A reversible reaction can supply either side for the sequential test.

## What never creates connectivity

* a shared enzyme, protein or gene
* a shared gene symbol
* a shared pathway name
* a shared paper
* similar vocabulary
* semantic similarity or embedding distance
* co-occurrence in a passage
* a shared disease or phenotype
* a model asserting that two reactions "seem related"

This is enforced **structurally**, not by discipline.
`graph.connection_between` is the only function in the package that can answer
"are these two reactions connected", and it reads nothing but
`substrate_ids`/`product_ids` and the entity registry. It is never passed
enzymes, papers, pathway names, text or vectors, so it cannot consult them. An
enzyme entity that somehow landed on a participant list is still refused,
because the registry types it as catalytic.

### Cofactors

By default a shared cofactor (ATP, ADP, NAD+, water, CO2, phosphate, CoA, ...)
does **not** by itself make a connection. Nearly every reaction touches ATP or
water, so admitting cofactor bridges would collapse the literature into one
component and make the primary/candidate distinction meaningless.

This is a conservative, documented default, switchable per graph
(`allow_cofactor_bridges=True`), and the setting is recorded in
`primary_graph.json` and in the run manifest.

---

## Placement is not admission

Two separate decisions, made by two separate modules:

* **`admission.judge`** — is the biology defensible? (evidence, over-claiming,
  completeness, hypothesis, species, conflict)
* **`graph._place`** — where does an *admitted* reaction connect?

Neither can do the other's job. The consequence that matters: a wrong-species
reaction that shares a metabolite with the primary graph **stays quarantined**.
Chemistry decides where an admitted reaction sits, never whether it is admitted.

## The primary graph

Holds admitted reactions that are chemically connected to the rest of the
primary network. A reaction is in it only if it connects to another primary
reaction, or it belongs to the component the run selected as primary.

## Candidate subgraphs

Two kinds, distinguished by `quarantine_reason`:

| `quarantine_reason` | Cause | Promotable by a bridge? |
|---|---|---|
| `""` | admitted, relevant, but **not chemically connected** to the primary graph | **yes** |
| `species:<state>` | the species guardrail held it back | no |
| `hypothesis_only` | the paper proposes rather than reports it | no |

A candidate subgraph:

* stays **visibly distinct** from the primary graph — dashed outline in the
  network view, its own Streamlit tab, its own file `candidate_subgraphs.json`;
* preserves complete paper provenance;
* remains eligible for bridge-seeking queries (the highest-priority frontier
  kind);
* is promoted **only** after a chemically supported bridge is found;
* is never displayed as connected merely because it came from related papers;
* retains its original `created_round` and its full `promotion_history`.

Nothing is ever discarded for being disconnected.

## Promotion through a bridge

When a newly admitted reaction chemically connects a candidate component to the
primary graph, `graph._promote_now_connected` runs to a fixed point (promoting A
can connect B through A) and records on each promoted component:

```json
{
  "round": 3,
  "event": "promoted_to_primary",
  "bridge_reaction_id": "rx_...",
  "bridge_kind": "product_to_substrate",
  "via_entity": "12-OPDA",
  "bridge_paper_ids": ["PMC:PMC1000004"]
}
```

Each moved reaction keeps its original `discovery_round`, gains a
`promoted_round`, and gains an `admitted_to_primary` history event after its
earlier `quarantined` one. **Nothing is rewritten on promotion.**

## Choosing the primary component

A topic-only run often produces several disconnected components at first. **No
connection is ever fabricated to merge them.** The run selects one by a
documented deterministic policy:

```
score(component) = sum over reactions of  (distinct supporting papers)
                                        x (1 + topic relevance in [0,1])
```

Topic relevance is the fraction of topic terms appearing in the reaction's own
entity, enzyme and pathway names. It ranks components; it never creates
connectivity and never admits a reaction.

Ties break on: more reactions, then more distinct papers, then the
lexicographically smallest component id — so the choice is reproducible. Quarantine
components are never eligible to *be* the primary graph.

The losers stay candidates, and the reason the winner won is written into
`primary_graph.json` and `report.md`:

> selected as the primary component: evidence-weighted topic relevance 5.0000
> over 2 reaction(s) and 2 distinct paper(s); 1 other component(s) retained as
> candidates

## Deduplication and reaction identity

```python
reaction_identity(substrates, products, species=...)
```

is deliberately independent of enzyme, compartment, direction, paper and
wording, so two papers describing the same chemistry with different enzyme
attributions land on the **same** identity and their disagreement becomes a
visible conflict rather than two unrelated reactions. Direction is folded into
the identity core (the two sides are sorted) so a forward and a reverse report
of the same chemistry collide and can be compared; direction itself is an
attribute that can conflict.

**Species is part of the identity**, because the same chemistry in two species is
two different biological claims and folding them would be the cross-species
transfer the guardrail forbids. An *ambiguous*-species claim about the target
shares the target's identity on purpose — that is what lets a later round with
real species evidence reconsider it rather than create a second, disconnected
copy. A claim with no species at all gets its own namespace
(`unspecified:<state>`) so it can never silently merge into the target graph.

## Conflicts

Same identity, different non-empty value for `enzyme`, `direction`,
`reversible` or `compartment` = a `ConflictRecord`. Substrates, products and
species are excluded because they *are* the identity: a difference there is a
different reaction, not a disagreement about one.

Both positions are kept. Nothing picks a winner. The decision status becomes
`conflicting` — still **admitted**, because a disagreement about an enzyme does
not unmake the chemistry and hiding the reaction would hide the conflict with it
— and the disputed attribute is **cleared on the graph node** so the node stops
asserting one of the answers. A reaction admitted in round 1 with one enzyme
attribution and contradicted in round 2 stops displaying the round-1 value; both
values stay in `conflicts.json` and on the claims.

A conflict also generates a `conflict_resolution` frontier item, so the next
round goes looking for more evidence rather than choosing a side.

## Reconsideration

Decisions are **versioned, not final**. A claim held as `incomplete`,
`candidate`, `hypothesis_only` or `rejected` stays reconsiderable; each new
decision records `supersedes` pointing at the previous one, and
`ClaimHistory.narrative()` renders the chain:

```
round 2: candidate (species state 'species_ambiguous' ...)
round 4: accepted -> graph (paper-supported, target-species reaction with
         multiple independent papers; no external database lookup was required)
```

An already-accepted claim is not re-decided downward by this mechanism; new
contradicting evidence becomes a conflict record instead, which stays visible.

Nothing in this history is ever erased.
