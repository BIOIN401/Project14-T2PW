# The two research views

Two views, deliberately two different graphs.

> **Paper / provenance edges and biochemical connectivity are different
> relationships.** The provenance view's payload carries
> `"provenance_edges_are_not_chemical_connectivity": true`, its `note` says so,
> the Streamlit page prints it above the canvas, the renderer prints it on the
> canvas, and the Graphviz source puts it in the graph label. Two reactions
> sharing a paper are **not** connected.

---

## 1. Reaction network — `graph_views/reaction_network.json`

The biochemical view.

### Guarantees

* **Metabolites and reactions are visually distinguishable**: metabolites are
  ellipses, reactions are boxes, enzymes are hexagons.
* **Multi-substrate / multi-product reactions are accurate**: one reaction node
  with all its incident edges. There is no metabolite-to-metabolite edge
  anywhere in the payload, and a test asserts it.
* **Enzymes are annotations**, drawn with a dashed edge, an open arrowhead, a
  distinct colour and `"chemical": false`. A test asserts no enzyme node is an
  endpoint of any chemical edge.
* **Primary vs candidate is unmistakable**: solid vs dashed outline, different
  fill, and a legend that says so in words.
* **Directionality and reversibility are visible**: the node label renders `->`
  or `<=>`, and `direction` / `reversible` are on the node.
* A reaction with an unresolved conflict carries a `⚠ conflict` badge.

### Filters

Round slider (the network *as it stood* at the end of round N), graph status,
supporting paper, enzyme, species state, and "only reactions supported by more
than one paper". Implemented server-side in `viz.filter_reaction_network`, so
the filtering is testable rather than a UI detail.

The round slider reconstructs history properly: a reaction promoted in round 3
reads as **candidate** when the slider is at round 2, because that is what it
was.

### Selecting a reaction reveals

complete reaction · species and species state · enzymes · compartment ·
conditions and tissue · discovery round · graph status and component ·
promoted round · every supporting paper with title, year and evidence type ·
every evidence id · every conflict with both positions and their papers ·
the full admission history (`discovered`, `quarantined`, `joined_component`,
`admitted_to_primary`, `disputed_attribute_cleared`, `quarantine_lifted`,
`component_merged`) · decision status and reason · external validation status.

### What it will not draw

A biochemical edge based on semantic similarity, embedding distance, paper
co-occurrence, a shared enzyme, a shared gene or a shared pathway name. The edge
list comes from `substrate_ids` / `product_ids` / `enzyme_entity_ids` only, and
the enzyme edges are typed as annotations.

---

## 2. Paper / evidence provenance — `graph_views/provenance_network.json`

```
paper -> evidence span -> claim -> reaction
```

### Shows

* which papers support each reaction (`supporting_papers` on every reaction node)
* which reactions are supported by multiple papers (`multi_paper_supported`)
* which paper introduced a reaction first (`first_support`, and `first_support`
  on the claim->reaction edge)
* the retrieval query and mining round that surfaced each span
* conflicting papers and claims
* primary vs review evidence where known (`evidence_type`, honestly `unknown`
  otherwise)
* papers that support candidate subgraphs (`supports_candidate_subgraph`)
* papers involved in discovering a bridge (`is_bridge_paper`)
* papers retrieved but **excluded**, with their exclusion reasons — available
  through a filter, faded and dashed when shown, never mixed into accepted
  evidence

### Large graphs

Layer selection (papers / evidence / claims / reactions) is the primary
progressive-disclosure control; evidence spans are numerous and are hidden by
default. Beyond `max_nodes` (default 420) the renderer trims, keeping reaction
nodes first and reporting `trimmed_from` in the payload.

For exact auditing the diagram is explicitly **not enough** — use the table.

---

## 3. The evidence table — `graph_views/evidence_table.json`

One row per (reaction, claim, evidence span). This is the authoritative audit
view and it is available both in Streamlit and as a downloadable JSON.

Columns: reaction id and label · graph status · component · discovery and
promoted round · species and species state · enzyme · compartment · direction ·
reversible · decision status and reason · evidence level · claim id · paper id,
title and year · evidence type · **source identifier** · evidence id · section ·
locator · the **exact quote** · retrieval query id · frontier item id ·
conflicts · external validation status.

---

## Exports

| File | Use |
|---|---|
| `reaction_network.json` / `provenance_network.json` | feed any other renderer |
| `reaction_network.dot` / `provenance_network.dot` | Graphviz source; `dot -Tsvg` if the binary is available |
| `evidence_table.json` | spreadsheet / audit |
| `primary_graph.json` / `candidate_subgraphs.json` | the graph itself |

All are offered as download buttons in the Streamlit **Logs, cost & exports**
tab, and all are written to the run directory automatically.
