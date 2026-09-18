# Species policy

A mining run declares **exactly one target species** and it is mandatory.
`MiningConfig` raises `ValueError` without it: the guardrail cannot classify
evidence against a target that does not exist, and "whatever the papers say" is
not a species.

Both forms are preserved, always:

```json
"target_species": "A. thaliana",
"target_species_normalized": "arabidopsis thaliana"
```

Normalization never replaces what the user wrote.

## Local normalization is not verification

`species.normalize_species` lowercases, collapses whitespace, strips
parenthetical strain/cultivar notes, expands the abbreviated binomial
(`A. thaliana`), trims a strain tail (`Escherichia coli K-12 MG1655` ->
`escherichia coli`) and maps a short table of common names (`human`, `mouse`,
`yeast`, `E. coli`, ...).

It makes **no external call** — not to NCBI Taxonomy, not to anything. An
unrecognised organism normalizes to itself. A normalized string is a comparison
key, never a resolved identifier.

`species_matches_target` requires exact normalized equality. A genus does not
satisfy a binomial: `Arabidopsis lyrata` never matches `Arabidopsis thaliana`.
(The common-name table does map the bare genus `arabidopsis`; that is a single
documented, auditable convenience, not an inference rule.)

## The seven states

Assigned by `species.classify_species`, which returns exactly one, with a reason.

| State | Meaning | Admissible? |
|---|---|---|
| `target_species_supported` | the target species is named **and** at least one evidence span is cited for the species claim specifically | **yes — the only one** |
| `non_target_species` | the evidence is for a different species | no |
| `multiple_species` | the evidence names more than one species (including when one of them is the target) | no |
| `species_ambiguous` | the target is named somewhere but no span ties it to *this* claim | no |
| `species_not_stated` | no species anywhere in the supporting evidence | no |
| `in_vitro_or_cell_free` | the passage describes a cell-free / reconstituted / purified-enzyme system — no organism performed the reaction | no |
| `inferred_only` | the species support is an inference, not an observation | no |

### Test order, and why

1. **inference** — an ortholog / analogy / "by similarity" claim is
   `inferred_only` *even when it names the target species and cites a span*.
   This is the ortholog-transfer guard and it deliberately outranks a naive name
   match.
2. **in vitro / cell-free** — no organism was involved.
3. **nothing observed** — `species_not_stated`.
4. **several distinct species** — `multiple_species`, never "one of them is the
   target, good enough".
5. **one species** — is it the target? If yes **and** a span is cited for the
   species specifically: `target_species_supported`. If yes but uncited:
   `species_ambiguous`. If no: `non_target_species`.

Detected inference markers include: `ortholog`, `orthologue`, `homolog of`,
`by analogy`, `by similarity`, `presumably`, `is expected to`,
`likely catalyzes`, `extrapolat`, `inferred from`. The extraction schema also
carries an explicit `species_inferred` boolean the model can set.

## The rule that matters most

> **A shared enzyme name, a shared gene symbol or an ortholog relationship never
> transfers a reaction between species.**

Mouse evidence stays mouse evidence however confidently a model asserts the human
orthologue does the same thing. Such a claim is **retained** — labelled,
auditable, available as a lead — but it is not an established target-species
reaction and cannot enter the accepted graph.

## What happens to a non-admissible claim

It is **never discarded**. `admission.judge` gives it decision status
`candidate` with the reason naming its species state, builds the reaction node
with `admitted=False`, and `graph._quarantine` places it in a labelled quarantine
component whose `quarantine_reason` is e.g. `species:non_target_species`.

Consequences of quarantine:

* it can never reach the primary graph **by chemistry alone** — sharing a
  metabolite with the primary network says nothing about species;
* a quarantine component is never promoted by a bridge — a bridge answers "is it
  connected", not "is it the right species";
* it keeps its full provenance, is shown in the Candidate subgraphs tab with its
  reason, and remains eligible for a `missing_species_evidence` frontier query;
* if a later round produces the same normalized claim **with** real species
  evidence, `graph.add_reaction` lifts it out of quarantine, records a
  `quarantine_lifted` history event, and places it properly. The earlier
  quarantine decision stays in the history chain.

## Where it shows up

* `claims.jsonl` — `species_state`, `species_reason`, `field_evidence.species`
* `decisions.jsonl` — `species_state` on every decision
* `summary.json` — `graph.species_states`, a census of every reaction node
* `report.md` — the census, marking which state is admitted
* Streamlit — a **Species state** filter on the reaction network, and the
  quarantine reason on every candidate component
