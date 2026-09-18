# This mode produces a research graph, not PWML

## What is not done

Version 1 of pathway mining does **not**:

* invoke the PWML exporter;
* require or look up PathWhiz IDs;
* create technical protein-complex wrappers for PathWhiz;
* apply importer-only layout rules;
* invoke strict PWML gates or the pre-export quarantine;
* represent a research-graph failure as a PWML failure.

## How that is enforced

* **No mining module references the exporter.** A test walks every `.py` under
  `src/t2pw/mining/` asserting that none of `t2pw.pwml`, `json_to_sbml`,
  `strict_quarantine` or `build_sbml` appears anywhere in the source.
* **No run directory ever contains one.** A test asserts
  `list(run_dir.rglob("*.pwml")) == []`, and the same for `*.sbml` and `*.xml`.
* **Every artifact says so.** `run_manifest.json` and `summary.json` both carry
  `"produces_pwml": false`; `report.md` states it in the first screen; the
  preflight has a `policy.pwml` check; `python -m t2pw.mining status` prints
  `PWML none - this mode produces a research graph only`.
* **An existing pathway supplied as a starting graph is read-only.**
  `load_starting_graph` reads structure and provenance and nothing is ever
  written back, exported, or converted.

## Failures stay in their own vocabulary

A mining run that produced few reactions reports a mining reason: a stop reason
from the closed set, model-call failures with their real `finish_reason`, papers
excluded with their screening reasons, claims rejected with their gate reasons.
None of that is expressible as, or reported as, a PWML outcome, and no
`release_ready` / `review_required` / `diagnostic_only` state is produced.

## What a later conversion layer would need

The research graph deliberately retains enough structured information that a
**separately authorized** conversion layer could be added later without
re-mining. It would find, per reaction:

| Available today | Where |
|---|---|
| substrates and products as typed entities, with roles | `ResearchReaction.substrate_ids` / `product_ids`, `EntityRegistry` |
| stoichiometry as stated by the paper | `ReactionParticipant.stoichiometry` |
| enzyme / protein / gene, separated from chemistry | `enzyme_names`, `enzyme_entity_ids`, `gene` |
| direction and reversibility, or an explicit absence | `direction`, `reversible` |
| compartment, conditions, tissue | `compartment`, `conditions`, `tissue` |
| organism context and how well it is supported | `species_text`, `species_state` |
| connectivity and its kind | `ResearchGraph.connections`, `ConnectionEvidence` |
| the full evidence chain for every field | `field_evidence` -> `EvidenceSpan` -> `PaperRecord` |
| what is disputed | `ConflictRecord`, `conflict_ids` |
| what is quarantined and why | `quarantine_reason`, `CandidateSubgraph` |

What such a layer would have to add — and what this version deliberately does
not do — is the external-identifier resolution the PWML exporter needs (HMDB,
KEGG, ChEBI, UniProt, PathWhiz IDs). That is the deferred work described in
[`evidence_policy.md`](evidence_policy.md), and its seam is
`t2pw.mining.resolver`.

## Relationship to the frozen production path

Production PWML engineering is frozen under `D-090`. This is a separate,
experimental research feature on its own branch. It:

* adds no behaviour to any production module;
* is imported by no production module;
* changes nothing when its flag is off;
* has been measured against the production regression gate with the feature
  disabled — **SMOKE: 508 passed**, the expected count
  (`evidence/001-smoke-feature-disabled.json`).

No contract boundary in `docs/pwml_recovery_sprint/PRODUCT_CONTRACT.md` is
touched, weakened or reinterpreted by this work. That document governs the PWML
product; this mode produces no PWML and makes no claim under it.
