# Replay idx 10, flag on: why 4 of 7 transports (d2) versus 6 of 7 (P3)

Method: files only (no Docker, no source edits). PWML diff of `final_payload_p3/replay/on/pwml/10.pwml` against `final_payload_p4/d2/replay/on/pwml/10.pwml`, importer results from `final_payload_p3/import/on/b_00/results.jsonl` and `d2/import/on59_results.jsonl`, importer code `C:/t/pwi/smpdb/lib/pwml_parser.rb`.

## 1. What changed in the PWML (the whole diff is 3 hunks)

| | P3 | d2 |
|---|---|---|
| location 3 `saliva` | present, `<ontology-id nil="true"/>` | dropped (writer.py ~1097, D-W20a: id-less location dropped) |
| location 7 `synaptic vesicle` | no ontology id | `GO:0008021` (task A) |
| state 14 `extracellular state` | `subcellular-location-id` 3, tissue and cell type nil | `<subcellular-location-id nil="true" type="integer"/>`, so tissue, location and cell type all nil, species 12 only |

Tissues, cell types, the other 7 locations, the other 9 states and all 7 transports (element ids, left/right state ids) are byte identical. Ontology ids added by task A for idx 10: only GO:0008021 (location 7), which no state uses (states use location 4 `synaptic vesicles`).

## 2. Mechanism: an importer wildcard match, not a transport change

`pwml_parser.rb` `store_object` resolves a biological state with `BiologicalState.find_by(attribute_hash)`. `add_content` skips any element whose content is empty (`node.content.present?`), so a nil id is omitted from the hash, whereas an id that is present but failed to map is set to nil (a strict NULL match). Therefore:

* P3: state 14 has location 3 present; location 3 cannot be created (`subcellular_location_3: Ontology can't be blank`), so the hash is `{subcellular_location_id: nil, species_id: 12}`. `find_by` returns the first species-12 state with a NULL location, row R2 (the `apical state`, tissue 10). So 14 = R2, different from state 12 (R1, location 9).
* d2: state 14 is species-only, so the hash is `{species_id: 12}`, and `find_by` returns the very first species-12 row, R1, which is state 12 (`Type III cell state`, location 9). So 14 = 12.

Replaying the importer's mapping over the doc reproduces both results exactly:

| transport (left to right) | P3 rows | d2 rows | P3 result | d2 result |
|---|---|---|---|---|
| 22 (14 to 12) | R2 to R1 | R1 to R1 | ok | `transport_element_23: Biological state must have unique biological states` |
| 26 (12 to 14) | R1 to R2 | R1 to R1 | ok | `transport_element_27: ... unique biological states` |
| 33 (21 to 21) | R1 to R1 | R1 to R1 | `transport_element_34` (same state both sides) | same |
| 28, 31, 36, 38 | distinct | distinct | ok | ok |

Each failure is followed by `transport_22/26/33: Transport elements is too short` (message count 3 left=right in d2 vs 1 in P3). Created transports 4 (d2) vs 6 (P3). The remaining d2 messages (`transport_visualization_80/86`, `transport_transporter_visualization_83`) are the cascade of the two lost transports. The validation is `transport_element.rb:50` `biological_states_unique` (left == right after remapping).

`identical_states.txt` shows the same group count for P3 and d2 because it models the writer's key (species, tissue, location, cell type) and treats an omitted component as a distinct value. The importer treats an omitted component as a wildcard. That is why the metric did not predict the loss.

## 3. Real regression or import artefact?

Both, in different respects.

* Artefact (importer): `find_by` with omitted keys is a wildcard. In production it would bind a species-only state to an arbitrary existing species state (worse than a miss). It is the same family as the known empty-state fall-through.
* Real change in the final PWML versus P3: the writer's collision guard (`writer.py` `keep` rule, commit 02eff361) checks the key (species, tissue, cell type) and treats species as a component. It therefore allows a drop that leaves a species-only state. P3 kept `saliva` as an extracellular location with name only; d2 removes the only information that distinguishes "extracellular state" (it now reads as "any state of Mus musculus"). That is a loss of biology relative to P3 and breaks the "never worsen output" rule on the importable result.
* Scope: `species-only` states in the on-set: P3 0, d2 8 states in 8 files (10 `extracellular state`, 27 `extracellular state`, 116 `medial Golgi state`, 119 `leaf surface`, 125 `microsomal fraction state`, 151 `saliva`, 3 `afferent nerve fiber state`, 70 `bacterial cell state`); flag-off also 0. Idx 10 is the only one of those files in which a transport pair then collapses in the harness; the other 7 are exposed the same way.

## 4. Minimal fix proposal (writer only, flag-on, no biology)

`src/t2pw/pwml/writer.py`, in the drop block at about lines 1112 to 1130 (the `keep` computation): extend the "no component" rule. Currently `if loc in droppable_locations and all(x is None for x in rest)` where `rest = (species, tissue, cell_type)`. Change it to test only `(tissue, cell_type)`:

```
if loc in droppable_locations and all(x is None for x in rest[1:]):   # species is not a distinguishing component
    keep.add(loc)
```

Rule: an id-less location is kept by name whenever dropping it would leave the state with no tissue, cell type or location, because PathWhiz `find_by` treats the omitted keys as a wildcard on the species alone (PathWhiz also requires `has_at_least_one_component`). The 8 affected states then keep their location by name exactly as in P3 (so those files are byte identical to P3 for those states, output not worse), and every state that has a tissue or cell type still drops the id-less location. A test: a state with only species plus an id-less location must retain the location row and give the state a non-nil `subcellular-location-id`; with the flag off nothing changes. Expected effect on idx 10: transports 6 of 7 and left=right 1, as in P3.

Not recommended: changing the importer here (it is out of this track's scope), but the same `find_by` wildcard should go in the importer bug list (`importer_bugs_p3.md`) as "state lookup omits nil ids".
