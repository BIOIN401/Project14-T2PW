# ontoBE replay (flag ON, 14 stored runs) against the same replay on the parent commit 010d2f39

`p5_replay.py` (adapted copy of the P4 replay): stored `final_mapped.json`, stage-0 context and `01_source_text.txt` (when the run folder has one) through `run_pwml_export`. Only the vocabulary rows each PWML writes are compared (name, ontology id); the rest of the PWML is not diffed here.

## run 2 `55/legs/sour_taste-research/2026-10-21_0709/papers/PMC7059786/research`
- ontoBE: ok=False pwml_written=False paper_text_found=True stage='stage3_revalidation'; base: ok=False pwml_written=False
- P5 vocabulary/drop counters: {}
- P5 ontology extra: {'id_used_by_other_row': 1}
  - fill entry: location `synaptic vesicle` -> ontology_id GO:0008021 (OLS4, exact_label) {}
- PWML: base absent, ontoBE absent

## run 3 `CH-755/legs/sour_taste-strict/2026-10-21_0139/papers/PMC7059786/strict`
- ontoBE: ok=True pwml_written=True paper_text_found=True stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_unmapped': 3, 'vocabulary_mapped_by_id': 1, 'vocabulary_mapped': 2, 'vocabulary_exact': 3, 'vocabulary_new_row': 2, 'components_kept_wildcard_guard': 2, 'locations_kept_state_collision': 1, 'tissues_dropped_no_id': 1}
- P5 ontology extra: {'singular_form_resolved': 1, 'numbered_context_resolved': 1, 'id_used_by_other_row': 1}
  - fill entry: location `synaptic vesicles` -> ontology_id GO:0008021 (OLS4, exact_label) {'matched_form': 'synaptic vesicle'}
  - fill entry: cell_type `Type III cell` -> ontology_id CL:0002285 (OLS4, reviewed_synonym) {'via': 'paper_context', 'context_phrase': 'type iii taste cell', 'context_source': '01_source_text.txt', 'context_count': 4}
- location: only in base [('synaptic vesicles', '')]; only in ontoBE [('synaptic vesicles', 'GO:0008021')]
- cell-type: only in base [('Type III cell', '')]; only in ontoBE [('type III taste bud cell', '')]

## run 10 `59p/legs/sour_taste_v2-strict/2026-10-22_2233/papers/PMC7059786/strict`
- ontoBE: ok=True pwml_written=True paper_text_found=True stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_mapped_by_id': 1, 'vocabulary_unmapped': 3, 'vocabulary_mapped': 2, 'vocabulary_exact': 3, 'vocabulary_new_row': 2, 'components_kept_wildcard_guard': 3, 'locations_kept_state_collision': 3}
- P5 ontology extra: {'numbered_context_resolved': 1, 'id_used_by_other_row': 1}
  - fill entry: location `synaptic vesicle` -> ontology_id GO:0008021 (OLS4, exact_label) {}
  - fill entry: cell_type `Type III cells` -> ontology_id CL:0002285 (OLS4, reviewed_synonym) {'via': 'paper_context', 'context_phrase': 'type iii taste cell', 'context_source': '01_source_text.txt', 'context_count': 4}
- cell-type: only in base []; only in ontoBE [('type III taste bud cell', '')]

## run 23 `N/Project14-T2PW/runs_verify/2026-08-21_1822/papers/PMC12782028/strict`
- ontoBE: ok=True pwml_written=True paper_text_found=True stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_unmapped': 2, 'vocabulary_exact': 3, 'vocabulary_new_row': 1, 'cell_types_dropped_no_id': 2}
- P5 ontology extra: {'cellosaurus_found': 2}
  - fill entry: cell_type `MG-63` -> cellosaurus_id CVCL_0426 (Cellosaurus, exact_label) {}
  - fill entry: cell_type `Saos-2` -> cellosaurus_id CVCL_0548 (Cellosaurus, exact_label) {}
- PWML vocabulary rows identical to base

## run 30 `Project14-T2PW/runs_verify/2026-08-21_2239/papers/PMC12096016/research`
- ontoBE: ok=False pwml_written=False paper_text_found=True stage='stage3_revalidation'; base: ok=False pwml_written=False
- P5 vocabulary/drop counters: {}
- P5 ontology extra: {}
- PWML: base absent, ontoBE absent

## run 36 `Project14-T2PW/runs_verify/2026-08-21_2239/papers/PMC12782028/research`
- ontoBE: ok=False pwml_written=False paper_text_found=True stage='required_field_gate'; base: ok=False pwml_written=False
- P5 vocabulary/drop counters: {}
- P5 ontology extra: {'cellosaurus_found': 2}
  - fill entry: cell_type `MG-63` -> cellosaurus_id CVCL_0426 (Cellosaurus, exact_label) {}
  - fill entry: cell_type `Saos-2` -> cellosaurus_id CVCL_0548 (Cellosaurus, exact_label) {}
- PWML: base absent, ontoBE absent

## run 64 `Project14-T2PW/runs_verify/2026-08-24_1402/papers/PMC12782028/research`
- ontoBE: ok=False pwml_written=False paper_text_found=True stage='required_field_gate'; base: ok=False pwml_written=False
- P5 vocabulary/drop counters: {}
- P5 ontology extra: {}
- PWML: base absent, ontoBE absent

## run 68 `N/Project14-T2PW/runs_verify/2026-08-24_1428/papers/PMC12444477/strict`
- ontoBE: ok=True pwml_written=True paper_text_found=True stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_exact': 2, 'vocabulary_unmapped': 3, 'vocabulary_mapped': 1, 'vocabulary_new_row': 1, 'locations_dropped_no_id': 1, 'components_kept_wildcard_guard': 2, 'locations_kept_state_collision': 2}
- P5 ontology extra: {}
- location: only in base [('inner membrane', ''), ('outer membrane', '')]; only in ontoBE [('Outer Membrane', '')]

## run 76 `N/Project14-T2PW/runs_verify/2026-08-25_1216/papers/PMC12444477/strict`
- ontoBE: ok=True pwml_written=True paper_text_found=True stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_exact': 2, 'vocabulary_unmapped': 2, 'vocabulary_mapped': 2, 'vocabulary_new_row': 1, 'components_kept_wildcard_guard': 2, 'locations_kept_state_collision': 2}
- P5 ontology extra: {}
- location: only in base [('inner membrane (IM)', ''), ('outer membrane (OM)', '')]; only in ontoBE [('Inner Membrane', ''), ('Outer Membrane', '')]

## run 94 `Project14-T2PW/runs_verify/2026-09-01_1612/papers/PMC12444477/research`
- ontoBE: ok=False pwml_written=False paper_text_found=True stage='stage3_revalidation'; base: ok=False pwml_written=False
- P5 vocabulary/drop counters: {}
- P5 ontology extra: {}
- PWML: base absent, ontoBE absent

## run 110 `Project14-T2PW/runs_verify/2026-09-06_1425/papers/PMC11172790/research`
- ontoBE: ok=False pwml_written=False paper_text_found=True stage='stage3_revalidation'; base: ok=False pwml_written=False
- P5 vocabulary/drop counters: {}
- P5 ontology extra: {}
- PWML: base absent, ontoBE absent

## run 146 `IN/Project14-T2PW/runs_smoke/2026-09-10_1135/papers/PMC13405594/strict`
- ontoBE: ok=True pwml_written=True paper_text_found=True stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_mapped': 1, 'vocabulary_exact': 6, 'vocabulary_new_row': 1}
- P5 ontology extra: {}
- cell-type: only in base []; only in ontoBE [('Granulosa Cell', '')]

## run 150 `C:/t/e2e4-run/work/item-1/attempt-1/out`
- ontoBE: ok=True pwml_written=True paper_text_found=False stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_unmapped': 5, 'vocabulary_mapped': 4, 'vocabulary_exact': 2, 'vocabulary_new_row': 2, 'components_kept_wildcard_guard': 4, 'locations_kept_state_collision': 2, 'cell_types_dropped_no_id': 1}
- P5 ontology extra: {'id_used_by_other_row': 1, 'cellosaurus_refused': {'no_exact_match': 3}}
  - fill entry: location `synaptic vesicle` -> ontology_id GO:0008021 (OLS4, exact_label) {}
- PWML vocabulary rows identical to base

## run 151 `C:/t/e2e5-run/work/item-1/attempt-1/out`
- ontoBE: ok=True pwml_written=True paper_text_found=False stage=''; base: ok=True pwml_written=True
- P5 vocabulary/drop counters: {'vocabulary_unmapped': 4, 'vocabulary_mapped': 3, 'vocabulary_new_row': 3, 'vocabulary_exact': 2, 'components_kept_wildcard_guard': 1, 'locations_kept_state_collision': 1, 'cell_types_dropped_no_id': 3}
- P5 ontology extra: {'id_used_by_other_row': 1, 'cellosaurus_refused': {'no_exact_match': 2}}
  - fill entry: location `synaptic vesicle` -> ontology_id GO:0008021 (OLS4, exact_label) {}
  - fill entry: location `synaptic cleft` -> ontology_id GO:0043083 (OLS4, exact_label) {}
- PWML vocabulary rows identical to base
