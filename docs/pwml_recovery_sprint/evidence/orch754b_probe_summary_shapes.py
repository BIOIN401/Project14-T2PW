"""When does compact_mapping_misses copy a Stage-6 summary dict into the curator prompt?"""
import json

from t2pw.curation.interactive_curator import compact_mapping_misses

cases = {
    "summary_only": {"summary": {"proteins_unmapped": 2, "generated_wrappers_created": 1}},
    "enzyme_conv": {"enzyme_complex_conversion": {"summary": {"reaction_enzyme_complexes_novel": 2,
                                                                "interaction_protein_endpoints_rewritten_to_complexes": 3},
                                                    "actions": [{"type": "interaction_protein_endpoint_rewritten_to_complex",
                                                                 "json_pointer": "/p", "protein": "A", "protein_complex": "A complex"}]}},
}
for name, report in cases.items():
    print(name, json.dumps(compact_mapping_misses(report))[:400])
