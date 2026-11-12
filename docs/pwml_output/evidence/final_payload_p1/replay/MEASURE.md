# MEASURE off4 vs on4

## (a) PWML written, flag off -> on

| group | runs | off | on |
|---|---|---|---|
| P0 | 40 | 37 | 37 |
| P1 | 61 | 60 | 60 |
| RS | 51 | 0 | 0 |

State changes: none

## (b) ids gained: 547 in 127 runs; refused 53 {'short_abbreviation_synonym': 36, 'ambiguous': 1, 'no_structure_in_chebi': 15, 'id_used_by_other_row': 1}

- by kind: {'compound': 547}
- by field: {'drugbank_id': 138, 'kegg_id': 149, 'hmdb_id': 27, 'cas': 127, 'chebi_id': 106}
- by source: {'ChEBI': 547}
- by match_type: {'cross_reference': 441, 'exact_label': 52, 'exact_synonym+charged_form': 22, 'exact_synonym': 32}
- by kind_field: {'compound.drugbank_id': 138, 'compound.kegg_id': 149, 'compound.hmdb_id': 27, 'compound.cas': 127, 'compound.chebi_id': 106}

## (c)
- written only with flag on: []
- compounds with any db id (runs written both ways): {"runs": 97, "total_compounds_off": 687, "with_id_off": 406, "with_id_on": 484, "runs_increased": 48}

## (d) titles/references
{"written": 97, "real_title": 97, "pubmed_id_nonempty": 97, "any_reference_element": 97, "title_source:NCBI": 97, "off_title_real": 0}

## (e) PWML differing from flag-off: 97 (identical: 0)

## QA/well-formedness
- not well-formed: 0 of 97
- qa off: {'written': 97, 'qa_ok_false': [], 'qa_errors_total': 0, 'qa_warnings_total': 388}
- qa on: {'written': 97, 'qa_ok_false': [], 'qa_errors_total': 0, 'qa_warnings_total': 467}

## Live requests
- total 0 (errors 0), by pass {1: 0}
