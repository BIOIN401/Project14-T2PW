# KGML export of all canonical runs

XML written to `C:/t/ev-cm/kgml_out` (outside the repo). Structural check = required attributes, unique ids, references resolve (no KEGG service is called).

- runs: 152, files written: 152, structural check passed: 152, failures: 0
- entries: 2293; KEGG id 556 (24.2%), non-KEGG prefix 747 (32.6%), undefined 990 (43.2%)
- reactions (incl. transports): 724; rn: 0, rhea: 0, undefined 724
- relations: 325; signed 106, unsigned 219

## Omission totals

- direction_placeholder: 724
- lost_catalysts_and_modifiers: 1503
- lost_compartments: 414
- lost_quotes: 1051
- lost_stoichiometry_participants: 1544
- lost_transport_locations: 103
- runs_using_prefix_chebi: 53
- runs_using_prefix_uniprot: 127
- skipped_group_members: 154
- skipped_interactions: 27
- skipped_reactions: 4
- unmapped_entities: 990
