import json, sys
d = json.load(open('C:/t/corr-assess/a1/per_pathway.json', encoding='utf-8'))
for k in sys.argv[1:]:
    v = d[k]
    print('==', k, v['canvas'])
    print(' payload states', v['states']['payload'], 'dup names', v['states']['payload_dup_names'])
    print(' used', v['states']['used_in_pwml'])
    print(' unused', v['states']['defined_unused'])
    print(' syn', v['states']['synonym_groups_used'])
    print(' species', v['states']['species_among_used'])
    print(' labels', v['compartment_boxes']['labels'], 'dup', v['compartment_boxes']['dup_labels'], 'strain', v['compartment_boxes']['organism_or_strain_named'])
    print(' rx state payload', v['states']['reaction_state_payload'])
    print(' rx state pwml', v['states']['reaction_state_pwml'])
    print(' transports', {x: v['transports'][x] for x in ('payload', 'pwml', 'missing_in_pwml', 'extra_in_pwml', 'transporters_payload', 'transporters_pwml', 'same_state_both_ends_pwml')})
