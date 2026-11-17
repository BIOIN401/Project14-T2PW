import json, sys
for i in sys.argv[1:]:
    d = json.load(open('C:/t/corr-assess/base/%s/final_payload.json' % i, encoding='utf-8'))
    print(i, [(s['name'], s['subcellular_location']) for s in d['biological_states']], 'default(first)=', d['biological_states'][0]['name'] if d['biological_states'] else None)
    print('   rx states', sorted({str(r.get('biological_state')) for r in d['processes']['reactions']}), 'n', len(d['processes']['reactions']),
          'compound_locations states', sorted({x.get('biological_state') for x in d['element_locations'].get('compound_locations', [])}))
