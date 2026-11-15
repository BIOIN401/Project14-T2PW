import json, sys
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c
i = int(sys.argv[1])
d = json.load(open('C:/t/corr-assess/base/%d/final_payload.json' % i, encoding='utf-8'))
print('PAYLOAD')
for t in d['processes']['transports']:
    print(' ', t['name'], '| cargo', t['cargo'], '|', t['from_biological_state'], '->', t['to_biological_state'], '| transporters', [x.get('entity') if isinstance(x, dict) else x for x in t.get('transporters', [])])
    for e in t.get('elements_with_states', []):
        print('      ', e.get('element'), e.get('biological_state'), e.get('side'))
for t in d['processes']['interactions']:
    print(' INT', t['name'], t['entity_1'], t['relationship'], t['entity_2'], t.get('biological_state'))
P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
nm, st = P['names'], P['states']
print('PWML')
for t in P['transports']:
    print(' ', t['id'], [(nm.get((e[0], e[1])), st.get(e[2], {}).get('name'), '->', st.get(e[3], {}).get('name'), e[4]) for e in t['elements']], [nm.get(('ProteinComplex', x[0])) for x in t['transporters']])
for t in P['interactions']:
    print(' INT', t['type'], [nm.get(x) for x in t['left']], '->', [nm.get(x) for x in t['right']])
