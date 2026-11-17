"""Reactions with no biological_state fall to biological_states[0] (ir.py:1836). Which compartment do they land in, per pathway?"""
import json
res = {}
CELL = {'cell', 'cytoplasm', 'cytosol', 'intracellular', 'cytoplasmic'}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    d = json.load(open('C:/t/corr-assess/base/%d/final_payload.json' % i, encoding='utf-8'))
    rx = d['processes']['reactions']
    nulls = [r for r in rx if not r.get('biological_state')]
    bs = d['biological_states']
    first = bs[0] if bs else None
    named = {r.get('biological_state') for r in rx if r.get('biological_state')}
    res[i] = {'reactions': len(rx), 'null_state': len(nulls), 'default_state': first['name'] if first else None,
              'default_location': first['subcellular_location'] if first else None,
              'other_states_used_by_named_reactions': sorted(named)}
    flag = ''
    if nulls and first and first['subcellular_location'].lower() not in CELL:
        flag = '  <-- default is NOT a plain cell/cytosol compartment'
    if nulls:
        print(i, res[i]['null_state'], '/', len(rx), 'default:', res[i]['default_state'], '|', res[i]['default_location'], '| named:', res[i]['other_states_used_by_named_reactions'], flag)
json.dump(res, open('C:/t/corr-assess/a1/null_state_default.json', 'w', encoding='utf-8'), indent=1)
