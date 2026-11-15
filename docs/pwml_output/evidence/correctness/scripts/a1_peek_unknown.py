import json, sys, collections
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c

d = json.load(open('C:/t/corr-assess/base/134/final_payload.json', encoding='utf-8'))
print([(s['name'], s['species'], s['subcellular_location']) for s in d['biological_states']])
print([s['name'] for s in d['entities']['species']], [(p['name'], p.get('organism')) for p in d['entities']['proteins']][:5])
print([(p['name'], p.get('species')) for p in d['entities']['protein_complexes']][:5])
print(open('C:/t/corr-assess/base/134/meta.json', encoding='utf-8').read()[:300])
# Unknown protein use
tot = collections.Counter()
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
    unk = [k for k, v in P['names'].items() if k[0] == 'Protein' and v.lower() == 'unknown']
    ids = {k[1] for k in unk}
    in_cx = sum(1 for cp in P['complex_proteins'].values() for x in cp if x == 'Unknown')
    drawn = sum(1 for p in P['ploc'].values() if p['protein'] in ids)
    if unk:
        tot['files_with_Unknown'] += 1
        print(i, 'Unknown protein ids', ids, 'in complexes', in_cx, 'drawn', drawn)
print(tot)
