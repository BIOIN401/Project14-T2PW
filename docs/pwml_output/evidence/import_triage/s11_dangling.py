import json, collections
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter(); shown=0
for r in rows:
    if not any(k.startswith('transport_compound_visualization_') for k in r['errors']): continue
    t=ET.parse(r['file']).getroot()
    owner={}
    for el in t.iter():
        i=el.find('id')
        if i is not None and i.text: owner.setdefault(i.text,[]).append(el.tag)
    cl={c.findtext('id') for c in t.iter('compound-location') if c.find('compound-id') is not None}
    for e in t.iter('transport-compound-visualization'):
        if e.find('id') is None: continue
        ref=e.findtext('compound-location-id')
        if ref not in cl:
            C[tuple(owner.get(ref,['<nowhere>']))]+=1
            if shown<3: shown+=1; print(r['file'][-60:], 'ref',ref, owner.get(ref))
print(C)
