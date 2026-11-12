"""transport_compound_visualization 'Compound location can't be blank': are the referenced compound-locations in the PWML?"""
import json, collections
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter()
for r in rows:
    errs=[k for k in r['errors'] if k.startswith('transport_compound_visualization_')]
    if not errs: continue
    t=ET.parse(r['file']).getroot()
    cl={c.findtext('id'):c for c in t.iter('compound-location') if c.find('compound-id') is not None}
    for e in t.iter('transport-compound-visualization'):
        if e.find('id') is None: continue
        eid=e.findtext('id'); ref=e.findtext('compound-location-id')
        failed = ('transport_compound_visualization_'+eid) in r['errors']
        c=cl.get(ref)
        C[('failed' if failed else 'ok', 'cl in PWML' if c is not None else 'cl MISSING in PWML')]+=1
        if failed and c is not None:
            C[('failed; compound-location bs', c.findtext('biological-state-id') is not None)]+=1
    cl_err=[k for k in r['errors'] if k.startswith('compound_location_')]
    if cl_err: C['file with compound_location error']+=1
for k,v in sorted(C.items(),key=str): print(k,v)
