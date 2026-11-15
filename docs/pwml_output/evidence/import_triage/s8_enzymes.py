"""Reaction enzymes: protein-id vs protein-complex-id, and whether the referenced complex failed on import."""
import json, collections, re
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter()
for r in rows:
    t=ET.parse(r['file']).getroot()
    bad_cx={m.group(1) for k in r['errors'] for m in [re.fullmatch(r'protein_complex_(\d+)',k)] if m}
    cx_ids={c.findtext('id') for c in t.iter('protein-complex')}
    for e in t.iter('reaction-enzyme'):
        if e.find('id') is None: continue
        pc=e.findtext('protein-complex-id'); p=e.findtext('protein-id')
        if pc: C[('enzyme via protein-complex-id', 'complex failed import' if pc in bad_cx else 'complex ok')]+=1
        elif p: C[('enzyme via protein-id only (no complex)', '')]+=1
    for c in t.iter('protein-complex'):
        if c.find('protein_complex-proteins') is None: continue
        cid=c.findtext('id')
        C[('complex','failed' if cid in bad_cx else 'ok')]+=1
        if cid in bad_cx:
            prots=[x.findtext('protein-id') for x in c.findall('protein_complex-proteins/protein-complex-protein')]
            C[('failed complex n_members',len(prots))]+=1
    # extra file for complexes too short beyond sentinel
for k,v in sorted(C.items(),key=str): print(k,v)
