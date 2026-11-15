"""Trace each of the 37 transports in the 60 imported files: what would still fail after the left/right state-id remap."""
import json, collections
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter(); n=0
for r in rows:
    t=ET.parse(r['file']).getroot()
    for tr in t.iter('transport'):
        if tr.find('transport-elements') is None: continue
        n+=1
        tes=tr.findall('transport-elements/transport-element')
        trs=tr.findall('transport-transporters/transport-transporter')
        te=tes[0]
        same = te.findtext('left-biological-state-id')==te.findtext('right-biological-state-id')
        C[('element_type',te.findtext('element-type'))]+=1
        C[('left==right state',same)]+=1
        C[('n_transporters',len(trs))]+=1
        for x in trs:
            C[('transporter_field','protein-complex-id' if x.find('protein-complex-id') is not None else 'protein-id' if x.find('protein-id') is not None else 'none')]+=1
        # would it survive after state remap fix?
        issues=[]
        if same: issues.append('left==right state (TransportElement#biological_states_unique)')
        if te.findtext('element-type')=='Protein': issues.append('element-type Protein (not a PathWhiz transport element type)')
        if any(x.find('protein-complex-id') is None for x in trs): issues.append('transporter has protein-id only')
        C[('residual', tuple(issues))]+=1
print('transports',n)
for k,v in sorted(C.items(),key=str): print(k,v)
