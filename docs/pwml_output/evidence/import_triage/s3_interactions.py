"""Per-interaction trace: type validity, elements present, whether referenced element survived the importer."""
import json, re
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
OK={'Activation','Inhibition'}
tot={'valid_type':0,'invalid_type':0,'invalid_with_elems':0}
for r in rows:
    f=r['file']; date=re.search(r'(20dd-dd-dd)',f.replace(chr(92),'/'))
    t=ET.parse(f).getroot()
    ids={'ProteinComplex':{e.findtext('id') for e in t.iter('protein-complex')},
         'Compound':{e.findtext('id') for e in t.iter('compound') if e.find('hmdb-id') is not None or e.find('name') is not None},
         'NucleicAcid':{e.findtext('id') for e in t.iter('nucleic-acid')}}
    for i in t.iter('interaction'):
        if i.find('interaction-type') is None: continue
        ty=i.findtext('interaction-type')
        els=[]
        for side in ('left','right'):
            for e in i.findall(f'interaction-{side}-elements/interaction-{side}-element'):
                et=e.findtext('element-type'); eid=e.findtext('element-id')
                els.append((side,et,eid,eid in ids.get(et,set())))
        if ty in OK:
            tot['valid_type']+=1
            msgs=[k for k in r['errors'] if k.startswith('interaction_') and re.search(r'_(left|right)_element_',k)]
            ce=[ (k,r['errors'][k]) for k in r['errors'] if k in {f'protein_complex_{e[2]}' for e in els}]
            print(date.group(1) if date else '?', f[-60:], ty, els, 'complex_errors=',[k for k,_ in ce], 'interaction created=',r['created'].get('interactions',0))
        else:
            tot['invalid_type']+=1
            if els: tot['invalid_with_elems']+=1
print(tot)
# date histogram of files with invalid-type interactions
