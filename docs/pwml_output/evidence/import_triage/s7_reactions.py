"""Reactions failing 'Spontaneous cannot be No': enzymes absent in the PWML itself (content gap) or lost in import (cascade)?"""
import json, collections
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter(); files=collections.defaultdict(set)
for r in rows:
    t=ET.parse(r['file']).getroot()
    failing={k.split('_',1)[1] for k in r['errors'] if k.startswith('reaction_') and not k.startswith('reaction_visual') and not k.startswith('reaction_enzyme') and not k.startswith('reaction_compound') and not k.startswith('reaction_element') and not k.startswith('reaction_coupled')}
    bad_cx={k.split('_',2)[2] for k in r['errors'] if k.startswith('protein_complex_') and k.count('_')==2 and k.split('_',2)[2].isdigit()}
    for rx in t.iter('reaction'):
        if rx.find('reaction-left-elements') is None: continue
        rid=rx.findtext('id'); sp=rx.findtext('spontaneous')
        enz=rx.findall('reaction-enzymes/reaction-enzyme')
        C[('spont_text',sp, 'nil' if rx.find('spontaneous') is not None and rx.find('spontaneous').get('nil') else '')]+=1
        if rid in failing:
            kind = 'no enzyme in PWML' if not enz else 'enzymes in PWML but complex lost on import'
            C[('failing reaction',kind)]+=1; files[kind].add(r['file'])
        else:
            C[('imported reaction','with enzymes' if enz else 'no enzymes (spontaneous nil/true ok)')]+=1
for k,v in sorted(C.items(),key=str): print(k,v)
for k,v in files.items(): print(k,'files:',len(v))
