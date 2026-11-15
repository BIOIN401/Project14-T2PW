"""Which proteins/species fail and why: species classification blank vs protein species."""
import json, collections
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter()
for r in rows:
    t=ET.parse(r['file']).getroot()
    sp={}
    for s in t.iter('species'):
        if s.find('taxonomy-id') is None: continue
        sp[s.findtext('id')]=(s.findtext('name'),s.findtext('taxonomy-id'),s.findtext('classification'))
    perr=[k for k in r['errors'] if k.startswith('protein_') and not k.startswith('protein_complex') and not k.startswith('protein_location') and not k.startswith('protein_complex')]
    sperr=[k for k in r['errors'] if k.startswith('species_')]
    blank=[k for k,v in sp.items() if not v[2]]
    prots=[p for p in t.iter('protein') if p.find('uniprot-id') is not None]
    bad_ids=[k.split('_',1)[1] for k in perr]
    for p in prots:
        pid=p.findtext('id'); sid=p.findtext('species-id')
        if pid in bad_ids:
            C[('failing protein species-id in blank-class species', sid in blank)]+=1
    C[('file has blank-class species',bool(blank))]+=1
    C[('file has species error',bool(sperr))]+=1
    C[('file has protein species error', bool(perr))]+=1
    C[('same set',bool(blank)==bool(perr))]+=1
    # how many proteins use a blank-class species
    nbad=sum(1 for p in prots if p.findtext('species-id') in blank)
    C['proteins using blank-class species']+=nbad; C['proteins total']+=len(prots)
    if perr: print(r['file'][-70:], perr[:3], [sp[p.findtext('species-id')] for p in prots if p.findtext('id') in bad_ids][:2], 'n_prot',len(prots),'blank-class',nbad)
for k,v in sorted(C.items(),key=str): print(k,v)
