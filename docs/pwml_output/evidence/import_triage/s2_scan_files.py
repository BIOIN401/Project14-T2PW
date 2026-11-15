"""Scan all 60 imported T2PW PWML files: interaction types, element types, ontology ids, species classification, transports, etc."""
import json, collections, re
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
C=collections.Counter()
per=collections.defaultdict(set)
for r in rows:
    try: t=ET.parse(r['file']).getroot()
    except Exception as e: print('ERR',r['file'],e); continue
    for i in t.iter('interaction'):
        ty=i.findtext('interaction-type'); C[('itype',ty)]+=1; per[('itype',ty)].add(r['file'])
        for side in ('left','right'):
            els=i.findall(f'interaction-{side}-elements/interaction-{side}-element')
            C[('i_n_'+side,len(els))]+=1
            for e in els: C[('i_etype',e.findtext('element-type'))]+=1
    for tag in ('subcellular-location','cell-type','tissue'):
        for e in t.iter(tag):
            if e.find('ontology-id') is None: continue
            oid=e.findtext('ontology-id')
            k=(tag,'nil' if oid is None else 'set'); C[k]+=1; per[k].add(r['file'])
    for s in t.iter('species'):
        if s.find('taxonomy-id') is None: continue
        k=('species_class',s.findtext('classification')); C[k]+=1; per[k].add(r['file'])
        C[('species_tax_set',bool(s.findtext('taxonomy-id')))]+=1
    for tr in t.iter('transport'):
        C[('transport_n_elems',len(tr.findall('transport-elements/transport-element')))]+=1
        C[('transport_type',tr.findtext('transport-type'))]+=1
        for te in tr.findall('transport-elements/transport-element'):
            C[('te_etype',te.findtext('element-type'))]+=1
            C[('te_dir',te.findtext('direction'))]+=1
            C[('te_states_same',te.findtext('left-biological-state-id')==te.findtext('right-biological-state-id'))]+=1
    for bv in t.iter('bound-visualization'):
        C[('bv_has_boundid',bv.find('bound-id') is not None)]+=1
    for b in t.iter('bound'): C['bound_elems']+=1
    for p in t.iter('protein'):
        if p.find('uniprot-id') is None: continue
        C[('protein_species_set',bool(p.findtext('species-id')))]+=1
for k,v in sorted(C.items(),key=lambda x:str(x[0])): print(k,v,len(per[k]) if k in per else '')
