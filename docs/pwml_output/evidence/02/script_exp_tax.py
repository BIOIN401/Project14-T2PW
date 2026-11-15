import json,netc
tax=json.load(open('org_tax.json',encoding='utf-8'))
S=json.load(open('frame_proteins.json',encoding='utf-8'))
orgs=sorted({u['organism'] for u in S if u['organism']})
out={}
for o in orgs:
    t=tax.get(o)
    if not t: continue
    txt=netc.get('https://rest.uniprot.org/taxonomy/%s'%t,'uniprot')
    try:
        d=json.loads(txt); st=d.get('statistics',{})
        out[o]=dict(taxid=t,reviewed=st.get('reviewedProteinCount'),unreviewed=st.get('unreviewedProteinCount'),proteomes=st.get('proteomeCount'),refprot=st.get('referenceProteomeCount'),rank=d.get('rank'),lineage=[l['scientificName'] for l in d.get('lineage',[])][-4:])
    except Exception as e: out[o]=dict(taxid=t,err=txt[:100])
json.dump(out,open('org_stats.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
for o,v in out.items(): print(o,v)
