import json,random,collections,re
from cause import protein_cause
random.seed(20261110)
U=json.load(open('units.json',encoding='utf-8'))
tax=json.load(open('org_tax.json',encoding='utf-8'))
P=[u for u in U if u['kind']=='protein' and u['cls']!='mapped']
for u in P:
    u['cause']=protein_cause(u['name']); u['taxid']=tax.get(u['organism'])
    # manual overrides after reading names
OV={'sphingomyelinase':'family_or_class_term','ceramidase':'family_or_class_term','c-Jun':'gene_symbol','Frizzled':'family_or_class_term','Strabismus (Stbm)':'gene_symbol',
 'cAMP receptor protein':'gene_symbol','catabolite control protein CcpA':'gene_symbol','AraC dimer':'modified_form','Dvl-DAAM complex':'complex_or_multi_entity','siderosomes':'complex_or_multi_entity',
 'TcyJKLMN':'complex_or_multi_entity','NIT-7AB':'complex_or_multi_entity','HvACT-2HS1':'gene_symbol','HvACT-2HL1':'gene_symbol','HvACT-2HL2':'gene_symbol','PhBSα':'gene_symbol','β-GD1':'gene_symbol',"AmI3'H":'gene_symbol'}
for u in P:
    if u['name'] in OV: u['cause']=OV[u['name']]
print(collections.Counter(u['cause'] for u in P))
def oclass(o):
    return 'model' if o in ('Homo sapiens','Mus musculus','Escherichia coli','Arabidopsis thaliana','Bacillus subtilis','Bacillus subtilis (strain 168)') else 'other'
A=[u for u in P if u['cause']!='gene_symbol']
B=[u for u in P if u['cause']=='gene_symbol']
# stratify B by organism class and cls (placeholder/unmapped)
strata=collections.defaultdict(list)
for u in B: strata[(oclass(u['organism']),u['cls'])].append(u)
print({k:len(v) for k,v in strata.items()})
target=44
tot=len(B); S=[]
for k,v in strata.items():
    n=max(2,round(target*len(v)/tot)); random.shuffle(v); S+=v[:n]
S=S[:target+4]
sample=A+S
for u in sample: u['stratum']=u['cause']
print(len(A),len(S),len(sample))
json.dump(sample,open('sample_proteins.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
json.dump(P,open('frame_proteins.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
C=[u for u in U if u['kind']=='compound' and u['cls']!='mapped']
json.dump(C,open('sample_compounds.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
print('compounds',len(C))
print(collections.Counter((u['organism'] or '')[:20] for u in sample).most_common(12))
