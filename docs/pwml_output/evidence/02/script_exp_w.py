import json,re,random,collections
import netc
random.seed(11)
E=[json.loads(l) for l in open('entities.jsonl',encoding='utf-8')]
def norm(s):
    s=re.sub(r'<[^>]+>','',s or '').lower()
    for k,v in {'α':'alpha','β':'beta','γ':'gamma','δ':'delta'}.items(): s=s.replace(k,v)
    s=re.sub(r'^[dl]-','',s); s=re.sub(r'\b(acid|ate|ic)\b','',s)
    return re.sub(r'[^a-z0-9]+','',s)
def chebi_names(cid):
    cid=str(cid).replace('CHEBI:','')
    t=netc.get('https://www.ebi.ac.uk/chebi/backend/api/public/compound/CHEBI:%s/'%cid,'chebi')
    try: d=json.loads(t)
    except Exception: return None
    names=[d.get('name') or '',d.get('ascii_name') or '']
    for k,v in (d.get('names') or {}).items():
        names+= [x.get('name') or '' for x in v]
    inchikey=None
    ds=d.get('default_structure') or {}
    inchikey=ds.get('standard_inchi_key')
    return dict(label=d.get('ascii_name'),names=names,inchikey=inchikey,formula=(d.get('chemical_data') or {}).get('formula'),stars=d.get('stars'))
def pairs(sel):
    seen={}
    for e in E:
        if e['bucket']!='compounds' or not sel(e): continue
        c=e['ids'].get('chebi_id')
        if not c: continue
        seen[(e['name'].lower(),str(c).replace('CHEBI:',''))]=e
    return list(seen.items())
amb=pairs(lambda e: e['status']=='ambiguous')
base=pairs(lambda e: e['status']=='matched' and e['step']=='pathbank_compound_id')
random.shuffle(base); base=base[:30]
res=[]
for tag,L in (('ambiguous_status',amb),('matched_by_pathbank_id_baseline',base)):
    for (nm,c),e in L:
        info=chebi_names(c)
        if not info: res.append(dict(tag=tag,name=e['name'],chebi=c,err=True)); continue
        nn=norm(e['name']); nn2=norm(re.sub(r'\s*\(.*\)$','',e['name']))
        ab=re.findall(r'\(([^()]+)\)\s*$',e['name']); nab=[norm(a) for a in ab]
        ns=[norm(x) for x in info['names']]
        exact= nn in ns or nn2 in ns or any(a and a in ns for a in nab)
        part= any(nn2 and len(nn2)>3 and (nn2 in x or x in nn2) for x in ns if len(x)>3)
        res.append(dict(tag=tag,name=e['name'],chebi=c,label=info['label'],inchikey=info['inchikey'],agree='exact' if exact else ('partial' if part else 'none'),paper=e['paper'],step=e['step'],chosen=e['chosen']))
json.dump(res,open('results_w.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
for tag in ('ambiguous_status','matched_by_pathbank_id_baseline'):
    r=[x for x in res if x['tag']==tag]
    print(tag,len(r),collections.Counter(x.get('agree','err') for x in r))
for x in res:
    if x.get('agree')=='none': print(x['tag'][:4],x['name'],'->',x['chebi'],x['label'])
print(netc.ncached('chebi'))
