import json,re,urllib.parse,collections,sys
import netc
from exp_a import us as _us, base_name
F2='accession,gene_names,gene_primary,protein_name,reviewed,organism_name,organism_id,ec,sequence,length'
def us2(q,size=8):
    u='https://rest.uniprot.org/uniprotkb/search?query='+urllib.parse.quote(q)+'&fields='+F2+'&size=%d&format=json'%size
    t=netc.get(u,'uniprot')
    if t.startswith('HTTPERR') or t.startswith('NETERR'): return []
    try: return json.loads(t).get('results',[])
    except Exception: return []
def P(r):
    pd=r.get('proteinDescription') or {}
    rn=pd.get('recommendedName') or ((pd.get('submissionNames') or [{}])[0])
    genes=r.get('genes') or []
    return dict(acc=r['primaryAccession'],rev=r['entryType'].startswith('UniProtKB reviewed'),gene=(genes[0].get('geneName') or {}).get('value') if genes else None,
        pn=((rn or {}).get('fullName') or {}).get('value'),org=r['organism']['scientificName'],orgid=r['organism']['taxonId'],
        lin=r['organism'].get('lineage',[]),ec=[e.get('value') for e in (pd.get('recommendedName') or {}).get('ecNumbers',[])],
        seq=(r.get('sequence') or {}).get('value',''),len=(r.get('sequence') or {}).get('length'))
def align(a,b,match=5,mis=-3,gap=-6):
    """semi-global (free end gaps) linear-gap alignment. returns identity over aligned columns, aligned cols, matches"""
    n,m=len(a),len(b)
    if not n or not m: return 0,0,0
    prev=[0]*(m+1); trace=[]
    H=[[0]*(m+1) for _ in range(n+1)]
    T=[[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        ai=a[i-1]; Hi=H[i]; Hp=H[i-1]; Ti=T[i]
        for j in range(1,m+1):
            d=Hp[j-1]+(match if ai==b[j-1] else mis)
            u=Hp[j]+gap
            l=Hi[j-1]+gap
            if d>=u and d>=l: Hi[j]=d; Ti[j]=0
            elif u>=l: Hi[j]=u; Ti[j]=1
            else: Hi[j]=l; Ti[j]=2
    # best end: last row or last col (free end gaps)
    best=(-10**9,n,m)
    for j in range(m+1):
        if H[n][j]>best[0]: best=(H[n][j],n,j)
    for i in range(n+1):
        if H[i][m]>best[0]: best=(H[i][m],i,m)
    _,i,j=best; cols=0; matches=0
    while i>0 and j>0:
        t=T[i][j]
        if t==0:
            cols+=1; matches+= (a[i-1]==b[j-1]); i-=1; j-=1
        elif t==1: i-=1
        else: j-=1
    return (matches/cols if cols else 0),cols,matches
def metrics(seed,cand):
    idt,cols,mt=align(seed['seq'],cand['seq'])
    ls,lc=len(seed['seq']),len(cand['seq'])
    return dict(acc=cand['acc'],gene=cand['gene'],pn=cand['pn'],org=cand['org'],rev=cand['rev'],ident=round(idt,3),cov_seed=round(cols/ls,3),cov_cand=round(cols/lc,3),len=lc)
def target_pool(seed,T):
    pool=[]
    if seed['pn']:
        pool=us2('protein_name:"%s" AND taxonomy_id:%s'%(seed['pn'].replace('"',''),T),size=10)
    if not pool and seed['ec']:
        pool=us2('ec:%s AND taxonomy_id:%s'%(seed['ec'][0],T),size=10)
    return [P(r) for r in pool]
def pick_seed(rows,name,prefer_orgids=()):
    rows=[P(r) for r in rows if r.get('sequence')]
    if not rows: return None
    ex=[r for r in rows if (r['gene'] or '').lower()==name.lower()] or rows
    for pid in prefer_orgids:
        for r in ex:
            if r['orgid']==pid: return r
    return ex[0]
def run_unit(u):
    T=u['taxid']; b=base_name(u['name'])
    out=dict(name=u['name'],base=b,organism=u['organism'],taxid=T,cause=u['cause'],cls=u['cls'])
    rows=us2('gene_exact:%s AND reviewed:true'%b,size=8)
    how='gene_exact'
    if not rows and u['cause'] in ('descriptive_enzyme_name','other'):
        rows=us2('protein_name:"%s" AND reviewed:true'%b,size=8); how='protein_name'
    seed=pick_seed(rows,b)
    if not seed:
        out['seed']=None; return out
    out['seed']=dict(acc=seed['acc'],gene=seed['gene'],pn=seed['pn'],org=seed['org'],len=seed['len'],how=how,n_seed_rows=len(rows))
    pool=target_pool(seed,T)
    out['pool_n']=len(pool)
    m=[metrics(seed,c) for c in pool if c['seq'] and c['acc']!=seed['acc']]
    m.sort(key=lambda x:-x['ident']*min(1,x['cov_seed']))
    out['hits']=m[:5]
    return out
if __name__=='__main__':
    A=json.load(open('results_a.json',encoding='utf-8'))
    S=json.load(open('sample_proteins.json',encoding='utf-8'))
    by={u['name']+'|'+u['organism']:u for u in S}
    todo=[]
    for r in A:
        if r['result']['tier'] in ('none','ambiguous') and r['cause'] in ('gene_symbol','descriptive_enzyme_name','other'):
            u=by.get(r['name']+'|'+r['organism'])
            if u: todo.append(u)
    print('todo',len(todo),flush=True)
    res=[]
    for i,u in enumerate(todo):
        res.append(run_unit(u))
        if i%8==0: print(i,netc.ncached('uniprot'),flush=True)
    json.dump(res,open('results_b.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
    print('done',netc.ncached('uniprot'))
