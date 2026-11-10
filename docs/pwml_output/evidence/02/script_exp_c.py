import json,re,urllib.parse,collections,sys
import netc
GREEK={'α':'alpha','β':'beta','γ':'gamma','δ':'delta','ω':'omega','Δ':'delta'}
def norm(s):
    s=s.lower()
    for k,v in GREEK.items(): s=s.replace(k,v)
    return re.sub(r'[^a-z0-9]+','',s)
def variants(name):
    n=name.strip()
    out=[n]
    m=re.match(r'^(.*?)\s*\(([^()]+)\)\s*$',n)
    if m and len(m.group(1))>2:
        out.append(m.group(1).strip())
        if len(m.group(2))<=10: out.append(m.group(2).strip())   # abbreviation in parentheses
    # ions
    ion={'Zn2+':'Zn(2+)','Zn²⁺':'Zn(2+)','Ca++':'Ca(2+)','Ca2+':'Ca(2+)','FeII':'Fe(2+)','Fe3+':'Fe(3+)','Fe3':'Fe(3+)','Mg2+':'Mg(2+)'}
    if n in ion: out.append(ion[n])
    g=n
    for k,v in GREEK.items(): g=g.replace(k,v)
    if g!=n: out.append(g)
    seen=[];[seen.append(x) for x in out if x and x not in seen]
    return seen[:3]
def ols(v):
    q='https://www.ebi.ac.uk/ols4/api/search?q='+urllib.parse.quote(v)+'&ontology=chebi&exact=true&queryFields=label,synonym&rows=6&fieldList=obo_id,label,synonym'
    t=netc.get(q,'ols')
    try: d=json.loads(t)['response']['docs']
    except Exception: return None
    return [dict(id=x['obo_id'],label=x.get('label'),syn=x.get('synonym',[])[:8]) for x in d]
def kegg(v):
    t=netc.get('https://rest.kegg.jp/find/compound/'+urllib.parse.quote(v),'kegg','text/plain')
    if t.startswith('HTTPERR') or t.startswith('NETERR'): return None
    out=[]
    for line in t.splitlines():
        if '\t' in line:
            a,b=line.split('\t',1); out.append(dict(id=a.replace('cpd:',''),names=[x.strip() for x in b.split(';')]))
    return out
def run(u):
    name=u['name']; vs=variants(name)
    r=dict(name=name,variants=vs,organism=u['organism'],paper=u['paper'],chebi=[],kegg=[],tries=0)
    for v in vs[:2]:
        h=ols(v)
        r['tries']+=1
        if h:
            nv=norm(v)
            exact=[x for x in h if nv==norm(x['label'] or '') or nv in [norm(s) for s in x['syn']]]
            if exact:
                r['chebi']=[dict(x,matched_variant=v) for x in exact]; break
    # KEGG once, on the first two variants
    for v in vs[:2]:
        k=kegg(v)
        if k:
            nv=norm(v)
            exact=[x for x in k if nv in [norm(s) for s in x['names']]]
            if exact:
                r['kegg']=[dict(x,matched_variant=v) for x in exact]; break
            r.setdefault('kegg_near',[x['id'] for x in k[:3]])
    return r
if __name__=='__main__':
    C=json.load(open('sample_compounds.json',encoding='utf-8'))
    res=[]
    for i,u in enumerate(C):
        res.append(run(u))
        if i%20==0: print(i,netc.ncached('ols'),netc.ncached('kegg'),flush=True)
    json.dump(res,open('results_c.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
    print(sum(1 for r in res if r['chebi']),sum(1 for r in res if r['kegg']),len(res),netc.ncached('ols'),netc.ncached('kegg'))
