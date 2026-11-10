import json,re,urllib.parse,collections,sys
import netc
FIELDS='accession,gene_names,gene_primary,protein_name,reviewed,organism_name,organism_id,ec'
def us(query,fields=FIELDS,size=25):
    u='https://rest.uniprot.org/uniprotkb/search?query='+urllib.parse.quote(query)+'&fields='+fields+'&size=%d&format=json'%size
    t=netc.get(u,'uniprot')
    if t.startswith('HTTPERR') or t.startswith('NETERR'): return None,t
    try: return json.loads(t).get('results',[]),None
    except Exception as e: return None,str(e)
def parse(r):
    pn=((r.get('proteinDescription') or {}).get('recommendedName') or (r.get('proteinDescription') or {}).get('submissionNames',[{}])[0] if (r.get('proteinDescription') or {}).get('recommendedName') else ((r.get('proteinDescription') or {}).get('submissionNames') or [{}])[0])
    pnv=((pn or {}).get('fullName') or {}).get('value')
    genes=r.get('genes') or []
    prim=(genes[0].get('geneName') or {}).get('value') if genes else None
    syn=[s['value'] for g in genes for s in (g.get('synonyms') or [])]
    return dict(acc=r['primaryAccession'],rev=r['entryType'].startswith('UniProtKB reviewed'),gene=prim,syn=syn,pn=pnv,org=r['organism']['scientificName'],orgid=r['organism']['taxonId'],ec=[e.get('value') for e in ((r.get('proteinDescription') or {}).get('recommendedName') or {}).get('ecNumbers',[])])
def base_name(n):
    n=re.sub(r'\s+complex$','',n.strip(),flags=re.I)
    n=re.sub(r'^(apo|holo|phosphorylated|human|porcine)[- ]','',n,flags=re.I)
    m=re.match(r'^(.*?)\s*\((.*)\)$',n)
    if m: n=m.group(1).strip()
    return n
def decide(hits,T):
    if not hits: return dict(tier='none')
    rev=[h for h in hits if h['rev']]
    pool=rev or hits
    keys=collections.OrderedDict()
    for h in pool:
        k=(h['gene'] or h['pn'] or '').lower()
        keys.setdefault(k,[]).append(h)
    # same-protein different gene-symbol casing handled by lower
    if len(keys)==1:
        g=list(keys.values())[0]
        exact=[h for h in g if h['orgid']==T]
        k12=[h for h in g if h['orgid']==83333]
        rep=(exact or k12 or g)[0]
        return dict(tier='unique',reviewed=bool(rev),rep=rep,n_strain=len(g),n_groups=1)
    return dict(tier='ambiguous',reviewed=bool(rev),n_groups=len(keys),groups=[(k,v[0]['acc'],v[0]['pn']) for k,v in list(keys.items())[:5]])
def run(u):
    T=u['taxid']; name=u['name']; b=base_name(name); out=dict(name=name,base=b,organism=u['organism'],taxid=T,cause=u['cause'],cls=u['cls'],queries=[])
    if u['cause'] in ('rna_or_gene_not_protein',):
        out['result']=dict(tier='skipped',why='not a protein'); return out
    if u['cause']=='complex_or_multi_entity':
        out['result']=dict(tier='skipped',why='multi-entity: needs split into subunits'); return out
    qs=[]
    if u['cause'] in ('gene_symbol','modified_form','other') and re.match(r'^[\w\-\.\']+$',b): qs.append(('gene_exact','gene_exact:%s AND taxonomy_id:%s'%(b,T)))
    qs.append(('protein_name','protein_name:"%s" AND taxonomy_id:%s'%(b,T)))
    if u['cause']=='gene_symbol': qs.append(('text','(gene:%s OR protein_name:%s) AND taxonomy_id:%s'%(b,b,T)))
    for lab,q in qs:
        res,err=us(q)
        out['queries'].append((lab,q,len(res) if res is not None else err))
        if res:
            hits=[parse(r) for r in res]
            d=decide(hits,T); d['via']=lab
            if d['tier']=='unique': out['result']=d; return out
            out.setdefault('result',d)
    out.setdefault('result',dict(tier='none'))
    return out
if __name__=='__main__':
    S=json.load(open('sample_proteins.json',encoding='utf-8'))
    res=[]
    for i,u in enumerate(S):
        res.append(run(u))
        if i%10==0: print(i,netc.ncached('uniprot'),flush=True)
    json.dump(res,open('results_a.json','w',encoding='utf-8'),ensure_ascii=False,indent=0)
    print(collections.Counter(r['result']['tier'] for r in res), netc.ncached('uniprot'))
