import json,re,collections
E=[json.loads(l) for l in open('entities.jsonl',encoding='utf-8')]
runs=json.load(open('runs.json')); RUN={r['path']:r for r in runs}
Q=json.load(open('quar_rows.json'))
def norm(n):
    n=(n or '').lower()
    n=re.sub(r'\bcomplex(es)?\b','',n)
    return re.sub(r'[^a-z0-9α-ωβ]+','',n)
def pstat(e):
    ids=e['ids']; b=e['bucket']
    if b=='proteins':
        if e['name']=='Unknown': return None
        if e['step']=='pathbank_unknown_protein_fallback' or (e['idstatus'] or '')=='placeholder': return 'placeholder'
        if any(k in ids for k in ('uniprot','uniprot_id','drugbank','drugbank_id')): return 'mapped'
        return 'unmapped'
    if b=='protein_complexes':
        if e['status']=='fallback': return 'placeholder'
        if e['status']=='matched': return 'mapped'
        if e['step']=='novel_complex_from_resolved_components': return 'mapped'
        return 'unmapped'
    if b=='compounds': return 'mapped' if ids else 'unmapped'
    return None
U={}
for e in E:
    s=pstat(e)
    if s is None: continue
    kind='compound' if e['bucket']=='compounds' else 'protein'
    k=(e['paper'],kind,norm(e['name']))
    u=U.setdefault(k,dict(paper=e['paper'],kind=kind,names=collections.Counter(),runs=set(),states=collections.Counter(),rows=[],buckets=set()))
    u['names'][e['name']]+=1; u['runs'].add(e['path']); u['states'][s]+=1; u['rows'].append(e); u['buckets'].add(e['bucket'])
# quarantine-only participants
qonly=0
for o in Q:
    if o['reason']=='protein_missing_external_identity' and o['in_final'] is None:
        k=(RUN[o['path']]['paper'],'protein',norm(o['participant']))
        u=U.setdefault(k,dict(paper=RUN[o['path']]['paper'],kind='protein',names=collections.Counter(),runs=set(),states=collections.Counter(),rows=[],buckets=set()))
        if k not in U or not u['rows']: qonly+=1
        u['names'][o['participant']]+=1; u['runs'].add(o['path']); u['states']['quarantined_removed']+=1; u['buckets'].add('quarantine_only')
print('units',len(U),collections.Counter(k[1] for k in U))
def unitclass(u):
    st=u['states']
    if st.get('mapped') and not (st.get('unmapped') or st.get('placeholder') or st.get('quarantined_removed')): return 'mapped'
    if st.get('mapped'): return 'mixed'
    if st.get('unmapped') or st.get('quarantined_removed'):
        return 'unmapped'
    if st.get('placeholder'): return 'placeholder'
for kind in ('protein','compound'):
    c=collections.Counter(unitclass(u) for k,u in U.items() if k[1]==kind)
    print(kind,c)
import pickle
pickle.dump(U,open('units.pkl','wb'))
