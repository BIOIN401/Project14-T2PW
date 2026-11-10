import json,re,collections,pickle
src=open('units.py').read().split("print('units'")[0]
exec(src)
runs_by={r['path']:r for r in runs}
def uclass(u):
    rows=u['rows']
    if any(r['bucket']=='proteins' and pstat(r)=='mapped' for r in rows): return 'mapped'
    if any(pstat(r)=='mapped' for r in rows): return 'mapped'
    if any(pstat(r)=='placeholder' for r in rows): return 'placeholder'
    return 'unmapped'
out=[]
for k,u in U.items():
    cls=uclass(u)
    rows=u['rows']
    # primary reason
    reasons=collections.Counter()
    for r in rows:
        s=pstat(r)
        if s=='mapped': continue
        if r['bucket']=='protein_complexes' and r['status']=='fallback': reasons['placeholder:'+str(r['rmfr'])]+=1
        else: reasons[str(r['status'])+'/'+str(r['issue'])+'/'+str(r['failed'])]+=1
    if u['states'].get('quarantined_removed') and not rows: reasons['quarantine_only(meta lost)']+=1
    orgs=collections.Counter(); 
    for r in rows:
        o=r['tgt'] or r['organism'] if (r['bucket']!='protein_complexes' or r['status']!='fallback') else (r['tgt'] or r['organism'])
        if r['bucket']=='compounds': o=r['main_org']
        orgs[o]+=1
    runs_here=[runs_by[p] for p in u['runs']]
    for rr in runs_here: orgs[rr['main_org']]+=0
    mo=collections.Counter(rr['main_org'] for rr in runs_here).most_common(1)[0][0]
    ex=[r for r in rows if pstat(r)!='mapped']
    exr=ex[0] if ex else (rows[0] if rows else None)
    out.append(dict(paper=u['paper'],kind=u['kind'],cls=cls,name=u['names'].most_common(1)[0][0],all_names=list(u['names'])[:4],
        n_runs=len(u['runs']),organism=mo,reasons=dict(reasons),
        buckets=sorted(u['buckets']),states=dict(u['states']),
        gene=(exr or {}).get('gene'),ec=(exr or {}).get('ec'),cls_row=(exr or {}).get('cls'),syn=(exr or {}).get('syn'),
        amb_acc=(exr or {}).get('amb_acc'),amb_rej=(exr or {}).get('amb_rej'),raw=(exr or {}).get('raw_name'),
        lit=(exr or {}).get('lit_alias'),jc=[c for r in ex for c in (r.get('jc') or [])][:8],ngate=(exr or {}).get('ngate'),vreason=(exr or {}).get('vreason'),rmfr=[r.get('rmfr') for r in ex if r.get('rmfr')][:1],rej=(exr or {}).get('rejected_ids'),failed=(exr or {}).get('failed'),fen=(exr or {}).get('fen'),query=(exr or {}).get('query')))
json.dump(out,open('units.json','w',encoding='utf-8'),ensure_ascii=False)
for kind in ('protein','compound'):
    print(kind,collections.Counter(o['cls'] for o in out if o['kind']==kind))
# unit-level: papers
print('papers with any unmapped/placeholder protein unit:',len({o['paper'] for o in out if o['kind']=='protein' and o['cls']!='mapped'}))
print('papers with any unmapped compound:',len({o['paper'] for o in out if o['kind']=='compound' and o['cls']!='mapped'}))
