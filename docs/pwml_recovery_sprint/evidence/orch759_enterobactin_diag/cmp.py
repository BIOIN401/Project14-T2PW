import json,sys
def load(p): return json.load(open(p,encoding='utf-8'))
def ents(d):
    e=d.get('entities',{})
    out={}
    for k,v in e.items():
        if isinstance(v,list):
            out[k]=[(x.get('name'), x.get('generation_reason'), [c if isinstance(c,str) else c.get('name') for c in (x.get('components') or [])]) if isinstance(x,dict) else x for x in v]
    return out
def procs(d):
    pr=d.get('processes',{})
    res={}
    for k in ('reactions','interactions','transports'):
        v=pr.get(k) or d.get(k) or []
        rows=[]
        for x in v:
            if k=='reactions':
                rows.append((x.get('name'), x.get('inputs'), x.get('outputs'), [m.get('entity') for m in x.get('enzymes') or []], [ (m.get('entity'),m.get('role')) for m in x.get('modifiers') or []]))
            elif k=='interactions':
                rows.append((x.get('name'),x.get('entity_1'),x.get('entity_2'),x.get('relationship'), x.get('entity_1_type'), x.get('entity_2_type')))
            else:
                rows.append((x.get('name'),x.get('cargo'),[t.get('entity') for t in x.get('transporters') or []]))
        res[k]=rows
    return res
for p in sys.argv[1:]:
    d=load(p)
    print('#####',p[-80:])
    for k,v in ents(d).items():
        if k in ('proteins','protein_complexes','compounds','element_collections'):
            print(' ',k)
            for r in v: print('    ',r)
    for k,v in procs(d).items():
        print(' ',k)
        for r in v: print('    ',r)
