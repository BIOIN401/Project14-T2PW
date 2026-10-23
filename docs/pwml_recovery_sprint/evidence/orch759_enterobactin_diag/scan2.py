import json,os,collections
R='C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/'
out=collections.Counter()
for p in open('fm_list.txt',encoding='utf-8').read().splitlines():
    try: s=open(R+p,encoding='utf-8').read()
    except Exception: continue
    if 'complex_named_source_entity_wrapper' not in s: continue
    d=json.loads(s); e=d.get('entities',{})
    pn={x.get('name') for x in e.get('proteins',[])}
    allcomp=set()
    for c in e.get('protein_complexes',[]):
        for x in c.get('components') or []: allcomp.add(x.get('name') if isinstance(x,dict) else x)
    for c in e.get('protein_complexes',[]):
        if c.get('generation_reason')!='complex_named_source_entity_wrapper': continue
        sub=c['name'][:-8]
        if sub in pn and sub not in allcomp:
            g=os.path.join(os.path.dirname(R+p),'final_stage3_gate_report.json')
            st='nogate'
            if os.path.exists(g):
                gd=json.load(open(g,encoding='utf-8')); st=('ok' if gd.get('ok') else 'FAIL')+' deg0=%s'%gd.get('connectivity',{}).get('proteins_degree0')
            mode=p.rstrip('/').split('/')[-2]
            out[(p.split('/papers/')[1].split('/')[0],mode,tuple(sorted(x.get('name') if isinstance(x,dict) else x for x in c.get('components') or [])),st)]+=1
for k,v in sorted(out.items()): print(v,k)
