import json,sys,collections
seen=collections.Counter()
for p in open('fm_list.txt',encoding='utf-8').read().splitlines():
    try: s=open('C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/'+p,encoding='utf-8').read()
    except Exception: continue
    if 'complex_named_source_entity_wrapper' not in s: continue
    d=json.loads(s); e=d.get('entities',{})
    prots=e.get('proteins',[]); pn={x.get('name') for x in prots}
    for c in e.get('protein_complexes',[]):
        if c.get('generation_reason')!='complex_named_source_entity_wrapper': continue
        comps=[x.get('name') if isinstance(x,dict) else x for x in c.get('components') or []]
        sub=c['name'][:-len(' complex')] if c['name'].lower().endswith(' complex') else None
        key=(p.split('/papers/')[1].split('/')[0], c['name'], tuple(comps), sub in pn)
        seen[key]+=1
for k,v in sorted(seen.items()): print(v,k)
