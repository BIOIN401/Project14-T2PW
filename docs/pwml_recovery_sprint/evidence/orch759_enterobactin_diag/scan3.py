import json,os
R='C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/'
for p in open('fm_list.txt',encoding='utf-8').read().splitlines():
    try: s=open(R+p,encoding='utf-8').read()
    except Exception: continue
    if 'complex_named_source_entity_wrapper' not in s: continue
    d=json.loads(s); e=d.get('entities',{})
    for c in e.get('protein_complexes',[]):
        if c.get('generation_reason')=='complex_named_source_entity_wrapper' and [x.get('name') for x in c.get('components')]==['EntE']:
            dd=os.path.dirname(R+p); print(p.split('/papers/')[0], sorted(f for f in os.listdir(dd) if f.endswith('.pwml') or f=='RESULT.txt'))
