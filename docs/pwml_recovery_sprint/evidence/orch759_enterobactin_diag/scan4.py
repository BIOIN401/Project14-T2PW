import json,os,collections
R='C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/'
out=collections.Counter()
for p in open('fm_list.txt',encoding='utf-8').read().splitlines():
    try: s=open(R+p,encoding='utf-8').read()
    except Exception: continue
    if 'complex_named_source_entity_wrapper' not in s: continue
    d=json.loads(s); e=d.get('entities',{})
    for c in e.get('protein_complexes',[]):
        if c.get('generation_reason')=='complex_named_source_entity_wrapper' and [x.get('name') for x in c.get('components')]==['enterobactin synthase']:
            dd=os.path.dirname(R+p)
            row=next((x for x in e.get('proteins',[]) if x.get('name')=='enterobactin synthase'),{})
            up=(row.get('mapped_ids') or {}).get('uniprot')
            dup=[x.get('name') for x in e.get('proteins',[]) if x is not row and (x.get('mapped_ids') or {}).get('uniprot')==up and up]
            g=os.path.join(dd,'final_stage3_gate_report.json')
            st=(json.load(open(g,encoding='utf-8')).get('ok') if os.path.exists(g) else 'nogate')
            pw=any(f.endswith('.pwml') for f in os.listdir(dd))
            out[(p.split('/papers/')[1].split('/')[0], p.rstrip('/').split('/')[-2], up, tuple(dup), st, pw)]+=1
for k,v in sorted(out.items(), key=str): print(v,k)
