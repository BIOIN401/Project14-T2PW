import json,sys
for p in sys.argv[1:]:
    d=json.load(open(p,encoding='utf-8'))
    pr=d.get('processes',{})
    print('#####',p[-90:])
    for x in pr.get('reactions') or d.get('reactions') or []:
        print(' R',x.get('name'))
        for m in x.get('enzymes') or []:
            print('    enz', json.dumps({k:v for k,v in m.items() if k not in ('evidence_spans',)})[:400])
        for m in x.get('modifiers') or []:
            print('    mod', json.dumps({k:v for k,v in m.items() if k not in ('evidence_spans',)})[:300])
