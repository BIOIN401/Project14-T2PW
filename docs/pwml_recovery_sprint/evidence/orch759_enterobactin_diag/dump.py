import json,sys
p=sys.argv[1]
d=json.load(open(p,encoding='utf-8'))
ents=d.get('entities',d)
for k in ('proteins','protein_complexes'):
    print('==',k)
    for i,e in enumerate(ents.get(k,[])):
        e2={kk:v for kk,v in e.items() if kk not in ('evidence','mapping_meta','provenance_detail')}
        print(i,json.dumps(e2)[:900])
for k in ('reactions','interactions','transports'):
    print('==',k)
    for i,e in enumerate(d.get(k,[]) or ents.get(k,[]) or []):
        e2={kk:v for kk,v in e.items() if kk not in ('evidence','mapping_meta','provenance_detail','evidence_spans')}
        print(i,json.dumps(e2)[:1200])
print(list(d.keys()))
