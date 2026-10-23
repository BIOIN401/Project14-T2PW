import json,sys
p=sys.argv[1]
d=json.load(open(p,encoding='utf-8'))
pr=d.get('processes',{})
print(type(pr), list(pr.keys()) if isinstance(pr,dict) else len(pr))
drop=('evidence','mapping_meta','provenance_detail','evidence_spans','provenance_lineage','source_text','evidence_text')
for k,v in (pr.items() if isinstance(pr,dict) else []):
    print('==',k, len(v) if isinstance(v,list) else '')
    if isinstance(v,list):
        for i,e in enumerate(v):
            e2={kk:vv for kk,vv in e.items() if kk not in drop} if isinstance(e,dict) else e
            print(i,json.dumps(e2)[:1500])
