import json, collections
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
c=collections.Counter()
for r in rows:
    for k,v in r['errors'].items():
        if k.startswith('interaction_') and not k.startswith('interaction_vis'):
            c[(k.rsplit('_',1)[0], tuple(v) if isinstance(v,list) else v)]+=1
for k,v in c.most_common(): print(v,k)
print(rows[1]['file'])
for r in rows:
    ie=[(k,v) for k,v in r['errors'].items() if k.startswith('interaction_')]
    if ie: print(r['file'][-80:], r['expected'].get('interactions'), r['created'].get('interactions',0), len(ie)); 
