import json, os, collections
rows = [json.loads(l) for l in open('C:/t/ev-dedup/out/runs.jsonl', encoding='utf-8')]
print(len(rows), collections.Counter((r['root'], r['quarantine_ok']) for r in rows))
R = 'C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/'
BS = chr(92)
c = collections.Counter(); out = []
for r in rows:
    p = r['path'].replace(BS, '/')
    d = p if p[1:2] == ':' else R + p
    d = os.path.dirname(d)
    fs = os.listdir(d) if os.path.isdir(d) else []
    has_ir = 'pwml_ir.json' in fs
    pw = [f for f in fs if f.endswith('.pwml')]
    c[(r['quarantine_ok'], has_ir, bool(pw), 'merged_payload.json' in fs)] += 1
    out.append(dict(dir=d, q=r['quarantine_ok'], ir=has_ir, pwml=pw, ne=r['n_entities'], nr=r['n_reactions'], nt=r['n_transports'], ni=r['n_interactions'], root=r['root']))
print(c)
json.dump(out, open('C:/t/ev-oa/payload/runlist.json', 'w'), indent=0)
