import json, sys, os, collections
sys.path.insert(0, 'C:/t/oa/src')
from t2pw.pipeline.gate_reports import payload_sha256
L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
c = collections.Counter(); bad = []
for x in L:
    if x['q'] is not True: continue
    p = os.path.join(x['dir'], 'final_stage3_gate_report.json')
    if not os.path.exists(p): c['no_report'] += 1; continue
    rep = json.load(open(p, encoding='utf-8'))
    want = rep.get('payload_sha256') or (rep.get('binding') or {}).get('payload_sha256')
    if not want: c['no_hash_field'] += 1; continue
    got = payload_sha256(json.load(open(os.path.join(x['dir'], 'final_mapped.json'), encoding='utf-8')))
    c['match' if got == want else 'mismatch'] += 1
    if got != want and len(bad) < 3: bad.append(x['dir'][-50:])
print(dict(c), bad)
