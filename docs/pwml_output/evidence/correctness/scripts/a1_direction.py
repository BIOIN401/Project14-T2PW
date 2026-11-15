import json, collections
d = json.load(open('C:/t/corr-assess/a1/per_pathway.json', encoding='utf-8'))
dv = collections.Counter()
rev = []
for k, v in d.items():
    for dd, n in v['reactions']['direction_values'].items():
        dv[dd] += n
    p = json.load(open('C:/t/corr-assess/base/%s/final_payload.json' % k, encoding='utf-8'))
    keys = collections.Counter()
    nrev = 0
    for r in p['processes']['reactions']:
        keys.update(r.keys())
        if 'reversib' in (r.get('evidence') or '').lower() or 'reversib' in (r.get('name') or '').lower() or '<->' in (r.get('evidence') or '') or '⇌' in (r.get('evidence') or ''):
            nrev += 1
    if nrev:
        rev.append((int(k), nrev))
    if 'direction' in keys or 'reversible' in keys:
        print('payload has direction key', k)
print('PWML direction values over all reactions', dict(dv))
print('payload reactions whose evidence/name says reversible:', rev, sum(n for _, n in rev))
