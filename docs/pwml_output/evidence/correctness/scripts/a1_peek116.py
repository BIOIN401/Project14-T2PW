import json, sys
i = sys.argv[1]
d = json.load(open('C:/t/corr-assess/base/%s/final_payload.json' % i, encoding='utf-8'))
for r in d['processes']['reactions']:
    print(r.get('locked_reaction_id'), r.get('name'), '|', r['inputs'], '->', r['outputs'], [e['entity'] for e in r['enzymes']], r.get('biological_state'), r.get('scope_membership'))
print(d['reaction_order'])
print(json.dumps(d['quarantined_locked_reactions'])[:800])
