import json, sys, collections
for i in sys.argv[1:]:
    d = json.load(open('C:/t/corr-assess/base/%s/final_payload.json' % i, encoding='utf-8'))
    q = json.load(open('C:/t/corr-assess/base/%s/outputs__quarantine_report.json' % i, encoding='utf-8'))
    ints = d['processes']['interactions']
    print('==', i, 'payload interactions', len(ints), 'relationships', dict(collections.Counter(x.get('relationship') for x in ints)))
    for x in ints[:4]:
        print('   ', x.get('entity_1'), '|', x.get('relationship'), '|', x.get('entity_2'), '|', x.get('biological_state'))
    qi = [x for x in q.get('quarantined', []) if x.get('process_kind') == 'interaction']
    print('  quarantined interactions in report:', len(qi), dict(collections.Counter(x.get('reason') for x in qi)), dict(collections.Counter(x.get('state') for x in qi)))
    print('  counts', q.get('counts'))
    adm = [x for x in q.get('admissions', []) if x.get('process_kind') == 'interaction']
    print('  admissions of interactions', len(adm), dict(collections.Counter((x.get('state'), x.get('reason')) for x in adm)))
