import json, collections, re
rows = [json.loads(l) for l in open('C:/t/corr-assess/a1/import_results.jsonl', encoding='utf-8') if l.strip()]
print('rows', len(rows), 'import_ok', sum(r['import_ok'] for r in rows))
ex = collections.Counter()
causes = collections.Counter()
files_by_cause = collections.defaultdict(set)
for r in rows:
    e = r.get('exports') or {}
    for f in ('sbml', 'sbgn', 'pwml'):
        ex[f] += 1 if (e.get(f) or {}).get('ok') else 0
    for k, v in (r.get('errors') or {}).items():
        base = re.sub(r'_\d+$', '', k)
        msg = '; '.join(v) if isinstance(v, list) else str(v)
        c = '%s: %s' % (base, msg[:110])
        causes[c] += 1
        files_by_cause[c].add(r['idx'])
print('exports ok', dict(ex), 'screenshots ok', sum(1 for r in rows if (r.get('screenshot') or {}).get('ok')))
for r in rows:
    cm, exc = r.get('created_main') or {}, r.get('expected') or {}
    print(r['idx'], 'ok' if r['import_ok'] else 'FAIL', 'errs', len(r.get('errors') or {}), 'react created/expected', cm.get('reactions'), exc.get('reactions'),
          'compounds', cm.get('compounds'), exc.get('compounds'), 'proteins', cm.get('proteins'), exc.get('proteins'), 'cx', cm.get('protein_complexes'), exc.get('protein_complexes'),
          'transports', cm.get('transports'), exc.get('transports'), 'clocs', cm.get('compound_locations'), exc.get('compound_locations'), 'pexc', r.get('parser_exception'))
print()
for c, n in causes.most_common():
    print(n, sorted(files_by_cause[c]), c)
