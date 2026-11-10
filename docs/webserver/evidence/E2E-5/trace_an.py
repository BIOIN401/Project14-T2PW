import json, re, collections, statistics
T = 'C:/t/ev-E2E5/LEG_TRACE.jsonl'
OUT = 'C:/t/e2e5-run/work/item-1/attempt-1/out/'
raw = open(T, encoding='utf-8').read()
rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
m = [r for r in rows if r.get('kind') == 'model_attempt']
print('rows', len(rows), 'model_attempt', len(m), collections.Counter(r['kind'] for r in rows))
for f in ['started_at', 'duration_s', 'prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'provider', 'generation_id', 'response_model']:
    ok = sum(1 for r in m if r.get(f) not in (None, '', 0))
    pres = sum(1 for r in m if f in r)
    print(f'{f}: present {pres}/{len(m)} ({100*pres/len(m):.0f}%), non-empty/non-zero {ok}/{len(m)} ({100*ok/len(m):.0f}%)')
print('\nper stage')
st = collections.defaultdict(list)
for r in m: st[r['stage']].append(r)
for s, rs in st.items():
    d = [r.get('duration_s') or 0 for r in rs]
    print(f'{s}: n={len(rs)} sum={sum(d):.1f}s max={max(d):.1f}s status={dict(collections.Counter(r["status"] for r in rs))}')
print('\nten slowest')
for r in sorted(m, key=lambda r: -(r.get('duration_s') or 0))[:10]:
    print(r['stage'], r['status'], r.get('duration_s'), r.get('provider'), r.get('finish_reason'))
tot = sum(r.get('duration_s') or 0 for r in m)
je = [r for r in rows if r['kind'] == 'job_end'][0]
print(f'\nLLM time sum {tot:.1f}s ({tot/60:.1f} min); job_end elapsed {je["elapsed_seconds"]}s; unaccounted {je["elapsed_seconds"]-tot:.1f}s ({100*(je["elapsed_seconds"]-tot)/je["elapsed_seconds"]:.0f}%)')
cap = [r for r in m if r['status'] == 'call_cap']
print('\ncall_cap rows', len(cap), [(r['stage'], r.get('duration_s')) for r in cap])
print('statuses overall', collections.Counter(r['status'] for r in m))
print('\nlongest successful call per capped stage (margin to 720s)')
for s in ['preprocessor', 'Stage 1 extraction', 'Stage 2 inference', 'rag_prose_extraction']:
    ok = [r.get('duration_s') or 0 for r in st.get(s, []) if r['status'] == 'ok']
    print(s, 'n_ok', len(ok), 'max_ok', max(ok) if ok else None, 'margin', (720 - max(ok)) if ok else None)
print('stages seen:', sorted(st))
print('\nPRIVACY')
for pat in ['sk-or', r'sk-[A-Za-z0-9_-]{20,}', r'C:[\\/]', r'C:\\\\', r'Users[\\/]+Angad', 'Angad', r'Bearer ']:
    print(pat, len(re.findall(pat, raw)))
# paper sentences from evidence in final_mapped
sents = set()
for fn in ['final_mapped.json', 'merged_payload.json']:
    for s in re.findall(r'"evidence": "([^"]{60,220})', open(OUT + fn, encoding='utf-8').read()):
        sents.add(s[:60])
hits = {s: raw.count(s) for s in sents}
print('paper-evidence snippets tested', len(hits), 'with hits', sum(1 for v in hits.values() if v))
print('keys in model_attempt rows:', sorted({k for r in m for k in r}))
long_str = [(k, len(v)) for r in m for k, v in r.items() if isinstance(v, str) and len(v) > 80]
print('string fields >80 chars:', collections.Counter(k for k, _ in long_str))
