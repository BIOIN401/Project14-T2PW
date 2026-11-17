import json, sys, collections
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c
d = json.load(open('C:/t/corr-assess/a1/per_pathway.json', encoding='utf-8'))
cx_total = cx_unknown_only = cx_with_unknown = 0
files_unknown_only = []
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
    uo = 0
    for k, names in P['complex_proteins'].items():
        cx_total += 1
        if names and all(n == 'Unknown' for n in names):
            cx_unknown_only += 1
            uo += 1
        if 'Unknown' in names:
            cx_with_unknown += 1
    if uo:
        files_unknown_only.append((i, uo, len(P['complex_proteins'])))
print('complexes', cx_total, 'only Unknown', cx_unknown_only, 'with Unknown', cx_with_unknown, 'files', len(files_unknown_only))
print(files_unknown_only)
auto = []
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    p = json.load(open('C:/t/corr-assess/base/%d/final_payload.json' % i, encoding='utf-8'))
    n = sum(1 for b in p['biological_states'] if b['name'] == '__auto_state__')
    t = json.dumps(p['processes']['transports'])
    if n or '__auto_state__' in t:
        auto.append((i, n, t.count('__auto_state__')))
print('payload files with __auto_state__ (state rows, uses in transports):', auto)
print('drawn components', sum(v['drawing']['drawn_components'] for v in d.values()), 'logical', sum(v['drawing']['logical_components'] for v in d.values()))
sh = json.load(open('C:/t/corr-assess/a1/shared_enzyme_nodes.json'))
print('files with enzyme node shared by >1 reaction', sum(1 for v in sh.values() if v['enzyme_nodes_shared_by_multiple_reactions']), 'max', max(v['max_reactions_on_one_enzyme_node'] for v in sh.values()))
print('reactions on shared nodes files', [(k, v['max_reactions_on_one_enzyme_node']) for k, v in sh.items() if v['max_reactions_on_one_enzyme_node'] >= 3])
