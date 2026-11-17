"""Per pathway: how many reactions share one drawn enzyme node (protein-complex-visualization) -> converging stacks; and duplicate left elements."""
import json, sys, collections
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c

out = {}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
    use = collections.Counter()
    for rid, v in P['rviz'].items():
        for e in v['evis']:
            use[e] += 1
    shared = {e: n for e, n in use.items() if n > 1}
    dup_members = 0
    for r in P['reactions']:
        for side in ('left', 'right'):
            cnt = collections.Counter(e[1] for e in r[side])
            dup_members += sum(v - 1 for v in cnt.values() if v > 1)
    # reactions per enzyme viz max
    out[i] = {'reactions': len(P['reactions']), 'enzyme_nodes': len(P['pcv']), 'enzyme_nodes_shared_by_multiple_reactions': len(shared),
              'max_reactions_on_one_enzyme_node': max(use.values()) if use else 0, 'duplicate_members_within_a_reaction_side': dup_members}
    print(i, out[i])
json.dump(out, open('C:/t/corr-assess/a1/shared_enzyme_nodes.json', 'w'), indent=1)
