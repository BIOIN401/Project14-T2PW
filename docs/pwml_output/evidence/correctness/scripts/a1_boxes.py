"""Node-in-compartment-box check for the PWML (v2.1): which drawn nodes lie outside every box, and whether a box holds nodes of >1 state."""
import json, sys, collections
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c

out = {}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
    boxes = []
    for m in P['membranes']:
        if not m:
            continue
        lab = [l['text'] for l in P['labels'] if m['x0'] <= l['x'] <= m['x1'] and m['y0'] <= l['y'] <= m['y1']]
        boxes.append({**m, 'label': lab[0] if lab else None, 'states': set()})
    nodes = [('c', x['x'] + x['w'] / 2, x['y'] + x['h'] / 2, x['state']) for x in P['cloc'].values() if x['hidden'] != 'true'] + \
            [('p', x['x'] + x['w'] / 2, x['y'] + x['h'] / 2, x['state']) for x in P['ploc'].values() if x['hidden'] != 'true']
    outside = 0
    for kind, x, y, st in nodes:
        hit = [b for b in boxes if b['x0'] <= x <= b['x1'] and b['y0'] <= y <= b['y1']]
        if not hit:
            outside += 1
        for b in hit:
            b['states'].add(P['states'].get(st, {}).get('name'))
    mixed = [(b['label'], sorted(x for x in b['states'] if x)) for b in boxes if len(b['states']) > 1]
    states_in_no_box = sorted({P['states'].get(st, {}).get('name') for kind, x, y, st in nodes
                               if not any(b['x0'] <= x <= b['x1'] and b['y0'] <= y <= b['y1'] for b in boxes)})
    out[i] = {'nodes': len(nodes), 'nodes_outside_all_boxes': outside, 'mixed_state_boxes': mixed, 'states_with_unboxed_nodes': states_in_no_box,
              'boxes': [(b['label'], sorted(x for x in b['states'] if x)) for b in boxes]}
    print(i, 'nodes', len(nodes), 'outside', outside, 'mixed', mixed, 'unboxed states', states_in_no_box)
json.dump(out, open('C:/t/corr-assess/a1/box_check.json', 'w', encoding='utf-8'), indent=1)
