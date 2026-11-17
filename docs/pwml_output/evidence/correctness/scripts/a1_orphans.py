"""Drawn enzyme/protein-complex nodes that no process visualization references (orphans: drawn in the picture with no edge) and what they are."""
import json, sys, collections
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/corr-assess/scripts')
import a1_check as c

out = {}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    P = c.parse_pwml('%s/%d.pwml' % (c.PWML, i))
    pv = ET.parse('%s/%d.pwml' % (c.PWML, i)).getroot().find('.//pathway-visualization')
    ref = set()
    for v in P['rviz'].values():
        ref.update(v['evis'])
    for v in P['tviz'].values():
        ref.update(v['tvis'])
    for x in pv.iter('interaction-protein-complex-visualization'):
        ref.add(c.txt(x, 'protein-complex-visualization-id'))
    for x in pv.iter('protein-complex-protein-visualization'):
        pass
    orphans = [k for k in P['pcv'] if k not in ref]
    names = collections.Counter(P['names'].get(('ProteinComplex', P['pcv'][k]['complex']), '?') for k in orphans)
    # enzymes in the payload that never appear on any reaction/transport (extra complexes)
    out[i] = {'complex_nodes': len(P['pcv']), 'orphan_complex_nodes': len(orphans), 'orphan_names': dict(names.most_common(6))}
    print(i, out[i])
json.dump(out, open('C:/t/corr-assess/a1/orphan_enzyme_nodes.json', 'w', encoding='utf-8'), indent=1)
