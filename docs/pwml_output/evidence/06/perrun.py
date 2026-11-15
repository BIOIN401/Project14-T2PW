import json, os, collections
import xml.etree.ElementTree as ET
L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
picks = {
    'OTOP1 sour taste (transport/signalling), e2e5': 'C:/t/e2e5-run/work/item-1/attempt-1/out',
    'Chlorophyll PMC5130987 (metabolic + gene regulation)': 'ORCH-759p/legs/chlorophyll-strict/2026-10-22_2301/papers/PMC5130987/strict',
    'Wnt/PCP PMC10425814 (signalling)': 'ORCH-759/legs/wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict',
    'Enterobactin PMC12096016 (metabolic + transport)': 'ORCH-759p/legs/enterobactin-strict/2026-10-22_2101/papers/PMC12096016/strict',
}
PV = './pathway-visualization-contexts/pathway-visualization-context/pathway-visualization'
rows = []
for lab, key in picks.items():
    x = [y for y in L if y['dir'].replace('\\', '/').endswith(key)][0]
    d = x['dir']
    mp = json.load(open(d + '/merged_payload.json', encoding='utf-8'))
    fm = json.load(open(d + '/final_mapped.json', encoding='utf-8'))
    ir = json.load(open(d + '/pwml_ir.json', encoding='utf-8'))
    pv = ET.parse(os.path.join(d, x['pwml'][0])).getroot().find(PV)
    def n(o, *p):
        for k in p:
            o = o.get(k) if isinstance(o, dict) else None
        return len(o) if isinstance(o, list) else 0
    full_int = len([i for i in pv.findall('./interactions/interaction') if i.find('interaction-left-elements') is not None])
    row = [lab, d[-60:]]
    for name, a, b, c, e in (
        ('compounds', n(mp, 'entities', 'compounds'), n(fm, 'entities', 'compounds'), n(ir, 'entities', 'compounds'), len(pv.findall('./compounds/compound'))),
        ('proteins', n(mp, 'entities', 'proteins'), n(fm, 'entities', 'proteins'), n(ir, 'entities', 'proteins'), len(pv.findall('./proteins/protein'))),
        ('complexes', n(mp, 'entities', 'protein_complexes'), n(fm, 'entities', 'protein_complexes'), n(ir, 'entities', 'protein_complexes'), len(pv.findall('./protein-complexes/protein-complex'))),
        ('nucleic acids', n(mp, 'entities', 'nucleic_acids'), n(fm, 'entities', 'nucleic_acids'), n(ir, 'entities', 'nucleic_acids'), len(pv.findall('./nucleic-acids/nucleic-acid'))),
        ('reactions', n(mp, 'processes', 'reactions'), n(fm, 'processes', 'reactions'), n(ir, 'processes', 'reactions'), len(pv.findall('./reactions/reaction'))),
        ('transports', n(mp, 'processes', 'transports'), n(fm, 'processes', 'transports'), n(ir, 'processes', 'transports'), len(pv.findall('./transports/transport'))),
        ('interactions', n(mp, 'processes', 'interactions'), n(fm, 'processes', 'interactions'), n(ir, 'processes', 'interactions'), full_int),
        ('bio. states', n(mp, 'biological_states'), n(fm, 'biological_states'), n(ir, 'biological_states'), len(pv.findall('./biological-states/biological-state'))),
        ('locations (subcell.)', n(mp, 'entities', 'subcellular_locations'), n(fm, 'entities', 'subcellular_locations'), n(ir, 'subcellular_locations'), len(pv.findall('./subcellular-locations/subcellular-location'))),
    ):
        row.append((name, a, b, c, e))
    rows.append(row)
md = ['| run | entity / process | merged_payload | final_mapped (frozen) | pwml_ir | PWML |', '|---|---|---|---|---|---|']
for r in rows:
    for t in r[2:]:
        md.append(f'| {r[0]} | {t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} |')
open('C:/t/ev-oa/payload/per_run_stage_counts.md', 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))
# per-run interaction spellings & transporter / unknown
for lab, key in picks.items():
    x = [y for y in L if y['dir'].replace('\\', '/').endswith(key)][0]
    ir = json.load(open(x['dir'] + '/pwml_ir.json', encoding='utf-8'))
    c = collections.Counter((i['interaction_type'], i.get('export_status')) for i in ir['processes']['interactions'])
    print(lab, dict(c))
