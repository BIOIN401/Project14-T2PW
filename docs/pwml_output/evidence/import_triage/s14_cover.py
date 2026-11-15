"""Group every importer error key into a root cause, then count how many of the 60 files become error-free
if a set of causes is removed (cause-set cover from the recorded errors; NOT a re-import)."""
import json, re, collections, itertools
import xml.etree.ElementTree as ET
R='C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl'
rows=[json.loads(l) for l in open(R)]
OK={'Activation','Inhibition'}
def group(key, ctx):
    p=re.sub(r'_\d+$','',key)
    if p=='bound_visualization': return 'BOUND'
    if p in ('cell_type','tissue','subcellular_location'): return 'ONTO'
    if p in ('species','protein','protein_complex_protein','protein_location','protein_complex_visualization',
             'protein_complex_protein_visualization','reaction_enzyme','reaction_enzyme_visualization'): return 'SENT'
    if p=='protein_complex':
        return 'ZEROCX' if re.search(r'_(\d+)$',key).group(1) in ctx['zero_cx'] else 'SENT'
    if p in ('reaction','reaction_visualization'): return 'SPONT'
    if p in ('transport_element','transport','transport_visualization'): return 'TSTATE'
    if p=='transport_compound_visualization': return 'TCVLOC'
    if p in ('transport_transporter','transport_transporter_visualization'): return 'TTRANS'
    if p.startswith('nucleic_acid') or p=='interaction_nucleic_acid_visualization': return 'NUCLEO'
    if p.startswith('interaction'):
        iid=re.search(r'_(\d+)$',key).group(1)
        if p in ('interaction','interaction_visualization') and iid in ctx['bad_int']: return 'STALEINT'
        return 'NUCLEO' if ctx['has_na'] else 'SENT'
    return 'OTHER'
per={}; cnt=collections.Counter(); keyscount=collections.Counter()
for r in rows:
    t=ET.parse(r['file']).getroot()
    bad_int={i.findtext('id') for i in t.iter('interaction') if i.find('interaction-type') is not None and i.findtext('interaction-type') not in OK}
    # interaction_visualization ids refer to viz ids; map viz -> interaction
    viz2int={v.findtext('id'):v.findtext('interaction-id') for v in t.iter('interaction-visualization') if v.find('interaction-id') is not None}
    zero={c.findtext('id') for c in t.iter('protein-complex') if c.find('protein_complex-proteins') is not None and not c.findall('protein_complex-proteins/protein-complex-protein')}
    has_na=any(k.startswith('nucleic_acid_') for k in r['errors'])
    ctx={'bad_int':bad_int,'zero_cx':zero,'has_na':has_na}
    gs=set()
    for k in r['errors']:
        m=re.fullmatch(r'interaction_visualization_(\d+)',k)
        if m and viz2int.get(m.group(1)) in bad_int:
            g='STALEINT'
        else: g=group(k,ctx)
        gs.add(g); keyscount[g]+=1
    per[r['file']]=gs
    for g in gs: cnt[g]+=1
print('files per root cause:',dict(cnt.most_common()))
print('error messages (keys) per root cause:',dict(keyscount.most_common()))
allg=sorted(cnt)
def clean(fixed): return sum(1 for gs in per.values() if gs<=set(fixed))
order=[]; fixed=set()
while True:
    best=max((g for g in allg if g not in fixed), key=lambda g:(clean(fixed|{g}), cnt[g]), default=None)
    if best is None: break
    fixed.add(best); order.append(best); print(f'+ {best:9s} -> files with zero errors: {clean(fixed)}')
json.dump({f:sorted(g) for f,g in per.items()},open('s14_cover_per_file.json','w'),indent=1)
