"""Gap quantification, part 2: coefficients, reversibility, step order, nesting, PWML population, stage flow."""
import json, os, re, sys, collections
import xml.etree.ElementTree as ET
sys.path.insert(0, 'C:/t/oa/src')
try:
    from t2pw.pwml.writer import _CURRENCY_COMPOUND_NAMES as CUR
except Exception as e:
    print('currency import failed', e); CUR = set()
from t2pw.pipeline.stage_one_ensemble import _LOCATION_PARENT, _location

L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
Q = [x for x in L if x['q'] is True]
QIR = [x for x in Q if x['ir']]
QPW = [x for x in QIR if x['pwml']]


def J(d, f):
    p = os.path.join(d, f)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None

out = {}

# 1/2/3 coefficients, reversibility words, reaction_order
coef = tot_part = 0; rev_ev = rx_n = 0; rev_ex = []
ro_runs = ro_merged = 0; ro_len_match = 0
coef_ex = []
for x in Q:
    fm = J(x['dir'], 'final_mapped.json'); mp = J(x['dir'], 'merged_payload.json')
    if 'reaction_order' in fm: ro_runs += 1
    if 'reaction_order' in mp: ro_merged += 1
    for r in fm['processes']['reactions']:
        rx_n += 1
        for side in ('inputs', 'outputs'):
            for it in r.get(side) or []:
                nm = it if isinstance(it, str) else (it.get('name') or '')
                tot_part += 1
                if re.match(r'^\s*(\d+|two|three|four|n)\s*[x\u00d7]?\s+\S', nm, re.I):
                    coef += 1
                    if len(coef_ex) < 5: coef_ex.append(nm)
        ev = (r.get('evidence') or '')
        if re.search(r'reversib|\u21cc|<->|\u2194|<=>|bidirection|interconver', ev, re.I):
            rev_ev += 1
            if len(rev_ex) < 4: rev_ex.append(ev[:140])
out['G7b_coefficients_reversibility_152'] = dict(reactions=rx_n, participants=tot_part, names_with_leading_coefficient=coef, coef_examples=coef_ex,
                                                 reactions_whose_evidence_mentions_reversibility=rev_ev, rev_examples=rev_ex)
out['G5_reaction_order_key'] = dict(final_mapped_runs_with_reaction_order_of_152=ro_runs, merged_runs_with_reaction_order_of_152=ro_merged)

# 5 step order derivable from shared compounds (IR, 62 runs)
def nrm(s): return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()
curn = {nrm(c) for c in CUR}
tot_rx = linked_rx = 0; runs_with_edges = 0; runs_forward = 0; runs_topo_ok = 0; runs_cycle = 0
n_edges = 0; n_back = 0
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    cn = {c['key']: nrm(c['name']) for c in d['entities']['compounds']}
    rx = d['processes']['reactions']
    prod = collections.defaultdict(list); subs = collections.defaultdict(list)
    for i, r in enumerate(rx):
        for m in r['left']:
            if m['entity_type'] == 'compound' and cn.get(m['entity_key']) not in curn: subs[cn[m['entity_key']]].append(i)
        for m in r['right']:
            if m['entity_type'] == 'compound' and cn.get(m['entity_key']) not in curn: prod[cn[m['entity_key']]].append(i)
    edges = set()
    for c, ps in prod.items():
        for p in ps:
            for s in subs.get(c, []):
                if s != p: edges.add((p, s))
    linked = set(a for e in edges for a in e)
    tot_rx += len(rx); linked_rx += len(linked)
    if edges: runs_with_edges += 1
    n_edges += len(edges); n_back += sum(1 for a, b in edges if a > b)
    if edges and all(a < b for a, b in edges): runs_forward += 1
out['G5_step_order_from_shared_substrates_62'] = dict(reactions=tot_rx, reactions_linked_by_a_noncurrency_compound=linked_rx, runs=len(QIR), runs_with_any_link=runs_with_edges,
                                                      links=n_edges, links_pointing_backward_in_list_order=n_back, runs_with_links_all_forward=runs_forward,
                                                      currency_set_size=len(curn))

# 4 nesting
names = collections.Counter(); per_run_loc = collections.Counter(); nest_pair_runs = 0; ofin = 0; states_tot = 0
rel_keys = collections.Counter()
for x in Q:
    fm = J(x['dir'], 'final_mapped.json')
    sl = fm['entities'].get('subcellular_locations') or []
    ls = set()
    for r in sl:
        ls.add(_location(r.get('name', '')))
        for k in r:
            if re.search(r'parent|part_of|contain|within|inside|outside|enclos', k, re.I): rel_keys['subcellular_locations.' + k] += 1
    for s in fm.get('biological_states') or []:
        states_tot += 1
        for k in s:
            if re.search(r'parent|part_of|contain|within|inside|outside|enclos', k, re.I): rel_keys['biological_states.' + k] += 1
        if re.search(r'\b(of|in|within)\b', str(s.get('name', '')), re.I): ofin += 1
        if s.get('subcellular_location'): ls.add(_location(s['subcellular_location']))
    for n in ls: names[n] += 1
    per_run_loc[min(len(ls), 8)] += 1
    if any(c in ls for n in ls for c in [_LOCATION_PARENT.get(n)] if c):
        nest_pair_runs += 1
in_table = sum(1 for n in names if n in _LOCATION_PARENT)
out['G2_nesting_152'] = dict(runs=len(Q), relation_fields_found=dict(rel_keys), distinct_location_names=len(names),
                             names_with_container_in_D109_table=in_table, runs_with_two_locations_where_table_nests_one_in_other=nest_pair_runs,
                             states_total=states_tot, state_names_with_of_in_within=ofin,
                             locations_per_run_hist=dict(sorted(per_run_loc.items())),
                             top_names=names.most_common(25))
# IR / PWML: any outside/parent field
ir_keys = collections.Counter()
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    for b in ('subcellular_locations', 'biological_states'):
        for r in d[b]:
            for k in r:
                if re.search(r'parent|part_of|contain|within|inside|outside|enclos', k, re.I): ir_keys[b + '.' + k] += 1
out['G2_nesting_ir_relation_fields_62'] = dict(ir_keys)

# 6 PWML text population (61 files)
pop = collections.defaultdict(lambda: [0, 0])  # tag -> [items, nonempty]
def popu(parent_xpath, item_tag, fields, label):
    for x in QPW:
        r = ET.parse(os.path.join(x['dir'], x['pwml'][0])).getroot()
        for it in r.iter(item_tag):
            if it.getparent if False else False: pass
    return
PV = './pathway-visualization-contexts/pathway-visualization-context/pathway-visualization'
sections = {
    'compounds/compound': ['name', 'description', 'cas', 'hmdb-id', 'kegg-id', 'chebi-id', 'pubchem-cid', 'biocyc-id', 'chemspider-id', 'drugbank-id', 'pwc-id', 'short-name',
                           'moldb-formula', 'moldb-smiles', 'moldb-inchi', 'moldb-inchikey', 'moldb-iupac', 'moldb-average-mass', 'moldb-mono-mass', 'foodb-id', 'synonyms'],
    'proteins/protein': ['name', 'uniprot-id', 'gene-name', 'locus', 'genbank-id', 'hmdbp-id', 'description', 'drugbank-id', 'synonyms'],
    'protein-complexes/protein-complex': ['name', 'pwp-id', 'species-id'],
    'species/species': ['name', 'taxonomy-id', 'classification', 'common-name'],
    'subcellular-locations/subcellular-location': ['name', 'ontology-id'],
    'reactions/reaction': ['pwr-id', 'direction', 'spontaneous'],
    'transports/transport': ['pwt-id', 'transport-type'],
    'nucleic-acids/nucleic-acid': ['name', 'chebi-id', 'genbank-id', 'gene-name', 'nucleotide-type', 'pwna-id'],
}
items_ct = collections.Counter()
for x in QPW:
    r = ET.parse(os.path.join(x['dir'], x['pwml'][0])).getroot()
    pv = r.find(PV)
    for sec, fields in sections.items():
        for it in pv.findall('./' + sec):
            items_ct[sec] += 1
            for f in fields:
                e = it.find(f)
                has = e is not None and e.get('nil') != 'true' and ((e.text or '').strip() != '' or len(e) > 0)
                pop[sec + '|' + f][0] += 1
                pop[sec + '|' + f][1] += 1 if has else 0
    pe = pv.find('pathway')
    for f in ('name', 'description', 'subject', 'species-id'):
        e = pe.find(f); pop['pathway|' + f][0] += 1; pop['pathway|' + f][1] += 1 if (e is not None and (e.text or '').strip()) else 0
    refs = pe.find('references'); pop['pathway|references(children)'][0] += 1; pop['pathway|references(children)'][1] += 1 if (refs is not None and len(refs)) else 0
    nm = pe.find('name').text
    pop['pathway_name_values|' + str(nm)][0] += 1
out['G8_pwml_field_population_61'] = {k: v for k, v in sorted(pop.items())}
out['G8_pwml_items_61'] = dict(items_ct)

# moldb value examples
x = QPW[0]
r = ET.parse(os.path.join(x['dir'], x['pwml'][0])).getroot()
c = r.find(PV).find('./compounds/compound')
out['G8_example_compound_fields'] = {e.tag: (e.text or '')[:50] + (' nil' if e.get('nil') else '') for e in c}

# 7 stage flow (62 runs; PWML-bearing 61)
flow = collections.Counter()
for x in QPW:
    mp = J(x['dir'], 'merged_payload.json'); fm = J(x['dir'], 'final_mapped.json'); ir = J(x['dir'], 'pwml_ir.json')
    r = ET.parse(os.path.join(x['dir'], x['pwml'][0])).getroot(); pv = r.find(PV)
    def ln(d, *p):
        for k in p: d = (d or {}).get(k) if isinstance(d, dict) else None
        return len(d) if isinstance(d, list) else 0
    for lab, mpn, fmn, irn, pwn in (
        ('compounds', ln(mp, 'entities', 'compounds'), ln(fm, 'entities', 'compounds'), ln(ir, 'entities', 'compounds'), len(pv.findall('./compounds/compound'))),
        ('proteins', ln(mp, 'entities', 'proteins'), ln(fm, 'entities', 'proteins'), ln(ir, 'entities', 'proteins'), len(pv.findall('./proteins/protein'))),
        ('protein_complexes', ln(mp, 'entities', 'protein_complexes'), ln(fm, 'entities', 'protein_complexes'), ln(ir, 'entities', 'protein_complexes'), len(pv.findall('./protein-complexes/protein-complex'))),
        ('nucleic_acids', ln(mp, 'entities', 'nucleic_acids'), ln(fm, 'entities', 'nucleic_acids'), ln(ir, 'entities', 'nucleic_acids'), len(pv.findall('./nucleic-acids/nucleic-acid'))),
        ('reactions', ln(mp, 'processes', 'reactions'), ln(fm, 'processes', 'reactions'), ln(ir, 'processes', 'reactions'), len(pv.findall('./reactions/reaction'))),
        ('transports', ln(mp, 'processes', 'transports'), ln(fm, 'processes', 'transports'), ln(ir, 'processes', 'transports'), len(pv.findall('./transports/transport'))),
        ('interactions', ln(mp, 'processes', 'interactions'), ln(fm, 'processes', 'interactions'), ln(ir, 'processes', 'interactions'), len([i for i in pv.findall('./interactions/interaction') if i.find('interaction-left-elements') is not None])),
        ('interaction_stubs_in_pwml', 0, 0, 0, len([i for i in pv.findall('./interactions/interaction') if i.find('interaction-left-elements') is None])),
        ('biological_states', ln(mp, 'biological_states'), ln(fm, 'biological_states'), ln(ir, 'biological_states'), len(pv.findall('./biological-states/biological-state'))),
        ('subcellular_locations', ln(mp, 'entities', 'subcellular_locations'), ln(fm, 'entities', 'subcellular_locations'), ln(ir, 'subcellular_locations'), len(pv.findall('./subcellular-locations/subcellular-location'))),
    ):
        flow[lab + '|merged'] += mpn; flow[lab + '|final_mapped'] += fmn; flow[lab + '|ir'] += irn; flow[lab + '|pwml'] += pwn
out['G9_stage_flow_61'] = dict(flow)

# per-run IR->PWML differences for reactions/compounds
diffs = []
for x in QPW:
    ir = J(x['dir'], 'pwml_ir.json')
    r = ET.parse(os.path.join(x['dir'], x['pwml'][0])).getroot(); pv = r.find(PV)
    a = len(ir['entities']['compounds']); b = len(pv.findall('./compounds/compound'))
    a2 = len(ir['processes']['reactions']); b2 = len(pv.findall('./reactions/reaction'))
    a3 = len(ir['entities']['protein_complexes']); b3 = len(pv.findall('./protein-complexes/protein-complex'))
    a4 = len(ir['entities']['proteins']); b4 = len(pv.findall('./proteins/protein'))
    if (a, a2, a3, a4) != (b, b2, b3, b4):
        diffs.append((x['dir'][-50:], (a, b), (a2, b2), (a3, b3), (a4, b4)))
out['G9b_ir_vs_pwml_count_mismatch_runs'] = diffs

json.dump(out, open('C:/t/ev-oa/payload/gaps2.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ('G8_pwml_field_population_61',)}, indent=1, default=str)[:9000])
