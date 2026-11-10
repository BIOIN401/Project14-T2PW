"""Gap quantification, part 1: interactions, transporters, auto states, complexes, stoichiometry, direction."""
import json, os, re, sys, collections
sys.path.insert(0, 'C:/t/oa/src')
from t2pw.pwml.interaction_types import PWML_INTERACTION_TYPES, _interaction_type_key

L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
Q = [x for x in L if x['q'] is True]
QIR = [x for x in Q if x['ir']]
NONQ = [x for x in L if x['q'] is not True]


def J(d, f):
    p = os.path.join(d, f)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


out = {}

# ---------- G1 interaction spellings ----------
def spell_counts(runs, fname, getter):
    exact = collections.Counter(); key = collections.Counter(); n_int = 0; nr = 0
    for x in runs:
        d = J(x['dir'], fname)
        if d is None:
            continue
        nr += 1
        for it in getter(d):
            n_int += 1
            raw = getter_raw(it)
            exact[raw] += 1
            key[_interaction_type_key(raw)] += 1
    return dict(n_runs=nr, n_interactions=n_int, exact=exact, key=key)


def getter_raw(it):
    return str(it.get('relationship') or it.get('interaction_type') or it.get('raw_interaction_type') or '')


def ints_payload(d):
    return (d.get('processes') or {}).get('interactions') or []


def ints_ir(d):
    r = []
    for it in (d.get('processes') or {}).get('interactions') or []:
        if it.get('export_status') == 'withheld':
            it = dict(it); it['interaction_type'] = (it.get('withheld') or {}).get('raw_interaction_type') or it.get('interaction_type')
        r.append(it)
    return r

res = {}
for name, runs, fname, g in (('merged_152', Q, 'merged_payload.json', ints_payload),
                             ('final_mapped_152', Q, 'final_mapped.json', ints_payload),
                             ('final_mapped_nonq_40', NONQ, 'final_mapped.json', ints_payload),
                             ('ir_62', QIR, 'pwml_ir.json', ints_ir)):
    r = spell_counts(runs, fname, g)
    mapped = sum(c for k, c in r['key'].items() if k in PWML_INTERACTION_TYPES)
    r['mapped_to_activation_inhibition'] = mapped
    r['exact_distinct'] = len(r['exact']); r['key_distinct'] = len(r['key'])
    r['exact'] = r['exact'].most_common(60); r['key'] = r['key'].most_common(60)
    res[name] = r
out['G1_interaction_spellings'] = res

# IR export status & withheld reasons
st = collections.Counter(); why = collections.Counter(); n = 0
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    for it in (d['processes'].get('interactions') or []):
        n += 1; st[it.get('export_status')] += 1
        if it.get('export_status') == 'withheld':
            why[(it.get('withheld') or {}).get('reason')] += 1
out['G1b_ir_interaction_status_62'] = dict(n=n, status=dict(st), withheld_reasons=why.most_common(12))

# ---------- G3 transporter states ----------
tr_total = tr_with = tr_default = tr_no_state_field = 0
state_loc_kind = collections.Counter(); ex = []
tr_proc_total = 0; tr_proc_no_transporter = 0
cargo_state_loc = collections.Counter()
payload_tr_with_state = 0; payload_tr_n = 0
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    fm = J(x['dir'], 'final_mapped.json')
    sloc = {l['key']: l['name'] for l in d.get('subcellular_locations', [])}
    st2loc = {s['key']: sloc.get(s.get('subcellular_location_key'), '?') for s in d['biological_states']}
    st2name = {s['key']: s['name'] for s in d['biological_states']}
    first = d['biological_states'][0]['key'] if d['biological_states'] else None
    for t in d['processes'].get('transports', []):
        tr_proc_total += 1
        if not t['transporters']:
            tr_proc_no_transporter += 1
        for tt in t['transporters']:
            tr_with += 1
            bs = tt.get('biological_state_key')
            if bs == first:
                tr_default += 1
            nm = st2loc.get(bs, '?').lower()
            kind = 'membrane' if ('membrane' in nm or 'leaflet' in nm or 'vesicle' in nm or 'envelope' in nm) else ('cell/cytosol' if nm in ('cell', 'cytosol', 'cytoplasm', 'whole cell') else 'other')
            state_loc_kind[kind] += 1
            if len(ex) < 6:
                ex.append((x['dir'][-40:], t['name'], st2name.get(bs), nm))
        for el in t['transport_elements']:
            for side in ('left_biological_state_key', 'right_biological_state_key'):
                nm = st2loc.get(el[side], '?').lower()
                k = 'membrane' if ('membrane' in nm or 'leaflet' in nm) else 'other'
                cargo_state_loc[k] += 1
    for t in (fm.get('processes') or {}).get('transports', []):
        payload_tr_n += 1
        for a in t.get('transporters') or []:
            if isinstance(a, dict) and (a.get('biological_state') or a.get('state')):
                payload_tr_with_state += 1
out['G3_transporters'] = dict(ir_transport_processes_62=tr_proc_total, without_transporter=tr_proc_no_transporter,
                              ir_transporter_members=tr_with, transporter_state_equals_first_state=tr_default,
                              transporter_state_location_kind=dict(state_loc_kind), cargo_end_state_location_kind=dict(cargo_state_loc),
                              payload_transports_62=payload_tr_n, payload_transporters_with_state_field=payload_tr_with_state,
                              examples=ex)

# ---------- G6 auto states / default states ----------
auto_fm = auto_ir = 0; states_fm = states_ir = 0; runs_auto = 0
auto_use = 0
for x in Q:
    fm = J(x['dir'], 'final_mapped.json')
    sts = fm.get('biological_states') or []
    states_fm += len(sts)
    a = [s for s in sts if str(s.get('name', '')).startswith('__auto')]
    auto_fm += len(a)
    if a:
        runs_auto += 1
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    states_ir += len(d['biological_states'])
    auto_ir += sum(1 for s in d['biological_states'] if str(s['name']).startswith('__auto'))
out['G6_auto_states'] = dict(final_mapped_states_152=states_fm, auto_states_152=auto_fm, runs_with_auto_152=runs_auto, ir_states_62=states_ir, ir_auto_62=auto_ir)

# ---------- G4 complexes / Unknown members ----------
def is_unknown_member(c):
    if isinstance(c, str):
        return c.strip().lower() == 'unknown'
    nm = str(c.get('name') or '').strip().lower()
    up = str(c.get('uniprot') or (c.get('mapped_ids') or {}).get('uniprot') or '').strip().lower()
    pid = str(c.get('pathbank_protein_id') or (c.get('mapped_ids') or {}).get('pathbank_protein_id') or '')
    return nm == 'unknown' or up == 'unknown' or pid == '9659'

cx = collections.Counter(); gen_reasons = collections.Counter()
sizes = collections.Counter()
for x in Q:
    fm = J(x['dir'], 'final_mapped.json')
    for c in (fm.get('entities') or {}).get('protein_complexes') or []:
        cx['complexes'] += 1
        comps = c.get('components') or []
        sizes[min(len(comps), 6)] += 1
        gen = bool(c.get('generated'))
        if gen:
            cx['generated'] += 1; gen_reasons[str(c.get('generation_reason'))[:90]] += 1
        un = [is_unknown_member(k) for k in comps]
        if comps and all(un):
            cx['all_members_unknown'] += 1
            if gen:
                cx['generated_all_unknown'] += 1
        elif any(un):
            cx['some_unknown'] += 1
        if not comps:
            cx['no_members'] += 1
        if len(comps) == 1 and gen:
            cx['generated_single_member_wrapper'] += 1
out['G4_complexes_152'] = dict(cx)
out['G4_complex_size_hist_152'] = dict(sizes)
out['G4_generation_reasons_152'] = gen_reasons.most_common(8)

cx2 = collections.Counter()
rx_enz_unknown = rx_total = rx_with_enz = 0
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    prot = {p['key']: p for p in d['entities']['proteins']}
    cmap = {c['key']: c for c in d['entities']['protein_complexes']}
    def unk(pk):
        p = prot.get(pk)
        return bool(p) and (str(p.get('name')).lower() == 'unknown' or str(p.get('uniprot')).lower() == 'unknown' or str(p.get('pathbank_protein_id') or p.get('pathwhiz_id')) == '9659')
    unk_cx = set()
    for c in cmap.values():
        cx2['ir_complexes'] += 1
        comps = c.get('components') or []
        if comps and all(unk(k.get('protein_key')) for k in comps):
            cx2['ir_complex_all_members_unknown'] += 1; unk_cx.add(c['key'])
        if c.get('generated'):
            cx2['ir_generated'] += 1
    for r in d['processes']['reactions']:
        rx_total += 1
        if r['enzymes']:
            rx_with_enz += 1
            if all(e['entity_key'] in unk_cx for e in r['enzymes'] if e['entity_type'] == 'protein_complex') and any(e['entity_type'] == 'protein_complex' for e in r['enzymes']):
                rx_enz_unknown += 1
    for p in prot.values():
        cx2['ir_proteins'] += 1
        if unk(p['key']):
            cx2['ir_protein_unknown_sentinel'] += 1
out['G4_ir_62'] = dict(cx2, reactions=rx_total, reactions_with_enzyme=rx_with_enz, reactions_whose_enzyme_complexes_are_all_unknown_backed=rx_enz_unknown)

# ---------- stoichiometry / direction / spontaneous ----------
stc = collections.Counter(); dr = collections.Counter(); sp = collections.Counter(); pay_dict = collections.Counter()
for x in QIR:
    d = J(x['dir'], 'pwml_ir.json')
    fm = J(x['dir'], 'final_mapped.json')
    for r in d['processes']['reactions']:
        dr[r['direction']] += 1; sp[r['spontaneous']] += 1
        for m in r['left'] + r['right']:
            stc[m['stoichiometry']] += 1
    for r in fm['processes']['reactions']:
        for side in ('inputs', 'outputs'):
            for it in r.get(side) or []:
                pay_dict[type(it).__name__] += 1
        sp[('payload_spont', r.get('spontaneous'))] += 1
        pay_dict[('payload_direction_present', 'direction' in r)] += 1
        pay_dict[('payload_stoich_present', any(isinstance(i, dict) and ('stoichiometry' in i or 'coefficient' in i) for i in (r.get('inputs') or []) + (r.get('outputs') or [])))] += 1
out['G7_reaction_defaults_62'] = dict(ir_stoich={str(k): v for k, v in stc.items()}, ir_direction=dict(dr), spontaneous={str(k): v for k, v in sp.items()},
                                      payload_participant_types={str(k): v for k, v in pay_dict.items()})

json.dump(out, open('C:/t/ev-oa/payload/gaps1.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'G1_interaction_spellings'}, indent=1, default=str)[:7000])
for k, v in out['G1_interaction_spellings'].items():
    print(k, {a: b for a, b in v.items() if a not in ('exact', 'key')})
