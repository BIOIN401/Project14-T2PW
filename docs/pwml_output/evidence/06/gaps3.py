"""Gap part 3: how many reactions are fully identifiable (KEGG/ChEBI) and which interaction words fall in which family."""
import json, os, re, sys, collections
sys.path.insert(0, 'C:/t/oa/src')
from t2pw.pwml.interaction_types import _interaction_type_key, PWML_INTERACTION_TYPES
L = json.load(open('C:/t/ev-oa/payload/runlist.json'))
Q = [x for x in L if x['q'] is True]


def nrm(s): return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

tot = all_kegg = all_chebi = all_any = 0; part_unres = 0
rx_unmatched_names = 0
for x in Q:
    d = json.load(open(os.path.join(x['dir'], 'final_mapped.json'), encoding='utf-8'))
    comp = {}
    for c in d['entities'].get('compounds', []):
        m = c.get('mapped_ids') or {}
        rec = dict(kegg=bool(m.get('kegg') or c.get('kegg_id')), chebi=bool(m.get('chebi') or c.get('chebi_id')),
                   anyid=any(m.get(k) or c.get(k + '_id') for k in ('chebi', 'kegg', 'hmdb', 'pubchem')))
        comp[nrm(c['name'])] = rec
        for s in c.get('synonyms') or []:
            comp.setdefault(nrm(s), rec)
    others = {nrm(p['name']) for b in ('proteins', 'protein_complexes', 'nucleic_acids') for p in d['entities'].get(b, [])}
    for r in d['processes']['reactions']:
        tot += 1
        names = [nrm(i if isinstance(i, str) else i.get('name', '')) for i in (r.get('inputs') or []) + (r.get('outputs') or [])]
        recs = []
        ok = True
        for n in names:
            if n in comp: recs.append(comp[n])
            elif n in others: continue
            else: ok = False
        if not ok:
            rx_unmatched_names += 1
        if recs and all(c['kegg'] for c in recs) and ok: all_kegg += 1
        if recs and all(c['chebi'] for c in recs) and ok: all_chebi += 1
        if recs and all(c['anyid'] for c in recs) and ok: all_any += 1

fam = collections.Counter(); famkeys = collections.defaultdict(collections.Counter)
FAM = {
    'signed (PWML-exportable verb)': lambda k: k in PWML_INTERACTION_TYPES,
    'binding / association (unsigned)': lambda k: re.search(r'bind|associat|interact|co-?local|complex|component|same_as|requires', k),
    'catalysis / substrate / cofactor / transport / degradation': lambda k: re.search(r'catal|substrate|cofactor|transport|degrad|modif|phospho|hydrol', k),
    'regulation, no sign (regulates, modulates, regulation)': lambda k: re.search(r'regulat|modulat', k),
}
n = 0
for x in Q:
    d = json.load(open(os.path.join(x['dir'], 'final_mapped.json'), encoding='utf-8'))
    for i in d['processes']['interactions']:
        n += 1
        k = _interaction_type_key(i.get('relationship'))
        for lab, f in FAM.items():
            if f(k):
                fam[lab] += 1; famkeys[lab][k] += 1; break
        else:
            fam['other'] += 1; famkeys['other'][k] += 1
out = dict(reactions_152=tot, reactions_all_participants_have_kegg=all_kegg, reactions_all_participants_have_chebi=all_chebi,
           reactions_all_participants_have_any_of_chebi_kegg_hmdb_pubchem=all_any, reactions_with_a_participant_name_not_found=rx_unmatched_names,
           interactions_152=n, families=dict(fam), family_keys={k: v.most_common(8) for k, v in famkeys.items()})
json.dump(out, open('C:/t/ev-oa/payload/gaps3.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
