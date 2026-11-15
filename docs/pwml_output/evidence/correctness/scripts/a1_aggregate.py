import json, collections
d = json.load(open('C:/t/corr-assess/a1/per_pathway.json', encoding='utf-8'))
imp = {json.loads(l)['idx']: json.loads(l) for l in open('C:/t/corr-assess/a1/import_results.jsonl', encoding='utf-8') if l.strip()}
tot = collections.Counter()
lists = collections.defaultdict(list)
for k, v in d.items():
    i = int(k)
    r = v['reactions']
    tot['rx_payload'] += r['payload']
    tot['rx_pwml'] += r['pwml']
    if r['missing_in_pwml']:
        lists['reaction_missing_in_pwml'].append(i)
    db = v.get('db') or {}
    tot['rx_db'] += db.get('reactions_db', 0)
    if db.get('reactions_merged_by_importer'):
        lists['importer_merged_reactions'].append((i, db['reactions_merged_by_importer']))
    if v['payload_dups']['exact_dup']:
        lists['payload_exact_dup_reactions'].append((i, v['payload_dups']['exact_dup']))
    if v['payload_dups']['reverse_pairs_any_enzyme']:
        lists['payload_reverse_pairs'].append((i, v['payload_dups']['reverse_pairs_any_enzyme']))
    if v['payload_dups']['same_inout_diff_enzyme']:
        lists['payload_same_inout_diff_enzyme'].append((i, v['payload_dups']['same_inout_diff_enzyme']))
    if v['noncatalyst_modifiers']['written_as_reaction_enzyme']:
        lists['noncatalyst_as_enzyme'].append((i, v['noncatalyst_modifiers']['payload_roles']))
    if v['side_vs_role']['side_contradicts_list']:
        lists['side_wrong'].append(i)
    if v['enzymes']['viz_complex_mismatch'] or v['enzymes']['enzyme_not_drawn']:
        lists['enzyme_wrong'].append(i)
    if v['states']['null_state_reactions_payload']:
        lists['null_state_reactions'].append((i, v['states']['null_state_reactions_payload']))
    if v['compartment_boxes']['dup_labels']:
        lists['dup_box_labels'].append((i, v['compartment_boxes']['dup_labels']))
    sg = v['states']['synonym_groups_used']
    multi = {g: m for g, m in sg.items() if len(m) > 1}
    if multi:
        lists['near_synonym_state_groups'].append((i, len(multi)))
    if v['compartment_boxes']['organism_or_strain_named']:
        lists['organism_named_box'].append((i, v['compartment_boxes']['organism_or_strain_named']))
    dr = v['drawing']
    if dr['shared_redrawn']:
        lists['shared_compound_redrawn'].append(i)
    if dr['drawn_components'] > dr['logical_components']:
        lists['chain_fragmented'].append((i, dr['drawn_components'], dr['logical_components']))
    tot['loc'] += dr['compound_locations']
    tot['distinct'] += dr['distinct_compound_state']
    tot['prec'] += v['flow']['precedence_pairs']
    tot['back'] += v['flow']['right_to_left_pairs']
    if v['flow']['right_to_left_pairs']:
        lists['flow_backward'].append((i, v['flow']['right_to_left_pairs'], v['flow']['precedence_pairs']))
    if v['canvas_use']['fit_scale_1600x900'] < 0.3:
        lists['fit_scale_lt_0.3'].append((i, v['canvas_use']['fit_scale_1600x900']))
    t = v['transports']
    if t['missing_in_pwml'] and any(x[1] != '__auto_state__' for x in t['missing_in_pwml']):
        pass
    tr_pl, tr_pw = t['payload'], t['pwml']
    tot['tr_payload'] += tr_pl
    tot['tr_pwml'] += tr_pw
    tot['tr_db'] += (db.get('counts') or {}).get('transports', 0)
    tot['int_payload'] += v['interactions']['payload']
    tot['int_pwml'] += v['interactions']['pwml']
    tot['int_db'] += (db.get('counts') or {}).get('interactions', 0)
    if tr_pw > (db.get('counts') or {}).get('transports', 0) and imp.get(i, {}).get('import_ok'):
        lists['importer_dropped_or_merged_transport'].append((i, tr_pw, (db.get('counts') or {}).get('transports', 0)))
    if v['interactions']['pwml'] > (db.get('counts') or {}).get('interactions', 0):
        lists['importer_dropped_interaction'].append((i, v['interactions']['pwml'], (db.get('counts') or {}).get('interactions', 0)))
    if v['transports']['same_state_both_ends_pwml']:
        lists['transport_same_state_both_ends'].append((i, v['transports']['same_state_both_ends_pwml']))
print(dict(tot))
for k, v in lists.items():
    print(k, len(v), v)
