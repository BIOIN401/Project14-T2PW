import json
d = json.load(open('C:/t/corr-assess/a1/per_pathway.json', encoding='utf-8'))
print('idx  rxnP/W dupP(ex,rev,sameIO) nc_mod  states(P/W/used/unused) boxes dupLbl syn  drawn(loc/dist/multi/sharedRedrawn/comp drawn-logical) flow(b/p) fit canvas')
for k, v in d.items():
    if 'error' in v:
        print(k, 'ERR', v['error'][-200:]); continue
    r, s, dr, f = v['reactions'], v['states'], v['drawing'], v['flow']
    pd = v['payload_dups']
    print(k.rjust(4), '%d/%d' % (r['payload'], r['pwml']), (pd['exact_dup'], pd['reverse_pairs_any_enzyme'], pd['same_inout_diff_enzyme']),
          v['noncatalyst_modifiers']['written_as_reaction_enzyme'], (s['payload'], s['pwml'], len(s['used_in_pwml']), len(s['defined_unused'])),
          v['compartment_boxes']['boxes'], len(v['compartment_boxes']['dup_labels']), len(s['synonym_groups_used']),
          (dr['compound_locations'], dr['distinct_compound_state'], dr['compounds_with_multiple_locs'], dr['shared_redrawn'], dr['drawn_components'], dr['logical_components']),
          '%d/%d' % (f['right_to_left_pairs'], f['precedence_pairs']), v['canvas_use']['fit_scale_1600x900'], v['canvas'])
