"""Count what PathWhiz's own SBML / SBGN-ML exports contain, next to the DB row counts and PWML (for A3 and the completeness table)."""
import json, os, re, collections
EXP = 'C:/t/corr-assess/pathwhiz_exports'
out = {}
for s in json.load(open('C:/t/corr-assess/sample.json', encoding='utf-8')):
    i = s['idx']
    r = {}
    p = '%s/%d.sbml' % (EXP, i)
    if os.path.exists(p):
        t = open(p, encoding='utf-8', errors='replace').read()
        r['sbml_bytes'] = len(t)
        r['sbml_reactions'] = len(re.findall(r'<reaction\b', t))
        r['sbml_species'] = len(re.findall(r'<species\b', t))
        r['sbml_compartments'] = len(re.findall(r'<compartment\b', t))
        r['sbml_reactants'] = len(re.findall(r'<speciesReference\b', t))
        r['sbml_modifiers'] = len(re.findall(r'<modifierSpeciesReference\b', t))
        r['sbml_listOfReactants'] = len(re.findall(r'<listOfReactants', t))
        r['sbml_listOfProducts'] = len(re.findall(r'<listOfProducts', t))
    p = '%s/%d.sbgn' % (EXP, i)
    if os.path.exists(p):
        t = open(p, encoding='utf-8', errors='replace').read()
        r['sbgn_bytes'] = len(t)
        r['sbgn_glyphs'] = len(re.findall(r'<glyph\b', t))
        cls = collections.Counter(re.findall(r'<glyph\b[^>]*\bclass="([^"]+)"', t))
        r['sbgn_glyph_classes'] = dict(cls)
        r['sbgn_arcs'] = len(re.findall(r'<arc\b', t))
        r['sbgn_arc_classes'] = dict(collections.Counter(re.findall(r'<arc\b[^>]*\bclass="([^"]+)"', t)))
    p = '%s/%d.pwml' % (EXP, i)
    if os.path.exists(p):
        t = open(p, encoding='utf-8', errors='replace').read()
        r['reexport_pwml_bytes'] = len(t)
        r['reexport_reactions'] = len(re.findall(r'<reaction>', t))
    p = '%s/%d.db.json' % (EXP, i)
    if os.path.exists(p):
        d = json.load(open(p, encoding='utf-8'))
        r['db_reactions'] = d['counts'].get('reactions', 0)
        r['db_reaction_enzymes'] = d['counts'].get('reaction_enzymes', 0)
        r['db_reaction_elements'] = d['counts'].get('reaction_elements', 0)
    out[i] = r
    print(i, {k: v for k, v in r.items() if 'classes' not in k})
json.dump(out, open('C:/t/corr-assess/a1/export_counts.json', 'w', encoding='utf-8'), indent=1)
