"""Read-only: sample curated PathBank visualizations per membrane template (newest first), de-duplicate propagated
copies (identical membrane geometry), and dump the raw geometry each needs for the convention statistics.
usage: db2_sample.py OUT.json [PER_TEMPLATE]"""
import hashlib
import json
import sys
import time

from t2pw.bench.eval_v2.pathbank_gold import connect

PER = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TEMPLATES = [1, 84, 91, 92, 93, 94, 146, 220, 237, 238, 291, 304]
t0 = time.time()
conn = connect({"read_timeout": 600})
cur = conn.cursor()


def q(sql, args=()):
    cur.execute(sql, args)
    return cur.fetchall()


picked, seen_sig = [], set()
for t in TEMPLATES:
    cands = [r["pv"] for r in q("SELECT DISTINCT pathway_visualization_id pv FROM membrane_visualizations "
                                 "WHERE visualization_template_id=%s ORDER BY pv DESC LIMIT 600", (t,))]
    n = 0
    for pv in cands:
        if n >= PER:
            break
        mems = q("SELECT path FROM membrane_visualizations WHERE pathway_visualization_id=%s", (pv,))
        sig = hashlib.sha1("|".join(sorted(m["path"] or "" for m in mems)).encode()).hexdigest()
        if sig in seen_sig:
            continue  # propagated copy of a diagram already sampled
        seen_sig.add(sig)
        picked.append((t, pv))
        n += 1
    print("template", t, "candidates", len(cands), "picked", n, round(time.time() - t0, 1), flush=True)

out = []
for t, pv in picked:
    rec = {"via_template": t, "pv": pv}
    rec["meta"] = (q("SELECT width, height, created_at, updated_at, pathway_id FROM pathway_visualizations WHERE id=%s", (pv,)) or [{}])[0]
    rec["membranes"] = q("SELECT id, path, visualization_template_id t, complete_membrane c, options, zindex FROM membrane_visualizations "
                         "WHERE pathway_visualization_id=%s", (pv,))
    rec["labels"] = q("SELECT x, y, label_text, scale_x, visualization_template_id t FROM label_locations WHERE pathway_visualization_id=%s", (pv,))
    rec["compounds"] = q("SELECT id, x, y, visualization_template_id t, biological_state_id s, hidden FROM compound_locations "
                         "WHERE pathway_visualization_id=%s", (pv,))
    rec["proteins"] = q("SELECT id, x, y, visualization_template_id t, biological_state_id s, hidden FROM protein_locations "
                        "WHERE pathway_visualization_id=%s", (pv,))
    rec["pcv_prot"] = q("SELECT p.protein_complex_visualization_id pcv, p.protein_location_id pl FROM protein_complex_protein_visualizations p "
                        "JOIN protein_complex_visualizations c ON c.id=p.protein_complex_visualization_id WHERE c.pathway_visualization_id=%s", (pv,))
    rec["rxn_cmp"] = q("SELECT r.id rv, m.compound_location_id cl, m.side, m.edge_id e FROM reaction_visualizations r "
                       "JOIN reaction_compound_visualizations m ON m.reaction_visualization_id=r.id WHERE r.pathway_visualization_id=%s", (pv,))
    rec["rxn_enz"] = q("SELECT r.id rv, m.protein_complex_visualization_id pcv FROM reaction_visualizations r "
                       "JOIN reaction_enzyme_visualizations m ON m.reaction_visualization_id=r.id WHERE r.pathway_visualization_id=%s", (pv,))
    rec["tr_cmp"] = q("SELECT tv.id tv, m.compound_location_id cl, m.side, m.edge_id e FROM transport_visualizations tv "
                      "JOIN transport_compound_visualizations m ON m.transport_visualization_id=tv.id WHERE tv.pathway_visualization_id=%s", (pv,))
    rec["tr_tr"] = q("SELECT tv.id tv, m.protein_complex_visualization_id pcv FROM transport_visualizations tv "
                     "JOIN transport_transporter_visualizations m ON m.transport_visualization_id=tv.id WHERE tv.pathway_visualization_id=%s", (pv,))
    rec["edges"] = q("SELECT id, path, visualization_template_id t, hidden FROM edges WHERE pathway_visualization_id=%s", (pv,))
    rec["drawables"] = q("SELECT e.x, e.y, e.scale_x, e.scale_y, d.name, d.width, d.height FROM drawable_element_locations e "
                         "JOIN drawable_images d ON d.id=e.drawable_image_id WHERE e.pathway_visualization_id=%s", (pv,))
    out.append(rec)
print("fetched", len(out), round(time.time() - t0, 1), flush=True)
tpl = q("SELECT id, description, visualization_type, width, height FROM visualization_templates")
conn.close()
json.dump({"picked": picked, "pvs": out, "templates": tpl}, open(sys.argv[1], "w"), default=str)
print("done", round(time.time() - t0, 1))
