"""A1: import sample PWML files one at a time into the local PathWhiz, with screenshot, SVG, PathWhiz exports
(SBML/SBGN/PWML) and a DB snapshot while each is the loaded pathway.

usage: python a1_import_all.py [--idx 7 9 ...] [--limit N] [--resume]
Reuses functions of the harness run_import.py (imported, never edited). Output:
  C:/t/corr-assess/a1/shots/<idx>.png|svg|_view.png
  C:/t/corr-assess/pathwhiz_exports/<idx>.sbml|.sbgn|.pwml|.db.json
  C:/t/corr-assess/a1/import_results.jsonl   (one row per idx)
"""
import argparse, json, os, shutil, subprocess, sys, time

sys.dont_write_bytecode = True
HARNESS = "C:/t/pwi/t2pw/tools/pathwhiz_import"
sys.path.insert(0, HARNESS)
import run_import as ri  # noqa: E402

PWML_DIR = "C:/t/corr-assess/replay/on_v2/pwml"
SHOTS = "C:/t/corr-assess/a1/shots"
EXPORTS = "C:/t/corr-assess/pathwhiz_exports"
RESULTS = "C:/t/corr-assess/a1/import_results.jsonl"
OUT = "C:/t/corr-assess/a1/import_out"
EXPORT_PY = "C:/t/corr-assess/scripts/a1_export.py"


def q(env, sql, cols):
    out = ri.mysql(env, sql)
    if out is None:
        return None
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        parts += [""] * (len(cols) - len(parts))
        rows.append(dict(zip(cols, parts)))
    return rows


def snapshot(env):
    tables = (ri.mysql(env, "SHOW TABLES") or "").split()
    keep = [t for t in tables if t not in ("users", "sessions", "schema_migrations", "visualization_templates", "drawable_images")]
    union = " UNION ALL ".join("SELECT '%s', COUNT(*) FROM `%s`" % (t, t) for t in keep)
    counts = {r["t"]: int(r["n"]) for r in q(env, union, ["t", "n"]) or []}
    counts = {k: v for k, v in sorted(counts.items()) if v}
    name = ("COALESCE((SELECT name FROM compounds c WHERE c.id=e.element_id AND e.element_type='Compound'),"
            "(SELECT name FROM nucleic_acids n WHERE n.id=e.element_id AND e.element_type='NucleicAcid'),"
            "(SELECT name FROM element_collections x WHERE x.id=e.element_id AND e.element_type='ElementCollection'),"
            "(SELECT name FROM protein_complexes pc WHERE pc.id=e.element_id AND e.element_type='ProteinComplex'),'?')")
    snap = {"counts": counts}
    snap["reaction_elements"] = q(env, "SELECT e.reaction_id, e.type, e.element_type, e.element_id, e.stoichiometry, %s FROM reaction_elements e ORDER BY e.reaction_id, e.type, e.id" % name,
                                  ["reaction_id", "type", "element_type", "element_id", "stoichiometry", "name"])
    snap["reactions"] = q(env, "SELECT id, direction, spontaneous, pwr_id FROM reactions ORDER BY id", ["id", "direction", "spontaneous", "pwr_id"])
    snap["reaction_enzymes"] = q(env, "SELECT re.reaction_id, re.protein_complex_id, pc.name, (SELECT GROUP_CONCAT(p.name SEPARATOR ' | ') FROM protein_complex_proteins cp JOIN proteins p ON p.id=cp.protein_id WHERE cp.protein_complex_id=pc.id) FROM reaction_enzymes re LEFT JOIN protein_complexes pc ON pc.id=re.protein_complex_id ORDER BY re.reaction_id, re.id",
                                 ["reaction_id", "protein_complex_id", "complex_name", "proteins"])
    snap["biological_states"] = q(env, "SELECT b.id, sl.name, b.species_id FROM biological_states b LEFT JOIN subcellular_locations sl ON sl.id=b.subcellular_location_id ORDER BY b.id", ["id", "location", "species_id"])
    snap["pathway_elements"] = q(env, "SELECT element_type, element_id, biological_state_id, process_type, process_id FROM pathway_elements ORDER BY id", ["element_type", "element_id", "biological_state_id", "process_type", "process_id"])
    snap["element_states"] = q(env, "SELECT element_type, element_id, biological_state_id FROM element_states ORDER BY id", ["element_type", "element_id", "biological_state_id"])
    snap["transports"] = q(env, "SELECT id, transport_type FROM transports ORDER BY id", ["id", "transport_type"])
    snap["transport_elements"] = q(env, "SELECT e.transport_id, e.element_type, e.element_id, e.stoichiometry, e.left_biological_state_id, e.right_biological_state_id, e.direction, %s FROM transport_elements e ORDER BY e.transport_id, e.id" % name,
                                   ["transport_id", "element_type", "element_id", "stoichiometry", "left_state", "right_state", "direction", "name"])
    snap["transport_transporters"] = q(env, "SELECT t.transport_id, t.protein_complex_id, pc.name, t.biological_state_id FROM transport_transporters t LEFT JOIN protein_complexes pc ON pc.id=t.protein_complex_id ORDER BY t.transport_id, t.id",
                                       ["transport_id", "protein_complex_id", "name", "state"])
    snap["interactions"] = q(env, "SELECT id, interaction_type, direction FROM interactions ORDER BY id", ["id", "interaction_type", "direction"])
    snap["interaction_elements"] = q(env, "SELECT e.interaction_id, e.type, e.element_type, e.element_id, e.stoichiometry, %s FROM interaction_elements e ORDER BY e.interaction_id, e.id" % name,
                                     ["interaction_id", "type", "element_type", "element_id", "stoichiometry", "name"])
    snap["compounds"] = q(env, "SELECT id, name FROM compounds ORDER BY id", ["id", "name"])
    snap["proteins"] = q(env, "SELECT id, name FROM proteins ORDER BY id", ["id", "name"])
    snap["protein_complexes"] = q(env, "SELECT id, name FROM protein_complexes ORDER BY id", ["id", "name"])
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    sample = json.load(open("C:/t/corr-assess/sample.json", encoding="utf-8"))
    idxs = a.idx or [s["idx"] for s in sample]
    done = set()
    if a.resume and os.path.exists(RESULTS):
        done = {json.loads(l)["idx"] for l in open(RESULTS, encoding="utf-8") if l.strip()}
    todo = [i for i in idxs if i not in done]
    if a.limit:
        todo = todo[:a.limit]
    for d in (SHOTS, EXPORTS, OUT, os.path.dirname(RESULTS)):
        os.makedirs(d, exist_ok=True)

    work = os.path.abspath(os.path.join(HARNESS, "work", "smpdb"))
    ri.ensure_work_copy(ri.DEFAULT_SRC, work)
    ri.ensure_config(work)
    ri.ensure_images_and_volume()
    pwml_in = os.path.join(HARNESS, "pwml_in")
    env = dict(os.environ, SMPDB_SRC=work, OUT_DIR=os.path.abspath(OUT), PWML_DIR=pwml_in)
    ri.dc(["up", "-d"], env, timeout=300)
    ri.ensure_schema_and_seed(env)

    for idx in todo:
        t0 = time.time()
        src = "%s/%d.pwml" % (PWML_DIR, idx)
        rec = {"idx": idx, "file": src, "import_ok": False}
        if not os.path.exists(src):
            rec["error"] = "pwml missing"
        else:
            name = "a1_%d.pwml" % idx
            shutil.copy(src, os.path.join(pwml_in, name))
            rows = list(ri.run_rake(env, [(name, src)], 1500))
            if not rows:
                rec["error"] = "no rake result"
            else:
                row = rows[0]
                rec.update(import_ok=bool(row["success"]), errors=row.get("errors"), created=row.get("created"),
                           created_main=row.get("created_main"), pw_id=row.get("pw_id"),
                           parser_exception=row.get("parser_exception"), expected=ri.expected_counts(src),
                           import_seconds=row.get("seconds"))
                if row["success"] and row.get("pw_id"):
                    pw = row["pw_id"]
                    try:
                        rec["snapshot_ok"] = True
                        snap = snapshot(env)
                        snap["idx"], snap["pw_id"] = idx, pw
                        json.dump(snap, open("%s/%d.db.json" % (EXPORTS, idx), "w", encoding="utf-8"), indent=1)
                    except Exception as e:
                        rec["snapshot_ok"] = False
                        rec["snapshot_error"] = repr(e)[:300]
                    ri.ensure_server(env)
                    # exports first (cheap), then the screenshot
                    r = subprocess.run([ri.PLAYWRIGHT_PY, EXPORT_PY, pw, "%s/%d" % (EXPORTS, idx)], capture_output=True, text=True, timeout=900)
                    try:
                        rec["exports"] = json.loads(r.stdout.strip().splitlines()[-1])
                    except Exception:
                        rec["exports"] = {"error": (r.stdout + r.stderr)[-300:]}
                    png, svg, info = ri.take_screenshot(pw, SHOTS, str(idx))
                    rec["screenshot"] = {"ok": info.get("ok"), "svg_bytes": info.get("svg_bytes"), "svg_elements": info.get("svg_elements"),
                                         "notes": info.get("notes"), "console_errors": info.get("console_errors")}
        rec["seconds"] = round(time.time() - t0, 1)
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        ri.log("idx %d import_ok=%s exports=%s %.0fs" % (idx, rec["import_ok"], {k: v.get("ok") for k, v in (rec.get("exports") or {}).items() if isinstance(v, dict)}, rec["seconds"]))


if __name__ == "__main__":
    main()
