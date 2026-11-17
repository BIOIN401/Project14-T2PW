import sys, json
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
B = M.B
i = 9
for nm, f in (("payload", lambda: M.from_payload(json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8")))),
              ("ir", lambda: M.from_ir(json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8")))),
              ("pwml", lambda: M.from_pwml(f"{B}/replay/on_v2/pwml/{i}.pwml")),
              ("db", lambda: M.from_db(json.load(open(f"{B}/pathwhiz_exports/{i}.db.json", encoding="utf-8")))),
              ("pwsbml", lambda: M.from_pw_sbml(f"{B}/pathwhiz_exports/{i}.sbml")),
              ("web", lambda: M.from_web(json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8")))),
              ("pwsbgn", lambda: M.from_pw_sbgn(f"{B}/pathwhiz_exports/{i}.sbgn"))):
    m = f()
    print(nm, len(m["comps"]), len(m["prots"]), len(m["cplx"]), len(m["nuc"]), "rxn", len(m["rxn"]), "trn", len(m["trn"]), "itn", len(m["itn"]), sorted(m["locs"])[:6])
    print("   ", m["itn"][:1], m["trn"][:1])
