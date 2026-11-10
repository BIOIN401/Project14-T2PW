"""libsbml check of the filled SBML files (scratch venv). Reuses validate_sbml.check."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_sbml
out = {}
for f in sorted(glob.glob("C:/t/ev-cm/sbml_out_fill/*.xml")):
    r = validate_sbml.check(f); r.pop("messages", None); out[os.path.basename(f)] = r
    print(os.path.basename(f), r.get("read"), r.get("errors"), r.get("warnings"), r.get("read_errors"))
json.dump(out, open("C:/t/cm/docs/pwml_output/evidence/clean_model/measure_fill_validate.json", "w"), indent=1)
