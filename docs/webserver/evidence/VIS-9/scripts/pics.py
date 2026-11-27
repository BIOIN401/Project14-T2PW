"""Per pathway, after a PathWhiz import: every compound record of OUR input PWML classified by what PathWhiz holds for it
in its own export (matched by name): dropped (absent from the export), imported with OUR pwc-id (the right picture when
MolDB has a structure for it: moldb/<pwc>.png present and not the 'No Structure Available' placeholder), imported with
our pwc-id but MolDB has no structure, or imported with a dev-DB pwc-id (no correct picture). Split by HMDB presence.
usage: pics.py RESULTS.jsonl OUT.json"""
import hashlib, json, sys
from pathlib import Path
from lxml import etree
W = Path("C:/t/vis-pw9-work"); res = Path(sys.argv[1]); base = res.parent
PLACEHOLDER = hashlib.md5((W / "moldb/PW_C118678.png").read_bytes()).hexdigest()
def comps(p):
    root = etree.parse(str(p), etree.XMLParser(recover=True, huge_tree=True)).getroot()
    return [{"name": (c.findtext("name") or "").strip(), "pwc": (c.findtext("pwc-id") or "").strip(), "hmdb": (c.findtext("hmdb-id") or "").strip()}
            for c in root.iter("compound") if c.getparent() is not None and c.getparent().tag == "compounds"]
out = {}
for line in res.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    name = Path(r["file"].replace("\\", "/")).stem.replace("ref_", "")
    exp = base / "exports" / (r.get("staged", "") + ".pwml")
    ein = {c["name"].casefold(): c for c in comps(exp)} if exp.exists() else None
    rows, cnt = [], {}
    for c in comps(Path(r["file"].replace("\\", "/"))):
        e = (ein or {}).get(c["name"].casefold())
        if ein is None: k = "no_export"
        elif e is None:
            same = [x for x in ein.values() if (c["hmdb"] and x["hmdb"] == c["hmdb"]) or (c["pwc"] and x["pwc"] == c["pwc"])]
            k = ("merged_into:" + same[0]["name"]) if same else "dropped"
        elif c["pwc"] and e["pwc"] == c["pwc"]:
            png = W / "moldb" / (c["pwc"] + ".png")
            k = "missing_moldb_check" if not png.exists() else ("our_pwc_no_structure" if hashlib.md5(png.read_bytes()).hexdigest() == PLACEHOLDER else "our_pwc_picture")
        elif c["pwc"]: k = "our_pwc_replaced:" + e["pwc"]
        else: k = "dev_pwc_no_correct_picture"
        key = ("hmdb" if c["hmdb"] else "no_hmdb") + "/" + k.split(":")[0]
        cnt[key] = cnt.get(key, 0) + 1
        rows.append({**c, "class": k, "export_pwc": e and e["pwc"]})
    out[name] = {"success": r.get("success"), "error_count": r.get("error_count"), "counts": cnt, "rows": rows}
    print(name, r.get("success"), r.get("error_count"), json.dumps(cnt, sort_keys=True))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
