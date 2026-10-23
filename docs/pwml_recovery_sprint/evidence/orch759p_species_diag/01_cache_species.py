"""Stdlib-only: dump species_taxonomy cache entries and any lactis/KF147 keys."""
import json
import sys

ROOT = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW"
path = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/data/id_mapping_cache.json"
d = json.load(open(path, encoding="utf-8"))
print("top keys:", list(d.keys())[:40] if isinstance(d, dict) else type(d))
sec = d.get("species_taxonomy") if isinstance(d, dict) else None
if sec is None and isinstance(d, dict):
    for k, v in d.items():
        if isinstance(v, dict) and "species_taxonomy" in v:
            sec = v["species_taxonomy"]
            print("found under", k)
print("species_taxonomy entries:", len(sec or {}))
for k, v in sorted((sec or {}).items()):
    print(" ", k, "->", json.dumps(v))


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = path + "/" + str(k)
            if "lactis" in str(k).lower() or "kf147" in str(k).lower():
                print("KEY", p[:200], "->", json.dumps(v)[:300])
            else:
                walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, path + f"[{i}]")


walk(d)
