"""Stdlib-only: CPO-keyed entries in the leg's id_mapping_cache snapshot."""
import json, re, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("top:", type(p).__name__, list(p.keys())[:20] if isinstance(p, dict) else len(p))
def it(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            if re.search(r"(^|[^a-z])cpo(1)?([^a-z]|$)", str(k).lower()) or "coproporphyrinogen" in str(k).lower():
                print(path + "/" + str(k), "=>", json.dumps(v, ensure_ascii=False)[:900])
            elif isinstance(v, dict) and len(path.split("/")) < 3:
                it(v, path + "/" + str(k))
it(p)
