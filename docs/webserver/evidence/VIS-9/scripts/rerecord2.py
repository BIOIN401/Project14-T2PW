"""VIS-9 re-record, second field: recorded.outcome.message of the equivalence fixture names the PWML byte count
("... pathway.review_required.pwml is 53474 bytes"); set it to the new count. usage: rerecord2.py TREE"""
import gzip, json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "tests/fixtures/equivalence/sour_taste_v2_strict.json.gz"
F = json.loads(gzip.decompress(p.read_bytes()))
o = F["recorded"]["outcome"]; old = o["message"]
assert old == "strict run completed; pathway.review_required.pwml is 53474 bytes", old
assert F["recorded"]["pwml_bytes"] == 53554
o["message"] = old.replace("53474 bytes", "53554 bytes")
p.write_bytes(gzip.compress(json.dumps(F, indent=1, ensure_ascii=False).encode("utf-8"), mtime=0))
line = f"tests/fixtures/equivalence/sour_taste_v2_strict.json.gz: recorded.outcome.message {old!r} -> {o['message']!r}"
open("C:/t/vis-pw9-work/rerec/rerecord_log.txt", "a", encoding="utf-8").write(line + "\n")
print(line)
