"""C-108: flatten rev107_corpus.py's {"verdicts":..,"meta":..} dump into the flat
key->bool shape c107_corpus_diff.py consumes, so the corpus can be scored with the
DATA root held fixed and only the CODE root varied.

Usage:  <python> c108_flatten_corpus.py <in.json> <out.json>
"""
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
v = d["verdicts"] if isinstance(d, dict) and "verdicts" in d else d
Path(sys.argv[2]).write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
print("rows:", len(v), " accepted:", sum(1 for x in v.values() if x),
      " refused:", sum(1 for x in v.values() if not x))
