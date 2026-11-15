"""Noise floor: size of the symmetric difference of normalized-payload facts (reaction, interaction,
state, location, entities; transports excluded) for off-vs-off, off-vs-on and on-vs-on pairs.

usage: python noise.py <runs_dir> <out_md>     (reuses compare.py's facts() and diff())
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare as c  # noqa: E402

CATS = ["reaction", "interaction", "state", "location"]
out = ["| paper | pair kind | n pairs | mean sym.diff (reaction+interaction+state+location+entities) | min | max | per-pair totals |", "|---|---|---|---|---|---|---|"]
for pdir in sorted(Path(sys.argv[1]).glob("PMC*")):
    norm = {}
    for r in sorted(d for d in pdir.iterdir() if d.is_dir()):
        p = r / "normalized_payload.json"
        if p.exists():
            norm[r.name] = c.facts(json.loads(p.read_text(encoding="utf-8")))
    kinds = {}
    for a, b in itertools.combinations(sorted(norm), 2):
        d = c.diff(norm[a], norm[b])
        tot = sum(len(v[0]) + len(v[1]) for k, v in d.items() if k in CATS or k.startswith("entity:"))
        kind = "off-off" if a.startswith("off") and b.startswith("off") else "on-on" if a.startswith("on") and b.startswith("on") else "off-on"
        kinds.setdefault(kind, []).append(tot)
    for kind in ("off-off", "off-on", "on-on"):
        v = kinds.get(kind, [])
        if v:
            out.append(f"| {pdir.name} | {kind} | {len(v)} | {sum(v) / len(v):.1f} | {min(v)} | {max(v)} | {v} |")
Path(sys.argv[2]).write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
