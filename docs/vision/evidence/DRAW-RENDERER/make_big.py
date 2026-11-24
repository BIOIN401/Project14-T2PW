"""Synthetic large IR for timing only: several stored IRs side by side, every key prefixed so nothing collides.
usage: make_big.py <out.json> <ir.json>..."""
import json
import sys


def pre(x, p):
    if isinstance(x, dict):
        return {k: (p + v if isinstance(v, str) and (k == "key" or k.endswith("_key")) else pre(v, p)) for k, v in x.items()}
    if isinstance(x, list):
        return [pre(v, p) for v in x]
    return x


out = None
for i, path in enumerate(sys.argv[2:]):
    ir = pre(json.load(open(path, encoding="utf-8")), "z%d_" % i)
    if out is None:
        out = ir
        continue
    for k, v in ir.items():
        if isinstance(v, list) and isinstance(out.get(k), list):
            out[k] += v
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            for kk, vv in v.items():
                if isinstance(vv, list):
                    out[k].setdefault(kk, []).extend(vv)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"))
print("ok")
