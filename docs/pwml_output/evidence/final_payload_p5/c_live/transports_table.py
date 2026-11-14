"""Per run: every Stage-1 transport (before and after the step) and every normalized transport.

usage: python transports_table.py <runs_dir> <out_md>
Reuses compare.py's side reading and health classes (no new logic).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compare as c  # noqa: E402


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rows(payload):
    if not payload:
        return []
    return [(t.get("name"), t.get("cargo"), *c.sides(t)[:2], kind) for (_, _, _, _, kind), t in
            zip(c.transport_health(payload)["rows"], (payload.get("processes") or {}).get("transports") or [])]


def fmt(rs):
    if not rs:
        return "  (no transports)"
    return "\n".join(f"  - {n} | cargo {g} | from `{a or '-'}` | to `{b or '-'}` | {k}" for n, g, a, b, k in rs)


out = []
tot = {}
for pdir in sorted(Path(sys.argv[1]).glob("PMC*")):
    out.append(f"## {pdir.name}")
    for r in sorted(d for d in pdir.iterdir() if d.is_dir()):
        pre, post, nm = load(r / "stage1_pre.json"), load(r / "stage1_post.json"), load(r / "normalized_payload.json")
        out.append(f"### {r.name}")
        if r.name.startswith("on") and pre != post:
            out.append("Stage 1 before the step:\n" + fmt(rows(pre)))
            out.append("Stage 1 after the step:\n" + fmt(rows(post)))
        else:
            out.append("Stage 1 (step changed nothing):\n" + fmt(rows(post)))
        if nm:
            out.append("normalized:\n" + fmt(rows(nm)))
        h = c.transport_health(post) if post else None
        if h:
            t = tot.setdefault((pdir.name, r.name[:2]), [0, 0, 0])
            t[0] += h["ok"]; t[1] += h["one_sided"]; t[2] += h["equal_or_neither"]
        out.append("")
Path(sys.argv[2]).write_text("\n".join(out), encoding="utf-8")
print("wrote", sys.argv[2])
