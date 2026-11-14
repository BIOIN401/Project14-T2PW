"""Compare the arms of the T2PW_TRANSPORT_SIDES live test.

usage: python compare.py <live_out_dir> [<md_out>]

For every paper dir <live_out_dir>/<paper>/ with runs named off_<n> / on_<n>:
  * per run: fact counts by category (entities by type, reactions, transports, interactions,
    biological states, locations) at Stage 1 (stage1_post.json) and after the normalizer;
    transport sides: ok / one-sided / equal at both levels
  * paired (same Stage-1 draw): stage1_pre vs stage1_post of every on run -- exact set differences
  * unpaired: every on-vs-off pair, and every off-vs-off pair (the noise floor), as
    only-in-first / only-in-second counts per category on the normalized payload
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

AUTO = "__auto_state__"


def n(v):
    return " ".join(str(v or "").casefold().split())


def facts(payload):
    f = {}
    ents = payload.get("entities") or {}
    for typ, rows in ents.items():
        if isinstance(rows, list):
            f[f"entity:{typ}"] = {n(r.get("name") if isinstance(r, dict) else r) for r in rows}
    p = payload.get("processes") or {}
    f["reaction"] = {(tuple(sorted(n(x) for x in r.get("inputs") or [])), tuple(sorted(n(x) for x in r.get("outputs") or [])))
                     for r in p.get("reactions") or [] if isinstance(r, dict)}
    f["transport"] = {(n(t.get("cargo")), n(sides(t)[0]), n(sides(t)[1])) for t in p.get("transports") or [] if isinstance(t, dict)}
    f["interaction"] = {(n(i.get("entity_1")), n(i.get("entity_2")), n(i.get("relationship")))
                        for i in p.get("interactions") or [] if isinstance(i, dict)}
    f["state"] = {(n(s.get("name")), n(s.get("subcellular_location"))) for s in payload.get("biological_states") or [] if isinstance(s, dict)}
    loc = set()
    for key, rows in (payload.get("element_locations") or {}).items():
        for r in rows or []:
            if isinstance(r, dict):
                ent = next((v for k, v in r.items() if k not in ("biological_state", "evidence") and isinstance(v, str)), "")
                loc.add((key, n(ent), n(r.get("biological_state"))))
    f["location"] = loc
    return f


def sides(t):
    def one(field, side):
        v = str(t.get(field) or "").strip()
        if v:
            return v
        names = {str(e.get("biological_state") or "").strip() for e in t.get("elements_with_states") or []
                 if isinstance(e, dict) and str(e.get("side", "")).strip().lower() == side}
        names.discard("")
        return names.pop() if len(names) == 1 else ""
    return one("from_biological_state", "left"), one("to_biological_state", "right")


def transport_health(payload):
    ok = one = eq = 0
    rows = []
    for t in (payload.get("processes") or {}).get("transports") or []:
        a, b = sides(t)
        if not a and not b or n(a) == n(b):
            eq += 1
            kind = "equal/neither"
        elif not a or not b or AUTO in (a, b):
            one += 1
            kind = "one-sided"
        else:
            ok += 1
            kind = "ok"
        rows.append((t.get("name"), t.get("cargo"), a, b, kind))
    return {"ok": ok, "one_sided": one, "equal_or_neither": eq, "rows": rows}


def diff(a, b):
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, set()), b.get(k, set())
        out[k] = (sorted(map(str, x - y)), sorted(map(str, y - x)), len(x), len(y))
    return out


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def counts(f):
    return {k: len(v) for k, v in f.items()}


def main():
    root = Path(sys.argv[1])
    md = []
    machine = {}
    for pdir in sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("PMC")):
        runs = {r.name: r for r in sorted(pdir.iterdir()) if r.is_dir()}
        md += [f"## {pdir.name}", ""]
        norm = {}
        md += ["| run | wall s | LLM requests | deficient | repairs | repair calls | S1 pre==post | transports ok/one-sided/equal (S1) | (normalized) | reactions | entities | interactions |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in runs.items():
            s = load(r / "summary.json") or {}
            s1 = load(r / "stage1_post.json")
            nm = load(r / "normalized_payload.json")
            h1 = transport_health(s1) if s1 else None
            hn = transport_health(nm) if nm else None
            if nm:
                norm[name] = facts(nm)
            ent = sum(v for k, v in counts(norm[name]).items() if k.startswith("entity:")) if name in norm else "-"
            fmt = lambda h: f"{h['ok']}/{h['one_sided']}/{h['equal_or_neither']}" if h else "-"
            md.append(f"| {name} | {s.get('wall_seconds')} | {s.get('llm_requests_total')} | {s.get('deficient', '-')} | {s.get('repairs', '-')} | "
                      f"{s.get('repair_calls', '-')} | {s.get('stage1_identical_pre_post', '-')} | {fmt(h1)} | {fmt(hn)} | "
                      f"{counts(norm[name]).get('reaction', '-') if name in norm else '-'} | {ent} | {counts(norm[name]).get('interaction', '-') if name in norm else '-'} |")
        md.append("")
        # defective / repaired transports, per run
        for name, r in runs.items():
            rep = load(r / "transport_sides_report.json")
            if rep and (rep.get("deficient") or rep.get("error")):
                md += [f"### {pdir.name} {name}: repair record", "", "```json", json.dumps(rep, indent=1, ensure_ascii=False)[:6000], "```", ""]
        # paired Stage-1 differences
        for name, r in runs.items():
            pre, post = load(r / "stage1_pre.json"), load(r / "stage1_post.json")
            if name.startswith("on_") and pre and post:
                d = diff(facts(pre), facts(post))
                changed = {k: v for k, v in d.items() if v[0] or v[1]}
                md.append(f"paired Stage-1 diff {name} (only-in-pre / only-in-post): " + (json.dumps(changed, ensure_ascii=False) if changed else "no fact differs (byte-identical)"))
        md.append("")
        # unpaired pairs, normalized
        names = sorted(norm)
        md += ["| pair | kind | " + " | ".join(["reaction", "transport", "interaction", "state", "location", "entities"]) + " |", "|---|---|---|---|---|---|---|---|"]
        for a, b in itertools.combinations(names, 2):
            kind = "off-off" if a.startswith("off") and b.startswith("off") else "on-on" if a.startswith("on") and b.startswith("on") else "off-on"
            d = diff(norm[a], norm[b])
            ent_a = sum(len(v[0]) for k, v in d.items() if k.startswith("entity:"))
            ent_b = sum(len(v[1]) for k, v in d.items() if k.startswith("entity:"))
            cell = lambda k: f"{len(d[k][0])}/{len(d[k][1])}" if k in d else "-"
            md.append(f"| {a} vs {b} | {kind} | {cell('reaction')} | {cell('transport')} | {cell('interaction')} | {cell('state')} | {cell('location')} | {ent_a}/{ent_b} |")
        md += ["", "(cells: only-in-first / only-in-second)", ""]
        machine[pdir.name] = {k: counts(v) for k, v in norm.items()}
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "COMPARE.md"
    out.write_text("\n".join(md), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(machine, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
