"""ORCH-773: base vs tip corpus, leg by leg: verdicts, PWML bytes, gold, every omission and its KI id.

Usage: python corpus_delta.py <base.json> <tip.json> <base out-dir> <tip out-dir> <PWML_KNOWN_ISSUES.md>

* Every leg passing at base: is the tip PWML byte-identical (pwml_sha256)? Which other written files differ?
* Every leg whose verdict moved: before/after.
* Every leg at tip: each process the per-row pass omitted (quarantine_report quarantined[] rows whose
  detail carries "omission"), and each entity it removed (removed_entity_report rows with detail), with the
  KI id the catalogue appendix maps the reason to (parsed from the catalogue itself; "unclassified" if none).
* Cause counts: process vs actor (and actor slot), and D-113 actor removals.
Read-only; stdlib only.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def slug(leg_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", leg_id)


def appendix_rules(md_path):
    """[(kind, a, b, ki)] from the appendix tables: kind exact/prefix/suffix/both/message."""
    text = Path(md_path).read_text(encoding="utf-8")
    text = text[text.index("## Appendix"):]
    rules = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| reason code") or line.startswith("| message") or line.startswith("|---"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) == 4:
            code_col, match_col, ki_col = cols[0], cols[1], cols[2]
            ki = re.search(r"KI-\d+", ki_col).group(0) if re.search(r"KI-\d+", ki_col) else None
            code = re.findall(r"`([^`]*)`", code_col)
            ticks = re.findall(r"`([^`]*)`", match_col)
            if match_col.startswith("exact") and code:
                rules.append(("exact", code[0], None, ki))
            elif "prefix" in match_col and "suffix" in match_col and len(ticks) >= 2:
                rules.append(("both", ticks[0], ticks[1], ki))
            elif match_col.startswith("prefix") and ticks:
                rules.append(("prefix", ticks[0], None, ki))
            elif match_col.startswith("suffix") and ticks:
                rules.append(("suffix", ticks[0], None, ki))
        elif len(cols) == 3:
            msg, ki_col = cols[0].strip('"'), cols[1]
            ki = re.search(r"KI-\d+", ki_col).group(0) if re.search(r"KI-\d+", ki_col) else None
            parts = re.split(r"`<[^`]*>`", msg.replace("“", '"').replace("”", '"'))
            pattern = ".*".join(re.escape(p.replace("`", "")) for p in parts)
            rules.append(("message", re.compile("^" + pattern), None, ki))
    return rules


def ki_for(reason, rules):
    for kind, a, b, ki in rules:
        if kind == "exact" and reason == a:
            return ki
    for kind, a, b, ki in rules:
        if (kind == "prefix" and reason.startswith(a)) or (kind == "suffix" and reason.endswith(a)) or \
                (kind == "both" and reason.startswith(a) and reason.endswith(b)):
            label = ki
            m = re.match(r"^protein_complex_component_unrepresentable:.*?:(.*)$", reason)
            if m and kind == "prefix":
                inner = ki_for(m.group(1), rules)
                label = f"{ki} (inner: {inner})"
            return label
    for kind, a, b, ki in rules:
        if kind == "message" and a.match(reason):
            return ki
    return "unclassified"


def main():
    base, tip = load(sys.argv[1]), load(sys.argv[2])
    out_b, out_t = Path(sys.argv[3]), Path(sys.argv[4])
    rules = appendix_rules(sys.argv[5])
    b = {r["id"]: r for r in base["legs"]}
    t = {r["id"]: r for r in tip["legs"]}
    passing = lambda r: bool((r.get("replay") or {}).get("pwml_written"))  # noqa: E731

    print("== base-passing legs: PWML bytes at tip ==")
    same = diff = 0
    for i in sorted(b):
        if not passing(b[i]):
            continue
        rb, rt = b[i]["replay"], t[i]["replay"]
        ok = rb.get("pwml_sha256") == rt.get("pwml_sha256") and (rb.get("export") or {}).get("pwml_name") == (rt.get("export") or {}).get("pwml_name")
        same += ok
        diff += not ok
        changed_files = sorted(k for k in set(rb.get("files") or {}) | set(rt.get("files") or {})
                               if (rb.get("files") or {}).get(k) != (rt.get("files") or {}).get(k))
        print(f"  {'IDENTICAL' if ok else 'DIFFERENT'} {i}  pwml={rt.get('pwml_sha256', '')[:16]}  other files changed: {changed_files}")
    print(f"  -> {same} identical, {diff} different")

    print("\n== verdicts that moved ==")
    for i in sorted(b):
        sb = "pwml" if passing(b[i]) else ((b[i].get("replay") or {}).get("failure_stage") or b[i]["class"])
        st = "pwml" if passing(t[i]) else ((t[i].get("replay") or {}).get("failure_stage") or t[i]["class"])
        if sb != st:
            rel = ((t[i].get("replay") or {}).get("export") or {}).get("release")
            print(f"  {i}: {sb} -> {st}  (tip release {rel}, name {((t[i].get('replay') or {}).get('export') or {}).get('pwml_name')})")

    print("\n== gold (base -> tip) ==")
    for i in sorted(b):
        gb, gt = (b[i].get("replay") or {}).get("gold") or {}, (t[i].get("replay") or {}).get("gold") or {}
        if gb or gt:
            fmt = lambda g: " ".join(f"{lvl} R{v['R']:.3f} ({v['matched']}/{v['gold']})" for lvl, v in g.items() if isinstance(v, dict) and v.get("R") is not None)  # noqa: E731
            print(f"  {i}: {fmt(gb) or '-'}  ->  {fmt(gt) or '-'}")

    print("\n== omissions at tip, per leg ==")
    causes, slots, kis, drops = Counter(), Counter(), Counter(), Counter()
    legs_with = 0
    for i in sorted(t):
        r = t[i].get("replay")
        if not r or r.get("error"):
            continue
        work = out_t / slug(i) / "outputs"
        if not (work / "quarantine_report.json").exists():
            continue
        q, rem = load(work / "quarantine_report.json"), load(work / "removed_entity_report.json")
        procs = [x for x in q.get("quarantined") or [] if "omission" in ((x.get("detail") or {}))]
        ents = [x for x in rem.get("removed_entities") or [] if (x.get("detail") or {}).get("omission")]
        if not procs and not ents:
            continue
        legs_with += 1
        rel = (q.get("release") or {}).get("status")
        print(f"\n  {i}  [{('PWML ' + str((r.get('export') or {}).get('pwml_name'))) if r.get('pwml_written') else 'no PWML: ' + str(r.get('failure_stage'))}; release {rel}]")
        for x in procs:
            o = x["detail"]["omission"]
            ki = ki_for(x.get("reason") or "", rules)
            causes[o.get("cause")] += 1
            if o.get("cause") == "actor":
                slots[o.get("slot")] += 1
            kis[ki] += 1
            print(f"    PROCESS {x.get('process_kind')} {x.get('name')!r}: reason={x.get('reason')!r} KI={ki} cause={o.get('cause')}"
                  f" slot={o.get('slot')} via={o.get('name')!r} gate={o.get('gate')}"
                  + (f" failing={o.get('failing_entity')}" if o.get("failing_entity") else ""))
        for x in ents:
            d = x["detail"]
            ki = ki_for(x.get("reason") or "", rules)
            kis[ki] += 1
            for entry in d.get("actor_removed_from") or []:
                drops[entry.get("decision")] += 1
            kind = "TRANSPORTER DROPPED" if x.get("bucket") == "transporters" else "ENTITY"
            print(f"    {kind} {x.get('bucket')} {x.get('name')!r}: reason={x.get('reason')!r} KI={ki} gate={d.get('gate')}"
                  + (f" actor_removed_from={d.get('actor_removed_from')}" if d.get("actor_removed_from") else ""))
        print(f"    release reasons: {[r for r in (q.get('release') or {}).get('reasons') or [] if r.startswith(('omitted_from_pwml', 'per_row'))]}")
    print(f"\n== totals: {legs_with} legs with omissions; process omissions by cause {dict(causes)}; "
          f"actor slots {dict(slots)}; actors dropped from a kept process by decision {dict(drops)}; KI ids {dict(kis)}")

    print("\n== release status base -> tip, and the omission cap reason at tip ==")
    capped = 0
    for i in sorted(t):
        rb, rt = (b[i].get("replay") or {}), (t[i].get("replay") or {})
        sb, st = (rb.get("quarantine") or {}).get("release"), (rt.get("quarantine") or {}).get("release")
        wt = out_t / slug(i) / "outputs" / "quarantine_report.json"
        reasons = [r for r in ((load(wt).get("release") or {}).get("reasons") or [])
                   if r.startswith("omitted_from_pwml")] if wt.exists() else []
        capped += bool(reasons)
        if sb != st:
            print(f"  RELEASE MOVED {i}: {sb} -> {st} {reasons}")
        elif reasons:
            print(f"  cap reason, release unchanged ({st}) {i}: {reasons}")
    print(f"  legs whose frozen record carries omitted_from_pwml: {capped}")

    print("\n== every quarantine/removal record that differs base -> tip (closure consequences included) ==")
    for i in sorted(t):
        wb, wt = out_b / slug(i) / "outputs", out_t / slug(i) / "outputs"
        if not (wb / "quarantine_report.json").exists() or not (wt / "quarantine_report.json").exists():
            continue
        def rows(w):
            q, rem = load(w / "quarantine_report.json"), load(w / "removed_entity_report.json")
            procs = {(x.get("process_kind"), x.get("name"), x.get("state"), x.get("reason")) for x in q.get("quarantined") or []}
            ents = {(x.get("bucket"), x.get("name"), x.get("reason")) for key in ("removed_entities", "removed_locations", "removed_biological_states")
                    for x in rem.get(key) or []}
            return procs, ents
        pb, eb = rows(wb)
        pt, et = rows(wt)
        if pb == pt and eb == et:
            continue
        print(f"\n  {i}")
        for x in sorted(pt - pb, key=str):
            print(f"    + quarantined {x}")
        for x in sorted(pb - pt, key=str):
            print(f"    - quarantined {x}")
        for x in sorted(et - eb, key=str):
            print(f"    + removed {x}")
        for x in sorted(eb - et, key=str):
            print(f"    - removed {x}")


if __name__ == "__main__":
    main()
