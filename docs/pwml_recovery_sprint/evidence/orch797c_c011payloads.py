"""ORCH-797c: dump the c011 seam's frozen payload for chosen cohort legs under a chosen source tree.

Usage: python orch797c_c011payloads.py --src <tree>/src --out <dir> [--compare <other dir>]
For every cohort leg whose name contains one of LEGS, runs the c011 seam exactly as ``_leg_projection`` does and
writes the frozen payload to <dir>/<n>.json. With --compare, prints per leg what differs from <other dir>'s dump:
entity rows per bucket (names), process rows per bucket (name + participants), and removed rows.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

LEGS = ("2026-07-28_2122/papers/PMC12180156", "2026-07-28_2122/papers/PMC12444477", "2026-07-28_2122/papers/PMC12657337",
        "2026-07-28_2122/papers/PMC12856317", "2026-07-28_2122/papers/PMC13233763", "2026-08-02_2130/papers/PMC12096016/strict")
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--compare")
args = ap.parse_args()
sys.path.insert(0, str(Path(args.src).resolve()))
import t2pw  # noqa: E402

print("t2pw resolved:", Path(t2pw.__file__).resolve().parent)
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "tests"))
import test_c011_freeze_seam_golden_equivalence as C  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
seam = C._seam(None)


def summary(payload):
    ents = {b: sorted(str(r.get("name")) for r in rows if isinstance(r, dict))
            for b, rows in (payload.get("entities") or {}).items() if isinstance(rows, list)}
    procs = {}
    for b, rows in (payload.get("processes") or {}).items():
        for r in rows if isinstance(rows, list) else []:
            if isinstance(r, dict):
                parts = {k: r.get(k) for k in ("inputs", "outputs", "entity_1", "entity_2", "cargo", "enzymes", "transporters")
                         if r.get(k)}
                procs.setdefault(b, []).append(f"{r.get('name')} :: {json.dumps(parts, sort_keys=True, default=str)[:300]}")
    return ents, {b: sorted(v) for b, v in procs.items()}


with tempfile.TemporaryDirectory() as raw:
    for n, leg in enumerate(C.replay._legs()):
        name = leg.relative_to(C.ROOT).as_posix()
        if not any(key in name for key in LEGS):
            continue
        work = Path(raw) / f"leg{n}"
        for d in (work / "outputs", work / "tmp"):
            d.mkdir(parents=True, exist_ok=True)
        normalized = C.replay._normalized(C.replay._payload_for(leg)[1])
        result = seam(normalized, strict_db=True, outputs_dir=work / "outputs", pathway_context=None,
                      export_mode=leg.name, research_mode=C.is_research(leg.name), tmp=work / "tmp")
        payload = result["payload"] or normalized
        fname = name.replace("/", "__") + ".json"
        (out / fname).write_text(json.dumps(payload, indent=1, sort_keys=True, default=str, ensure_ascii=False),
                                 encoding="utf-8")
        print("dumped", name)
        if args.compare and (Path(args.compare) / fname).is_file():
            other = json.loads((Path(args.compare) / fname).read_text(encoding="utf-8"))
            (e1, p1), (e2, p2) = summary(other), summary(payload)
            for b in sorted(set(e1) | set(e2)):
                gone, new = sorted(set(e1.get(b, [])) - set(e2.get(b, []))), sorted(set(e2.get(b, [])) - set(e1.get(b, [])))
                if gone or new:
                    print(f"  ENTITIES {b}: only-in-compare {gone} | only-here {new}")
            for b in sorted(set(p1) | set(p2)):
                gone, new = sorted(set(p1.get(b, [])) - set(p2.get(b, []))), sorted(set(p2.get(b, [])) - set(p1.get(b, [])))
                for g in gone:
                    print(f"  PROCESS {b} only-in-compare: {g}")
                for g in new:
                    print(f"  PROCESS {b} only-here     : {g}")
            if e1 == e2 and p1 == p2:
                keys = sorted(k for k in set(other) | set(payload) if other.get(k) != payload.get(k))
                print(f"  same entities and processes; differing top-level keys: {keys}")
