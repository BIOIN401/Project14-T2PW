"""ORCH-759e: compare the Stage-6 replay census at base, at tip, and under the placement variants.

Stdlib only, reads the ``census-*.jsonl`` files ``stage6_replay.py`` wrote next to this file, prints.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ["base", "tip", "m1", "m2_full", "m1_full"]


def load(name):
    return {r["leg"]: r for r in map(json.loads, (HERE / f"census-{name}.jsonl").read_text(encoding="utf-8").splitlines())}


def brief(checks):
    if not checks:
        return "-"
    q = checks["quarantine"]
    c = q.get("counts") or {}
    return (f"quarantine ok={q['ok']} release={q['release']} core/aux={c.get('core_accepted')}/{c.get('auxiliary_accepted')} "
            f"broken_ref={c.get('quarantined_broken_reference')} unmapped={c.get('quarantined_unmapped_entity')} | "
            f"gate={checks.get('final_gate')} | contract={checks['contract']} | pre_export={checks['pre_export']} | ir={checks['ir']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    runs = {v: load(v) for v in VARIANTS}
    legs = list(runs["base"])
    assert all(set(r) == set(legs) for r in runs.values()), "the variants did not replay the same legs"
    errors = {v: [leg for leg, r in rs.items() if "error" in r] for v, rs in runs.items()}
    print("legs:", len(legs), "| replay errors:", {v: len(e) for v, e in errors.items()})
    late = [leg for leg in legs if runs["base"][leg].get("late")]
    print("legs with a protein the Phase-2 fallback resolved:", len(late))
    quiet = [leg for leg in legs if leg not in late]
    same = [leg for leg in quiet if len({runs[v][leg]["payload_digest"] for v in VARIANTS}) == 1
            and len({runs[v][leg]["report_digest"] for v in VARIANTS}) == 1]
    print(f"legs with none: {len(quiet)}; payload AND report byte-identical in all five runs: {len(same)}")
    for leg in quiet:
        if leg not in same:
            print("   DIFFERS:", leg, {v: runs[v][leg]["payload_digest"] for v in VARIANTS})
    for v in VARIANTS[1:]:
        changed = [leg for leg in legs if runs[v][leg]["payload_digest"] != runs["base"][leg]["payload_digest"]]
        print(f"\n== {v} vs base: {len(changed)} leg(s) change")
        for leg in changed:
            b, t = runs["base"][leg], runs[v][leg]
            print(f"  {leg}  late={t['late']}  actors in/out base={b['actors_in']}/{b['actors_out']} {v}={t['actors_in']}/{t['actors_out']}")
            print(f"     base : {b['late_actors']}")
            print(f"     {v:5}: {t['late_actors']}")
            print(f"     base  replay: {brief(b.get('replay_checks'))}")
            print(f"     {v:5} replay: {brief(t.get('replay_checks'))}")
    print("\n== live artifact vs base replay (does the replay reproduce the leg?)")
    for leg in late:
        b = runs["base"][leg]
        print(f"  {leg}\n     live       : {brief(b.get('live_checks'))}\n     base replay: {brief(b.get('replay_checks'))}")


if __name__ == "__main__":
    main()
