"""C-051b P2/P3: attribute the golden's second baseline move, per leg.

``c045a_golden_rebaseline.py --mode delta`` aggregates every ``(leg, config)``
pair into one path summary. That is unusable here, because four pairs stop
pre-freeze on ``AMBIGUOUS_RENAME_TARGET`` (D-015 clause 6) and a stopped pair
has *no* leaves at all -- so every one of the ~350 paths under ``ir`` reads as
"changed" on those two legs and buries the question P3 actually asks. This
splits the comparison instead:

* pairs that stop, or that the exporter refuses, are reported **as results**,
  by code, and excluded from the path diff;
* the remaining pairs -- both sides real IR -- are diffed, and their changed
  paths are partitioned into ``/0`` (the IR) and ``/1`` (the report), with the
  compound verdict field ``ir.COMPOUND_RESOLUTION_VERDICT_FIELD`` named
  separately, because the pre-freeze stage legitimately writes it.

Three steps decompose the move, so no part of it rests on assertion:

    A  raw       @ 328862a  -- what the SUPERSEDED digests hash (P1 reproduces them)
    B  prefreeze @ 328862a  -- the same routing, at pre-C-051 code
    C  prefreeze @ tip      -- what the NEW digests hash
    D  raw       @ tip      -- C-051's refusal, measured directly

``A->B`` is the routing alone; ``B->C`` is the production-code change alone;
``A->C`` is the whole move. ``D`` is why ``A`` has no successor at the tip.

Usage::  <python> c051b_delta_attribution.py --work <dir> --out F
It imports no ``t2pw`` -- it reads the sweeps and the digest lists only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

VERDICT_FIELD = "db_status"  # t2pw.pwml.ir.COMPOUND_RESOLUTION_VERDICT_FIELD


def _generalize(key: str) -> str:
    return "/".join("[]" if p[:1] + p[-1:] == "[]" else p for p in key.split("/"))


def _stops(sweep: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """``{leg: {config: raised}}`` -- the pairs that produced no IR."""
    out: Dict[str, Dict[str, str]] = {}
    for leg, per_config in sorted(sweep.items()):
        raised = {c: v["#raised"] for c, v in sorted(per_config.items()) if "#raised" in v}
        if raised:
            out[leg] = raised
    return out


def _step(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    excluded: Dict[str, List[str]] = {}
    ir_paths: Dict[str, List[str]] = {}
    report_paths: Dict[str, List[str]] = {}
    per_leg: Dict[str, Dict[str, List[str]]] = {}
    samples: Dict[str, Dict[str, Any]] = {}
    compared = 0
    for leg in sorted(before):
        for config in sorted(before[leg]):
            lhs, rhs = before[leg][config], after[leg][config]
            if "#raised" in lhs or "#raised" in rhs:
                excluded.setdefault(leg, []).append(config)
                continue
            compared += 1
            for key in sorted(set(lhs) | set(rhs)):
                if lhs.get(key, "\0absent") == rhs.get(key, "\0absent"):
                    continue
                general = _generalize(key)
                bucket = ir_paths if general.startswith("/0") else report_paths
                bucket.setdefault(general, []).append(f"{leg}::{config}")
                per_leg.setdefault(leg, {}).setdefault(general, []).append(config)
                if general.startswith("/0"):
                    samples.setdefault(general, {
                        "leg": leg, "config": config, "path": key,
                        "before": lhs.get(key, "<absent>"), "after": rhs.get(key, "<absent>")})

    ir_changed = sorted(ir_paths)
    return {
        "pairs_compared": compared,
        "pairs_excluded_because_one_side_produced_no_ir": excluded,
        "legs_changed": len(per_leg),
        "ir_paths_changed": ir_changed,
        "ir_path_samples": samples,
        "ir_pair_counts": {p: len(set(v)) for p, v in sorted(ir_paths.items())},
        "ir_paths_changed_count": len(ir_changed),
        "ir_paths_beyond_the_compound_verdict_field": [
            p for p in ir_changed if not p.endswith("/" + VERDICT_FIELD)],
        "report_top_level_keys_changed": sorted(
            {p.split("/")[2].split("#")[0] for p in report_paths if p.startswith("/1/")}),
        "report_paths_changed_count": len(report_paths),
        "per_leg_ir_paths": {leg: sorted(d) for leg, d in sorted(per_leg.items())
                             if any(p.startswith("/0") for p in d)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    work = Path(args.work)
    load = lambda n: json.loads((work / n).read_text(encoding="utf-8"))  # noqa: E731

    a, b, c, d = (load(f"{n}.json") for n in
                  ("A_raw_base", "B_prod_base", "C_prod_tip", "D_raw_tip"))
    base_digests = load("p1_digest_base.json")
    tip_digests = load("digest_tip.json")

    moved = {leg: {"from": base_digests[leg], "to": tip_digests[leg][0],
                   "changed": base_digests[leg] != tip_digests[leg][0],
                   "prefreeze_stops": tip_digests[leg][1]}
             for leg in sorted(tip_digests)}

    payload = {
        "legs": len(moved),
        "digests_changed": sum(1 for v in moved.values() if v["changed"]),
        "digests_unchanged": sorted(k for k, v in moved.items() if not v["changed"]),
        "prefreeze_stops_A_raw_base": _stops(a),
        "prefreeze_stops_B_prefreeze_base": _stops(b),
        "prefreeze_stops_C_prefreeze_tip": _stops(c),
        "refusals_D_raw_tip": _stops(d),
        "refusals_D_raw_tip_pair_count": sum(len(v) for v in _stops(d).values()),
        "step_A_to_B_routing_only_at_base_code": _step(a, b),
        "step_B_to_C_production_code_only": _step(b, c),
        "step_A_to_C_the_whole_move": _step(a, c),
        "per_leg_digest_move": moved,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
