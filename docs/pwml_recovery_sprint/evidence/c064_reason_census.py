"""C-064 reason census: what every reachable loop state reports, base vs tip.

Orchestration tooling, not pipeline code. It imports ``t2pw`` from whatever
``PYTHONPATH`` names, so the SAME script produces the base census and the tip
census and the two are directly comparable -- the c061 idiom.

Why a census and not a handful of asserts. Charter C-064 § 3 requires every
reason-combination whose REPORTED reason changes to be enumerated, and RULING 9
requires the exact set of differing keys, not one field. The honest corpus for
that is the whole reachable condition space of ``loop_policy._conditions``: the
round bound, the time bound and the nine observation axes, 3 072 states. A census
over those answers "which states moved, and to what" exhaustively, so nothing
changes as an unnamed side effect.

The axes are spelled out here rather than imported from the test file on purpose:
this script must run unchanged at the base SHA, where the test file's tip-only
constants do not exist.

Usage::

    PYTHONPATH=<tree>/src python c064_reason_census.py dump <out.json>
    PYTHONPATH=<tree>/src python c064_reason_census.py compare <base> <tip> <out>
"""

from __future__ import annotations

import json
import sys
from itertools import product

from t2pw.rag.loop_policy import TERMINATION_PRECEDENCE, LoopState, decide

#: (axis name, the LoopState overrides that select each value). The product is the
#: whole reachable condition space of the policy.
AXES = (
    ("round_bound", (dict(rounds_completed=1, max_rounds=3),
                     dict(rounds_completed=3, max_rounds=3))),
    ("time_bound", (dict(now=100.0, deadline=1000.0),
                    dict(now=1000.0, deadline=1000.0))),
    ("ladder_completed", (dict(ladder_completed=False), dict(ladder_completed=True))),
    ("round_retrieval_completed", (dict(round_retrieval_completed=False),
                                   dict(round_retrieval_completed=True))),
    ("new_admissible_claims", (dict(new_admissible_claims=0),
                               dict(new_admissible_claims=2))),
    ("graph_delta", (dict(graph_delta=0), dict(graph_delta=2))),
    ("operation_timed_out", (dict(operation_timed_out=False),
                             dict(operation_timed_out=True))),
    ("identical_empty_responses", (dict(identical_empty_responses=0),
                                   dict(identical_empty_responses=1),
                                   dict(identical_empty_responses=2))),
    ("evidence_sources_exhausted", (dict(evidence_sources_exhausted=False),
                                    dict(evidence_sources_exhausted=True))),
    ("defensible_core", (dict(defensible_core=True), dict(defensible_core=False))),
    ("attempt_cap_reached", (dict(attempt_cap_reached=False),
                             dict(attempt_cap_reached=True))),
)
NAMES = tuple(name for name, _ in AXES)


def _states():
    """Every state, with the axis-index key that identifies it across both trees."""
    for indices in product(*(range(len(values)) for _, values in AXES)):
        over = {}
        for axis, index in enumerate(indices):
            over.update(AXES[axis][1][index])
        yield "-".join(str(i) for i in indices), LoopState(**over)


def _readable(key: str) -> dict:
    """The axis-index key, back as names -- so a census row is diagnosable."""
    return {name: int(index) for name, index in zip(NAMES, key.split("-"))}


def dump(path: str) -> int:
    rows = {}
    for key, state in _states():
        decision = decide(state)
        rows[key] = [decision.reason or ""] + list(decision.also_true)
    payload = {"axes": NAMES, "precedence": list(TERMINATION_PRECEDENCE),
               "state_count": len(rows), "rows": rows}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"states={len(rows)} precedence={len(TERMINATION_PRECEDENCE)} -> {path}")
    return 0


def compare(base_path: str, tip_path: str, out_path: str) -> int:
    base = json.load(open(base_path, encoding="utf-8"))
    tip = json.load(open(tip_path, encoding="utf-8"))
    if set(base["rows"]) != set(tip["rows"]):
        print("FATAL: the two censuses do not cover the same state space")
        return 2

    classes: dict = {}
    reason_changed = also_changed = identical = 0
    lost_a_reason, gained_a_reason = [], []
    for key in sorted(base["rows"]):
        before, after = base["rows"][key], tip["rows"][key]
        if before == after:
            identical += 1
            continue
        if before[0] and not after[0]:
            lost_a_reason.append(key)       # BOUNDEDNESS BREAK: must stay empty
        if not before[0] and after[0]:
            gained_a_reason.append(key)
        if before[0] != after[0]:
            reason_changed += 1
        else:
            also_changed += 1
        signature = json.dumps({"base": before, "tip": after}, sort_keys=True)
        entry = classes.setdefault(signature, {
            "base_reason": before[0], "base_also": before[1:],
            "tip_reason": after[0], "tip_also": after[1:],
            "reason_changed": before[0] != after[0], "count": 0,
            "example_states": []})
        entry["count"] += 1
        if len(entry["example_states"]) < 3:
            entry["example_states"].append(_readable(key))

    ordered = sorted(classes.values(),
                     key=lambda e: (not e["reason_changed"], -e["count"]))
    payload = {
        "base_census": base_path, "tip_census": tip_path,
        "axes": list(base["axes"]),
        "base_precedence": base["precedence"], "tip_precedence": tip["precedence"],
        "state_count": base["state_count"],
        "states_identical": identical,
        "states_reason_changed": reason_changed,
        "states_only_also_true_changed": also_changed,
        "states_that_lost_their_reason": lost_a_reason,
        "states_that_gained_a_reason": gained_a_reason,
        "equivalence_classes": ordered,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"states={base['state_count']} identical={identical} "
          f"reason_changed={reason_changed} also_only={also_changed} "
          f"classes={len(ordered)} lost_reason={len(lost_a_reason)} -> {out_path}")
    for entry in ordered:
        flag = "REASON" if entry["reason_changed"] else "also  "
        print(f"  {flag} {entry['count']:5d}  {entry['base_reason']}"
              f"{list(entry['base_also'])} -> {entry['tip_reason']}"
              f"{list(entry['tip_also'])}")
    return 0


def main(argv) -> int:
    if len(argv) >= 3 and argv[1] == "dump":
        return dump(argv[2])
    if len(argv) >= 5 and argv[1] == "compare":
        return compare(argv[2], argv[3], argv[4])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
