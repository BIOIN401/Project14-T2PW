"""Read-only replay of frontier.generate() over the persisted pilot-001 state.

Imports the FROZEN mining code at C:/t/mining-pilot/src (run with
PYTHONPATH=C:/t/mining-pilot/src and PYTHONDONTWRITEBYTECODE=1 so nothing is
written into the worktree). No network, no model, no run-dir writes.

Purpose: the controller caps each round's generated frontier at
max_frontier_size (60) and never records what the cap cut. Replaying generate()
with the cap lifted shows how many generated questions were never asked.

Validation (asserted):
* every recorded non-bridge round-2 input item is reproduced by the replay, and
  they are exactly the highest-ranked non-bridge items that fit under the cap;
* the round-2 replay generates no non-bridge item (the live run generated nothing).
Bridge targets are NOT reproduced exactly: generate() anchors a bridge on the
first chemical name in component order, and the order restored by
ResearchGraph.from_dict differs from the in-memory order of the live run. That
divergence is recorded, not hidden.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from t2pw.mining import frontier as fm
from t2pw.mining.claims import ClaimStore
from t2pw.mining.graph import ResearchGraph
from t2pw.mining.models import FrontierItem, GraphDelta

RUN = Path("C:/t/mining-pilot-runs/pilot-001-jasmonate-ath")
OUT = Path("C:/t/mining-pilot-runs/audit/frontier_replay.json")
BRIDGE = "bridge_search"


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def cfg(cap):
    return SimpleNamespace(topic="jasmonate biosynthesis", target_species="Arabidopsis thaliana",
                           budgets=SimpleNamespace(max_frontier_size=cap))


def by_kind(items):
    out = {}
    for i in items:
        out[i.kind] = out.get(i.kind, 0) + 1
    return dict(sorted(out.items()))


state = json.loads((RUN / "state.json").read_text(encoding="utf-8"))
graph = ResearchGraph.from_dict(state["graph"])   # graph hash unchanged by round 2
store = ClaimStore.from_dict(state["store"])        # no claims added in round 2
deltas = [GraphDelta.from_dict(d) for d in state["deltas"]]
r1_in = [FrontierItem.from_dict(d) for d in jl(RUN / "rounds/round_001/input_frontier.jsonl")]
r2_in = [FrontierItem.from_dict(d) for d in jl(RUN / "rounds/round_002/input_frontier.jsonl")]

r1_spent = {i.frontier_item_id for i in r1_in if i.exhausted}
uncapped = fm.generate(graph=graph, store=store, delta=deltas[0], config=cfg(10**6),
                       round_index=1, previous=r1_in, exhausted_ids=r1_spent)

rec_nb = [(i.kind, i.target) for i in r2_in if i.kind != BRIDGE and i.origin_round == 1]
rec_bridges = [i for i in r2_in if i.kind == BRIDGE]
carried = [i for i in r2_in if i.origin_round == 0]
rep_nb = [(i.kind, i.target) for i in uncapped if i.kind != BRIDGE]
rep_bridges = [i for i in uncapped if i.kind == BRIDGE]
slots = 60 - len(carried) - len(rec_bridges)
assert set(rec_nb) <= set(rep_nb), "a recorded non-bridge item is not reproduced"
assert rep_nb[:slots] == rec_nb, "recorded non-bridge items are not the top-ranked replay items"
cut = [i for i in uncapped if i.kind != BRIDGE and (i.kind, i.target) not in set(rec_nb)]

r2 = fm.generate(graph=graph, store=store, delta=deltas[1], config=cfg(60), round_index=2,
                 previous=r2_in, exhausted_ids=set(state["exhausted_frontier"]))
r2_nb = [i for i in r2 if i.kind != BRIDGE]
assert not r2_nb, "round-2 replay should generate no non-bridge item"

result = {
    "method": "frontier.generate() replayed on state.json with max_frontier_size lifted; frozen code @ 23e1ae83",
    "round1_carried_unexhausted_items": [i.target for i in carried],
    "round1_recorded_generated_items": len(r2_in) - len(carried),
    "round1_recorded_bridge_items": len(rec_bridges),
    "round1_recorded_nonbridge_items": len(rec_nb),
    "replay_uncapped_generated": len(uncapped),
    "replay_uncapped_by_kind": by_kind(uncapped),
    "replay_bridge_items": len(rep_bridges),
    "bridge_targets_recorded_not_in_replay": sorted({i.target for i in rec_bridges} - {i.target for i in rep_bridges}),
    "bridge_targets_replay_not_recorded": sorted({i.target for i in rep_bridges} - {i.target for i in rec_bridges}),
    "nonbridge_generated_uncapped": len(rep_nb),
    "cut_by_cap_count": len(cut),
    "cut_by_cap_by_kind": by_kind(cut),
    "cut_by_cap_items": [{"kind": i.kind, "target": i.target, "priority": i.priority} for i in cut],
    "round2_replay_nonbridge_items": len(r2_nb),
    "round2_replay_bridge_items_from_reordered_anchors": [i.target for i in r2 if i.kind == BRIDGE],
    "validation": "recorded non-bridge items == top-ranked replay non-bridge items (asserted); round-2 replay generates no non-bridge item (asserted); its bridge items exist only because the restored component order changes the anchor name",
}
OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "cut_by_cap_items"}, indent=1, ensure_ascii=False))
