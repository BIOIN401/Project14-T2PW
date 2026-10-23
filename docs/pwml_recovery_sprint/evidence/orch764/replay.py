"""ORCH-764: offline replay of pre-freeze compound resolution on the two ORCH-759 3-draw legs, scored
against the certified gold exactly as ``orch756_gold_accuracy.py`` scores level 2.

Offline: no model, no network, no database. Run once with the base code and once with the tip:

    PYTHONPATH=<worktree>/src python -u docs/pwml_recovery_sprint/evidence/orch764/replay.py

Per leg: the recorded ``final_mapped.json`` with its pre-freeze resolution undone
(``make_fixture.pre_prefreeze``: the one recorded rename reverted, the stamps removed), then
``resolve_compounds_prefreeze`` re-run over it with the PathBank answers the leg recorded (the
fixture's row 874; every other database-consulted row of both legs recorded ``unmatched``, which an
empty answer reproduces) and the production default name index. The base replay must reproduce the
recorded payload exactly; that equality is printed as the fidelity check.

Scored: ``R.score_leg(case, payload, [], text, expected_species=case.species)`` on the recorded
Stage-1 payload, the recorded final payload and the replayed final payload.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORKTREE / "tests"))

from make_fixture import PRIMARY, pre_prefreeze, sha  # noqa: E402
from test_orch764_class_term_not_narrowed import FIXTURE, FixtureCompoundDb  # noqa: E402

import t2pw.pwml.compound_resolution as cr  # noqa: E402
from t2pw.bench.eval_v2 import reactions as R  # noqa: E402
from t2pw.bench.eval_v2.gold import load_fulltext, load_supplement  # noqa: E402
from t2pw.pwml.prefreeze_resolution import resolve_compounds_prefreeze  # noqa: E402

FULLTEXT = Path("C:/t/eval-fulltext")
LEGS = {
    "menaquinone": ("PMC12421875", "runs_papers/ORCH-759/legs/menaquinone-strict/2026-10-22_1148/papers/PMC12421875/strict"),
    "enterobactin": ("PMC12096016", "runs_papers/ORCH-759/legs/enterobactin-strict/2026-10-22_1037/papers/PMC12096016/strict"),
}


def score(case, payload, text) -> str:
    t1 = R.score_leg(case, payload, [], text, expected_species=case.species)["table1_paper_extraction"]
    return (f"P {t1['precision']:.3f}  R {t1['recall']:.3f}  F1 {t1['f1']:.3f}  "
            f"matched {len(t1['gold_reactions_matched'])}/{t1['gold_reactions']} "
            f"{sorted(t1['gold_reactions_matched'])}")


def main() -> int:
    print("compound_resolution:", cr.__file__, "| D-109 class gate present:", hasattr(cr, "is_class_participant_term"))
    cases = load_supplement()
    for name, (pid, rel) in LEGS.items():
        leg = PRIMARY / rel
        case = cases[pid]
        text = load_fulltext(case, FULLTEXT)
        assert text is not None, pid
        final = json.loads((leg / "final_mapped.json").read_text(encoding="utf-8"))
        stage1 = json.loads((leg / "stage1_payload.json").read_text(encoding="utf-8"))
        replay, renames = pre_prefreeze(final)
        summary = resolve_compounds_prefreeze(
            replay, db_resolver=FixtureCompoundDb(copy.deepcopy(FIXTURE["pathbank_compounds"])), report={})
        same = json.dumps(replay, sort_keys=True) == json.dumps(final, sort_keys=True)
        print(f"\n=== {name} {pid}  leg {rel}")
        print(f"final_mapped.json sha256 {sha(leg / 'final_mapped.json')}")
        print(f"recorded pre-freeze renames undone: {renames}")
        print(f"replay rename_map: {summary['rename_map']}")
        print(f"replay reproduces the recorded final_mapped.json exactly: {same}")
        for row in replay["entities"]["compounds"]:
            if (row.get("db_match") or {}).get("status") == "matched":
                print(f"  DB-matched row: name {row['name']!r} raw_name {row.get('raw_name')!r} "
                      f"db_status {row.get('db_status')!r} ids {sorted((row.get('mapped_ids') or {}))}")
        print(f"  stage1 (recorded)   {score(case, stage1, text)}")
        print(f"  final  (recorded)   {score(case, final, text)}")
        print(f"  final  (replayed)   {score(case, replay, text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
