"""ORCH-759e, D-104 amendment 2: what of the Stage-6 mapping report the interactive curator's prompt sees.

Runs the late-enzyme fixture of ``tests/test_orch759e_rewrap_after_phase2.py`` through ``map_payload``
and writes ``compact_mapping_misses(report)`` -- the curator's LLM-bound view of the report -- and the
report's key paths to ``argv[1]``. Run at base and at tip; diff the two files with ``llm_view_diff.py``.
"""

import json
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(WT / "src"), str(WT / "tests")]

import test_orch759e_rewrap_after_phase2 as T  # noqa: E402
from t2pw.curation.interactive_curator import compact_mapping_misses  # noqa: E402


def key_paths(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield f"{prefix}/{key}"
            yield from key_paths(child, f"{prefix}/{key}")
    elif isinstance(value, list):
        for child in value:
            yield from key_paths(child, f"{prefix}/[]")


out = {}
scratch = Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)
for label, payload, main, phase2 in (
    ("late_enzyme", T._payload(["lactate dehydrogenase"], ["lactate dehydrogenase"]), {},
     {"lactate dehydrogenase": T._mapped("lactate dehydrogenase", "P52643")}),
    ("cpo_shape", T._payload(["CPO", "CPO1"], ["CPO", "CPO1"], mirror_modifiers=True), {},
     {"CPO": T._mapped("CPO", "Q9LR75", T.CPO_CANDIDATE)}),
):
    report = T._run(payload, scratch, main=main, phase2=phase2)["report"]
    out[label] = {"llm_view": compact_mapping_misses(report), "report_key_paths": sorted(set(key_paths(report)))}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True, default=str), encoding="utf-8")
print("written", sys.argv[1])
