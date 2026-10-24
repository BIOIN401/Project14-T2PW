"""ORCH-769c G9 proof on the real legs: the review diagram, exactly as
``initialize_refinement_review_state`` draws it (``build_draft_graph`` then
``render_draft_graph_to_png_bytes(..., dpi=100)``), on two saved WNT PCP payloads.

* ORCH-767 ``wnt_pcp-strict/2026-10-24_0131``: 0 reactions, 0 transports, 18 interactions.
  Base raises "DraftGraph has no reaction nodes"; tip returns ``b""``.
* ORCH-759 ``wnt_pcp-strict/2026-10-22_1432``: a leg that drew a diagram and wrote a PWML.
  Its PNG must be byte-identical at base and tip.

Offline, no LLM, reads the legs and writes only ``<out.json>``. Usage (PYTHONPATH = <tree>/src)::

    bounded_run.py ... -- python g9_render_legs.py <tree>/src <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(src))
import t2pw  # noqa: E402

assert Path(t2pw.__file__).resolve().parent == src / "t2pw", f"wrong tree: {t2pw.__file__}"
from t2pw.pipeline.draft_graph import build_draft_graph  # noqa: E402
from t2pw.pipeline.draft_graph_render import render_draft_graph_to_png_bytes  # noqa: E402

PRIMARY = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
LEGS = {
    "ORCH-767/wnt_pcp-strict/2026-10-24_0131": PRIMARY / ".claude/worktrees/orch767/runs_papers/ORCH-767/legs/"
    "wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict/final_mapped.json",
    "ORCH-759/wnt_pcp-strict/2026-10-22_1432": PRIMARY / "runs_papers/ORCH-759/legs/"
    "wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict/final_mapped.json",
}

doc = {"src": str(src), "t2pw": t2pw.__file__, "legs": {}}
for leg, path in LEGS.items():
    raw = path.read_bytes()
    payload = json.loads(raw)
    graph = build_draft_graph(payload).to_dict()
    rec = {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "processes": {k: len(v) for k, v in payload["processes"].items() if isinstance(v, list)},
        "node_kinds": sorted({n["kind"] for n in graph["nodes"]}),
    }
    try:
        png = render_draft_graph_to_png_bytes(graph, dpi=100)
        rec.update(outcome="returned", bytes=len(png), png_sha256=hashlib.sha256(png).hexdigest())
    except Exception as exc:  # noqa: BLE001 -- the base behaviour under test
        rec.update(outcome="raised", error=f"{type(exc).__name__}: {exc}")
    doc["legs"][leg] = rec
    print(leg, json.dumps(rec, ensure_ascii=False))

out.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
