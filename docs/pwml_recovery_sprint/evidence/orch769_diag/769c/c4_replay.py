"""ORCH-769c offline replay: the exact functions the app's review-init calls, on the saved legs.

No LLM, no network, writes nothing. Run from C:/t/bt-769c through bounded_run.py.
"""
import json
from collections import Counter

from t2pw.pipeline.draft_graph import build_draft_graph
from t2pw.pipeline.draft_graph_render import render_draft_graph_to_png_bytes
from t2pw.pipeline.qa_graph import generate_qa_report
from t2pw.pwml.ir import build_pwml_ir

LEGS = {
    "767": r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees/orch767/runs_papers/ORCH-767/legs/wnt_pcp-strict/2026-10-24_0131/papers/PMC10425814/strict/final_mapped.json",
    "759": r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict/final_mapped.json",
}


class NoDb:
    """Stub resolver: never connects (D-015 / no-network rule)."""

    last_error = "offline replay stub"

    def available(self):
        return False


for tag, path in LEGS.items():
    print("=" * 20, tag, path)
    payload = json.load(open(path, encoding="utf-8"))
    print("processes", {k: len(v) for k, v in payload["processes"].items() if isinstance(v, list)})
    graph = build_draft_graph(payload)
    gd = graph.to_dict()
    print("draft graph node kinds", dict(Counter(n.get("kind") for n in gd.get("nodes", []))))
    print("draft graph edges", len(gd.get("edges", [])))
    try:
        png = render_draft_graph_to_png_bytes(gd, dpi=100)
        print("RENDER ok bytes", len(png))
    except Exception as exc:  # noqa: BLE001
        print("RENDER RAISED", type(exc).__name__, exc)
    try:
        qa = generate_qa_report(graph, payload)
        print("QA ok keys", sorted(qa)[:12])
    except Exception as exc:  # noqa: BLE001
        print("QA RAISED", type(exc).__name__, exc)
    try:
        ir, rep = build_pwml_ir(payload, strict_db=True, db_resolver=NoDb())
        print("IR processes", {k: len(v) for k, v in ir.get("processes", {}).items()})
        print("IR entities", {k: len(v) for k, v in ir.get("entities", {}).items()})
        print("IR edges", len(ir.get("edges", [])), "process_visualizations", len(ir.get("process_visualizations", [])))
        errs = rep.get("errors", [])
        print("IR errors", len(errs), [e.get("code", e.get("message")) if isinstance(e, dict) else e for e in errs][:10])
        withheld = [w.get("message") for w in rep.get("warnings", []) if isinstance(w, dict) and "withheld" in str(w.get("message"))]
        print("IR interactions withheld", len(withheld))
        for w in withheld[:20]:
            print("   ", w)
    except Exception as exc:  # noqa: BLE001
        print("IR RAISED", type(exc).__name__, exc)
