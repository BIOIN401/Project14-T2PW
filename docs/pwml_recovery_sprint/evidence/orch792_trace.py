"""ORCH-792 step 1: offline Stage-2 -> normalise -> Stage-6 replay with a protein-row removal tracer.

usage: python orch792_trace.py <tree_root> <workdir> [leg ...]
Reuses tests/orch762_harness.py (no PathBank DB, no LLM, HTTP answered 404 unless recorded). Stage-6
protein answers are the leg's own recorded resolutions (``recorded_resolutions(final_mapped)``), the
ORCH-759a technique. Every function of ``map_ids`` and ``process_normalizer`` is wrapped: a call that
returns with fewer protein rows (by name) than it entered with is logged with the wrapped call stack,
so the innermost logged frame is the deleting step. Rows lost in map_payload but in no wrapped callee
are inline in map_payload. Read-only on the legs; writes only under <workdir>.
"""
import functools
import inspect
import json
import shutil
import sys
from pathlib import Path

tree = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(tree / "src"), str(tree / "tests")]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import orch762_harness as h  # noqa: E402
from t2pw.mapping import map_ids  # noqa: E402
from t2pw.pipeline import process_normalizer  # noqa: E402

W = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.claude/worktrees")
LEGS = {
    "chlorophyll": ("orch785run", "ORCH-785/legs/chlorophyll-strict/2026-10-28_1743/papers/PMC5130987/strict"),
    "wnt_ca": ("orch785run", "ORCH-785/legs/wnt_ca-strict/2026-10-28_1910/papers/PMC10425814/strict"),
    "wnt_pcp": ("orch784run", "ORCH-784/legs/wnt_pcp-strict/2026-10-27_2309/papers/PMC10425814/strict"),
    "menaquinone": ("orch784run", "ORCH-784/legs/menaquinone-strict/2026-10-28_0208/papers/PMC12421875/strict"),
}

STACK = []
LOG = []


def names_of(obj):
    if not isinstance(obj, dict):
        return None
    ents = obj.get("entities") if isinstance(obj.get("entities"), dict) else obj
    rows = ents.get("proteins") if isinstance(ents, dict) else None
    if not isinstance(rows, list):
        return None
    return {str(r.get("name")) for r in rows if isinstance(r, dict)}


def wrap(mod, name, fn):
    @functools.wraps(fn)
    def inner(*a, **k):
        args = [x for x in list(a) + list(k.values()) if isinstance(x, dict)]
        before = [(x, names_of(x)) for x in args]
        before = [(x, n) for x, n in before if n is not None]
        STACK.append(name)
        try:
            out = fn(*a, **k)
        finally:
            STACK.pop()
        for x, n in before:
            after = names_of(x)
            if name == "map_payload" and isinstance(out, dict):
                after = names_of(out.get("payload"))
            lost = sorted(n - (after or set()))
            if lost:
                LOG.append({"fn": f"{mod}.{name}", "stack": list(STACK), "lost": lost})
        return out
    return inner


for mod in (map_ids, process_normalizer):
    for name, fn in list(vars(mod).items()):
        if inspect.isfunction(fn) and fn.__module__ == mod.__name__:
            setattr(mod, name, wrap(mod.__name__.rsplit(".", 1)[-1], name, fn))

for leg in sys.argv[3:] or list(LEGS):
    run, rel = LEGS[leg]
    d = W / run / "runs_papers" / rel
    lw = work / leg
    lw.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(W / run / "data" / "id_mapping_cache.json", lw / "id_mapping_cache.json")
    merged = json.loads((d / "merged_payload.json").read_text(encoding="utf-8"))
    fm = json.loads((d / "final_mapped.json").read_text(encoding="utf-8"))
    LOG.clear()
    http = h.ReplayHttp({})
    with h.Doors(http, resolutions=h.recorded_resolutions(fm)):
        out = h.replay(merged, lw)
    st6 = out["stage6"]["payload"]
    print(f"\n######## {leg}  tree={tree.name}")
    print("merged proteins:", sorted(names_of(merged)))
    print("stage6 proteins:", sorted(names_of(st6)))
    print("quarantined_proteins:", sorted({q.get("name") for q in st6.get("quarantined_proteins") or []}))
    for e in LOG:
        print("LOST", e["fn"], "via", " > ".join(e["stack"]) or "(top)", "->", e["lost"])
