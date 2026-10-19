"""G9 proof for the ORCH-747 graph kind-resolution fix: it FAILS on the base commit.

Merge gate 9 requires a claimed correction of pre-existing behaviour to carry a proof that
fails behaviourally at the base SHA and passes at the tip. Symbol absence is not proof, so
this does not look for a function or a constant: it imports ``draft_graph.py`` AS COMMITTED
at the chosen revision, builds a payload whose participants name nucleic acids and an
element collection, and reports the mis-typed duplicates and orphans in the graph it
returns. The base commit produces them; the tip produces none.

The committed copy is loaded from ``git show`` into a temporary module so the working tree
is never modified, stashed or checked out.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch747_graph_kind_base_proof.py [--rev HEAD]
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
TARGET_REL = "src/t2pw/pipeline/draft_graph.py"

PAYLOAD: Dict[str, Any] = {
    "pathway_context": {"name": "ara operon", "organism": "Escherichia coli"},
    "entities": {
        "species": [{"name": "Escherichia coli"}],
        "compounds": [{"name": "L-arabinose", "class": "compound"}],
        "proteins": [{"name": "araC protein", "class": "protein"}],
        "protein_complexes": [{"name": "RNA polymerase holoenzyme",
                               "class": "protein_complex"}],
        "nucleic_acids": [{"name": "A site", "class": "nucleic_acid"},
                          {"name": "Pc promoter", "class": "nucleic_acid"}],
        "element_collections": [{"name": "ara regulatory elements",
                                 "class": "element_collection"}],
    },
    "processes": {
        "interactions": [
            {"name": "araC protein binds A site", "entity_1": "araC protein",
             "relationship": "binding", "entity_2": "A site", "evidence": "q"},
            {"name": "araC protein inhibits Pc promoter", "entity_1": "araC protein",
             "relationship": "inhibition", "entity_2": "Pc promoter", "evidence": "q"},
            {"name": "polymerase binds the elements",
             "entity_1": "RNA polymerase holoenzyme", "relationship": "binding",
             "entity_2": "ara regulatory elements", "evidence": "q"},
        ],
        # Every declared entity must take part in SOME process, or it is an orphan for a
        # legitimate reason and the orphan count stops being evidence of mis-typing.
        "reactions": [
            {"name": "transcription", "inputs": ["Pc promoter"], "outputs": ["L-arabinose"],
             "modifiers": [{"entity": "RNA polymerase holoenzyme", "role": "catalyst"}],
             "evidence": "q"},
        ],
    },
}


def _load_at_revision(rev: str) -> Any:
    """``draft_graph`` as committed at ``rev``, imported without touching the tree."""

    out = subprocess.run(["git", "show", f"{rev}:{TARGET_REL}"], cwd=REPO,
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"cannot read {TARGET_REL} at {rev}: {out.stderr.strip()}")
    tmp = Path(tempfile.mkdtemp(prefix="orch747_base_")) / "draft_graph_base.py"
    tmp.write_text(out.stdout, encoding="utf-8")
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("draft_graph_base", tmp)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # @dataclass resolves its own module from sys.modules while the class body runs, so
    # the module must be registered BEFORE exec_module or the decorator raises.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inspect(build) -> Dict[str, Any]:
    graph = build(PAYLOAD)
    deg: Dict[str, int] = collections.Counter()
    for edge in graph.edges:
        deg[edge.source] += 1
        deg[edge.target] += 1
    by_label: Dict[str, List[str]] = collections.defaultdict(list)
    for node in graph.nodes:
        if node.kind != "interaction":
            by_label[node.label].append(node.kind)
    return {
        "kinds": dict(collections.Counter(n.kind for n in graph.nodes)),
        "duplicated_across_kinds": {k: sorted(v) for k, v in by_label.items() if len(v) > 1},
        "orphan_entity_nodes": sorted(
            n.label for n in graph.nodes
            if n.kind != "interaction" and deg.get(n.id, 0) == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="HEAD", help="the base revision to prove against")
    args = ap.parse_args()

    base = _load_at_revision(args.rev)
    base_result = _inspect(base.build_draft_graph)

    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from t2pw.pipeline.draft_graph import build_draft_graph as tip_build
    tip_result = _inspect(tip_build)

    print(f"BASE ({args.rev}:{TARGET_REL})")
    print(json.dumps(base_result, indent=1))
    print()
    print("TIP (working tree)")
    print(json.dumps(tip_result, indent=1))
    print()

    base_broken = bool(base_result["duplicated_across_kinds"]) or bool(
        base_result["orphan_entity_nodes"])
    tip_clean = (not tip_result["duplicated_across_kinds"]
                 and not tip_result["orphan_entity_nodes"])

    print(f"base fails behaviourally : {base_broken}")
    print(f"tip passes               : {tip_clean}")
    if base_broken and tip_clean:
        print("VERDICT: G9 satisfied -- the base mis-types and orphans these entities, "
              "the tip does not.")
        return 0
    print("VERDICT: NOT a valid G9 proof for this change.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
