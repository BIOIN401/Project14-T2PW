"""ORCH-754a: goldens proving what a model reads is the base's bytes (ruling 2026-10-20).

``reaction_summary.txt`` is not only for people: the curator (``pathway_curator.py``) and the
enrichment agent (``gap_resolver.py``) receive it. The product owner ruled that every
model input must stay byte-identical to base ``f9892cdf`` for the same payload, and gain
no support mark or merge note at N>1. This script renders the reaction summary of two
fixture payloads with the BASE modules -- ``draft_graph.py`` and ``reaction_summary.py``
as committed at ``--rev``, loaded from ``git show`` -- and writes the text as goldens that
``tests/test_stage_one_paper_dictionary.py::test_what_a_model_reads_is_the_bases_bytes``
compares the tip against:

* ``chlorophyll_draw1_payload.json`` -- one saved Stage-1 draw, i.e. an N=1 payload;
* ``chlorophyll_pooled_payload.json`` -- the TIP's pooled merge of the three chlorophyll
  draws with the paper text, i.e. exactly what an N=3 run hands the next stage.

Both are written with their interactions removed: interaction rows belong to lane
ORCH-754b, and a golden of them would couple the two lanes' merges.

Usage::

    PYTHONPATH=src python docs/pwml_recovery_sprint/evidence/orch754a_llm_input_base_golden.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.pipeline.stage_one_ensemble import merge_draws  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "orch754a"
DRAWS = [HERE / "orch752_chlorophyll_payloads" / f"control__run{i}__payload.json" for i in (1, 2, 3)]
PAPER = HERE / "orch754a_inputs" / "papers" / "PMC5130987.txt"


def _load(rev: str, rel: str, name: str) -> Any:
    text = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout
    path = Path(tempfile.mkdtemp(prefix="orch754a_golden_")) / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # @dataclass resolves its module while the class body runs
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _reactions_only(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(payload))
    out.get("processes", {}).pop("interactions", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="f9892cdf")
    args = ap.parse_args()

    graph = _load(args.rev, "src/t2pw/pipeline/draft_graph.py", "draft_graph_base")
    summary = _load(args.rev, "src/t2pw/pipeline/reaction_summary.py", "reaction_summary_base")
    draws = [json.loads(p.read_text(encoding="utf-8")) for p in DRAWS]
    pooled, _report = merge_draws(draws, paper_text=PAPER.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"base_rev": args.rev, "files": {}}
    for stem, payload in (("chlorophyll_draw1", _reactions_only(draws[0])), ("chlorophyll_pooled", _reactions_only(pooled))):
        payload_path = OUT / f"{stem}_payload.json"
        payload_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        text = summary.generate_reaction_summary(graph.build_draft_graph(payload), {})
        golden = OUT / f"{stem}_reaction_summary.base.txt"
        golden.write_text(text, encoding="utf-8")
        record["files"][golden.name] = {"payload": payload_path.name, "chars": len(text),
                                        "sha256_lf": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                        "reactions": len(payload["processes"]["reactions"])}
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
