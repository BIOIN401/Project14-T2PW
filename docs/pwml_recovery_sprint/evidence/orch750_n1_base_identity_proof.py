"""ORCH-750 (dispatched as ORCH-750; renumbered because another session had already claimed 748 in the ledger for the pathway-mining lane) identity proof: at N=1 the tip's ``run_extraction_pipeline`` IS the base's.

Ensembling is a new capability (G9: labelled new acceptance tests, no fabricated base
failure). What the change must PRESERVE is that with ``T2PW_EXTRACTION_DRAWS`` unset (or
``1``) Stage 1 behaves byte-identically to the committed base: the same single
``_run_json_stage`` call, the same keyword set, the same values, the same rendered user
prompts, the same return value passed through untouched. This script proves that by
behaviour, not by symbol inspection.

``pipeline.py`` AS COMMITTED at ``--rev`` is loaded from ``git show`` into a temporary
module -- the working tree is never checked out, stashed or reset -- and both modules get
the same recording stub for ``_run_json_stage``. Every captured call and return is compared.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch750_n1_base_identity_proof.py \
        [--rev HEAD] [--out docs/pwml_recovery_sprint/evidence/orch750_n1_identity_verdict.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]
TARGET_REL = "src/t2pw/pipeline/pipeline.py"

TEXT = "Glucose is phosphorylated by hexokinase to glucose 6-phosphate using ATP."
CONTEXT = {"name": "glycolysis", "organism": "Homo sapiens", "document_type": "primary_research"}
CANNED = {
    "entities": {"compounds": [{"name": "ATP", "class": "compound"}]},
    "processes": {"reactions": [{"name": "r1", "inputs": ["ATP"], "outputs": ["ADP"], "evidence": "q"}]},
}
ARG_SETS: List[Dict[str, Any]] = [
    {},
    {"pathway_context": CONTEXT, "pathway_scope": "glycolysis", "user_task_context": "ctx",
     "max_attempts": 3, "temperature": 0.2, "max_tokens": 64000},
]
ENV_CASES: List[Optional[str]] = [None, "1"]


def _load_at_revision(rev: str) -> Any:
    out = subprocess.run(["git", "show", f"{rev}:{TARGET_REL}"], cwd=REPO,
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"cannot read {TARGET_REL} at {rev}: {out.stderr.strip()}")
    tmp = Path(tempfile.mkdtemp(prefix="orch750_base_")) / "pipeline_base.py"
    tmp.write_text(out.stdout, encoding="utf-8")
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("pipeline_base", tmp)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # @dataclass resolves its module from sys.modules while the class body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stub:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.returned: List[Any] = []

    def __call__(self, **kwargs: Any) -> Any:
        rendered = dict(kwargs)
        build = rendered.pop("build_user_prompt")
        rendered["build_user_prompt(None, None)"] = build(None, None)
        rendered["build_user_prompt('{}', 'err')"] = build("{}", "err")
        self.calls.append(rendered)
        result = (json.loads(json.dumps(CANNED)), [{"attempt": 1, "raw": json.dumps(CANNED)}])
        self.returned.append(result)
        return result


def _observe(module: Any, env: Optional[str], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if env is None:
        os.environ.pop("T2PW_EXTRACTION_DRAWS", None)
    else:
        os.environ["T2PW_EXTRACTION_DRAWS"] = env
    stub = Stub()
    original = module._run_json_stage
    module._run_json_stage = stub
    try:
        result = module.run_extraction_pipeline(TEXT, **kwargs)
    finally:
        module._run_json_stage = original
    return {
        "calls": stub.calls,
        "result": result,
        "result_is_stage_return_object": bool(stub.returned) and result is stub.returned[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default=None, help="write the JSON verdict here")
    args = ap.parse_args()

    base = _load_at_revision(args.rev)
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    import t2pw.pipeline.pipeline as tip

    cases = []
    identical = True
    for env in ENV_CASES:
        for kwargs in ARG_SETS:
            b = _observe(base, env, kwargs)
            t = _observe(tip, env, kwargs)
            same = json.dumps(b, sort_keys=True, default=str) == json.dumps(t, sort_keys=True, default=str)
            identical &= same and len(t["calls"]) == 1 and t["result_is_stage_return_object"]
            cases.append({
                "env_T2PW_EXTRACTION_DRAWS": env,
                "kwargs": {k: (v if k != "pathway_context" else "<context>") for k, v in kwargs.items()},
                "base_calls": len(b["calls"]),
                "tip_calls": len(t["calls"]),
                "call_keyword_sets_equal": [sorted(c) for c in b["calls"]] == [sorted(c) for c in t["calls"]],
                "captured_observation_json_equal": same,
                "tip_passes_stage_return_through": t["result_is_stage_return_object"],
            })

    verdict = {
        "base_rev": args.rev,
        "base_file": TARGET_REL,
        "base_has_draws_parameter": "draws" in inspect.signature(base.run_extraction_pipeline).parameters,
        "tip_has_draws_parameter": "draws" in inspect.signature(tip.run_extraction_pipeline).parameters,
        "cases": cases,
        "n1_behaviourally_identical": identical,
    }
    print(json.dumps(verdict, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print("VERDICT:", "N=1 IDENTICAL to base" if identical else "N=1 DIFFERS from base")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
