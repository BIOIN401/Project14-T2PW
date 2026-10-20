"""ORCH-754a round 4: ``map_payload`` is base-identical for every non-Stage-6 call (D-104 am. 5).

The authorisation: ``map_payload`` gains ``merge_identity_duplicates=False``, and at False
its behaviour and output stay byte-identical to base for every caller. This script loads
``src/t2pw/mapping/map_ids.py`` AS COMMITTED at ``--rev`` from ``git show`` into its own
module, runs it and the tip offline on the same fixture payloads -- every external door
shut, as ``tests/test_c073_identity_admission.run_offline`` does -- and compares the full
``{"payload", "report"}`` results as canonical JSON:

* tip default call vs base, and tip ``merge_identity_duplicates=False`` vs base, inside a
  run whose Stage 1 POOLED 3 draws (the condition under which rule 3 would act);
* tip ``merge_identity_duplicates=True`` vs base inside a run that did NOT pool, i.e. N=1
  at Stage 6.

Fixtures: one saved chlorophyll draw (N=1 shape), the delivered pooled chlorophyll merge,
and a protoheme/heme b payload whose two ferrochelatase rows would merge under rule 3.

Usage::

    PYTHONPATH=src python docs/pwml_recovery_sprint/evidence/orch754a_map_payload_base_identity_proof.py \\
        [--rev f9892cdf] [--out docs/pwml_recovery_sprint/evidence/orch754a_map_payload_base_identity_verdict.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from t2pw.mapping import map_ids as tip  # noqa: E402
from t2pw.pipeline import extraction_diagnostics as diag  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import BOUNDARY_STAGE1_ENSEMBLE, ENV_VAR  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "orch754a"
FERRO = {
    "entities": {"compounds": [{"name": n} for n in ("protoporphyrin IX", "Fe2+", "protoheme", "heme b")]},
    "processes": {"reactions": [
        {"name": "ferro A", "inputs": ["protoporphyrin IX", "Fe2+"], "outputs": ["protoheme"], "evidence": "q1"},
        {"name": "ferro B", "inputs": ["protoporphyrin IX", "Fe2+"], "outputs": ["heme b"], "evidence": "q3"}]},
}


def _base(rev: str) -> Any:
    text = subprocess.run(["git", "show", f"{rev}:src/t2pw/mapping/map_ids.py"], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout
    path = Path(tempfile.mkdtemp(prefix="orch754a_mapids_")) / "map_ids_base.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("map_ids_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["map_ids_base"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(module: Any, payload: Dict[str, Any], tmp: Path, **kwargs: Any) -> str:
    with patch.dict(os.environ, {"T2PW_SPECIES_LLM": "0", "T2PW_SPECIES_NCBI": "0"}), \
            patch.object(module.PathBankDbResolver, "from_env", classmethod(lambda cls, overrides=None: None)), \
            patch.object(module, "_ai_protein_synonym_lookup", return_value=[]), \
            patch.object(module.HttpClient, "get", side_effect=RuntimeError("network call during an offline run")):
        result = module.map_payload(deepcopy(payload), cache_path=tmp / "cache.json", id_source="db",
                                    use_cache=False, **kwargs)
    return json.dumps(result, sort_keys=True, ensure_ascii=False, default=str)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="f9892cdf")
    ap.add_argument("--out", default=str(HERE / "orch754a_map_payload_base_identity_verdict.json"))
    args = ap.parse_args()
    base = _base(args.rev)
    fixtures = {
        "chlorophyll_draw1": json.loads((FIXTURES / "chlorophyll_draw1_payload.json").read_text(encoding="utf-8")),
        "chlorophyll_pooled": json.loads((FIXTURES / "chlorophyll_pooled_payload.json").read_text(encoding="utf-8")),
        "protoheme_heme_b": FERRO,
    }
    os.environ[ENV_VAR] = "3"
    cases, identical = [], True
    for name, payload in fixtures.items():
        tmp = Path(tempfile.mkdtemp(prefix="orch754a_mp_"))
        for pooled in (True, False):
            diag.activate(run_id=f"proof-{name}")
            if pooled:
                diag.current().record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_ENSEMBLE, attempts=3,
                                               ensemble={"draws_succeeded": 3})
            try:
                reference = _run(base, payload, tmp)
                runs = {"tip_default": _run(tip, payload, tmp), "tip_false": _run(tip, payload, tmp, merge_identity_duplicates=False)}
                if not pooled:
                    runs["tip_true_unpooled"] = _run(tip, payload, tmp, merge_identity_duplicates=True)
            finally:
                diag.deactivate()
            for label, got in runs.items():
                same = got == reference
                identical &= same
                cases.append({"fixture": name, "run_pooled_3_draws": pooled, "call": label,
                              "identical_to_base": same, "chars": len(got)})
    verdict = {"base_rev": args.rev, "file": "src/t2pw/mapping/map_ids.py", "cases": cases,
               "non_stage6_and_n1_base_identical": identical}
    Path(args.out).write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    print("VERDICT:", "map_payload BASE-IDENTICAL" if identical else "map_payload DIFFERS from base")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
