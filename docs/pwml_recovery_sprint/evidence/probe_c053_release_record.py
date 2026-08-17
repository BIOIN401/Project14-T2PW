"""C-053 SS0: is the FROZEN release record actually there on a passing strict export?

D-038 SS2 makes the whole of C-053 rest on ONE expression --
``pwml_result["quarantine_report"]["release"]`` -- and the chain proving it reaches
the batch strict PASS path was established STATICALLY. D-033 forbids inheriting an
unmeasured claim, so this runs the production export and reads the value.

It measures the real thing rather than a stand-in: ``run_pwml_export`` is the
function ``_generate_pwml_from_refinement_working_json`` (EP3) tail-calls, and its
return value is what the app stores in ``session_state["pwml_export_result"]`` --
which is exactly what ``driver._find_pwml_result`` hands to ``_add_strict_artifacts``
and ``_finalize_pwml_export``. Both of the boundary's dispositions are measured:

``fresh``    ``run_pwml_export`` was given no stored decision, so it runs
             ``quarantine_and_close`` itself.
``carried``  the app shape: ``run_quarantine_boundary`` decides first and the
             decision is handed in, so ``decision_matches`` reuses it. This is the
             disposition a batch strict leg actually takes.

Nothing is written into the checkout: ``project_root`` is a caller-supplied
temporary directory, and the output path is resolved BEFORE anything else runs
(F-045). Read-only with respect to ``src/`` -- it imports and calls, nothing more.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ``t2pw.app.streamlit_app`` executes a Streamlit script at import. The stub is
# the established convention (tests/test_prefreeze_third_export_seam.py:52-63);
# neither ``run_pwml_export`` nor ``run_quarantine_boundary`` makes a Streamlit
# call that matters here -- the second only stores its decision in session state,
# and this probe reads the decision from its RETURN value instead.
if "streamlit" not in sys.modules:
    _st = MagicMock(name="streamlit")
    _st.columns.side_effect = lambda spec=1, **__: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    _st.tabs.side_effect = lambda labels, **__: [MagicMock() for _ in labels]
    _st.form_submit_button.return_value = False
    _st.checkbox.return_value = False
    _st.session_state = MagicMock()
    _st.session_state.get.return_value = None
    _st.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = _st


def _payload() -> Dict[str, Any]:
    """The pinned payload every export gate already accepts.

    Imported rather than re-typed so a refusal here can only be the seam.
    """

    from test_prefreeze_third_export_seam import _raw_extraction_payload

    return _raw_extraction_payload()


def _observe(result: Dict[str, Any]) -> Dict[str, Any]:
    """What SS0 has to answer: is it there, is it non-empty, what shape is it."""

    report = result.get("quarantine_report")
    release = report.get("release") if isinstance(report, dict) else None
    seen: Dict[str, Any] = {
        "export_ok": bool(result.get("ok")),
        "export_error": str(result.get("error") or ""),
        "quarantine_report_present": isinstance(report, dict) and bool(report),
        "quarantine_reused": bool(result.get("quarantine_reused")),
        "release_present": isinstance(release, dict) and bool(release),
        "release_type": type(release).__name__,
        "release_keys": sorted(release) if isinstance(release, dict) else [],
        "release_value": release if isinstance(release, dict) else None,
    }
    return seen


def _fresh(payload: Dict[str, Any], root: Path) -> Dict[str, Any]:
    from t2pw.app.streamlit_app import run_pwml_export

    (root / "outputs").mkdir(parents=True, exist_ok=True)
    return run_pwml_export(
        payload,
        pathway_name="Caffeine demethylation",
        pathway_description="",
        pathway_subject="Metabolic",
        project_root=root,
        ref_path=ROOT / "reference" / "PW000001.pwml",
        strict_db=False,
    )


def _carried(payload: Dict[str, Any], root: Path) -> Dict[str, Any]:
    from t2pw.app.streamlit_app import run_pwml_export, run_quarantine_boundary

    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    decision = run_quarantine_boundary(payload, strict_db=False, outputs_dir=outputs)
    return run_pwml_export(
        decision["payload"],
        pathway_name="Caffeine demethylation",
        pathway_description="",
        pathway_subject="Metabolic",
        project_root=root,
        ref_path=ROOT / "reference" / "PW000001.pwml",
        strict_db=False,
        quarantine_report=decision["quarantine_report"],
        quarantine_artifacts=decision["quarantine_artifacts"],
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="where to write the measurement")
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()  # BEFORE anything runs (F-045)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import t2pw

    record: Dict[str, Any] = {
        "probe": "C-053 SS0 -- pwml_result['quarantine_report']['release']",
        "t2pw_file": t2pw.__file__,
        "expected_tree": str(ROOT),
        "legs": {},
    }

    with tempfile.TemporaryDirectory(prefix="c053probe") as tmp:
        for name, run in (("fresh", _fresh), ("carried", _carried)):
            leg_root = Path(tmp) / name
            leg_root.mkdir(parents=True, exist_ok=True)
            record["legs"][name] = _observe(run(_payload(), leg_root))

    record["verdict"] = (
        "PRESENT_AND_NON_EMPTY"
        if all(
            leg["export_ok"] and leg["release_present"] for leg in record["legs"].values()
        )
        else "ABSENT_OR_EMPTY"
    )
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"t2pw.__file__ = {t2pw.__file__}")
    print(f"wrote {out_path}  (exists={out_path.is_file()})")
    print(json.dumps({k: v for k, v in record.items() if k != "legs"}, indent=2))
    for name, leg in record["legs"].items():
        print(f"--- {name}: ok={leg['export_ok']} release_present={leg['release_present']}")
        print(f"    error={leg['export_error'][:200]}")
        print(f"    keys={leg['release_keys']}")
    return 0 if record["verdict"] == "PRESENT_AND_NON_EMPTY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
