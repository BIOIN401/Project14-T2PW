"""C-011: the canonical-freeze seam is a PURE MOVE, proved by behaviour.

``evidence/c011_freeze_seam_before.json`` records what the freeze block DID at
BASE, where it was still inline inside ``run_post_pipeline_sbml_artifacts``: it
is built by executing the BASE blob's OWN statements -- lifted out of the BASE
AST by line span, never retyped -- over all 39 legs of the frozen baseline
cohort. The tests rebuild that document from the working tree's extracted
``freeze_canonical_payload`` and compare BYTES, so a seam that kept the payload
while dropping a gate report, changing ``sbml_input_source`` or replacing object
sharing with a copy fails here.

Regenerate, byte-identically, from any checkout of this branch:
    .venv/Scripts/python.exe tests/test_c011_freeze_seam_golden_equivalence.py
"""

from __future__ import annotations

import ast, copy, hashlib, json, subprocess, sys, tempfile, textwrap  # noqa: E401
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (ROOT / "src", ROOT / "tests") if str(p) not in sys.path]

from t2pw.paths import PROJECT_ROOT  # noqa: E402
from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE, coerce_mode, is_research  # noqa: E402
from t2pw.pipeline.gate_reports import (  # noqa: E402
    CANONICAL_PAYLOAD_KEY, PHASE_FINAL_PRE_EXPORT, payload_sha256, stamp_report)
from t2pw.pipeline.process_normalizer import (  # noqa: E402
    GateValidationError, run_strict_post_normalization_gates)
from t2pw.pipeline.strict_quarantine import (  # noqa: E402
    quarantine_and_close, quarantine_review_flags, write_quarantine_artifacts)
import test_strict_quarantine_real_artifact_replay as replay  # noqa: E402

APP_REL = "src/t2pw/app/streamlit_app.py"
ORIGIN_SHA = "9e1b9abe7ba8a1a228558fd03ca6c394cc22c31e"
BASE_SHA = "72ee20f54b713c8477dc28d38f8053141f239307"
ORCHESTRATOR = "run_post_pipeline_sbml_artifacts"
SEAM = "freeze_canonical_payload"
GENERATOR = "tests/test_c011_freeze_seam_golden_equivalence.py"
MANIFEST = ROOT / "tests" / "data" / "baseline_cohort_manifest.json"
FIXTURE = ROOT / "docs/pwml_recovery_sprint/evidence/c011_freeze_seam_before.json"

#: Exactly the module globals the freeze block and the quarantine boundary read.
_GLOBALS: dict[str, Any] = {
    "Any": Any, "Dict": dict, "Optional": Any, "Path": Path, "json": json,
    "PROJECT_ROOT": PROJECT_ROOT, "ExportMode": str, "deepcopy": copy.deepcopy,
    "DEFAULT_EXPORT_MODE": DEFAULT_EXPORT_MODE, "coerce_mode": coerce_mode,
    "st": SimpleNamespace(session_state={}), "payload_sha256": payload_sha256,
    "_safe_dict": lambda value: value if isinstance(value, dict) else {},
    "quarantine_and_close": quarantine_and_close, "stamp_report": stamp_report,
    "write_quarantine_artifacts": write_quarantine_artifacts,
    "quarantine_review_flags": quarantine_review_flags,
    "run_strict_post_normalization_gates": run_strict_post_normalization_gates,
    "GateValidationError": GateValidationError,
    "PHASE_FINAL_PRE_EXPORT": PHASE_FINAL_PRE_EXPORT,
    "CANONICAL_PAYLOAD_KEY": CANONICAL_PAYLOAD_KEY,
}

#: The block's frame at BASE: ``project_root`` is the local rebinding of
#: ``PROJECT_ROOT`` it read, ``sbml_input_path`` the local it assigned.
_BEFORE = '''\
def {seam}(final_export_payload, *, strict_db, outputs_dir, pathway_context,
           export_mode, research_mode, tmp):
    project_root = PROJECT_ROOT
    sbml_input_path = None
{block}
    return dict(
        payload=canonical_export_payload, payload_hash=canonical_payload_hash,
        final_stage3_gate_report=final_stage3_gate_report,
        quarantine_result=quarantine_result, quarantine_ok=quarantine_ok,
        canonical_json_path=sbml_input_path, sbml_input_source=sbml_input_source)
'''


def _source(ref: str | None) -> str:
    if ref is None:
        return (ROOT / APP_REL).read_text(encoding="utf-8")
    shown = subprocess.run(["git", "show", f"{ref}:{APP_REL}"], cwd=ROOT,
                           capture_output=True, check=True)
    return shown.stdout.decode("utf-8")


def _fdef(source: str, name: str) -> ast.FunctionDef:
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _exec_function(source: str, name: str, namespace: dict[str, Any]) -> Any:
    """``_load_function``'s mechanism, against an arbitrary source string."""
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0),
        _fdef(source, name)], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, APP_REL, "exec"), namespace)
    return namespace[name]


def _inline_block(source: str) -> str:
    """The freeze block's own source lines, lifted out of the orchestrator."""
    def declares(statement: ast.stmt, name: str) -> bool:
        return (isinstance(statement, ast.AnnAssign)
                and getattr(statement.target, "id", "") == name)

    lines = source.splitlines(keepends=True)
    for node in ast.walk(_fdef(source, ORCHESTRATOR)):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        opens = [s for s in body if declares(s, "canonical_export_payload")]
        closes = [s for s in body if declares(s, "sbml_overwatch_report")]
        if opens and closes:
            return "".join(lines[opens[0].lineno - 1: closes[0].lineno - 1])
    raise AssertionError("the inline freeze block moved")


def _seam(ref: str | None) -> Any:
    """``freeze_canonical_payload`` at ``ref``: extracted, or the inline block."""
    source, namespace = _source(ref), dict(_GLOBALS)
    _exec_function(source, "run_quarantine_boundary", namespace)
    if any(isinstance(n, ast.FunctionDef) and n.name == SEAM
           for n in ast.parse(source).body):
        return _exec_function(source, SEAM, namespace)
    exec(compile(_BEFORE.format(seam=SEAM, block=textwrap.indent(
        textwrap.dedent(_inline_block(source)), "    ")),
        "<base-inline-freeze-block>", "exec"), namespace)
    return namespace[SEAM]


def _leg_projection(seam: Any, leg: Path, work: Path) -> dict[str, Any]:
    outputs, tmp = work / "outputs", work / "tmp"
    for directory in (outputs, tmp):
        directory.mkdir(parents=True, exist_ok=True)
    canonical = tmp / "final.canonical.json"
    canonical.unlink(missing_ok=True)
    mode, normalized = leg.name, replay._normalized(replay._payload_for(leg)[1])
    result = seam(normalized, strict_db=True, outputs_dir=outputs, pathway_context=None,
                  export_mode=mode, research_mode=is_research(mode), tmp=tmp)
    quarantine = result["quarantine_result"]
    # ``final_mapped`` and ``final_mapped_quarantined`` are the artifacts dict's
    # two views of ONE object, so this is captured with ``is``: value equality
    # would still pass if the seam started handing out a copy.
    final_mapped = result["payload"] or normalized
    return {
        "seam_result_keys": sorted(result),
        "canonical_payload_sha256": result["payload_hash"],
        "final_stage3_gate_report": result["final_stage3_gate_report"],
        "quarantine_ok": result["quarantine_ok"],
        "quarantine_refusal_reasons": list(quarantine.get("refusal_reasons") or []),
        "quarantine_coverage": quarantine.get("coverage") or {},
        "sbml_input_source": result["sbml_input_source"],
        "final_mapped_is_final_mapped_quarantined": final_mapped is result["payload"],
        "tmp_final_canonical_json_written": canonical.exists(),
    }


def _document(ref: str | None) -> dict[str, Any]:
    def orchestrator_hash(sha: str) -> str:
        dumped = ast.dump(_fdef(_source(sha), ORCHESTRATOR), include_attributes=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    origin_hash, base_hash = orchestrator_hash(ORIGIN_SHA), orchestrator_hash(BASE_SHA)
    # HARD STOP: if the orchestrator moved between the sprint's cut point and
    # this branch's parent, "before" is unproven and this is not a baseline.
    assert origin_hash == base_hash, (
        f"{ORCHESTRATOR} differs between ORIGIN_SHA {ORIGIN_SHA} and BASE {BASE_SHA}")
    seam = _seam(ref)
    with tempfile.TemporaryDirectory() as raw:
        legs = {leg.relative_to(ROOT).as_posix():
                _leg_projection(seam, leg, Path(raw) / f"leg{n}")
                for n, leg in enumerate(replay._legs())}
    return {
        "schema_version": 1,
        "cohort_id": json.loads(MANIFEST.read_text(encoding="utf-8"))["cohort_id"],
        "origin_sha": ORIGIN_SHA, "base_sha": BASE_SHA,
        "source_equivalence": {
            "symbol": f"{APP_REL}::{ORCHESTRATOR}", "equal": origin_hash == base_hash,
            "normalization": "ast.dump(include_attributes=False) then sha256",
            "origin_sha_hash": origin_hash, "base_sha_hash": base_hash},
        "generator": GENERATOR, "command": f".venv/Scripts/python.exe {GENERATOR}",
        "legs": legs,
    }


def _serialize(document: dict[str, Any]) -> bytes:
    body = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
    return (body + "\n").encode("utf-8")


def _fixture_bytes() -> bytes:
    """The fixture with git's checkout filter undone. NOT a weakening: ``json.dumps``
    writes a carriage return inside a string value as the two-character escape
    ``\\r``, never as a raw byte, so a raw ``\\r\\n`` here can only be git's doing --
    ``core.autocrlf`` is true system-wide, nothing overrides it, and every tracked
    JSON is therefore LF in the object store and CRLF in the working tree. Line
    endings in the STORED artifact are a property of the VCS, not of the seam; every
    content byte is still compared and every field still perturbation-tested. Only
    this read side is normalized -- ``_serialize`` builds bytes in memory git never
    touches, and the regeneration path below still writes LF.
    """

    return FIXTURE.read_bytes().replace(b"\r\n", b"\n")


def test_the_extracted_seam_reproduces_the_before_fixture_byte_for_byte() -> None:
    after = _document(None)
    assert _serialize(after) == _fixture_bytes()
    declared = [e["path"] for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["legs"]]
    assert sorted(after["legs"]) == sorted(declared) == sorted(set(declared))
    assert len(after["legs"]) == 39
    assert any(isinstance(n, ast.FunctionDef) and n.name == SEAM
               for n in ast.parse(_source(None)).body), f"{SEAM} is not module-level"
    # Every captured field is load-bearing: perturbing any ONE of them, on any
    # leg, must break the byte comparison above.
    leg = sorted(after["legs"])[0]
    for field in sorted(after["legs"][leg]):
        perturbed = copy.deepcopy(after)
        perturbed["legs"][leg][field] = "PERTURBED"
        assert _serialize(perturbed) != _fixture_bytes(), field


def test_the_fixture_regenerates_byte_identically_from_the_base_blob() -> None:
    first = _serialize(_document(BASE_SHA))
    assert first == _fixture_bytes()
    assert _serialize(_document(BASE_SHA)) == first


if __name__ == "__main__":  # regeneration, documented in the module docstring
    FIXTURE.write_bytes(_serialize(_document(BASE_SHA)))
