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
        # ── C-052 / A0-C8, cohort half (D-040 §5 part 2) ───────────────────
        # The projection is GROWTH-ONLY: nothing above is removed, renamed or
        # re-valued. A0-C8 asks for "the actual ``canonical_json_path`` for all
        # 39 legs", and ``sbml_input_source`` -- which says which *kind* of
        # payload was used -- is recorded there as insufficient for it.
        #
        # NOT the absolute path. It lives under
        # ``temp_root / f"post_pipeline_{uuid4().hex}"`` in production and under
        # a ``TemporaryDirectory`` here, so the string is different on every run
        # and recording it would make this byte-pinned fixture unregenerable --
        # it would stop being a fixture and become a nonce. The NAME and the
        # containment fact are the parts that are properties of the seam.
        "canonical_json_path_name": (
            "" if result["canonical_json_path"] is None
            else Path(result["canonical_json_path"]).name),
        "canonical_json_path_in_tmp": (
            result["canonical_json_path"] is not None
            and Path(result["canonical_json_path"]).parent == tmp),
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


#: C-030's DELIBERATE BASELINE MOVE (merge rule 4), stated here instead of being
#: absorbed by a rewritten fixture. Wiring hash schema 2 through the seam's one
#: ``stamp_report`` call adds these three keys to ``final_stage3_gate_report`` --
#: and does nothing else: no key removed, no existing key's value changed,
#: ``payload_sha256`` byte for byte what it was, and the four quarantine-refusal
#: legs write no report at all, so they stay untouched. The fixture therefore
#: remains the BEFORE document it is named for and still regenerates from the
#: BASE blob below, while a FOURTH key, a changed value, a moved
#: ``payload_sha256`` or a refusal leg that grew a report each fail right here.
_C030_KEYS = ("canonical_graph_sha256", "canonical_payload_sha256", "hash_schema_version")
_C030_LEGS = 35


def _with_c030_hash_keys(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """``before`` plus exactly C-030's delta, and only where the seam wrote one."""
    document = copy.deepcopy(before)
    moved = 0
    for name, leg in document["legs"].items():
        report = leg["final_stage3_gate_report"]
        produced = after["legs"][name]["final_stage3_gate_report"]
        if not report:
            assert not produced, f"{name}: a refusal leg wrote a gate report"
            continue
        assert set(produced) - set(report) == set(_C030_KEYS), name
        # Not merely "three new keys": schema 2's payload projection must agree
        # with the digest the seam already published, and the graph projection
        # must be a different digest -- one hash renamed twice would pass a
        # key-name check and fail these.
        assert produced["hash_schema_version"] == 2, name
        assert produced["canonical_payload_sha256"] == report["payload_sha256"], name
        assert produced["canonical_graph_sha256"] != report["payload_sha256"], name
        report.update({key: produced[key] for key in _C030_KEYS})
        moved += 1
    assert moved == _C030_LEGS, moved
    return document


#: C-052's PROJECTION delta, stated here for the same reason ``_C030_KEYS`` is:
#: so the fixture stays the document it is named for instead of being quietly
#: rewritten. The two keys are read off ``canonical_json_path`` -- data the
#: seam's seven-field return has published since C-011 -- so the SEAM did not
#: move; only what this harness looks at did. That distinction is the whole
#: point of writing it as a delta: a fixture regenerated to absorb it would look
#: identical whether the seam had changed or not.
#:
#: Unlike C-030's, this delta appears in BOTH documents -- the base blob writes
#: ``tmp / "final.canonical.json"`` and assigns it to ``sbml_input_path`` exactly
#: as the extracted seam assigns it to ``canonical_json_path`` -- so both callers
#: below go through it, and a base blob that produced a different name or put the
#: file somewhere else would fail here rather than being papered over.
_C052_KEYS = ("canonical_json_path_name", "canonical_json_path_in_tmp")
#: 35 legs freeze a canonical payload; the other 4 are quarantine refusals that
#: write no file and must therefore report ``""`` / ``False``. Same split as
#: ``_C030_LEGS``, and asserted rather than assumed.
_C052_LEGS = 35


def _with_c052_path_keys(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """``before`` plus exactly C-052's projection delta, leg by leg."""

    document = copy.deepcopy(before)
    written = 0
    for name, leg in document["legs"].items():
        produced = after["legs"][name]
        assert set(produced) - set(leg) == set(_C052_KEYS), name
        # Not merely "two new keys": the path the seam returns must be the file
        # it wrote, under the tmp it was given, on exactly the legs that wrote
        # one. A seam that returned some other path, or the right name from the
        # wrong directory, or a path on a refusal leg, passes a key-name check
        # and fails these.
        wrote = leg["tmp_final_canonical_json_written"]
        assert produced["canonical_json_path_in_tmp"] is wrote, name
        assert produced["canonical_json_path_name"] == (
            "final.canonical.json" if wrote else ""), name
        assert (produced["sbml_input_source"] == CANONICAL_PAYLOAD_KEY) is wrote, name
        leg.update({key: produced[key] for key in _C052_KEYS})
        written += bool(wrote)
    assert written == _C052_LEGS, written
    return document


#: C-057's LINEAGE delta, stated here for exactly the reason ``_C030_KEYS`` and
#: ``_C052_KEYS`` are: so the fixture stays the BEFORE document it is named for
#: instead of being quietly rewritten. C-057 writes ``provenance_lineage`` onto
#: the process rows strict quarantine EXCLUDES. A quarantined row is deleted from
#: the payload, so on most legs the attribution never reaches the frozen graph at
#: all -- but where the excluded row is a LOCKED reaction,
#: ``_reconcile_locked_reactions`` keeps its retained copy under
#: ``payload["quarantined_locked_reactions"]``, and the frozen payload's digest
#: moves with it. ``PRODUCT_CONTRACT`` 178 REQUIRES that it move: "Lineage must
#: not change graph equivalence, but lineage changes must remain detectable."
#:
#: Unlike C-030's and C-052's, these name keys whose VALUE moves, not keys that
#: appear: C-057 adds nothing to this projection. What moves is ONE digest
#: recorded twice -- the leg's ``canonical_payload_sha256`` and its gate report's
#: ``payload_sha256``, equal on all 35 report-writing legs -- on the 7 legs that
#: quarantine a locked reaction. Nothing else moves: not ``connectivity``, not
#: ``normalization_stats``, not ``errors``, not ``ok``, not ``phase``, and no leg
#: field outside the report. Those are the biology, and a lineage write that had
#: moved any of them fails right here instead of being absorbed. Measured over
#: the whole cohort in ``evidence/c057_cohort_equivalence.json``:
#: ``canonical_graph_sha256`` -- the EXPORTERS' hash -- moves on 0 of 78 leg
#: runs, and so do the surviving row set, the admission states and the removals.
#:
#: ONE-WAY, and that is the limitation to know: ``_without_c052_path_keys`` can
#: strip C-052's delta because its keys are additive, but the digest this one
#: replaces is not stored anywhere, so its inverse is not expressible. Running
#: ``__main__`` below would therefore ABSORB this delta into the fixture, which
#: is the "regenerate to absorb it" failure the whole mechanism exists to
#: prevent. The fixture must not be regenerated while this delta stands.
_C057_KEYS = ("canonical_payload_sha256", "final_stage3_gate_report")
_C057_LEGS = 7


def _with_c057_lineage_hashes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """``before`` plus exactly C-057's payload-digest delta, leg by leg.

    Runs INNERMOST, before :func:`_with_c030_hash_keys`, deliberately. C-030's
    ``produced["canonical_payload_sha256"] == report["payload_sha256"]`` is what
    proves schema 2's payload projection agrees with the digest the seam already
    published; moving the stored digest first keeps that assertion running
    against a live value rather than weakening it to accommodate this delta.
    """

    document = copy.deepcopy(before)
    moved = 0
    for name, leg in document["legs"].items():
        produced = after["legs"][name]
        report = leg["final_stage3_gate_report"]
        # Everything this delta does NOT touch, checked on EVERY leg and not only
        # on the ones that move. A comparison of the digests alone would absorb a
        # lineage write that had also changed the graph; these are where such a
        # change would land.
        for field in leg:
            if field not in _C057_KEYS:
                assert leg[field] == produced[field], f"{name}: {field}"
        for key in report:
            if key != "payload_sha256":
                assert report[key] == produced["final_stage3_gate_report"][key], f"{name}: {key}"
        if not report:
            # A refusal leg freezes nothing, so it has no digest that can move.
            assert not produced["final_stage3_gate_report"], name
            assert leg["canonical_payload_sha256"] == "", name
            assert produced["canonical_payload_sha256"] == "", name
            continue
        # One digest recorded twice, and it must be recorded twice on BOTH sides
        # -- otherwise "the payload hash moved" is two claims and this helper
        # would let the two halves drift apart.
        assert leg["canonical_payload_sha256"] == report["payload_sha256"], name
        digest = produced["final_stage3_gate_report"]["payload_sha256"]
        assert produced["canonical_payload_sha256"] == digest, name
        if digest == report["payload_sha256"]:
            continue
        report["payload_sha256"] = digest
        leg["canonical_payload_sha256"] = digest
        moved += 1
    assert moved == _C057_LEGS, moved
    return document


def _without_c052_path_keys(document: dict[str, Any]) -> dict[str, Any]:
    """``document`` reduced to the 9-field projection the FIXTURE stores.

    The regeneration path below writes this, so ``__main__`` can never silently
    move the fixture by the delta above -- which is precisely the "regenerate to
    absorb it" failure the delta exists to prevent.
    """

    stripped = copy.deepcopy(document)
    for leg in stripped["legs"].values():
        for key in _C052_KEYS:
            leg.pop(key, None)
    return stripped


def test_the_extracted_seam_reproduces_the_before_fixture_byte_for_byte() -> None:
    after = _document(None)
    before = json.loads(_fixture_bytes())
    assert _serialize(before) == _fixture_bytes(), "the fixture is no longer its own bytes"
    expected = _serialize(_with_c052_path_keys(
        _with_c030_hash_keys(_with_c057_lineage_hashes(before, after), after), after))
    assert _serialize(after) == expected
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
        assert _serialize(perturbed) != expected, field


def test_the_fixture_regenerates_byte_identically_from_the_base_blob() -> None:
    base = _document(BASE_SHA)
    # The base blob's own document IS the fixture plus exactly C-052's projection
    # delta and C-057's digest delta -- both proved through the same helpers the
    # tip goes through, so the two sides cannot drift apart -- and the fixture
    # bytes themselves never move.
    before = _with_c057_lineage_hashes(json.loads(_fixture_bytes()), base)
    assert _serialize(base) == _serialize(_with_c052_path_keys(before, base))
    first = _serialize(_without_c052_path_keys(base))
    assert first == _serialize(before)
    assert _serialize(_without_c052_path_keys(_document(BASE_SHA))) == first


if __name__ == "__main__":  # regeneration, documented in the module docstring
    FIXTURE.write_bytes(_serialize(_without_c052_path_keys(_document(BASE_SHA))))
