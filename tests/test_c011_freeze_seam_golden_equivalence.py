"""C-011: the canonical-freeze seam is a PURE MOVE, proved by behaviour.

``evidence/c011_freeze_seam_before.json`` records what the freeze block DID at
BASE, where it was still inline inside ``run_post_pipeline_sbml_artifacts``: it
is built by executing the BASE blob's OWN statements -- lifted out of the BASE
AST by line span, never retyped -- over all 39 legs of the frozen baseline
cohort. The tests rebuild that document from the working tree's extracted
``freeze_canonical_payload`` and compare BYTES, so a seam that kept the payload
while dropping a gate report, changing ``sbml_input_source`` or replacing object
sharing with a copy fails here.

REGENERATION IS REFUSED, and that belongs here, beside the command, not 300 lines
below it. This fixture is a BEFORE document with a historical identity: the
per-card helpers below STATE each baseline move instead of absorbing it, so a
rebuilt fixture stops being the document those moves are stated AGAINST. C-057's
delta REPLACES a digest whose pre-move value is stored nowhere else, so it cannot
be inverted the way C-052's additive keys can, and regenerating would destroy
three cards' evidence while the suite stayed green (F-076). The command below
therefore exits NON-ZERO and writes NOTHING, printing which delta stands, why no
inverse is expressible, and what overriding would destroy:

    .venv/Scripts/python.exe tests/test_c011_freeze_seam_golden_equivalence.py

Retiring the BEFORE baseline is a PRODUCT DECISION, not maintenance. Taking it
means naming the exact standing set, which the refusal prints for you:

    ... test_c011_freeze_seam_golden_equivalence.py --retire-the-before-baseline=C-057

``_DELTAS`` below is the registry the guard reads. A per-card delta helper with no
entry in it blocks regeneration by itself, so a later card cannot add one silently.
"""

from __future__ import annotations

import ast, copy, hashlib, json, re, subprocess, sys, tempfile, textwrap  # noqa: E401
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

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
#: ONE-WAY. DO NOT READ THE HELPERS ABOVE AND ASSUME THIS ONE REVERSES LIKE
#: THEIRS -- it does not, and the asymmetry is structural rather than an
#: oversight:
#:
#: 1. **There is no ``_without_c057_lineage_hashes`` and none can be written.**
#:    ``_without_c052_path_keys`` works because C-052's delta ADDS keys, so
#:    dropping them restores the fixture exactly. This delta REPLACES a digest,
#:    and the value it replaces -- the pre-C-057 ``payload_sha256`` -- is stored
#:    nowhere but in the fixture itself. Once overwritten there is nothing left
#:    to restore it from, so the inverse is not expressible at any price.
#: 2. **Running ``__main__`` below WILL silently absorb this delta**, because
#:    that path regenerates from ``_document(BASE_SHA)``, which now carries it.
#:    The write would succeed, the suite would pass, and the fixture would have
#:    stopped being the BEFORE document it is named for -- the exact
#:    "regenerate to absorb it" failure this whole mechanism exists to prevent,
#:    committed by the one command documented as safe.
#: 3. **So the fixture must not be regenerated while this delta stands.** If a
#:    later card needs to regenerate it, that is a product decision about
#:    retiring the BEFORE baseline, not a housekeeping step, and it has to be
#:    taken deliberately rather than by running the block at the bottom of this
#:    file.
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


# ─── C-065: the regeneration guard that closes F-076 ────────────────────────
#: Each helper above states a baseline move. What ``__main__`` needs is not WHAT
#: each does but whether the fixture can still be REBUILT while it stands, so
#: every helper carries a REGENERATION DISPOSITION here -- a registry, not an
#: ``if C-057``, because the guard must keep working for a delta nobody has
#: written yet, and because an unregistered helper is itself a refusal.
_INVERTIBLE = "invertible"
_NOT_IN_BASE_DOCUMENT = "not-in-base-document"
_ONE_WAY = "one-way"
_DISPOSITIONS = (_INVERTIBLE, _NOT_IN_BASE_DOCUMENT, _ONE_WAY)
#: The override. Not a bare ``--force``: its VALUE must be the exact standing set,
#: so registering a second one-way delta expires every invocation naming the first.
OVERRIDE = "--retire-the-before-baseline"
WRITE_TO = "--write-to"
USAGE = f"""usage: {sys.executable} {GENERATOR} [{OVERRIDE}=CARDS] [{WRITE_TO}=PATH]

  (no arguments)   rebuild the fixture from the BASE blob. REFUSED, non-zero and
                   writing nothing, while any one-way delta is registered below.
  {OVERRIDE}=CARDS
                   a product decision to retire the BEFORE baseline. CARDS must
                   be the exact comma-separated set of standing one-way deltas.
  {WRITE_TO}=PATH  write somewhere other than the tracked fixture. Used by this
                   file's own guard tests; it does NOT bypass the refusal."""


class _Delta(NamedTuple):
    """One card's stated baseline move, as the regeneration guard sees it.
    ``moves`` and ``why`` are DATA, not commentary: the refusal prints them.
    """

    card: str
    helper: str
    disposition: str
    inverse: str | None
    moves: str
    why: str


_DELTAS: tuple[_Delta, ...] = (
    _Delta(
        card="C-030", helper="_with_c030_hash_keys",
        disposition=_NOT_IN_BASE_DOCUMENT, inverse=None,
        moves="adds three hash-schema-2 keys to final_stage3_gate_report on 35 legs",
        why="the keys come from the SEAM's own stamp_report call site, which the "
            "BASE blob does not have, so _document(BASE_SHA) never carries this "
            "delta and a rebuild cannot absorb it",
    ),
    _Delta(
        card="C-052", helper="_with_c052_path_keys",
        disposition=_INVERTIBLE, inverse="_without_c052_path_keys",
        moves="adds canonical_json_path_name and canonical_json_path_in_tmp to "
              "every leg of the projection",
        why="the delta is ADDITIVE, so dropping the two keys recovers the "
            "document exactly; __main__ applies that inverse before writing",
    ),
    _Delta(
        card="C-057", helper="_with_c057_lineage_hashes",
        disposition=_ONE_WAY, inverse=None,
        moves="REPLACES one digest recorded twice -- canonical_payload_sha256 and "
              "its gate report's payload_sha256 -- on the 7 legs that quarantine "
              "a locked reaction",
        why="the value it replaces, the pre-C-057 payload_sha256, is stored "
            "nowhere but in the fixture itself, so once overwritten there is "
            "nothing left to restore it from and no _without_c057_* is "
            "expressible at any price (F-076)",
    ),
)

_HELPER_RE = re.compile(r"^_with_(c\d+)_\w+$")
_INVERSE_RE = re.compile(r"^_without_(c\d+)_\w+$")


def _named(pattern: re.Pattern[str], prefix: str) -> tuple[dict[str, str], list[str]]:
    """``({card key: function name}, [misnamed function names])`` in this module."""

    keyed: dict[str, str] = {}
    misnamed: list[str] = []
    for name in sorted(globals()):
        if not name.startswith(prefix) or not callable(globals()[name]):
            continue
        match = pattern.match(name)
        if match is None:
            misnamed.append(name)
        else:
            keyed.setdefault(match.group(1), name)
    return keyed, misnamed


def _registry_defects() -> list[str]:
    """Every disagreement between ``_DELTAS`` and the module it describes.

    A defect blocks regeneration exactly as a standing one-way delta does, and no
    override dismisses one: the guard fails CLOSED, so the next card to add a
    replacing helper is stopped by its own omission rather than permitted by it.
    """

    defects: list[str] = []
    registered: dict[str, _Delta] = {}
    for delta in _DELTAS:
        key = delta.card.replace("-", "").lower()
        if key in registered:
            defects.append(f"{delta.card}: registered more than once")
        registered[key] = delta
        if delta.disposition not in _DISPOSITIONS:
            defects.append(f"{delta.card}: disposition {delta.disposition!r} is not "
                           f"one of {_DISPOSITIONS}")
        if delta.disposition == _INVERTIBLE:
            if not callable(globals().get(delta.inverse or "")):
                defects.append(f"{delta.card}: declared {_INVERTIBLE} but its inverse "
                               f"{delta.inverse!r} is not defined here")
        elif delta.inverse is not None:
            defects.append(f"{delta.card}: only an {_INVERTIBLE} delta may name an inverse")
    helpers, misnamed_helpers = _named(_HELPER_RE, "_with_")
    inverses, misnamed_inverses = _named(_INVERSE_RE, "_without_")
    for name in misnamed_helpers + misnamed_inverses:
        defects.append(f"{name}: a per-card delta helper must be named "
                       f"_with_c<NNN>_... or _without_c<NNN>_... so the guard can "
                       f"see it. Rename it and register it in _DELTAS.")
    for key, name in helpers.items():
        if key not in registered:
            defects.append(f"{name}: a per-card delta helper with NO entry in _DELTAS. "
                           f"Add one saying whether __main__ can still rebuild the "
                           f"fixture while your delta stands, and why. Until then it "
                           f"blocks regeneration.")
    for key, delta in registered.items():
        if helpers.get(key) != delta.helper:
            defects.append(f"{delta.card}: registered helper {delta.helper} is not a "
                           f"delta helper defined in this module")
        if key in inverses and delta.disposition != _INVERTIBLE:
            defects.append(f"{delta.card}: {inverses[key]} exists, so this delta IS "
                           f"invertible and must be registered {_INVERTIBLE!r}, not "
                           f"{delta.disposition!r}")
    return defects


def _standing_one_way_deltas() -> tuple[_Delta, ...]:
    return tuple(delta for delta in _DELTAS if delta.disposition == _ONE_WAY)


def _regeneration_inverses() -> list[Callable[..., dict[str, Any]]]:
    """The inverses ``__main__`` composes before writing, in registry order.

    With C-052 the only invertible delta this is exactly ``_without_c052_path_keys``:
    the path that already works is unchanged, merely read off the registry now.
    """

    return [globals()[delta.inverse] for delta in _DELTAS
            if delta.disposition == _INVERTIBLE]


def _regeneration_refusal() -> str | None:
    """Why the fixture must not be rebuilt right now, or ``None`` if it may be."""

    defects, standing = _registry_defects(), _standing_one_way_deltas()
    if not defects and not standing:
        return None
    lines = [f"REFUSING to regenerate {FIXTURE.relative_to(ROOT).as_posix()}", ""]
    for delta in standing:
        lines += [f"  ONE-WAY DELTA STANDS: {delta.card} ({delta.helper})",
                  f"    what it does : it {delta.moves}",
                  f"    no inverse   : {delta.why}", ""]
    for defect in defects:
        lines += [f"  REGISTRY DEFECT: {defect}", ""]
    lines += [
        "  WHAT REGENERATING WOULD DESTROY:",
        "    _document(BASE_SHA) already carries the delta above, so a rebuilt",
        "    fixture absorbs it. It stops being the BEFORE document it is named",
        "    for, and every card's stated delta -- C-030's, C-052's and C-057's",
        "    alike -- becomes unverifiable at the same stroke, because the",
        "    document they are stated AGAINST no longer exists. The suite would",
        "    still pass, so nothing would report it (F-076).", ""]
    if defects:
        lines += ["  Fix the registry defect(s) above. The override cannot dismiss one.", ""]
    else:
        lines += ["  Retiring the BEFORE baseline is a PRODUCT DECISION, not maintenance.",
                  "  If that decision has been taken, name the exact standing set:", "",
                  f"    {sys.executable} {GENERATOR} \\",
                  f"      {OVERRIDE}={','.join(delta.card for delta in standing)}", ""]
    return "\n".join(lines)


def _override_mismatch(named: list[str], standing: tuple[_Delta, ...]) -> str | None:
    """``None`` when ``named`` is exactly the standing set, else why it is not."""

    expected = [delta.card for delta in standing]
    if sorted(named) == sorted(expected):
        return None
    return (f"{OVERRIDE} must name EXACTLY the standing one-way deltas "
            f"[{','.join(expected) or 'none'}]; got [{','.join(named) or 'none'}]. "
            f"It names them so it cannot be pasted forward: registering another "
            f"one-way delta must expire this invocation rather than carry it.")


def _destruction_notice(target: Path, standing: tuple[_Delta, ...]) -> str:
    stored = hashlib.sha256(FIXTURE.read_bytes()).hexdigest() if FIXTURE.is_file() else "absent"
    return "\n".join([
        "RETIRING THE BEFORE BASELINE. This is what is being destroyed:", "",
        f"  document   : {FIXTURE.relative_to(ROOT).as_posix()}",
        f"  its bytes  : sha256 {stored}",
        f"  written to : {target}"]
        + [f"  absorbing  : {delta.card} -- it {delta.moves}" for delta in standing]
        + ["",
           "  The rebuilt document is no longer the BEFORE document the entries in",
           "  _DELTAS are stated AGAINST, so C-030's, C-052's and C-057's stated",
           "  moves stop being verifiable against it and the pre-move digests",
           "  survive only in this file's git history.", ""])


def _parse_argv(argv: list[str]) -> tuple[Path, list[str] | None]:
    target, named = FIXTURE, None
    for argument in argv:
        if argument.startswith(f"{OVERRIDE}="):
            named = [card.strip() for card in argument.split("=", 1)[1].split(",")
                     if card.strip()]
        elif argument.startswith(f"{WRITE_TO}="):
            target = Path(argument.split("=", 1)[1])
        else:
            raise SystemExit(f"{USAGE}\n\nunrecognized argument: {argument!r}")
    return target, named


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


def _regenerate(argv: list[str]) -> int:
    """``__main__``'s body. Returns the process exit code; writes only on 0.

    The guard runs BEFORE the 39-leg rebuild, so a refusal is instant and cannot
    half-write; the inverses come from ``_DELTAS``, so a later card extends its
    registry entry rather than this function.
    """

    target, named = _parse_argv(argv)
    refusal = _regeneration_refusal()
    if refusal is None:
        if named is not None:
            print(f"{USAGE}\n\nno one-way delta is registered: there is nothing to "
                  f"retire, so {OVERRIDE} is refused rather than ignored.",
                  file=sys.stderr)
            return 2
    else:
        if named is None:
            print(refusal, file=sys.stderr)
            return 2
        standing = _standing_one_way_deltas()
        rejection = ("the override cannot dismiss a registry defect"
                     if _registry_defects() else _override_mismatch(named, standing))
        if rejection is not None:
            print(f"{refusal}\n  OVERRIDE REJECTED: {rejection}\n", file=sys.stderr)
            return 2
        print(_destruction_notice(target, standing))
    document = _document(BASE_SHA)
    for inverse in _regeneration_inverses():
        document = inverse(document)
    target.write_bytes(_serialize(document))
    return 0


def _run_main(*argv: str) -> subprocess.CompletedProcess[str]:
    """This file's ``__main__``, as a real process, for the guard tests below."""

    return subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def test_regeneration_is_refused_while_a_one_way_delta_is_registered() -> None:
    """Bare ``__main__`` must exit non-zero, write nothing, and say why.

    At BASE this block exited 0 and rewrote the fixture, absorbing C-057's
    digest move on 7 legs -- a stated baseline move turned silent, with the
    suite still green (F-076).
    """

    before = FIXTURE.read_bytes()
    done = _run_main()
    assert FIXTURE.read_bytes() == before, "the refusal wrote to the fixture"
    assert done.returncode != 0, done.stdout
    message = done.stdout + done.stderr
    for required in ("REFUSING to regenerate", "C-057", "_with_c057_lineage_hashes",
                     "no inverse", "unverifiable", OVERRIDE):
        assert required in message, (required, message)


def test_the_override_regenerates_and_absorbs_exactly_the_one_way_delta(
        tmp_path: Any) -> None:
    """The escape hatch is real, and what it destroys is exactly what was stated.

    The rebuilt document must be the fixture plus C-057's delta and NOTHING else:
    ``_with_c057_lineage_hashes`` asserts every untouched field on every leg and
    that the digest moved on exactly ``_C057_LEGS``, so a rebuild that had also
    moved the biology fails here instead of being absorbed.
    """

    target = tmp_path / "retired.json"
    before = FIXTURE.read_bytes()
    done = _run_main(f"{WRITE_TO}={target}", f"{OVERRIDE}=C-057")
    assert done.returncode == 0, done.stdout + done.stderr
    assert FIXTURE.read_bytes() == before, "the override wrote to the TRACKED fixture"
    for required in ("RETIRING THE BEFORE BASELINE", "what is being destroyed",
                     "C-057", "sha256", str(target)):
        assert required in done.stdout, (required, done.stdout)
    rebuilt = json.loads(target.read_bytes())
    assert _serialize(_with_c057_lineage_hashes(
        json.loads(_fixture_bytes()), rebuilt)) == _serialize(rebuilt)


def test_the_override_must_name_exactly_the_standing_one_way_deltas(
        tmp_path: Any) -> None:
    """A bare force flag would be pasted forward; naming the set expires it."""

    target = tmp_path / "must-not-exist.json"
    standing = _standing_one_way_deltas()
    assert [delta.card for delta in standing] == ["C-057"], standing
    assert _override_mismatch(["C-057"], standing) is None
    for wrong in ("", "C-030", "C-057,C-099", "c-057"):
        assert _override_mismatch([c for c in wrong.split(",") if c],
                                  standing) is not None, wrong
        done = _run_main(f"{WRITE_TO}={target}", f"{OVERRIDE}={wrong}")
        assert done.returncode != 0, wrong
        assert not target.exists(), wrong
    assert _override_mismatch(["C-057", "C-099"], standing + (
        _Delta("C-099", "_with_c099_x", _ONE_WAY, None, "x", "y"),)) is None


def test_an_unregistered_delta_helper_blocks_regeneration_by_itself(
        monkeypatch: Any) -> None:
    """The registry is hard to skip: omitting an entry is itself a refusal.

    Catches the next card adding a replacing helper without declaring whether the
    fixture can still be rebuilt while its delta stands -- the exact omission that
    made F-076 latent instead of loud.
    """

    assert _registry_defects() == []
    monkeypatch.setitem(globals(), "_with_c099_demo_delta", lambda before, after: before)
    defects = _registry_defects()
    assert any("_with_c099_demo_delta" in defect and "_DELTAS" in defect
               for defect in defects), defects
    refusal = _regeneration_refusal()
    assert refusal is not None and "REGISTRY DEFECT" in refusal
    assert "override cannot dismiss one" in refusal
    # And a helper named so the scanner cannot see it is a defect too, so the
    # registry cannot be dodged by choosing a different name.
    monkeypatch.setitem(globals(), "_with_lineage_v2", lambda before, after: before)
    assert any("_with_lineage_v2" in defect for defect in _registry_defects())


def test_the_registry_describes_the_helpers_it_guards(monkeypatch: Any) -> None:
    """Every helper registered, every disposition honest, C-052's path unchanged."""

    assert _registry_defects() == []
    assert {delta.card for delta in _DELTAS} == {"C-030", "C-052", "C-057"}
    assert all(delta.moves and delta.why for delta in _DELTAS)
    # The inverse composition IS the pre-C-065 regeneration path, read off the
    # registry: identical function, identical order, nothing added.
    assert _regeneration_inverses() == [_without_c052_path_keys]
    # A delta that claims to be one-way while its inverse exists is a defect, so
    # the disposition cannot drift away from the code it describes.
    monkeypatch.setattr(sys.modules[__name__], "_DELTAS", tuple(
        delta._replace(disposition=_ONE_WAY, inverse=None)
        if delta.card == "C-052" else delta for delta in _DELTAS))
    assert any("_without_c052_path_keys" in defect and _INVERTIBLE in defect
               for defect in _registry_defects()), _registry_defects()


def test_the_docstring_states_the_refusal_beside_the_regeneration_command() -> None:
    """F-076's second defect: the promise was 320 lines from its own warning."""

    doc = __doc__ or ""
    command = f".venv/Scripts/python.exe {GENERATOR}"
    assert command in doc and OVERRIDE in doc
    assert 0 < doc.index("REFUSED") < doc.index(command), "the warning follows the command"
    assert "byte-identically" not in doc.split("REFUSED")[0], "the false promise survives"


if __name__ == "__main__":  # GUARDED regeneration -- see the module docstring
    raise SystemExit(_regenerate(sys.argv[1:]))
