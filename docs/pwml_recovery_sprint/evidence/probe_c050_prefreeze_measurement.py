"""C-050's headline measurement: the five categories over a LIVE re-derivation.

``probe_exporter_identity_mutation.py`` measurement A compares two **committed**
files. Those artifacts were produced before this fix and are protected, so they
can never show the fix. This probe runs the same measurement over a payload
re-derived live, through the region C-050 changed.

**The classifier is imported, not copied.** ``_report`` and ``_pair_rows`` come
straight from ``probe_exporter_identity_mutation``, so "the same five categories
by the same one-to-one pairing methodology" is literal: it is the same code,
including the H-005 correction that pairs on ``mapping_meta.query.name`` and
never on ``raw_name``. ``RESULT: UNDETERMINED`` still exits 2 and is still not a
pass.

THE TWO LEGS
------------
``--leg base`` -- load the committed canonical payload and build the IR from it,
which is exactly what production does at ``bcc0bfe``: no pre-freeze resolution,
the exporter resolves. Expected to show nonzero counts. This is the G9
behavioural base failure, and it is a *behaviour*, not a symbol check: the base
leg never imports :mod:`t2pw.pwml.prefreeze_resolution`.

``--leg tip`` -- the same payload, but ``run_prefreeze_resolution`` first, then
``freeze_canonical_payload``, then build the IR from the frozen canonical
payload and compare **that** payload against the IR. Every count must be 0 with
``RESULT: MEASURED``.

Running ``--leg tip`` at ``bcc0bfe`` fails at import: the module does not exist
there. That is *not* the base proof and is not presented as one -- ``--leg base``
is, and it runs identically at both SHAs.

SCOPE -- READ BEFORE QUOTING THE RESULT
---------------------------------------
This measures **compound rows in the canonical JSON**. It is the C-050 half of
what ``MASTER_PLAN.md`` §1.2 states as a joint C-050/C-051 acceptance; C-051's
assert-only conversion of the in-IR call site is what makes the zeros structural
rather than a consequence of the resolution being idempotent. **It is not
T-102.** ``DECISIONS.md`` **D-016 (LOCKED)** requires T-102 to verify
organism/species equivalence across canonical JSON, PWML *and* SBML, which is
C-045's and C-052's territory.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from _repo_root import add_src_to_path, require
from probe_exporter_identity_mutation import PASSING_LEG, _report

add_src_to_path()

import t2pw  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir  # noqa: E402

PATHWAY_CONTEXT: Dict[str, Any] = {
    "pathway_name": "Heme biosynthesis",
    "name": "Heme biosynthesis",
    "pathway_subject": "Metabolic",
    "subject": "Metabolic",
}


def _load_source_payload() -> Dict[str, Any] | None:
    leg = require(PASSING_LEG)
    if leg is None:
        return None
    path = leg / "final_mapped.json"
    if not path.is_file():
        print(f"  [skip] {PASSING_LEG}/final_mapped.json is absent")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("metadata", {}).update(PATHWAY_CONTEXT)
    return payload


def _freeze(payload: Dict[str, Any], tmp: Path) -> Dict[str, Any]:
    """Run the real seam, not a stand-in for it."""

    from t2pw.app.streamlit_app import freeze_canonical_payload
    from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE

    outputs = tmp / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    result = freeze_canonical_payload(
        payload,
        strict_db=False,
        outputs_dir=outputs,
        pathway_context=dict(PATHWAY_CONTEXT),
        export_mode=DEFAULT_EXPORT_MODE,
        research_mode=True,
        tmp=tmp,
    )
    frozen = result.get("payload") or {}
    if not frozen:
        print("  quarantine refused the payload; falling back to the pre-freeze object")
        print(f"  quarantine_ok={result.get('quarantine_ok')} "
              f"sbml_input_source={result.get('sbml_input_source')!r}")
        return payload
    print(f"  freeze: sbml_input_source={result.get('sbml_input_source')!r} "
          f"canonical_json_path={'set' if result.get('canonical_json_path') else 'None'}")
    return frozen


def _measure(canonical: Dict[str, Any], label: str) -> tuple[int, List[str]]:
    ir, ir_report = build_pwml_ir(
        deepcopy(canonical),
        pathway_name=str(PATHWAY_CONTEXT["pathway_name"]),
        pathway_subject=str(PATHWAY_CONTEXT["pathway_subject"]),
        strict_db=False,
    )
    canonical_rows: List[Dict[str, Any]] = (
        canonical.get("entities", {}).get("compounds") or []
    )
    ir_rows: List[Dict[str, Any]] = ir.get("entities", {}).get("compounds") or []
    print(f"=== {label}: canonical payload vs LIVE-built IR ===")
    print(f"    db_resolution available: {(ir_report.get('db_resolution') or {}).get('available')}")
    lines, code = _report(canonical_rows, ir_rows)
    for line in lines:
        print(line)
    return code, lines


def _check_canonical_path(payload: Dict[str, Any]) -> int:
    """A10 / A0-C8: C-050 must not change which file the canonical path names."""

    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    observed: Dict[str, Dict[str, Any]] = {}
    for label, prepare in (("base_semantics", False), ("with_prefreeze", True)):
        candidate = deepcopy(payload)
        if prepare:
            run_prefreeze_resolution(candidate, strict_db=False)
        with tempfile.TemporaryDirectory(prefix="c050path") as raw_tmp:
            tmp = Path(raw_tmp)
            from t2pw.app.streamlit_app import freeze_canonical_payload
            from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE

            outputs = tmp / "outputs"
            outputs.mkdir(parents=True, exist_ok=True)
            result = freeze_canonical_payload(
                candidate, strict_db=False, outputs_dir=outputs,
                pathway_context=dict(PATHWAY_CONTEXT), export_mode=DEFAULT_EXPORT_MODE,
                research_mode=True, tmp=tmp,
            )
            canonical_path = result.get("canonical_json_path")
            observed[label] = {
                "sbml_input_source": result.get("sbml_input_source"),
                "canonical_json_name": Path(canonical_path).name if canonical_path else None,
                "canonical_json_relative_to_tmp": (
                    str(Path(canonical_path).relative_to(tmp)) if canonical_path else None
                ),
                "quarantine_ok": result.get("quarantine_ok"),
            }
        print(f"  {label:<16}: {json.dumps(observed[label], sort_keys=True)}")

    if observed["base_semantics"] == observed["with_prefreeze"]:
        print("\nA10: PASSED -- the canonical path and its source are unchanged.")
        return 0
    print("\nA10: FAILED -- C-050 changed which file the canonical path names.")
    return 1


def _verdict(code: int, lines: List[str]) -> int:
    """Exit 0 only for all-five-zero AND ``RESULT: MEASURED``. UNDETERMINED never passes."""

    from probe_exporter_identity_mutation import CATEGORIES

    body = "\n".join(lines)
    measured = code == 0 and "RESULT: MEASURED" in body
    zeros = [title for _key, title in CATEGORIES if f"{title:<32}:   0 instance" not in body]
    if measured and not zeros:
        print("\nC-050 ACCEPTANCE: PASSED -- five categories 0, RESULT: MEASURED.")
        print("  Scope: compound rows in the canonical JSON. NOT T-102 (D-016).")
        return 0
    print("\nC-050 ACCEPTANCE: FAILED")
    if not measured:
        print("  the population was not fully measured (see RESULT above); a lower")
        print("  bound of zero is not a zero-mutation result")
    for title in zeros:
        print(f"  nonzero category: {title}")
    return 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=("base", "tip", "auto"), default="auto")
    parser.add_argument(
        "--check-canonical-path", action="store_true",
        help="A0-C8 / A10: freeze the same payload with and without pre-freeze "
             "resolution and show that the canonical path names the same file.",
    )
    args = parser.parse_args(argv)

    print(f"T2PW: {t2pw.__file__}")
    payload = _load_source_payload()
    if payload is None:
        # A9: "UNDETERMINED is never a pass" -- and an absent input leg is the
        # purest UNDETERMINED there is. Returning 0 here let a run that measured
        # NOTHING report success through a different door than the verdict
        # function guards, which is exactly the failure mode A9 names.
        print(f"\nRESULT: UNDETERMINED -- {PASSING_LEG} carries no payload to measure")
        print("C-050 ACCEPTANCE: FAILED -- nothing was measured; this is not a zero")
        return 2

    if args.check_canonical_path:
        return _check_canonical_path(payload)

    leg = args.leg
    if leg == "auto":
        # SHA-invariant: replicate whatever the enrichment->freeze region of the
        # tree this probe is running in actually does. At bcc0bfe that region has
        # no pre-freeze call, so the measurement is of the base behaviour and it
        # is nonzero; here it has one, so it is zero. Same command, same file,
        # opposite verdicts -- the difference is the production path, not a
        # symbol lookup, and the assertion below is on the five counts.
        import importlib.util

        leg = "tip" if importlib.util.find_spec("t2pw.pwml.prefreeze_resolution") else "base"
        print(f"leg=auto resolved to {leg!r} for this tree")

    if leg == "base":
        print("leg=base -- no pre-freeze resolution; the exporter resolves, as at bcc0bfe")
        code, lines = _measure(payload, "BASE")
        return _verdict(code, lines)

    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    report = run_prefreeze_resolution(payload, strict_db=False)
    compounds = report.get("compounds") or {}
    print("leg=tip -- pre-freeze resolution ran on the payload before the freeze")
    print(f"    rename_map        : {json.dumps(compounds.get('rename_map'), ensure_ascii=False)}")
    print(f"    aliases_preserved : {json.dumps(compounds.get('aliases_preserved'), ensure_ascii=False)}")
    print(f"    refs updated      : {len(compounds.get('references_updated') or [])}")
    for update in compounds.get("references_updated") or []:
        print(f"        {update['pointer']}: {update['from']!r} -> {update['to']!r}")

    with tempfile.TemporaryDirectory(prefix="c050probe") as raw_tmp:
        canonical = _freeze(payload, Path(raw_tmp))
        code, lines = _measure(canonical, "TIP")
    return _verdict(code, lines)


if __name__ == "__main__":
    sys.exit(main())
