"""C-045's measurements: the species rename moves to BEFORE the freeze (D-016).

Every mode runs the **same production region** at both SHAs -- pre-freeze
resolution, then the freeze, then the IR build -- so nothing here is a symbol
check. At ``0ec64d2c`` that region has no species canonicalizer registered and
``build_pwml_ir`` still does the rename; at the tip the pre-freeze stage does it
and ``build_pwml_ir`` only replays the record. Same command, same file, opposite
verdicts, and the difference is the production path.

``--mode census``  (A3, A7)
    Over every committed payload that carries ``entities.species``: the
    ``name_canonicalization["species"]`` log the IR report ends up with, the
    ``preflight["species"]`` set ``_emit_canonicalization_preflight`` publishes
    from ``_species_at_risk``, and the emitted IR species names. A3 holds when
    the base and tip dumps are equal; A7 holds when the preflight sets are.

``--mode trace``  (A1)
    Counts every entry into ``ir._canonicalize_species_offline`` on one real run
    and records which function called it. A1 wants "exactly once, before the
    freeze", which is a statement about the caller, not about the symbol.

``--mode g9``  (A8)
    Drives a species the ladder genuinely renames -- a strain-qualified name with
    a taxonomy id, which step 4 normalizes offline and with no database -- and
    asks the only question merge rule 8 asks: does the **frozen** payload carry
    the canonical name, or does the exporter still have to produce it? Exits 1 at
    base, 0 at tip.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()

import t2pw  # noqa: E402
from t2pw.pipeline.canonical_hash import canonical_graph_sha256  # noqa: E402
from t2pw.pwml import ir as ir_module  # noqa: E402

PATHWAY_CONTEXT: Dict[str, Any] = {
    "pathway_name": "Heme biosynthesis",
    "name": "Heme biosynthesis",
    "pathway_subject": "Metabolic",
    "subject": "Metabolic",
}
SOURCE_LEG = "runs_verify/2026-08-04_1647/papers/PMC12856317/strict/final_mapped.json"

#: A strain-qualified organism with a taxonomy id. Step 4 of the ladder collapses
#: it to the binomial deterministically, offline, with no database -- which is
#: what makes this measurement valid in a worktree that has no ``.env`` (P4-01).
G9_SPECIES = {"name": "Lactococcus lactis subsp. lactis KF147", "taxonomy_id": "1091041"}
G9_EXPECTED = "Lactococcus lactis"

#: Where the committed payloads are read from. Separate from where ``t2pw`` is
#: imported from **on purpose**: the base leg runs base *code* over byte-identical
#: *inputs*, so any difference the census reports is the production path and
#: cannot be the corpus. ``_repo_root.add_src_to_path`` inserts the containing
#: checkout's ``src`` at ``sys.path[0]``, so the base leg is driven by running a
#: copy of this file out of a base export -- never by re-pointing ``PYTHONPATH``,
#: which that insert would silently override (S2).
CORPUS_ROOT: Path = REPO_ROOT


def _corpus_payloads() -> List[Tuple[str, Dict[str, Any]]]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=CORPUS_ROOT, capture_output=True, check=True,
    ).stdout.decode("utf-8").splitlines()
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name in listed:
        if not name.endswith(("final_mapped.json", "merged_payload.json", "stage1_payload.json")):
            continue
        try:
            payload = json.loads((CORPUS_ROOT / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(
            (payload.get("entities") or {}).get("species"), list
        ):
            out.append((name, payload))
    return sorted(out)


def _production_region(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``run_prefreeze_resolution`` then ``build_pwml_ir`` -- the real order."""

    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    run_prefreeze_resolution(payload, strict_db=False)
    return ir_module.build_pwml_ir(
        payload,
        pathway_name=str(PATHWAY_CONTEXT["pathway_name"]),
        pathway_subject=str(PATHWAY_CONTEXT["pathway_subject"]),
        strict_db=False,
    )


def _census() -> int:
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError

    rows: Dict[str, Any] = {}
    for name, payload in _corpus_payloads():
        try:
            ir, report = _production_region(deepcopy(payload))
        except PrefreezeResolutionError as exc:
            # Recorded, not skipped: a file that stops the region at one SHA and
            # not the other is precisely the kind of divergence A3 is looking for.
            rows[name] = {"prefreeze_error": exc.code}
            continue
        preflight = report.get("preflight") or {}
        rows[name] = {
            "name_canonicalization_species": (report.get("name_canonicalization") or {}).get(
                "species", []
            ),
            "preflight_species": preflight.get("species", []),
            "warning_species": [
                issue.get("species")
                for issue in report.get("warnings") or []
                if issue.get("code") == "noncanonical_names_collision_risk"
            ],
            "ir_species": [row.get("name") for row in ir.get("species") or []],
        }
    renamed = sum(len(row.get("name_canonicalization_species") or []) for row in rows.values())
    at_risk = sum(len(row.get("preflight_species") or []) for row in rows.values())
    errors = sorted(name for name, row in rows.items() if "prefreeze_error" in row)
    print(json.dumps(
        {"files": len(rows), "renames": renamed, "at_risk": at_risk, "errored": len(errors)},
        sort_keys=True,
    ))
    for name in errors:
        print(f"  prefreeze_error {rows[name]['prefreeze_error']}  {name}")
    print("--- CENSUS-JSON-BEGIN ---")
    print(json.dumps(rows, sort_keys=True, indent=1))
    print("--- CENSUS-JSON-END ---")
    return 0


def _g9_payload() -> Dict[str, Any]:
    payload = json.loads((CORPUS_ROOT / SOURCE_LEG).read_text(encoding="utf-8"))
    payload.setdefault("metadata", {}).update(PATHWAY_CONTEXT)
    payload["entities"]["species"] = [dict(G9_SPECIES)]
    return payload


def _standalone() -> int:
    """The exact delta on ``build_pwml_ir`` called WITHOUT the pre-freeze stage.

    C-040's golden (``test_build_pwml_ir_matches_the_pre_extraction_golden``)
    hashes that call, and it is the one caller shape the move cannot preserve:
    with the ladder upstream, a standalone exporter run has no way to know a
    species was only *deterministically* normalized without running the ladder a
    second time -- which is precisely the duplication D-016 forbids. This dumps
    the whole report so the delta can be characterized exactly rather than
    asserted, and so a reviewer can confirm it is confined to the two keys the
    preflight writes.
    """

    rows: Dict[str, Any] = {}
    for name, payload in _corpus_payloads():
        ir, report = ir_module.build_pwml_ir(
            deepcopy(payload),
            pathway_name=str(PATHWAY_CONTEXT["pathway_name"]),
            pathway_subject=str(PATHWAY_CONTEXT["pathway_subject"]),
            strict_db=False,
        )
        rows[name] = {
            "report": json.loads(json.dumps(report, sort_keys=True, default=str)),
            "ir_species": [row.get("name") for row in ir.get("species") or []],
            "ir_entity_names": {
                bucket: [row.get("name") for row in (ir.get("entities") or {}).get(bucket) or []]
                for bucket in sorted((ir.get("entities") or {}))
            },
        }
    print(json.dumps({"files": len(rows)}, sort_keys=True))
    print("--- CENSUS-JSON-BEGIN ---")
    print(json.dumps(rows, sort_keys=True, indent=1))
    print("--- CENSUS-JSON-END ---")
    return 0


def _trace() -> int:
    payload = _g9_payload()

    original = ir_module._canonicalize_species_offline
    calls: List[Dict[str, Any]] = []

    def _counting(record: Any, **kwargs: Any) -> str:
        frame = sys._getframe(1)
        calls.append({
            "name": record.get("name") if isinstance(record, dict) else None,
            "caller": frame.f_code.co_name,
            "caller_file": Path(frame.f_code.co_filename).name,
        })
        return original(record, **kwargs)

    # Both bindings, so the count cannot depend on import order: the exporter
    # looks the ladder up as a module global, and the pre-freeze stage holds an
    # imported reference to it. At the base SHA the second binding does not
    # exist, which is itself part of what the two legs show.
    import t2pw.pwml.prefreeze_resolution as prefreeze_module  # noqa: PLC0415

    patched = [ir_module] + (
        [prefreeze_module] if hasattr(prefreeze_module, "_canonicalize_species_offline") else []
    )
    for module in patched:
        module._canonicalize_species_offline = _counting
    print(f"  patched bindings: {[m.__name__ for m in patched]}")
    try:
        with tempfile.TemporaryDirectory(prefix="c045tr") as raw:
            frozen = _freeze(_prefreeze(payload), Path(raw))
        ir, _ = ir_module.build_pwml_ir(
            deepcopy(frozen),
            pathway_name=str(PATHWAY_CONTEXT["pathway_name"]),
            pathway_subject=str(PATHWAY_CONTEXT["pathway_subject"]),
            strict_db=False,
        )
    finally:
        for module in patched:
            module._canonicalize_species_offline = original

    print(json.dumps({"invocations": len(calls), "calls": calls}, sort_keys=True, indent=1))
    print(f"ir species: {[row.get('name') for row in ir.get('species') or []]}")
    from_prefreeze = [c for c in calls if c["caller_file"] == "prefreeze_resolution.py"]
    from_ir = [c for c in calls if c["caller_file"] == "ir.py"]
    print(f"A1: {len(calls)} invocation(s); pre-freeze={len(from_prefreeze)} exporter={len(from_ir)}")
    if len(calls) == 1 and len(from_prefreeze) == 1 and not from_ir:
        print("A1: PASSED -- one species row, one invocation, and the caller is the "
              "pre-freeze stage; the exporter never enters the ladder.")
        return 0
    print("A1: FAILED -- see the call list above.")
    return 1


def _prefreeze(payload: Dict[str, Any]) -> Dict[str, Any]:
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    candidate = deepcopy(payload)
    report = run_prefreeze_resolution(candidate, strict_db=False)
    species = report.get("species")
    print(f"  prefreeze species summary: {json.dumps(species, sort_keys=True, default=str)}")
    return candidate


def _freeze(payload: Dict[str, Any], tmp: Path) -> Dict[str, Any]:
    from t2pw.app.streamlit_app import freeze_canonical_payload
    from t2pw.pipeline.export_mode import DEFAULT_EXPORT_MODE

    outputs = tmp / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    result = freeze_canonical_payload(
        payload, strict_db=False, outputs_dir=outputs,
        pathway_context=dict(PATHWAY_CONTEXT), export_mode=DEFAULT_EXPORT_MODE,
        research_mode=True, tmp=tmp,
    )
    frozen = result.get("payload") or {}
    if not frozen:
        print(f"  quarantine refused the payload (quarantine_ok={result.get('quarantine_ok')}); "
              "falling back to the pre-freeze object")
        return payload
    return frozen


def _g9() -> int:
    payload = _g9_payload()

    with tempfile.TemporaryDirectory(prefix="c045g9") as raw:
        frozen = _freeze(_prefreeze(payload), Path(raw))

    frozen_species = [row.get("name") for row in (frozen.get("entities") or {}).get("species") or []]
    ir, report = ir_module.build_pwml_ir(
        deepcopy(frozen),
        pathway_name=str(PATHWAY_CONTEXT["pathway_name"]),
        pathway_subject=str(PATHWAY_CONTEXT["pathway_subject"]),
        strict_db=False,
    )
    ir_species = [row.get("name") for row in ir.get("species") or []]
    observed = {
        "raw_name": G9_SPECIES["name"],
        "expected_canonical": G9_EXPECTED,
        "frozen_payload_species": frozen_species,
        "ir_species": ir_species,
        "canonical_graph_sha256": canonical_graph_sha256(deepcopy(frozen)),
        "ir_report_name_canonicalization": (report.get("name_canonicalization") or {}).get(
            "species", []
        ),
    }
    print(json.dumps(observed, sort_keys=True, indent=1))

    # ``Homo sapiens`` is in both lists at both SHAs -- ``build_pwml_ir`` hydrates
    # it from ``biological_states``, which is not a rename and not this card's
    # subject. The question is only which side of the freeze the *strain*
    # collapse happens on, so ask it about that one organism.
    raw = str(G9_SPECIES["name"])
    if G9_EXPECTED in ir_species and raw in frozen_species:
        print(f"\nA8: the frozen payload still says {raw!r} while the exporter emits "
              f"{G9_EXPECTED!r} -- the species rename happens AFTER the freeze, so the "
              "canonical name is outside canonical_graph_sha256.")
        print("A8: FAILED (this is the expected behavioural failure at the base SHA)")
        return 1
    if G9_EXPECTED in frozen_species and raw not in frozen_species and raw not in ir_species:
        print(f"\nA8: the frozen payload already carries {G9_EXPECTED!r} and no longer "
              f"carries {raw!r}; the exporter changes nothing -- the rename landed "
              "BEFORE the freeze and is inside canonical_graph_sha256.")
        print("A8: PASSED")
        return 0
    print("\nA8: UNDETERMINED -- neither shape was observed; the ladder renamed nothing.")
    return 2


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("census", "standalone", "trace", "g9"), required=True)
    parser.add_argument(
        "--corpus-root", default=None,
        help="checkout to read the committed payloads from; defaults to this "
             "script's own checkout. The base leg points it at the tip worktree "
             "so both legs measure byte-identical inputs.",
    )
    args = parser.parse_args(argv)
    global CORPUS_ROOT
    if args.corpus_root:
        CORPUS_ROOT = Path(args.corpus_root).resolve()
    print(f"T2PW: {t2pw.__file__}")
    print(f"CORPUS_ROOT: {CORPUS_ROOT}")
    return {
        "census": _census, "standalone": _standalone, "trace": _trace, "g9": _g9,
    }[args.mode]()


if __name__ == "__main__":
    raise SystemExit(main())
