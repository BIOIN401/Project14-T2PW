"""C-045b: does the **CLI** export path run the pre-freeze sequence, once?

Measures ``t2pw.pwml.writer.run_pwml_pipeline_export`` -- the second production
entry point (``README.md:40`` -> ``scripts/run_pwml.py:12-16`` -> here), the one
``docs/pathwhiz_requirements.md`` §316-318 names by symbol.

SHA-invariant on purpose. The same command runs at ``d146be48`` and at this
card's tip and returns **opposite** verdicts, because what it measures is the
production path, not a symbol. At the base the trace shows
``run_prefreeze_resolution`` called **0** times and the exporter's own
``_resolve_compound_rows`` supplying the only compound resolution the CLI gets;
at the tip it shows the pre-freeze sequence running exactly once, both
canonicalizers with it. Nothing here imports a symbol that exists at only one of
the two SHAs.

The four sections
-----------------
``trace``   A2/A4. ``sys.setprofile`` counts real Python calls by code object --
            no monkeypatching, so ``PREFREEZE_CANONICALIZERS`` is exercised as
            the tuple production actually holds, and a wrapper cannot be
            mistaken for the thing it wraps. Reports call counts and first-call
            order.
``idem``    A5. A second pass of the whole sequence over an already-resolved
            payload must rename nothing, log nothing and move no byte of it; two
            exports of one input must be byte-identical.
``order``   A6 + the rationale for the seam's position: the measured reason it
            runs after ``normalize_process_payload`` rather than before.
``error``   A5. ``PrefreezeResolutionError`` out of the seam: the run must fail
            intelligibly (named code, nonzero exit), the input file must be
            byte-unchanged, and the report must say which row.
``review``  A5/A6/D-029. ``strict_db`` yields
            ``review_required = compounds: resolution_report_not_ok:db_unavailable``.
            D-029 (LOCKED) rules that outcome must **not** abort by itself, and
            merge rule 7 keeps the pathway. The export must still complete and
            the verdict must still be visible.

Usage::

    <python> probe_c045b_cli_prefreeze_seam.py --tmp C:/t/c045b/probe
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from _repo_root import add_src_to_path

add_src_to_path()

import t2pw  # noqa: E402
from t2pw.pwml import ir as ir_mod  # noqa: E402
from t2pw.pwml import prefreeze_resolution as pf  # noqa: E402
from t2pw.pwml.writer import run_pwml_pipeline_export  # noqa: E402

#: The REV-045a case: a taxonomy-identified strain the offline index does not
#: cover, so the species ladder lands on rung 4 (deterministic strain
#: normalization), plus two compounds the offline index *does* answer by KEGG id
#: so the compound canonicalizer has real work on the same payload.
PAYLOAD: Dict[str, Any] = {
    "entities": {
        "species": [{"name": "Lactococcus lactis subsp. lactis KF147", "taxonomy_id": "1091041"}],
        "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
        "compounds": [
            {"name": "gly", "kegg_id": "C00037"},
            {"name": "pyr", "kegg_id": "C00022"},
        ],
        "proteins": [{
            "name": "GlyA", "species": "Lactococcus lactis subsp. lactis KF147",
            "uniprot": "Q00001", "pathbank_protein_id": 201,
        }],
        "protein_complexes": [{
            "name": "GlyA complex", "species": "Lactococcus lactis subsp. lactis KF147",
            "components": ["GlyA"], "generated": True,
            "generation_reason": "single_protein_pathwhiz_wrapper",
        }],
    },
    "biological_states": [
        {"name": "cyto_state", "species": "Lactococcus lactis subsp. lactis KF147",
         "subcellular_location": "cytosol"},
    ],
    "element_locations": {
        "compound_locations": [
            {"compound": "gly", "biological_state": "cyto_state"},
            {"compound": "pyr", "biological_state": "cyto_state"},
        ],
        "protein_locations": [{"protein": "GlyA", "biological_state": "cyto_state"}],
    },
    "processes": {
        "reactions": [{
            "name": "gly to pyr", "inputs": ["gly"], "outputs": ["pyr"],
            "biological_state": "cyto_state",
            "enzymes": [{"entity": "GlyA complex", "entity_type": "protein_complex"}],
        }],
        "transports": [],
        "interactions": [],
    },
}

#: label -> code object. Names, not wrappers: the profiler sees the real frames.
WATCHED: Tuple[Tuple[str, Any], ...] = (
    ("run_prefreeze_resolution", pf.run_prefreeze_resolution.__code__),
    ("resolve_compounds_prefreeze", pf.resolve_compounds_prefreeze.__code__),
    ("resolve_species_prefreeze", pf.resolve_species_prefreeze.__code__),
    ("build_pwml_ir", ir_mod.build_pwml_ir.__code__),
    ("_canonicalize_species_offline", ir_mod._canonicalize_species_offline.__code__),
)


def _args(input_path: Path, out_dir: Path, *, strict: bool) -> SimpleNamespace:
    return SimpleNamespace(
        input_path=str(input_path), out_dir=str(out_dir),
        ref=str(Path(t2pw.__file__).resolve().parents[2] / "reference" / "PW000001.pwml"),
        name="Generated Pathway", subject="Metabolic", description="",
        width=3200, height=1400, background_color="#FFFFFF",
        non_strict_db=not strict,
    )


def _write(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _export(tmp: Path, leg: str, payload: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    out = tmp / leg
    out.mkdir(parents=True, exist_ok=True)
    return run_pwml_pipeline_export(_args(_write(tmp / f"{leg}.json", payload), out, strict=strict))


def _prefreeze_report(result: Dict[str, Any]) -> Dict[str, Any] | None:
    """The seam's report, or ``None`` where the CLI has no seam.

    At ``d146be48`` this key does not exist, because nothing wired the pre-freeze
    sequence into this entry point. Reading it as absence rather than crashing is
    what lets one unmodified script produce the base leg and the tip leg.
    """

    path = result.get("prefreeze_resolution_report")
    if not path:
        print("    prefreeze report              : ABSENT -- this entry point runs no "
              "pre-freeze sequence")
        return None
    return json.loads(Path(str(path)).read_text(encoding="utf-8"))


def _trace(tmp: Path) -> int:
    """A2 + A4: which stages ran, how many times, in what order."""

    counts: Dict[str, int] = {label: 0 for label, _ in WATCHED}
    order: List[str] = []
    by_code = {code: label for label, code in WATCHED}

    def profile(frame: Any, event: str, _arg: Any) -> None:
        if event != "call":
            return
        label = by_code.get(frame.f_code)
        if label is not None:
            counts[label] += 1
            order.append(label)

    sys.setprofile(profile)
    try:
        result = _export(tmp, "trace", deepcopy(PAYLOAD))
    finally:
        sys.setprofile(None)

    print("=== A2/A4 invocation trace over ONE run_pwml_pipeline_export ===")
    for label, _ in WATCHED:
        print(f"    {label:<32}: {counts[label]}")
    print(f"    first-call order              : {' -> '.join(dict.fromkeys(order))}")

    failures: List[str] = []
    if counts["run_prefreeze_resolution"] != 1:
        failures.append("A4: the pre-freeze sequence did not run exactly once")
    for label in ("resolve_compounds_prefreeze", "resolve_species_prefreeze"):
        if counts[label] != 1:
            failures.append(f"A4: {label} ran {counts[label]} times, expected 1")
    if counts["build_pwml_ir"] != 1:
        failures.append("A4: build_pwml_ir did not run exactly once")
    seen = list(dict.fromkeys(order))
    if "run_prefreeze_resolution" not in seen or "build_pwml_ir" not in seen:
        failures.append("A2: a required stage never ran on the CLI path")
    elif seen.index("run_prefreeze_resolution") > seen.index("build_pwml_ir"):
        failures.append("A2: the pre-freeze sequence ran AFTER the exporter")

    report = _prefreeze_report(result)
    if report is None:
        failures.append("A2: the CLI export produced no pre-freeze resolution report")
    else:
        print(f"    compounds rename_map          : {report['compounds']['rename_map']}")
        print(f"    species   rename_map          : {report['species']['rename_map']}")
        if not report["compounds"]["rename_map"]:
            failures.append("A2: resolve_compounds_prefreeze renamed nothing -- vacuous")
        if not report["species"]["rename_map"]:
            failures.append("A1: resolve_species_prefreeze renamed nothing -- vacuous")

    pwml = Path(result["pwml_file"]).read_text(encoding="utf-8")
    print(f"    exported organism             : "
          f"{'Lactococcus lactis subsp. lactis KF147' if 'KF147' in pwml else 'Lactococcus lactis'}")
    if "KF147" in pwml:
        failures.append("A1: the un-normalized strain name still reaches pathway.pwml")
    for line in failures:
        print(f"  FAIL {line}")
    return 1 if failures else 0


def _idem(tmp: Path) -> int:
    """A5: the stage is a no-op over an already-resolved payload, and the export
    is deterministic.

    Two separate claims, deliberately measured separately -- ``_order`` below
    shows why conflating them would have charged this card with a defect it does
    not own.
    """

    print("\n=== A5 idempotence ===")
    failures: List[str] = []

    # (a) the STAGE. Resolve once, then run the whole sequence again over the
    #     same object: no rename, no fresh log, and not one byte of the payload
    #     moved by the second pass.
    once = deepcopy(PAYLOAD)
    pf.run_prefreeze_resolution(once, strict_db=False)
    settled = deepcopy(once)
    again = pf.run_prefreeze_resolution(once, strict_db=False)
    unchanged = once == settled
    print(f"    2nd-pass renames              : compounds={again['compounds']['renamed']} "
          f"species={again['species']['renamed']}")
    print(f"    2nd-pass species log          : {again['species']['name_canonicalization']}")
    print(f"    payload unchanged by 2nd pass : {unchanged}")
    if again["compounds"]["renamed"] or again["species"]["renamed"]:
        failures.append("A5: the second pass renamed something")
    if again["species"]["name_canonicalization"]:
        failures.append("A5: the second pass logged a fresh canonicalization")
    if not unchanged:
        failures.append("A5: the second pass mutated an already-resolved payload")

    # (b) the EXPORT. Same input file twice -> byte-identical PWML.
    first = _export(tmp, "idem-a", deepcopy(PAYLOAD))
    second = _export(tmp, "idem-b", deepcopy(PAYLOAD))
    same = Path(first["pwml_file"]).read_bytes() == Path(second["pwml_file"]).read_bytes()
    print(f"    pathway.pwml byte-identical   : {same}")
    if not same:
        failures.append("A5: two exports of the same input produced different PWML")

    for line in failures:
        print(f"  FAIL {line}")
    return 1 if failures else 0


def _order(tmp: Path) -> int:
    """Why the seam sits AFTER ``normalize_process_payload``, measured.

    ``process_normalizer`` carries a participant-name synonym map
    (``process_normalizer.py:82``, ``"pyruvic acid" -> "pyruvate"``) that rewrites
    reaction references but not the compound rows they point at, and then adds
    the now-dangling name as a **new compound**. Hand it a payload whose compound
    names are already canonical and it invents a row and splits the connectivity.

    So the seam runs after normalization, which is also where D-015 clause 7
    wants it -- "finish all network-dependent resolution before the freeze", on
    the payload that is actually frozen. The rejected ordering is measured here
    rather than argued, because it is the ordering a reader would otherwise
    assume from the Streamlit seam's position.

    This is a **finding about ``process_normalizer``, not about this seam**: the
    third leg below runs the normalizer alone, with no pre-freeze call anywhere,
    and reproduces it. It is out of this card's boundary and is not fixed here.
    """

    from t2pw.pipeline.process_normalizer import normalize_process_payload

    print("\n=== ordering: normalize -> prefreeze, and why not the reverse ===")

    shipped = _export(tmp, "order-shipped", deepcopy(PAYLOAD))
    ir = json.loads(Path(shipped["pwml_ir"]).read_text(encoding="utf-8"))
    shipped_names = [row["name"] for row in ir["entities"]["compounds"]]

    reversed_payload = deepcopy(PAYLOAD)
    pf.run_prefreeze_resolution(reversed_payload, strict_db=False)
    reversed_norm, reversed_report = normalize_process_payload(reversed_payload)
    reversed_names = [row.get("name") for row in reversed_norm["entities"]["compounds"]]

    # The same normalizer, on the same already-canonical names, with no
    # pre-freeze call in the leg at all.
    control = deepcopy(PAYLOAD)
    for row, name in zip(control["entities"]["compounds"], ("Glycine", "Pyruvic acid")):
        row["name"] = name
    control["processes"]["reactions"][0]["inputs"] = ["Glycine"]
    control["processes"]["reactions"][0]["outputs"] = ["Pyruvic acid"]
    for row, name in zip(control["element_locations"]["compound_locations"],
                         ("Glycine", "Pyruvic acid")):
        row["compound"] = name
    control_norm, control_report = normalize_process_payload(control)
    control_names = [row.get("name") for row in control_norm["entities"]["compounds"]]

    print(f"    shipped  normalize->prefreeze : {shipped_names}")
    print(f"    rejected prefreeze->normalize : {reversed_names} "
          f"(entities_added_as_compounds="
          f"{reversed_report['summary']['entities_added_as_compounds']})")
    print(f"    control  normalizer alone     : {control_names} "
          f"(entities_added_as_compounds="
          f"{control_report['summary']['entities_added_as_compounds']})")

    failures = []
    if len(shipped_names) != len(PAYLOAD["entities"]["compounds"]):
        failures.append("A6: the shipped ordering changed the compound row count")
    if control_report["summary"]["entities_added_as_compounds"] == 0:
        failures.append("the process_normalizer finding did not reproduce in the control leg; "
                        "re-measure before quoting the ordering rationale")
    for line in failures:
        print(f"  FAIL {line}")
    return 1 if failures else 0


def _error(tmp: Path) -> int:
    """A5: a raise is reported intelligibly and leaves the input untouched."""

    payload = deepcopy(PAYLOAD)
    # A rename onto an organism another row keeps -- C-045 correction round 1's
    # AMBIGUOUS_RENAME_TARGET, one of the six codes the seam can see.
    payload["entities"]["species"].append({"name": "Lactococcus lactis", "taxonomy_id": "1358"})
    out = tmp / "error"
    out.mkdir(parents=True, exist_ok=True)
    src = _write(tmp / "error.json", payload)
    before = src.read_bytes()
    result = run_pwml_pipeline_export(_args(src, out, strict=False))

    print("\n=== A5 error semantics: PrefreezeResolutionError on the CLI path ===")
    print(f"    summary ok                    : {result['ok']}")
    print(f"    summary error                 : {result.get('error')}")
    report = _prefreeze_report(result)
    if report is not None:
        print(f"    report code                   : {report.get('code')}")
        print(f"    report details                : "
              f"{json.dumps(report.get('details'), sort_keys=True)}")
    print(f"    input file byte-unchanged     : {src.read_bytes() == before}")
    print(f"    pathway.pwml written          : {(out / 'pathway.pwml').exists()}")
    failures = []
    if result["ok"] is not False or "AMBIGUOUS_RENAME_TARGET" not in str(result.get("error")):
        failures.append("A5: the raise was not reported intelligibly")
    if src.read_bytes() != before:
        failures.append("A5: the input payload was mutated by a failing run")
    if (out / "pathway.pwml").exists():
        failures.append("A6: a PWML was written despite a refused canonicalization")
    for line in failures:
        print(f"  FAIL {line}")
    return 1 if failures else 0


def _review(tmp: Path) -> int:
    """A5/A6 + D-029: review_required is surfaced and does not kill the run."""

    result = _export(tmp, "review", deepcopy(PAYLOAD), strict=True)

    print("\n=== D-029: an unreachable DB is review_required, not death ===")
    report = _prefreeze_report(result)
    failures = []
    if report is None:
        failures.append("D-029: no pre-freeze verdict is produced on this entry point")
    else:
        print(f"    prefreeze ok                  : {report['ok']}")
        print(f"    review_required               : {report['review_required']}")
        print(f"    surfaced in summary           : {result.get('prefreeze_review_required')}")
        if report["review_required"] != {"compounds": "resolution_report_not_ok:db_unavailable"}:
            failures.append("D-029: the expected review_required verdict was not produced")
        if result.get("prefreeze_review_required") != report["review_required"]:
            failures.append("D-029: the verdict is not surfaced to the caller")
    print(f"    export still produced PWML    : {'pwml_file' in result}")
    if "pwml_file" not in result:
        failures.append("D-029/merge rule 7: review_required aborted the export")
    for line in failures:
        print(f"  FAIL {line}")
    return 1 if failures else 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-045b CLI pre-freeze seam probe")
    parser.add_argument("--tmp", required=True)
    args = parser.parse_args(argv)
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"T2PW: {t2pw.__file__}")

    code = _trace(tmp) | _idem(tmp) | _order(tmp) | _error(tmp) | _review(tmp)
    print("\nC-045b PROBE: " + ("PASSED" if code == 0 else "FAILED"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
