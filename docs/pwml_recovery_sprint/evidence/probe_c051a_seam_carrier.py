"""C-051a: the third export seam, and the ``db_resolution.available`` carrier.

Runs UNCHANGED at the base SHA and at the tip, so every section is a
behavioural base-vs-tip measurement rather than a symbol check (G9). Nothing
here imports a symbol this card adds: the carrier field is referenced by its
literal name, and the seam is measured by tracing frames, not by reading source.

WHY A FRAME TRACE AND NOT A MONKEYPATCH (charter A1). ``run_prefreeze_
resolution``'s ``canonicalizers`` default argument binds
``PREFREEZE_CANONICALIZERS`` at *definition* time, so patching a module
attribute is invisible to the call and reports a false zero -- C-045b's reviewer
hit exactly that. ``sys.setprofile`` sees the interpreter's own ``call`` events.

SECTIONS. ``seam`` (A1) drives ``run_pwml_export`` -- the third production
export entry point (D-033) -- over a raw extraction payload and records the
ordered call trace; at base, zero pre-freeze calls and a refused export.
``carrier`` (A3) runs two compound populations x two DB-reachability legs and
reports what the EXPORTER says about ``available`` beside the pre-freeze value
it must equal, with the row digests that show ``all_legacy`` indistinguishable
from the rows alone. ``warning`` (A4) is a reachable DB with no matching row:
at base the export claims "Resolution DB unavailable (db_not_configured)", a
false statement in ``report['preflight']``, which D-032 clause 6 rules
product-visible export content.

Usage::  <python> probe_c051a_seam_carrier.py --tmp <dir> --section all --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()

import t2pw  # noqa: E402

#: The payload key the pre-freeze sequence uses to carry the DB-reachability
#: verdict to the exporter. Referenced as a LITERAL so this file imports the
#: same names at both SHAs; the constant does not exist at base.
CARRIER_FIELD = "prefreeze_db_resolution"

#: ``ir._emit_canonicalization_preflight``'s message stem. The whole of A4 is
#: that this sentence is emitted about a DB that was in fact reachable.
FALSE_WARNING_STEM = "Resolution DB unavailable"


# Fixtures


class _CannedDb:
    """A reachable PathBank DB that answers with a Glycine row."""

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        return [{
            "id": 78, "name": "Glycine", "short_name": "Gly", "hmdb_id": "HMDB0000123",
            "kegg_id": "C00037", "chebi_id": "15428", "pubchem_cid": "750",
            "cas": "56-40-6", "biocyc_id": "GLY", "chemspider_id": "730",
            "drugbank_id": "DB00145", "pwc_id": "PW_C000123", "description": "canned",
            "synonyms": "Glycine; Gly",
        }]


class _EmptyReachableDb:
    """REACHABLE, and it has no row for the query. The A4 population.

    ``available()`` True is what makes ``_resolve_compound_rows`` build a live
    resolver and record ``db_resolution['available'] = True``; the empty answer
    is what leaves the compound un-canonicalized and therefore at risk. Both at
    once is the case the base misreports: a DB that WAS consulted, described in
    product-visible content as one that was not configured.
    """

    def available(self) -> bool:
        return True

    def _query(self, sql: str, params: Any) -> List[Dict[str, Any]]:  # noqa: ARG002
        return []


#: ``all_legacy`` is the decisive population (charter A3): every row carries a
#: ``pathbank_compound_id``, so ``compound_resolution.py`` takes its legacy-id
#: branch and ``continue``s BEFORE the resolver is consulted. Nothing about DB
#: reachability can reach such a row, so no exporter can infer it from them.
CARRIER_POPULATIONS: Dict[str, List[Dict[str, Any]]] = {
    "all_legacy": [
        {"name": "Glycine", "pathbank_compound_id": 78},
        {"name": "Pyruvic acid", "pathbank_compound_id": 91},
    ],
    "mixed": [
        {"name": "Glycine", "pathbank_compound_id": 78},
        {"name": "gly", "kegg_id": "C00037"},
    ],
}


def _shell_payload(compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": deepcopy(compounds),
        },
        "biological_states": [
            {"name": "cyto_state", "species": "Homo sapiens",
             "subcellular_location": "cytosol"}
        ],
        "element_locations": {"compound_locations": []},
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _stage6_extraction_payload() -> Dict[str, Any]:
    """A RAW extraction payload no canonicalizer has touched.

    Copied verbatim from ``tests/test_streamlit_stage8_export_contract.py``'s
    ``_stage6_extraction_payload`` (the pre-``prefrozen_when_compounded`` form),
    because that fixture is already pinned as one the Stage-8 gate, the
    pre-export contract and the IR build all accept -- so a refusal in this
    probe is the seam, never the fixture. Its compounds are ``all_legacy``,
    which is also the population no exporter can resolve by inference.
    """

    return {
        "metadata": {
            "name": "Caffeine demethylation",
            "pathway_name": "Caffeine demethylation",
            "subject": "Metabolic",
            "pathway_subject": "Metabolic",
        },
        "entities": {
            "species": [{"name": "Pseudomonas putida", "pathwhiz_id": 1}],
            "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
            "compounds": [
                {"name": "caffeine", "pathbank_compound_id": 101},
                {"name": "theobromine", "pathbank_compound_id": 102},
            ],
            "proteins": [
                {
                    "name": "NdmA",
                    "species": "Pseudomonas putida",
                    "mapped_ids": {"uniprot": "Q9I147"},
                }
            ],
            "protein_complexes": [
                {
                    "name": "NdmA complex",
                    "species": "Pseudomonas putida",
                    "generated": True,
                    "generation_reason": "single_protein_pathwhiz_wrapper",
                    "components": [{"name": "NdmA", "stoichiometry": 1}],
                }
            ],
        },
        "biological_states": [
            {
                "name": "Pseudomonas putida cytosol",
                "species": "Pseudomonas putida",
                "subcellular_location": "cytosol",
            }
        ],
        "processes": {
            "reactions": [
                {
                    "name": "caffeine demethylation",
                    "inputs": ["caffeine"],
                    "outputs": ["theobromine"],
                    "biological_state": "Pseudomonas putida cytosol",
                    "enzymes": [
                        {"entity": "NdmA complex", "entity_type": "protein_complex",
                         "role": "catalyst"}
                    ],
                    "modifiers": [
                        {"entity": "NdmA complex", "entity_type": "protein_complex",
                         "role": "catalyst"}
                    ],
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }


# seam -- A1, Part 1

#: The frames whose ``call`` events answer the question. ``build_pwml_ir`` is in
#: the set so that a trace with no pre-freeze call can be told apart from a run
#: that never reached the exporter at all.
TRACED = ("run_prefreeze_resolution", "build_pwml_ir", "resolve_compounds_prefreeze",
          "resolve_species_prefreeze", "_resolve_compound_rows")


def _seam(tmp: Path) -> Tuple[int, Dict[str, Any]]:
    from t2pw.app.streamlit_app import run_pwml_export

    print("\n=== seam: does run_pwml_export run the pre-freeze sequence? ===")
    project_root = tmp / "seam_project"
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    ref_path = REPO_ROOT / "reference" / "PW000001.pwml"

    trace: List[str] = []

    def _profiler(frame: Any, event: str, arg: Any) -> None:  # noqa: ARG001
        if event != "call":
            return
        name = frame.f_code.co_name
        if name in TRACED:
            trace.append(name)

    payload = _stage6_extraction_payload()
    sys.setprofile(_profiler)
    try:
        result = run_pwml_export(
            payload,
            pathway_name="Caffeine demethylation",
            pathway_description="",
            pathway_subject="Metabolic",
            project_root=project_root,
            ref_path=ref_path,
            strict_db=False,
        )
    finally:
        sys.setprofile(None)

    prefreeze_calls = trace.count("run_prefreeze_resolution")
    build_calls = trace.count("build_pwml_ir")
    order_ok = (
        prefreeze_calls == 1
        and build_calls >= 1
        and trace.index("run_prefreeze_resolution") < trace.index("build_pwml_ir")
    ) if (prefreeze_calls and build_calls) else False

    out_pwml = project_root / "outputs" / "pathway.pwml"
    observed: Dict[str, Any] = {
        "trace": trace,
        "prefreeze_calls_in_run_pwml_export": prefreeze_calls,
        "build_pwml_ir_calls": build_calls,
        # Merge rule 8 / PRODUCT_CONTRACT §5: resolution must not happen inside
        # the exporter. Counted by POSITION, not by name -- everything after the
        # ``build_pwml_ir`` call event is the exporter's own frame stack.
        "resolve_compound_rows_calls_total": trace.count("_resolve_compound_rows"),
        "resolve_compound_rows_after_build_pwml_ir": (
            trace[trace.index("build_pwml_ir"):].count("_resolve_compound_rows")
            if "build_pwml_ir" in trace else None
        ),
        "prefreeze_runs_exactly_once_before_build": order_ok,
        "result_ok": bool(result.get("ok")),
        "result_error": str(result.get("error") or "")[:400],
        "pathway_pwml_written": out_pwml.is_file(),
        "pathway_pwml_sha256": (
            hashlib.sha256(out_pwml.read_bytes()).hexdigest() if out_pwml.is_file() else ""
        ),
        # ``run_pwml_export`` deep-copies its input, so the caller's object is
        # expected to stay clean; the carrier is observable in the IR report
        # below, which is what the exporter actually emitted.
        "caller_payload_marker_expected_none": payload.get(CARRIER_FIELD),
        "ir_report_db_resolution": (
            result.get("pwml_ir_report") or {}
        ).get("db_resolution") if isinstance(result.get("pwml_ir_report"), dict) else None,
    }
    for key, value in observed.items():
        if key != "trace":
            print(f"    {key:44s}: {value}")
    print(f"    trace                                       : {trace}")

    ok = order_ok and observed["result_ok"] and observed["pathway_pwml_written"]
    print(f"    SEAM WIRED                                  : {ok}")
    return (0 if ok else 1), observed


# carrier -- A3, Part 2


def _carrier() -> Tuple[int, Dict[str, Any]]:
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    print("\n=== carrier: does the exporter's report tell the truth about the DB? ===")
    results: Dict[str, Any] = {}
    for population, rows in CARRIER_POPULATIONS.items():
        legs: Dict[str, Any] = {}
        row_blobs: Dict[str, str] = {}
        for leg, resolver in (("db_reachable", _CannedDb()), ("db_not_configured", None)):
            payload = _shell_payload(rows)
            prefreeze = run_prefreeze_resolution(
                payload, strict_db=False, db_resolver=resolver
            )
            prefreeze_available = (
                ((prefreeze.get("compounds") or {}).get("resolution_report") or {})
                .get("db_resolution") or {}
            ).get("available")
            row_blobs[leg] = json.dumps(
                payload["entities"]["compounds"], sort_keys=True, default=str
            )
            _ir, ir_report = build_pwml_ir(payload, strict_db=False)
            exporter_available = (ir_report.get("db_resolution") or {}).get(
                "available", "<ABSENT>"
            )
            legs[leg] = {
                "prefreeze_available": prefreeze_available,
                "exporter_available": exporter_available,
                "carrier_marker_on_payload": payload.get(CARRIER_FIELD),
                "faithful": exporter_available == prefreeze_available,
            }
        entry = {
            "legs": legs,
            "rows_identical_across_legs": row_blobs["db_reachable"] == row_blobs["db_not_configured"],
            "exporter_distinguishes_the_legs": (
                legs["db_reachable"]["exporter_available"]
                != legs["db_not_configured"]["exporter_available"]
            ),
            "both_legs_faithful": all(leg["faithful"] for leg in legs.values()),
        }
        results[population] = entry
        print(f"    {population}")
        for leg, value in legs.items():
            print(f"        {leg:20s} prefreeze={value['prefreeze_available']!r:>10} "
                  f"exporter={value['exporter_available']!r:>10} "
                  f"faithful={value['faithful']}")
        print(f"        rows identical across legs : {entry['rows_identical_across_legs']}")
        print(f"        exporter distinguishes     : {entry['exporter_distinguishes_the_legs']}")

    ok = all(
        entry["both_legs_faithful"] and entry["exporter_distinguishes_the_legs"]
        for entry in results.values()
    )
    results["_carrier_faithful_in_both_directions"] = ok
    print(f"    CARRIER FAITHFUL BOTH DIRECTIONS            : {ok}")
    return (0 if ok else 1), results


# warning -- A4

#: A compound the offline name index cannot answer, carrying an external id, no
#: legacy PathBank id and no DB row -- so it lands in the preflight's at-risk
#: set and the warning has something to be about.
AT_RISK_COMPOUND: List[Dict[str, Any]] = [
    {"name": "norbelladine", "hmdb_id": "HMDB9999999"},
]


def _warning() -> Tuple[int, Dict[str, Any]]:
    from t2pw.pwml.ir import build_pwml_ir
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    print("\n=== warning: a REACHABLE DB with no matching row ===")
    payload = _shell_payload(AT_RISK_COMPOUND)
    prefreeze = run_prefreeze_resolution(
        payload, strict_db=False, db_resolver=_EmptyReachableDb()
    )
    prefreeze_available = (
        ((prefreeze.get("compounds") or {}).get("resolution_report") or {})
        .get("db_resolution") or {}
    ).get("available")
    _ir, ir_report = build_pwml_ir(payload, strict_db=False)

    warnings = [str(item.get("message") or "") for item in (ir_report.get("warnings") or [])]
    false_claims = [text for text in warnings if FALSE_WARNING_STEM in text]
    observed = {
        "db_was_actually_reachable": prefreeze_available,
        "exporter_available": (ir_report.get("db_resolution") or {}).get(
            "available", "<ABSENT>"
        ),
        "preflight": ir_report.get("preflight"),
        "false_unavailability_claims": false_claims,
        "claim_count": len(false_claims),
    }
    for key, value in observed.items():
        print(f"    {key:34s}: {str(value)[:220]}")
    ok = prefreeze_available is True and not false_claims and ir_report.get("preflight") is None
    print(f"    NO FALSE CLAIM                    : {ok}")
    return (0 if ok else 1), observed




def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-051a seam + carrier probe")
    parser.add_argument("--tmp", required=True)
    parser.add_argument(
        "--section", default="all", choices=("all", "seam", "carrier", "warning")
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"T2PW: {t2pw.__file__}")

    code = 0
    payload: Dict[str, Any] = {"t2pw": t2pw.__file__}
    if args.section in ("all", "seam"):
        rc, payload["seam"] = _seam(tmp)
        code |= rc
    if args.section in ("all", "carrier"):
        rc, payload["carrier"] = _carrier()
        code |= rc
    if args.section in ("all", "warning"):
        rc, payload["warning"] = _warning()
        code |= rc

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print(f"\nmeasurements written to {args.out}")

    print("\nC-051a PROBE: " + ("PASSED" if code == 0 else "FAILED"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
