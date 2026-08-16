"""C-051: the exporter asserts a compound verdict it no longer produces.

**SHA-invariant.** The same file runs at the dispatch base ``328862ab`` and at
this card's tip and returns opposite verdicts, because every section measures
*behaviour*, not a symbol. Nothing here imports a name that exists at only one
of the two SHAs; the one place a tip-only exception type is mentioned it is read
with :func:`getattr` and used to *label* a refusal that was already established
by observing what the exporter did.

Sections
--------
``refuse``   A2 / A7 / A8. Drive one unresolved compound row into
             ``build_pwml_ir`` with no pre-freeze stage anywhere in the leg.
             **BASE**: no raise, and the exporter renames the row and stamps a
             ``db_status`` the payload never had -- a biological repair after the
             freeze, which is the behaviour this card removes. **TIP**: a named
             refusal that identifies the offending row. This is the G9
             behavioural proof and it fails at the base by *doing the repair*,
             not by missing a symbol.

``identity`` A5 / D-027, all four cases, at **both** ``pathwhiz_id``
             materialization sites -- ``_entity_record`` and
             ``_component_record``. Compares the row as the freeze left it
             against the record the exporter built from it, so a value that
             appears only in the IR is a post-freeze materialization by
             definition.

``cli``      A9 / D-032 clause 4. ``sys.setprofile`` over one
             ``run_pwml_pipeline_export``: compound resolution must still happen
             exactly once, before the exporter, on the CLI entry point after the
             ``ir.py`` call site is gone. Module-attribute patching is
             deliberately NOT used -- ``run_prefreeze_resolution`` binds
             ``PREFREEZE_CANONICALIZERS`` as a default argument at definition
             time, so a patched attribute is invisible to it and reports a false
             zero (found by ``REV-045b``). The Streamlit entry point is measured
             in the same way through ``freeze_canonical_payload``.

``corpus``   A4 / A8. Export all 32 committed leg fixtures through the CLI and
             emit ``{leg: sha256(pathway.pwml), ...}`` plus the compound row
             names. Run it at both SHAs and diff the two files: identical
             digests are what shows the deleted pass was redundant rather than
             lossy, and identical row name lists are what shows no biological
             gate moved.

Usage::

    <python> probe_c051_assert_only.py --tmp C:/t/c051/probe [--corpus-root <dir>]
             [--section refuse|identity|cli|corpus|all] [--out <json>]

``--corpus-root`` exists for the base leg: ``runs/`` and ``runs_verify/`` are
committed artifacts this card does not touch (verified: ``git diff
328862ab..HEAD -- runs runs_verify`` is empty), and the hash-verified base tree
does not carry them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()

import t2pw  # noqa: E402
from t2pw.pwml import ir as ir_mod  # noqa: E402
from t2pw.pwml.writer import run_pwml_pipeline_export  # noqa: E402

VERDICT_FIELD = "db_status"

#: One compound the offline index answers (``gly`` -> ``Glycine`` by KEGG id),
#: so the base leg has a real rename to perform and the measurement is not
#: vacuous.
UNRESOLVED_PAYLOAD: Dict[str, Any] = {
    "entities": {
        "species": [{"name": "Homo sapiens", "pathwhiz_id": 1}],
        "subcellular_locations": [{"name": "cytosol", "pathwhiz_id": 2}],
        "compounds": [{"name": "gly", "kegg_id": "C00037"}],
    },
    "biological_states": [
        {"name": "cyto_state", "species": "Homo sapiens", "subcellular_location": "cytosol"}
    ],
    "element_locations": {
        "compound_locations": [{"compound": "gly", "biological_state": "cyto_state"}]
    },
    "processes": {"reactions": [], "transports": [], "interactions": []},
}


def _ir_compounds(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list((ir.get("entities") or {}).get("compounds") or [])


# ---------------------------------------------------------------------------
# refuse -- A2 / A7 / A8
# ---------------------------------------------------------------------------


def _refuse() -> Tuple[int, Dict[str, Any]]:
    payload = deepcopy(UNRESOLVED_PAYLOAD)
    row_before = deepcopy(payload["entities"]["compounds"][0])
    observed: Dict[str, Any] = {
        "payload_row_name": row_before.get("name"),
        "payload_row_has_verdict": bool(str(row_before.get(VERDICT_FIELD) or "").strip()),
    }

    try:
        ir, _report = ir_mod.build_pwml_ir(payload, strict_db=False)
    except Exception as exc:  # noqa: BLE001
        observed["outcome"] = "refused"
        observed["exception_type"] = type(exc).__name__
        observed["exception_module"] = type(exc).__module__
        observed["message"] = str(exc)
        named = getattr(ir_mod, "UnresolvedCompoundRowError", None)
        observed["is_the_named_error"] = named is not None and isinstance(exc, named)
        observed["message_names_the_row"] = str(row_before.get("name")) in str(exc)
        observed["code"] = getattr(exc, "code", None)
    else:
        rows = _ir_compounds(ir)
        after = rows[0] if rows else {}
        observed["outcome"] = "exported"
        observed["ir_row_name"] = after.get("name")
        observed["ir_row_verdict"] = after.get(VERDICT_FIELD)
        observed["renamed_after_the_freeze"] = after.get("name") != row_before.get("name")
        observed["verdict_stamped_after_the_freeze"] = bool(
            str(after.get(VERDICT_FIELD) or "").strip()
        )

    print("=== refuse: one unresolved compound row into build_pwml_ir ===")
    for key, value in observed.items():
        print(f"    {key:<32}: {value}")

    failures: List[str] = []
    if observed["outcome"] == "exported":
        # The BASE shape. It is a PASS *as a base measurement* only when the
        # late repair is actually visible; a silent no-op would mean the payload
        # was already resolved and the leg proves nothing.
        if not (
            observed["renamed_after_the_freeze"]
            or observed["verdict_stamped_after_the_freeze"]
        ):
            failures.append(
                "the base leg is vacuous: the exporter neither renamed the row nor "
                "stamped a verdict, so there is no late repair to remove"
            )
        print("    LEG READS AS: BASE -- the exporter repaired biology post-freeze")
    else:
        if not observed["message_names_the_row"]:
            failures.append("A2: the refusal does not identify the offending row")
        if not observed["is_the_named_error"]:
            failures.append("A2: the refusal is not the card's named error type")
        print("    LEG READS AS: TIP -- the exporter refused instead of repairing")

    for line in failures:
        print(f"  FAIL {line}")
    return (1 if failures else 0), observed


# ---------------------------------------------------------------------------
# identity -- A5 / D-027
# ---------------------------------------------------------------------------

#: D-027's four cases. ``site`` names which materialization block the row's
#: class reaches: compounds/proteins go through ``_entity_record``, species and
#: the other component buckets through ``_component_record``.
IDENTITY_CASES: Tuple[Tuple[str, str, str, Dict[str, Any]], ...] = (
    (
        "1_valid_prefreeze_identity",
        "compounds",
        "_entity_record",
        {"name": "Glycine", "pathwhiz_id": 78, "kegg_id": "C00037"},
    ),
    (
        "2_missing_at_freeze",
        "compounds",
        "_entity_record",
        {"name": "Zzqxx unmatchable compound"},
    ),
    (
        "3_identity_only_in_mapping_metadata__compound",
        "compounds",
        "_entity_record",
        {
            "name": "Qqvv metadata-only compound",
            "mapping_meta": {"candidates": [{"pathbank_compound_id": 4242}]},
        },
    ),
    (
        "3b_identity_only_in_mapping_metadata__protein",
        "proteins",
        "_entity_record",
        {
            "name": "Qqvv metadata-only protein",
            "species": "Homo sapiens",
            "uniprot": "Q00099",
            "mapping_meta": {"candidates": [{"pathbank_protein_id": 5353}]},
        },
    ),
    (
        "3c_identity_only_in_mapping_metadata__species",
        "species",
        "_component_record",
        {
            "name": "Qqvv metadata-only organism",
            "mapping_meta": {"candidates": [{"pathbank_species_id": 6464}]},
        },
    ),
)

_IR_BUCKET = {"compounds": "compounds", "proteins": "proteins"}


def _identity() -> Tuple[int, Dict[str, Any]]:
    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    print("\n=== identity: D-027's four cases at BOTH pathwhiz_id sites ===")
    results: Dict[str, Any] = {}
    failures: List[str] = []

    for label, bucket, site, row in IDENTITY_CASES:
        payload = deepcopy(UNRESOLVED_PAYLOAD)
        payload["entities"]["compounds"] = []
        payload["element_locations"] = {"compound_locations": []}
        payload["entities"].setdefault(bucket, [])
        if bucket == "species":
            payload["entities"]["species"] = [deepcopy(row)]
            payload["biological_states"] = [
                {
                    "name": "cyto_state",
                    "species": row["name"],
                    "subcellular_location": "cytosol",
                }
            ]
        else:
            payload["entities"][bucket] = [deepcopy(row)]

        # The production sequence, exactly as both entry points run it.
        run_prefreeze_resolution(payload, strict_db=False)
        frozen_row = deepcopy(payload["entities"][bucket][0])
        ir, _report = ir_mod.build_pwml_ir(payload, strict_db=False)
        ir_rows = (ir.get("entities") or {}).get(_IR_BUCKET.get(bucket, bucket)) or []
        if bucket == "species":
            ir_rows = ir.get("species") or []
        ir_row = ir_rows[0] if ir_rows else {}

        frozen_id = frozen_row.get("pathwhiz_id")
        ir_id = ir_row.get("pathwhiz_id")
        entry = {
            "bucket": bucket,
            "site": site,
            "frozen_pathwhiz_id": frozen_id,
            "ir_pathwhiz_id": ir_id,
            "survives_unchanged": frozen_id is not None and frozen_id == ir_id,
            "newly_materialized_post_freeze": frozen_id is None and ir_id is not None,
            "dropped_by_the_exporter": frozen_id is not None and ir_id is None,
        }
        results[label] = entry
        print(f"    {label}")
        for key in (
            "bucket", "site", "frozen_pathwhiz_id", "ir_pathwhiz_id",
            "survives_unchanged", "newly_materialized_post_freeze",
            "dropped_by_the_exporter",
        ):
            print(f"        {key:<32}: {entry[key]}")

        if entry["dropped_by_the_exporter"]:
            failures.append(f"{label}: a correct pre-freeze identifier was dropped")

    compound_cases = [
        label for label, entry in results.items()
        if entry["bucket"] == "compounds" and entry["newly_materialized_post_freeze"]
    ]
    if compound_cases:
        failures.append(
            "D-027: a COMPOUND identity is still newly materialized after the freeze: "
            + ", ".join(compound_cases)
        )
    results["_compound_path_foreclosed"] = not compound_cases
    results["_other_classes_still_materializing"] = sorted(
        label for label, entry in results.items()
        if isinstance(entry, dict)
        and entry.get("bucket") != "compounds"
        and entry.get("newly_materialized_post_freeze")
    )
    print(f"    compound path foreclosed        : {results['_compound_path_foreclosed']}")
    print(
        "    other classes still materializing: "
        f"{results['_other_classes_still_materializing']}"
    )
    for line in failures:
        print(f"  FAIL {line}")
    return (1 if failures else 0), results


# ---------------------------------------------------------------------------
# cli -- A9 / D-032 clause 4
# ---------------------------------------------------------------------------


def _args(input_path: Path, out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_path=str(input_path), out_dir=str(out_dir),
        ref=str(REPO_ROOT / "reference" / "PW000001.pwml"),
        name="Generated Pathway", subject="Metabolic", description="",
        width=3200, height=1400, background_color="#FFFFFF", non_strict_db=True,
    )


def _cli(tmp: Path) -> Tuple[int, Dict[str, Any]]:
    from t2pw.pwml import compound_resolution as cr
    from t2pw.pwml import prefreeze_resolution as pf

    watched = {
        "run_prefreeze_resolution": pf.run_prefreeze_resolution.__code__,
        "resolve_compounds_prefreeze": pf.resolve_compounds_prefreeze.__code__,
        "resolve_species_prefreeze": pf.resolve_species_prefreeze.__code__,
        "_resolve_compound_rows": cr._resolve_compound_rows.__code__,
        "build_pwml_ir": ir_mod.build_pwml_ir.__code__,
    }
    by_code = {code: label for label, code in watched.items()}
    counts = {label: 0 for label in watched}
    order: List[str] = []

    def profile(frame: Any, event: str, _arg: Any) -> None:
        if event != "call":
            return
        label = by_code.get(frame.f_code)
        if label is not None:
            counts[label] += 1
            order.append(label)

    out = tmp / "cli"
    out.mkdir(parents=True, exist_ok=True)
    src = tmp / "cli.json"
    src.write_text(json.dumps(UNRESOLVED_PAYLOAD, indent=2), encoding="utf-8")

    sys.setprofile(profile)
    try:
        result = run_pwml_pipeline_export(_args(src, out))
    finally:
        sys.setprofile(None)

    ir = json.loads(Path(result["pwml_ir"]).read_text(encoding="utf-8"))
    rows = _ir_compounds(ir)
    seen = list(dict.fromkeys(order))
    observed: Dict[str, Any] = {
        "counts": counts,
        "first_call_order": seen,
        "exported_compound_names": [row.get("name") for row in rows],
        "exported_compound_verdicts": [row.get(VERDICT_FIELD) for row in rows],
        # ``raw_name`` is the extraction name a row was resolved FROM. Recorded
        # at both SHAs because ``test_pwml_writer`` pins it: the pre-freeze
        # fixed-point loop's ``_PROVENANCE_FIELDS`` does not include it, so the
        # question "is the drift mine or was it already production's?" is
        # answered by the base leg of this same probe, not by argument.
        "exported_compound_raw_names": [row.get("raw_name") for row in rows],
        "exported_compound_aliases": [row.get("aliases") for row in rows],
        "ok": result.get("ok"),
    }

    print("\n=== cli: one run_pwml_pipeline_export, sys.setprofile ===")
    for label in watched:
        print(f"    {label:<32}: {counts[label]}")
    print(f"    first-call order                : {' -> '.join(seen)}")
    print(f"    exported compound names         : {observed['exported_compound_names']}")
    print(f"    exported compound verdicts      : {observed['exported_compound_verdicts']}")
    print(f"    exported compound raw_names     : {observed['exported_compound_raw_names']}")
    print(f"    exported compound aliases       : {observed['exported_compound_aliases']}")

    failures: List[str] = []
    if counts["run_prefreeze_resolution"] != 1:
        failures.append("A9: the pre-freeze sequence did not run exactly once on the CLI")
    if counts["resolve_compounds_prefreeze"] != 1:
        failures.append("A9: compound resolution did not run pre-freeze on the CLI")
    if counts["_resolve_compound_rows"] < 1:
        failures.append("A9: compound resolution did not run AT ALL on the CLI path")
    if counts["build_pwml_ir"] != 1:
        failures.append("A9: build_pwml_ir did not run exactly once")
    if observed["exported_compound_names"] != ["Glycine"]:
        failures.append(
            "A9: the CLI export lost the compound canonicalization "
            f"({observed['exported_compound_names']})"
        )
    if not all(observed["exported_compound_verdicts"]):
        failures.append("A9: an exported compound row carries no resolution verdict")
    for line in failures:
        print(f"  FAIL {line}")
    return (1 if failures else 0), observed


# ---------------------------------------------------------------------------
# corpus -- A4 / A8
# ---------------------------------------------------------------------------


def _legs(corpus_root: Path) -> List[str]:
    legs: List[str] = []
    for top in ("runs", "runs_verify"):
        base = corpus_root / top
        if not base.is_dir():
            continue
        legs.extend(
            str(path.relative_to(corpus_root)).replace("\\", "/")
            for path in sorted(base.glob("*/papers/*/*/final_mapped.json"))
        )
    return legs


def _corpus(tmp: Path, corpus_root: Path) -> Tuple[int, Dict[str, Any]]:
    print(f"\n=== corpus: every committed leg through the CLI ({corpus_root}) ===")
    digests: Dict[str, Any] = {}
    for index, leg in enumerate(_legs(corpus_root)):
        out = tmp / "corpus" / f"leg{index:03d}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            result = run_pwml_pipeline_export(_args(corpus_root / leg, out))
        except Exception as exc:  # noqa: BLE001
            digests[leg] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        pwml = result.get("pwml_file")
        entry: Dict[str, Any] = {"ok": result.get("ok")}
        if pwml and Path(pwml).is_file():
            entry["pwml_sha256"] = hashlib.sha256(Path(pwml).read_bytes()).hexdigest()
        if result.get("pwml_ir") and Path(str(result["pwml_ir"])).is_file():
            ir = json.loads(Path(str(result["pwml_ir"])).read_text(encoding="utf-8"))
            rows = _ir_compounds(ir)
            entry["compound_names"] = [row.get("name") for row in rows]
            entry["compound_verdicts"] = [row.get(VERDICT_FIELD) for row in rows]
        digests[leg] = entry
    exported = sum(1 for entry in digests.values() if entry.get("pwml_sha256"))
    errored = sorted(leg for leg, entry in digests.items() if entry.get("error"))
    print(f"    legs measured                   : {len(digests)}")
    print(f"    legs that produced a pathway.pwml: {exported}")
    print(f"    legs that raised                 : {len(errored)}")
    for leg in errored[:5]:
        print(f"        {leg}: {digests[leg]['error'][:120]}")
    return (1 if not digests else 0), digests


# ---------------------------------------------------------------------------
# carrier -- can build_pwml_ir derive db_resolution.available from the rows?
# ---------------------------------------------------------------------------


class _CannedDb:
    """A reachable PathBank DB. ``available()`` True is what makes
    ``_resolve_compound_rows`` construct a live resolver and record
    ``db_resolution['available'] = True``."""

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


#: label -> the compound rows. ``all_legacy`` is the decisive population: every
#: row carries a ``pathbank_compound_id``, so ``_resolve_compound_rows`` takes
#: its legacy-id branch, which ``continue``s **before** the resolver is ever
#: consulted. Nothing about DB reachability can reach such a row.
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


def _carrier() -> Tuple[int, Dict[str, Any]]:
    """Two legs differing ONLY in DB reachability. Are the rows distinguishable?

    ``_emit_canonicalization_preflight`` (``ir.py:797-798``) reads
    ``report['db_resolution']['available']``, which the deleted ``ir.py:979``
    call used to populate. The orchestrator's ruling: set it inside
    ``build_pwml_ir`` **only** if it can be derived faithfully from the verdicts
    already on the rows, and stop and report if that would be a guess.
    """

    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    print("\n=== carrier: is db_resolution.available derivable from the rows? ===")
    results: Dict[str, Any] = {}
    for population, rows in CARRIER_POPULATIONS.items():
        leg_rows: Dict[str, Any] = {}
        leg_available: Dict[str, Any] = {}
        for leg, resolver in (("db_reachable", _CannedDb()), ("db_not_configured", None)):
            payload = deepcopy(UNRESOLVED_PAYLOAD)
            payload["entities"]["compounds"] = deepcopy(rows)
            payload["element_locations"] = {"compound_locations": []}
            payload["processes"] = {"reactions": [], "transports": [], "interactions": []}
            report = run_prefreeze_resolution(payload, strict_db=False, db_resolver=resolver)
            resolution = (report["compounds"] or {}).get("resolution_report") or {}
            leg_available[leg] = {
                "available": (resolution.get("db_resolution") or {}).get("available"),
                "reason": (resolution.get("db_resolution") or {}).get("reason"),
            }
            leg_rows[leg] = json.dumps(
                payload["entities"]["compounds"], sort_keys=True, default=str
            )
        rows_identical = leg_rows["db_reachable"] == leg_rows["db_not_configured"]
        available_differs = (
            leg_available["db_reachable"]["available"]
            != leg_available["db_not_configured"]["available"]
        )
        results[population] = {
            "available_by_leg": leg_available,
            "rows_identical_across_legs": rows_identical,
            "available_differs_across_legs": available_differs,
            "derivable_from_rows": not (rows_identical and available_differs),
        }
        print(f"    {population}")
        print(f"        available (db reachable)    : {leg_available['db_reachable']}")
        print(f"        available (not configured)  : {leg_available['db_not_configured']}")
        print(f"        rows identical across legs  : {rows_identical}")
        print(f"        available differs           : {available_differs}")
        print(f"        DERIVABLE FROM THE ROWS     : {results[population]['derivable_from_rows']}")

    undecidable = sorted(
        name for name, entry in results.items() if not entry["derivable_from_rows"]
    )
    results["_undecidable_populations"] = undecidable
    results["_verdict"] = (
        "NOT_DERIVABLE" if undecidable else "DERIVABLE"
    )
    print(f"    VERDICT                         : {results['_verdict']}")
    if undecidable:
        print(f"    indistinguishable populations   : {undecidable}")
        print("    -> build_pwml_ir would be GUESSING; the carrier belongs upstream.")
    # This section reports a fact; it is not a pass/fail gate on the card.
    return 0, results


# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-051 assert-only probe")
    parser.add_argument("--tmp", required=True)
    parser.add_argument("--corpus-root", default=str(REPO_ROOT))
    parser.add_argument(
        "--section", default="all",
        choices=("all", "refuse", "identity", "cli", "corpus", "carrier"),
    )
    parser.add_argument("--out", default=None, help="write the measurements as JSON")
    args = parser.parse_args(argv)

    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"T2PW: {t2pw.__file__}")

    code = 0
    payload: Dict[str, Any] = {"t2pw": t2pw.__file__}
    if args.section in ("all", "refuse"):
        rc, payload["refuse"] = _refuse()
        code |= rc
    if args.section in ("all", "identity"):
        rc, payload["identity"] = _identity()
        code |= rc
    if args.section in ("all", "cli"):
        rc, payload["cli"] = _cli(tmp)
        code |= rc
    if args.section in ("all", "corpus"):
        rc, payload["corpus"] = _corpus(tmp, Path(args.corpus_root))
        code |= rc
    if args.section in ("all", "carrier"):
        rc, payload["carrier"] = _carrier()
        code |= rc

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nmeasurements written to {args.out}")

    print("\nC-051 PROBE: " + ("PASSED" if code == 0 else "FAILED"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
