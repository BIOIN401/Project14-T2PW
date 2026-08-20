"""C-050j census: how exposed is the create-defaults-MANUFACTURED component collision?

**This script is a read-only measurement. It changes nothing and asserts no verdict.**

D-050 section 3 makes this the card's first act and lets it decide the G9 arm:

* **zero** exposure  -> C-050j hardens a *latent* path: G9 **new capability**, an
  explicitly labelled new acceptance test, and **no** fabricated base failure.
* **non-zero**       -> the card stops for a fresh ruling, because G9 flips to a
  **correction** needing a real base-failing proof.

What is measured, and why the earlier census could not have seen it
-------------------------------------------------------------------
``build_pwml_ir`` applies ``_apply_create_defaults`` to the **species** and
**subcellular_locations** component rows (``ir.py:1181`` / ``:1187``) -- i.e. it
**renames rows after the canonical hash** (``ir.py:344``) -- and then hands them to
``_dedupe_named_rows`` at ``:1191-1195`` **with no** ``refuse_duplicates``. So the
exporter can *manufacture* a ``_norm`` collision the payload never had and then
silently collapse it. C-050i's zero-of-32 census replayed ``_norm(_canonical(...))``
over committed ``final_mapped.json``, i.e. state from **before**
``_apply_create_defaults`` runs, and is structurally blind to this.

Method
------
For every committed leg, and for **both** production ``strict_db`` settings, this
reproduces **EP3** (``streamlit_app.py:4154`` then ``:4241``) exactly::

    run_prefreeze_resolution(payload, strict_db=<arm>)   # db_resolver=None -> .env
    build_pwml_ir(payload, strict_db=<arm>)              # db_resolver=None, default index

``_apply_create_defaults`` and ``_dedupe_named_rows`` are **wrapped, not
reimplemented**: the real functions run, and the wrappers record, for every row
reaching the component dedupe, its ``_norm`` **before** the post-freeze rename and
its ``_norm`` **after**. Reproducing the loop by hand would measure a copy of the
exporter rather than the exporter.

Every ``_norm`` group with more than one row that reaches the component dedupe is
then classified:

``post_freeze_manufactured_merge``
    the members do **not** all share one pre-rename ``_norm``: the exporter's own
    post-freeze rename **merged two previously distinct name groups**.
    *This is the census number.*
``preexisting``
    every member already shared the group ``_norm`` before create-defaults ran.
    Either a deliberate pre-freeze convergence (``species_canonicalization`` marker
    with ``followed_leader``, ``prefreeze_resolution.py:1242``) or the
    payload-authored residual F-046 registers. Neither is counted.

``preexisting`` groups are reported in full anyway, because D-050 section 3 warns
that "deliberate convergences that must keep collapsing have never been
enumerated". This is that enumeration.

``--mode control`` is the **instrument control**: a synthetic payload carrying both
``ir.py:230-238`` spellings, which the create-defaults table renames onto one name.
A census that cannot see a manufactured merge cannot report a meaningful zero.

Usage::

    <python> probe_c050j_component_collision_census.py --mode census --out <json>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()

import t2pw  # noqa: E402
from t2pw.pwml import ir as ir_mod  # noqa: E402

#: The four component buckets ``build_pwml_ir`` routes through the *component*
#: call site (``ir.py:1157-1166``). Only these reach the unguarded dedupe.
COMPONENT_POINTERS = {
    "/entities/species",
    "/entities/subcellular_locations",
    "/entities/cell_types",
    "/entities/tissues",
}


def _marker(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get(ir_mod.SPECIES_CANONICALIZATION_FIELD)
    return value if isinstance(value, dict) else {}


def _classify(rows: List[Any], pointer_prefix: str, origin: Dict[int, str]) -> Dict[str, Any]:
    """Group the rows arriving at ``_dedupe_named_rows`` exactly as it does."""

    groups: Dict[str, List[Dict[str, Any]]] = {}
    renamed = 0
    named = 0
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        name = ir_mod._canonical(raw.get("name"))
        if not name:
            continue
        named += 1
        now = ir_mod._norm(name)
        before = origin.get(id(raw), now)
        if before != now:
            renamed += 1
        marker = _marker(raw)
        groups.setdefault(now, []).append(
            {
                "name": name,
                "origin_norm": before,
                "renamed_post_freeze": before != now,
                "followed_leader": marker.get("followed_leader"),
                "canonicalization_status": marker.get("status"),
            }
        )

    collisions: Dict[str, Any] = {}
    for norm, members in groups.items():
        if len(members) < 2:
            continue
        origins = sorted({m["origin_norm"] for m in members})
        collisions[norm] = {
            "kind": "post_freeze_manufactured_merge" if len(origins) > 1 else "preexisting",
            "origin_norms": origins,
            "members": members,
        }
    return {
        "pointer_prefix": pointer_prefix,
        "named_rows": named,
        "rows_renamed_post_freeze": renamed,
        "collisions": collisions,
    }


def _run_ep3(payload: Dict[str, Any], *, strict_db: bool) -> Dict[str, Any]:
    """EP3, with the two exporter helpers wrapped so the merge class is observable."""

    from t2pw.pwml.prefreeze_resolution import (
        PrefreezeResolutionError,
        run_prefreeze_resolution,
    )

    out: Dict[str, Any] = {"strict_db": strict_db}
    started = time.time()
    try:
        run_prefreeze_resolution(payload, strict_db=strict_db)
    except PrefreezeResolutionError as stop:
        out["status"] = "PREFREEZE_STOP"
        out["code"] = stop.code
        out["seconds"] = round(time.time() - started, 2)
        return out
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        out["status"] = "PREFREEZE_RAISED_OTHER"
        out["detail"] = f"{type(exc).__name__}: {exc}"[:300]
        out["seconds"] = round(time.time() - started, 2)
        return out

    origin: Dict[int, str] = {}
    keepalive: List[Any] = []  # ids must not be recycled mid-build
    observed: List[Dict[str, Any]] = []

    real_defaults = ir_mod._apply_create_defaults
    real_dedupe = ir_mod._dedupe_named_rows

    def spy_defaults(raw: Any, defaults: Any) -> Any:
        before = ir_mod._norm(ir_mod._canonical(raw.get("name"))) if isinstance(raw, dict) else ""
        record = real_defaults(raw, defaults)
        origin[id(record)] = before
        keepalive.append(record)
        return record

    def spy_dedupe(rows: Any, **kwargs: Any) -> Any:
        materialized = list(rows)
        keepalive.extend(materialized)
        pointer_prefix = kwargs.get("pointer_prefix", "")
        if pointer_prefix in COMPONENT_POINTERS:
            observed.append(_classify(materialized, pointer_prefix, origin))
        return real_dedupe(materialized, **kwargs)

    ir_mod._apply_create_defaults = spy_defaults
    ir_mod._dedupe_named_rows = spy_dedupe
    try:
        ir_mod.build_pwml_ir(payload, strict_db=strict_db)
        out["status"] = "BUILT"
    except Exception as exc:  # noqa: BLE001 - the classification IS the measurement
        out["status"] = "BUILD_RAISED"
        out["detail"] = f"{type(exc).__name__}: {getattr(exc, 'code', '')}: {exc}"[:300]
    finally:
        ir_mod._apply_create_defaults = real_defaults
        ir_mod._dedupe_named_rows = real_dedupe

    out["seconds"] = round(time.time() - started, 2)
    out["buckets_observed"] = len(observed)
    out["rows_renamed_post_freeze"] = sum(b["rows_renamed_post_freeze"] for b in observed)
    manufactured: Dict[str, Any] = {}
    preexisting: Dict[str, Any] = {}
    for bucket in observed:
        for norm, detail in bucket["collisions"].items():
            key = f"{bucket['pointer_prefix']}|{norm}"
            bucketed = (manufactured if detail["kind"] == "post_freeze_manufactured_merge"
                        else preexisting)
            bucketed[key] = detail
    out["manufactured_merges"] = manufactured
    out["preexisting_collisions"] = preexisting
    return out


def _legs() -> List[str]:
    found = set()
    for root in ("runs", "runs_verify"):
        base = REPO_ROOT / root
        if base.is_dir():
            for path in base.rglob("final_mapped.json"):
                found.add(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return sorted(found)


def _mode_census() -> Dict[str, Any]:
    legs = _legs()
    per_leg: Dict[str, Any] = {}
    manufactured_total = 0
    preexisting_total = 0
    renamed_total = 0
    stops: Dict[str, str] = {}
    for leg in legs:
        payload = json.loads((REPO_ROOT / leg).read_text(encoding="utf-8"))
        arms: Dict[str, Any] = {}
        for arm, strict in (("lenient", False), ("strict", True)):
            result = _run_ep3(copy.deepcopy(payload), strict_db=strict)
            manufactured_total += len(result.get("manufactured_merges", {}))
            preexisting_total += len(result.get("preexisting_collisions", {}))
            renamed_total += int(result.get("rows_renamed_post_freeze", 0) or 0)
            if result["status"] in ("PREFREEZE_STOP", "PREFREEZE_RAISED_OTHER"):
                stops[f"{leg}|{arm}"] = result.get("code") or result.get("detail", "")
            arms[arm] = result
        per_leg[leg] = arms
        print(f"  [leg] {leg}")
        for arm, result in arms.items():
            print(f"        {arm:<8} {result['status']:<22} "
                  f"renamed={result.get('rows_renamed_post_freeze', 0)} "
                  f"manufactured={len(result.get('manufactured_merges', {}))} "
                  f"preexisting={len(result.get('preexisting_collisions', {}))} "
                  f"{result.get('seconds', 0)}s", flush=True)
    return {
        "legs_measured": len(legs),
        "arms_per_leg": 2,
        "measurements": len(legs) * 2,
        "manufactured_merge_count": manufactured_total,
        "preexisting_collision_count": preexisting_total,
        "rows_renamed_post_freeze_total": renamed_total,
        "prefreeze_stops": stops,
        "per_leg": per_leg,
    }


def _control_payload() -> Dict[str, Any]:
    """Both spellings ``SPECIES_CREATE_DEFAULTS`` folds together (``ir.py:230-238``).

    Neither name is the other's ``_norm``, so no pre-freeze guard sees a collision;
    it exists only after ``_apply_create_defaults`` renames the first row,
    post-freeze. The two rows carry **different** taxonomy ids, which is why
    dropping one is not a cosmetic loss -- and why F-043 forbids settling the
    question on identifiers in either direction.
    """

    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {
            "compounds": [
                {"name": "glycine", "db_status": "matched", "pathbank_compound_id": 1},
            ],
            "species": [
                {"name": "Narcissus sp. aff. pseudonarcissus", "taxonomy_id": "1540222"},
                {"name": "Narcissus aff. pseudonarcissus MK-2014", "taxonomy_id": "999999"},
            ],
        },
        "processes": {"reactions": []},
    }


def _mode_control() -> Dict[str, Any]:
    result = _run_ep3(_control_payload(), strict_db=False)
    result["instrument_sees_a_manufactured_merge"] = bool(result.get("manufactured_merges"))
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="C-050j create-defaults collision census")
    parser.add_argument("--mode", required=True, choices=("census", "control", "all"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()
    resolved = Path(t2pw.__file__).resolve()
    assert resolved.is_relative_to(REPO_ROOT), (
        f"wrong tree: t2pw resolved to {resolved}, expected under {REPO_ROOT}")

    payload: Dict[str, Any] = {
        "mode": args.mode,
        "repo_root": str(REPO_ROOT),
        "t2pw_file": str(resolved),
    }
    if args.mode in ("control", "all"):
        payload["control"] = _mode_control()
    if args.mode in ("census", "all"):
        payload["census"] = _mode_census()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=repr), encoding="utf-8")
    print("=" * 72)
    print(f"t2pw resolved        : {resolved}")
    if "control" in payload:
        control = payload["control"]
        print(f"CONTROL status       : {control['status']}")
        print(f"CONTROL sees a merge : {control['instrument_sees_a_manufactured_merge']}")
        print(f"CONTROL merges       : {json.dumps(control.get('manufactured_merges', {}))[:600]}")
    if "census" in payload:
        summary = payload["census"]
        print(f"legs measured        : {summary['legs_measured']}")
        print(f"measurements         : {summary['measurements']}")
        print(f"rows renamed (total) : {summary['rows_renamed_post_freeze_total']}")
        print(f"PREEXISTING          : {summary['preexisting_collision_count']}")
        print(f"MANUFACTURED MERGES  : {summary['manufactured_merge_count']}"
              "   <== THE CENSUS NUMBER")
    print(f"artifact landed      : {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
