"""C-050i: the post-freeze row-dedupe refusal, measured at base and at tip.

**G9 labelling, up front.** C-050i is a **new capability** -- ``ir._dedupe_named_rows``
gains a refusal it never had -- so it carries an explicitly labelled new acceptance
test (``tests/test_pwml_ir_duplicate_row_refusal.py``) and needs **no** base-SHA
behavioural failure. None is fabricated here. What this probe adds is the same
fixtures run through **both real implementations**, base tree (exported by
``c051a_base_tree_batch.py``) and tip, so the latent defect is demonstrated:

``--mode g9``
    (a) **Constructed fixture** -- two compound rows differing only in punctuation,
        carrying different PathBank / ChEBI ids, with a reaction consuming the
        second. Base drops it and re-binds the reaction to the *first* row's ids;
        tip refuses. A *constructed fixture demonstrating a latent defect*, **not**
        a corpus regression.
    (b) **Real committed leg** ``PMC12444477…/strict`` (F-039's) under the two
        golden-sweep configurations that still reach the exporter, ``B`` and ``E``.
        ``A``/``C``/``D`` stop in pre-freeze at production defaults, which is why
        live *production* exposure is zero. Base turns 44 payload rows into 43 IR
        compounds and binds reaction 9 to the wrong molecule; tip refuses. Real
        data, still **not** a production-path regression -- ``B``/``E`` are offline
        test configurations.
    (c) **R3 control** -- the IR digest of a *non-colliding* leg, which must be
        byte-identical at base and tip. The refusal must cost nothing off the
        colliding path.

``--mode residual``  (charter section 5, REV-050h finding 4)
    EP3 runs ``run_prefreeze_resolution`` a **second** time on a deepcopy of the
    frozen payload (``streamlit_app.py:4091``); that pass renames compounds and can
    therefore *create* a ``_norm`` collision absent from the committed
    ``final_mapped.json``. D-034 clause 5 records ``_reject_ambiguous_renames`` as
    structurally **blind** to a collision between a rename target and a row that is
    not itself renamed, and that class is caught today only when the collided row is
    a *participant* (via ``PREFREEZE_CONNECTIVITY_BROKEN``); a **non-participant**
    row in that shape reaches the dedupe. This mode replays that second pass with
    the **live** resolver over every committed leg and counts the residual. DB
    access is read-only. If the resolver is unreachable it reports
    ``db_available: false`` and classifies **DB_UNAVAILABLE** rather than reporting
    a zero it never measured.

``--tree`` is the checkout whose **code** is imported and pinned; ``--fixtures`` is
where the committed ``runs/`` legs are read from, defaulting to ``--tree``. They are
separable because ``c045b_base_tree.PATHSPEC`` deliberately omits ``runs/`` (**F-042**),
so an exported base tree has the source but not the legs. **PATHSPEC is not
broadened**: the caller points ``--fixtures`` at a checkout that has them, having
verified them byte-identical at both revisions (``git diff --name-only <base> <tip>
-- runs runs_verify`` empty). Both paths are recorded so the reader can check that.

Usage::

    <python> probe_c050i_dedupe_refusal.py --mode g9|residual --tree <dir>
             [--fixtures <dir>] [--pin-verdict <json>] [--dotenv <path>] --out <json>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

#: F-039's leg: 44 compound rows, of which rows 5 and 23 share the ``_norm`` key
#: ``lipid iv a`` while carrying different PathBank, ChEBI and KEGG identifiers.
F039_LEG = ("runs/2026-07-28_0919/papers/"
            "PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict/final_mapped.json")

#: An R3 control leg with no ``_norm`` collision in any bucket (census: 31 of the
#: 32 committed legs are collision-free; this is one of them).
CLEAN_LEG = ("runs/2026-07-27_1623/papers/"
             "PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/"
             "final_mapped.json")


def _constructed_fixture() -> Dict[str, Any]:
    """Two spellings of F-039's real pair with **different** identifiers. The reaction
    consumes the spelling on the dropped row -- what turns a "duplicate" into a
    re-bound substrate."""

    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {
            "compounds": [
                {"name": "lipid IV_A", "db_status": "unmatched",
                 "pathbank_compound_id": 40982, "chebi_id": "CHEBI:58603"},
                {"name": "lipid IV A", "db_status": "unmatched",
                 "pathbank_compound_id": 40738, "chebi_id": "CHEBI:60365",
                 "kegg_id": "C06025"},
                {"name": "lipid A precursor", "db_status": "unmatched",
                 "pathbank_compound_id": 111},
            ],
            "proteins": [],
        },
        "processes": {
            "reactions": [
                {"name": "lipid IV A -> lipid A precursor",
                 "inputs": ["lipid IV A"], "outputs": ["lipid A precursor"]},
            ],
        },
    }


def _build(tree: Path, payload: Dict[str, Any], **kwargs: Any) -> Tuple[str, Any]:
    """``build_pwml_ir`` on ``payload``, classified rather than allowed to escape."""
    from t2pw.pwml.ir import build_pwml_ir

    dup = getattr(sys.modules["t2pw.pwml.ir"], "DuplicateNamedRowError", None)
    try:
        ir, report = build_pwml_ir(copy.deepcopy(payload), **kwargs)
    except Exception as exc:  # noqa: BLE001 - the classification IS the measurement
        if dup is not None and isinstance(exc, dup):
            return "REFUSED", {
                "code": getattr(exc, "code", ""),
                "norm_key": getattr(exc, "norm_key", ""),
                "pointer_prefix": getattr(exc, "pointer_prefix", ""),
                "names": getattr(exc, "names", []),
                "keys": getattr(exc, "keys", []),
                "pointers": getattr(exc, "pointers", []),
            }
        return "RAISED_OTHER", {"type": type(exc).__name__, "message": str(exc)[:400]}
    return "BUILT", (ir, report)


def _severity(report: Dict[str, Any], code: str) -> str:
    for bucket, name in (("errors", "error"), ("warnings", "warning")):
        if any(i.get("code") == code for i in report.get(bucket, [])):
            return name
    return "none"


def _describe_binding(ir: Dict[str, Any], *, reaction_index: int) -> Dict[str, Any]:
    """THE HARM, in one record: which compound row a reaction actually got bound to.
    The reaction declares ``'lipid IV A'`` (PathBank 40738 / ChEBI 60365); if that row
    was dropped, the binding lands on a row with *different* ids and nothing says so."""

    compounds = ir["entities"]["compounds"]
    out: Dict[str, Any] = {
        "ir_compound_count": len(compounds),
        "dropped_spelling_in_ir": "lipid IV A" in json.dumps(ir),
    }
    reaction = ir["processes"]["reactions"][reaction_index]
    out["reaction_name"] = reaction.get("name")
    left = reaction.get("left") or []
    if left:
        key = left[0]["entity_key"]
        bound = next(row for row in compounds if row["key"] == key)
        out["bound_entity_key"] = key
        out["bound_row_name"] = bound["name"]
        out["bound_row_pathwhiz_id"] = bound.get("pathwhiz_id")
        out["bound_row_chebi"] = bound.get("chebi_id")
    return out


def _mode_g9(tree: Path, fixtures: Path) -> Dict[str, Any]:
    from t2pw.pwml import ir as ir_mod
    from t2pw.pwml.name_index import PathwhizNameIndex
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution

    out: Dict[str, Any] = {"t2pw_file": ir_mod.__file__}

    # (a) the constructed fixture -----------------------------------------------
    status, detail = _build(tree, _constructed_fixture(), strict_db=False, name_index=None)
    arm: Dict[str, Any] = {"status": status}
    if status == "BUILT":
        ir, report = detail
        arm.update(_describe_binding(ir, reaction_index=0))
        arm["compound_names"] = [row["name"] for row in ir["entities"]["compounds"]]
        arm["report_ok"] = report["ok"]
        arm["duplicate_issue_severity"] = _severity(report, "duplicate_named_record")
    else:
        arm["detail"] = detail
    out["a_constructed_fixture"] = arm

    # (b) the real committed leg, golden-sweep configurations B and E ------------
    class _DownDb:
        last_error = "harvest_db_down"

        def available(self) -> bool:
            return False

    leg_payload = json.loads((fixtures / F039_LEG).read_text(encoding="utf-8"))
    leg_arms: Dict[str, Any] = {"payload_compound_rows": len(
        leg_payload["entities"]["compounds"])}
    for name, kwargs in (
        ("B_dbdown_noindex_strict",
         dict(db_resolver=_DownDb(), strict_db=True, name_index=None)),
        ("E_fromenv_raises_emptyindex_lenient",
         dict(db_resolver=None, strict_db=False, name_index=PathwhizNameIndex({}))),
    ):
        staged = copy.deepcopy(leg_payload)
        try:
            run_prefreeze_resolution(
                staged, strict_db=kwargs["strict_db"],
                db_resolver=kwargs["db_resolver"], name_index=kwargs["name_index"])
        except PrefreezeResolutionError as stop:
            leg_arms[name] = {"status": "PREFREEZE_STOP", "code": stop.code}
            continue
        status, detail = _build(tree, staged, **kwargs)
        entry: Dict[str, Any] = {"status": status}
        if status == "BUILT":
            entry.update(_describe_binding(detail[0], reaction_index=9))
        else:
            entry["detail"] = detail
        leg_arms[name] = entry
    out["b_committed_leg"] = leg_arms

    # (c) R3 control: a non-colliding leg must be byte-identical ------------------
    clean = json.loads((fixtures / CLEAN_LEG).read_text(encoding="utf-8"))
    staged = copy.deepcopy(clean)
    run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)
    status, detail = _build(tree, staged, db_resolver=_DownDb(), strict_db=True, name_index=None)
    control: Dict[str, Any] = {"status": status}
    if status == "BUILT":
        ir, report = detail
        control["ir_digest"] = hashlib.sha256(
            json.dumps(ir, sort_keys=True, indent=1, default=repr).encode()).hexdigest()
        control["report_digest"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, indent=1, default=repr).encode()).hexdigest()
        control["compound_keys"] = [r["key"] for r in ir["entities"]["compounds"]]
    else:
        control["detail"] = detail
    out["c_r3_control_clean_leg"] = control
    return out


def _mode_residual(tree: Path, fixtures: Path) -> Dict[str, Any]:
    """Charter section 5. EP3's SECOND pre-freeze pass, with the live resolver."""
    from t2pw.pwml import ir as ir_mod
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution
    # The same import production uses (``compound_resolution.py:478``), so this
    # measures the real resolver rather than a look-alike.
    from t2pw.mapping.map_ids import PathBankDbResolver

    out: Dict[str, Any] = {"t2pw_file": ir_mod.__file__}
    try:
        # ``from_env`` returns None when host/user are absent -- an unconfigured DB
        # is a reportable outcome, never an exception to swallow into a zero.
        resolver = PathBankDbResolver.from_env()
        if resolver is None:
            available, reason = False, "from_env returned None (host/user not configured)"
        else:
            available = bool(resolver.available())
            reason = "" if available else str(getattr(resolver, "last_error", "") or "unknown")
    except Exception as exc:  # noqa: BLE001 - unreachable is a reportable outcome
        resolver, available, reason = None, False, f"{type(exc).__name__}: {exc}"[:300]
    out["db_available"] = available
    out["db_reason"] = reason
    if not available:
        out["classification"] = "DB_UNAVAILABLE"
        out["note"] = ("The live resolver was not reachable, so the residual class was "
                       "NOT measured. No number is reported. This is an honest "
                       "DB-unavailable outcome, not a measured zero.")
        return out

    legs = sorted(
        str(p.relative_to(fixtures)).replace("\\", "/")
        for root in ("runs", "runs_verify")
        for p in (fixtures / root).rglob("final_mapped.json"))
    created: Dict[str, Any] = {}
    stops: Dict[str, str] = {}
    for leg in legs:
        # EP3 hands build_pwml_ir a deepcopy of the FROZEN payload, then runs the
        # pre-freeze sequence over it a second time. That is reproduced exactly.
        staged = copy.deepcopy(json.loads((fixtures / leg).read_text(encoding="utf-8")))
        before = _collision_map(ir_mod, staged)
        try:
            run_prefreeze_resolution(staged, strict_db=False, db_resolver=resolver)
        except PrefreezeResolutionError as stop:
            stops[leg] = stop.code
            continue
        after = _collision_map(ir_mod, staged)
        fresh = {k: v for k, v in after.items() if k not in before}
        if fresh:
            created[leg] = fresh
    out["legs_measured"] = len(legs)
    out["prefreeze_stops"] = stops
    out["created_collisions"] = created
    out["classification"] = "MEASURED"
    out["residual_count"] = sum(len(v) for v in created.values())
    return out


def _collision_map(ir_mod: Any, payload: Dict[str, Any]) -> Dict[str, List[str]]:
    """Every ``bucket|_norm`` group holding more than one row, with its names."""
    entities = payload.get("entities") or {}
    groups: Dict[str, List[str]] = {}
    for bucket in ("compounds", "proteins", "nucleic_acids", "element_collections",
                   "protein_complexes", "species", "subcellular_locations",
                   "cell_types", "tissues"):
        seen: Dict[str, List[str]] = {}
        for row in (entities.get(bucket) or []):
            if not isinstance(row, dict):
                continue
            name = ir_mod._canonical(row.get("name"))
            if not name:
                continue
            seen.setdefault(ir_mod._norm(name), []).append(name)
        for norm, names in seen.items():
            if len(names) > 1:
                groups[f"{bucket}|{norm}"] = names
    return groups


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C-050i base/tip probe")
    parser.add_argument("--mode", required=True, choices=("g9", "residual"))
    parser.add_argument("--tree", required=True, help="the tree under measurement")
    parser.add_argument("--fixtures", default=None,
                        help="checkout to read committed runs/ legs from (default: --tree)")
    parser.add_argument("--pin-verdict", default=None,
                        help="where tree_pin writes its verdict (TEST_MATRIX section 0 rule 10)")
    parser.add_argument("--dotenv", default=None,
                        help="a .env to read into os.environ before the tree is pinned; "
                             "READ-ONLY and never copied. Charter section 5 requires the "
                             "primary checkout's .env, which must not be placed in a worktree.")
    parser.add_argument("--out", required=True, help="JSON destination")
    args = parser.parse_args(argv)

    # F-045: resolve EVERY path before anything chdirs, and never write into the
    # tree being audited. A measurement script has already once written its
    # evidence into the checkout it was auditing.
    out_path = Path(args.out).resolve()
    tree = Path(args.tree).resolve()
    fixtures = Path(args.fixtures).resolve() if args.fixtures else tree
    verdict_path = str(Path(args.pin_verdict).resolve()) if args.pin_verdict else None
    dotenv_path = Path(args.dotenv).resolve() if args.dotenv else None
    cwd_at_entry = os.getcwd()

    # The .env is READ where it lives, its values placed in this process's
    # environment. Nothing is written or copied: the DB settings are read through
    # ``os.getenv``, so this suffices and a file copy never would.
    if dotenv_path is not None:
        assert dotenv_path.is_file(), f"--dotenv not found: {dotenv_path}"
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(dotenv_path), override=False)

    # H-010's measurement-tree pin, called rather than reimplemented and NOT weakened:
    # this deliberately does not put ``<tree>/src`` on ``sys.path``, because repairing
    # the path here would make the wrong-tree refusal unreachable and turn
    # ``PYTHONPATH`` into decoration. The caller pins ``PYTHONPATH=<tree>/src`` and the
    # verdict settles which tree ran. Tree root onto ``sys.path`` + cwd, as
    # ``pinned_pytest`` does, because ``check`` refuses ``CWD_NOT_EXPECTED_ROOT``.
    sys.path.insert(0, str(evidence_dir()))
    import tree_pin

    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    os.chdir(tree)
    facts = tree_pin.enforce(expected=tree, require_scripts=False,
                             verdict_path=verdict_path, cwd_at_entry=cwd_at_entry)

    payload = (_mode_g9(tree, fixtures) if args.mode == "g9"
               else _mode_residual(tree, fixtures))
    payload["mode"] = args.mode
    payload["tree"] = str(tree)
    payload["fixtures"] = str(fixtures)
    payload["dotenv"] = str(dotenv_path) if dotenv_path else None
    payload["pin_verdict"] = facts.get("verdict_path")
    payload["resolved_t2pw"] = facts.get("t2pw_file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=repr), encoding="utf-8")
    assert out_path.is_file(), f"artifact did not land at {out_path}"
    assert not out_path.is_relative_to(tree), (
        f"F-045: refusing to leave evidence inside the audited tree at {out_path}")
    print(f"mode           : {args.mode}")
    print(f"t2pw resolved  : {facts.get('t2pw_file')}")
    print(f"fixtures from  : {fixtures}")
    print(f"artifact landed: {out_path} ({out_path.stat().st_size} bytes)")
    print(json.dumps(payload, indent=2, default=repr)[:4000])
    return 0


def evidence_dir() -> Path:
    """This file's own directory -- where ``tree_pin`` lives."""
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    sys.exit(main())
