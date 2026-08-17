"""C-050k: the alias-index ambiguity census (F-048), and the base-vs-tip differential.

**G9 labelling, up front. CORRECTED 2026-08-17 by D-043 -- read this before the rest.**
This file was written before the census ran, and labelled C-050k a *new capability*. The
census falsified that: **61 live ``resolve_entity`` consultations through an ambiguous
alias key, across 8 legs, with the diagnostic empty on all 8.** There is pre-existing
observable behaviour here and it is wrong, so under **D-043 section 4** C-050k is a
**CORRECTION**, carries a genuine base-SHA behavioural failure, and must not be presented
as new functionality. ``--mode g9`` is that proof and **exits 1 where the diagnostic is
absent**, so it fails at the base SHA and passes at the tip.

``--mode differential`` remains what its name says -- *differential evidence*, the same
fixture through two real implementations, "what moved" and never "what broke". The
committed ``c050k_differential_base.json`` was produced under the old labelling and is
still valid as differential evidence; it is simply not the G9 proof.

``--mode census``  (charter section 2 / D-041 section 3)
    **Exposure was UNMEASURED before this probe ran.** C-050i's and REV-050i's censuses
    measured ``_norm(name)`` collisions only, because that is what ``_dedupe_named_rows``
    groups on. This one measures the surface F-048 owns: the **alias** index
    ``entity_by_name``, which is populated from ``[name, raw_name, short_name,
    common_name, *synonyms]`` (``ir.py:1249-1261``), and where a *name* on one row can
    collide with an *alias* on another without ever being a dedupe collision.

    EP3 is reproduced exactly (``streamlit_app.py:4087-4141``): a deepcopy of the frozen
    ``final_mapped.json``, then ``run_prefreeze_resolution`` a **second** time with
    ``strict_db`` alone -- no resolver and no name index, exactly as the production seam
    passes them -- then ``build_pwml_ir``. Four quantities per leg:

    1. ``_norm`` keys in ``entity_by_name`` mapping to **more than one distinct entity
       key**  -- the real ambiguity signal;
    2. of those, how many are **actually consulted** by a ``resolve_entity`` call;
    3. for each such consultation: role, pointer, every candidate, and the entity it
       binds to today;
    4. ``_norm`` keys carrying a **repeated same-key** candidate -- the false positive
       (**D-041 section 5**): ``ir.py:1249-1261`` appends once per alias slot with no
       dedupe, so one entity whose ``name`` and ``raw_name`` normalize alike appears
       twice under one key and ``len(candidates) > 1`` is true for a **single** entity.
       Counted here so the number is on the record, and excluded from (1) by
       construction.

    **(2) was non-zero and the card DID stop**, at tip ``dd5da13``. A ``pwml-bio-auditor``
    then adjudicated all 20 same-type bindings against the committed frozen payloads and
    **D-043** ruled every one of them biologically **correct**: no
    ``product_contract_violation``. The violation is the **missing diagnostic** --
    ``PRODUCT_CONTRACT`` section 3 traceability -- so the obligation is to record the
    choice, not to have been lucky.

``--mode differential``
    F-048's decoy -- rows ``[serine (synonym "Glycine"), glycine]`` and a reaction input
    ``"Glycine"`` -- in **both row orders**, plus two controls. Run once in a real git
    worktree at the base SHA and once at the tip. **Never** on a ``c045b_base_tree.py``
    export (**F-042**: ``PATHSPEC`` omits ``scripts/``).

**How the index is observed, and why nothing is reimplemented.** ``entity_by_name`` is a
``build_pwml_ir`` local, so it is captured rather than rebuilt: ``ir.defaultdict`` is
temporarily bound to a factory that returns a recording ``defaultdict`` **only** for the
``defaultdict(list)`` at ``ir.py:1186`` (the module's only other use is
``defaultdict(int)`` at ``:450``), and restored in ``finally``. The recorder observes the
real index the real code built, and its ``get`` -- ``resolve_entity``'s single lookup at
``ir.py:1515`` -- records every consultation with the caller's own ``role``/``pointer``
read off the live frame. ``_norm`` / ``_canonical`` are **imported, never reimplemented**
(R5). Production code is untouched by this file.

Usage::

    <python> probe_c050k_alias_ambiguity.py --mode census|differential|g9 --tree <dir>
             [--fixtures <dir>] [--pin-verdict <json>] [--dotenv <path>] --out <json>
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: EP3 passes exactly these to ``build_pwml_ir`` (``streamlit_app.py:4135-4141``):
#: no ``db_resolver``, no ``name_index``.
EP3_IR_KWARGS = dict(pathway_name="Generated Pathway", pathway_subject="Metabolic",
                     width=6400, height=1400)


class _RecordingIndex(collections.defaultdict):
    """The real ``entity_by_name``, with its lookups recorded.

    Subclasses ``defaultdict`` so ``__missing__``/``append`` behave identically; only
    ``get`` is observed, and it returns exactly what ``defaultdict.get`` returns.
    ``resolve_entity`` is the sole caller of ``.get`` on this object (``ir.py:1515``).
    """

    def __init__(self, default_factory: Any) -> None:
        super().__init__(default_factory)
        self.consultations: List[Dict[str, Any]] = []
        self.preferred_order: Optional[Dict[str, List[str]]] = None

    def get(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        value = super().get(key, default)
        record: Dict[str, Any] = {"key": key, "n_candidates": len(value or [])}
        try:
            caller = sys._getframe(1)
            local = caller.f_locals
            record["caller"] = caller.f_code.co_name
            for field in ("clean", "role", "pointer", "hint"):
                record[field] = local.get(field)
            # ``preferred_order`` is a free variable of ``resolve_entity``; free
            # variables are present in ``f_locals``. Captured rather than restated so
            # the replay in (3) uses the live table.
            if self.preferred_order is None:
                order = local.get("preferred_order")
                if isinstance(order, dict):
                    self.preferred_order = copy.deepcopy(order)
        except Exception as exc:  # noqa: BLE001 - introspection failure is reportable
            record["frame_error"] = f"{type(exc).__name__}: {exc}"[:200]
        self.consultations.append(record)
        return value


@contextlib.contextmanager
def _recording(ir_mod: Any) -> Iterator[List[_RecordingIndex]]:
    """Bind ``ir.defaultdict`` to a recording factory for ``defaultdict(list)`` only."""

    real = ir_mod.defaultdict
    created: List[_RecordingIndex] = []

    def factory(default_factory: Any = None, *args: Any, **kwargs: Any) -> Any:
        if default_factory is list and not args and not kwargs:
            index = _RecordingIndex(list)
            created.append(index)
            return index
        return real(default_factory, *args, **kwargs)

    ir_mod.defaultdict = factory
    try:
        yield created
    finally:
        ir_mod.defaultdict = real


def _candidate_view(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {"entity_key": candidate.get("key"), "entity_type": candidate.get("entity_type"),
            "name": candidate.get("name"), "pathwhiz_id": candidate.get("pathwhiz_id")}


def _distinct_keys(candidates: List[Dict[str, Any]]) -> List[Any]:
    seen: List[Any] = []
    for candidate in candidates:
        key = candidate.get("key")
        if key not in seen:
            seen.append(key)
    return seen


def _replay_binding(ir_mod: Any, candidates: List[Dict[str, Any]], role: Any, hint: Any,
                    preferred_order: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
    """A replay of ``ir.py:1528-1544`` over the captured candidates and the **captured**
    ``preferred_order``. Labelled a replay: the raw candidate list is reported beside it
    so a reader can check the answer rather than trust it."""

    if preferred_order is None:
        return {"replay": "UNAVAILABLE", "reason": "preferred_order not captured"}
    ordered = [hint] if hint in ir_mod.ENTITY_BUCKETS else []
    ordered.extend(preferred_order.get(role, []))
    for wanted in ordered:
        for candidate in candidates:
            if candidate.get("entity_type") == wanted:
                return {"replay": "PREFERRED_TYPE_EARLY_RETURN", "wanted": wanted,
                        "bound": _candidate_view(candidate)}
    return {"replay": "FALL_THROUGH_CANDIDATES_0",
            "bound": _candidate_view(candidates[0]) if candidates else None}


def _analyse(ir_mod: Any, index: _RecordingIndex) -> Dict[str, Any]:
    """The four-part census over one leg's captured index.

    **Both the index and the consultation log are snapshotted first.** ``index.get``
    is instrumented, so reading through it here would append to the very list being
    iterated -- an unbounded loop. Measured the hard way: the first run of this probe
    timed out at 311 s having grown to 3.3 GB, and the failing bounded report is
    committed at ``g11/C-050k/01-differential-base.json`` rather than deleted.
    """

    snapshot: Dict[Any, List[Dict[str, Any]]] = {key: list(value) for key, value in index.items()}
    consultations = list(index.consultations)

    ambiguous: Dict[str, Any] = {}
    same_key_repeats: Dict[str, Any] = {}
    naive_multi = 0
    for norm_key, candidates in snapshot.items():
        if len(candidates) > 1:
            naive_multi += 1
        keys = _distinct_keys(candidates)
        if len(keys) > 1:
            ambiguous[norm_key] = [_candidate_view(c) for c in candidates]
        if len(candidates) > len(keys):
            same_key_repeats[norm_key] = {
                "candidate_slots": len(candidates), "distinct_entity_keys": len(keys),
                "entity_types": [c.get("entity_type") for c in candidates]}

    consulted: List[Dict[str, Any]] = []
    for record in consultations:
        if record.get("key") in ambiguous:
            candidates = snapshot.get(record["key"], [])
            entry = dict(record)
            entry["candidates"] = [_candidate_view(c) for c in candidates]
            entry["binding_today"] = _replay_binding(
                ir_mod, candidates, record.get("role"), record.get("hint"),
                index.preferred_order)
            consulted.append(entry)

    return {
        # (1)
        "part1_ambiguous_norm_keys": len(ambiguous),
        "part1_detail": ambiguous,
        # (2)
        "part2_ambiguous_consultations": len(consulted),
        "part2_distinct_ambiguous_keys_consulted": len({c["key"] for c in consulted}),
        # (3)
        "part3_consultations": consulted,
        # (4)
        "part4_same_key_repeat_norm_keys": len(same_key_repeats),
        "part4_detail": same_key_repeats,
        # context
        "norm_keys_total": len(snapshot),
        "norm_keys_with_multiple_candidate_slots": naive_multi,
        "resolve_entity_consultations_total": len(consultations),
    }


def _census_leg(ir_mod: Any, prefreeze_mod: Any, fixtures: Path, leg: str) -> Dict[str, Any]:
    strict_db = "/strict/" in leg
    entry: Dict[str, Any] = {"leg": leg, "strict_db": strict_db}
    staged = copy.deepcopy(json.loads((fixtures / leg).read_text(encoding="utf-8")))
    try:
        report = prefreeze_mod.run_prefreeze_resolution(staged, strict_db=strict_db)
        entry["prefreeze"] = {"ok": report.get("ok"),
                              "review_required": report.get("review_required"),
                              "failures": report.get("failures")}
    except prefreeze_mod.PrefreezeResolutionError as stop:
        entry["status"] = "PREFREEZE_STOP"
        entry["prefreeze_stop_code"] = stop.code
        return entry
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        entry["status"] = "PREFREEZE_RAISED_OTHER"
        entry["detail"] = f"{type(exc).__name__}: {exc}"[:400]
        return entry

    with _recording(ir_mod) as created:
        try:
            _ir, ir_report = ir_mod.build_pwml_ir(staged, strict_db=strict_db, **EP3_IR_KWARGS)
            entry["status"] = "BUILT"
            entry["report_ok"] = ir_report["ok"]
            entry["ambiguous_entity_reference_issues"] = [
                issue for bucket in ("errors", "warnings")
                for issue in ir_report.get(bucket, [])
                if issue.get("code") == "ambiguous_entity_reference"]
        except Exception as exc:  # noqa: BLE001 - a refusal is a measurement, not a crash
            entry["status"] = "IR_RAISED"
            entry["detail"] = f"{type(exc).__name__}: {exc}"[:400]
    indexes = [index for index in created if len(index) or index.consultations]
    entry["indexes_captured"] = len(created)
    if not indexes:
        entry["census"] = None
        entry["census_note"] = "no entity_by_name index was populated"
        return entry
    # ``build_pwml_ir`` builds exactly one ``defaultdict(list)``. More than one would
    # mean the capture assumption is stale; it is reported, never averaged away.
    entry["census"] = _analyse(ir_mod, indexes[-1])
    if len(indexes) > 1:
        entry["census_warning"] = f"{len(indexes)} list-defaultdicts captured; analysed the last"
    return entry


def _mode_census(tree: Path, fixtures: Path) -> Dict[str, Any]:
    from t2pw.pwml import ir as ir_mod
    from t2pw.pwml import prefreeze_resolution as prefreeze_mod

    out: Dict[str, Any] = {"t2pw_file": ir_mod.__file__}
    try:
        from t2pw.mapping.map_ids import PathBankDbResolver
        resolver = PathBankDbResolver.from_env()
        out["db_available"] = bool(resolver.available()) if resolver is not None else False
        out["db_reason"] = "" if out["db_available"] else (
            "from_env returned None (host/user not configured)" if resolver is None
            else str(getattr(resolver, "last_error", "") or "unknown"))
    except Exception as exc:  # noqa: BLE001 - unreachable is a reportable outcome
        out["db_available"] = False
        out["db_reason"] = f"{type(exc).__name__}: {exc}"[:300]

    legs = sorted(
        str(path.relative_to(fixtures)).replace("\\", "/")
        for root in ("runs", "runs_verify")
        for path in (fixtures / root).rglob("final_mapped.json"))
    results = [_census_leg(ir_mod, prefreeze_mod, fixtures, leg) for leg in legs]
    out["legs_measured"] = len(results)
    out["legs"] = results

    built = [r for r in results if r.get("status") == "BUILT"]
    censused = [r for r in results if r.get("census")]
    out["totals"] = {
        "legs": len(results),
        "legs_built": len(built),
        "legs_with_index": len(censused),
        "legs_prefreeze_stopped": len([r for r in results if r.get("status") == "PREFREEZE_STOP"]),
        "legs_ir_raised": len([r for r in results if r.get("status") == "IR_RAISED"]),
        "part1_ambiguous_norm_keys": sum(r["census"]["part1_ambiguous_norm_keys"] for r in censused),
        "part2_ambiguous_consultations": sum(
            r["census"]["part2_ambiguous_consultations"] for r in censused),
        "part4_same_key_repeat_norm_keys": sum(
            r["census"]["part4_same_key_repeat_norm_keys"] for r in censused),
        "legs_with_part1_nonzero": len(
            [r for r in censused if r["census"]["part1_ambiguous_norm_keys"]]),
        "legs_with_part2_nonzero": len(
            [r for r in censused if r["census"]["part2_ambiguous_consultations"]]),
        "norm_keys_total": sum(r["census"]["norm_keys_total"] for r in censused),
        "resolve_entity_consultations_total": sum(
            r["census"]["resolve_entity_consultations_total"] for r in censused),
    }
    out["stop_condition_2_triggered"] = bool(out["totals"]["part2_ambiguous_consultations"])
    out["classification"] = "MEASURED"
    return out


def _decoy_payload(order: str) -> Dict[str, Any]:
    """F-048's decoy, byte-for-byte the shape pinned at
    ``tests/test_pwml_ir_duplicate_row_refusal.py:493-515``."""

    serine = {"name": "serine", "db_status": "unmatched", "synonyms": ["Glycine"]}
    glycine = {"name": "glycine", "db_status": "unmatched"}
    rows = [serine, glycine] if order == "serine_first" else [glycine, serine]
    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {"compounds": copy.deepcopy(rows), "proteins": []},
        "processes": {"reactions": [{"name": "R1", "inputs": ["Glycine"], "outputs": []}]},
    }


def _auditor_payload(order: str) -> Dict[str, Any]:
    """The regression fixture ``pwml-bio-auditor`` specified in **D-043**, transcribed
    exactly from the real ``PMC12312563/strict`` collision and DB-free."""

    naphthoate = {"name": "1,4-dihydroxy-2-naphthoic acid", "synonyms": ["DHNA"],
                  "pathbank_compound_id": 40747, "db_status": "unmatched"}
    dhna = {"name": "DHNA", "db_status": "unmatched"}
    rows = [naphthoate, dhna] if order == "naphthoate_first" else [dhna, naphthoate]
    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {"compounds": copy.deepcopy(rows), "proteins": []},
        "processes": {"reactions": [
            {"name": "R1", "inputs": ["DHNA-CoA"], "outputs": ["DHNA"]}]},
    }


def _control_payload() -> Dict[str, Any]:
    """One row, no overlap: the diagnostic must stay silent and nothing may move."""

    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {"compounds": [{"name": "glycine", "db_status": "unmatched"}],
                     "proteins": []},
        "processes": {"reactions": [{"name": "R1", "inputs": ["Glycine"], "outputs": []}]},
    }


def _slot_duplicate_payload() -> Dict[str, Any]:
    """D-041 section 5's false positive: ONE entity occupying two alias slots under one
    ``_norm`` key. ``len(candidates) > 1`` is true; the ambiguity is not."""

    return {
        "metadata": {"pathway_name": "P", "pathway_subject": "Metabolic"},
        "entities": {"compounds": [{"name": "glycine", "raw_name": "Glycine",
                                    "synonyms": ["GLYCINE"], "db_status": "unmatched"}],
                     "proteins": []},
        "processes": {"reactions": [{"name": "R1", "inputs": ["Glycine"], "outputs": []}]},
    }


def _differential_arm(ir_mod: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    arm: Dict[str, Any] = {}
    with _recording(ir_mod) as created:
        try:
            ir, report = ir_mod.build_pwml_ir(
                copy.deepcopy(payload), pathway_name="P", pathway_subject="Metabolic",
                strict_db=False, name_index=None)
        except Exception as exc:  # noqa: BLE001
            arm["status"] = "RAISED"
            arm["detail"] = f"{type(exc).__name__}: {exc}"[:400]
            return arm
    arm["status"] = "BUILT"
    arm["report_ok"] = report["ok"]
    reaction = ir["processes"]["reactions"][0]
    # ``left`` for an inputs-side fixture, ``right`` for the auditor's outputs-side one.
    left = (reaction.get("left") or []) or (reaction.get("right") or [])
    if left:
        bound_key = left[0]["entity_key"]
        bound = next(row for row in ir["entities"]["compounds"] if row["key"] == bound_key)
        arm["bound_entity_key"] = bound_key
        arm["bound_row_name"] = bound["name"]
    else:
        arm["bound_entity_key"] = None
    arm["ambiguous_entity_reference"] = [
        issue for bucket in ("errors", "warnings") for issue in report.get(bucket, [])
        if issue.get("code") == "ambiguous_entity_reference"]
    # C-050k's own code, kept SEPARATE from the type-based one above (D-043 section 4).
    arm["ambiguous_entity_row_reference"] = [
        issue for bucket in ("errors", "warnings") for issue in report.get(bucket, [])
        if issue.get("code") == "ambiguous_entity_row_reference"]
    arm["row_reference_severity"] = ("error" if any(
        i.get("code") == "ambiguous_entity_row_reference" for i in report.get("errors", []))
        else "warning" if arm["ambiguous_entity_row_reference"] else "none")
    arm["ambiguous_severity"] = ("error" if any(
        i.get("code") == "ambiguous_entity_reference" for i in report.get("errors", []))
        else "warning" if arm["ambiguous_entity_reference"] else "none")
    arm["error_codes"] = [i.get("code") for i in report.get("errors", [])]
    arm["warning_codes"] = [i.get("code") for i in report.get("warnings", [])]
    index = created[-1] if created else None
    if index is not None:
        arm["index"] = _analyse(ir_mod, index)
    return arm


#: C-050i's R3 control -- "non-colliding" for ``_dedupe_named_rows``, which groups on
#: ``_norm(name)``, and therefore exactly the leg that shows the alias surface is a
#: different surface: the census found the ``dhna`` ambiguity on it.
R3_CONTROL_LEG = ("runs/2026-07-27_1623/papers/"
                  "PMC12312563__structures-of-listeria-monocytogenes-mend-in-th/strict/"
                  "final_mapped.json")


def _r3_control(ir_mod: Any, fixtures: Path) -> Dict[str, Any]:
    """The R3 control leg under golden configuration B, digested and itemised."""

    from t2pw.pwml.prefreeze_resolution import run_prefreeze_resolution

    class _DownDb:
        last_error = "harvest_db_down"

        def available(self) -> bool:
            return False

    staged = json.loads((fixtures / R3_CONTROL_LEG).read_text(encoding="utf-8"))
    run_prefreeze_resolution(staged, strict_db=True, db_resolver=_DownDb(), name_index=None)
    ir, report = ir_mod.build_pwml_ir(staged, db_resolver=_DownDb(), strict_db=True,
                                      name_index=None)
    return {
        "ir_digest": hashlib.sha256(
            json.dumps(ir, sort_keys=True, indent=1, default=repr).encode()).hexdigest(),
        "report_digest": hashlib.sha256(
            json.dumps(report, sort_keys=True, indent=1, default=repr).encode()).hexdigest(),
        "report_ok": report["ok"],
        "error_codes": [i.get("code") for i in report.get("errors", [])],
        "warning_codes": [i.get("code") for i in report.get("warnings", [])],
        "row_reference_issues": [i for i in report.get("warnings", [])
                                 if i.get("code") == "ambiguous_entity_row_reference"],
    }


def _mode_g9(tree: Path, fixtures: Path) -> Dict[str, Any]:
    """**The G9 proof, and it is a CORRECTION, not new functionality** (D-043 section 4).

    ``main`` returns **exit 1** when the row-level diagnostic is missing on the
    auditor's fixture, so this fails *behaviourally* in a real git worktree at the base
    SHA and passes at the tip. The base failure is not fabricated and is not a symbol
    check: the base runs the same fixture through the same real ``build_pwml_ir`` and
    silently binds a row while recording nothing, which is the ``PRODUCT_CONTRACT``
    section 3 violation D-043 section 2 names.

    Three claims, all of which must hold at the tip:

    1. **fires** on the auditor's fixture, naming both candidate keys;
    2. **binding unchanged** -- ``cmp_1`` in both payload row orders;
    3. **silent** on one entity occupying several alias slots (D-041 section 5's 209
       false positives). Claim 3 is what stops claim 1 being satisfied by a diagnostic
       that fires on everything.
    """

    from t2pw.pwml import ir as ir_mod

    out: Dict[str, Any] = {"t2pw_file": ir_mod.__file__,
                           "label": "G9 CORRECTION PROOF: exits 1 where the diagnostic is absent"}
    arms = {
        "auditor_fixture_naphthoate_first": _differential_arm(
            ir_mod, _auditor_payload("naphthoate_first")),
        "auditor_fixture_reversed": _differential_arm(
            ir_mod, _auditor_payload("dhna_first")),
        "control_slot_duplicate_single_entity": _differential_arm(
            ir_mod, _slot_duplicate_payload()),
        "control_single_row": _differential_arm(ir_mod, _control_payload()),
    }
    out["arms"] = arms
    fires = [name for name in ("auditor_fixture_naphthoate_first", "auditor_fixture_reversed")
             if arms[name].get("ambiguous_entity_row_reference")]
    silent = [name for name in ("control_slot_duplicate_single_entity", "control_single_row")
              if not arms[name].get("ambiguous_entity_row_reference")]
    bindings = {name: arms[name].get("bound_entity_key") for name in arms}
    out["claim_1_fires_on_ambiguity"] = len(fires) == 2
    out["claim_2_binding_unchanged"] = all(key == "cmp_1" for key in bindings.values())
    out["claim_3_silent_on_slot_duplication"] = len(silent) == 2
    out["severities"] = {name: arms[name].get("row_reference_severity") for name in arms}
    out["claim_4_severity_is_warning_only"] = all(
        value in ("warning", "none") for value in out["severities"].values())
    out["bindings"] = bindings
    # The merge rule 4 delta, captured with the SAME harness on BOTH trees, which is
    # what F-047 requires of any harness that is not pytest. Configuration B
    # (DB down, no index, strict) is exactly what the R3 control test runs.
    out["r3_control_leg"] = _r3_control(ir_mod, fixtures)
    out["verdict"] = "PASS" if all(
        out[key] for key in ("claim_1_fires_on_ambiguity", "claim_2_binding_unchanged",
                             "claim_3_silent_on_slot_duplication",
                             "claim_4_severity_is_warning_only")) else "FAIL"
    return out


def _mode_differential(tree: Path, fixtures: Path) -> Dict[str, Any]:
    from t2pw.pwml import ir as ir_mod

    return {
        "t2pw_file": ir_mod.__file__,
        "label": "DIFFERENTIAL EVIDENCE, NOT A G9 REGRESSION PROOF",
        "serine_first": _differential_arm(ir_mod, _decoy_payload("serine_first")),
        "glycine_first": _differential_arm(ir_mod, _decoy_payload("glycine_first")),
        "control_single_row": _differential_arm(ir_mod, _control_payload()),
        "control_slot_duplicate_single_entity": _differential_arm(
            ir_mod, _slot_duplicate_payload()),
    }


def evidence_dir() -> Path:
    """This file's own directory -- where ``tree_pin`` lives."""
    return Path(__file__).resolve().parent


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="C-050k alias-ambiguity probe")
    parser.add_argument("--mode", required=True, choices=("census", "differential", "g9"))
    parser.add_argument("--tree", required=True, help="the tree under measurement")
    parser.add_argument("--fixtures", default=None,
                        help="checkout to read committed runs/ legs from (default: --tree)")
    parser.add_argument("--pin-verdict", default=None,
                        help="where tree_pin writes its verdict (TEST_MATRIX section 0 rule 10)")
    parser.add_argument("--dotenv", default=None,
                        help="a .env READ where it lives and never copied; EP3 reaches the "
                             "PathBank DB through os.getenv, so this reproduces it")
    parser.add_argument("--out", required=True, help="JSON destination")
    args = parser.parse_args(argv)

    # F-045: resolve EVERY path before anything chdirs, and never write into the tree
    # being audited.
    out_path = Path(args.out).resolve()
    tree = Path(args.tree).resolve()
    fixtures = Path(args.fixtures).resolve() if args.fixtures else tree
    verdict_path = str(Path(args.pin_verdict).resolve()) if args.pin_verdict else None
    dotenv_path = Path(args.dotenv).resolve() if args.dotenv else None
    cwd_at_entry = os.getcwd()

    if dotenv_path is not None:
        assert dotenv_path.is_file(), f"--dotenv not found: {dotenv_path}"
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=str(dotenv_path), override=False)

    # H-010's measurement-tree pin, called rather than reimplemented and NOT weakened:
    # ``<tree>/src`` is deliberately not placed on ``sys.path`` -- repairing the path
    # here would make the wrong-tree refusal unreachable. The caller pins
    # ``PYTHONPATH=<tree>/src``; the verdict settles which tree ran.
    sys.path.insert(0, str(evidence_dir()))
    import tree_pin

    if str(tree) not in sys.path:
        sys.path.insert(0, str(tree))
    os.chdir(tree)
    facts = tree_pin.enforce(expected=tree, require_scripts=False,
                             verdict_path=verdict_path, cwd_at_entry=cwd_at_entry)

    modes = {"census": _mode_census, "differential": _mode_differential, "g9": _mode_g9}
    payload = modes[args.mode](tree, fixtures)
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
    if args.mode == "g9":
        print(json.dumps({k: v for k, v in payload.items()
                          if k.startswith(("claim_", "verdict", "bindings", "severities"))},
                         indent=2))
        # The behavioural failure IS the exit code. At the base SHA claim 1 is false
        # and this returns 1; at the tip every claim holds and it returns 0.
        return 0 if payload["verdict"] == "PASS" else 1
    if args.mode == "census":
        print(json.dumps(payload["totals"], indent=2))
        print(f"stop_condition_2_triggered: {payload['stop_condition_2_triggered']}")
        print(f"db_available: {payload['db_available']} ({payload['db_reason']})")
    else:
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "index"}
                          for k, v in payload.items()
                          if isinstance(v, dict) and "status" in v}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
