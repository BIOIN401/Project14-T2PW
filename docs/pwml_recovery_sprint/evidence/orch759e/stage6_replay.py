"""ORCH-759e: replay Stage 6 (``map_payload``) on archived strict legs, offline, at whatever code is checked out.

Run it at base and at tip; it imports nothing ORCH-759e adds, so the same file measures both.

WHAT IS REPLAYED. The leg's ``final_mapped.json`` is turned back into what Stage 6 saw when its
wrapper rewrite ran, as far as the defect needs, and fed to ``map_payload`` with the Stage-6 flags
(``allow_complex_wrapper_creation=True``, ``allow_structural_cleanup=True``,
``merge_identity_duplicates=True``):

* a protein the Phase-2 UniProt fallback resolved (``mapping_meta.chosen_rule ==
  "api_uniprot_fallback"`` with an identity) loses that identity again: the main loop answers
  "unmapped" for it and the Phase-2 fallback answers with the accession the leg recorded;
* a PathBank-Unknown placeholder complex standing beside such a protein in an actor list was a
  bare unresolved protein at the rewrite (otherwise the rewrite's drop rule would have removed the
  late protein, and the leg shows it did not), so it is reverted to one, everywhere it is named,
  as ``orch759_cpo_diag/replay_cpo.py`` M2 did by hand for CPO1. The Unknown fallback rebuilds it.

Every identity DECISION is the leg's, replayed, never re-judged: the main loop returns each row's
recorded mapping, the name gates (``_enforce_shipped_identity_names``,
``_apply_name_plausibility_gate``, ``_break_gene_synonym_collision``) are pass-throughs, species
hydration and the taxonomy scope are off, ``id_source="api"`` (no PathBank DB), and every HTTP
request is refused and counted (must be zero). Only the Stage-6 sequence runs live.

CHECKS (``--checks``, legs with a late protein): strict quarantine with the leg's own requested
context, the final pre-export gate, the required-field contract and ``validate_pre_export``, and
the IR, exactly as ``replay_cpo.py`` runs them -- on the live artifact and on the replay.

``--variant`` (tip only, measurement of the placement choice): ``as_is`` (the code checked out),
``m1`` (the targeted rewrite moved after both Unknown fallbacks), ``m2_full`` / ``m1_full`` (the
WHOLE rewrite re-run at either point, the ORCH-759 diagnosis counterfactuals).

Output: one JSON line per leg to ``--out``; a summary on stdout. Writes nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

WT = Path(__file__).resolve().parents[4]
PRIMARY = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
for entry in (WT / "src", WT / "tests"):
    sys.path.insert(0, str(entry))

from helpers_prefreeze import prefrozen_when_compounded  # noqa: E402
from test_strict_quarantine_real_artifact_replay import _as_export_payload  # noqa: E402
from t2pw.mapping import map_ids as M  # noqa: E402
from t2pw.pipeline.entity_identity import has_protein_external_identity, is_pathbank_unknown_protein  # noqa: E402
from t2pw.pipeline.process_normalizer import GateValidationError, run_strict_post_normalization_gates  # noqa: E402
from t2pw.pipeline.stage_contracts import validate_pre_export  # noqa: E402
from t2pw.pipeline.strict_quarantine import quarantine_and_close  # noqa: E402
from t2pw.pwml.ir import build_pwml_ir, validate_pwml_ir, validate_required_pwml_contract  # noqa: E402
from t2pw.pwml.writer import blocking_pwml_ir_errors  # noqa: E402

DIRS = ["runs_papers", "runs_verify", "runs", "runs_validation", "runs_smoke"]
ID_KEYS = ("uniprot", "uniprot_id", "uniprot-id", "drugbank", "drugbank_id", "drugbank-id")
FALLBACK_RULE = "pathbank_unknown_protein_fallback"
META_KEYS = ("provider", "source", "chosen_rule", "confidence", "reviewed", "resolution", "candidates",
             "matched_alias", "alias_source", "resolved_name", "queries_tried", "identity_verdict")


def norm(value: Any) -> str:
    return M._normalize_name(str(value or ""))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def actor_lists(payload: Dict[str, Any]):
    procs = payload.get("processes") or {}
    for ridx, rx in enumerate(procs.get("reactions") or []):
        for key in ("enzymes", "modifiers"):
            if isinstance(rx, dict) and isinstance(rx.get(key), list):
                yield f"/processes/reactions/{ridx}/{key}", rx[key]
    for tidx, tx in enumerate(procs.get("transports") or []):
        if isinstance(tx, dict) and isinstance(tx.get("transporters"), list):
            yield f"/processes/transports/{tidx}/transporters", tx["transporters"]


def actor_name(actor: Any) -> str:
    return M._reaction_actor_name_and_type(actor)[0]


def ids_of(row: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for container in (row, row.get("mapped_ids") or {}):
        for key in ("uniprot", "drugbank"):
            value = str((container or {}).get(key) or "").strip()
            if value and key not in out:
                out[key] = value
    return out


# ── the Stage-6 input ─────────────────────────────────────────────────────────────────────


def reconstruct(fm: Dict[str, Any]):
    """``(input payload, late names, reverted placeholder names, recorded rows)``."""

    payload = deepcopy(fm)
    ents = payload.setdefault("entities", {})
    proteins = ents.setdefault("proteins", [])
    recorded = {norm(r.get("name")): deepcopy(r) for r in proteins if isinstance(r, dict)}
    late = {
        n for n, r in recorded.items()
        if (r.get("mapping_meta") or {}).get("chosen_rule") == "api_uniprot_fallback"
        and has_protein_external_identity(r) and not is_pathbank_unknown_protein(r)
    }
    for row in proteins:
        if isinstance(row, dict) and norm(row.get("name")) in late:
            for container in (row, row.get("mapped_ids"), row.get("ids"), row.get("mapping_meta")):
                if isinstance(container, dict):
                    for key in ID_KEYS:
                        container.pop(key, None)
    complexes = {norm(c.get("name")): c for c in ents.get("protein_complexes") or [] if isinstance(c, dict)}
    placeholders = {n for n, c in complexes.items() if (c.get("mapping_meta") or {}).get("chosen_rule") == FALLBACK_RULE}
    revert = set()
    for _pointer, rows in actor_lists(payload):
        names = {norm(actor_name(a)) for a in rows}
        if names & late:
            revert |= names & placeholders
    for n in sorted(revert):
        cx = complexes[n]
        meta = cx.get("mapping_meta") or {}
        organism = meta.get("target_organism") or cx.get("organism") or cx.get("species")
        row = {"name": cx["name"], "species": organism, "organism": organism}
        for key in ("species_id", "pathbank_species_id", "taxonomy_id", "species_ref"):
            if key in cx:
                row[key] = deepcopy(cx[key])
        row["mapping_meta"] = {"chosen_rule": "novel_protein", "resolution": {
            "status": "novel", "issue": meta.get("real_mapping_failure_reason") or "no_match", "order_step": "novel_protein"}}
        for key in ("provenance", "source_refs"):
            if key in meta:
                row[key] = deepcopy(meta[key])
        proteins.append(row)
        recorded[n] = deepcopy(row)
        for _pointer, rows in actor_lists(payload):
            for actor in rows:
                if isinstance(actor, dict) and norm(actor_name(actor)) == n:
                    actor["entity_type"] = "protein"
    ents["protein_complexes"] = [c for c in ents.get("protein_complexes") or []
                                 if not (isinstance(c, dict) and norm(c.get("name")) in revert)]
    return payload, late, revert, recorded


# ── the outside world, replayed ─────────────────────────────────────────────────────────


class NoNetwork:
    calls: List[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get(self, url: str, *args: Any, **kwargs: Any) -> Any:
        NoNetwork.calls.append(str(url))
        raise ConnectionError("ORCH-759e replay is offline")

    post = get


def recorded_result(row: Optional[Dict[str, Any]], *, identified: bool) -> Dict[str, Any]:
    meta = dict((row or {}).get("mapping_meta") or {})
    out = {key: deepcopy(meta[key]) for key in META_KEYS if key in meta}
    out.setdefault("provider", "UniProt")
    out.setdefault("source", "api")
    out.setdefault("candidates", [])
    out.setdefault("confidence", 0.0)
    if identified:
        out.update(status="mapped", mapped_ids=ids_of(row or {}))
        if (row or {}).get("pathbank_protein_id"):
            out["pathbank_protein_id"] = row["pathbank_protein_id"]
        out.setdefault("resolution", {"status": "matched", "order_step": "replayed"})
    else:
        resolution = out.get("resolution") or {"status": "unresolved", "issue": "no_match"}
        out.update(status="unmapped", reason=str(resolution.get("issue") or "no_match"), resolution=resolution)
    return out


@contextmanager
def offline_stage6(recorded: Dict[str, Dict[str, Any]], late: set, compounds: Dict[str, Dict[str, Any]]):
    def main_loop(**kw: Any) -> Dict[str, Any]:
        key = norm(kw.get("name"))
        row = recorded.get(key)
        if key in late or row is None:
            return recorded_result(row, identified=False)
        return recorded_result(row, identified=has_protein_external_identity(row))

    def phase2(_client: Any, name: str, _organism: str, **_kw: Any) -> Dict[str, Any]:
        key = norm(name)
        row = recorded.get(key)
        if key not in late:
            return {"status": "unmapped", "reason": "no_match", "provider": "UniProt", "source": "api",
                    "confidence": 0.0, "candidates": []}
        verdict = (row.get("mapping_meta") or {}).get("identity_verdict") or {}
        cand = verdict.get("judged_candidate")
        return {"status": "mapped", "provider": "UniProt", "source": "api", "confidence": 1.0,
                "chosen_rule": "replayed_phase2", "mapped_ids": ids_of(row),
                "candidates": [deepcopy(cand)] if cand else [], "identity_verdict": deepcopy(verdict) or None,
                "resolution": {"status": "matched", "order_step": "api_uniprot"}}

    def compound(**kw: Any) -> Dict[str, Any]:
        row = compounds.get(norm(kw.get("name"))) or kw.get("compound_row") or {}
        meta = dict(row.get("mapping_meta") or {})
        ids = dict(row.get("mapped_ids") or {})
        out = {key: deepcopy(meta[key]) for key in ("provider", "source", "chosen_rule", "confidence", "resolution",
                                                     "candidates") if key in meta}
        out.setdefault("candidates", [])
        out.setdefault("confidence", 0.0)
        if ids:
            out.update(status="mapped", mapped_ids=ids)
            if row.get("pathbank_compound_id"):
                out["pathbank_compound_id"] = row["pathbank_compound_id"]
        else:
            out.update(status="unmapped", reason="no_match")
        return out

    NoNetwork.calls = []
    with ExitStack() as stack:
        for name, value in (
            ("HttpClient", NoNetwork),
            ("hydrate_species_references", lambda *a, **k: {"hydrated": 0, "matched": 0, "novel": 0}),
            ("_memo_taxon_scope", lambda *a, **k: None),
            ("_map_protein_with_strategy", main_loop),
            ("map_protein_uniprot", phase2),
            ("_map_compound_with_strategy", compound),
            ("_enforce_shipped_identity_names", lambda *a, **k: {}),
            ("_apply_name_plausibility_gate", lambda result, *a, **k: result),
            ("_break_gene_synonym_collision", lambda result, *a, **k: result),
        ):
            stack.enter_context(patch.object(M, name, value))
        yield


ORIG_REWRITE = M._rewrite_reaction_protein_enzymes_to_complexes
ORIG_CFB = M._apply_pathbank_unknown_complex_fallback


@contextmanager
def placement(variant: str):
    """Tip only: move or widen the call ``map_payload`` makes with ``only_proteins``."""

    if variant == "as_is":
        yield
        return
    deferred: List[Any] = []

    def rewrite(mapped: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
        if kw.get("only_proteins") is None:
            return ORIG_REWRITE(mapped, **kw)
        if variant == "m2_full":
            kw.pop("only_proteins")
            return ORIG_REWRITE(mapped, **kw)
        deferred.append((mapped, kw))
        return {"summary": {}, "actions": []}

    def complex_fallback(mapped: Dict[str, Any]) -> Dict[str, Any]:
        out = ORIG_CFB(mapped)
        for target, kw in deferred:
            if variant == "m1_full":
                kw = {k: v for k, v in kw.items() if k != "only_proteins"}
            ORIG_REWRITE(target, **kw)
        deferred.clear()
        return out

    with patch.object(M, "_rewrite_reaction_protein_enzymes_to_complexes", side_effect=rewrite), \
            patch.object(M, "_apply_pathbank_unknown_complex_fallback", side_effect=complex_fallback):
        yield


def replay(fm: Dict[str, Any], scratch: Path, variant: str):
    payload, late, revert, recorded = reconstruct(fm)
    compounds = {norm(r.get("name")): r for r in (fm.get("entities") or {}).get("compounds") or [] if isinstance(r, dict)}
    with offline_stage6(recorded, late, compounds), placement(variant):
        result = M.map_payload(payload, cache_path=scratch / "never-written-cache.json", id_source="api",
                               use_cache=False, allow_complex_wrapper_creation=True, allow_structural_cleanup=True,
                               merge_identity_duplicates=True)
    assert NoNetwork.calls == [], NoNetwork.calls[:3]
    return result, payload, late, revert


# ── checks, as replay_cpo.py runs them ─────────────────────────────────────────────────────


def rxn_name(payload: Dict[str, Any], pointer: str) -> str:
    parts = (pointer or "").split("/")
    try:
        if parts[1] == "processes":
            return str(payload["processes"][parts[2]][int(parts[3])].get("name"))
    except Exception:  # noqa: BLE001
        return ""
    return ""


def checks(payload: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    res = quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=ctx, mode="pathwhiz")
    rep = res.quarantine_report
    rel = rep.get("release") or {}
    out["quarantine"] = {"ok": res.ok, "release": rel.get("status"), "reasons": rel.get("reasons"),
                         "review_reasons": rel.get("review_reasons"), "counts": rep.get("counts"),
                         "refusal": rep.get("refusal_reasons")}
    shipped = res.payload or {}
    if res.ok:
        try:
            run_strict_post_normalization_gates(deepcopy(shipped), enforce_all_proteins_connected=True)
            out["final_gate"] = "ok"
        except GateValidationError as exc:
            out["final_gate"] = f"FAIL {str(exc)[:160]}"
    contract = validate_required_pwml_contract(_as_export_payload(shipped), strict_db=True)
    out["contract"] = sorted({f"{e.get('code')} @ {rxn_name(shipped, e.get('pointer'))}" for e in contract.get("errors") or []})
    try:
        validate_pre_export(_as_export_payload(shipped), strict_db=True)
        out["pre_export"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- StageContractError
        det = getattr(exc, "report", None) or getattr(exc, "details", None) or {}
        out["pre_export"] = sorted({str(i.get("code")) for i in (det.get("errors") or [])}) if isinstance(det, dict) else str(exc)[:160]
    try:
        try:
            staged = prefrozen_when_compounded(deepcopy(shipped))
        except AssertionError:
            staged = deepcopy(shipped)  # a leg from after C-051: its compounds already carry their verdicts
        ir, ir_report = build_pwml_ir(staged, pathway_name="Replayed pathway",
                                      pathway_subject="Metabolic", strict_db=True, width=3200, height=1400)
        out["ir"] = sorted({f"{i.get('code')} @ {rxn_name(shipped, i.get('pointer'))}" for i in blocking_pwml_ir_errors(ir_report)}
                           | {f"{i.get('code')}" for i in validate_pwml_ir(ir).get("errors", [])})
    except Exception as exc:  # noqa: BLE001
        out["ir"] = [f"ir_build_crashed:{type(exc).__name__}:{str(exc)[:120]}"]
    return out


def late_actor_view(payload: Dict[str, Any], late: set, revert: set) -> List[List[Any]]:
    """Where each late protein and each reverted placeholder stands after the replay."""

    view = []
    for pointer, rows in actor_lists(payload):
        for actor in rows:
            name, typ, _ = M._reaction_actor_name_and_type(actor)
            stem = norm(name[: -len(" complex")] if name.lower().endswith(" complex") else name)
            if norm(name) in late | revert or stem in late:
                view.append([pointer, name, typ])
    for iidx, row in enumerate((payload.get("processes") or {}).get("interactions") or []):
        for field in ("entity_1", "entity_2", "left", "right", "source", "target"):
            name = str((row or {}).get(field) or "")
            stem = norm(name[: -len(" complex")] if name.lower().endswith(" complex") else name)
            if name and (norm(name) in late or stem in late):
                view.append([f"/processes/interactions/{iidx}/{field}", name, "endpoint"])
    return view


def actor_count(payload: Dict[str, Any]) -> int:
    return sum(len(rows) for _p, rows in actor_lists(payload))


def legs(args: argparse.Namespace) -> List[Path]:
    if args.legs:
        return [PRIMARY / leg for leg in args.legs]
    found = []
    for d in DIRS:
        for fm in sorted((PRIMARY / d).rglob("final_mapped.json")):
            posix = fm.as_posix()
            if "/research/" in posix or "worktrees" in posix:
                continue  # research mode never wraps; same population as scan_shape.py
            found.append(fm.parent)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--variant", default="as_is", choices=["as_is", "m1", "m2_full", "m1_full"])
    ap.add_argument("--checks", action="store_true", help="quarantine/gate/contract/IR on legs with a late protein")
    ap.add_argument("--legs", nargs="*", help="leg directories relative to the primary checkout (default: all)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"variant={args.variant} worktree={WT}")
    n_late = n_err = 0
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for leg in legs(args):
            rel = leg.relative_to(PRIMARY).as_posix()
            rec: Dict[str, Any] = {"leg": rel}
            try:
                fm = json.loads((leg / "final_mapped.json").read_text(encoding="utf-8"))
                result, before, late, revert = replay(fm, scratch, args.variant)
                after = result["payload"]
                rec.update(late=sorted(late), reverted_placeholders=sorted(revert),
                           payload_digest=digest(after), report_digest=digest(result["report"]),
                           actors_in=actor_count(before), actors_out=actor_count(after),
                           late_actors=late_actor_view(after, late, revert))
                if late:
                    n_late += 1
                    if args.checks:
                        ctx = None
                        qpath = leg / "quarantine_report.json"
                        if qpath.exists():
                            ctx = (json.loads(qpath.read_text(encoding="utf-8")).get("coverage") or {}).get("requested_context")
                        rec["live_checks"] = checks(fm, ctx)
                        rec["replay_checks"] = checks(after, ctx)
            except Exception as exc:  # noqa: BLE001 -- a leg that cannot be replayed is reported, not hidden
                n_err += 1
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            flag = "LATE" if rec.get("late") else ("ERROR" if "error" in rec else "")
            print(f"{flag:5} {rel}  digest={rec.get('payload_digest')} late={rec.get('late')}")
            if rec.get("late"):
                print("      actors:", rec["late_actors"], "| in/out:", rec.get("actors_in"), rec.get("actors_out"))
                for side in ("live_checks", "replay_checks"):
                    if side in rec:
                        c = rec[side]
                        print(f"      {side}: quarantine={c['quarantine']['ok']}/{c['quarantine']['release']} "
                              f"counts={c['quarantine']['counts']} gate={c.get('final_gate')} "
                              f"contract={c['contract']} pre_export={c['pre_export']} ir={c['ir']}")
            if "error" in rec:
                print("      ", rec["error"])
    print(f"legs={len(legs(args))} with_late={n_late} errors={n_err} seconds={time.time() - t0:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
