"""ORCH-754b -- base-vs-tip record for interaction-endpoint wrappers (D-104).

Two claims, both measured on the same runs, base ``f9892cdf`` against the working tree:

1. **Preservation (G9).** On every input whose interactions name no protein, the Stage-6
   wrapper step, the IR (bar the ``evidence`` this card adds to each withheld row's record),
   the IR report and the PWML bytes are byte-identical base vs tip.
   Reaction enzymes and their wrappers are unchanged.
3. **Correction (G9).** ``competes_with`` is drawn as Inhibition on the base and withheld
   as untyped at the tip (input ``correction:competes_with``).
2. **New capability, measured (not a G9 base failure).** On the inputs that do name a
   protein endpoint, how many interactions reach the PWML before and after.

Inputs: every committed ``final_mapped.json`` under ``runs/`` and ``runs_verify/`` with at
least one interaction (the ORCH-749 corpus), plus two synthetic fixtures (an enzyme-only
payload and the protein-endpoint payload of ``tests/test_pwml_interaction_protein_wrappers.py``).

Two configurations per input, both offline (no DB, no LLM, no network, no cache file):

* ``direct`` -- ``build_pwml_ir`` on the payload exactly as committed, DB down: the ORCH-749
  census configuration. The committed payloads were mapped before this card, so they carry
  no interaction wrapper; this measures what the exporter alone now says about them.
* ``replay`` -- the Stage-6 wrapper step ``_rewrite_reaction_protein_enzymes_to_complexes``
  re-run offline on the payload (``db=None``, cache disabled: the no-database branch, so a
  protein never maps to a PathBank complex here), then ``run_prefreeze_resolution`` with the
  DB down, then the IR, then the writer. **This is an estimate of what a real Stage 6 would
  do, not a re-run of it**: with the database up a protein can resolve to a PathBank
  complex instead, and the later mapping passes (UniProt fallback, identity admission) can
  change which proteins carry an identity.

Each tree runs in its own child interpreter that verifies which ``t2pw`` it imported.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch754b_wrapper_base_proof.py [--rev f9892cdf] [--out FILE]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
PACKAGE_REL = "src/t2pw"
DEFAULT_OUT = Path(__file__).resolve().parent / "orch754b_wrapper_base_proof.json"
REFERENCE = REPO / "reference" / "PW000001.pwml"


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


def _sha(value: Any) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, indent=1, default=repr).encode()
    return hashlib.sha256(data).hexdigest()


def _fixtures() -> Dict[str, Dict[str, Any]]:
    sys.path.insert(0, str(REPO / "tests"))
    import test_c086_superset_complex_promotion as c086  # noqa: E402
    import test_pwml_interaction_protein_wrappers as w  # noqa: E402

    enzymes = c086._enterobactin_payload()
    enzymes["entities"]["species"] = [{"name": "Escherichia coli", "pathwhiz_id": 3}]
    # A reaction whose bare enzyme Stage 6 must wrap, plus a compound-only interaction:
    # the shape that has to export byte-identically.
    reaction = w._payload()
    reaction["entities"]["compounds"] = [
        {"name": "glycine", "pathbank_compound_id": 21, "mapped_ids": {"chebi": "CHEBI:57305"}},
        {"name": "5-aminolevulinate", "pathbank_compound_id": 1180, "mapped_ids": {"chebi": "CHEBI:356416"}},
    ]
    reaction["processes"]["reactions"] = [{
        "name": "ALAS2 step", "biological_state": "mitochondrion",
        "inputs": [{"name": "glycine", "stoichiometry": 1}], "outputs": [{"name": "5-aminolevulinate", "stoichiometry": 1}],
        "enzymes": [{"protein": "ALAS2", "provenance": "extracted"}],
    }]
    reaction["processes"]["interactions"] = [
        w._interaction("glycine activates 5-aminolevulinate", "glycine", "activates", "5-aminolevulinate"),
    ]
    competes = deepcopy(reaction)
    competes["processes"]["interactions"] = [
        w._interaction("glycine competes with 5-aminolevulinate", "glycine", "competes_with", "5-aminolevulinate"),
    ]
    return {"fixture:enzymes_only": enzymes, "fixture:enzyme_reaction_no_protein_endpoint": reaction,
            "fixture:protein_endpoints": w._payload(), "correction:competes_with": competes}


def _without_withheld_evidence(ir: Dict[str, Any]) -> Dict[str, Any]:
    """The IR minus the one annotation this card adds to EVERY withheld row (its evidence),
    so preservation is judged on everything else."""
    ir = deepcopy(ir)
    for row in ir["processes"]["interactions"]:
        (row.get("withheld") or {}).pop("evidence", None)
    return ir


def _protein_endpoint_rows(payload: Dict[str, Any]) -> int:
    proteins = {str(p.get("name") or "").strip().casefold() for p in (payload.get("entities") or {}).get("proteins") or []
                if isinstance(p, dict)} - {""}
    return sum(
        1
        for row in (payload.get("processes") or {}).get("interactions") or []
        if isinstance(row, dict)
        and any(str(row.get(f) or "").strip().casefold() in proteins for f in ("entity_1", "entity_2", "left", "right", "source", "target"))
    )


# --------------------------------------------------------------------------- child
def _child(expect_tree: Path, inputs_file: Path, out_file: Path) -> int:
    logging.disable(logging.CRITICAL)
    import t2pw
    from lxml import etree
    from t2pw.mapping.map_ids import MappingCache, _extract_global_organism, _rewrite_reaction_protein_enzymes_to_complexes
    from t2pw.pwml.ir import build_pwml_ir, validate_required_pwml_contract
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError, run_prefreeze_resolution
    from t2pw.pwml.validate import discover_structure_signature, repair_tree
    from t2pw.pwml.writer import DeterministicPwmlBuilder

    resolved = Path(t2pw.__file__).resolve()
    if (expect_tree / PACKAGE_REL).resolve() not in resolved.parents:
        print(json.dumps({"error": "T2PW_MEASUREMENT_TREE_REFUSED", "t2pw_file": str(resolved)}))
        return 98
    signature = discover_structure_signature(REFERENCE)
    args = SimpleNamespace(name="Generated Pathway", description="", subject="Metabolic", pw_id="PW000000",
                           height=1400, width=3200, background_color="#FFFFFF", ref=str(REFERENCE))

    def outcomes(ir: Dict[str, Any]) -> Dict[str, int]:
        return dict(Counter((r.get("withheld") or {}).get("reason", "exportable") for r in ir["processes"]["interactions"]))

    results: Dict[str, Any] = {"t2pw_file": str(resolved), "inputs": {}}
    for name, payload in json.loads(inputs_file.read_text(encoding="utf-8")).items():
        entry: Dict[str, Any] = {}
        try:
            ir, _report = build_pwml_ir(deepcopy(payload), db_resolver=_DownDb(), strict_db=True, name_index=None)
            entry["direct"] = outcomes(ir)
        except Exception as exc:  # noqa: BLE001 -- recorded and compared across trees
            entry["direct"] = {"stop": type(exc).__name__}

        mapped = deepcopy(payload)
        stage6 = _rewrite_reaction_protein_enzymes_to_complexes(
            mapped, db=None, cache=MappingCache(Path("unused.json"), enabled=False),
            global_organism=_extract_global_organism(mapped),
        )
        replay: Dict[str, Any] = {"stage6_payload_sha256": _sha(mapped), "stage6_report_sha256": _sha(stage6),
                                  "stage6_summary": stage6.get("summary"), "stage6_payload": deepcopy(mapped),
                                  "contract_error_codes": sorted({e.get("code") for e in
                                                                  validate_required_pwml_contract(deepcopy(mapped))["errors"]})}
        try:
            run_prefreeze_resolution(mapped, strict_db=True, db_resolver=_DownDb(), name_index=None)
            ir, report = build_pwml_ir(mapped, db_resolver=_DownDb(), strict_db=True, name_index=None)
        except PrefreezeResolutionError as exc:
            replay["stop"] = f"prefreeze:{exc.code}"
        except Exception as exc:  # noqa: BLE001
            replay["stop"] = f"build:{type(exc).__name__}"
        else:
            replay.update(outcomes=outcomes(ir), ir_sha256=_sha(ir), report_sha256=_sha(report), ir=ir,
                          ir_without_withheld_evidence_sha256=_sha(_without_withheld_evidence(ir)),
                          ir_error_codes=sorted({e.get("code") for e in report.get("errors") or []}))
            try:
                build = DeterministicPwmlBuilder(extraction=ir, signature=signature, args=args).build()
                xml = etree.tostring(repair_tree(etree.ElementTree(build.root), signature).getroot(),
                                     encoding="utf-8", xml_declaration=True, pretty_print=True)
                replay["pwml_sha256"] = _sha(xml)
                replay["pwml_interactions"] = xml.count(b"<interaction>")
                replay["pwml_protein_complex_interaction_elements"] = xml.count(b"<element-type>ProteinComplex</element-type>")
            except Exception as exc:  # noqa: BLE001
                replay["pwml_stop"] = type(exc).__name__
        entry["replay"] = replay
        results["inputs"][name] = entry
    out_file.write_text(json.dumps(results, default=repr), encoding="utf-8")
    print(json.dumps({"t2pw_file": str(resolved), "inputs": len(results["inputs"])}))
    return 0


# -------------------------------------------------------------------------- parent
def _export_base_tree(rev: str) -> Path:
    out = subprocess.run(["git", "archive", "--format=tar", rev, PACKAGE_REL], cwd=REPO, capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"git archive {rev} failed: {out.stderr.decode(errors='replace').strip()}")
    tree = Path(tempfile.mkdtemp(prefix="orch754b_base_"))
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        tar.extractall(tree)
    return tree


def _run_child(tree: Path, inputs_file: Path, out_file: Path) -> Dict[str, Any]:
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), T2PW_OFFLINE_CURATOR="1", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", "--expect-tree", str(tree),
                          "--inputs-file", str(inputs_file), "--out-file", str(out_file)],
                         cwd=tree, env=env, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"child failed in {tree} (exit {out.returncode}):\n{out.stdout}\n{out.stderr}")
    return json.loads(out_file.read_text(encoding="utf-8"))


def _sum(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    total: Counter = Counter()
    for e in entries:
        total.update({k: v for k, v in e.items() if isinstance(v, int)})
    return dict(total)


ENDPOINT_FIELDS = ("entity_1", "entity_2", "left", "right", "source", "target")


def _confinement(b: Dict[str, Any], t: Dict[str, Any]) -> List[str]:
    """Where tip differs from base outside what an interaction-endpoint wrapper may touch.

    Allowed: an interaction endpoint renamed to a NEW single-protein wrapper row appended
    to ``protein_complexes``; in the IR, the interaction rows, their geometry and the new
    complexes. Everything about reactions, transports, proteins and every pre-existing
    complex must be identical. ``[]`` means confined.
    """
    out: List[str] = []
    bp, tp = b["stage6_payload"], t["stage6_payload"]
    for key in sorted(set(bp) | set(tp)):
        if key not in ("entities", "processes") and bp.get(key) != tp.get(key):
            out.append(f"payload/{key}")
    be, te = bp.get("entities") or {}, tp.get("entities") or {}
    base_complexes, tip_complexes = be.get("protein_complexes") or [], te.get("protein_complexes") or []
    added = tip_complexes[len(base_complexes):]
    if tip_complexes[: len(base_complexes)] != base_complexes:
        out.append("payload/entities/protein_complexes: a pre-existing row changed")
    proteins = {str(r.get("name")): r for r in te.get("proteins") or [] if isinstance(r, dict)}
    for row in added:
        comps = row.get("components") or []
        if not (row.get("generated") is True and row.get("generation_reason") == "single_protein_pathwhiz_wrapper"
                and len(comps) == 1 and str((comps[0] or {}).get("name")) in proteins):
            out.append(f"payload/entities/protein_complexes: added row {row.get('name')!r} is not a one-protein wrapper")
    for key in sorted(set(be) | set(te)):
        if key != "protein_complexes" and be.get(key) != te.get(key):
            out.append(f"payload/entities/{key}")
    bpr, tpr = bp.get("processes") or {}, tp.get("processes") or {}
    for key in sorted(set(bpr) | set(tpr)):
        if key != "interactions" and bpr.get(key) != tpr.get(key):
            out.append(f"payload/processes/{key}")
    added_names = {str(r.get("name")) for r in added}
    b_rows, t_rows = bpr.get("interactions") or [], tpr.get("interactions") or []
    if len(b_rows) != len(t_rows):
        out.append("payload/processes/interactions: row count")
    for i, (br, tr) in enumerate(zip(b_rows, t_rows)):
        for key in sorted(set(br) | set(tr)):
            if br.get(key) == tr.get(key):
                continue
            if key in ENDPOINT_FIELDS and tr.get(key) in added_names | {str(r.get("name")) for r in base_complexes}:
                continue
            out.append(f"payload/processes/interactions/{i}/{key}")
    bi, ti = b.get("ir"), t.get("ir")
    if bi is not None and ti is not None:
        for key in ("reactions", "transports", "reaction_coupled_transports", "sub_pathways"):
            if bi["processes"].get(key) != ti["processes"].get(key):
                out.append(f"ir/processes/{key}")
        for key in sorted(set(bi["entities"]) | set(ti["entities"])):
            bl, tl = bi["entities"].get(key) or [], ti["entities"].get(key) or []
            if (tl[: len(bl)] != bl) if key == "protein_complexes" else (bl != tl):
                out.append(f"ir/entities/{key}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--rev", default="f9892cdf")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-tree", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--inputs-file", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return _child(Path(args.expect_tree), Path(args.inputs_file), Path(args.out_file))

    legs = sorted(p for p in subprocess.run(["git", "ls-files", "runs", "runs_verify"], cwd=REPO, capture_output=True,
                                            text=True, check=True).stdout.split("\n") if p.endswith("/final_mapped.json"))
    inputs: Dict[str, Any] = {}
    for leg in legs:
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        if (payload.get("processes") or {}).get("interactions"):
            inputs[leg] = payload
    inputs.update(_fixtures())
    work = Path(tempfile.mkdtemp(prefix="orch754b_proof_"))
    inputs_file = work / "inputs.json"
    inputs_file.write_text(json.dumps(inputs), encoding="utf-8")

    sha = subprocess.run(["git", "rev-parse", args.rev], cwd=REPO, capture_output=True, text=True).stdout.strip()
    base = _run_child(_export_base_tree(args.rev), inputs_file, work / "base.json")
    tip = _run_child(REPO, inputs_file, work / "tip.json")

    per_input: Dict[str, Any] = {}
    preservation_failures: List[str] = []
    for name, payload in inputs.items():
        b, t = base["inputs"][name], tip["inputs"][name]
        protein_rows = _protein_endpoint_rows(payload)
        keys = ("stage6_payload_sha256", "stage6_report_sha256", "ir_without_withheld_evidence_sha256",
                "report_sha256", "pwml_sha256", "stop", "pwml_stop")
        identical = {k: b["replay"].get(k) == t["replay"].get(k) for k in keys}
        entry = {
            "interaction_rows": len((payload.get("processes") or {}).get("interactions") or []),
            "protein_endpoint_rows": protein_rows,
            "direct": {"base": b["direct"], "tip": t["direct"]},
            "replay": {"base": b["replay"].get("outcomes", b["replay"].get("stop")),
                       "tip": t["replay"].get("outcomes", t["replay"].get("stop")),
                       "tip_stage6_summary_interaction_keys": {k: v for k, v in (t["replay"].get("stage6_summary") or {}).items()
                                                               if k.startswith("interaction_")},
                       "pwml_interactions": [b["replay"].get("pwml_interactions"), t["replay"].get("pwml_interactions")],
                       "pwml_built": [bool(b["replay"].get("pwml_sha256")), bool(t["replay"].get("pwml_sha256"))],
                       "pwml_stop": [b["replay"].get("pwml_stop"), t["replay"].get("pwml_stop")],
                       "tip_ir_error_codes_added": sorted(set(t["replay"].get("ir_error_codes") or [])
                                                          - set(b["replay"].get("ir_error_codes") or []))},
            "byte_identical": identical,
            "confinement_violations": _confinement(b["replay"], t["replay"]),
            "contract_error_codes_added": sorted(set(t["replay"]["contract_error_codes"])
                                                 - set(b["replay"]["contract_error_codes"])),
        }
        if protein_rows == 0 and not name.startswith("correction:") and not all(identical.values()):
            preservation_failures.append(name)
        per_input[name] = entry

    def collect(group: List[str], config: str, side: str) -> Dict[str, int]:
        return _sum([per_input[n][config][side] for n in group if isinstance(per_input[n][config][side], dict)
                     and "stop" not in per_input[n][config][side]])

    corpus = [n for n in per_input if not n.startswith(("fixture:", "correction:"))]
    no_protein = [n for n in per_input if per_input[n]["protein_endpoint_rows"] == 0 and not n.startswith("correction:")]
    error_added = {n: e["replay"]["tip_ir_error_codes_added"] for n, e in per_input.items() if e["replay"]["tip_ir_error_codes_added"]}
    contract_added = {n: e["contract_error_codes_added"] for n, e in per_input.items() if e["contract_error_codes_added"]}
    unconfined = {n: e["confinement_violations"] for n, e in per_input.items() if e["confinement_violations"]}
    summary = {
        "base_rev": args.rev,
        "base_sha": sha,
        "base_t2pw_file": base["t2pw_file"],
        "tip_t2pw_file": tip["t2pw_file"],
        "inputs": len(inputs),
        "corpus_legs": len(corpus),
        "corpus_interaction_rows": sum(per_input[n]["interaction_rows"] for n in corpus),
        "preservation": {
            "inputs_with_no_protein_endpoint": len(no_protein),
            "of_which_pwml_built_on_both": sum(1 for n in no_protein if all(per_input[n]["replay"]["pwml_built"])),
            "names": no_protein,
            "byte_identical_failures": preservation_failures,
            "verdict": "PRESERVED" if not preservation_failures else "NOT PRESERVED",
        },
        "corpus_direct": {
            "legs_built": sum(1 for n in corpus if "stop" not in per_input[n]["direct"]["tip"]),
            "base": collect(corpus, "direct", "base"),
            "tip": collect(corpus, "direct", "tip"),
        },
        "corpus_replay": {
            "legs_built": sum(1 for n in corpus if isinstance(per_input[n]["replay"]["tip"], dict)),
            "base": collect(corpus, "replay", "base"),
            "tip": collect(corpus, "replay", "tip"),
            "pwml_interactions": [sum((per_input[n]["replay"]["pwml_interactions"][i] or 0) for n in corpus) for i in (0, 1)],
        },
        "ir_error_codes_added_on_tip": error_added,
        "contract_error_codes_added_on_tip": contract_added,
        "confinement": {"inputs_checked": len(per_input), "violations": unconfined,
                        "verdict": "CONFINED" if not unconfined else "NOT CONFINED"},
        "per_input": per_input,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")  # per-input carries hashes, not payloads
    lines = [
        f"ORCH-754b wrapper proof  base {args.rev} = {sha}",
        f"base t2pw: {base['t2pw_file']}",
        f"tip  t2pw: {tip['t2pw_file']}",
        f"inputs: {len(inputs)} ({len(corpus)} committed legs with interactions, {summary['corpus_interaction_rows']} rows; + fixtures)",
        "",
        f"PRESERVATION on {len(no_protein)} inputs with no protein endpoint: "
        f"{summary['preservation']['verdict']}  failures={preservation_failures}  "
        f"(PWML bytes compared on {summary['preservation']['of_which_pwml_built_on_both']})",
        "",
        f"corpus, direct (ORCH-749 configuration), {summary['corpus_direct']['legs_built']} legs built:",
        f"  base {summary['corpus_direct']['base']}",
        f"  tip  {summary['corpus_direct']['tip']}",
        f"corpus, offline Stage-6 replay + prefreeze + IR, {summary['corpus_replay']['legs_built']} legs built:",
        f"  base {summary['corpus_replay']['base']}",
        f"  tip  {summary['corpus_replay']['tip']}",
        f"  <interaction> elements in the PWML, base/tip: {summary['corpus_replay']['pwml_interactions']}",
        f"IR error codes added on tip: {error_added or 'none'}",
        f"contract (validate_required_pwml_contract) error codes added on tip: {contract_added or 'none'}",
        f"CONFINEMENT on all {len(per_input)} inputs (reactions, transports, proteins, pre-existing complexes, "
        f"IR reactions/entities untouched): {summary['confinement']['verdict']}  {unconfined or ''}",
        "",
        "fixtures:",
    ]
    for name in ("fixture:enzymes_only", "fixture:enzyme_reaction_no_protein_endpoint", "fixture:protein_endpoints",
                 "correction:competes_with"):
        e = per_input[name]
        lines.append(f"  {name}: replay base {e['replay']['base']} -> tip {e['replay']['tip']}; "
                     f"PWML <interaction> {e['replay']['pwml_interactions'][0]} -> {e['replay']['pwml_interactions'][1]}; "
                     f"byte_identical {all(e['byte_identical'].values())}")
    text = "\n".join(lines) + "\n"
    Path(args.out).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {args.out}")
    c = per_input["correction:competes_with"]["replay"]
    correction_holds = c["base"] == {"exportable": 1} and c["tip"] == {"untyped_interaction": 1}
    print(f"G9 correction, competes_with: base {c['base']} -> tip {c['tip']}: "
          + ("base FAILS (draws it), tip withholds" if correction_holds else "NOT DEMONSTRATED"))
    summary["g9_correction_competes_with"] = {"base": c["base"], "tip": c["tip"], "holds": correction_holds}
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    ok = not preservation_failures and not error_added and not contract_added and not unconfined and correction_holds
    print("VERDICT: " + ("preservation holds, delta confined, no IR or contract error added" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
