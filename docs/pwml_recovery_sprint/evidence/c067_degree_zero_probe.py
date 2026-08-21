"""C-067 G9 probe: does the degree-zero detector resolve names the way the pruner does?

Run at the base SHA and again at the tip; the two JSON reports are the behavioural
delta. **Symbol absence is not proof** -- nothing here reads the source of
``_degree_zero_exports``. Every arm calls the shipped function and records the
rows it returned.

The fixture is deliberately, obviously synthetic. The seam under test is NAME
RESOLUTION, not chemistry: ``Enzyme X`` with synonym ``ExA`` catalysing ``A -> B``
asserts nothing about any organism, and therefore cannot later be replayed or
quoted as a pathway claim. No accession is minted anywhere in this file -- the
direct-drive arms never reach an identity check, and the one integration arm
passes ``strict_db=False``, which ``strict_quarantine.py:697-698`` short-circuits
on before any identity test. (It does NOT short-circuit before
``placeholder_claims_real_identity`` at ``:692``, so a forged accession would
still be caught here; there simply is not one.)

Four arms plus one integration observation:

``divergence``
    The protein's primary name is unreferenced; its SYNONYM is what the reaction's
    ``enzymes[].entity`` says. The pruner keeps the row. At base the detector
    flags it -- a false positive about a row that IS referenced. At tip: no rows.

``control_true_orphan``
    Same graph, synonym removed. The pruner DELETES the row, so afterwards the
    detector has nothing to flag. Non-vacuity for the closure leg: this arm also
    records the detector's answer BEFORE the prune, where it must still flag.

``unreferenced_synonym``
    A protein carrying a synonym that nothing references. Non-vacuity for the NEW
    code path specifically: widening the test to name-union-synonyms must not
    blanket-exempt every row that merely HAS a synonym.

``complex_component_exempt``
    A protein complex referenced only under its synonym, with a component protein.
    Exercises the ``exempt`` construction rather than the row loop.

``integration``
    ``quarantine_and_close`` over the ``divergence`` graph, recording ``ok``,
    ``strict_gates_passed`` (captured at the ``classify_release_status`` call
    boundary, not inferred) and ``release.status`` together, so a report where
    they disagree is visible rather than assumed impossible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

CLOSURE_ITERATIONS = 50

#: Bound by :func:`_bind_source` before any arm runs. The module is NOT imported
#: at import time, so ``--src`` decides which bytes of ``strict_quarantine.py``
#: the measurement runs against -- the base tree or the branch tree -- and one
#: committed script produces both halves of the G9 delta.
SQ: Any = None


def _bind_source(src: str) -> str:
    """Import ``t2pw`` from ``src`` and prove that is where it came from."""

    global SQ
    src = os.path.abspath(src)
    sys.path.insert(0, src)
    from t2pw.pipeline import strict_quarantine as module

    loaded_from = os.path.abspath(module.__file__)
    if not loaded_from.startswith(src):
        raise SystemExit(f"refusing to measure: t2pw loaded from {loaded_from}, not {src}")
    SQ = module
    return loaded_from


# ── Fixtures ────────────────────────────────────────────────────────────────


def _accepted(index: int = 0) -> Dict[str, Any]:
    return {"state": SQ.CORE_ACCEPTED, "bucket": "reactions", "index": index}


def _shell() -> Dict[str, Any]:
    """An empty graph with every bucket the walkers expect."""

    return {
        "entities": {
            "species": [],
            "subcellular_locations": [],
            "compounds": [],
            "proteins": [],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "biological_states": [],
        "element_locations": {},
        "processes": {"reactions": [], "transports": [], "interactions": []},
    }


def _one_reaction(enzyme_reference: str) -> Dict[str, Any]:
    return {
        "name": "A to B",
        "inputs": ["A"],
        "outputs": ["B"],
        "enzymes": [{"entity": enzyme_reference}],
    }


def divergence_payload() -> Dict[str, Any]:
    """The reaction names the protein by its SYNONYM, never by its name."""

    payload = _shell()
    payload["entities"]["compounds"] = [{"name": "A"}, {"name": "B"}]
    payload["entities"]["proteins"] = [{"name": "Enzyme X", "synonyms": ["ExA"]}]
    payload["processes"]["reactions"] = [_one_reaction("ExA")]
    return payload


def control_payload() -> Dict[str, Any]:
    """The same graph with the synonym removed: a genuine orphan."""

    payload = divergence_payload()
    payload["entities"]["proteins"] = [{"name": "Enzyme X"}]
    return payload


def unreferenced_synonym_payload() -> Dict[str, Any]:
    """A synonym-carrying protein that nothing reaches, under either name."""

    payload = _shell()
    payload["entities"]["compounds"] = [{"name": "A"}, {"name": "B"}]
    payload["entities"]["proteins"] = [
        {"name": "Enzyme X", "synonyms": ["ExA"]},
        {"name": "Enzyme Y", "synonyms": ["EyB"]},
    ]
    payload["processes"]["reactions"] = [_one_reaction("ExA")]
    return payload


def complex_component_payload() -> Dict[str, Any]:
    """A complex referenced only by synonym, plus the component it carries."""

    payload = _shell()
    payload["entities"]["compounds"] = [{"name": "A"}, {"name": "B"}]
    payload["entities"]["proteins"] = [{"name": "Sub One"}]
    payload["entities"]["protein_complexes"] = [
        {"name": "Complex C", "synonyms": ["CxC"], "components": ["Sub One"]}
    ]
    payload["processes"]["reactions"] = [_one_reaction("CxC")]
    return payload


# ── Drive ───────────────────────────────────────────────────────────────────


def prune_to_fixpoint(
    payload: Mapping[str, Any], admissions: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    """The closure loop's ENTITY leg, reproduced from the shipped helpers.

    ``strict_quarantine.py:2080-2088`` -- same ``referenced``, same
    ``surviving_complex_norms``, same ``keep_norms``, same ``_prune_entities``.
    Location, biological-state and process revalidation legs are omitted: this
    fixture has no locations or states, and no reaction here can be revalidated
    away, so the entity leg alone reaches the same fixpoint. Nothing is copied
    from the module -- every step calls it.
    """

    working = deepcopy(dict(payload))
    removed: List[Dict[str, Any]] = []
    converged = False
    for _ in range(CLOSURE_ITERATIONS):
        referenced = SQ._referenced_entity_norms(working, admissions)
        surviving_complex_norms = {
            SQ._normalize(SQ._row_name(row))
            for row in SQ._safe_list(
                SQ._safe_dict(working.get("entities")).get("protein_complexes")
            )
            if isinstance(row, dict) and SQ._normalize(SQ._row_name(row)) in referenced
        }
        keep_norms = referenced | SQ._complex_component_norms(working, surviving_complex_norms)
        step = SQ._prune_entities(working, keep_norms)
        removed.extend(step)
        if not step:
            converged = True
            break
    return working, removed, converged


def _detect(payload: Mapping[str, Any], admissions: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    return SQ._degree_zero_exports(
        payload, admissions, process_snapshot=SQ._safe_dict(payload.get("processes"))
    )


def _names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(str(row.get("name")) for row in rows)


def _surviving_protein_names(payload: Mapping[str, Any]) -> List[str]:
    entities = SQ._safe_dict(payload.get("entities"))
    out: List[str] = []
    for bucket in ("proteins", "protein_complexes"):
        for row in SQ._safe_list(entities.get(bucket)):
            if isinstance(row, dict) and SQ._row_name(row):
                out.append(SQ._row_name(row))
    return sorted(out)


def _arm(payload: Dict[str, Any]) -> Dict[str, Any]:
    admissions = [_accepted(0)]
    before = _detect(payload, admissions)
    pruned, removed, converged = prune_to_fixpoint(payload, admissions)
    after = _detect(pruned, admissions)
    return {
        "detector_before_prune": _names(before),
        "closure_converged": converged,
        "pruner_removed": [
            {"bucket": row.get("bucket"), "name": row.get("name"), "reason": row.get("reason")}
            for row in removed
        ],
        "surviving_protein_rows": _surviving_protein_names(pruned),
        "detector_after_prune": _names(after),
    }


# ── The integration observation (section 4 of the card) ─────────────────────


def integration_arm() -> Dict[str, Any]:
    """``ok`` and ``strict_gates_passed``, captured together at one call boundary."""

    from t2pw.pipeline import release_status as RS

    captured: Dict[str, Any] = {}
    shipped = RS.classify_release_status

    def _spy(*args: Any, **kwargs: Any) -> Any:
        captured["strict_gates_passed"] = bool(kwargs.get("strict_gates_passed"))
        captured["serializable_without_invention"] = bool(
            kwargs.get("serializable_without_invention")
        )
        return shipped(*args, **kwargs)

    RS.classify_release_status = _spy  # type: ignore[assignment]
    try:
        result = SQ.quarantine_and_close(divergence_payload(), strict_db=False)
    finally:
        RS.classify_release_status = shipped  # type: ignore[assignment]

    invariants = result.quarantine_report.get("strict_invariants") or {}
    release = result.quarantine_report.get("release") or {}
    return {
        "ok": bool(result.ok),
        "strict_gates_passed": captured.get("strict_gates_passed"),
        "serializable_without_invention": captured.get("serializable_without_invention"),
        "degree_zero_exports": _names(invariants.get("degree_zero_exports") or []),
        "closure_converged": bool(invariants.get("closure_converged")),
        "refusal_reasons": list(result.refusal_reasons),
        "review_reasons": list(getattr(result, "review_reasons", []) or []),
        "release_status": release.get("status"),
        "strict_acceptance_eligible": release.get("strict_acceptance_eligible"),
        "surviving_protein_rows": _surviving_protein_names(result.payload),
        "ok_and_gates_agree": bool(result.ok) == bool(captured.get("strict_gates_passed")),
    }


def measure(loaded_from: str) -> Dict[str, Any]:
    return {
        "probe": "c067_degree_zero_probe",
        "module": loaded_from.replace("\\", "/"),
        "arms": {
            "divergence": _arm(divergence_payload()),
            "control_true_orphan": _arm(control_payload()),
            "unreferenced_synonym": _arm(unreferenced_synonym_payload()),
            "complex_component_exempt": _arm(complex_component_payload()),
        },
        "integration": integration_arm(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--src",
        default=os.path.join(ROOT, "src"),
        help="source tree to import t2pw from; point it at a base checkout for the base arm",
    )
    args = parser.parse_args(argv)

    loaded_from = _bind_source(args.src)
    report = measure(loaded_from)
    report["label"] = args.label
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)

    divergence = report["arms"]["divergence"]
    print()
    print("divergence arm: pruner kept", divergence["surviving_protein_rows"])
    print("divergence arm: detector after prune ->", divergence["detector_after_prune"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
