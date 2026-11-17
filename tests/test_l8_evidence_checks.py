"""L8 (D-W24 item 6): every reaction, enzyme assignment and estimated compartment is checked against the paper.

A failing check flags the element (``review_required``, a ``review_reasons`` entry naming the check, the evidence read);
nothing is removed unless the paper contradicts it. Fixtures are shaped on the A4 assessment cases (idx 126, 144, 116, 138).
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List

PAPER = (
    "Benzyl alcohol biosynthesis in Petunia hybrida. "
    "PhBR1 and PhBR2 catalyze the NADPH-dependent reduction of benzaldehyde to benzyl alcohol. "
    "Recombinant PhCAD1 also reduced benzaldehyde in vitro. "
    "Benzyl benzoate is first hydroxylated by benzyl benzoate oxidase (BBO) to form benzyl salicylate. "
    "It is very interesting to ask why the enzymes for the sequential transformation of 3 to 4, and to 5 are missing. "
    "Dihydrosphingosine is N-acylated by the ceramide synthases to dihydroceramide. "
    "The reaction takes place in the cytosol of petal cells."
)
CONTEXT = {"pathway_name": "benzyl alcohol biosynthesis", "organism": "Petunia hybrida"}


def _rx(name: str, inputs: List[str], outputs: List[str], evidence: str, enzymes: List[Dict[str, Any]] = ()) -> Dict[str, Any]:
    return {"name": name, "inputs": list(inputs), "outputs": list(outputs), "evidence": evidence,
            "enzymes": [deepcopy(e) for e in enzymes], "biological_state": "cytosol"}


def _enz(name: str, evidence: str) -> Dict[str, Any]:
    return {"entity": name, "entity_type": "protein_complex", "role": "catalyst", "evidence": evidence}


def _payload(*reactions: Dict[str, Any], transports: List[Dict[str, Any]] = ()) -> Dict[str, Any]:
    names = {n for r in reactions for n in r["inputs"] + r["outputs"]}
    return {
        "entities": {"compounds": [{"name": n} for n in sorted(names)],
                     "protein_complexes": [{"name": a["entity"]} for r in reactions for a in r["enzymes"]]},
        "biological_states": [{"name": "cytosol"}],
        "processes": {"reactions": list(reactions), "transports": list(transports), "interactions": []},
    }


def _checks(row: Dict[str, Any]) -> List[str]:
    return [c["check"] for c in row.get("evidence_checks") or []]


def _run(payload: Dict[str, Any], paper: str = PAPER, **kw: Any) -> Dict[str, Any]:
    from t2pw.service.final_payload_evidence import check_evidence

    return check_evidence(payload, paper, CONTEXT, **kw)


BR = _rx("PhBR reduction", ["benzaldehyde", "NADPH"], ["benzyl alcohol", "NADP+"],
         "PhBR1 and PhBR2 catalyze the NADPH-dependent reduction of benzaldehyde",
         [_enz("PhBR1", "PhBR1 and PhBR2 catalyze the NADPH-dependent reduction of benzaldehyde")])


def test_l8_a_supported_step_is_not_flagged() -> None:
    payload = _payload(deepcopy(BR))
    report = _run(payload)
    row = payload["processes"]["reactions"][0]
    assert _checks(row) == [] and "review_required" not in row and "review_reasons" not in row
    assert report["flagged"] == [] and report["removed"] == []


def test_l8_an_invented_step_whose_evidence_is_the_pathway_name_is_flagged_not_removed() -> None:
    """idx 126 R2: benzaldehyde -> salicylic acid, evidence 'benzyl alcohol biosynthesis in Petunia hybrida'."""
    invented = _rx("PhBBO benzaldehyde hydroxylase reaction", ["benzaldehyde", "O2"], ["salicylic acid"],
                   "benzyl alcohol biosynthesis in Petunia hybrida")
    payload = _payload(deepcopy(BR), invented)
    report = _run(payload)
    rows = payload["processes"]["reactions"]
    assert len(rows) == 2 and "scope_membership" not in rows[1]
    assert _checks(rows[1]) == ["evidence_is_pathway_name", "evidence_names_no_participant"]
    assert rows[1]["review_required"] is True
    assert all(r.endswith("(D-W24 6, L8)") for r in rows[1]["review_reasons"])
    assert rows[1]["evidence_checks"][0]["evidence"] == "benzyl alcohol biosynthesis in Petunia hybrida"
    assert report["counts"] == {"evidence_is_pathway_name": 1, "evidence_names_no_participant": 1}
    assert report["removed"] == []


def test_l8_evidence_that_is_the_article_header_is_flagged() -> None:
    """idx 144 R5: the evidence is the article metadata block."""
    header = _rx("isoflavone backbone formation", ["liquiritigenin"], ["isoflavone backbone"],
                 "4803 psa Pharm Sci Adv Elsevier PMC13084691 10.1016/j.pscia.2026.100118 Functional identification")
    payload = _payload(header)
    _run(payload)
    checks = _checks(payload["processes"]["reactions"][0])
    assert "evidence_is_article_header" in checks and "evidence_names_no_participant" in checks
    assert "evidence_not_in_paper" in checks


def test_l8_a_step_that_skips_an_intermediate_the_payload_has_is_flagged() -> None:
    """idx 116 R2: sphinganine -> ceramide beside sphinganine -> dihydroceramide -> ceramide."""
    a = _rx("ceramide synthase reaction", ["Sphinganine"], ["dihydroceramide"],
            "Dihydrosphingosine is N-acylated by the ceramide synthases to dihydroceramide")
    b = _rx("desaturase reaction", ["dihydroceramide"], ["ceramide"], "desaturation of dihydroceramide produces ceramide")
    merged = _rx("Ceramide synthase reaction", ["Sphinganine"], ["ceramide"],
                 "Dihydrosphingosine is N-acylated by the ceramide synthases")
    payload = _payload(a, b, merged)
    _run(payload)
    rows = payload["processes"]["reactions"]
    assert "step_skips_intermediate" not in _checks(rows[0]) + _checks(rows[1])
    (rec,) = [c for c in rows[2]["evidence_checks"] if c["check"] == "step_skips_intermediate"]
    assert rec["intermediate"] == "dihydroceramide"
    assert rec["via"] == ["ceramide synthase reaction", "desaturase reaction"]


def test_l8_evidence_that_describes_several_steps_is_flagged() -> None:
    """idx 138 R7: compound 3 -> compound 5, 'the sequential transformation of 3 to 4, and to 5'."""
    row = _rx("", ["compound 3"], ["compound 5"],
              "It is very interesting to ask why the enzymes for the sequential transformation of 3 to 4, and to 5 are missing")
    payload = _payload(row)
    _run(payload)
    assert "evidence_describes_several_steps" in _checks(payload["processes"]["reactions"][0])


def test_l8_an_enzyme_not_tied_to_its_step_is_flagged_unless_a_paper_sentence_ties_it() -> None:
    """idx 126 R1: PhBSalpha's evidence is the pathway name; PhCAD1's evidence names no step but the paper ties it."""
    row = _rx("Benzaldehyde reductase reaction", ["benzaldehyde", "NADPH"], ["benzyl alcohol", "NADP+"],
              "PhBR1 and PhBR2 catalyze the NADPH-dependent reduction of benzaldehyde",
              [_enz("PhBR1", "PhBR1 and PhBR2 catalyze the NADPH-dependent reduction of benzaldehyde"),
               _enz("PhCAD1", "purified MBP-tagged recombinant PhBR1, PhCAD1"),
               _enz("PhBSα", "benzyl alcohol biosynthesis in Petunia hybrida")])
    payload = _payload(row)
    report = _run(payload)
    (rec,) = payload["processes"]["reactions"][0]["evidence_checks"]
    assert (rec["check"], rec["kind"], rec["element"]) == ("enzyme_not_tied_to_step", "enzyme", "PhBSα")
    assert [t["enzyme"] for t in report["tied_by_paper_sentence"]] == ["PhCAD1"]
    assert payload["processes"]["reactions"][0]["enzymes"][2]["entity"] == "PhBSα"  # flagged, not removed


def test_l8_estimated_compartments_must_carry_evidence() -> None:
    llm_ok = dict(_rx("a", ["benzaldehyde"], ["benzyl alcohol"], "reduction of benzaldehyde to benzyl alcohol"),
                  compartment_source="llm_estimate",
                  compartment_estimate=[{"source": "llm_estimate", "reason": "estimated from the paper",
                                         "quote": "takes place in the cytosol of petal cells"}])
    llm_bad = dict(deepcopy(llm_ok), name="b", compartment_estimate=[
        {"source": "llm_estimate", "reason": "estimated", "quote": "occurs in the peroxisome"}])
    neighbour_bad = dict(deepcopy(llm_ok), name="c", compartment_source="neighbour_estimate",
                         compartment_estimate=[{"source": "neighbour_estimate", "reason": "neighbours", "neighbours": []}])
    merged_ok = dict(deepcopy(llm_ok), name="d", compartment_source="estimated",
                     compartment_estimate=[{"field": "biological_state", "reason": "placeholder merged (D-W24 5, L6)"}])
    no_reason = dict(deepcopy(llm_ok), name="e", compartment_source="estimated", compartment_estimate=[])
    unresolved = dict(deepcopy(llm_ok), name="f", compartment_source="unresolved", compartment_estimate=[])
    transport = {"name": "t", "compartment_source": "llm_estimate",
                 "compartment_estimate": [{"source": "llm_estimate", "reason": "x", "quote": "not a quote of the paper"}]}
    payload = _payload(llm_ok, llm_bad, neighbour_bad, merged_ok, no_reason, unresolved, transports=[transport])
    _run(payload)
    flagged = {r["name"]: _checks(r) for r in payload["processes"]["reactions"]}
    assert flagged == {"a": [], "b": ["estimated_compartment_without_evidence"],
                       "c": ["estimated_compartment_without_evidence"], "d": [],
                       "e": ["estimated_compartment_without_evidence"], "f": []}
    assert _checks(payload["processes"]["transports"][0]) == ["estimated_compartment_without_evidence"]


def test_l8_only_a_contradiction_quoted_from_the_paper_removes_an_element() -> None:
    invented = _rx("PhBBO benzaldehyde hydroxylase reaction", ["benzaldehyde"], ["salicylic acid"],
                   "benzyl alcohol biosynthesis in Petunia hybrida",
                   [_enz("PhBBO", "benzyl alcohol biosynthesis in Petunia hybrida")])
    # A judge whose quote is not in the paper removes nothing.
    payload = _payload(deepcopy(invented))
    report = _run(payload, judge=lambda rec, row, text: {"contradicted": True, "quote": "BBO never touches benzaldehyde"})
    assert report["removed"] == [] and len(report["judge_rejected"]) == 3
    assert "scope_membership" not in payload["processes"]["reactions"][0]
    # A quote that is in the paper: the reaction goes to the quarantine as out_of_scope, the enzyme to the withheld record.
    quote = "Benzyl benzoate is first hydroxylated by benzyl benzoate oxidase (BBO) to form benzyl salicylate"
    payload = _payload(deepcopy(invented))
    report = _run(payload, judge=lambda rec, row, text: {"contradicted": True, "quote": quote})
    row = payload["processes"]["reactions"][0]
    assert row["scope_membership"]["reason"] == "contradicted_by_paper" and row["scope_membership"]["quote"] == quote
    assert row["enzymes"] == []
    assert payload["withheld_reaction_actors"][0]["quote"] == quote
    assert {r["kind"] for r in report["removed"]} == {"reaction", "enzyme"}


def test_l8_a_second_call_without_the_paper_keeps_the_first_result() -> None:
    from t2pw.service.final_payload_evidence import check_evidence

    invented = _rx("x", ["benzaldehyde"], ["salicylic acid"], "benzyl alcohol biosynthesis in Petunia hybrida")
    payload = _payload(invented)
    check_evidence(payload, PAPER, CONTEXT)
    first = deepcopy(payload)
    assert check_evidence(payload, None, CONTEXT) == {"skipped": "already checked with the paper text"}
    assert payload == first
    check_evidence(payload, PAPER, CONTEXT)  # a re-run with the paper replaces, never duplicates
    assert payload == first


def test_l8_the_final_payload_seam_runs_the_checks_with_the_flag_on(monkeypatch) -> None:
    from t2pw.service import final_payload as fp

    class _NoNet:
        def get_json(self, *a, **k):
            raise RuntimeError("offline")

        get_text = get_json

    for step in ("apply_final_id_fill", "maybe_apply_paper_nucleic_types", "maybe_apply_final_pathway_meta"):
        monkeypatch.setattr(fp, step, lambda *a, **k: {})
    invented = _rx("x", ["benzaldehyde"], ["salicylic acid"], "benzyl alcohol biosynthesis in Petunia hybrida")
    payload = _payload(invented)
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "0")
    assert fp.maybe_apply_final_id_fill(deepcopy(payload), {}, {"final_payload_paper_text": PAPER}) == {}
    monkeypatch.setenv("T2PW_FINAL_PAYLOAD", "1")
    report = fp.maybe_apply_final_id_fill(payload, {"pathway_name": CONTEXT["pathway_name"]},
                                          {"final_payload_paper_text": PAPER}, fetcher=_NoNet())
    assert report["evidence"]["paper_text"] is True
    assert "evidence_names_no_participant" in _checks(payload["processes"]["reactions"][0])
    assert payload["final_payload_evidence"]["flagged_elements"] >= 1
