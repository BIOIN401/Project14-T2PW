"""Research mode in Stage-3 normalization: keep the content, flag the problem.

Three classes of strict-mode behaviour are covered, because a "make the gates
non-blocking" change misses all three:

1. **Uncaught ``ValueError``s.** Composite materialization raises from inside
   pass 3 of 18, long before any gate report exists, so strict mode loses the
   entire run over a name PathWhiz cannot model.
2. **A hidden ``assert``.** ``validate_post_normalization`` never raises, so the
   ``assert actor_contract.get("ok") is True`` at the end of
   ``normalize_process_payload`` is what actually aborts on a canonical
   actor-row failure.
3. **Destructive passes that never raise at all.** They delete proteins and
   catalysts to pre-empt a downstream gate, so relaxing the gate alone leaves
   the content already gone.

Every research-mode assertion has a strict-mode twin, because the point is the
difference between them -- and the strict twin is the regression firewall.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.export_mode import PATHWHIZ, RESEARCH  # noqa: E402
from t2pw.pipeline.process_normalizer import normalize_process_payload  # noqa: E402


def _entity_names(payload: dict, bucket: str) -> set:
    return {
        str(row.get("name"))
        for row in payload.get("entities", {}).get(bucket, [])
        if isinstance(row, dict)
    }


def _relaxations(report: dict) -> set:
    return {
        str(action.get("type"))
        for action in report.get("actions", [])
        if isinstance(action, dict) and str(action.get("type", "")).startswith("research_mode_")
    }


def _base_payload() -> dict:
    return {
        "entities": {
            "species": [{"name": "Homo sapiens"}],
            "compounds": [{"name": "ATP"}, {"name": "ADP"}],
            "proteins": [{"name": "Wired kinase", "species": "Homo sapiens"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "Step",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "enzymes": [{"entity": "Wired kinase", "entity_type": "protein"}],
                }
            ]
        },
    }


# --------------------------------------------------------------------------
# 1. The uncaught ValueError that kills the whole run.
# --------------------------------------------------------------------------


def _composite_payload() -> dict:
    payload = _base_payload()
    # "glucose+fructose" has no protein-like left component, so it has no
    # PathWhiz protein_complex representation.
    payload["entities"]["compounds"].append({"name": "glucose+fructose"})
    return payload


def test_composite_compound_aborts_the_whole_run_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="no protein-like left component"):
        normalize_process_payload(_composite_payload(), mode=PATHWHIZ)


def test_composite_compound_is_kept_and_flagged_in_research_mode() -> None:
    data, report = normalize_process_payload(_composite_payload(), mode=RESEARCH)

    assert "glucose+fructose" in _entity_names(data, "compounds")
    assert "research_mode_composite_compound_kept" in _relaxations(report)


def test_composite_relaxation_is_counted_so_it_is_never_silent() -> None:
    _, report = normalize_process_payload(_composite_payload(), mode=RESEARCH)

    assert report["summary"]["research_mode_relaxations"] >= 1


# --------------------------------------------------------------------------
# 2. Destructive passes: proteins a paper stated but that are not yet wired.
# --------------------------------------------------------------------------


def _orphan_payload() -> dict:
    payload = _base_payload()
    payload["entities"]["proteins"].append(
        {"name": "Orphan factor Y", "species": "Homo sapiens"}
    )
    return payload


def test_strict_mode_deletes_the_unwired_protein() -> None:
    data, _ = normalize_process_payload(_orphan_payload(), mode=PATHWHIZ)

    assert "Orphan factor Y" not in _entity_names(data, "proteins")


def test_research_mode_keeps_the_unwired_protein_for_review() -> None:
    data, report = normalize_process_payload(_orphan_payload(), mode=RESEARCH)

    assert "Orphan factor Y" in _entity_names(data, "proteins")
    assert "research_mode_destructive_cleanup_skipped" in _relaxations(report)


# --------------------------------------------------------------------------
# 3. Non-protein catalysts: a real claim PathWhiz cannot represent.
# --------------------------------------------------------------------------


def _cofactor_payload() -> dict:
    payload = _base_payload()
    payload["entities"]["compounds"].append({"name": "Mg2+"})
    payload["processes"]["reactions"][0]["enzymes"].append(
        {"entity": "Mg2+", "entity_type": "compound", "role": "catalyst"}
    )
    return payload


def test_strict_mode_drops_the_non_protein_catalyst() -> None:
    _, report = normalize_process_payload(_cofactor_payload(), mode=PATHWHIZ)

    dropped = {
        str(action.get("type"))
        for action in report["actions"]
        if isinstance(action, dict)
    }
    assert "non_protein_catalyst_dropped" in dropped


def test_research_mode_keeps_the_non_protein_catalyst_and_flags_it() -> None:
    _, report = normalize_process_payload(_cofactor_payload(), mode=RESEARCH)

    assert "research_mode_non_protein_catalyst_kept" in _relaxations(report)
    # The strict drop must not also have happened.
    assert "non_protein_catalyst_dropped" not in {
        str(action.get("type")) for action in report["actions"] if isinstance(action, dict)
    }


# --------------------------------------------------------------------------
# Defaults and non-mutation.
# --------------------------------------------------------------------------


def test_default_mode_is_strict() -> None:
    with pytest.raises(ValueError):
        normalize_process_payload(_composite_payload())


def test_strict_report_carries_no_research_bookkeeping() -> None:
    _, report = normalize_process_payload(_orphan_payload(), mode=PATHWHIZ)

    assert "export_mode" not in report
    assert _relaxations(report) == set()
    assert "research_mode_relaxations" not in report["summary"]


@pytest.mark.parametrize("mode", [PATHWHIZ, RESEARCH])
def test_input_payload_is_never_mutated(mode) -> None:
    payload = _orphan_payload()
    before = deepcopy(payload)

    normalize_process_payload(payload, mode=mode)

    assert payload == before


# --------------------------------------------------------------------------
# The headline claim, end to end.
# --------------------------------------------------------------------------


def _everything_wrong_payload() -> dict:
    """One payload that trips a format rule, a fatal rewrite, and a biology rule."""

    payload = _base_payload()
    payload["entities"]["compounds"] += [{"name": "glucose+fructose"}, {"name": "Mg2+"}]
    payload["entities"]["proteins"] = [
        {"name": "IL-6:IL-6R signalling protein", "species": "Homo sapiens"},
        {"name": "Orphan factor Y", "species": "Homo sapiens"},
    ]
    payload["processes"]["reactions"][0]["enzymes"] = [
        {"entity": "IL-6:IL-6R signalling protein", "entity_type": "protein"},
        {"entity": "Mg2+", "entity_type": "compound", "role": "catalyst"},
    ]
    payload["processes"]["reactions"].append(
        {"name": "Empty claim", "inputs": [], "outputs": []}
    )
    return payload


def test_strict_mode_cannot_produce_this_pathway_at_all() -> None:
    with pytest.raises(ValueError):
        normalize_process_payload(_everything_wrong_payload(), mode=PATHWHIZ)


def test_research_mode_always_produces_the_pathway_and_flags_the_biology() -> None:
    from t2pw.pipeline.export_mode import review_flags
    from t2pw.pipeline.stage_contracts import run_stage_contract, validate_post_extraction

    data, report = normalize_process_payload(_everything_wrong_payload(), mode=RESEARCH)
    contract = run_stage_contract(validate_post_extraction, data, mode=RESEARCH)

    # A pathway exists, with the ':' name, the unwired protein and the cofactor.
    assert "IL-6:IL-6R signalling protein" in _entity_names(data, "proteins")
    assert "Orphan factor Y" in _entity_names(data, "proteins")
    assert {"Step", "Empty claim"} <= {
        str(r.get("name")) for r in data["processes"]["reactions"]
    }

    # ...and the biology problem is loud, not swallowed.
    assert contract["ok"] is True
    assert "reaction_missing_participants" in {f["code"] for f in review_flags(contract)}
    assert contract["research_review_required"] is True
    assert _relaxations(report) >= {
        "research_mode_composite_compound_kept",
        "research_mode_destructive_cleanup_skipped",
        "research_mode_non_protein_catalyst_kept",
    }
