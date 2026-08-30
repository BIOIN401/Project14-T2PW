"""C-105 -- an audit patch may not invent an actor role it has no evidence for.

The defect these tests pin is measured, not hypothetical:
``runs_verify/2026-08-28_1816/papers/PMC13231680/research/final_mapped.json``
``/processes/reactions/0`` was frozen naming one protein as both the reaction's
enzyme and its inhibitor, and the audit round that wrote it recorded why -- to
"resolve the structural inconsistency where an inhibitor is listed without a
target enzyme". PRODUCT_CONTRACT section 1 forbids inventing an enzyme, and a
schema complaint is not evidence.

**Every test here drives the public seam ``apply_patch_with_policy`` and asserts on
behaviour and on literal reason text -- no new symbol is imported.** That is
deliberate (G9): at base ``36f773c`` these tests RUN and FAIL on what the function
does, rather than erroring on a name that does not exist yet.

The fixture is generic on purpose. ``P``, ``A`` and ``B`` carry no paper, no
protein and no gold value: the rule under test is about the shape of a patch.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


#: The greppable prefix the guard stamps on its rejections. Spelled literally
#: rather than imported so this file exercises behaviour at any SHA.
REASON_PREFIX = "unevidenced_actor_role"

#: A span that names P doing the thing the paper says it does -- being inhibited.
#: This is the span the real audit round argued from.
INHIBITOR_SPAN = "A significantly inhibited P activity in the assay"

#: The reaction's own evidence. It names the chemistry and NO actor at all, which
#: is what made the reaction look structurally incomplete in the first place.
REACTION_SPAN = "A is decomposed in the intestine, resulting in an antibacterial effect"

#: The rationale the audit stage actually gave, generalised off the protein name.
STRUCTURAL_RATIONALE = (
    "add P as an enzyme to the decomposition reaction to resolve the structural "
    "inconsistency where an inhibitor is listed without a target enzyme"
)

#: A real catalysis claim: P is named performing the catalytic role, on this
#: reaction's own chemistry.
CATALYSIS_SPAN = "P catalyses the conversion of A to B under physiological conditions"


def _payload() -> Dict[str, Any]:
    """One reaction whose only actor is an inhibitor, exactly as the card specifies."""

    return {
        "entities": {
            "compounds": [{"name": "A"}, {"name": "B"}],
            "proteins": [{"name": "P"}],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "A decomposition to B",
                    "inputs": ["A"],
                    "outputs": ["B"],
                    "enzymes": [],
                    "modifiers": [
                        {"entity": "P", "role": "inhibitor", "evidence": INHIBITOR_SPAN}
                    ],
                    "evidence": REACTION_SPAN,
                }
            ]
        },
    }


def _enzyme_add(evidence: Any) -> Dict[str, Any]:
    """The op shape the audit stage emitted: a bare-string enzyme at full confidence."""

    op: Dict[str, Any] = {
        "op": "add",
        "path": "/processes/reactions/0/enzymes/-",
        "value": "P",
        "confidence": 1.0,
    }
    if evidence is not None:
        op["evidence"] = evidence
    return op


def _reaction(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload["processes"]["reactions"][0]


# ---------------------------------------------------------------------------
# 1. The rejection case -- the measured defect. FAILS at base 36f773c.
# ---------------------------------------------------------------------------


def test_unevidenced_enzyme_add_is_rejected(tmp_path: Path) -> None:
    """A structural-consistency argument does not license inventing a catalyst.

    This is T-107's Priority-2 failure in miniature. At base the op clears on
    ``confidence >= 0.75`` alone and ``enzymes`` becomes ``["P"]``.
    """

    payload = _payload()
    before = copy.deepcopy(payload)
    rejected_log = tmp_path / "rejected_patch_log.json"

    result, report = apply_patch_with_policy(
        payload,
        [_enzyme_add(STRUCTURAL_RATIONALE)],
        stage="audit",
        rejected_log_path=rejected_log,
    )

    assert report["summary"]["accepted_count"] == 0, report
    assert report["summary"]["rejected_count"] == 1, report
    assert _reaction(result)["enzymes"] == [], _reaction(result)

    reason = report["rejected"][0]["reason"]
    assert reason.startswith(REASON_PREFIX), reason
    assert "catalysis" in reason, reason

    # Merge rule 7: the row survives, reviewable, exactly as it was. The guard
    # refuses the patch; it never deletes a reaction or strips the modifier.
    assert result["processes"] == before["processes"], result["processes"]
    assert result == before

    # Recorded, not silently dropped -- through the existing rejected-log channel.
    assert [entry["reason"] for entry in report["rejected_patch_log"]] == [reason]
    logged = json.loads(rejected_log.read_text(encoding="utf-8"))
    assert any(str(entry.get("reason", "")).startswith(REASON_PREFIX) for entry in logged), logged


# ---------------------------------------------------------------------------
# 2. The preservation case -- mandatory. A guard that rejects both is a defect.
# ---------------------------------------------------------------------------


def test_evidenced_enzyme_add_is_still_accepted() -> None:
    """The same patch, carrying a span that names P as the catalyst, is applied.

    PRODUCT_CONTRACT section 2 keeps depth recoverable through bounded repair;
    section 1 forbids a technically recoverable problem from producing no PWML.
    A guard that refused this would be a new defect, not a fix.
    """

    payload = _payload()

    result, report = apply_patch_with_policy(
        payload, [_enzyme_add(CATALYSIS_SPAN)], stage="audit"
    )

    assert report["summary"]["accepted_count"] == 1, report
    assert report["summary"]["rejected_count"] == 0, report
    assert report["rejected"] == []
    assert _reaction(result)["enzymes"] == ["P"]
    assert report["transaction"]["committed"] is True


# ---------------------------------------------------------------------------
# 3. Presence is not evidence. This is the case a reviewer will construct.
# ---------------------------------------------------------------------------


def test_actor_already_in_the_reaction_is_not_evidence_for_a_new_role(tmp_path: Path) -> None:
    """P is already an actor of this very reaction -- and that licenses nothing.

    The measured defect IS a protein legitimately present as an inhibitor being
    promoted to catalyst on the same reaction. A guard that read "the entity is
    already here" as corroboration would accept the exact patch this card exists
    to reject. The op below offers, as its evidence, the inhibitor row's own span
    -- which does name P, by name, on this reaction. It is still refused, because
    naming an actor is not naming it performing the role being added.
    """

    payload = _payload()
    assert _reaction(payload)["modifiers"][0]["entity"] == "P", "fixture precondition"

    result, report = apply_patch_with_policy(
        payload, [_enzyme_add(INHIBITOR_SPAN)], stage="audit"
    )

    assert report["summary"]["accepted_count"] == 0, report
    assert _reaction(result)["enzymes"] == []
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)

    # And the same span DOES license the role it actually attests, so the refusal
    # above is about the role, not about the span being unusable.
    payload2 = _payload()
    payload2["processes"]["reactions"][0]["modifiers"] = []
    result2, report2 = apply_patch_with_policy(
        payload2,
        [
            {
                "op": "add",
                "path": "/processes/reactions/0/modifiers/-",
                "value": {"entity": "P", "role": "inhibitor", "evidence": INHIBITOR_SPAN},
                "confidence": 0.9,
            }
        ],
        stage="audit",
    )
    assert report2["summary"]["accepted_count"] == 1, report2
    assert _reaction(result2)["modifiers"][0]["role"] == "inhibitor"


# ---------------------------------------------------------------------------
# 4. Shape coverage -- the guard is scoped, and it fails closed.
# ---------------------------------------------------------------------------


def test_enzyme_add_with_no_evidence_field_at_all_is_rejected() -> None:
    payload = _payload()
    result, report = apply_patch_with_policy(payload, [_enzyme_add(None)], stage="audit")

    reason = report["rejected"][0]["reason"]
    assert reason.startswith(REASON_PREFIX), reason
    assert "no evidence span" in reason, reason
    assert _reaction(result)["enzymes"] == []


@pytest.mark.parametrize(
    "path",
    [
        "/processes/reactions/0/modifiers/-",
        "/processes/reactions/0/modifiers_or_enzymes/-",
        "/processes/reactions/0/catalysts/-",
    ],
)
def test_every_actor_container_on_a_reaction_is_guarded(path: str) -> None:
    payload = _payload()
    op = {
        "op": "add",
        "path": path,
        "value": {"entity": "P", "role": "catalyst"},
        "confidence": 1.0,
        "evidence": STRUCTURAL_RATIONALE,
    }
    _result, report = apply_patch_with_policy(payload, [op], stage="audit")
    assert report["summary"]["accepted_count"] == 0, (path, report)
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)


def test_transport_actor_add_is_guarded_and_a_transport_span_licenses_it() -> None:
    payload = {
        "entities": {"compounds": [{"name": "A"}], "proteins": [{"name": "P"}]},
        "processes": {
            "transports": [
                {"name": "A import", "cargo": "A", "transporters": [], "evidence": REACTION_SPAN}
            ]
        },
    }
    op = {
        "op": "add",
        "path": "/processes/transports/0/transporters/-",
        "value": "P",
        "confidence": 1.0,
        "evidence": "P is required for the reaction to proceed",
    }
    _result, report = apply_patch_with_policy(copy.deepcopy(payload), [op], stage="audit")
    assert report["summary"]["accepted_count"] == 0, report
    assert report["rejected"][0]["reason"].startswith(REASON_PREFIX)

    licensed = dict(op, evidence="P transports A across the inner membrane")
    result2, report2 = apply_patch_with_policy(copy.deepcopy(payload), [licensed], stage="audit")
    assert report2["summary"]["accepted_count"] == 1, report2
    assert result2["processes"]["transports"][0]["transporters"] == ["P"]


# ---------------------------------------------------------------------------
# 5. The guard stays inside its boundary -- no collateral tightening.
# ---------------------------------------------------------------------------


def test_guard_does_not_touch_non_actor_paths_removes_or_clears() -> None:
    """Three ops the guard must leave exactly as it found them.

    An entity add is not an actor role; a remove is judged by the core-semantics
    guard that already exists; and emptying a role introduces no actor, so there
    is nothing to license.
    """

    payload = _payload()
    _result, report = apply_patch_with_policy(
        payload,
        [
            {
                "op": "add",
                "path": "/entities/compounds/-",
                "value": {"name": "C"},
                "confidence": 0.9,
                "evidence": "no actor named anywhere in this string",
            },
            {
                "op": "replace",
                "path": "/processes/reactions/0/enzymes",
                "value": [],
                "confidence": 0.9,
                "evidence": "no actor named anywhere in this string",
            },
        ],
        stage="audit",
    )
    reasons = [entry["reason"] for entry in report["rejected"]]
    assert not any(str(reason).startswith(REASON_PREFIX) for reason in reasons), reasons


def test_confidence_rejection_keeps_its_own_reason() -> None:
    """A patch below the confidence bar is still reported as below the bar.

    The guard runs last precisely so reports that already grep the existing reason
    strings keep reading what they always read.
    """

    payload = _payload()
    op = dict(_enzyme_add(STRUCTURAL_RATIONALE), confidence=0.10)
    _result, report = apply_patch_with_policy(payload, [op], stage="audit")
    assert report["rejected"][0]["reason"].startswith("Confidence "), report
