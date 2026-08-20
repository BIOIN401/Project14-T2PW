"""F-058 — the transport branch of ``_inject_name_based_modifiers`` fabricated its transporter.

``pipeline._inject_name_based_modifiers`` has two branches. The **reaction**
branch was hardened in 2026-07 after a bare substring test turned 9 Stage-1
enzymes into 204 shipped enzyme rows: it now requires the actor name to sit
inside a catalysis-cue window and to be the only actor that qualifies, and three
tests in ``test_rag_payload_gate_guardrails.py`` pin it.

The **transport** branch never received any of that, and **nothing in the
repository asserted on it at all** — ``test_rag_payload_gate_guardrails.py`` is
the only file that calls this function and all three of its name-heuristic tests
are on the reaction branch. This file closes that seam.

What the untested branch shipped, measured by replaying the production function
over the 64 committed legs in ``runs/`` and ``runs_verify/``
(``evidence/c058_corpus_base.json``): **24 fabricated transporter rows**, 23 of
them the adenylation enzyme ``EntE`` attached as the transporter of an
enterobactin transport step purely because ``"ente"`` is a prefix of
``"ent-erobactin"``, each citing a span that names TolC or TonB and never names
EntE. The gold set is explicit that these steps must not be emitted —
PMC12452463 ``notes``: *"Export of enterobactin from the cytoplasm is never
described at all, so no efflux step may be emitted"*; PMC12096016
``export_rationale``: *"Export must exclude MenD, LDH and the transport
mentions."*

**Merge rule 7 is the constraint that shapes the fix**: on ``PMC12452463`` the
Stage-1 row already carried the correct ``FepA`` and the defect *appended* an
``EntE`` beside it. A guard that dropped both would trade a fabrication for a
deletion, so ``test_leg_a_keeps_its_correct_fepa_transporter`` asserts the
correct row survives byte-identically on the committed artifact.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.pipeline import _inject_name_based_modifiers  # noqa: E402

RUN = ROOT / "runs_verify" / "2026-08-18_1328" / "papers"
LEG_A = RUN / "PMC12452463" / "strict"


def _payload(
    *,
    transports: List[Dict[str, Any]],
    proteins: List[str] = (),
    complexes: List[str] = (),
) -> Dict[str, Any]:
    return {
        "entities": {
            "proteins": [{"name": name} for name in proteins],
            "protein_complexes": [{"name": name} for name in complexes],
        },
        "processes": {"reactions": [], "transports": transports},
    }


def _transporters(payload: Dict[str, Any], index: int = 0) -> List[Any]:
    return payload["processes"]["transports"][index].get("transporters") or []


def _replay(leg: Path) -> Dict[str, Any]:
    """Run the production injector on a deep copy of a committed Stage-1 payload."""
    stage1 = leg / "stage1_payload.json"
    assert stage1.exists(), f"committed Stage-1 artifact missing: {stage1}"
    payload = deepcopy(json.loads(stage1.read_text(encoding="utf-8")))
    _inject_name_based_modifiers(payload)
    return payload


def _actor_names(rows: List[Any]) -> List[str]:
    return [
        str(
            row.get("entity")
            or row.get("protein")
            or row.get("protein_complex")
            or row.get("name")
            or ""
        )
        for row in rows
        if isinstance(row, dict)
    ]


# --------------------------------------------------------------------------
# The committed legs — the behaviour that shipped, replayed offline
# --------------------------------------------------------------------------
def test_no_committed_leg_gains_a_fabricated_ente_transporter() -> None:
    """The F-058 regression, asserted on committed production artifacts.

    Deterministic, offline, no Stage-2 additions and no RAG: the injector alone
    reproduces the committed ``merged_payload.json`` transporter entries
    character for character on all five transport rows of these four legs
    (``evidence/c058_replay_base.json``, ``rows_reproducing_committed: 5``).
    ``EntE`` is declared in ``entities.proteins`` on every one of them and is
    named by none of their evidence spans.
    """
    fabricated = []
    for leg in sorted(RUN.glob("*/*")):
        if not (leg / "stage1_payload.json").exists():
            continue
        payload = _replay(leg)
        for transport in payload["processes"]["transports"]:
            for name in _actor_names(transport.get("transporters") or []):
                if name.strip().lower() == "ente":
                    fabricated.append(f"{leg.parent.name}/{leg.name}: {transport.get('name')}")
    assert fabricated == []


def test_leg_a_keeps_its_correct_fepa_transporter() -> None:
    """Merge rule 7 — the correct transporter must survive the guard.

    ``PMC12452463/strict`` transport 0 arrives from Stage 1 already carrying
    ``{"protein": "FepA", ...}``; the defect appended an ``EntE`` beside it. The
    fix must refuse the ``EntE`` **without** disturbing the ``FepA`` row, so this
    asserts the Stage-1 entry is preserved byte-identically *and* is now the only
    transporter on the row.
    """
    stage1 = json.loads((LEG_A / "stage1_payload.json").read_text(encoding="utf-8"))
    seeded = stage1["processes"]["transports"][0]["transporters"]
    assert _actor_names(seeded) == ["FepA"], "fixture drift: Stage 1 no longer seeds FepA"

    replayed = _replay(LEG_A)["processes"]["transports"][0]["transporters"]

    assert replayed[0] == seeded[0]
    assert _actor_names(replayed) == ["FepA"]


def test_the_committed_leg_a_merged_payload_is_the_row_this_card_removes() -> None:
    """The shipped defect, read straight off the committed merged artifact.

    Pins what the fix is removing so a future reader does not have to trust a
    prose record: the merged payload carries a second transporter, typed
    ``protein_complex`` although ``EntE`` is declared in ``entities.proteins``,
    citing a span that never names it.
    """
    merged = json.loads((LEG_A / "merged_payload.json").read_text(encoding="utf-8"))
    shipped = merged["processes"]["transports"][0]["transporters"]
    assert _actor_names(shipped) == ["FepA", "EntE"]
    assert shipped[1]["protein_complex"] == "EntE"
    assert shipped[1]["provenance"] == "inferred"
    assert "EntE" not in shipped[1]["evidence"]


# --------------------------------------------------------------------------
# The four guards, each attacked on its own
# --------------------------------------------------------------------------
def test_transport_heuristic_refuses_a_name_that_is_a_substring_of_a_longer_word() -> None:
    """Guard 1 — whole-token match. ``EntE`` is not a mention inside ``enterobactin``.

    This is F-058's entire mechanism in one payload, and the cue window alone
    does **not** catch it: "TolC-dependent" puts a cue right beside the
    accidental match.
    """
    payload = _payload(
        proteins=["EntE"],
        transports=[
            {
                "name": "enterobactin export",
                "cargo": "enterobactin",
                "evidence": "secreted to the extracellular environment by a TolC-dependent process",
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    assert _transporters(payload) == []


def test_transport_heuristic_requires_a_transport_role_cue_near_the_name() -> None:
    """Guard 2 — cue window. A protein merely named in a transport row is not its transporter.

    ``Fur`` is a whole token here and is the only declared actor, so guards 1, 3
    and 4 all pass it. Only the 80-character cue window refuses it: the row's own
    ``uptake`` sits far outside the window and the sentence naming ``Fur`` makes
    a regulatory claim, not a transport one.
    """
    payload = _payload(
        proteins=["Fur"],
        transports=[
            {
                "name": "citrate uptake",
                "cargo": "citrate",
                "evidence": (
                    "Citrate crosses the inner membrane. Growth was assayed in minimal "
                    "medium supplemented with glucose and casamino acids under aerobic "
                    "conditions. Fur represses the operon in iron-replete cells."
                ),
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    assert _transporters(payload) == []


def test_transport_heuristic_refuses_a_row_naming_two_candidate_actors() -> None:
    """Guard 3 — exactly one candidate, mirroring the reaction branch's refusal.

    Ambiguity is not a licence to guess: two declared actors both sit in a
    transport-role cue window here, and the row says nothing about which one
    moves the cargo.
    """
    payload = _payload(
        proteins=["TolC", "AcrB"],
        transports=[
            {
                "name": "siderophore export",
                "cargo": "siderophore",
                "evidence": "Export across the outer membrane requires the TolC channel and the AcrB pump.",
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    assert _transporters(payload) == []


def test_transport_heuristic_refuses_the_rows_own_declared_cargo() -> None:
    """Guard 4 — the thing being moved is not the thing moving it.

    Found by replaying the injector over the committed corpus, not predicted:
    ``PMC12856317/strict`` declares ``{"name": "ALAS2 import into mitochondrial
    matrix", "cargo": "ALAS2"}`` and shipped ``ALAS2`` as its own transporter.
    A cargo name always sits in a transport-role cue window, because the row is
    *about* transporting it, so guards 1-3 cannot refuse this on their own.
    """
    payload = _payload(
        proteins=["ALAS2"],
        transports=[
            {
                "name": "ALAS2 import into mitochondrial matrix",
                "cargo": "ALAS2",
                "evidence": (
                    "heme binds directly to the ALAS mitochondrial targeting sequence, "
                    "preventing protein translocation into the matrix"
                ),
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    assert _transporters(payload) == []


def test_transport_heuristic_ignores_an_oversized_evidence_blob() -> None:
    """The ``MAX_INJECTOR_EVIDENCE_CHARS`` bound reaches the transport branch too.

    ``rag/conform.py`` flattens a retrieved corpus into one string before this
    pass runs. Matching names against a corpus attaches every actor to every row.
    """
    filler = "Unrelated background prose about membranes. " * 40
    payload = _payload(
        proteins=["FepA"],
        transports=[
            {
                "name": "step",
                "evidence": filler + "imported by the FepA receptor. " + filler,
            }
        ],
    )
    assert len(payload["processes"]["transports"][0]["evidence"]) > 400

    _inject_name_based_modifiers(payload)

    assert _transporters(payload) == []


# --------------------------------------------------------------------------
# Non-vacuity in the other direction — the guard is not a blanket refusal
# --------------------------------------------------------------------------
def test_transport_heuristic_still_attaches_a_correctly_named_transporter() -> None:
    """A deliberate, correct name-based attachment is still made.

    Shaped on ``PMC12452463``'s real sentence, with the Stage-1 ``FepA`` row
    removed so the heuristic is the only thing that could supply it. If this
    goes red the card has refused correct biology, which merge rule 7 forbids.
    """
    payload = _payload(
        proteins=["FepA"],
        transports=[
            {
                "name": "ferric-enterobactin import",
                "cargo": "ferric enterobactin",
                "evidence": (
                    "The iron-bound enterobactin is recognized by specific outer membrane "
                    "receptors (e.g. FepA in E. coli), which transport it into the bacterial cell"
                ),
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    rows = _transporters(payload)
    assert _actor_names(rows) == ["FepA"]
    assert rows[0]["provenance"] == "inferred"
    assert rows[0]["confidence"] == 0.9


def test_injected_transporter_uses_the_protein_key_for_a_protein_actor() -> None:
    """The second defect on the same statement: ``protein_complex`` was written unconditionally.

    ``EntE`` is declared in ``entities.proteins`` yet shipped as a
    ``protein_complex``, which ``map_ids`` then had to rewrite back to
    ``"entity_type": "protein"``. The key now follows the bucket the actor was
    collected from.
    """
    payload = _payload(
        proteins=["FepA"],
        transports=[
            {
                "name": "ferric-enterobactin import",
                "evidence": "the FepA receptor transports ferric enterobactin into the cell",
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    row = _transporters(payload)[0]
    assert row["protein"] == "FepA"
    assert "protein_complex" not in row


def test_injected_transporter_keeps_protein_complex_for_a_genuine_complex() -> None:
    """...and a genuine complex is still emitted under ``protein_complex``.

    The key fix must be a *typing* fix, not a blanket rename: an actor declared
    in ``entities.protein_complexes`` keeps the complex key.
    """
    payload = _payload(
        complexes=["TonB-ExbB-ExbD complex"],
        transports=[
            {
                "name": "ferric siderophore import",
                "evidence": "The TonB-ExbB-ExbD complex energizes transport across the outer membrane.",
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    row = _transporters(payload)[0]
    assert row["protein_complex"] == "TonB-ExbB-ExbD complex"
    assert "protein" not in row


def test_injected_transporter_cites_the_matched_window_not_a_blind_prefix() -> None:
    """The stored evidence is the window that justified the attachment.

    The old branch wrote ``(transport.get("evidence") or "")[:120]`` — read off
    the row's *raw* evidence, bypassing the size bound applied to the text the
    match is made against, so a row matched on its name while carrying a
    flattened RAG corpus cited 120 characters of a corpus that took no part in
    the match. This mirrors what the reaction branch already stores.
    """
    evidence = (
        "Iron limitation induces the siderophore regulon. " * 2
        + "The ferric-enterobactin complex is imported by the FepA receptor."
    )
    payload = _payload(
        proteins=["FepA"],
        transports=[{"name": "ferric-enterobactin import", "evidence": evidence}],
    )

    _inject_name_based_modifiers(payload)

    row = _transporters(payload)[0]
    assert "FepA" in row["evidence"]
    assert row["evidence"] != evidence[:120]
    assert row["source_refs"] == [row["evidence"]]


def test_a_transporter_already_credited_under_the_typed_key_is_not_rewritten() -> None:
    """An actor named under ``entity`` counted as an *empty* slot and was overwritten.

    The old ``already_present`` test read only ``protein``/``protein_complex``,
    and the patch loop treated any row without those two keys as unfilled — so a
    row already naming ``FepA`` under the typed ``entity`` key had a second name
    written into it.
    """
    seeded = {"entity": "FepA", "role": "transporter"}
    payload = _payload(
        proteins=["FepA"],
        transports=[
            {
                "name": "ferric-enterobactin import",
                "evidence": "the FepA receptor transports ferric enterobactin into the cell",
                "transporters": [dict(seeded)],
            }
        ],
    )

    _inject_name_based_modifiers(payload)

    assert _transporters(payload) == [seeded]


# --------------------------------------------------------------------------
# The reaction branch is out of this card's boundary and must not move
# --------------------------------------------------------------------------
def test_the_reaction_branch_emits_exactly_what_it_emitted_before() -> None:
    """Pins the reaction branch's emitted row, unchanged, at base and at tip.

    The transport guards live entirely below the reaction loop and share no
    mutable state with it, but "the diff does not touch it" is a claim about
    source, not behaviour. This asserts the behaviour. The corpus replay
    (``evidence/c058_corpus_{base,tip}.json``) carries the same claim over all
    64 committed legs as a per-leg digest of every reaction's ``enzymes`` and
    ``modifiers``: identical on all 64.
    """
    payload = {
        "entities": {
            "proteins": [],
            "protein_complexes": [{"name": "Hexokinase complex"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "glucose phosphorylation",
                    "evidence": (
                        "Glucose is phosphorylated in the cytosol; the reaction is "
                        "catalyzed by Hexokinase complex."
                    ),
                }
            ],
            "transports": [],
        },
    }

    _inject_name_based_modifiers(payload)

    modifiers = payload["processes"]["reactions"][0]["modifiers"]
    assert len(modifiers) == 1
    assert modifiers[0]["entity"] == "Hexokinase complex"
    assert modifiers[0]["entity_type"] == "protein_complex"
    assert modifiers[0]["role"] == "catalyst"
    assert "catalyzed by Hexokinase complex" in modifiers[0]["evidence"]
    assert modifiers[0]["source_refs"] == [modifiers[0]["evidence"]]
