"""Conform a RAG-synthesized payload into a Stage-2 ``additions`` envelope.

The multi-paper RAG synthesis (:mod:`t2pw.rag.synthesize`) emits a *thin*,
reaction-only :class:`~t2pw.schema.Payload`: reactions whose ``evidence`` is a
LIST of records, participants that may be ``{"name", "stoichiometry"}`` dicts,
and evidence-built ``compounds`` / ``proteins`` buckets. The rich single-paper
(Stage-1) payload, by contrast, carries ``element_locations``, per-reaction
``compartment`` / ``biological_state``, per-entity ``mapping_meta`` and a
``string`` ``evidence`` field — the shape every downstream stage assumes.

Rather than letting synthesis REPLACE the rich seed payload with its thin one
(which crashed the post-pipeline reading fields RAG never emitted), the seam
INJECTS RAG's new reactions / entities INTO the seed with the *same* additive
machinery Stage 2 uses (:func:`t2pw.pipeline.pipeline.merge_additions`). That
function consumes an ``{"additions": {...}}`` envelope. This module builds that
envelope from a synthesized payload, coercing each row to the CORE shape:

* ``evidence`` is coerced from a list of records to a single **string** (matching
  ``ReactionModel.evidence: str`` / ``ActorModel.evidence: str`` in
  :mod:`t2pw.pipeline.payload_models`), on both reactions and enzyme actors.
* Reaction participants are coerced to their plain **name string** — the core
  representation the whole pipeline uses (``clean_inference_output`` accepts only
  string participants, ``draft_graph`` reads them with ``_safe_str``, and
  ``_reaction_io_key`` stringifies them). A ``{"name", "stoichiometry"}`` dict is
  reduced to its name; stoichiometry is not carried on reaction participants
  anywhere in the core pipeline, so this invents no new shape.
* The carried-forward scaffolding buckets (``species`` / ``subcellular_locations``
  / ``cell_types`` / ``tissues``) are EXCLUDED — they already live on the seed, so
  re-adding them would double-list context entities.

Deliberately absent: ``element_locations`` and per-reaction ``compartment``. Those
are left for the existing ``default_compartment="cell"`` / compartment-backfill
path to populate on the merged, seed-shaped payload — exactly as for a
single-paper run.

Import-light and offline: only the standard library is used, so importing this
module never pulls the chromadb / retrieval / LLM stack.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Evidence-built entity buckets synthesis produces that are genuine additive
# chemistry (safe to merge into the seed). The scaffolding buckets that
# ``t2pw.rag.synthesize._carry_forward_scaffolding`` copies from the seed
# (species / subcellular_locations / cell_types / tissues) are intentionally
# absent so they are not double-added.
_MERGE_ENTITY_BUCKETS = ("compounds", "proteins", "protein_complexes")


def _evidence_to_str(value: Any) -> str:
    """Flatten a RAG ``evidence`` value to a single plain string.

    Mirrors :func:`t2pw.pipeline.pipeline._evidence_text`: a string is returned
    as-is, a ``{"text": ...}`` record yields its text, and a list of records /
    strings is joined into one human-readable string.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                parts.append(item["text"].strip())
        return " ".join(parts)
    return ""


def _coerce_evidence_in_place(row: Dict[str, Any]) -> None:
    """Coerce ``row['evidence']`` from a list/record to a string (or drop it)."""
    if "evidence" not in row:
        return
    text = _evidence_to_str(row["evidence"])
    if text:
        row["evidence"] = text
    else:
        row.pop("evidence", None)


def _conform_participant(participant: Any) -> str:
    """Reduce a reaction participant to its core plain-name string.

    RAG emits a string when stoichiometry is absent / 1, else a
    ``{"name", "stoichiometry"}`` dict (:func:`t2pw.rag.synthesize._participant_row`).
    The core pipeline represents every reaction participant as a plain string, so
    a dict is reduced to its name here — no new shape is invented and no participant
    is silently dropped by the string-only ``clean_inference_output`` path.
    """
    if isinstance(participant, str):
        return participant.strip()
    if isinstance(participant, dict):
        for key in ("name", "entity", "compound", "protein", "element", "nucleic_acid"):
            value = participant.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _conform_reaction(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce one RAG reaction row to the core additive shape."""
    conformed = dict(row)

    conformed["inputs"] = [
        name for name in (_conform_participant(p) for p in _safe_list(row.get("inputs"))) if name
    ]
    conformed["outputs"] = [
        name for name in (_conform_participant(p) for p in _safe_list(row.get("outputs"))) if name
    ]

    _coerce_evidence_in_place(conformed)

    enzymes = row.get("enzymes")
    if isinstance(enzymes, list):
        conformed_enzymes: List[Any] = []
        for actor in enzymes:
            if isinstance(actor, dict):
                actor = dict(actor)
                _coerce_evidence_in_place(actor)
            conformed_enzymes.append(actor)
        conformed["enzymes"] = conformed_enzymes

    return conformed


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def conform_rag_additions_for_merge(rag_payload: Any) -> Dict[str, Any]:
    """Build a ``merge_additions``-ready ``{"additions": {...}}`` envelope.

    Takes a RAG-synthesized payload and returns an additions envelope whose rows
    are coerced to the core shape (see module docstring). The result is suitable
    for :func:`t2pw.pipeline.pipeline.merge_additions`, which injects these rows
    into the rich seed payload. Never mutates ``rag_payload``.
    """
    payload = rag_payload if isinstance(rag_payload, dict) else {}
    additions: Dict[str, Any] = {}

    src_entities = payload.get("entities")
    if isinstance(src_entities, dict):
        entities_add: Dict[str, Any] = {}
        for bucket in _MERGE_ENTITY_BUCKETS:
            rows = src_entities.get(bucket)
            if not isinstance(rows, list):
                continue
            conformed_rows: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = row.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                conformed_row = dict(row)
                conformed_row["name"] = name.strip()
                _coerce_evidence_in_place(conformed_row)
                conformed_rows.append(conformed_row)
            if conformed_rows:
                entities_add[bucket] = conformed_rows
        if entities_add:
            additions["entities"] = entities_add

    src_processes = payload.get("processes")
    if isinstance(src_processes, dict):
        reactions = [
            _conform_reaction(row)
            for row in _safe_list(src_processes.get("reactions"))
            if isinstance(row, dict)
        ]
        if reactions:
            additions["processes"] = {"reactions": reactions}

    return {"additions": additions}


__all__ = ["conform_rag_additions_for_merge"]
