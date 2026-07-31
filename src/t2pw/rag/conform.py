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
* ``scope_membership`` — the admission gate's verdict on a RAG reaction — is
  carried through verbatim (see :func:`_conform_scope_membership_in_place`). It
  is a CORE reaction field, not a RAG additive one, so it must reach
  ``clean_inference_output`` intact or the label the gate wrote is erased at this
  boundary and the reaction arrives downstream indistinguishable from an
  unclassified one.

Deliberately absent: ``element_locations`` and per-reaction ``compartment``. Those
are left for the existing ``default_compartment="cell"`` / compartment-backfill
path to populate on the merged, seed-shaped payload — exactly as for a
single-paper run.

Also absent, when the caller supplies ``seed_payload``: any entity row the merge
BASE already registers. ``merge_additions`` dedupes with ``_extend_unique``, which
keys on the JSON signature of the whole row, so a row that differs from the base's
only by carrying ``rag_provenance`` is appended as a SECOND row for the same
species. See :func:`conform_rag_additions_for_merge` for what that cost.

Import-light and offline: the standard library plus
``t2pw.pipeline.process_normalizer``, from which the registry gate's own two name
functions are imported rather than re-implemented. That module is itself stdlib +
three small sibling modules (``entity_identity`` / ``enzyme_cues`` /
``export_mode``), so importing this one still never pulls the chromadb /
retrieval / LLM stack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from t2pw.pipeline.process_normalizer import (
    # The registry gate's OWN identity functions (see
    # ``validate_registry_references``), imported so "this entity is already in the
    # base" means exactly what the gate means by it — and so no second, subtly
    # different notion of entity identity is invented here. ``_entity_name_norms``
    # also folds a row's declared ``synonyms`` into the set, which is why a base
    # compound named "NAD+" listing "NAD" already covers a RAG row named "NAD".
    _entity_name_norms as _registry_name_norms,
    _normalize as _registry_normalize,
)

# Exactly the buckets ``process_normalizer.validate_registry_references`` unions
# into its registry. ``nucleic_acids`` is in the gate's union (it was added when
# 'pmrHFIJKLM operon' — a reaction input of PMC13278307 in run 2026-07-28_0919 —
# got relocated out of ``entities.compounds``), so it is in this one too: a RAG
# compound row whose name the base already registers as a nucleic acid is still a
# duplicate species, whichever bucket the base filed it under.
_REGISTRY_ENTITY_BUCKETS = ("compounds", "proteins", "protein_complexes", "nucleic_acids")

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

    The join deduplicates. This is a belt-and-braces boundary guard: the amplifying
    append was fixed at its origin in :func:`t2pw.rag.synthesize._merge_into`, but
    this function is the LAST point at which a repeated passage is still a list of
    separable parts — one line later it is a single opaque string that no downstream
    stage can split apart again. Run 2026-07-28_0919 is what that costs when it gets
    through: PMC12444477 (strict) exported a reaction whose evidence string was
    139,576 characters holding one 4,812-character passage 29 times, and a 4.70 MB
    merged payload of which 4.6 MB was duplicated evidence text. Guarding here also
    covers evidence lists that reach the seam from a producer other than
    ``_merge_into``.
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
        # ``dict.fromkeys`` is safe HERE and only here: ``parts`` is a List[str] —
        # every dict record was already unwrapped to its ``.text`` on the line above,
        # so nothing unhashable reaches it. (The same trick is NOT usable inside
        # ``t2pw.rag.synthesize``, where the elements are still dicts and carry a
        # per-retrieval ``score`` that differs for the same chunk; that side keys on
        # the explicit ``(chunk_id, text)`` pair instead.) It preserves first-seen
        # order, so the leading passage of the evidence list still leads the string.
        return " ".join(dict.fromkeys(parts))
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


def _conform_scope_membership_in_place(row: Dict[str, Any]) -> None:
    """Normalize ``row['scope_membership']`` the way the core cleaner does.

    ``scope_membership`` is the one label that has to survive the whole boundary
    chain — synthesis -> here -> ``clean_inference_output`` -> normalization — or
    a RAG-imported reaction arrives downstream unlabelled and the core
    out-of-scope filter and the lock manifest disagree about it (the split
    ``pipeline._carry_scope_membership`` was written to close for seed rows, and
    whose docstring records that RAG rows "cannot carry a scope label even in
    principle"). ``dict(row)`` already copies it; this makes the survival
    EXPLICIT and applies the same two rules ``_carry_scope_membership`` applies,
    so both sides of the boundary agree:

    * stripped, never case-folded — the lock manifest records the model's own
      spelling and the two artifacts must not disagree about what was said;
    * a non-string label is DROPPED rather than coerced, which lands it in the
      "absent" case that every downstream reader treats as KEEP.
    """
    if "scope_membership" not in row:
        return
    scope = row.get("scope_membership")
    if isinstance(scope, str) and scope.strip():
        row["scope_membership"] = scope.strip()
    else:
        row.pop("scope_membership", None)


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
    _conform_scope_membership_in_place(conformed)

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


def _base_entity_registry(seed_payload: Any) -> Set[str]:
    """Every normalized entity name (+ declared synonym) the merge BASE registers.

    Read-only; returns a fresh mutable set so the caller can keep adding to it as
    it accepts rows (that is what makes one pass do both base-vs-envelope and
    envelope-vs-envelope dedupe).
    """
    base = seed_payload if isinstance(seed_payload, dict) else {}
    entities = base.get("entities")
    if not isinstance(entities, dict):
        return set()
    norms: Set[str] = set()
    for bucket in _REGISTRY_ENTITY_BUCKETS:
        norms |= _registry_name_norms(_safe_list(entities.get(bucket)))
    norms.discard("")
    return norms


def conform_rag_additions_for_merge(
    rag_payload: Any,
    seed_payload: Any = None,
) -> Dict[str, Any]:
    """Build a ``merge_additions``-ready ``{"additions": {...}}`` envelope.

    Takes a RAG-synthesized payload and returns an additions envelope whose rows
    are coerced to the core shape (see module docstring). The result is suitable
    for :func:`t2pw.pipeline.pipeline.merge_additions`, which injects these rows
    into the rich seed payload. Never mutates ``rag_payload``.

    ``seed_payload`` is the payload this envelope will be merged INTO — i.e. the
    exact ``base`` argument the caller is about to hand ``merge_additions``, which
    at the S3 seam is the Stage-1+Stage-2 ``final_payload``, NOT the Stage-1 seed
    that synthesis was given. When supplied, an entity row whose registry-normalized
    name is already registered in that base is DROPPED instead of added.

    Why this seam needs the guard at all. ``merge_additions`` dedupes entity rows
    with ``_extend_unique``, whose signature is ``json.dumps(row, sort_keys=True)``.
    Synthesis rebuilds an entity row for every participant of every reaction it
    resolved — and it resolved the seed's own reactions too — so the base's
    ``{"name": "LpxA", "class": "protein", "confidence": 1.0, "provenance":
    "extracted", "organism": "Escherichia coli"}`` and RAG's ``{"name": "LpxA",
    "organism": ..., "rag_provenance": {"source_id": "seed_paper", ...},
    "source_refs": ["seed_paper"], ...}`` are two different signatures for one
    protein, and both survive. Run 2026-07-28_0919, PMC12444477 strict: 31
    ``entities.proteins`` rows for 22 distinct normalized names (all nine seed
    enzymes doubled) and 56 ``entities.compounds`` rows for 43 distinct names (13
    doubled). Byte-identical names, so the 2026-07-23 synonym resolver — which
    collapses SYNONYMS — correctly never touched them.

    Why HERE and not in synthesis, which also holds a seed payload. Two reasons.
    (1) The payload synthesis holds is the STAGE-1 seed; the merge base is Stage 1
    **plus Stage 2**. In the same run 'lipid IV_A precursor', 'Kdo-lipid A
    precursor' and 'tetra-acylated disaccharide intermediate' are Stage-1 reaction
    participants that Stage 1 never registered and Stage 2 did — invisible to
    synthesis, duplicated all the same. (2) ``synthesize_with_report`` returns a
    STANDALONE payload whose entity buckets must cover its own reactions and which
    feeds ``validate_provenance`` / ``assign_tiers`` / the citation report;
    tests/test_rag_synthesize.py pins that contract (a seed compound keeps
    ``source_refs == ["PMID:0001"]``; NdmA is listed). Dropping seed entities there
    would strand reaction references and delete the corroboration the seed is
    indexed to provide. Dropping them HERE removes only the duplicate ROW: the
    pre-merge payload still shows, e.g., 'LpxC' with
    ``source_refs == ["seed_paper", "PMC11046580"]``.

    ``seed_payload=None`` (the default) keeps the previous behavior except that the
    envelope is still deduped against ITSELF, so two synthesized rows for one
    normalized name can no longer both be appended. It does NOT fall back to
    guessing from ``rag_provenance.source_id == "seed_paper"``: that tag means
    "a seed reaction registered this entity first", not "the base already has it".
    Auditing every ``merged_payload.json`` under ``runs/`` found seed-tagged rows
    that are the ONLY row for their species — 'PSA' and 'sulfacetamide'
    (PMC13231680), 'PEtN', 'mcr genes' and 'pmrCAB operon' (PMC13278307),
    'α-ketoglutarate' (PMC12312563) — so a tag-based rule would have deleted real
    chemistry. Only name identity against the actual base is safe.

    Identity is ``process_normalizer._normalize`` — the registry gate's own — not a
    looser one. That matters in the false-merge direction: it preserves ``+`` and
    ``:``, so 'NAD+' does not collapse into 'NAD', and it is purely lexical, so
    'PhoP' does not collapse into 'phosphorylated PhoP'. Checked against the real
    corpus (four runs under ``runs/``): every collision it produces is between
    BYTE-IDENTICAL names. It under-merges rather than over-merges — 'lipid IV_A'
    and 'lipid IV A' stay two rows — which is the safe direction here and is the
    synonym resolver's job, not this guard's.
    """
    payload = rag_payload if isinstance(rag_payload, dict) else {}
    additions: Dict[str, Any] = {}

    src_entities = payload.get("entities")
    if isinstance(src_entities, dict):
        # Seeded with the base's registry, then grown as rows are accepted, so the
        # same pass rejects both "already in the base" and "already emitted by an
        # earlier bucket/row of this same envelope".
        registered = _base_entity_registry(seed_payload)
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
                norm = _registry_normalize(name)
                if norm:
                    if norm in registered:
                        continue
                    registered.add(norm)
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
