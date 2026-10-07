"""Structured extraction of biochemical claims from retrieved passages.

The contract with the model
---------------------------
The model is shown numbered passages, each carrying an ``evidence_id`` this run
minted. It must cite those ids for **every field it fills**. A field with no
citation, or with a citation to an id that was not shown, is treated as
unsupported and is dropped here -- before admission, before the graph, before
anything can act on it.

That is the mechanism behind "do not let the model create a stronger claim than
its evidence supports". It is not a prompt instruction the model may ignore; the
citation is checked against the passages that were actually sent.

Nothing in this module talks to ChEBI, KEGG, HMDB, UniProt or any other
biochemical database. Names stay as the paper wrote them, normalized locally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from t2pw.mining.config import PROMPT_VERSION
from t2pw.mining.models import (
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
    DIRECTION_UNSPECIFIED,
    EVIDENCE_REVIEW,
    EVIDENCE_UNKNOWN,
    EXTRACTION_OK,
    UNVERIFIED_EVIDENCE_TYPE_NOTE,
    EvidenceSpan,
    ExtractedClaim,
    ReactionParticipant,
    make_claim_id,
    reaction_identity,
    reads_as_review,
)
from t2pw.mining.modelclient import MiningModelClient, ModelCallOutcome
from t2pw.mining.species import (
    aggregate_species,
    fold_text,
    grade_sentences,
    species_span_names_target,
)

SYSTEM_PROMPT = """You are a biochemical literature analyst building a research graph.

You are given numbered passages from ONE scientific paper. Extract only the
biochemical reactions those passages actually support.

Absolute rules:
1. Cite evidence for every field you fill, in THAT field's own list:
   enzyme -> enzyme_evidence_ids, gene -> gene_evidence_ids,
   species -> species_evidence_ids, compartment -> compartment_evidence_ids,
   direction -> direction_evidence_ids, reversible -> reversible_evidence_ids,
   and each substrate / product -> its own evidence_ids. Put in each list the
   evidence_id of every passage that supports that field specifically. The
   claim-level evidence_ids list does not count for any other field: a filled
   field whose own list is missing or empty will be discarded.
2. Never fill a field the passages do not support. If the passages give a
   substrate and a product but never name the enzyme, leave "enzyme" empty. An
   empty field is correct; an invented one is a defect.
3. Use the paper's own names for metabolites, enzymes and genes. Do not
   translate them into database identifiers and do not add identifiers.
4. "species" is the organism the passages say the reaction occurs in. If the
   passages do not say, leave it empty. If the reaction is asserted for the
   organism only by analogy, orthology or similarity, set species_inferred=true.
5. Set hypothesis=true when the paper proposes or models the reaction rather
   than reporting an observation ("we propose", "may be converted",
   "a putative step").
6. One reaction per claim. A multi-step conversion described as a single
   sentence is several claims, each with its own substrates and products.
7. Return valid JSON of exactly this shape. Omit a field the passages do not
   support; never omit the *_evidence_ids list of a field you fill.
   {"claims": [{
     "substrates": [{"name": "...", "evidence_ids": ["ev_..."]}],
     "products": [{"name": "...", "evidence_ids": ["ev_..."]}],
     "enzyme": "...", "enzyme_evidence_ids": ["ev_..."],
     "gene": "...", "gene_evidence_ids": ["ev_..."],
     "species": "...", "species_evidence_ids": ["ev_..."],
     "direction": "forward", "direction_evidence_ids": ["ev_..."],
     "compartment": "...", "compartment_evidence_ids": ["ev_..."],
     "evidence_ids": ["ev_..."]}]}
   Every claim also carries "species_inferred" and "hypothesis", each set to
   true or false as rules 4 and 5 decide for THAT claim; neither has a default.
   If the passages support no reaction, return {"claims": []} -- an empty list
   is a correct answer.
"""

_PARTICIPANT_SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string"},
        "entity_type": {"type": "string"},
        "stoichiometry": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
}

CLAIM_SCHEMA: Dict[str, Any] = {
    "name": "mining_claims",
    "strict": False,
    "schema": {
        "type": "object",
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["substrates", "products", "evidence_ids"],
                    "properties": {
                        "substrates": {"type": "array", "items": _PARTICIPANT_SCHEMA},
                        "products": {"type": "array", "items": _PARTICIPANT_SCHEMA},
                        "enzyme": {"type": "string"},
                        "enzyme_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "gene": {"type": "string"},
                        "gene_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "direction": {
                            "type": "string",
                            "enum": [DIRECTION_FORWARD, DIRECTION_REVERSE, DIRECTION_UNSPECIFIED],
                        },
                        "direction_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "reversible": {"type": ["boolean", "null"]},
                        "reversible_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "compartment": {"type": "string"},
                        "compartment_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "species": {"type": "string"},
                        "species_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "species_inferred": {"type": "boolean"},
                        "conditions": {"type": "string"},
                        "tissue": {"type": "string"},
                        "hypothesis": {"type": "boolean"},
                        "pathway_names": {"type": "array", "items": {"type": "string"}},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string"},
                    },
                },
            }
        },
    },
}

#: The last-resort schema used when a full-schema attempt keeps failing. Asking
#: for less is a real retry; re-sending the same request is not.
TIGHT_CLAIM_SCHEMA: Dict[str, Any] = {
    "name": "mining_claims_minimal",
    "strict": False,
    "schema": {
        "type": "object",
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["substrates", "products", "evidence_ids"],
                    "properties": {
                        "substrates": {"type": "array", "items": _PARTICIPANT_SCHEMA},
                        "products": {"type": "array", "items": _PARTICIPANT_SCHEMA},
                        "species": {"type": "string"},
                        "species_evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    },
}


def render_passages(spans: Sequence[EvidenceSpan], *, paper_title: str = "") -> str:
    """The user message: the exact passages, each labelled with its evidence id."""
    header = [f"PAPER: {paper_title}"] if paper_title else []
    header.append(
        "PASSAGES (cite these evidence_id values; do not invent ids):"
    )
    body = []
    for span in spans:
        locator = " | ".join(
            part
            for part in (
                span.section or "",
                f"page {span.page}" if span.page else "",
                f"para {span.paragraph}" if span.paragraph else "",
                span.locator or "",
            )
            if part
        )
        body.append(
            f"[evidence_id: {span.evidence_id}]"
            + (f" ({locator})" if locator else "")
            + f"\n{span.text}\n"
        )
    return "\n".join([*header, "", *body])


def _valid_ids(raw: Any, allowed: Mapping[str, EvidenceSpan]) -> List[str]:
    """Keep only cited ids that were actually shown to the model."""
    out: List[str] = []
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    for item in raw or []:
        text = str(item or "").strip()
        if text in allowed and text not in out:
            out.append(text)
    return out


#: How a kept field got its evidence (``ExtractedClaim.field_attribution``).
ATTRIBUTION_MODEL_CITATION = "model_citation"
ATTRIBUTION_VERBATIM_CITED_SPAN = "verbatim_cited_span"

#: The identity namespace of a claim none of whose participants is supported
#: (ORCH-745 F-5): kept as a lead, never merged into a supported identity.
UNSUPPORTED_PARTICIPANTS_NAMESPACE = "unsupported-participants"

#: Fields the verbatim fallback may recover. Only names: a value that is a
#: name can be checked against a passage; a direction or a boolean cannot. A
#: recovered species is kept as the observed value but is never species evidence
#: (see ``build_claims``): the target species then classifies ``species_ambiguous``.
_VERBATIM_RECOVERABLE = ("enzyme", "gene", "species")


def _fold(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def _verbatim_span(value: str, span_ids: Sequence[str], allowed: Mapping[str, EvidenceSpan]) -> str:
    """The first of ``span_ids`` whose text contains ``value`` verbatim.

    Case- and whitespace-insensitive, and never inside a longer word ("X" does
    not occur in "LOX2"). ``span_ids`` are the claim's own cited participant
    spans, all already checked to be spans the model was shown.
    """
    needle = _fold(value)
    if not needle:
        return ""
    pattern = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
    for eid in span_ids:
        if eid in allowed and pattern.search(_fold(allowed[eid].text)):
            return eid
    return ""


#: A value that echoes the prompt's template instead of reading a passage
#: (pilot-003's thin answers stored ``"stoichiometry": "has_stoichiometry"`` and
#: ``"products_evidence_ids_placeholder"``), or an empty JSON-ish string. The
#: ``has_`` / ``_placeholder`` forms are case-sensitive, as in
#: ``modelclient.thin_answer``: a gene symbol such as ``HAS_1`` is not an echo.
_PLACEHOLDER = re.compile(
    r"^(?:has_\w+|\w+_placeholder|\.\.\.|\u2026|ev_\.\.\.|\[\s*\]|\{\s*\}|null|None|\"\"|'')$"
)
#: A stoichiometric coefficient: a number, a fraction, or n (2n, n+1).
_COEFFICIENT = re.compile(r"^(?:\d+(?:\.\d+)?|\d+/\d+|\d*n(?:\s*[+-]\s*\d+)?)$")


def _value(raw: Any, label: str, dropped: List[str]) -> str:
    """``raw`` as a clean string, or "" (recorded in ``dropped``) for a placeholder echo."""
    text = str(raw if raw is not None else "").strip()
    if text and _PLACEHOLDER.match(text):
        dropped.append(f"{label}={text!r} (placeholder echoed from the prompt template)")
        return ""
    return text


def _verbatim_in(value: str, span_ids: Sequence[str], allowed: Mapping[str, EvidenceSpan]) -> str:
    """The first of ``span_ids`` whose text contains ``value`` verbatim as a whole
    token: case, whitespace, Unicode hyphens and Greek letters folded, and a hyphen
    is part of the token ("JA" does not occur in "JA-Ile")."""
    needle = fold_text(value)
    if not needle:
        return ""
    pattern = re.compile(r"(?<![\w-])" + re.escape(needle) + r"(?![\w-])")
    for eid in span_ids:
        if eid in allowed and pattern.search(fold_text(allowed[eid].text)):
            return eid
    return ""


def _participants(rows: Any, allowed: Mapping[str, EvidenceSpan], *, label: str,
                  dropped: List[str]) -> List[ReactionParticipant]:
    """Every participant the model asserted, its own citation checked. Placeholder
    names are dropped here; :func:`_support_participants` decides the rest."""
    out: List[ReactionParticipant] = []
    seen = set()
    for index, row in enumerate(rows or []):
        if not isinstance(row, Mapping):
            row = {"name": str(row or "").strip()}
        name = _value(row.get("name"), f"{label}[{index}].name", dropped)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        stoichiometry = _value(row.get("stoichiometry"), f"{label}[{index}].stoichiometry", dropped)
        if stoichiometry and not _COEFFICIENT.match(stoichiometry):
            dropped.append(f"{label}[{index}].stoichiometry={stoichiometry!r} (not a coefficient)")
            stoichiometry = ""
        out.append(
            ReactionParticipant(
                name=name,
                declared_type=_value(row.get("entity_type"), f"{label}[{index}].entity_type", dropped),
                stoichiometry=stoichiometry,
                evidence_ids=_valid_ids(row.get("evidence_ids"), allowed),
            )
        )
    return out


def _support_participants(
    participants: Sequence[ReactionParticipant],
    cited: Sequence[str],
    allowed: Mapping[str, EvidenceSpan],
    *,
    side: str,
    label: str,
    dropped: List[str],
    unsupported: List[Dict[str, str]],
) -> List[ReactionParticipant]:
    """Keep a participant only on evidence (ORCH-745 F-5): a citation to a span the
    model was shown, or its name verbatim in a span the claim cites (any field's or
    the claim-level list). Pilot-003's thin answers cited nothing for any
    participant, and base kept every one."""
    kept: List[ReactionParticipant] = []
    for participant in participants:
        if participant.evidence_ids:
            participant.attribution = ATTRIBUTION_MODEL_CITATION
            kept.append(participant)
            continue
        span = _verbatim_in(participant.name, cited, allowed)
        if span:
            participant.evidence_ids = [span]
            participant.attribution = ATTRIBUTION_VERBATIM_CITED_SPAN
            kept.append(participant)
            continue
        reason = "no valid citation and not verbatim in a cited passage"
        dropped.append(f"{label}.{side}={participant.name!r} ({reason})")
        unsupported.append({"side": side, "name": participant.name, "reason": reason})
    return kept


@dataclass
class ExtractionResult:
    """Claims plus the record of the call that produced them."""

    claims: List[ExtractedClaim] = field(default_factory=list)
    outcome: Optional[ModelCallOutcome] = None
    dropped_fields: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.outcome.status if self.outcome else EXTRACTION_OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "claim_count": len(self.claims),
            "dropped_fields": list(self.dropped_fields),
            "notes": list(self.notes),
            "model_call": self.outcome.to_dict() if self.outcome else None,
        }


def build_claims(
    payload: Any,
    *,
    spans: Sequence[EvidenceSpan],
    paper_id: str,
    target_species_normalized: str,
    round_index: int,
    model: str,
    actual_model: str = "",
    model_call_id: str = "",
    prompt_version: str = PROMPT_VERSION,
    extraction_status: str = EXTRACTION_OK,
    paper_observed_organisms: Sequence[str] = (),
    paper_evidence_type: str = EVIDENCE_UNKNOWN,
    paper_evidence_type_verified: Optional[bool] = None,
) -> Tuple[List[ExtractedClaim], List[str]]:
    """Turn a validated model payload into claims. Returns ``(claims, dropped)``.

    ``dropped`` names every field and participant discarded for lack of a valid
    citation (or as a placeholder echo), so the round artifact can show what the
    model asserted and could not support.

    ``paper_evidence_type`` is the paper's grade (``review`` routes the species
    reading through the review rule) and ``paper_observed_organisms`` the
    organisms the screen read off the paper: context for the sentence rules only,
    never the claim's species (ORCH-745 B3). ``paper_evidence_type_verified`` is
    ``PaperRecord.evidence_type_verified``: an ``unknown`` grade after a failed lookup
    is read by the review rule too (:func:`t2pw.mining.models.reads_as_review`).
    """
    allowed = {s.evidence_id: s for s in spans}
    claims: List[ExtractedClaim] = []
    dropped: List[str] = []
    review = reads_as_review(paper_evidence_type, paper_evidence_type_verified)
    unverified = review and str(paper_evidence_type or "") != EVIDENCE_REVIEW

    rows = payload.get("claims") if isinstance(payload, Mapping) else payload
    for index, row in enumerate(rows or []):
        if not isinstance(row, Mapping):
            continue
        label = f"claim[{index}]"
        asserted_subs = _participants(row.get("substrates"), allowed, label=f"{label}.substrates", dropped=dropped)
        asserted_prods = _participants(row.get("products"), allowed, label=f"{label}.products", dropped=dropped)
        if not asserted_subs and not asserted_prods:
            continue

        claim_evidence = _valid_ids(row.get("evidence_ids"), allowed)
        for participant in (*asserted_subs, *asserted_prods):
            for eid in participant.evidence_ids:
                if eid not in claim_evidence:
                    claim_evidence.append(eid)

        field_evidence: Dict[str, List[str]] = {}

        def cited(key: str, field_name: str) -> List[str]:
            ids = _valid_ids(row.get(key), allowed)
            if ids:
                field_evidence[field_name] = ids
                for eid in ids:
                    if eid not in claim_evidence:
                        claim_evidence.append(eid)
            return ids

        enzyme_ids = cited("enzyme_evidence_ids", "enzyme")
        gene_ids = cited("gene_evidence_ids", "gene")
        direction_ids = cited("direction_evidence_ids", "direction")
        reversible_ids = cited("reversible_evidence_ids", "reversible")
        compartment_ids = cited("compartment_evidence_ids", "compartment")
        species_ids = cited("species_evidence_ids", "species")

        unsupported: List[Dict[str, str]] = []
        substrates = _support_participants(asserted_subs, claim_evidence, allowed, side="substrates",
                                           label=label, dropped=dropped, unsupported=unsupported)
        products = _support_participants(asserted_prods, claim_evidence, allowed, side="products",
                                         label=label, dropped=dropped, unsupported=unsupported)

        if substrates:
            sub_ids = [e for p in substrates for e in p.evidence_ids]
            if sub_ids:
                field_evidence["substrates"] = list(dict.fromkeys(sub_ids))
        if products:
            prod_ids = [e for p in products for e in p.evidence_ids]
            if prod_ids:
                field_evidence["products"] = list(dict.fromkeys(prod_ids))

        # Every kept field names how it got its evidence.
        attribution: Dict[str, str] = {}
        # The field fallback reads only spans the model cited for a participant; a
        # participant kept by its own verbatim match does not widen it.
        participant_ids = list(dict.fromkeys(
            e for p in (*substrates, *products) if p.attribution == ATTRIBUTION_MODEL_CITATION
            for e in p.evidence_ids))

        def keep(value: Any, ids: List[str], name: str) -> Tuple[str, List[str]]:
            text = _value(value, f"{label}.{name}", dropped)
            if not text:
                return "", ids
            if ids:
                attribution[name] = ATTRIBUTION_MODEL_CITATION
                return text, ids
            # The model wrote no citation list for this field at all (pilot-001's
            # failure). A name that occurs verbatim in one of THIS claim's own
            # cited participant spans is attributed to that span. An explicit
            # empty list, or one citing only unshown ids, is the model declining
            # to cite and is respected. Anything else is still dropped.
            recoverable = name in _VERBATIM_RECOVERABLE and row.get(f"{name}_evidence_ids") is None
            span = _verbatim_span(text, participant_ids, allowed) if recoverable else ""
            if span:
                field_evidence[name] = [span]
                attribution[name] = ATTRIBUTION_VERBATIM_CITED_SPAN
                return text, [span]
            dropped.append(f"{label}.{name}={text!r} (no valid citation)")
            return "", ids

        enzyme, enzyme_ids = keep(row.get("enzyme"), enzyme_ids, "enzyme")
        gene, gene_ids = keep(row.get("gene"), gene_ids, "gene")
        compartment, compartment_ids = keep(row.get("compartment"), compartment_ids, "compartment")
        species_text, species_ids = keep(row.get("species"), species_ids, "species")

        # Tissue has no citation list of its own: it is kept only where a passage the
        # claim cites says it (F-5). Pilot-003 stored "leaf, root, inflorescence,
        # wounded leaf" from passages that say nothing of the kind.
        tissue = _value(row.get("tissue"), f"{label}.tissue", dropped)
        if tissue:
            span = _verbatim_in(tissue, claim_evidence, allowed)
            if span:
                field_evidence["tissue"] = [span]
                attribution["tissue"] = ATTRIBUTION_VERBATIM_CITED_SPAN
            else:
                dropped.append(f"{label}.tissue={tissue!r} (not verbatim in a cited passage)")
                tissue = ""

        direction = _value(row.get("direction"), f"{label}.direction", dropped) or DIRECTION_UNSPECIFIED
        if direction != DIRECTION_UNSPECIFIED and not direction_ids:
            dropped.append(f"{label}.direction={direction!r} (no valid citation)")
            direction = DIRECTION_UNSPECIFIED
        elif direction != DIRECTION_UNSPECIFIED:
            attribution["direction"] = ATTRIBUTION_MODEL_CITATION

        reversible = row.get("reversible")
        if reversible is not None and not reversible_ids:
            dropped.append(f"{label}.reversible={reversible!r} (no valid citation)")
            reversible = None
        if reversible is not None:
            reversible = bool(reversible)
            attribution["reversible"] = ATTRIBUTION_MODEL_CITATION

        # A species kept only by the verbatim fallback is a name that occurs in a
        # cited passage, not evidence that the passage ties it to THIS reaction
        # ("expression of bryophyte genes in transgenic Arabidopsis", "equivalent
        # to Arabidopsis AtFAD3"). It is still the observed value, so a non-target
        # species is recognised and quarantines the claim; but it never counts as
        # species evidence, so the target species fails closed to
        # ``species_ambiguous`` and stays a lead.
        if attribution.get("species") == ATTRIBUTION_VERBATIM_CITED_SPAN:
            species_ids = []
        # F-1: read the sentences of the claim's own cited passages, not the whole
        # passage. B3: the species is what the model stated for THIS claim; the
        # paper's screen organisms are context for the sentence rules only (base
        # made "human LOX homologs" in an abstract a second species of every claim).
        claimed_inferred = bool(row.get("species_inferred"))
        species_span_ok = species_span_names_target(
            [allowed[e] for e in species_ids], target_species_normalized)
        sentences = grade_sentences(
            [allowed[e] for e in claim_evidence],
            target_species_normalized=target_species_normalized,
            substrates=[p.name for p in substrates],
            products=[p.name for p in products],
            enzyme=enzyme,
            gene=gene,
            asserted=[*(p.name for p in (*asserted_subs, *asserted_prods)), *_catalyst_values(row)],
            review=review,
            paper_observed_organisms=paper_observed_organisms,
        )
        verdict = aggregate_species(
            target_species_normalized=target_species_normalized,
            observed=[species_text] if species_text else [],
            species_evidence_ids=species_ids,
            sentences=sentences,
            species_span_ok=species_span_ok,
            claimed_inferred=claimed_inferred,
        )

        # Species is part of the reaction identity, so the same chemistry in two
        # species stays two claims. An ambiguous claim about the target species
        # shares the target's identity on purpose: that is what lets a later
        # round with real species evidence RECONSIDER it instead of creating a
        # second, disconnected copy. A claim with no species at all gets its own
        # namespace so it can never silently merge into the target graph, and so
        # does a claim none of whose participants is supported (F-5): it is a lead
        # for the frontier, never a reaction, and must not reach a supported identity.
        no_participants = not substrates and not products
        if no_participants:
            identity_species = UNSUPPORTED_PARTICIPANTS_NAMESPACE
        elif verdict.admissible:
            identity_species = target_species_normalized
        elif verdict.observed_normalized:
            identity_species = verdict.observed_normalized[0]
        else:
            identity_species = f"unspecified:{verdict.state}"
        named_subs = asserted_subs if no_participants else substrates
        named_prods = asserted_prods if no_participants else products
        identity = reaction_identity(
            [p.name for p in named_subs],
            [p.name for p in named_prods],
            species=identity_species,
        )
        claim = ExtractedClaim(
            claim_id=make_claim_id(identity, paper_id, round_index),
            claim_identity=identity,
            paper_id=paper_id,
            subject=" + ".join(p.name for p in named_subs),
            relationship="converted_to",
            object=" + ".join(p.name for p in named_prods),
            species_text=species_text,
            species_state=verdict.state,
            species_reason=(f"{UNVERIFIED_EVIDENCE_TYPE_NOTE}; read by the review rule. {verdict.reason}"
                            if unverified else verdict.reason),
            substrates=substrates,
            products=products,
            enzyme=enzyme,
            gene=gene,
            direction=direction,
            reversible=reversible,
            compartment=compartment,
            conditions=_value(row.get("conditions"), f"{label}.conditions", dropped),
            tissue=tissue,
            pathway_names=[v for v in (_value(p, f"{label}.pathway_names", dropped)
                                       for p in row.get("pathway_names") or []) if v],
            field_evidence=field_evidence,
            field_attribution=attribution,
            evidence_ids=claim_evidence,
            species_support={
                "grade": verdict.grade,
                "sentences": verdict.sentences,
                "species_evidence_ids": list(species_ids),
                "species_span_ok": species_span_ok,
                "claimed_inferred": claimed_inferred,
                "target_species": target_species_normalized,
                "paper_evidence_type": str(paper_evidence_type or ""),
                "paper_evidence_type_verified": paper_evidence_type_verified,
                **({"gate_evidence_type": EVIDENCE_REVIEW, "reason": UNVERIFIED_EVIDENCE_TYPE_NOTE}
                   if unverified else {}),
            },
            unsupported_participants=unsupported,
            model=model,
            actual_model=actual_model or model,
            prompt_version=prompt_version,
            model_call_id=model_call_id,
            extraction_status=extraction_status,
            hypothesis=bool(row.get("hypothesis")),
            first_seen_round=int(round_index),
            last_updated_round=int(round_index),
            confidence=_value(row.get("confidence"), f"{label}.confidence", dropped),
        )
        claims.append(claim)
    return claims, dropped


def _catalyst_values(row: Mapping[str, Any]) -> List[str]:
    """The enzyme and gene the model asserted, cited or not: their sentences'
    negative markers count even when the value itself is dropped."""
    return [str(row.get(k) or "").strip() for k in ("enzyme", "gene") if str(row.get(k) or "").strip()]


def extract_claims(
    *,
    client: MiningModelClient,
    spans: Sequence[EvidenceSpan],
    paper_id: str,
    paper_title: str,
    target_species: str,
    target_species_normalized: str,
    round_index: int,
    topic: str = "",
    paper_observed_organisms: Sequence[str] = (),
    paper_evidence_type: str = EVIDENCE_UNKNOWN,
    paper_evidence_type_verified: Optional[bool] = None,
) -> ExtractionResult:
    """Run one extraction call over ``spans`` and build the resulting claims.

    A failed call returns an :class:`ExtractionResult` with **no claims** and a
    non-OK status. It never returns a repaired fragment as a claim.
    """
    if not spans:
        return ExtractionResult(notes=["no passages supplied"])

    context = [
        f"TARGET SPECIES (declared by the run): {target_species}",
    ]
    if topic:
        context.append(f"TOPIC OF INTEREST: {topic}")
    context.append(
        "Do not assume the passages are about the target species. Report the species the "
        "passages actually state; leave it empty if they do not state one."
    )
    user = "\n".join(context) + "\n\n" + render_passages(spans, paper_title=paper_title)

    outcome = client.call_json(
        purpose="extract_claims",
        system=SYSTEM_PROMPT,
        user=user,
        schema=CLAIM_SCHEMA,
        tight_schema=TIGHT_CLAIM_SCHEMA,
        round_index=round_index,
        paper_id=paper_id,
        evidence_ids=[s.evidence_id for s in spans],
    )

    if not outcome.ok:
        return ExtractionResult(
            claims=[],
            outcome=outcome,
            notes=[
                f"extraction produced no admissible claims: status={outcome.status}. "
                "No reaction was added; the failure is recorded and the passages remain "
                "available for a later round."
            ],
        )

    claims, dropped = build_claims(
        outcome.payload,
        spans=spans,
        paper_id=paper_id,
        target_species_normalized=target_species_normalized,
        round_index=round_index,
        model=client.config.model,
        actual_model=outcome.record.actual_model,
        model_call_id=outcome.record.call_id,
        prompt_version=client.config.prompt_version,
        extraction_status=outcome.status,
        paper_observed_organisms=paper_observed_organisms,
        paper_evidence_type=paper_evidence_type,
        paper_evidence_type_verified=paper_evidence_type_verified,
    )
    notes = []
    if dropped:
        notes.append(f"{len(dropped)} asserted field(s) or participant(s) dropped: no valid citation, "
                     "or a placeholder echoed from the prompt")
    recovered = sum(
        1 for c in claims for how in c.field_attribution.values()
        if how == ATTRIBUTION_VERBATIM_CITED_SPAN
    )
    if recovered:
        notes.append(
            f"{recovered} uncited field(s) kept: the value occurs verbatim in one of the claim's "
            f"own cited spans ({ATTRIBUTION_VERBATIM_CITED_SPAN})"
        )
    participants = sum(
        1 for c in claims for p in (*c.substrates, *c.products)
        if p.attribution == ATTRIBUTION_VERBATIM_CITED_SPAN
    )
    if participants:
        notes.append(
            f"{participants} uncited participant(s) kept: the name occurs verbatim in a span the "
            f"claim cites ({ATTRIBUTION_VERBATIM_CITED_SPAN})"
        )
    return ExtractionResult(claims=claims, outcome=outcome, dropped_fields=dropped, notes=notes)


__all__ = [
    "SYSTEM_PROMPT",
    "CLAIM_SCHEMA",
    "TIGHT_CLAIM_SCHEMA",
    "ExtractionResult",
    "render_passages",
    "build_claims",
    "extract_claims",
]
