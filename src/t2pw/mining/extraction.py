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
    EXTRACTION_OK,
    EvidenceSpan,
    ExtractedClaim,
    ReactionParticipant,
    make_claim_id,
    reaction_identity,
)
from t2pw.mining.modelclient import MiningModelClient, ModelCallOutcome
from t2pw.mining.species import classify_species

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
     "species_inferred": false,
     "direction": "forward", "direction_evidence_ids": ["ev_..."],
     "compartment": "...", "compartment_evidence_ids": ["ev_..."],
     "hypothesis": false,
     "evidence_ids": ["ev_..."]}]}
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


def _participants(rows: Any, allowed: Mapping[str, EvidenceSpan]) -> List[ReactionParticipant]:
    out: List[ReactionParticipant] = []
    seen = set()
    for row in rows or []:
        if not isinstance(row, Mapping):
            name = str(row or "").strip()
            row = {"name": name}
        name = str(row.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(
            ReactionParticipant(
                name=name,
                declared_type=str(row.get("entity_type") or "").strip(),
                stoichiometry=str(row.get("stoichiometry") or "").strip(),
                evidence_ids=_valid_ids(row.get("evidence_ids"), allowed),
            )
        )
    return out


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
) -> Tuple[List[ExtractedClaim], List[str]]:
    """Turn a validated model payload into claims. Returns ``(claims, dropped)``.

    ``dropped`` names every field discarded for lack of a valid citation, so the
    round artifact can show what the model asserted and could not support.
    """
    allowed = {s.evidence_id: s for s in spans}
    claims: List[ExtractedClaim] = []
    dropped: List[str] = []

    rows = payload.get("claims") if isinstance(payload, Mapping) else payload
    for index, row in enumerate(rows or []):
        if not isinstance(row, Mapping):
            continue
        substrates = _participants(row.get("substrates"), allowed)
        products = _participants(row.get("products"), allowed)
        if not substrates and not products:
            continue

        claim_evidence = _valid_ids(row.get("evidence_ids"), allowed)
        for participant in (*substrates, *products):
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
        participant_ids = list(dict.fromkeys(e for p in (*substrates, *products) for e in p.evidence_ids))

        def keep(value: Any, ids: List[str], label: str) -> Tuple[str, List[str]]:
            text = str(value or "").strip()
            if not text:
                return "", ids
            if ids:
                attribution[label] = ATTRIBUTION_MODEL_CITATION
                return text, ids
            # The model wrote no citation list for this field at all (pilot-001's
            # failure). A name that occurs verbatim in one of THIS claim's own
            # cited participant spans is attributed to that span. An explicit
            # empty list, or one citing only unshown ids, is the model declining
            # to cite and is respected. Anything else is still dropped.
            recoverable = label in _VERBATIM_RECOVERABLE and row.get(f"{label}_evidence_ids") is None
            span = _verbatim_span(text, participant_ids, allowed) if recoverable else ""
            if span:
                field_evidence[label] = [span]
                attribution[label] = ATTRIBUTION_VERBATIM_CITED_SPAN
                return text, [span]
            dropped.append(f"claim[{index}].{label}={text!r} (no valid citation)")
            return "", ids

        enzyme, enzyme_ids = keep(row.get("enzyme"), enzyme_ids, "enzyme")
        gene, gene_ids = keep(row.get("gene"), gene_ids, "gene")
        compartment, compartment_ids = keep(row.get("compartment"), compartment_ids, "compartment")
        species_text, species_ids = keep(row.get("species"), species_ids, "species")

        direction = str(row.get("direction") or DIRECTION_UNSPECIFIED).strip() or DIRECTION_UNSPECIFIED
        if direction != DIRECTION_UNSPECIFIED and not direction_ids:
            dropped.append(f"claim[{index}].direction={direction!r} (no valid citation)")
            direction = DIRECTION_UNSPECIFIED
        elif direction != DIRECTION_UNSPECIFIED:
            attribution["direction"] = ATTRIBUTION_MODEL_CITATION

        reversible = row.get("reversible")
        if reversible is not None and not reversible_ids:
            dropped.append(f"claim[{index}].reversible={reversible!r} (no valid citation)")
            reversible = None
        if reversible is not None:
            reversible = bool(reversible)
            attribution["reversible"] = ATTRIBUTION_MODEL_CITATION

        evidence_text = "\n".join(allowed[e].text for e in claim_evidence if e in allowed)
        # When the claim cites no species evidence of its own, the paper's own
        # observed organisms are still a fact about the SOURCE -- but a fact
        # about the paper, not about this reaction. Feeding them in uncited is
        # exactly what yields `species_ambiguous`: a species is named somewhere
        # and nothing ties it to this claim.
        observed = (
            [species_text]
            if species_text
            else [str(o) for o in paper_observed_organisms if str(o or "").strip()]
        )
        # A species kept only by the verbatim fallback is a name that occurs in a
        # cited passage, not evidence that the passage ties it to THIS reaction
        # ("expression of bryophyte genes in transgenic Arabidopsis", "equivalent
        # to Arabidopsis AtFAD3"). It is still the observed value, so a non-target
        # or second species is recognised and quarantines the claim; but it never
        # counts as species evidence, so the target species fails closed to
        # ``species_ambiguous`` and stays a lead.
        if attribution.get("species") == ATTRIBUTION_VERBATIM_CITED_SPAN:
            species_ids = []
        verdict = classify_species(
            target_species_normalized=target_species_normalized,
            observed=observed,
            species_evidence_ids=species_ids,
            evidence_text=evidence_text,
            claimed_inferred=bool(row.get("species_inferred")),
        )

        # Species is part of the reaction identity, so the same chemistry in two
        # species stays two claims. An ambiguous claim about the target species
        # shares the target's identity on purpose: that is what lets a later
        # round with real species evidence RECONSIDER it instead of creating a
        # second, disconnected copy. A claim with no species at all gets its own
        # namespace so it can never silently merge into the target graph.
        if verdict.admissible:
            identity_species = target_species_normalized
        elif verdict.observed_normalized:
            identity_species = verdict.observed_normalized[0]
        else:
            identity_species = f"unspecified:{verdict.state}"
        identity = reaction_identity(
            [p.name for p in substrates],
            [p.name for p in products],
            species=identity_species,
        )
        claim = ExtractedClaim(
            claim_id=make_claim_id(identity, paper_id, round_index),
            claim_identity=identity,
            paper_id=paper_id,
            subject=" + ".join(p.name for p in substrates),
            relationship="converted_to",
            object=" + ".join(p.name for p in products),
            species_text=species_text,
            species_state=verdict.state,
            species_reason=verdict.reason,
            substrates=substrates,
            products=products,
            enzyme=enzyme,
            gene=gene,
            direction=direction,
            reversible=reversible,
            compartment=compartment,
            conditions=str(row.get("conditions") or "").strip(),
            tissue=str(row.get("tissue") or "").strip(),
            pathway_names=[str(p).strip() for p in row.get("pathway_names") or [] if str(p).strip()],
            field_evidence=field_evidence,
            field_attribution=attribution,
            evidence_ids=claim_evidence,
            model=model,
            actual_model=actual_model or model,
            prompt_version=prompt_version,
            model_call_id=model_call_id,
            extraction_status=extraction_status,
            hypothesis=bool(row.get("hypothesis")),
            first_seen_round=int(round_index),
            last_updated_round=int(round_index),
            confidence=str(row.get("confidence") or "").strip(),
        )
        claims.append(claim)
    return claims, dropped


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
    )
    notes = []
    if dropped:
        notes.append(f"{len(dropped)} asserted field(s) dropped for lack of a valid citation")
    recovered = sum(
        1 for c in claims for how in c.field_attribution.values()
        if how == ATTRIBUTION_VERBATIM_CITED_SPAN
    )
    if recovered:
        notes.append(
            f"{recovered} uncited field(s) kept: the value occurs verbatim in one of the claim's "
            f"own cited participant spans ({ATTRIBUTION_VERBATIM_CITED_SPAN})"
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
