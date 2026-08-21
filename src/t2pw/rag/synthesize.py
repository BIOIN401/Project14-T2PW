"""Stage R5 — evidence-bound pathway synthesis (WP5).

Merge the seed extraction plus the per-gap evidence bundles produced by WP4 into
**one connected pathway** and emit it as a **standard** ``Payload`` (the
``TypedDict`` shapes in :mod:`t2pw.schema`) at seam **S3** — the exact shape Stage
2B already consumes, plus only the *optional, additive* provenance keys WP0
defined in :mod:`t2pw.rag.provenance` (``rag_provenance`` / ``evidence`` /
``source_papers`` / ``rag_confidence``) which every stage that does not know about
them ignores.

What synthesis does (docs/rag/agents/wp5_synthesis.md):

1. **Stitch** — connect reactions end-to-end so a product feeds the next
   reaction's input across papers. A dangling end is closed *only* where a
   retrieved evidence reaction supplies the missing substrate/product; nothing is
   fabricated.
2. **Reconcile synonyms** — unify cross-paper names by canonicalizing through the
   core's ``BIOCHEMICAL_ALIAS_MAP`` (imported **read-only**; this module is never
   imported *by* ``process_normalizer``).
3. **Resolve conflicts** — when papers disagree on stoichiometry / compartment /
   reversible flag for the *same-direction* reaction, pick the variant with the
   greater evidence weight and record the alternatives in the returned report.
   Opposite directions (a forward/reverse pair) are *not* a conflict — they key to
   distinct reactions and both survive, matching the core pipeline.
4. **Attach provenance** — every synthesized reaction, and every non-cofactor
   entity, carries at least one provenance pointer keyed to ``source_id`` /
   ``source_uri``. An element with no supporting evidence is **omitted** and
   surfaced in the unresolved-gaps report — never invented (the no-invented-
   chemistry guardrail; WP6 hardens enforcement, WP5 already obeys it).

Separation invariant (docs/rag/03_separation_invariant.md)
----------------------------------------------------------
All of this lives in ``t2pw.rag``; the dependency arrow points **RAG -> core
only**. The output is validated against the core structural contract
(``validate_post_extraction``) before it is returned, but no stage-module file is
edited and no RAG-only *required* key is added to the payload — the four
provenance keys are optional/additive.

Offline / determinism
---------------------
Import and execution require no chromadb, no network, and no LLM. WP4's
``EvidenceBundle`` / ``Gap`` and the store ``Chunk`` / ``Retrieved`` are consumed
by *duck typing* (only their attributes are read) so importing this module does
not pull the retrieval/ingest stack. Synthesis is a pure, deterministic function
of its inputs.
"""

from __future__ import annotations

import copy
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from t2pw.curation.gap_resolver import _ensure_biological_state
from t2pw.pipeline.entity_identity import (
    PATHBANK_UNKNOWN_PROTEIN_ID,
    has_protein_external_identity,
    is_pathbank_unknown_protein,
)
from t2pw.pipeline.lineage import (
    LINEAGE_KEY,
    LineageEntry,
    LineageError,
    LineageSource,
    read as read_lineage,
    record as record_lineage,
)
from t2pw.pipeline.process_normalizer import BIOCHEMICAL_ALIAS_MAP
from t2pw.pipeline.stage_contracts import validate_post_extraction
from t2pw.rag.admission import (
    GAP_DANGLING_REACTION,
    GAP_MISSING_COMPARTMENT,
    GAP_MISSING_PRECURSOR,
    GAP_ORPHAN_METABOLITE,
    GAP_UNMAPPED_ENZYME,
    REASON_CONFLICTING_RESOLUTION,
    REASON_MULTI_PARTICIPANT_REPAIR,
    REASON_SIDE_NO_LONGER_MISSING,
    REASON_UNSUPPORTED_TARGET_TYPE,
    RESOLUTION_COMPARTMENT,
    RESOLUTION_IDENTIFIER,
    RESOLUTION_PRECURSOR,
    ROUTE_IDENTITY_RESOLVER,
    STATUS_APPLIED,
    STATUS_PROPOSED,
    STATUS_REJECTED_PROPOSAL,
    AdmissionPolicy,
    AdmissionReport,
    RagReactionCandidate,
    admission_lineage_entry,
    admit_candidates,
    compare_organism,
    compare_requested_pathway,
    locate_span,
    missing_reaction_side,
    organisms_in_span,
)
from t2pw.rag.provenance import RAG_ADDITIVE_KEYS

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps imports offline-light
    from t2pw.rag.retrieve import EvidenceBundle, Gap
    from t2pw.schema import Payload
else:  # runtime aliases so annotations don't require the heavy imports
    EvidenceBundle = Any
    Gap = Any
    Payload = Dict[str, Any]


# Ubiquitous cofactors / small molecules that are exempt from the mandatory
# provenance requirement (they appear in nearly every reaction and carry no
# pathway-specific evidential weight). Keys are ``.casefold()``.
COFACTOR_NAMES = frozenset(
    {
        "atp",
        "adp",
        "amp",
        "gtp",
        "gdp",
        "gmp",
        "utp",
        "udp",
        "ctp",
        "cdp",
        "nad+",
        "nadh",
        "nadp+",
        "nadph",
        "fad",
        "fadh2",
        "coa-sh",
        "pi",
        "ppi",
        "h2o",
        "water",
        "h+",
        "proton",
        "o2",
        "oxygen",
        "co2",
        "nh3",
        "nh4+",
        "h2o2",
    }
)

# Arrow tokens recognised in an evidence reaction equation. Reversible variants
# are listed so a direction conflict can be detected.
_ARROWS = ("<=>", "<->", "⇌", "=>", "->", "→", "⇒")
_REVERSIBLE_ARROWS = frozenset({"<=>", "<->", "⇌"})

_COEFF_RE = re.compile(r"^(\d+(?:\.\d+)?)\s+(.+)$")
_PLUS_SPLIT_RE = re.compile(r"\s\+\s")
# Participant sides are split on both the chemistry-standard " + " and the
# ingest metadata-bag separator ";". A bare "+" without surrounding spaces (a
# charge notation like ``NAD+`` / ``H+``) is deliberately NOT a split point.
_SIDE_SPLIT_RE = re.compile(r"\s\+\s|\s*;\s*")

# Source types whose chunk ``text`` is a genuine prose/equation passage that may
# be transcribed into reactions. Everything else — ``pwml_example`` corpus
# scaffolding and ``pathbank`` / ``kegg`` DB records — is a " ; "-joined
# metadata *bag* built for lexical scoring, never a clean single equation, so it
# is NEVER parsed into a reaction (its arrow is incidental).
_PARSEABLE_SOURCE_TYPES = frozenset({"paper", ""})

# A token that is clearly not a single chemical species. Used to reject the
# pathway-metadata garbage that a " ; "-joined blob collapses into when it is
# (defensively) parsed: a pathway id, a biological-state descriptor, or an
# absurdly long "name".
_PATHWAY_TOKEN_RE = re.compile(r"^pathway\d", re.IGNORECASE)
_BIO_STATE_RE = re.compile(
    r",\s*(cell membrane|cell|extracellular)\b", re.IGNORECASE
)
_MAX_SPECIES_WORDS = 12
_MAX_SPECIES_CHARS = 120


def _is_invalid_species_token(name: str) -> bool:
    """True when ``name`` cannot be a single chemical species / enzyme.

    Guards the (defensive) reaction parser against " ; "-joined pathway
    metadata: a token that names a whole pathway, carries a biological-state
    descriptor (``, Cell,``), or is far too long to be one species is rejected.
    """
    text = str(name or "").strip()
    if not text:
        return True
    if _PATHWAY_TOKEN_RE.match(text):
        return True
    if _BIO_STATE_RE.search(text):
        return True
    if len(text) > _MAX_SPECIES_CHARS:
        return True
    if len(text.split()) > _MAX_SPECIES_WORDS:
        return True
    return False


# ---------------------------------------------------------------------------
# Public result container.
# ---------------------------------------------------------------------------
@dataclass
class SynthesisResult:
    """The synthesized payload plus the reports that ride *alongside* it.

    ``payload`` is a standard :class:`~t2pw.schema.Payload` (seam S3).
    ``unresolved_gaps`` lists every gap that stayed unfilled because no evidence
    supported it (nothing was invented for these). ``conflicts`` records the
    alternatives that lost an evidence-weight decision. ``stitched`` records the
    cross-paper connections that were closed by evidence. ``contract_report`` is
    the ``validate_post_extraction`` report proving the payload passed the core
    structural contract.
    """

    payload: "Payload"
    unresolved_gaps: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    stitched: List[Dict[str, Any]] = field(default_factory=list)
    contract_report: Dict[str, Any] = field(default_factory=dict)
    #: The bounded :class:`~t2pw.rag.admission.AdmissionReport` for this run,
    #: holding every retrieved candidate that was accepted AND every one that was
    #: rejected, with the reasons. Empty when there were no evidence bundles.
    admission: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal reaction / participant model.
# ---------------------------------------------------------------------------
@dataclass
class _Participant:
    name: str  # canonical display name
    stoichiometry: Optional[float] = None


@dataclass
class _Reaction:
    name: str
    inputs: List[_Participant]
    outputs: List[_Participant]
    enzymes: List[str] = field(default_factory=list)
    reversible: bool = False
    compartment: str = ""
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    source_papers: List[Dict[str, Any]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    #: ``"seed"`` for a reaction the uploaded paper's own extraction produced,
    #: ``"rag"`` for one transcribed from a retrieved passage. Kept because the
    #: two are governed by different rules — a seed reaction is the pathway being
    #: extended and is never subject to gap admission, a RAG one only enters by
    #: passing it — and because a reader of the merged payload has to be able to
    #: tell a seed claim from a cross-paper one.
    origin: str = "rag"
    #: The gap this reaction was retrieved to fill ("" for a seed reaction).
    gap_id: str = ""
    #: Every gap this same claim was retrieved for. One passage is routinely
    #: top-k for several gaps, and when those duplicates merge into one row the
    #: row genuinely fills all of them — dropping the extra attributions would
    #: leave gaps looking unfilled that a delivered reaction actually closed.
    gap_ids: List[str] = field(default_factory=list)
    #: Stage-1 scope taxonomy label (core | anaplerotic | cataplerotic |
    #: auxiliary | out_of_scope). Carried from the seed row, or written by the
    #: admission gate for a RAG row.
    scope_membership: str = ""
    #: The organism the evidence itself reports ("" when it does not say).
    organism: str = ""
    #: The EXACT span of the parent chunk that states this reaction — the parsed
    #: source line for an arrow equation, the validated quote/sentence for a
    #: prose extraction. The parent chunk pointer lives in ``provenance`` /
    #: ``evidence`` and is never replaced by this.
    evidence_span: str = ""
    #: What the source PAPER was observed to be about (eligibility screen),
    #: carried down from the chunk. Never the requested values.
    observed_organisms: List[str] = field(default_factory=list)
    observed_pathways: List[str] = field(default_factory=list)
    #: Serialized :class:`~t2pw.pipeline.lineage.LineageEntry` records this
    #: reaction carries into row emission (C-035): whatever a seed row already
    #: held, plus the admission gate's verdict for a RAG import. Attribution is
    #: POSITIONAL -- it belongs to the row this reaction becomes -- and
    #: :func:`_reaction_row` builds that row from scratch, so without a carrier
    #: here an inbound attribution would be silently erased at every synthesis.
    #: A lineage is append-only, which is why :func:`_merge_into` unions this
    #: rather than letting the surviving row keep only its own.
    lineage: List[Dict[str, Any]] = field(default_factory=list)

    # --- derived helpers -------------------------------------------------
    def input_names(self) -> List[str]:
        return [p.name for p in self.inputs]

    def output_names(self) -> List[str]:
        return [p.name for p in self.outputs]

    def participant_names(self) -> List[str]:
        return self.input_names() + self.output_names()

    def source_ids(self) -> List[str]:
        ids = [str(p.get("source_id") or "") for p in self.provenance]
        return [i for i in ids if i]

    def weight(self) -> float:
        """Evidence weight: summed retrieval scores, else provenance count."""
        total = sum(float(s) for s in self.scores if _is_number(s))
        if total > 0.0:
            return total
        return float(len(self.provenance))

    def conflict_key(self, resolver: Optional[Any] = None) -> Tuple[Any, ...]:
        """Direction-aware reaction identity: same inputs->outputs is the same reaction.

        Keyed on the (sorted input names, sorted output names) PAIR, not one merged
        participant set, so a reversible reaction represented as an explicit forward
        and reverse pair (identical participants, inputs/outputs swapped) keys to two
        distinct groups and BOTH survive conflict resolution. This matches the core
        ``dedupe_processes`` reaction key (the pre-RAG single-paper behavior); the old
        merged-set key collapsed the pair into one group and silently dropped a
        direction — deleting a locked reaction and tripping the locked-reaction
        accounting gate.

        ``resolver`` (optional, GROUPING-ONLY) maps each participant name to a
        synonym-canonical token so cross-paper duplicates that differ only by a
        compound/enzyme SYNONYM key to the same group and merge. When it is ``None``
        (the default) the key is byte-identical to the pre-resolver behavior (plain
        ``casefold``) — the emitted rows always keep their original names regardless.
        """
        if resolver is None:
            ins = tuple(sorted(n.casefold() for n in self.input_names()))
            outs = tuple(sorted(n.casefold() for n in self.output_names()))
            return (ins, outs)
        ins = tuple(sorted(resolver(n) for n in self.input_names()))
        outs = tuple(sorted(resolver(n) for n in self.output_names()))
        return (ins, outs)

    def signature(self, resolver: Optional[Any] = None) -> Tuple[Any, ...]:
        """Distinguishes conflicting *variants* of the same reaction.

        ``resolver`` (optional, GROUPING-ONLY) canonicalizes participant names to
        synonym tokens exactly as :meth:`conflict_key` does. When ``None`` the
        signature is byte-identical to the pre-resolver behavior. When active, the
        sort is made robust to two synonyms collapsing to the SAME token within one
        reaction (a token tie would otherwise force an unorderable ``None`` vs float
        stoichiometry comparison) while preserving the (token, stoichiometry)
        equality semantics used to detect variant disagreements.
        """
        if resolver is None:
            ins = tuple(sorted((p.name.casefold(), p.stoichiometry) for p in self.inputs))
            outs = tuple(
                sorted((p.name.casefold(), p.stoichiometry) for p in self.outputs)
            )
            return (ins, outs, self.reversible, self.compartment.casefold())

        def _sort_key(pair: Tuple[str, Optional[float]]) -> Tuple[str, bool, float]:
            token, stoich = pair
            return (token, stoich is None, 0.0 if stoich is None else float(stoich))

        ins = tuple(
            sorted(((resolver(p.name), p.stoichiometry) for p in self.inputs), key=_sort_key)
        )
        outs = tuple(
            sorted(((resolver(p.name), p.stoichiometry) for p in self.outputs), key=_sort_key)
        )
        return (ins, outs, self.reversible, self.compartment.casefold())


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _text(value: Any) -> str:
    return str(value or "").strip()


def canonical_name(name: Any) -> str:
    """Collapse whitespace then map known synonyms via ``BIOCHEMICAL_ALIAS_MAP``.

    Read-only reuse of the core alias map (docs/rag/agents/wp5_synthesis.md step
    2): this is the *same* casefold-keyed lookup ``process_normalizer`` performs,
    reproduced here so cross-paper synonyms collapse to one canonical node —
    without importing ``t2pw.rag`` into ``process_normalizer``.
    """
    cleaned = re.sub(r"\s+", " ", str(name or "").strip())
    if not cleaned:
        return ""
    return BIOCHEMICAL_ALIAS_MAP.get(cleaned.casefold(), cleaned)


def _is_cofactor(canonical: str) -> bool:
    return canonical.casefold() in COFACTOR_NAMES


# ---------------------------------------------------------------------------
# Provenance construction from a retrieved chunk / seed source descriptor.
# ---------------------------------------------------------------------------
def _chunk_of(hit: Any) -> Any:
    return getattr(hit, "chunk", None)


def _provenance_from_chunk(chunk: Any) -> Dict[str, Any]:
    """Build a :class:`~t2pw.rag.provenance.RagProvenance`-shaped pointer."""
    return {
        "source_id": _text(getattr(chunk, "source_id", "")) or _text(getattr(chunk, "id", "")),
        "source_title": _text(getattr(chunk, "source_title", "")),
        "source_type": _text(getattr(chunk, "source_type", "")),
        "source_uri": _text(getattr(chunk, "source_uri", "")),
        "section": _text(getattr(chunk, "section", "")),
        "organism": _text(getattr(chunk, "organism", "")),
        "chunk_id": _text(getattr(chunk, "id", "")),
    }


def _evidence_from_hit(hit: Any) -> Dict[str, Any]:
    """Build a :class:`~t2pw.rag.provenance.RagEvidence`-shaped record."""
    chunk = _chunk_of(hit)
    return {
        "text": _text(getattr(chunk, "text", "")),
        "source_id": _text(getattr(chunk, "source_id", "")) or _text(getattr(chunk, "id", "")),
        "source_uri": _text(getattr(chunk, "source_uri", "")),
        "section": _text(getattr(chunk, "section", "")),
        "score": float(getattr(hit, "score", 0.0) or 0.0),
        "chunk_id": _text(getattr(chunk, "id", "")),
    }


def _source_paper_from_chunk(chunk: Any) -> Dict[str, Any]:
    """Build a :class:`~t2pw.rag.provenance.RagSourcePaper`-shaped record."""
    return {
        "source_id": _text(getattr(chunk, "source_id", "")) or _text(getattr(chunk, "id", "")),
        "title": _text(getattr(chunk, "source_title", "")),
        "uri": _text(getattr(chunk, "source_uri", "")),
        "source_type": _text(getattr(chunk, "source_type", "")),
        "organism": _text(getattr(chunk, "organism", "")),
    }


def _seed_source_descriptor(seed_context: Any) -> Optional[Dict[str, Any]]:
    """Extract a seed provenance pointer from ``seed_context`` when present.

    ``seed_context`` mirrors WP4's parameter and may be a plain string (pathway
    context text, in which case there is no seed source pointer) or a mapping
    that carries a ``source`` sub-dict (or the source fields inline). The seed
    paper is itself a source, so the uploaded document's identity travels here.
    """
    if not isinstance(seed_context, dict):
        return None
    source = seed_context.get("source")
    src = source if isinstance(source, dict) else seed_context
    source_id = _text(src.get("source_id") or src.get("id"))
    if not source_id:
        return None
    return {
        "source_id": source_id,
        "source_title": _text(src.get("source_title") or src.get("title")),
        "source_type": _text(src.get("source_type")) or "paper",
        "source_uri": _text(src.get("source_uri") or src.get("uri")),
        "section": _text(src.get("section")),
        "organism": _text(src.get("organism")),
        "chunk_id": _text(src.get("chunk_id")),
    }


# ---------------------------------------------------------------------------
# Evidence reaction parsing (deterministic, offline).
# ---------------------------------------------------------------------------
def _after_delim(segment: str) -> str:
    for delim in (":", "="):
        if delim in segment:
            return segment.split(delim, 1)[1].strip()
    return ""


def _split_arrow(equation: str) -> Optional[Tuple[str, str, bool]]:
    for arrow in _ARROWS:
        if arrow in equation:
            lhs, rhs = equation.split(arrow, 1)
            return lhs.strip(), rhs.strip(), arrow in _REVERSIBLE_ARROWS
    return None


def _parse_side(side: str) -> List[_Participant]:
    out: List[_Participant] = []
    for raw in _SIDE_SPLIT_RE.split(side.strip()):
        token = raw.strip()
        if not token:
            continue
        match = _COEFF_RE.match(token)
        if match:
            name = canonical_name(match.group(2))
            stoich: Optional[float] = float(match.group(1))
        else:
            name = canonical_name(token)
            stoich = None
        # Drop any token that is clearly not a single chemical species (a
        # " ; "-joined pathway-metadata blob, a pathway id, a biological-state
        # descriptor, or an absurdly long name).
        if _is_invalid_species_token(name):
            continue
        out.append(_Participant(name, stoich))
    return [p for p in out if p.name]


def _parse_reaction_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one evidence line into name/inputs/outputs/enzymes/reversible.

    Recognised shape (KEGG-style equations and paper prose both fit)::

        name: R4 theobromine demethylation | theobromine + O2 -> 7-methylxanthine
        + formaldehyde | enzyme: NdmB

    Segments are ``|``-separated; the segment containing an arrow is the
    equation, ``enzyme:``/``name:`` label the catalyst/name, and any bare
    label becomes the reaction name. A line with no arrow is not a reaction.
    """
    text = line.strip()
    if not text:
        return None
    segments = [seg.strip() for seg in text.split("|") if seg.strip()]
    equation = ""
    enzymes: List[str] = []
    name = ""
    for seg in segments:
        low = seg.lower()
        if low.startswith("enzyme") or low.startswith("catalyst"):
            value = _after_delim(seg)
            enzymes.extend(
                canonical_name(part) for part in value.split(",") if part.strip()
            )
        elif low.startswith("name"):
            name = _after_delim(seg)
        elif _split_arrow(seg) is not None:
            equation = seg
        elif not name:
            name = seg
    if not equation and _split_arrow(text) is not None and len(segments) == 1:
        equation = text
    parsed = _split_arrow(equation) if equation else None
    if parsed is None:
        return None
    lhs, rhs, reversible = parsed
    inputs = _parse_side(lhs)
    outputs = _parse_side(rhs)
    # A "reaction" that lost participants to the species guards was garbage (e.g.
    # a " ; "-joined metadata bag) — discard it entirely. Both sides are required:
    # a one-sided row cannot be expressed in PWML, and the name fallback below
    # would render it "<participant> -> ?", which the required-field gate rejects.
    if not inputs or not outputs:
        return None
    # Reject a reaction name that is itself pathway-metadata garbage; fall back
    # to a participant-derived name instead.
    if name and _is_invalid_species_token(name):
        name = ""
    if not name:
        left = inputs[0].name if inputs else "?"
        right = outputs[0].name if outputs else "?"
        name = f"{left} -> {right}"
    return {
        "name": _text(name),
        "inputs": inputs,
        "outputs": outputs,
        "enzymes": [e for e in enzymes if e and not _is_invalid_species_token(e)],
        "reversible": reversible,
    }


# ---------------------------------------------------------------------------
# LLM prose→reaction extraction (opt-in; the arrow parser handles equations).
# ---------------------------------------------------------------------------
# Hard cap on how many distinct evidence passages one synthesis run sends to the
# prose extractor, bounding LLM cost/latency regardless of gap/top-k counts.
_EXTRACT_MAX_PASSAGES = 24


def _reaction_from_extracted(
    parsed: Any,
    prov: Dict[str, Any],
    evidence: Dict[str, Any],
    paper: Dict[str, Any],
    score: float,
    gap_id: str = "",
    chunk: Any = None,
) -> Optional[_Reaction]:
    """Build a provenance-bound :class:`_Reaction` from one extracted reaction dict.

    ``parsed`` is a clean dict from :func:`t2pw.rag.extract.extract_reactions_from_text`
    (``{"name", "inputs", "outputs", "enzymes", "reversible", "quote"}``). Names
    are canonicalized and junk tokens rejected exactly as the arrow-parser path
    does, and the chunk's provenance is attached so the reaction stays
    evidence-bound. Returns ``None`` when nothing usable survives.

    The model's ``quote`` is **validated, never trusted**: it is resolved through
    :func:`t2pw.rag.admission.locate_span`, which accepts it only if it appears
    verbatim in this chunk and is a single statement, and otherwise falls back to
    locating the one sentence of the chunk that names every participant and
    catalyst. A claim with no such span keeps ``evidence_span=""`` and the
    admission gate refuses it — an extraction that stitched two sentences
    together has no single sentence backing it, which is precisely how
    "Enz1 catalyzes A to B. Enz2 catalyzes X to Y." is prevented from yielding
    ``A -> Y``.
    """
    if not isinstance(parsed, dict):
        return None
    inputs = [
        p
        for p in _participants_from_field(parsed.get("inputs"))
        if not _is_invalid_species_token(p.name)
    ]
    outputs = [
        p
        for p in _participants_from_field(parsed.get("outputs"))
        if not _is_invalid_species_token(p.name)
    ]
    # Both sides are required — see ``_parse_reaction_line``. A passage that names
    # only a substrate (or whose other side was entirely junk tokens) yields no
    # exportable reaction, so it is dropped here rather than named "<name> -> ?".
    if not inputs or not outputs:
        return None
    enzymes: List[str] = []
    for raw in _safe_list(parsed.get("enzymes")):
        if isinstance(raw, str):
            name = canonical_name(raw)
            if name and not _is_invalid_species_token(name):
                enzymes.append(name)
    name = _text(parsed.get("name"))
    if name and _is_invalid_species_token(name):
        name = ""
    if not name:
        left = inputs[0].name if inputs else "?"
        right = outputs[0].name if outputs else "?"
        name = f"{left} -> {right}"
    span = locate_span(
        _text(getattr(chunk, "text", "")),
        parsed.get("quote"),
        [p.name for p in inputs] + [p.name for p in outputs] + enzymes,
    )
    return _Reaction(
        name=name,
        inputs=inputs,
        outputs=outputs,
        enzymes=enzymes,
        reversible=bool(parsed.get("reversible")),
        provenance=[dict(prov)],
        evidence=[dict(evidence)],
        source_papers=[dict(paper)],
        scores=[float(score)],
        origin="rag",
        gap_id=gap_id,
        gap_ids=[gap_id] if gap_id else [],
        organism=_text(prov.get("organism")),
        evidence_span=span,
        observed_organisms=_safe_list(getattr(chunk, "observed_organisms", [])),
        observed_pathways=_safe_list(getattr(chunk, "observed_pathways", [])),
    )


def _make_memoized_extractor(prose_extractor: Optional[Any]) -> Optional[Any]:
    """Wrap a ``text -> [reaction dict]`` callable into a memoized ``chunk ->`` one.

    Returns ``None`` when ``prose_extractor`` is ``None`` (arrow-only, today's
    behavior). Otherwise each distinct chunk (by id) is extracted **at most once**
    — so the two passes over the bundles (main synthesis + unfilled-gap
    detection) never double-call the model — up to :data:`_EXTRACT_MAX_PASSAGES`
    passages per run. Every call fails closed: a per-chunk error yields ``[]`` so
    extraction can only add reactions, never break synthesis.
    """
    if prose_extractor is None:
        return None
    memo: Dict[str, List[Dict[str, Any]]] = {}
    budget = {"used": 0}

    def _run(chunk: Any) -> List[Dict[str, Any]]:
        cid = _text(getattr(chunk, "id", "")) or _text(getattr(chunk, "text", ""))[:80]
        if cid in memo:
            return memo[cid]
        text = _text(getattr(chunk, "text", ""))
        if not text or budget["used"] >= _EXTRACT_MAX_PASSAGES:
            memo[cid] = []
            return []
        budget["used"] += 1
        try:
            result = list(prose_extractor(text) or [])
        except Exception:  # noqa: BLE001 - extraction must never break synthesis
            result = []
        memo[cid] = result
        return result

    return _run


_DOI_RE = re.compile(r"doi:\s*10\.", re.IGNORECASE)
_ACCESSION_RE = re.compile(r"\b(?:PMC\d{4,}|PMID\s*:?\s*\d{4,})\b", re.IGNORECASE)
_CITATION_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\s*[.;]")

# A chunk needs this many citation markers, this many dated entries, and this
# marker density (per 1000 characters) before it is treated as a reference list.
_BIBLIOGRAPHY_MIN_MARKERS = 3
_BIBLIOGRAPHY_MIN_ENTRIES = 3
_BIBLIOGRAPHY_MARKERS_PER_1K = 2.0


def _is_bibliography_text(text: str) -> bool:
    """True when a passage reads as a reference list rather than prose.

    Keyed on density, not on absolute counts: ordinary discussion cites a DOI or
    two, whereas a bibliography packs an accession and a dated entry into every
    line. Requiring markers *and* dated entries *and* a per-length density keeps
    a paragraph that happens to quote a reference from being discarded.
    """
    body = text or ""
    if len(body) < 200:
        return False
    markers = len(_DOI_RE.findall(body)) + len(_ACCESSION_RE.findall(body))
    if markers < _BIBLIOGRAPHY_MIN_MARKERS:
        return False
    if len(_CITATION_YEAR_RE.findall(body)) < _BIBLIOGRAPHY_MIN_ENTRIES:
        return False
    return (markers * 1000.0 / len(body)) >= _BIBLIOGRAPHY_MARKERS_PER_1K


def _reactions_from_bundle(
    bundle: Any, extractor: Optional[Any] = None
) -> List[_Reaction]:
    """Transcribe every reaction stated in a bundle's evidence hits.

    Each parsed reaction inherits the provenance/evidence/source-paper pointers
    of the chunk that stated it — so it is evidence-bound by construction. Two
    transcription paths run per chunk: the deterministic arrow-equation parser
    (``A + B -> C``) and, when ``extractor`` is supplied, the LLM prose extractor
    (``t2pw.rag.extract``) that recovers reactions stated in ordinary sentences.
    ``extractor`` is the memoized per-chunk callable built by
    :func:`_make_memoized_extractor`; ``None`` (the default) means arrow-only,
    exactly today's behavior.

    Every reaction built here inherits the bundle's ``gap_id``. That stamp is
    what makes the admission gate possible at all: a reaction transcribed from a
    passage retrieved for gap G is a *claim about G*, and if it turns out not to
    fill G it is unrelated chemistry that happened to share a passage, not a
    contribution to the pathway.
    """
    gap_id = _text(getattr(_gap_of(bundle), "gap_id", ""))
    reactions: List[_Reaction] = []
    for hit in _safe_list(getattr(bundle, "hits", [])):
        chunk = _chunk_of(hit)
        if chunk is None:
            continue
        # Only transcribe reactions from genuine prose/equation chunks. A
        # ``pwml_example`` corpus chunk, or a ``pathbank`` / ``kegg`` DB record,
        # is a " ; "-joined metadata bag built for lexical scoring — never a
        # clean single equation — so its (incidental) arrow must not be parsed.
        source_type = _text(getattr(chunk, "source_type", "")).lower()
        if source_type not in _PARSEABLE_SOURCE_TYPES:
            continue
        # Back-matter is labelled and dropped at ingest, but header detection is
        # best-effort — a paper whose only "methods" match lands inside its
        # reference list leaves the bibliography tagged as body text. Cited titles
        # read as reaction descriptions ("... gene (FabZ) encoding
        # (3R)-hydroxymyristoyl acyl carrier protein dehydrase"), so a
        # citation-dense chunk is refused here as well.
        if _is_bibliography_text(_text(getattr(chunk, "text", ""))):
            continue
        prov = _provenance_from_chunk(chunk)
        evidence = _evidence_from_hit(hit)
        paper = _source_paper_from_chunk(chunk)
        score = float(getattr(hit, "score", 0.0) or 0.0)
        for line in _text(getattr(chunk, "text", "")).splitlines():
            parsed = _parse_reaction_line(line)
            if parsed is None:
                continue
            reactions.append(
                _Reaction(
                    name=parsed["name"],
                    inputs=parsed["inputs"],
                    outputs=parsed["outputs"],
                    enzymes=parsed["enzymes"],
                    reversible=parsed["reversible"],
                    provenance=[dict(prov)],
                    evidence=[dict(evidence)],
                    source_papers=[dict(paper)],
                    scores=[score],
                    origin="rag",
                    gap_id=gap_id,
                    gap_ids=[gap_id] if gap_id else [],
                    organism=_text(getattr(chunk, "organism", "")),
                    # The parsed SOURCE LINE is the evidence span, kept exactly
                    # as it appeared. The parent chunk pointer stays in
                    # ``provenance`` / ``evidence``, so the row still says which
                    # passage of which paper this came from — the span narrows
                    # the claim, it does not replace the provenance.
                    evidence_span=line.strip(),
                    observed_organisms=_safe_list(
                        getattr(chunk, "observed_organisms", [])
                    ),
                    observed_pathways=_safe_list(
                        getattr(chunk, "observed_pathways", [])
                    ),
                )
            )
        if extractor is not None:
            for pr in extractor(chunk):
                rxn = _reaction_from_extracted(
                    pr, prov, evidence, paper, score, gap_id, chunk
                )
                if rxn is not None:
                    reactions.append(rxn)
    return reactions


# ---------------------------------------------------------------------------
# Seed reaction ingestion.
# ---------------------------------------------------------------------------
def _participants_from_field(value: Any) -> List[_Participant]:
    out: List[_Participant] = []
    for item in _safe_list(value):
        if isinstance(item, str):
            name = canonical_name(item)
            if name:
                out.append(_Participant(name, None))
        elif isinstance(item, dict):
            raw = (
                item.get("name")
                or item.get("entity")
                or item.get("compound")
                or item.get("protein")
                or item.get("element")
                or item.get("element_collection")
                or item.get("nucleic_acid")
            )
            name = canonical_name(raw)
            if not name:
                continue
            stoich = item.get("stoichiometry")
            if stoich is None:
                stoich = item.get("coefficient")
            out.append(_Participant(name, float(stoich) if _is_number(stoich) else None))
    return out


def _enzymes_from_reaction(row: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("enzymes", "modifiers"):
        for actor in _safe_list(row.get(key)):
            if isinstance(actor, dict):
                for field_name in ("entity", "protein", "protein_complex", "name"):
                    val = actor.get(field_name)
                    if isinstance(val, str) and val.strip():
                        names.append(canonical_name(val))
                        break
            elif isinstance(actor, str) and actor.strip():
                names.append(canonical_name(actor))
    return [n for n in names if n]


def _seed_reactions(
    seed_payload: Any, seed_source: Optional[Dict[str, Any]]
) -> Tuple[List[_Reaction], List[Dict[str, Any]]]:
    """Read the seed extraction's reactions, attaching seed provenance.

    A seed reaction that carries no provenance of its own and has no seed source
    to fall back on is *unsupported* — it is omitted and reported, exactly the
    no-invention guardrail applied even to the seed paper.
    """
    processes = _safe_dict(_safe_dict(seed_payload).get("processes"))
    omitted: List[Dict[str, Any]] = []
    reactions: List[_Reaction] = []
    for idx, row in enumerate(_safe_list(processes.get("reactions"))):
        if not isinstance(row, dict):
            continue
        inputs = _participants_from_field(row.get("inputs"))
        outputs = _participants_from_field(row.get("outputs"))
        if not inputs and not outputs:
            continue
        prov = _seed_row_provenance(row, seed_source)
        name = _text(row.get("name")) or f"seed_reaction_{idx + 1}"
        if not prov:
            omitted.append(
                {
                    "kind": "reaction",
                    "label": name,
                    "reason": "seed reaction has no supporting evidence (omitted)",
                }
            )
            continue
        scope = row.get("scope_membership")
        reactions.append(
            _Reaction(
                name=name,
                inputs=inputs,
                outputs=outputs,
                enzymes=_enzymes_from_reaction(row),
                reversible=bool(row.get("reversible")),
                compartment=_text(row.get("compartment")),
                provenance=prov,
                evidence=_seed_row_evidence(row),
                source_papers=[_paper_from_provenance(p) for p in prov],
                scores=[],
                origin="seed",
                # Carried verbatim (stripped, never case-folded) for the same
                # reason ``pipeline._carry_scope_membership`` carries it: the lock
                # manifest records the model's own spelling and the two artifacts
                # must not disagree about what the model actually said. A
                # non-string label is dropped rather than coerced, which lands it
                # in the "absent" case the core filter treats as KEEP.
                scope_membership=scope.strip() if isinstance(scope, str) else "",
                organism=_text(prov[0].get("organism")) if prov else "",
                # Carried, never authored. Synthesis does not know how this seed
                # row came to exist -- in research mode the "seed" of round N+1 is
                # round N's payload, so a row here may already have been attributed
                # by paper extraction, gap resolution or an audit. Stamping it
                # ``paper_stated`` would be inventing an origin; dropping it would
                # erase another stage's. It is re-emitted verbatim.
                lineage=_row_lineage(row),
            )
        )
    return reactions, omitted


def _seed_row_provenance(
    row: Dict[str, Any], seed_source: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    existing = row.get("provenance")
    prov: List[Dict[str, Any]] = []
    if isinstance(existing, dict) and existing.get("source_id"):
        prov.append(dict(existing))
    for paper in _safe_list(row.get("source_papers")):
        if isinstance(paper, dict) and (paper.get("source_id") or paper.get("id")):
            prov.append(
                {
                    "source_id": _text(paper.get("source_id") or paper.get("id")),
                    "source_title": _text(paper.get("title")),
                    "source_type": _text(paper.get("source_type")) or "paper",
                    "source_uri": _text(paper.get("uri")),
                }
            )
    for ref in _safe_list(row.get("source_refs")):
        if isinstance(ref, str) and ref.strip():
            prov.append({"source_id": ref.strip(), "source_type": "paper"})
    if not prov and seed_source:
        prov.append(dict(seed_source))
    return prov


def _seed_row_evidence(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = row.get("evidence")
    if isinstance(ev, str) and ev.strip():
        return [{"text": ev.strip()}]
    if isinstance(ev, list):
        return [dict(e) for e in ev if isinstance(e, dict)]
    return []


def _paper_from_provenance(prov: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": _text(prov.get("source_id")),
        "title": _text(prov.get("source_title")),
        "uri": _text(prov.get("source_uri")),
        "source_type": _text(prov.get("source_type")) or "paper",
        "organism": _text(prov.get("organism")),
    }


# ---------------------------------------------------------------------------
# Merge / reconcile / resolve conflicts.
# ---------------------------------------------------------------------------
def _merge_into(target: _Reaction, other: _Reaction) -> None:
    """Fold ``other`` (same signature) into ``target``, unioning provenance.

    ``evidence`` and ``source_papers`` are UNIONED here, exactly like ``provenance``
    — they used to be blindly ``extend``-ed, and that asymmetry is the origin of the
    evidence amplification seen in run 2026-07-28_0919.

    Why the amplification happens: :func:`_reactions_from_bundle` is called once PER
    GAP BUNDLE, and every reaction it builds carries ``evidence=[record]`` holding
    the WHOLE chunk text. A chunk that is top-k for N different gaps therefore
    produces N identical ``_Reaction`` objects, all of which land on the same
    signature and fold in here. The repeat count IS the gap-bundle count. In that
    run, PMC12444477 "The regulation of lipid A biosynthesis" (strict) shipped a
    reaction (#14) whose evidence was 139,576 characters: ONE 4,812-character
    passage repeated 29 times, once per gap. Payload-wide, reaction evidence came to
    2,716,278 chars and enzyme-row evidence to 1,888,665 chars — 4.6 MB of a 4.70 MB
    merged payload — and 177 of the 204 enzyme rows carried evidence of exactly 119
    or 120 characters, i.e. the same short passage restated over and over.

    The dedup key is the explicit ``(chunk_id, text)`` identity pair (see
    :func:`_dedupe_evidence`), NOT ``set``/``dict.fromkeys`` and NOT whole-record
    equality. Both alternatives are broken here: the records are dicts, so they are
    unhashable and cannot enter a set at all; and whole-record equality would MISS
    the duplicates anyway, because :func:`_evidence_from_hit` stores
    ``"score": float(hit.score)`` and that score differs per gap for the very same
    chunk. Two records naming the same chunk with the same text are the same passage
    regardless of what any one retrieval scored it.

    ``scores`` is deliberately left as a plain ``extend``: a score is a per-retrieval
    observation and :meth:`_Reaction.weight` SUMS them to rank conflicting variants
    in :func:`_resolve_reactions`. Collapsing repeated scores would silently re-rank
    conflict resolution — a different behavior change than the one this fix makes.

    No cap on the number of distinct passages is applied. That idea was reviewed and
    rejected: a row genuinely supported by several papers must keep every distinct
    passage it rests on. Only exact repeats are removed.
    """
    seen = {
        (p.get("source_id"), p.get("chunk_id")) for p in target.provenance
    }
    for prov in other.provenance:
        key = (prov.get("source_id"), prov.get("chunk_id"))
        if key not in seen:
            target.provenance.append(prov)
            seen.add(key)
    target.evidence = _dedupe_evidence(target.evidence + other.evidence)
    target.source_papers = _dedupe_papers(target.source_papers + other.source_papers)
    target.scores.extend(other.scores)
    for enzyme in other.enzymes:
        if enzyme not in target.enzymes:
            target.enzymes.append(enzyme)
    # Attribution fields fill in but never overwrite: the surviving row keeps its
    # OWN origin / scope label / gap, so corroboration from a second paper cannot
    # relabel a seed reaction as a RAG import, and a RAG row that was admitted for
    # gap A does not silently become "the fill for gap B" because a duplicate of it
    # was also retrieved for B. (The unioned provenance already records both.)
    if not target.scope_membership and other.scope_membership:
        target.scope_membership = other.scope_membership
    if not target.gap_id and other.gap_id:
        target.gap_id = other.gap_id
    if not target.organism and other.organism:
        target.organism = other.organism
    if not target.evidence_span and other.evidence_span:
        target.evidence_span = other.evidence_span
    # ``gap_ids`` is the one attribution field that UNIONS rather than filling in.
    # A passage is routinely top-k for several gaps, so the same canonical claim
    # arrives once per gap and merges here; keeping only the first ``gap_id``
    # would leave the other gaps reading as unfilled even though the delivered
    # reaction closes them. ``gap_id`` stays the primary (the gap the admission
    # decision was made against) for backward compatibility.
    target.gap_ids = _dedupe_strs(
        list(target.gap_ids or ([target.gap_id] if target.gap_id else []))
        + list(other.gap_ids or ([other.gap_id] if other.gap_id else []))
    )
    # Lineage UNIONS, for the same reason ``gap_ids`` does and the opposite reason
    # to the fill-in fields above: it is append-only, so the surviving row may not
    # keep only its own attribution. Two candidates admitted against two different
    # gaps merge into one row that genuinely was admitted for both, and dropping
    # the folded-in entry would make the second admission unattributable. Deduped
    # by VALUE so folding an identical record twice does not restate it.
    for entry in other.lineage:
        if entry not in target.lineage:
            target.lineage.append(entry)


def _resolve_reactions(
    reactions: List[_Reaction],
    resolver: Optional[Any] = None,
) -> Tuple[List[_Reaction], List[Dict[str, Any]]]:
    """Dedupe identical reactions and resolve conflicting variants by weight.

    Reactions are grouped by their direction-aware ``conflict_key`` — the (sorted
    inputs, sorted outputs) pair, i.e. the same inputs->outputs mapping is the same
    underlying reaction. Opposite directions (a forward/reverse pair) key to
    *different* groups and both survive, mirroring the core pipeline. Within a
    group, identical *signatures* are merged; when two or more distinct signatures
    disagree on stoichiometry / compartment / reversible flag (a *same-direction*
    disagreement), the highest-evidence-weight variant wins and the losers are
    recorded as ``conflicts`` (nothing is dropped silently).

    ``resolver`` (optional, GROUPING-ONLY) feeds the synonym-canonical token into
    both keys so cross-paper duplicates differing only by a compound/enzyme SYNONYM
    collapse to one row (provenance unioned). ``None`` (default) = today's exact
    grouping. The surviving row always keeps its own real names — only the KEYS are
    synonym-aware, never the emitted display names.
    """
    groups: Dict[Tuple[Any, ...], List[_Reaction]] = {}
    order: List[Tuple[Any, ...]] = []
    for rxn in reactions:
        key = rxn.conflict_key(resolver)
        if key not in groups:
            groups[key] = []
            order.append(key)
        # Merge into an existing identical-signature variant if present.
        merged = False
        rxn_sig = rxn.signature(resolver)
        for existing in groups[key]:
            if existing.signature(resolver) == rxn_sig:
                _merge_into(existing, rxn)
                merged = True
                break
        if not merged:
            groups[key].append(rxn)

    resolved: List[_Reaction] = []
    conflicts: List[Dict[str, Any]] = []
    for key in order:
        variants = groups[key]
        if len(variants) == 1:
            resolved.append(variants[0])
            continue
        # Deterministic: max weight, tie-break by provenance count then order.
        ranked = sorted(
            enumerate(variants),
            key=lambda pair: (pair[1].weight(), len(pair[1].provenance), -pair[0]),
            reverse=True,
        )
        winner = ranked[0][1]
        resolved.append(winner)
        conflicts.append(
            {
                "participants": list(key),
                "chosen": _variant_summary(winner),
                "alternatives": [
                    _variant_summary(v) for _, v in ranked[1:]
                ],
            }
        )
    return resolved, conflicts


def _variant_summary(rxn: _Reaction) -> Dict[str, Any]:
    return {
        "name": rxn.name,
        "inputs": rxn.input_names(),
        "outputs": rxn.output_names(),
        "reversible": rxn.reversible,
        "compartment": rxn.compartment,
        "weight": round(rxn.weight(), 6),
        "source_ids": sorted(set(rxn.source_ids())),
    }


# ---------------------------------------------------------------------------
# Stitch detection (cross-paper connections closed by evidence).
# ---------------------------------------------------------------------------
def _detect_stitches(reactions: List[_Reaction]) -> List[Dict[str, Any]]:
    producers: Dict[str, List[int]] = {}
    consumers: Dict[str, List[int]] = {}
    display: Dict[str, str] = {}
    for idx, rxn in enumerate(reactions):
        for name in rxn.output_names():
            producers.setdefault(name.casefold(), []).append(idx)
            display.setdefault(name.casefold(), name)
        for name in rxn.input_names():
            consumers.setdefault(name.casefold(), []).append(idx)
            display.setdefault(name.casefold(), name)

    stitches: List[Dict[str, Any]] = []
    seen: set = set()
    for metabolite, prod_idxs in producers.items():
        if _is_cofactor(metabolite):
            continue
        for cons_idx in consumers.get(metabolite, []):
            for prod_idx in prod_idxs:
                if prod_idx == cons_idx:
                    continue
                prod_sources = set(reactions[prod_idx].source_ids())
                cons_sources = set(reactions[cons_idx].source_ids())
                # A stitch is cross-paper: the producer and consumer come from
                # different sources (no single paper stated the whole link).
                if prod_sources and cons_sources and prod_sources.isdisjoint(
                    cons_sources
                ):
                    marker = (metabolite, prod_idx, cons_idx)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    stitches.append(
                        {
                            "metabolite": display.get(metabolite, metabolite),
                            "producer_reaction": reactions[prod_idx].name,
                            "producer_sources": sorted(prod_sources),
                            "consumer_reaction": reactions[cons_idx].name,
                            "consumer_sources": sorted(cons_sources),
                        }
                    )
    return stitches


# ---------------------------------------------------------------------------
# Entity registry construction (with mandatory provenance / omission).
# ---------------------------------------------------------------------------
#: What makes a ``protein_complexes`` row a complex rather than a name. Carried
#: forward from the seed verbatim so a complex the seed declared keeps its
#: members and its identity story through synthesis; nothing else is copied.
_COMPLEX_IDENTITY_KEYS = (
    "components",
    "mapped_ids",
    "mapping_meta",
    "identity_status",
    "pathbank_complex_id",
    "pathbank_protein_complex_id",
    "pw_complex_id",
    "pathwhiz_id",
)
def _build_entities(
    reactions: List[_Reaction],
    resolver: Optional[Any] = None,
    seed_payload: Any = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Collect compounds & proteins referenced by evidence-backed reactions.

    Each entity's provenance is the union of the provenance of every reaction
    that references it. A **non-cofactor** entity that ends up with no provenance
    is omitted and reported (guardrail); cofactors are exempt from the
    requirement but still carry provenance when it is available.

    ``resolver`` (optional, GROUPING-ONLY) unifies entity nodes that are the same
    compound/protein under different SYNONYMS: they register under one synonym
    token key, so the payload lists the species ONCE. The kept display name is the
    first-seen original name (``setdefault``) — never a rewritten canonical form —
    so this only affects the ``entities`` buckets, never the reaction rows the
    locked-reaction gate matches on. ``None`` (default) = today's ``casefold`` key.

    This registry is deliberately COMPLETE for the synthesized payload, including
    the seed's own entities: ``reactions`` contains the seed's reactions
    (``_seed_reactions`` turns every Stage-1 reaction into a ``_Reaction``), and
    ``synthesize_with_report`` returns a standalone ``Payload`` whose entity buckets
    have to cover its own reactions. Suppressing the seed's entities HERE — to stop
    them being re-imported as duplicate rows by the downstream merge — is therefore
    the wrong seam, and breaks the tested contract that a seed compound still
    carries its provenance in the synthesized payload
    (tests/test_rag_synthesize.py asserts
    ``compounds["caffeine"]["source_refs"] == ["PMID:0001"]`` and that NdmA is
    listed). The duplicate is created at the MERGE boundary, where the base is
    known, and is removed there — see
    :func:`t2pw.rag.conform.conform_rag_additions_for_merge`.

    ``seed_payload`` supplies the ENTITY TYPING the reaction rows do not carry. A
    reaction knows only "this name catalyses me", so a name in that role was
    emitted as a ``proteins`` row — including when the seed declared it in
    ``entities.protein_complexes``. That silently re-typed a complex as a protein
    in the payload every downstream check then reads: the complex's components
    (and with them its PathBank Unknown-protein fallback) disappeared, and a
    UniProt accession became writable onto a row that should never take one.
    Complexes are therefore emitted into their own bucket, carrying the seed row's
    identity fields forward.
    """
    seed_entities = _safe_dict(_safe_dict(seed_payload).get("entities"))
    seed_complexes: Dict[str, Dict[str, Any]] = {}
    for row in _safe_list(seed_entities.get("protein_complexes")):
        if isinstance(row, dict) and _text(row.get("name")):
            seed_complexes[canonical_name(row["name"]).casefold()] = row

    def _entity_key(name: str) -> str:
        return name.casefold() if resolver is None else resolver(name)

    enzyme_names = {_entity_key(e) for rxn in reactions for e in rxn.enzymes}
    # Entities this stage did NOT introduce (C-035): anything the seed payload
    # already declared as an entity, plus anything a SEED reaction references.
    # ``_build_entities`` runs over seed and RAG reactions alike, so a blanket
    # ``rag_literature`` stamp would relabel the seed pathway's own chemistry as a
    # cross-paper import -- which is inventing an origin. An entity in this set
    # gets no entry from here; whichever stage did introduce it owns that. The
    # match is name-keyed and can only ever SUPPRESS a record, so any imprecision
    # errs toward recording nothing, which is the safe direction.
    already_present = {
        _entity_key(canonical_name(erow["name"]))
        for erows in seed_entities.values()
        for erow in _safe_list(erows)
        if isinstance(erow, dict) and _text(erow.get("name"))
    }
    already_present |= {
        _entity_key(name)
        for rxn in reactions
        if rxn.origin != "rag"
        for name in rxn.participant_names() + list(rxn.enzymes)
    }
    prov_by_name: Dict[str, List[Dict[str, Any]]] = {}
    display_by_name: Dict[str, str] = {}
    scores_by_name: Dict[str, List[float]] = {}

    def _register(name: str, prov: List[Dict[str, Any]], scores: List[float]) -> None:
        key = _entity_key(name)
        display_by_name.setdefault(key, name)
        bucket = prov_by_name.setdefault(key, [])
        seen = {(p.get("source_id"), p.get("chunk_id")) for p in bucket}
        for p in prov:
            marker = (p.get("source_id"), p.get("chunk_id"))
            if marker not in seen and p.get("source_id"):
                bucket.append(p)
                seen.add(marker)
        scores_by_name.setdefault(key, []).extend(scores)

    for rxn in reactions:
        for name in rxn.participant_names():
            _register(name, rxn.provenance, rxn.scores)
        for enzyme in rxn.enzymes:
            _register(enzyme, rxn.provenance, rxn.scores)

    compounds: List[Dict[str, Any]] = []
    proteins: List[Dict[str, Any]] = []
    complexes: List[Dict[str, Any]] = []
    omitted: List[Dict[str, Any]] = []
    for key in sorted(display_by_name):
        display = display_by_name[key]
        prov = prov_by_name.get(key, [])
        is_protein = key in enzyme_names
        seed_complex = seed_complexes.get(canonical_name(display).casefold())
        if not prov and not _is_cofactor(display):
            omitted.append(
                {
                    "kind": "protein" if is_protein else "compound",
                    "label": display,
                    "reason": "entity has no supporting evidence (omitted)",
                }
            )
            continue
        row = {"name": display}
        _attach_provenance(row, prov, [], scores_by_name.get(key, []))
        if prov and key not in already_present:
            # ``derived``, not ``direct``: the passage states a REACTION, and this
            # row is the deterministic projection of that reaction onto its
            # participants. The weaker level is the accurate one. A cofactor row
            # that reached here with no provenance at all gets nothing -- there is
            # no record to name, and a sourceless ``rag_literature`` claim is
            # exactly the assertion this stage is not entitled to make.
            _write_lineage(
                row,
                _retrieval_entry(
                    prov[0],
                    f"{display!r} entered the payload only through RAG-imported "
                    "reaction(s): it appears in no seed reaction and in no seed "
                    "entity row",
                    support="derived",
                ),
            )
        if is_protein and seed_complex is not None:
            # Keep the complex a complex, with the identity fields that make it
            # one. ``components`` is what carries the Unknown-protein fallback;
            # dropping it turns a functional complex into a nameless protein.
            # Copied AFTER ``_attach_provenance``, whose whitelist deliberately
            # covers only the additive provenance keys.
            for field_name in _COMPLEX_IDENTITY_KEYS:
                if field_name in seed_complex:
                    row[field_name] = copy.deepcopy(seed_complex[field_name])
            complexes.append(row)
            continue
        if is_protein:
            proteins.append(row)
        else:
            compounds.append(row)
    entities: Dict[str, List[Dict[str, Any]]] = {}
    if compounds:
        entities["compounds"] = compounds
    if proteins:
        entities["proteins"] = proteins
    if complexes:
        entities["protein_complexes"] = complexes
    return entities, omitted


# ---------------------------------------------------------------------------
# Provenance attachment (the four WP0 additive keys + core-safe source_refs).
# ---------------------------------------------------------------------------
def _attach_provenance(
    row: Dict[str, Any],
    provenance: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    scores: List[float],
    *,
    gap_id: str = "",
    gap_ids: Optional[List[str]] = None,
) -> None:
    """Attach the additive provenance keys defined in ``t2pw.rag.provenance``.

    ``rag_provenance`` (the primary pointer), ``evidence``, ``source_papers`` and
    ``rag_confidence`` are the four keys WP0 permits (``RAG_ADDITIVE_KEYS``); a
    core-typed ``source_refs`` list is also written so a strict, RAG-unaware
    consumer still sees a valid ``List[str]`` provenance pointer. The primary
    pointer is namespaced ``rag_provenance`` (never ``provenance``) so it never
    collides with the core ``provenance`` string field. Every key is
    optional/additive — a stage that ignores them reads a plain core row.
    """
    if not provenance:
        return
    primary = dict(provenance[0])
    if gap_id:
        # The gap this row was admitted for travels INSIDE the existing
        # ``rag_provenance`` pointer (an optional key of the ``RagProvenance``
        # TypedDict) rather than as a sixth top-level additive key: the additive
        # set is a fixed, asserted contract (``RAG_ADDITIVE_KEYS``), and a
        # per-row string costs ~30 bytes against a payload whose size history is
        # the reason ``_dedupe_evidence`` exists.
        primary["gap_id"] = gap_id
        # ...and the COMPLETE set beside it. ``gap_id`` alone is lossy: one
        # canonical claim retrieved for two gaps merges into one row, and if only
        # the first attribution survived, the second gap would be reported
        # unfilled while the reaction that fills it sits in the payload. Written
        # only when it adds something, so a single-gap row keeps its old shape.
        complete = _dedupe_strs(list(gap_ids or []) + [gap_id])
        if len(complete) > 1:
            primary["gap_ids"] = complete
    row["rag_provenance"] = primary
    if evidence:
        # Deduped for the same reason ``source_papers`` and ``source_refs`` are on
        # the next four lines. This assignment was the one member of the group with
        # no dedupe helper, and it is reached once per reaction row AND once per
        # enzyme actor on that reaction (:func:`_enzyme_actor` passes the reaction's
        # own evidence list), so a repeated passage was billed once per row and again
        # per catalyst. That is how run 2026-07-28_0919 turned 9 extracted reactions
        # into 204 enzyme rows carrying 1,888,665 characters of evidence.
        row["evidence"] = [dict(e) for e in _dedupe_evidence(evidence)]
    papers = _dedupe_papers(_paper_from_provenance(p) for p in provenance)
    if papers:
        row["source_papers"] = papers
    source_refs = _dedupe_strs(str(p.get("source_id") or "") for p in provenance)
    if source_refs:
        row["source_refs"] = source_refs
    row["rag_confidence"] = _confidence(scores, len(provenance))
    # Defensive: never emit a key outside the permitted additive set + the
    # core-owned fields this module writes.
    assert set(row) <= _ALLOWED_ROW_KEYS


#: Keys a synthesized row may carry. ``scope_membership`` is a CORE reaction
#: field (``schema.PayloadReaction``, ``payload_models.ReactionModel``) that RAG
#: rows were structurally unable to carry: this whitelist did not name it, so
#: ``_attach_provenance``'s assertion below would have fired on any row that had
#: one. That is why ``pipeline._carry_scope_membership`` documents its own scope
#: limit as "cross-paper RAG imports ... cannot carry a scope label even in
#: principle". They can now: the admission gate writes the label
#: (:data:`~t2pw.rag.admission.SCOPE_ADMITTED`) and it survives synthesis ->
#: conform -> ``clean_inference_output`` -> normalization, so the core
#: out-of-scope filter and the lock manifest see the same label on a RAG row as
#: on a seed one.
_ALLOWED_ROW_KEYS = frozenset(
    {
        "name",
        "inputs",
        "outputs",
        "enzymes",
        "entity",
        "entity_type",
        "role",
        "source_refs",
        "scope_membership",
    }
    | set(RAG_ADDITIVE_KEYS)
    # C-035. Widening the LITERAL set, never ``RAG_ADDITIVE_KEYS``: that tuple is
    # an asserted contract pinned verbatim by ``tests/test_rag_foundation.py``, and
    # lineage is not a RAG provenance carrier -- it is the cross-stage attribution
    # ``rag/graph_delta.py`` already reads and already classifies as NON_BIOLOGICAL.
    # This widens what a row may CARRY, not what any stage emits: every write below
    # is conditional on evidence the stage actually holds.
    | {LINEAGE_KEY}
)


# ---------------------------------------------------------------------------
# Lineage (C-035) -- ATTRIBUTION ONLY.
#
# Emission is conditional on what this stage genuinely knows, never on what would
# make a row look better attributed. A row this stage cannot attribute honestly
# gets NO entry: inventing an origin is inventing biology, and a row asserting
# provenance it does not have is worse than a row with none.
#
# Every write happens AFTER ``_attach_provenance`` has run on the row. That is not
# stylistic -- ``_attach_provenance`` ends in ``assert set(row) <=
# _ALLOWED_ROW_KEYS`` and also early-returns when a row has no provenance, so a
# write placed before it would fire on exactly the rows that have evidence and stay
# silent on the ones that do not.
# ---------------------------------------------------------------------------
def _row_lineage(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The attribution ``row`` already carries; ``[]`` when it carries none.

    A malformed lineage is DROPPED, never raised. Synthesis is not the lineage
    validator, and a payload that synthesized yesterday must not start crashing
    because some earlier stage wrote a bad record -- that would be this instrument
    changing what the pipeline produces. ``rag/graph_delta.py`` is the seam that
    REPORTS a malformed lineage as a violation, which is where that belongs.
    """
    try:
        return read_lineage(row).as_list()
    except LineageError:
        return []


def _lineage_source(prov: Dict[str, Any]) -> Optional[LineageSource]:
    """A POINTER to the paper/passage ``prov`` names, or ``None`` if it names none.

    ``None`` is the honest answer for a provenance pointer with neither an id nor a
    URI: a ``rag_literature`` origin has to identify its supporting record, and an
    anonymous source would let the entry claim backing it cannot produce.
    """
    source_id = _text(prov.get("source_id"))
    uri = _text(prov.get("source_uri"))
    if not (source_id or uri):
        return None
    return LineageSource(
        source_id=source_id,
        source_type=_text(prov.get("source_type")) or "paper",
        uri=uri,
        locator=_text(prov.get("chunk_id")) or _text(prov.get("section")),
    )


def _retrieval_entry(
    prov: Dict[str, Any], reason: str, *, support: str
) -> Optional[LineageEntry]:
    """A ``rag_retrieval`` entry backed by ``prov``, or ``None`` when unciteable.

    ``paper_explicit`` is ``not_evaluated``: this stage transcribes what a RETRIEVED
    passage states and never compares it against the supplied paper, so it may not
    answer whether that paper stated it -- and ``not_evaluated`` is never
    ``not_explicit``. ``review_required`` is ``False``: transcription found nothing
    wrong, and a stage that flagged every row it touched would make the flag
    meaningless. Only the PRIMARY provenance pointer is named, matching the primary
    ``rag_provenance`` pointer ``_attach_provenance`` writes -- the corroborating
    papers are already on the row in ``source_papers`` / ``source_refs``, and this
    is a pointer to a record, not a second copy of it.
    """
    source = _lineage_source(prov)
    if source is None:
        return None
    return LineageEntry(
        stage="rag_retrieval",
        origin="rag_literature",
        support=support,
        paper_explicit="not_evaluated",
        reason=reason,
        review_required=False,
        sources=(source,),
    )


def _write_lineage(row: Dict[str, Any], *entries: Any) -> None:
    """Append every non-``None`` entry to ``row``'s lineage, in place.

    Goes through :func:`t2pw.pipeline.lineage.record`, which re-emits every entry
    already present, so this can only ever ADD to an attribution.
    """
    for entry in entries:
        if entry is not None:
            record_lineage(row, entry)


def _confidence(scores: List[float], prov_count: int) -> float:
    usable = [float(s) for s in scores if _is_number(s) and float(s) > 0.0]
    if usable:
        return round(min(1.0, max(usable)), 6)
    return round(min(1.0, 0.5 + 0.1 * max(prov_count, 1)), 6)


def _dedupe_papers(papers: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for paper in papers:
        sid = paper.get("source_id")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(paper)
    return out


def _dedupe_evidence(records: Any) -> List[Dict[str, Any]]:
    """Collapse repeats of the same passage, keyed on ``(chunk_id, text)``.

    The evidence sibling of :func:`_dedupe_papers` / :func:`_dedupe_strs`. It exists
    because those two were applied to ``source_papers`` and ``source_refs`` while the
    ``evidence`` assignment right beside them had no dedupe at all — the asymmetry
    that let run 2026-07-28_0919 ship a 4.70 MB payload of which 4.6 MB was the same
    handful of passages restated (one reaction carried a single 4,812-char passage 29
    times, once per gap bundle; 177 of 204 enzyme rows carried the same 119/120-char
    passage).

    Keyed on the identity pair rather than on the record: ``_evidence_from_hit``
    writes a per-retrieval ``score``, so the SAME chunk retrieved for two different
    gaps yields two records that are unequal as dicts but are one passage. Dicts are
    also unhashable, so ``set``/``dict.fromkeys`` is not available here anyway.

    Unlike :func:`_dedupe_papers`, which drops a paper with no ``source_id`` (an
    unciteable pointer is worthless), a record with neither a ``chunk_id`` nor a
    ``text`` is KEPT verbatim: evidence lists carried over from a seed row
    (:func:`_seed_row_evidence`) may hold records this module did not build, and
    collapsing every identity-less record into one would lose real data to save
    nothing. First-seen order is preserved so the leading passage stays leading.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for record in records:
        key = (_text(record.get("chunk_id")), _text(record.get("text")))
        if key == ("", ""):
            out.append(record)
        elif key not in seen:
            seen.add(key)
            out.append(record)
    return out


def _dedupe_strs(values: Any) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for value in values:
        token = str(value or "").strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# Reaction row emission (standard Payload shape).
# ---------------------------------------------------------------------------
def _participant_row(participant: _Participant) -> Any:
    if participant.stoichiometry is None or participant.stoichiometry == 1:
        return participant.name
    return {"name": participant.name, "stoichiometry": participant.stoichiometry}


def _enzyme_actor(name: str, reaction: _Reaction) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "entity": name,
        "entity_type": "protein",
        "role": "catalyst",
    }
    _attach_provenance(row, reaction.provenance, reaction.evidence, reaction.scores)
    return row


def _reaction_row(reaction: _Reaction) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": reaction.name,
        "inputs": [_participant_row(p) for p in reaction.inputs],
        "outputs": [_participant_row(p) for p in reaction.outputs],
    }
    if reaction.enzymes:
        row["enzymes"] = [_enzyme_actor(e, reaction) for e in reaction.enzymes]
    # Only a real label is written. A reaction with no label gains no key at all,
    # which is the "absent" case every downstream reader treats as KEEP — the same
    # rule ``pipeline._carry_scope_membership`` follows, so a plain row's key set
    # is unchanged.
    if reaction.scope_membership:
        row["scope_membership"] = reaction.scope_membership
    _attach_provenance(
        row,
        reaction.provenance,
        reaction.evidence,
        reaction.scores,
        gap_id=reaction.gap_id,
        gap_ids=reaction.gap_ids,
    )
    # Attribution, AFTER ``_attach_provenance`` (see the lineage section). A RAG
    # import gets a ``rag_retrieval`` entry; a SEED reaction gets none from this
    # stage, because synthesis did not introduce it and does not know what did --
    # it only re-emits what the seed row already carried. ``support`` is ``direct``
    # when a span states the claim: the admission gate refuses an empty span, so an
    # admitted row always has one, but it is checked rather than assumed.
    retrieval = None
    if reaction.origin == "rag" and reaction.provenance:
        gaps = _dedupe_strs(
            list(reaction.gap_ids or []) + [reaction.gap_id]
        )
        retrieval = _retrieval_entry(
            reaction.provenance[0],
            "transcribed from a passage retrieved for gap(s) "
            + (", ".join(gaps) if gaps else "(none recorded)"),
            support="direct" if _text(reaction.evidence_span) else "indirect",
        )
    _write_lineage(row, *reaction.lineage, retrieval)
    # Reaction rows legitimately carry inputs/outputs/enzymes on top of the
    # allowed additive/core keys.
    return row


# ---------------------------------------------------------------------------
# to_payload — assemble the standard Payload (seam S3).
# ---------------------------------------------------------------------------
def to_payload(
    entities: Dict[str, List[Dict[str, Any]]],
    reactions: List[_Reaction],
) -> "Payload":
    """Assemble the standard :class:`~t2pw.schema.Payload` shape.

    Only core buckets (``entities`` / ``processes``) are emitted; provenance is
    additive/optional on each row. No RAG-only *required* key is introduced.
    """
    payload: Dict[str, Any] = {
        "entities": dict(entities),
        "processes": {"reactions": [_reaction_row(r) for r in reactions]},
    }
    return payload


# Contextual scaffolding entity buckets that synthesis intentionally does NOT
# rebuild from evidence (organism / compartment / cell / tissue context). They
# are not reaction participants and carry no pathway-specific chemistry, so they
# are exempt from the provenance requirement (see ``_EVIDENCE_ENTITY_BUCKETS`` in
# ``t2pw.rag.provenance``) and are copied from the seed as-is. The evidence-built
# ``compounds`` / ``proteins`` / ``protein_complexes`` are deliberately absent
# here — those stay rebuilt from evidence and are never overwritten.
_SCAFFOLDING_ENTITY_BUCKETS = (
    "species",
    "subcellular_locations",
    "cell_types",
    "tissues",
)


def _carry_forward_scaffolding(payload: Dict[str, Any], seed_payload: Any) -> None:
    """Copy the seed's contextual scaffolding into the synthesized payload.

    Synthesis rebuilds only evidence-bound chemistry (``compounds`` /
    ``proteins``). The contextual buckets above — plus top-level
    ``biological_states`` (a :class:`~t2pw.schema.Payload` top-level key, *not*
    under ``entities``) — are never rebuilt, so without this carry-forward Stage
    2B mapping would produce a payload with zero ``species`` rows and
    ``validate_post_mapping`` would abort with ``species_required``. These buckets
    are copied as-is and never clobber the evidence-built compounds/proteins
    (the bucket names are disjoint). Guards against a non-dict seed / entities.
    """
    seed = seed_payload if isinstance(seed_payload, dict) else {}
    seed_entities = _safe_dict(seed.get("entities"))
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        entities = {}
        payload["entities"] = entities
    for bucket in _SCAFFOLDING_ENTITY_BUCKETS:
        value = seed_entities.get(bucket)
        if isinstance(value, list) and value and bucket not in entities:
            entities[bucket] = deepcopy(value)
    bio_states = seed.get("biological_states")
    if isinstance(bio_states, list) and bio_states and "biological_states" not in payload:
        payload["biological_states"] = deepcopy(bio_states)


# ---------------------------------------------------------------------------
# synthesize — the public entry points.
# ---------------------------------------------------------------------------
def _candidate_from_reaction(
    reaction: _Reaction,
    *,
    requested_pathway: str,
    requested_organism: str,
) -> RagReactionCandidate:
    """Turn a transcribed RAG reaction into an admission candidate.

    Every field the admission contract requires is filled here from what the
    transcription already carries — nothing is invented, and nothing is copied
    from the REQUEST into an OBSERVED field (the stamping bug
    ``t2pw.rag.acquire`` documents at length). ``organism`` is what the retrieved
    chunk itself reports; ``requested_organism`` is what the run asked for; the
    comparison between them is a third, separate field.
    """
    prov = reaction.provenance[0] if reaction.provenance else {}
    evidence = reaction.evidence[0] if reaction.evidence else {}
    paper = reaction.source_papers[0] if reaction.source_papers else {}

    # The LOCAL observation wins. A review is "about" E. coli and human cells at
    # paper level; the sentence backing one reaction is about one of them, and
    # that is the claim being admitted. Paper-level observed metadata (from the
    # eligibility screen, carried down the chunk) is the fallback for a span that
    # names no organism -- never the requested value, which is not evidence.
    span_organisms = organisms_in_span(reaction.evidence_span)
    observed_organisms = span_organisms or _dedupe_strs(
        list(reaction.observed_organisms)
        + ([reaction.organism] if reaction.organism else [])
    )
    paper_haystack = " ".join(
        [_text(paper.get("title")), _text(prov.get("source_title"))]
    )
    return RagReactionCandidate(
        gap_id=reaction.gap_id,
        gap_ids=list(reaction.gap_ids or ([reaction.gap_id] if reaction.gap_id else [])),
        name=reaction.name,
        inputs=[p.name for p in reaction.inputs],
        outputs=[p.name for p in reaction.outputs],
        enzymes=list(reaction.enzymes),
        reversible=bool(reaction.reversible),
        source_paper=dict(paper),
        evidence=dict(evidence),
        evidence_span=reaction.evidence_span,
        organism=reaction.organism,
        observed_organisms=list(observed_organisms),
        observed_pathways=list(reaction.observed_pathways),
        requested_pathway=_text(requested_pathway),
        requested_organism=_text(requested_organism),
        requested_pathway_match=compare_requested_pathway(
            requested_pathway,
            reaction.evidence_span,
            reaction.observed_pathways,
            paper_haystack,
        ),
        organism_match=compare_organism(requested_organism, observed_organisms),
        confidence=_confidence(reaction.scores, len(reaction.provenance)),
    )


def _dedupe_candidates(
    pairs: List[Tuple[RagReactionCandidate, _Reaction]]
) -> List[Tuple[RagReactionCandidate, _Reaction]]:
    """Collapse candidates that are the same claim from the same passage.

    The key is ``(gap_id, claim_identity, provenance_identity)`` — the canonical
    substrates/products/catalysts/direction PLUS which paper and which passage
    said it — never the reaction name. Two consequences, both intended:

    * the SAME claim from the SAME passage, transcribed twice (a chunk whose text
      repeats an equation, or an arrow parse and a prose extraction of one
      sentence), collapses to one candidate, so a duplicated passage cannot
      inflate anything downstream. This is the candidate-level sibling of
      :func:`_dedupe_evidence`, which protects the merged row, and both are kept:
      this one stops the duplicate becoming a second candidate at all, that one
      stops repeated passages accumulating on a row that legitimately merges.
    * the same claim from a DIFFERENT passage survives as its own candidate.
      That is corroboration, and it must reach :func:`_resolve_reactions` so the
      row ends up carrying both source pointers.

    ``gap_id`` STAYS in the key, deliberately, and C-059 measured why. The same
    span retrieved for two gaps is collapsed by
    :data:`~t2pw.rag.admission.REASON_DUPLICATE_ACROSS_GAPS`, which runs AFTER the
    admission gate has judged the claim against each gap separately. Collapsing
    here instead would judge it once, against whichever gap sorted first — and the
    gate's verdict is gap-dependent by construction (``_gap_type_verdict``: a
    reaction fills a connectivity gap and cannot fill an ``unmapped_enzyme`` one).
    A claim refused for the first gap and admissible for the second would be lost
    outright, which is a merge-rule-7 deletion of a legitimate recovery.
    """
    out: List[Tuple[RagReactionCandidate, _Reaction]] = []
    seen: set = set()
    for candidate, reaction in pairs:
        key = (
            candidate.gap_id,
            candidate.claim_identity(),
            candidate.provenance_identity(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append((candidate, reaction))
    return out


def synthesize_with_report(
    seed_payload: Any,
    evidence_bundles: Optional[List[Any]] = None,
    seed_context: Any = "",
    *,
    prose_extractor: Optional[Any] = None,
    synonym_resolver: Optional[Any] = None,
    gaps: Optional[List[Any]] = None,
    requested_pathway: str = "",
    requested_organism: str = "",
    admission_policy: Optional[AdmissionPolicy] = None,
    identity_verifier: Optional[Any] = None,
) -> SynthesisResult:
    """Full synthesis: returns the payload *plus* the reports that ride with it.

    See module docstring for the four steps. The returned payload has already
    passed ``validate_post_extraction`` (raising ``StageContractError`` if it
    could not) so the caller receives a Stage-2B-ready payload or a clear failure.

    ``synonym_resolver`` (optional, mirrors ``prose_extractor``) is a
    ``name -> grouping-token`` callable — see :mod:`t2pw.rag.synonyms`. When
    supplied it drives GROUPING/merge KEYS ONLY, so reactions that are duplicates
    except for a compound/enzyme SYNONYM collapse to one row; the emitted names are
    never rewritten. ``None`` (default) reproduces today's behavior byte-for-byte.

    Gap admission
    -------------
    Every reaction transcribed from evidence is a *candidate* and enters the
    payload only by passing :func:`t2pw.rag.admission.admit_candidates`. Rejected
    candidates are dropped **before** conflict resolution, stitch detection and
    entity building, so there is no path by which a rejected claim can re-enter:
    stitching operates on the accepted set only, and a rejected candidate cannot
    donate its provenance to an accepted row through :func:`_merge_into` because
    it is not in the list being resolved.

    ``gaps`` are the detected gaps the bundles were retrieved for; when omitted
    they are taken from the bundles themselves, so a caller that already holds the
    gap list and one that only holds the bundles both get the same admission.
    ``requested_pathway`` / ``requested_organism`` are the RUN's request, used for
    the pathway/organism comparisons; ``admission_policy`` defaults to
    :func:`t2pw.rag.admission.policy_from_config`.
    """
    bundles = list(evidence_bundles or [])
    seed_source = _seed_source_descriptor(seed_context)
    extractor = _make_memoized_extractor(prose_extractor)

    seed_rxns, seed_omitted = _seed_reactions(seed_payload, seed_source)

    known_gaps = list(gaps) if gaps else _gaps_from_bundles(bundles)
    policy = admission_policy
    if policy is None:
        from t2pw.rag.admission import policy_from_config

        policy = policy_from_config()

    pairs: List[Tuple[RagReactionCandidate, _Reaction]] = []
    for bundle in bundles:
        for reaction in _reactions_from_bundle(bundle, extractor):
            pairs.append(
                (
                    _candidate_from_reaction(
                        reaction,
                        requested_pathway=requested_pathway,
                        requested_organism=requested_organism,
                    ),
                    reaction,
                )
            )
    pairs = _dedupe_candidates(pairs)

    accepted_candidates, admission_report = admit_candidates(
        [candidate for candidate, _ in pairs],
        gaps=known_gaps,
        seed_payload=seed_payload,
        policy=policy,
        # The same GROUPING-only resolver the merge uses, so the gate's
        # graph-connection rules see the same two spellings of one compound as
        # one node that ``_resolve_reactions`` will later merge.
        name_resolver=synonym_resolver,
    )
    # Typed-gap proposals come from the BUNDLES, not from rejected reactions: the
    # sentence that answers "which protein is EnzX?" usually states no reaction at
    # all, so a candidate-derived scan would miss exactly the evidence the gap
    # asked for. Whatever the gate also lifted out of a rejected reaction's span is
    # unioned in, deduplicated by (gap, kind, value).
    typed_resolutions = _collect_typed_resolutions(
        bundles,
        known_gaps,
        list(getattr(admission_report, "proposals", []) or []),
        seed_payload=seed_payload,
        policy=policy,
    )
    accepted_ids = {id(candidate) for candidate in accepted_candidates}
    evidence_rxns: List[_Reaction] = []
    for candidate, reaction in pairs:
        if id(candidate) not in accepted_ids:
            continue
        # The gate's verdict travels with the reaction: the label is what the
        # core out-of-scope filter and the lock manifest read downstream, and a
        # reversibility the gate normalized to the evidence has to reach the
        # payload or the row would ship one-way chemistry the paper called
        # reversible.
        reaction.scope_membership = candidate.scope_membership
        reaction.reversible = bool(candidate.reversible)
        # C-059: the gate may now collapse one canonical claim that several gaps
        # each retrieved (``REASON_DUPLICATE_ACROSS_GAPS``). The sibling rows
        # therefore never reach :func:`_resolve_reactions`, and the union that
        # merge used to perform has to happen here instead -- both halves of it,
        # or the collapse silently costs the row something:
        #
        # * ``gap_ids``, or a gap this delivered reaction genuinely fills is
        #   reported unfilled (pinned by
        #   ``test_rag_admission_adversarial.py::test_one_claim_admitted_for_two_gaps_keeps_both_attributions``);
        # * the collapsed group's best retrieval score, which ``_confidence``
        #   maxes over, or the surviving row ships a lower ``rag_confidence``
        #   than the same evidence produced before.
        #
        # Guarded on the union actually GROWING, so a claim the gate did not
        # collapse takes this branch never and its row is unchanged.
        merged_gap_ids = _dedupe_strs(
            list(reaction.gap_ids or []) + list(candidate.gap_ids or [])
        )
        if len(merged_gap_ids) > len(list(reaction.gap_ids or [])):
            reaction.gap_ids = merged_gap_ids
            if candidate.confidence:
                reaction.scores = list(reaction.scores) + [
                    float(candidate.confidence)
                ]
        # The gate's ATTRIBUTION travels with it too, and this is the only seam
        # where it can: the candidate holds the verdict (which gap it was admitted
        # against, the reasons, the hop, the organism/pathway comparisons) and the
        # reaction is what becomes the payload row. Reconstructing this later from
        # the row's ``scope_membership`` alone would lose every one of those facts.
        # Reads the verdict, never makes one.
        entry = admission_lineage_entry(candidate)
        if entry is not None:
            reaction.lineage.append(entry.as_dict())
        evidence_rxns.append(reaction)

    all_rxns = seed_rxns + evidence_rxns
    resolved, conflicts = _resolve_reactions(all_rxns, synonym_resolver)
    stitched = _detect_stitches(resolved)
    entities, entity_omitted = _build_entities(resolved, synonym_resolver, seed_payload)

    payload = to_payload(entities, resolved)
    # Carry the seed's contextual scaffolding (species / compartments / cell
    # types / tissues / biological_states) forward so the downstream mapping
    # stage still has the species row its contract requires (Defect 1). These are
    # not evidence-bound chemistry, so they are copied as-is.
    _carry_forward_scaffolding(payload, seed_payload)

    # Typed-gap resolutions are applied to the PAYLOAD -- or not. Only what
    # actually lands in it can close a typed gap, which is why this runs before
    # the unresolved-gap report is derived.
    resolution_records = _apply_typed_resolutions(
        payload, typed_resolutions, identity_verifier=identity_verifier
    )
    admission_dict = admission_report.to_dict()
    admission_dict["resolutions"] = resolution_records

    unresolved = list(seed_omitted) + list(entity_omitted)
    unresolved.extend(
        _unfilled_gap_reports(bundles, known_gaps, accepted_candidates, payload)
    )
    admission_dict["identity_outcomes"] = _identity_outcomes(
        resolution_records, payload, unresolved
    )

    contract_report = validate_post_extraction(payload)  # raises on structural fail

    return SynthesisResult(
        payload=payload,
        unresolved_gaps=unresolved,
        conflicts=conflicts,
        stitched=stitched,
        contract_report=contract_report,
        admission=admission_dict,
    )


def synthesize(
    seed_payload: Any,
    evidence_bundles: Optional[List[Any]] = None,
    seed_context: Any = "",
    *,
    prose_extractor: Optional[Any] = None,
    synonym_resolver: Optional[Any] = None,
    gaps: Optional[List[Any]] = None,
    requested_pathway: str = "",
    requested_organism: str = "",
    admission_policy: Optional[AdmissionPolicy] = None,
) -> "Payload":
    """Merge seed + evidence into one connected, validated standard ``Payload``.

    This is the seam-S3 entry point named in wp5_synthesis.md. The unresolved-
    gaps report and conflict record are exposed via
    :func:`synthesize_with_report`; this function returns only the ``Payload`` so
    its type matches the brief exactly. ``prose_extractor`` (optional) is the
    LLM prose→reaction callable; omit it for arrow-only synthesis.
    ``synonym_resolver`` (optional) is the synonym-canonical GROUPING resolver —
    see :func:`synthesize_with_report`; omit it for today's exact grouping.
    """
    return synthesize_with_report(
        seed_payload,
        evidence_bundles,
        seed_context,
        prose_extractor=prose_extractor,
        synonym_resolver=synonym_resolver,
        gaps=gaps,
        requested_pathway=requested_pathway,
        requested_organism=requested_organism,
        admission_policy=admission_policy,
    ).payload


# ---------------------------------------------------------------------------
# Typed-gap resolution: applying it to the payload, and reading closure back off.
# ---------------------------------------------------------------------------
def _find_entity_row(payload: Dict[str, Any], name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return ``(bucket, row)`` for the entity named ``name``, or ``("", None)``."""
    target = canonical_name(name).casefold()
    entities = _safe_dict(payload.get("entities"))
    for bucket in (
        "proteins",
        "protein_complexes",
        "compounds",
        "element_collections",
        "nucleic_acids",
    ):
        for row in _safe_list(entities.get(bucket)):
            if isinstance(row, dict) and canonical_name(row.get("name")).casefold() == target:
                return bucket, row
    return "", None


#: Entity bucket -> the ``element_locations`` bucket and field that addresses it.
#: Mirrors ``t2pw.pwml.ir._entity_type_from_location_bucket``, which is the
#: authority: it maps ``protein_locations`` to the entity type ``protein``, so
#: the IR resolves that field against ``entities.proteins`` and nothing else.
#:
#: ``protein_complexes`` is DELIBERATELY absent. It used to map here to
#: ``("protein_locations", "protein")``, which put a complex name in a field the
#: IR resolves against the protein bucket — an unresolvable reference dressed as
#: a located entity. A complex whose compartment is known has no representation
#: in this schema, so the proposal is refused with
#: :data:`~t2pw.rag.admission.REASON_UNSUPPORTED_TARGET_TYPE` and the gap stays
#: open, which is the honest report.
_LOCATION_BUCKETS = {
    "compounds": ("compound_locations", "compound"),
    "element_collections": ("element_collection_locations", "element_collection"),
    "nucleic_acids": ("nucleic_acid_locations", "nucleic_acid"),
    "proteins": ("protein_locations", "protein"),
}

#: The bucket a UniProt/DrugBank identity may be written to. One entry, stated as
#: a constant so the rule is greppable rather than implied by an ``if``.
_IDENTITY_BUCKET = "proteins"

# --- machine-readable identity outcome codes --------------------------------
# Stamped on the record by :func:`_apply_identity_decision`, which is the only
# place that KNOWS which branch was taken, and consumed verbatim by
# :func:`_identity_outcomes`. Deriving these by matching the human-readable
# reason text would tie a diagnostic count to prose that exists to be read, and
# would silently miscount the moment a sentence is reworded — the two branches
# below both used to say "identity not verified: ...".
IDENTITY_OUTCOME_APPLIED = "applied"
#: No verifier / candidate provider was available at all. The claim was never
#: judged; production is deliberately here (the RAG chain yields passages, not
#: resolver candidates).
IDENTITY_OUTCOME_NO_RESOLVER = "unverified_no_resolver"
#: A wired resolver RAN and refused: missing candidate evidence, name, species,
#: score, margin, ambiguity, or any other rung of the ladder. A judged rejection
#: is a different fact from an unasked question, and only this one says anything
#: about the claim.
IDENTITY_OUTCOME_RESOLVER_REJECTED = "rejected_by_identity_resolver"
#: Refused before any resolver was consulted: the target is not in the payload,
#: or is the wrong shape of row for an identity.
IDENTITY_OUTCOME_SCOPE_OR_TYPE = "rejected_scope_or_type"
#: Incompatible values for the same singular field, with no single winner.
IDENTITY_OUTCOME_CONFLICTING = "conflicting"
#: The REFUSAL codes a verifier may hand back. A verified result needs no code —
#: the boolean says it — so only these two are read off the verifier, and an
#: unrecognized one falls through to the safe default rather than silently
#: becoming a new bucket.
_IDENTITY_REFUSAL_CODES = frozenset(
    {IDENTITY_OUTCOME_NO_RESOLVER, IDENTITY_OUTCOME_RESOLVER_REJECTED}
)


def _has_complex_external_identity(row: Any) -> bool:
    """Whether a ``protein_complexes`` row carries a real complex-level identity.

    The predicate ``t2pw.pwml.ir`` itself applies when deciding whether a complex
    may stand without listed components: a PathBank/PathWhiz complex id. A UniProt
    accession is not one of these, which is the whole point — no sentence in a
    paper can supply a complex identity, so a complex identity gap is never closed
    by retrieval and keeps the functional-complex / Unknown-protein-component
    fallback that carries it honestly.
    """
    if not isinstance(row, dict):
        return False
    mapped = row.get("mapped_ids")
    mapped = mapped if isinstance(mapped, dict) else {}
    for key in (
        "pathbank_complex_id",
        "pathbank_protein_complex_id",
        "pw_complex_id",
        "pathwhiz_id",
    ):
        for source in (row, mapped):
            try:
                if int(source.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


# ---------------------------------------------------------------------------
# Reconciliation: ONE decision per (gap_id, kind, target), before any mutation.
# ---------------------------------------------------------------------------
#: Identity fields that hold exactly one value. Two distinct values for one of
#: these is a contradiction, not extra information — a protein has one UniProt
#: accession. ``uniprot`` and ``drugbank`` are export identity; ``ec`` is
#: annotation, and coexists with either because it says a different thing.
_SINGULAR_IDENTITY_FIELDS = ("uniprot", "drugbank", "ec")


def _member_sources(members: List[Any]) -> List[str]:
    """Every source id behind a group of proposals, deduplicated and ordered."""
    return _dedupe_strs(
        _text(_safe_dict(getattr(m, "evidence", {})).get("source_id")) for m in members
    )


def _member_span(members: List[Any]) -> str:
    """The span the merged mutation records. Members arrive sorted, so this is
    the same span under any retrieval order."""
    for member in members:
        span = _text(getattr(member, "evidence_span", ""))
        if span:
            return span
    return ""


def _identity_value(proposal: Any) -> Dict[str, str]:
    """A proposal's identifier value, normalized to the singular fields."""
    value = _safe_dict(getattr(proposal, "value", {}))
    out: Dict[str, str] = {}
    for field_name in _SINGULAR_IDENTITY_FIELDS:
        text = _text(value.get(field_name))
        if text:
            out[field_name] = text
    return out


def _participant_set(proposal: Any) -> Tuple[str, Tuple[str, ...]]:
    """A precursor proposal's ``(side, canonical participant set)``."""
    value = _safe_dict(getattr(proposal, "value", {}))
    names = [
        _text(p) for p in _safe_list(value.get("participants")) if _text(p)
    ] or ([_text(value.get("participant"))] if _text(value.get("participant")) else [])
    return (
        _text(value.get("side")) or "inputs",
        tuple(sorted({canonical_name(n).casefold() for n in names})),
    )


def _describe_values(variants: List[Dict[str, Any]], label: str) -> str:
    """"'P12345' (PMC_A), 'Q99999' (PMC_B)" — both values AND both sources."""
    parts = []
    for variant in variants:
        sources = ", ".join(_member_sources(variant["members"])) or "unknown source"
        parts.append(f"{variant[label]!r} ({sources})")
    return "; ".join(parts)


def _reconcile_typed_resolutions(proposals: List[Any]) -> List[Dict[str, Any]]:
    """Group admissible proposals and decide, per group, what the payload gets.

    Sorting the proposals made the ORDER deterministic; it did not make the
    OUTCOME evidence-based. Applying a group one member at a time still let the
    first member write and a contradicting second member overwrite it or be
    reported applied on top of it — a deterministic arbitrary write, where the
    winner is whichever value sorts first. So the group is classified first:

    * **corroborating** — the same scientific value from several papers. One
      mutation, provenance merged, every evidence record kept in the report.
    * **complementary** — values that say different things and can coexist
      (a verified UniProt accession and an EC annotation). Merged into one value.
    * **incompatible** — distinct values for the same singular field. Nothing is
      applied unless a resolver picks exactly one winner, and both values are
      reported with their sources.

    Returns one decision per thing-to-write, in a deterministic order. Members
    arrive already sorted by :func:`_proposal_identity`, so grouping preserves
    that order and two permutations of the same retrieval reconcile identically.
    """
    groups: Dict[Tuple[str, str, str], List[Any]] = {}
    for proposal in proposals:
        key = (
            _text(getattr(proposal, "gap_id", "")),
            _text(getattr(proposal, "kind", "")),
            canonical_name(getattr(proposal, "target", "")).casefold(),
        )
        groups.setdefault(key, []).append(proposal)

    decisions: List[Dict[str, Any]] = []
    for (gap_id, kind, _folded), members in groups.items():
        target = _text(getattr(members[0], "target", ""))
        base = {"gap_id": gap_id, "kind": kind, "target": target, "members": members}
        if kind == RESOLUTION_IDENTIFIER:
            decisions.append(_reconcile_identity(base, members))
        elif kind == RESOLUTION_PRECURSOR:
            decisions.append(_reconcile_precursor(base, members))
        elif kind == RESOLUTION_COMPARTMENT:
            decisions.extend(_reconcile_compartment(base, members))
        else:  # pragma: no cover - defensive
            decisions.append(dict(base, action="apply", value={}))
    return decisions


def _reconcile_identity(base: Dict[str, Any], members: List[Any]) -> Dict[str, Any]:
    """Corroborating / complementary identifiers merge; distinct ones conflict."""
    values = [(m, _identity_value(m)) for m in members]
    distinct: Dict[str, List[str]] = {}
    for field_name in _SINGULAR_IDENTITY_FIELDS:
        seen = _dedupe_strs(value.get(field_name, "") for _m, value in values)
        if seen:
            distinct[field_name] = seen

    contested = [f for f, seen in distinct.items() if len(seen) > 1]
    if not contested:
        # Every field agrees (or only one member states it): one mutation.
        merged = {f: seen[0] for f, seen in distinct.items()}
        return dict(base, action="apply", value=merged)

    variants: Dict[Tuple[str, ...], List[Any]] = {}
    for member, value in values:
        variants.setdefault(tuple(value.get(f, "") for f in contested), []).append(member)
    rendered = [
        {
            "value": {
                f: seen[0]
                for f, seen in (
                    (f, _dedupe_strs(_identity_value(m).get(f, "") for m in ms))
                    for f in _SINGULAR_IDENTITY_FIELDS
                )
                if seen
            },
            "label": ", ".join(v for v in key if v),
            "members": ms,
        }
        for key, ms in variants.items()
    ]
    return dict(
        base,
        action="adjudicate",
        variants=rendered,
        reasons=[
            f"{REASON_CONFLICTING_RESOLUTION}: {base['target']!r} has "
            f"{len(rendered)} incompatible proposed identities for "
            f"{contested} — {_describe_values(rendered, 'label')}"
        ],
    )


def _reconcile_precursor(base: Dict[str, Any], members: List[Any]) -> Dict[str, Any]:
    """Identical participant sets merge; different chemistry conflicts."""
    clusters: Dict[Tuple[str, Tuple[str, ...]], List[Any]] = {}
    for member in members:
        clusters.setdefault(_participant_set(member), []).append(member)
    if len(clusters) == 1:
        return dict(base, action="apply", value=dict(_safe_dict(members[0].value)))

    rendered = [
        {
            # Labelled with the names as the papers wrote them; the casefolded
            # canonical set is the grouping key, not something to report back.
            "label": "{}={}".format(
                side,
                sorted(
                    _dedupe_strs(
                        _text(p)
                        for m in ms
                        for p in _safe_list(_safe_dict(m.value).get("participants"))
                    )
                ),
            ),
            "members": ms,
        }
        for (side, _names), ms in clusters.items()
    ]
    return dict(
        base,
        action="conflict",
        reasons=[
            f"{REASON_CONFLICTING_RESOLUTION}: {base['target']!r} has "
            f"{len(rendered)} incompatible proposed participant sets for its "
            f"missing side — {_describe_values(rendered, 'label')}. Applying the "
            "one that sorts first would be an arbitrary write, not a repair"
        ],
    )


def _reconcile_compartment(
    base: Dict[str, Any], members: List[Any]
) -> List[Dict[str, Any]]:
    """One decision per DISTINCT location; identical ones merge.

    Two locations for one element is not a contradiction in this schema. The
    PWML IR keys a location by ``(entity_type, entity_key, biological_state_key)``
    (``t2pw.pwml.ir``: ``explicit_locations_to_register``), so an element in the
    periplasm and in the cytosol resolves to two distinct locations with
    ``unresolved["biological_state_references"] == []`` and a passing required
    contract. Each therefore gets its OWN ``element_locations`` row and keeps its
    own sources, rather than one being written and both reported applied.
    """
    clusters: Dict[str, List[Any]] = {}
    for member in members:
        location = _text(_safe_dict(getattr(member, "value", {})).get("location"))
        clusters.setdefault(location, []).append(member)
    return [
        dict(base, action="apply", members=ms, value={"location": location})
        for location, ms in clusters.items()
    ]


# ---------------------------------------------------------------------------
# Application.
# ---------------------------------------------------------------------------
def _apply_typed_resolutions(
    payload: Dict[str, Any],
    resolutions: List[Any],
    *,
    identity_verifier: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Apply the RECONCILED typed resolutions, and report what stuck.

    A proposal is only a reading of a sentence. This is where a *group* of
    proposals either becomes a real change to the payload — in the schema's own
    representation — or does not, and the difference is what decides whether the
    gap is closed:

    * **identifier** — the accession is handed to ``identity_verifier`` (the
      caller's protein identity policy / resolver). Only a VERIFIED identity is
      written to the protein's ``mapped_ids``. With no verifier wired the identity
      is unverified, nothing is written, and the gap stays open so the existing
      Unknown-protein fallback keeps working. An accession appearing in a sentence
      is not an identity resolution, and two accessions are not an identity at all
      unless the resolver names exactly one winner.
    * **compartment** — written as a ``subcellular_locations`` entry, a biological
      state, *and* an ``element_locations`` row against the referenced entity,
      which is how the schema expresses a location. Marked applied only once that
      exact target + state pair is really in ``element_locations``.
    * **precursor** — the named INCOMPLETE reaction is patched with the evidenced
      participants, keeping its provenance. Adding a separate reaction would leave
      the incomplete one exactly as incomplete as it was.

    Returns one record per PROPOSAL — corroborating proposals share an outcome but
    keep their individual evidence rows — in the order they were given. ``payload``
    is mutated in place; it is this module's own freshly built payload.
    """
    records: Dict[int, Dict[str, Any]] = {}
    active: List[Any] = []
    for proposal in resolutions:
        record = proposal.to_dict()
        record["applied"] = False
        records[id(proposal)] = record
        if getattr(proposal, "status", "") == STATUS_REJECTED_PROPOSAL:
            # Refused by scope/type admission before it could touch anything. Kept
            # in the report — a refusal nobody can see is indistinguishable from a
            # proposal that was never made.
            continue
        active.append(proposal)

    def _mark(members, applied, reasons, *, conflicting=False, outcome="") -> None:
        for member in members:
            record = records[id(member)]
            record["applied"] = bool(applied)
            record["reasons"] = list(reasons)
            if conflicting:
                record["conflicting"] = True
            if outcome:
                record["identity_outcome"] = outcome

    for decision in _reconcile_typed_resolutions(active):
        if decision["action"] == "conflict":
            _mark(
                decision["members"],
                False,
                decision["reasons"],
                conflicting=True,
                outcome=IDENTITY_OUTCOME_CONFLICTING,
            )
            continue
        kind = decision["kind"]
        if kind == RESOLUTION_IDENTIFIER:
            _apply_identity_decision(payload, decision, identity_verifier, _mark)
        elif kind == RESOLUTION_COMPARTMENT:
            _apply_compartment_decision(payload, decision, _mark)
        elif kind == RESOLUTION_PRECURSOR:
            _apply_precursor_decision(payload, decision, _mark)
        else:  # pragma: no cover - defensive
            _mark(decision["members"], False, [f"unknown resolution kind {kind!r}"])

    out: List[Dict[str, Any]] = []
    for proposal in resolutions:
        record = records[id(proposal)]
        if getattr(proposal, "status", "") != STATUS_REJECTED_PROPOSAL:
            proposal.applied = bool(record["applied"])
            if proposal.applied:
                proposal.status = STATUS_APPLIED
            elif record.get("conflicting"):
                proposal.status = STATUS_REJECTED_PROPOSAL
            else:
                proposal.status = STATUS_PROPOSED
        record["status"] = proposal.status
        out.append(record)
    return out


def _apply_identity_decision(payload, decision, identity_verifier, mark) -> None:
    """Write at most ONE identity for the target, or write none and say why.

    Every branch stamps a machine-readable :data:`IDENTITY_OUTCOME_APPLIED`-style
    code alongside its prose reason. This is the only place that knows which
    branch was taken — in particular, whether a resolver was ASKED — so it is the
    only place that can record it without guessing.
    """
    target = decision["target"]
    members = decision["members"]
    bucket, row = _find_entity_row(payload, target)
    if row is None:
        mark(
            members,
            False,
            [f"identity not applied: {target!r} is not an entity of this payload"],
            outcome=IDENTITY_OUTCOME_SCOPE_OR_TYPE,
        )
        return
    if bucket != _IDENTITY_BUCKET:
        # Enforced at the WRITE, independently of the gap contract that already
        # screened for it. A UniProt accession names a protein; putting one on a
        # ``protein_complexes`` row makes ``identity_status`` report ``verified``
        # for a complex whose real identity is still unknown.
        mark(
            members,
            False,
            [
                f"{REASON_UNSUPPORTED_TARGET_TYPE}: {target!r} is in "
                f"entities.{bucket}, and a UniProt/EC identity may only be written "
                f"to entities.{_IDENTITY_BUCKET}"
            ],
            outcome=IDENTITY_OUTCOME_SCOPE_OR_TYPE,
        )
        return

    def _verify(value: Dict[str, Any]) -> Tuple[bool, str]:
        """``(verified, outcome code)`` — never "guess from the reason text".

        The code comes from the verifier itself when it supplies one
        (:class:`t2pw.rag.identity.IdentityCheck`), because only the verifier knows
        whether it had candidate evidence to judge. A bare ``bool``-returning
        callable — a test stub — has no code to give, and a wired stub that says no
        HAS judged, so its refusal is a resolver rejection.
        """
        if identity_verifier is None:
            return False, IDENTITY_OUTCOME_NO_RESOLVER
        try:
            result = identity_verifier(target, dict(value))
        except Exception:  # noqa: BLE001 - a resolver that crashed judged nothing
            return False, IDENTITY_OUTCOME_NO_RESOLVER
        if result:
            return True, IDENTITY_OUTCOME_APPLIED
        code = _text(getattr(result, "identity_outcome", ""))
        return False, (
            code if code in _IDENTITY_REFUSAL_CODES else IDENTITY_OUTCOME_RESOLVER_REJECTED
        )

    if decision["action"] == "adjudicate":
        # FAIL CLOSED. Two accessions are only resolvable by a resolver that
        # names exactly one winner; a verifier that confirms both has confirmed
        # nothing, and neither is written.
        winners = [v for v in decision["variants"] if _verify(v["value"])[0]]
        if len(winners) != 1:
            detail = (
                "no identity resolver is wired to adjudicate"
                if identity_verifier is None
                else f"the identity policy confirmed {len(winners)} of them"
            )
            mark(
                members,
                False,
                [f"{decision['reasons'][0]}; {detail}, so neither is applied"],
                conflicting=True,
                outcome=IDENTITY_OUTCOME_CONFLICTING,
            )
            return
        winner = winners[0]
        losers = [m for m in members if m not in winner["members"]]
        _write_identity(payload, row, winner["value"], winner["members"])
        mark(
            winner["members"],
            True,
            [
                f"identity verified against {len(decision['variants'])} competing "
                f"proposals and written to {bucket}"
            ],
            outcome=IDENTITY_OUTCOME_APPLIED,
        )
        mark(
            losers,
            False,
            [decision["reasons"][0]],
            conflicting=True,
            outcome=IDENTITY_OUTCOME_CONFLICTING,
        )
        return

    value = _safe_dict(decision["value"])
    ok, code = _verify(value)
    if not ok:
        if code == IDENTITY_OUTCOME_NO_RESOLVER:
            # NOT the same fact as a rejection. Nothing judged this claim — no
            # verifier, or a verifier with no candidate evidence to weigh.
            # Production sits here on purpose: the RAG chain yields passages, not
            # resolver candidates.
            reason = (
                "identity not verified: no identity resolver evidence was "
                "available, so the accession stays a claim and the gap keeps the "
                "Unknown-protein fallback"
            )
        else:
            # A wired resolver ran the full ladder and refused — missing candidate
            # evidence for THIS accession, wrong species, an implausible name, too
            # low a score, too thin a margin. That is a judgement about the claim.
            reason = (
                "identity rejected by the identity resolver: the ladder did not "
                f"confirm {value!r} for {target!r}"
            )
        mark(members, False, [reason], outcome=code)
        return
    _write_identity(payload, row, value, members)
    mark(
        members,
        True,
        [
            f"identity verified and written to {bucket} "
            f"(corroborated by {len(_member_sources(members))} source(s))"
        ],
        outcome=IDENTITY_OUTCOME_APPLIED,
    )


def _write_identity(payload, row, value, members) -> None:
    """One mutation, provenance merged across every proposal behind it."""
    mapped = row.get("mapped_ids")
    if not isinstance(mapped, dict):
        mapped = {}
    for key in _SINGULAR_IDENTITY_FIELDS:
        if value.get(key):
            mapped[key] = value[key]
    row["mapped_ids"] = mapped
    refs = _dedupe_strs(list(row.get("source_refs") or []) + _member_sources(members))
    if refs:
        row["source_refs"] = refs


def _apply_compartment_decision(payload, decision, mark) -> None:
    """Write ONE location, and only claim it once the reference really exists."""
    target = decision["target"]
    members = decision["members"]
    location = _text(_safe_dict(decision["value"]).get("location"))
    sources = _member_sources(members)
    span = _member_span(members)
    bucket, row = _find_entity_row(payload, target)
    species = _payload_species(payload)

    if row is None or not location:
        mark(members, False, [
            f"compartment not applied: {target!r} is not an entity of this "
            "payload, so there is nothing to locate"
        ])
        return
    if bucket not in _LOCATION_BUCKETS:
        mark(members, False, [
            f"{REASON_UNSUPPORTED_TARGET_TYPE}: {target!r} is in entities.{bucket}, "
            "which element_locations has no bucket for; writing it into "
            "protein_locations.protein would be a reference the PWML IR resolves "
            "against entities.proteins and cannot find"
        ])
        return
    if not species:
        # ``_ensure_biological_state`` needs a species to build a state, and a
        # location row pointing at a state that does not exist is a dangling
        # reference the PWML IR would refuse. Better to leave the gap open than to
        # write half a structure.
        mark(members, False, [
            "compartment not applied: the payload declares no species, so no "
            "biological state can be constructed for the location"
        ])
        return

    # Everything that could refuse has refused. Only now is anything created, so
    # no unused state or subcellular-location row is left behind by a refusal.
    entities = payload.setdefault("entities", {})
    locations = entities.setdefault("subcellular_locations", [])
    loc_row = next(
        (
            r
            for r in locations
            if isinstance(r, dict)
            and _text(r.get("name")).casefold() == location.casefold()
        ),
        None,
    )
    if loc_row is None:
        loc_row = {"name": location}
        if span:
            loc_row["evidence"] = span
        locations.append(loc_row)
    if sources:
        loc_row["source_refs"] = _dedupe_strs(
            list(loc_row.get("source_refs") or []) + sources
        )

    # The biological state is built by the repository's own helper, so its NAME,
    # its ``compartment_canonical`` and its reuse semantics are the ones every
    # other producer of states uses.
    state_name = _ensure_biological_state(payload, location, species)
    if not state_name:
        mark(members, False, [
            "compartment not applied: no biological state could be constructed "
            f"for {location!r} in {species!r}"
        ])
        return
    state_row = next(
        (
            st
            for st in _safe_list(payload.get("biological_states"))
            if isinstance(st, dict) and _text(st.get("name")) == state_name
        ),
        None,
    )
    if isinstance(state_row, dict):
        if sources:
            state_row["source_refs"] = _dedupe_strs(
                list(state_row.get("source_refs") or []) + sources
            )
        if span and not _text(state_row.get("evidence")):
            state_row["evidence"] = span

    loc_bucket, key = _LOCATION_BUCKETS[bucket]
    element_locations = payload.setdefault("element_locations", {})
    rows = element_locations.setdefault(loc_bucket, [])
    # Keyed by target AND state: a second location for the same element is a
    # second row, not a silent no-op reported as success.
    entry = next(
        (
            r
            for r in rows
            if isinstance(r, dict)
            and _text(r.get(key)).casefold() == target.casefold()
            and _text(r.get("biological_state")) == state_name
        ),
        None,
    )
    if entry is None:
        entry = {
            key: row.get("name") or target,
            # References the state BY NAME, which is what the PWML IR resolves
            # against ``biological_states``.
            "biological_state": state_name,
        }
        if span:
            entry["evidence"] = span
        rows.append(entry)
    if sources:
        entry["source_refs"] = _dedupe_strs(
            list(entry.get("source_refs") or []) + sources
        )

    # "Applied" means the reference EXISTS, re-read off the payload. Anything
    # weaker reports a location that no exporter can follow.
    written = any(
        isinstance(r, dict)
        and _text(r.get(key)).casefold() == target.casefold()
        and _text(r.get("biological_state")) == state_name
        for r in _safe_list(_safe_dict(payload.get("element_locations")).get(loc_bucket))
    )
    if not written:  # pragma: no cover - defensive
        mark(members, False, [
            f"compartment not applied: no element_locations.{loc_bucket} row "
            f"references {target!r} in state {state_name!r}"
        ])
        return
    mark(members, True, [
        f"location {location!r} written as biological state {state_name!r} plus "
        f"entities.subcellular_locations and element_locations.{loc_bucket}"
    ])


def _apply_precursor_decision(payload, decision, mark) -> None:
    """Patch the incomplete reaction once, with the complete participant set."""
    members = decision["members"]
    value = _safe_dict(decision["value"])
    participants = [
        _text(p) for p in _safe_list(value.get("participants")) if _text(p)
    ] or ([_text(value.get("participant"))] if _text(value.get("participant")) else [])
    side = _text(value.get("side")) or "inputs"
    reaction_name = _text(value.get("reaction")) or decision["target"]
    sources = _member_sources(members)

    # RE-FETCHED from the payload being delivered, not from whatever the proposal
    # saw when it was made. Synthesis rebuilds reactions and other resolutions run
    # before this one, so the row's shape at proposal time is not evidence about
    # its shape now.
    rows = _safe_list(_safe_dict(payload.get("processes")).get("reactions"))
    row = next(
        (
            r
            for r in rows
            if isinstance(r, dict)
            and _text(r.get("name")).casefold() == reaction_name.casefold()
        ),
        None,
    )
    if row is None or not participants:
        mark(members, False, [
            f"precursor not applied: reaction {reaction_name!r} is not in this "
            "payload, so there is nothing to repair"
        ])
        return
    current_side = missing_reaction_side(row)
    if current_side != side:
        # Covers all three: both sides populated (``""``), both sides empty
        # (``""``), and the OTHER side being the empty one. Appending to a
        # reaction that is no longer missing this side invents chemistry.
        mark(members, False, [
            f"{REASON_SIDE_NO_LONGER_MISSING}: the proposal repairs {side!r} of "
            f"{reaction_name!r}, but the reaction in the delivered payload is "
            f"missing {current_side or 'neither side'}"
        ])
        return

    names = {
        canonical_name(p if isinstance(p, str) else _safe_dict(p).get("name")).casefold()
        for p in _safe_list(row.get(side))
        + _safe_list(row.get("outputs" if side == "inputs" else "inputs"))
    }
    clashes = [p for p in participants if canonical_name(p).casefold() in names]
    if clashes:
        # ATOMIC: the evidence-stated set goes in whole or not at all.
        mark(members, False, [
            f"{REASON_MULTI_PARTICIPANT_REPAIR}: {reaction_name!r} already lists "
            f"{sorted(clashes)}, so the evidence-stated set {sorted(participants)} "
            "cannot be added in full"
        ])
        return

    added = [canonical_name(p) for p in participants]
    row.setdefault(side, [])
    row[side].extend(added)
    refs = _dedupe_strs(list(row.get("source_refs") or []) + sources)
    if refs:
        row["source_refs"] = refs
    # A participant a reaction references but no bucket registers is a dangling
    # reference -- the repair has to keep the payload referentially intact, so the
    # compound is registered too, with the same provenance the patch rests on.
    for name in added:
        _bucket, existing = _find_entity_row(payload, name)
        if existing is not None:
            continue
        compounds = payload.setdefault("entities", {}).setdefault("compounds", [])
        entity_row: Dict[str, Any] = {"name": name}
        if sources:
            entity_row["source_refs"] = list(sources)
        compounds.append(entity_row)
    mark(members, True, [
        f"{sorted(added)} added to {reaction_name!r}.{side} and registered as "
        f"compounds (corroborated by {len(sources)} source(s))"
    ])


def _has_unknown_component(row: Any) -> bool:
    """Whether a complex row already carries a PathBank Unknown-protein member."""
    if not isinstance(row, dict):
        return False
    for component in _safe_list(row.get("components")):
        if not isinstance(component, dict):
            continue
        try:
            if int(component.get("pathbank_protein_id") or 0) == PATHBANK_UNKNOWN_PROTEIN_ID:
                return True
        except (TypeError, ValueError):
            continue
        if is_pathbank_unknown_protein(component):
            return True
    return False


def _identity_outcomes(
    records: List[Dict[str, Any]],
    payload: Dict[str, Any],
    unresolved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Identity outcomes, in the units they are actually measured in.

    The previous version mixed two different populations in one dict: four of its
    keys counted PROPOSALS and one counted GAPS. Ten corroborating proposals for
    one protein read as ten verified identities, and an unmapped-enzyme gap that
    nothing had yet fallen back on was reported as an ``unknown_fallback`` that
    did not exist in the payload. So there are now three blocks, and none of them
    is comparable to another by accident:

    * ``proposals`` — one entry per proposal record. ``verified`` /
      ``annotation_only`` / ``rejected_scope_or_type`` / ``unverified_no_resolver``
      / ``rejected_by_identity_resolver`` / ``conflicting``.
    * ``targets`` — one entry per DISTINCT ``(gap_id, target)``, which is what a
      "resolution outcome" is actually about. Corroboration raises confidence, not
      the count.
    * the gap-level numbers, named for what they are. ``unresolved_identity_gaps``
      is every unmapped-enzyme gap the payload leaves open;
      ``unresolved_for_unknown_fallback`` is the subset whose row does NOT yet
      carry an Unknown sentinel or component — this stage does not insert one, so
      calling those "unknown_fallback" would report a payload state that is not
      there. ``unknown_fallback_present`` is the subset that already has it.

    The classification READS the ``identity_outcome`` code that
    :func:`_apply_identity_decision` stamped on each record. It does not inspect
    the reason prose: "no resolver was wired" and "the resolver ran and refused"
    are different facts about the claim, they were both phrased "identity not
    verified: ...", and a diagnostic that tells them apart by string prefix would
    be one reworded sentence away from silently merging them again.
    """
    proposals = {
        "verified": 0,
        "annotation_only": 0,
        "rejected_scope_or_type": 0,
        "unverified_no_resolver": 0,
        "rejected_by_identity_resolver": 0,
        "conflicting": 0,
    }
    per_target: Dict[Tuple[str, str], str] = {}

    def _rank(current: str, incoming: str) -> str:
        # The outcome a TARGET got is the strongest thing that happened to it: a
        # verified write is the outcome even when a second paper's contradicting
        # proposal was refused alongside it.
        order = [
            "verified",
            "annotation_only",
            "conflicting",
            "rejected_by_identity_resolver",
            "rejected_scope_or_type",
            "unverified_no_resolver",
        ]
        if not current:
            return incoming
        return order[min(order.index(current), order.index(incoming))]

    for record in records:
        if _text(record.get("kind")) != RESOLUTION_IDENTIFIER:
            continue
        code = _text(record.get("identity_outcome"))
        if code == IDENTITY_OUTCOME_APPLIED or record.get("applied"):
            # The only split not decided at the write: whether what landed is
            # EXPORT identity or annotation. Read off the delivered payload rather
            # than off the write, because a second gap resolving the same protein
            # can turn an EC-only row into a verified one after the fact.
            bucket, row = _find_entity_row(payload, _text(record.get("target")))
            outcome = (
                "verified"
                if bucket == _IDENTITY_BUCKET and has_protein_external_identity(row)
                else "annotation_only"
            )
        elif code in proposals:
            outcome = code
        else:  # pragma: no cover - defensive; every branch stamps a code
            outcome = "rejected_scope_or_type"
        proposals[outcome] += 1
        key = (_text(record.get("gap_id")), _text(record.get("target")).casefold())
        per_target[key] = _rank(per_target.get(key, ""), outcome)

    targets: Dict[str, int] = {name: 0 for name in proposals}
    for outcome in per_target.values():
        targets[outcome] += 1

    open_gaps = [
        row for row in unresolved if _text(row.get("kind")) == GAP_UNMAPPED_ENZYME
    ]
    with_sentinel = 0
    for gap_row in open_gaps:
        _bucket, row = _find_entity_row(payload, _text(gap_row.get("label")))
        if is_pathbank_unknown_protein(row) or _has_unknown_component(row):
            with_sentinel += 1
    return {
        "proposals": proposals,
        "targets": targets,
        "unresolved_identity_gaps": len(open_gaps),
        "unresolved_for_unknown_fallback": len(open_gaps) - with_sentinel,
        "unknown_fallback_present": with_sentinel,
    }


def _open_metabolites(payload: Dict[str, Any]) -> set:
    """Canonical non-cofactor metabolites the payload still leaves as dead ends.

    Produced-but-never-consumed, or consumed-but-never-produced. This is the same
    predicate ``retrieve._connectivity_gaps`` uses to CREATE a connectivity gap,
    so "the gap is closed" means exactly "re-detecting on this payload would no
    longer raise it" — read off the delivered payload rather than inferred from a
    candidate having been accepted.
    """
    produced: set = set()
    consumed: set = set()
    for row in _safe_list(_safe_dict(payload.get("processes")).get("reactions")):
        if not isinstance(row, dict):
            continue
        for side, bucket in (("inputs", consumed), ("outputs", produced)):
            for token in _safe_list(row.get(side)):
                name = token if isinstance(token, str) else _safe_dict(token).get("name")
                canonical = canonical_name(name).casefold()
                if canonical and canonical not in COFACTOR_NAMES:
                    bucket.add(canonical)
    return (produced - consumed) | (consumed - produced)


def _payload_species(payload: Dict[str, Any]) -> str:
    """The payload's declared species, or ``""`` (read-only)."""
    for row in _safe_list(_safe_dict(payload.get("entities")).get("species")):
        if isinstance(row, dict) and _text(row.get("name")):
            return _text(row["name"])
        if isinstance(row, str) and row.strip():
            return row.strip()
    return ""


def _payload_participants(payload: Dict[str, Any]) -> set:
    """Every canonical participant name the payload's reactions mention."""
    out: set = set()
    for row in _safe_list(_safe_dict(payload.get("processes")).get("reactions")):
        if not isinstance(row, dict):
            continue
        for side in ("inputs", "outputs"):
            for token in _safe_list(row.get(side)):
                name = token if isinstance(token, str) else _safe_dict(token).get("name")
                canonical = canonical_name(name).casefold()
                if canonical:
                    out.add(canonical)
    return out


def _payload_closes_gap(
    payload: Dict[str, Any],
    gap: Any,
    open_metabolites: set,
    participants: set,
) -> bool:
    """Is this gap actually closed BY THE PAYLOAD? (never "a candidate passed")."""
    kind = _text(getattr(gap, "kind", ""))
    label = _text(getattr(gap, "label", ""))
    if not label:
        return False

    if kind == GAP_ORPHAN_METABOLITE:
        folded = canonical_name(label).casefold()
        # ABSENT is not closed. A metabolite nothing in the payload mentions has
        # no dead end only because it has no end at all -- reporting that as
        # resolved is how "we invented nothing for this gap" would read as
        # "we solved this gap".
        return folded in participants and folded not in open_metabolites
    if kind == GAP_DANGLING_REACTION:
        targets = {
            canonical_name(n).casefold()
            for n in (getattr(gap, "target_names", list)() or [])
        }
        targets -= set(COFACTOR_NAMES)
        if not targets or not (targets & participants):
            return False
        return not (targets & open_metabolites)
    if kind == GAP_UNMAPPED_ENZYME:
        # The STRICT export definition, reused rather than re-implemented:
        # ``has_protein_external_identity`` is the same predicate the Stage-3
        # gate and the PathWhiz export apply, i.e. a real UniProt/DrugBank id.
        # An EC number is useful annotation and is NOT identity — a gap "closed"
        # on an EC number ships a protein the exporter still cannot resolve, and
        # silently removes it from the Unknown-fallback path that would have
        # carried it honestly.
        bucket, row = _find_entity_row(payload, label)
        if not isinstance(row, dict):
            return False
        if bucket == "protein_complexes":
            # A COMPLEX is not closed by a protein identity, and
            # ``has_protein_external_identity`` applied to a complex row would say
            # it was: the predicate reads ``mapped_ids.uniprot``, which nothing may
            # write here. A complex needs its own PathBank/PathWhiz id, and until
            # it has one the gap stays open on the functional-complex /
            # Unknown-protein-component fallback.
            return _has_complex_external_identity(row)
        if bucket != _IDENTITY_BUCKET:
            return False
        if is_pathbank_unknown_protein(row):
            return False
        return has_protein_external_identity(row)
    if kind == GAP_MISSING_COMPARTMENT:
        locations = _safe_dict(payload.get("element_locations"))
        folded = canonical_name(label).casefold()
        # A non-empty string is not closure: the state has to EXIST. A location
        # row naming a state the payload does not declare is a dangling reference,
        # which ``build_pwml_ir`` reports in
        # ``unresolved["biological_state_references"]`` and which no exporter can
        # follow -- reporting that as a resolved compartment would be a lie the
        # rest of the pipeline then trusts.
        declared = {
            _text(st.get("name"))
            for st in _safe_list(payload.get("biological_states"))
            if isinstance(st, dict) and _text(st.get("name"))
        }
        for bucket, key in _LOCATION_BUCKETS.values():
            for row in _safe_list(locations.get(bucket)):
                if not isinstance(row, dict):
                    continue
                if canonical_name(row.get(key)).casefold() != folded:
                    continue
                state = _text(row.get("biological_state"))
                if state and state in declared:
                    return True
        return False
    if kind == GAP_MISSING_PRECURSOR:
        # "Some novel participant is somewhere on this reaction" is not closure.
        # A participant appended to the side that was already populated leaves the
        # empty side exactly as empty as it was, and reads as a repair.
        missing_side = _text(getattr(gap, "missing_side", ""))
        if missing_side not in ("inputs", "outputs"):
            return False
        known_side = "outputs" if missing_side == "inputs" else "inputs"
        anchors = {
            canonical_name(n).casefold()
            for n in (getattr(gap, "target_names", list)() or [])
        } - set(COFACTOR_NAMES)

        def _names(row: Dict[str, Any], side: str) -> set:
            return {
                canonical_name(
                    p if isinstance(p, str) else _safe_dict(p).get("name")
                ).casefold()
                for p in _safe_list(row.get(side))
            }

        for row in _safe_list(_safe_dict(payload.get("processes")).get("reactions")):
            if not isinstance(row, dict):
                continue
            if _text(row.get("name")).casefold() != label.casefold():
                continue
            filled = _names(row, missing_side) - set(COFACTOR_NAMES)
            kept = _names(row, known_side)
            # 1. the side that was empty carries a real participant now;
            # 2. it is one the gap did not already know about;
            # 3. and the anchors the gap was detected around are STILL on the
            #    side they were on — a repair that moved them has not filled the
            #    gap, it has rewritten the reaction.
            return bool(filled) and bool(filled - anchors) and anchors <= kept
        return False
    return False


def _collect_typed_resolutions(
    bundles: List[Any],
    gaps: List[Any],
    extra: List[Any],
    *,
    seed_payload: Any = None,
    policy: Optional[AdmissionPolicy] = None,
) -> List[Any]:
    """Typed-gap proposals from every bundle, plus ``extra``, in a stable order.

    Bundles are scanned per gap so a passage retrieved for gap G is only ever read
    as evidence about G — the same attribution discipline the reaction path
    follows.

    Collapsing on ``(gap_id, kind, value)`` — which this did — makes the result
    depend on which passage was retrieved FIRST. Two papers can assert the same
    accession for the same protein while disagreeing about the organism: one is
    rejected on scope, the other is not, and whichever arrived first silently
    decided the outcome. Both failure directions are real. A foreign paper's
    rejected proposal suppressed the local paper's valid one, so a resolvable gap
    stayed open; and an unknown-scope proposal suppressed a later explicit
    mismatch, so contradicting evidence vanished from the report.

    So the key retains the scope verdict AND the provenance, and the result is
    SORTED by that key rather than by arrival. Identical evidence from the same
    chunk still collapses; genuinely different evidence is kept and reported, and
    the order two permutations of the same retrieval produce is the same order.
    """
    from t2pw.rag.admission import propose_typed_resolutions

    by_id = {_text(getattr(g, "gap_id", "")): g for g in gaps if getattr(g, "gap_id", "")}
    out: List[Any] = []
    seen: set = set()

    def _add(proposal: Any) -> None:
        key = _proposal_identity(proposal)
        if key in seen:
            return
        seen.add(key)
        out.append(proposal)

    for bundle in bundles:
        gap = _gap_of(bundle)
        gap_id = _text(getattr(gap, "gap_id", ""))
        gap = by_id.get(gap_id, gap)
        if gap is None or not gap_id:
            continue
        for hit in _safe_list(getattr(bundle, "hits", [])):
            chunk = _chunk_of(hit)
            if chunk is None:
                continue
            for proposal in propose_typed_resolutions(
                gap,
                _text(getattr(chunk, "text", "")),
                evidence=_evidence_from_hit(hit),
                source_paper=_source_paper_from_chunk(chunk),
                payload=seed_payload,
                observed_organisms=(
                    list(getattr(chunk, "observed_organisms", []) or [])
                    + ([_text(getattr(chunk, "organism", ""))] if _text(getattr(chunk, "organism", "")) else [])
                ),
                observed_pathways=list(getattr(chunk, "observed_pathways", []) or []),
                policy=policy,
            ):
                _add(proposal)
    for proposal in extra:
        _add(proposal)
    out.sort(key=_proposal_identity)
    return out


def _proposal_identity(proposal: Any) -> tuple:
    """The full identity of a typed proposal: what, under what scope, from where.

    Also the sort key. The scope verdict comes before provenance so an applicable
    proposal is always considered before a rejected one for the same value —
    which is what makes "a valid local proposal survives a rejected foreign one"
    an ordering guarantee rather than an accident.
    """
    value = _safe_dict(getattr(proposal, "value", {}))
    evidence = _safe_dict(getattr(proposal, "evidence", {}))
    return (
        _text(getattr(proposal, "gap_id", "")),
        _text(getattr(proposal, "kind", "")),
        tuple(sorted((str(k), str(v)) for k, v in value.items())),
        1 if _text(getattr(proposal, "status", "")) == STATUS_REJECTED_PROPOSAL else 0,
        _text(getattr(proposal, "organism_match", "")),
        _text(getattr(proposal, "requested_pathway_match", "")),
        _text(_safe_dict(getattr(proposal, "source_paper", {})).get("source_id")),
        _text(evidence.get("source_id")),
        _text(evidence.get("chunk_id")),
        _text(getattr(proposal, "evidence_span", "")),
    )


def _gap_of(bundle: Any) -> Any:
    return getattr(bundle, "gap", None)


def _gaps_from_bundles(bundles: List[Any]) -> List[Any]:
    """The distinct gaps the bundles were retrieved for, in first-seen order."""
    out: List[Any] = []
    seen: set = set()
    for bundle in bundles:
        gap = _gap_of(bundle)
        gap_id = _text(getattr(gap, "gap_id", ""))
        if gap is None or not gap_id or gap_id in seen:
            continue
        seen.add(gap_id)
        out.append(gap)
    return out


def _unfilled_gap_reports(
    bundles: List[Any],
    gaps: List[Any],
    accepted: List[RagReactionCandidate],
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Report every gap the DELIVERED PAYLOAD does not close.

    Closure is read off ``payload``, never inferred from a candidate having been
    accepted. That distinction is load-bearing for typed gaps, where the two come
    apart completely: a reaction can be admitted from a passage that also quotes
    "UniProt P12345" without the payload gaining a single ``mapped_ids`` entry, so
    "a candidate passed" would report an unmapped-enzyme gap as resolved while the
    protein is still unmapped and the Unknown-protein fallback is still the only
    thing standing behind it. :func:`_payload_closes_gap` asks the concrete
    question per kind — is the dead end gone, does the protein carry an
    identifier, does the entity carry a location, does the incomplete reaction
    carry the missing participant.

    The remaining failure modes are named apart because they lead to different
    fixes: nothing retrieved (widen acquisition), something retrieved but nothing
    admissible (read the admission report), or something admitted that did not
    change the payload in the way this gap needed (read the resolutions).
    """
    admitted_for = {
        gid
        for candidate in accepted
        for gid in (candidate.gap_ids or [candidate.gap_id])
        if gid
    }
    hits_by_gap: Dict[str, int] = {}
    for bundle in bundles:
        gap_id = _text(getattr(_gap_of(bundle), "gap_id", ""))
        if gap_id:
            hits_by_gap[gap_id] = hits_by_gap.get(gap_id, 0) + len(
                _safe_list(getattr(bundle, "hits", []))
            )

    open_metabolites = _open_metabolites(payload)
    participants = _payload_participants(payload)
    reports: List[Dict[str, Any]] = []
    for gap in gaps:
        gap_id = _text(getattr(gap, "gap_id", ""))
        if _payload_closes_gap(payload, gap, open_metabolites, participants):
            continue
        retrieved = hits_by_gap.get(gap_id, 0)
        if not retrieved:
            reason = "no supporting evidence retrieved (gap left unresolved)"
        elif gap_id in admitted_for:
            reason = (
                "a reaction was admitted for this gap but the resulting payload "
                "still does not express the resolution (gap left unresolved)"
            )
        else:
            reason = (
                f"{retrieved} passage(s) retrieved but nothing admissible resolved "
                "this gap (gap left unresolved)"
            )
        row: Dict[str, Any] = {
            "gap_id": gap_id,
            "kind": _text(getattr(gap, "kind", "")) or "gap",
            "label": _text(getattr(gap, "label", "")),
            "detail": _text(getattr(gap, "detail", "")),
            "reason": reason,
        }
        # An unmapped-enzyme gap asks "which protein IS this?", which no reaction
        # can answer. The RECOMMENDATION is recorded; nothing here hands the gap
        # anywhere, and the Unknown-protein fallback downstream stays available
        # precisely because the gap is still open.
        if row["kind"] == "unmapped_enzyme":
            row["recommended_route"] = ROUTE_IDENTITY_RESOLVER
        reports.append(row)
    return reports


__all__ = [
    "SynthesisResult",
    "synthesize",
    "synthesize_with_report",
    "to_payload",
    "canonical_name",
    "COFACTOR_NAMES",
]
