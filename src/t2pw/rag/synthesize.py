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

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from t2pw.pipeline.process_normalizer import BIOCHEMICAL_ALIAS_MAP
from t2pw.pipeline.stage_contracts import validate_post_extraction
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
) -> Optional[_Reaction]:
    """Build a provenance-bound :class:`_Reaction` from one extracted reaction dict.

    ``parsed`` is a clean dict from :func:`t2pw.rag.extract.extract_reactions_from_text`
    (``{"name", "inputs", "outputs", "enzymes", "reversible"}``). Names are
    canonicalized and junk tokens rejected exactly as the arrow-parser path does,
    and the chunk's provenance is attached so the reaction stays evidence-bound.
    Returns ``None`` when nothing usable survives.
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
    """
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
                )
            )
        if extractor is not None:
            for pr in extractor(chunk):
                rxn = _reaction_from_extracted(pr, prov, evidence, paper, score)
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
    """Fold ``other`` (same signature) into ``target``, unioning provenance."""
    seen = {
        (p.get("source_id"), p.get("chunk_id")) for p in target.provenance
    }
    for prov in other.provenance:
        key = (prov.get("source_id"), prov.get("chunk_id"))
        if key not in seen:
            target.provenance.append(prov)
            seen.add(key)
    target.evidence.extend(other.evidence)
    target.source_papers.extend(other.source_papers)
    target.scores.extend(other.scores)
    for enzyme in other.enzymes:
        if enzyme not in target.enzymes:
            target.enzymes.append(enzyme)


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
def _build_entities(
    reactions: List[_Reaction],
    resolver: Optional[Any] = None,
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
    """

    def _entity_key(name: str) -> str:
        return name.casefold() if resolver is None else resolver(name)

    enzyme_names = {_entity_key(e) for rxn in reactions for e in rxn.enzymes}
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
    omitted: List[Dict[str, Any]] = []
    for key in sorted(display_by_name):
        display = display_by_name[key]
        prov = prov_by_name.get(key, [])
        is_protein = key in enzyme_names
        if not prov and not _is_cofactor(display):
            omitted.append(
                {
                    "kind": "protein" if is_protein else "compound",
                    "label": display,
                    "reason": "entity has no supporting evidence (omitted)",
                }
            )
            continue
        row: Dict[str, Any] = {"name": display}
        _attach_provenance(row, prov, [], scores_by_name.get(key, []))
        if is_protein:
            proteins.append(row)
        else:
            compounds.append(row)
    entities: Dict[str, List[Dict[str, Any]]] = {}
    if compounds:
        entities["compounds"] = compounds
    if proteins:
        entities["proteins"] = proteins
    return entities, omitted


# ---------------------------------------------------------------------------
# Provenance attachment (the four WP0 additive keys + core-safe source_refs).
# ---------------------------------------------------------------------------
def _attach_provenance(
    row: Dict[str, Any],
    provenance: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    scores: List[float],
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
    primary = provenance[0]
    row["rag_provenance"] = dict(primary)
    if evidence:
        row["evidence"] = [dict(e) for e in evidence]
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


_ALLOWED_ROW_KEYS = frozenset(
    {"name", "inputs", "outputs", "enzymes", "entity", "entity_type", "role", "source_refs"}
    | set(RAG_ADDITIVE_KEYS)
)


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
    _attach_provenance(row, reaction.provenance, reaction.evidence, reaction.scores)
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
def synthesize_with_report(
    seed_payload: Any,
    evidence_bundles: Optional[List[Any]] = None,
    seed_context: Any = "",
    *,
    prose_extractor: Optional[Any] = None,
    synonym_resolver: Optional[Any] = None,
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
    """
    bundles = list(evidence_bundles or [])
    seed_source = _seed_source_descriptor(seed_context)
    extractor = _make_memoized_extractor(prose_extractor)

    seed_rxns, seed_omitted = _seed_reactions(seed_payload, seed_source)
    evidence_rxns: List[_Reaction] = []
    for bundle in bundles:
        evidence_rxns.extend(_reactions_from_bundle(bundle, extractor))

    all_rxns = seed_rxns + evidence_rxns
    resolved, conflicts = _resolve_reactions(all_rxns, synonym_resolver)
    stitched = _detect_stitches(resolved)
    entities, entity_omitted = _build_entities(resolved, synonym_resolver)

    unresolved = list(seed_omitted) + list(entity_omitted)
    unresolved.extend(_unfilled_gap_reports(bundles, resolved, extractor))

    payload = to_payload(entities, resolved)
    # Carry the seed's contextual scaffolding (species / compartments / cell
    # types / tissues / biological_states) forward so the downstream mapping
    # stage still has the species row its contract requires (Defect 1). These are
    # not evidence-bound chemistry, so they are copied as-is.
    _carry_forward_scaffolding(payload, seed_payload)
    contract_report = validate_post_extraction(payload)  # raises on structural fail

    return SynthesisResult(
        payload=payload,
        unresolved_gaps=unresolved,
        conflicts=conflicts,
        stitched=stitched,
        contract_report=contract_report,
    )


def synthesize(
    seed_payload: Any,
    evidence_bundles: Optional[List[Any]] = None,
    seed_context: Any = "",
    *,
    prose_extractor: Optional[Any] = None,
    synonym_resolver: Optional[Any] = None,
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
    ).payload


def _gap_of(bundle: Any) -> Any:
    return getattr(bundle, "gap", None)


def _unfilled_gap_reports(
    bundles: List[Any], reactions: List[_Reaction], extractor: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Report every gap that no evidence could fill (nothing invented for it).

    A gap is *filled* only if at least one of its evidence hits stated a reaction
    that references the gap's target (so it now connects into the graph). A
    bundle with no hits — or hits that stated nothing about the target — leaves
    its gap unfilled and it is surfaced here. ``extractor`` is the same memoized
    prose extractor the main pass used, so a gap filled only by a prose-extracted
    reaction is correctly counted as filled (and the model is not re-called — the
    per-chunk results are memoized).
    """
    reports: List[Dict[str, Any]] = []
    for bundle in bundles:
        gap = _gap_of(bundle)
        if gap is None:
            continue
        label = _text(getattr(gap, "label", ""))
        target = canonical_name(label)
        bundle_rxns = _reactions_from_bundle(bundle, extractor)
        referenced = any(
            target.casefold() in {n.casefold() for n in rxn.participant_names()}
            or target.casefold() in {e.casefold() for e in rxn.enzymes}
            for rxn in bundle_rxns
        )
        if referenced and target:
            continue
        reports.append(
            {
                "kind": _text(getattr(gap, "kind", "")) or "gap",
                "label": label,
                "detail": _text(getattr(gap, "detail", "")),
                "reason": "no supporting evidence retrieved (gap left unfilled)",
            }
        )
    return reports


__all__ = [
    "SynthesisResult",
    "synthesize",
    "synthesize_with_report",
    "to_payload",
    "canonical_name",
    "COFACTOR_NAMES",
]
