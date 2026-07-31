"""Stage R4 — gap detection & evidence retrieval (WP4).

Find the specific gaps in the *current* pathway payload, turn each gap into a
query, retrieve top-k evidence from the WP3 hybrid retriever, and render the
evidence to the plain-text shape the extraction / audit prompts already expect —
so it can ride the **existing** injection seams:

* **S1** — the seed extraction's existing ``pathway_context`` /
  ``user_task_context`` parameters of ``run_extraction_pipeline`` (pipeline.py).
* **S2** — the audit's existing ``retrieval_context=""`` parameter of
  ``run_audit`` (audit_json_llm.py).

This module *produces* the evidence bundles and the formatted string; the actual
wiring of that string into ``run_extraction_pipeline`` / ``run_audit`` is WP7's
orchestration job (seam S5). WP4 never adds a parameter to, nor edits the body
of, any stage module.

Separation invariant (docs/rag/03_separation_invariant.md)
----------------------------------------------------------
All of this lives in ``t2pw.rag``; the dependency arrow points **RAG -> core
only**. It *reads* the core's gap signals through seam **S4** and never mutates
them:

* ``qa_graph`` connectivity/degree output (dangling nodes, orphan components,
  degree-0 entities);
* the Stage-3 strict gate report from ``run_strict_post_normalization_gates``
  (its ``errors`` list: unresolved refs, missing participants/identifiers);
* mapping reports (entities with ``status="unmapped"``).

Every report handed to :func:`detect_gaps` is treated as read-only input: it is
inspected, never written back. Retrieval reuses the WP3
``build_hybrid_scorer(store)`` (never a second scorer), and formatting **mirrors
/ wraps** the existing ``t2pw.sbml.examples.build_retrieval_context`` renderer
(imported, never edited) so the produced string is shape-compatible with what the
prompts already parse, plus the mandatory additive provenance lines.

Offline-first / determinism
---------------------------
No network, no chromadb, no LLM is required to import or run this module. With
the ``memory`` backend + a stubbed embedder the hybrid scorer's lexical half
still retrieves an exact gene/compound symbol (e.g. ``NdmA``). ``top_k`` defaults
to ``rag_config()["retrieve_top_k"]``, read at call time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from t2pw.config import rag_config
from t2pw.rag.ingest import build_hybrid_scorer
from t2pw.rag.store import Retrieved, VectorStore

# Mirror / wrap the existing renderer (imported, never edited). ``build_retrieval_context``
# emits the ``[Example i] / Source / Model / Compartments / Species / Reactions``
# block shape the extraction / audit prompts already expect; ``_tokenize`` is the
# same tokenizer its lexical scorer uses, reused here so a synthesized single-entry
# index is guaranteed to render for a given hit.
from t2pw.sbml.examples import _tokenize as _lexical_tokenize, build_retrieval_context

# The gap kinds this module classifies (docs/rag/agents/wp4_gap_retrieval.md).
GAP_DANGLING_REACTION = "dangling_reaction"
GAP_ORPHAN_METABOLITE = "orphan_metabolite"
GAP_UNMAPPED_ENZYME = "unmapped_enzyme"
GAP_MISSING_PRECURSOR = "missing_precursor"
GAP_MISSING_COMPARTMENT = "missing_compartment"

# qa_graph node-kind -> gap kind for degree/connectivity signals.
_METABOLITE_KINDS = {"compound", "element_collection", "nucleic_acid"}
_ENZYME_KINDS = {"protein", "protein_complex"}
_REACTION_KINDS = {"reaction", "transport", "reaction_coupled_transport"}

#: Payload entity bucket -> the singular type the schema calls it. This is
#: ``t2pw.pwml.ir.ENTITY_BUCKETS`` inverted (plus the two non-entity buckets a
#: gap can centre on), and it exists because ``expected_type`` is a per-KIND
#: constant while the thing actually being repaired is a specific ROW.
#: ``unmapped_enzyme`` expects a "protein" whether the row sits in
#: ``entities.proteins`` or ``entities.protein_complexes`` — and those two take
#: entirely different identifiers.
_ENTITY_TYPE_BY_BUCKET = {
    "compounds": "compound",
    "proteins": "protein",
    "protein_complexes": "protein_complex",
    "nucleic_acids": "nucleic_acid",
    "element_collections": "element_collection",
    "bounds": "bound",
    "subcellular_locations": "subcellular_location",
    "species": "species",
}
#: Searched in this order so the answer is deterministic when two buckets carry
#: the same name — the more specific protein shape wins over a bare protein.
_ENTITY_BUCKET_ORDER = (
    "protein_complexes",
    "proteins",
    "compounds",
    "nucleic_acids",
    "element_collections",
    "bounds",
    "subcellular_locations",
)


def entity_bucket_of(payload: Any, name: str) -> Tuple[str, str]:
    """``(bucket, entity_type)`` of the entity ``name`` names, or ``("", "")``.

    Read-only. Matching is by canonical name, the same way every other consumer
    of the payload finds a row.
    """
    target = str(name or "").strip().casefold()
    if not target:
        return ("", "")
    entities = _safe_dict(_safe_dict(payload).get("entities"))
    for bucket in _ENTITY_BUCKET_ORDER:
        for row in _safe_list(entities.get(bucket)):
            if not isinstance(row, dict):
                continue
            if str(row.get("name") or "").strip().casefold() == target:
                return (bucket, _ENTITY_TYPE_BY_BUCKET.get(bucket, ""))
    return ("", "")

# Per-kind static half of the gap contract: what shape of thing is expected to
# fill the gap, what relationship is missing, and why the gap matters. The
# ``{label}`` placeholder is filled per gap. Keeping these in one table is what
# makes a gap DESCRIPTION structured rather than a free-text ``detail`` blob —
# an admission rule can compare against ``expected_type`` without parsing prose.
_GAP_CONTRACT: Dict[str, Dict[str, str]] = {
    GAP_DANGLING_REACTION: {
        "expected_type": "reaction",
        "missing_relationship": (
            "no reaction shares a substrate or product with '{label}', so it is "
            "not linked to the rest of the pathway"
        ),
        "reason": (
            "the pathway cannot be traversed through '{label}': it is an isolated "
            "step, so any route that must pass through it is broken"
        ),
    },
    GAP_ORPHAN_METABOLITE: {
        "expected_type": "reaction",
        "missing_relationship": (
            "no reaction in the pathway both produces and consumes '{label}'"
        ),
        "reason": (
            "'{label}' is a dead end: the pathway either starts or stops there "
            "instead of connecting through, so the route is incomplete"
        ),
    },
    GAP_UNMAPPED_ENZYME: {
        "expected_type": "protein",
        "missing_relationship": (
            "'{label}' is named as a catalyst but no reaction/identifier evidence "
            "ties it to a step of the pathway"
        ),
        "reason": (
            "an unresolved catalyst cannot be mapped to an identifier or exported, "
            "so the step it catalyzes is unattributable"
        ),
    },
    GAP_MISSING_PRECURSOR: {
        "expected_type": "compound",
        "missing_relationship": (
            "reaction '{label}' is missing a substrate or a product, so its "
            "upstream/downstream partner is unknown"
        ),
        "reason": (
            "a half-specified reaction cannot be balanced or connected, so the "
            "pathway's precursor supply is unaccounted for"
        ),
    },
    GAP_MISSING_COMPARTMENT: {
        "expected_type": "subcellular_location",
        "missing_relationship": (
            "'{label}' has no subcellular location relating it to a compartment"
        ),
        "reason": (
            "without a compartment the element cannot be placed in the pathway "
            "diagram or checked for transport requirements"
        ),
    },
}

_GAP_ID_PREFIX = "gap"

# Adjacent-entity anchors quoted in a retrieval query. Capped because a hub
# metabolite can be adjacent to dozens of entities and an unbounded list would
# swamp the query's own gap terms (see :meth:`Gap.query_header`).
_MAX_QUERY_ADJACENT = 8


class GapContractError(ValueError):
    """A gap or a retrieval query violated the gap contract.

    Raised when a retrieval query would be issued for a gap that carries no
    ``gap_id`` — i.e. a *broad* retrieval that is not attributable to one
    specific detected gap. Retrieval is only ever allowed to run per-gap.
    """


def make_gap_id(kind: Any, label: Any) -> str:
    """Return the stable ``gap_id`` for ``(kind, label)``.

    Deterministic and content-addressed: the same gap in two runs of the same
    pathway gets the same id, so an admission report can be diffed across runs
    and a candidate's ``gap_id`` can be resolved back to the gap it claims to
    fill. The identity pair is exactly :meth:`Gap.key`, which is what
    :func:`detect_gaps` already deduplicates on, so ids are unique within a
    detection pass by construction.
    """
    kind_token = str(kind or "gap").strip().casefold() or "gap"
    identity = f"{kind_token}|{str(label or '').strip().casefold()}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:8]
    return f"{_GAP_ID_PREFIX}-{kind_token}-{digest}"


# ---------------------------------------------------------------------------
# Data shapes (defined within t2pw.rag).
# ---------------------------------------------------------------------------
@dataclass
class Gap:
    """One specific missing piece in the current pathway graph.

    ``kind`` is one of the five classified categories. ``label`` is the
    entity/reaction name the gap centers on; ``symbols`` are the exact
    gene/compound/reaction tokens to feed the lexical half of the hybrid
    retriever so an exact symbol is never lost. ``source`` records which
    read-only report (``qa_graph`` / ``gate`` / ``mapping``) flagged it.

    The gap contract (the fields below ``source``)
    ----------------------------------------------
    A gap is not just "something is missing near X" — it is a *named request*
    that a retrieved reaction can be checked against. Every gap therefore carries:

    * ``gap_id`` — stable, content-addressed (:func:`make_gap_id`). Every
      retrieval query names it and every RAG candidate must quote it, so no
      reaction can enter the pathway without saying which gap it fills.
    * ``missing_relationship`` — the relationship the graph lacks, in words.
    * ``adjacent_entities`` — the entities ALREADY in the pathway that sit next
      to the gap. This is the anchor set an admitted candidate has to connect to.
    * ``expected_type`` — the entity/reaction type expected to fill the gap.
    * ``requested_pathway`` / ``requested_organism`` — what the run asked for, so
      a candidate's own pathway/organism can be compared against the request
      rather than against whatever the retrieved paper happens to be about.
    * ``reason`` — why the gap matters (what stays broken if it is not filled).

    The description fields default to the per-kind :data:`_GAP_CONTRACT` text so
    a hand-built ``Gap(kind=..., label=...)`` is a complete, valid gap: nothing
    in the subsystem has to cope with a half-specified one.
    """

    kind: str
    label: str
    detail: str = ""
    node: str = ""
    symbols: List[str] = field(default_factory=list)
    source: str = ""
    #: --- gap contract ---
    gap_id: str = ""
    #: The names a candidate must touch to be FILLING this gap — a strict subset
    #: of ``symbols``. See :meth:`target_names`; empty means "derive it".
    target_symbols: List[str] = field(default_factory=list)
    missing_relationship: str = ""
    adjacent_entities: List[str] = field(default_factory=list)
    #: The substrate/product half of ``adjacent_entities``. This is the ONLY
    #: adjacency the admission graph may use — catalysts and reaction names are
    #: not chemical nodes and must never extend the frontier.
    adjacent_metabolites: List[str] = field(default_factory=list)
    #: For a ``missing_precursor`` gap: WHICH side of the incomplete reaction is
    #: empty (``"inputs"`` / ``"outputs"``). Recorded on the contract rather than
    #: re-derived per consumer, because a repair that guesses the side appends a
    #: substrate where a product belongs.
    missing_side: str = ""
    expected_type: str = ""
    #: The bucket the gap's TARGET ROW actually lives in
    #: (``proteins`` / ``protein_complexes`` / ``compounds`` / ...) and the
    #: singular type that bucket stands for. ``expected_type`` is a per-kind
    #: constant — every ``unmapped_enzyme`` gap "expects a protein" — but the row
    #: being repaired may be a protein COMPLEX, and a complex takes a
    #: complex-level identifier, never a UniProt accession. Carrying the real type
    #: through is what lets the applier refuse the write instead of performing it
    #: on the wrong shape of row.
    target_entity_bucket: str = ""
    target_entity_type: str = ""
    requested_pathway: str = ""
    requested_organism: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        """Fill the contract fields that were not supplied.

        Auto-deriving rather than requiring them keeps every existing
        ``Gap(kind=..., label=...)`` construction valid while guaranteeing the
        invariant the admission gate depends on: **a Gap always has a gap_id**.
        A caller that deliberately blanks ``gap_id`` afterwards is the one case
        the gate treats as "no gap named" and rejects.
        """
        if not str(self.gap_id or "").strip():
            self.gap_id = make_gap_id(self.kind, self.label)
        contract = _GAP_CONTRACT.get(self.kind, {})
        if not str(self.expected_type or "").strip():
            self.expected_type = contract.get("expected_type", "reaction")
        if not str(self.missing_relationship or "").strip():
            template = contract.get(
                "missing_relationship", "'{label}' is missing a pathway relationship"
            )
            self.missing_relationship = template.format(label=self.label)
        if not str(self.reason or "").strip():
            template = contract.get(
                "reason", "the pathway is incomplete around '{label}'"
            )
            self.reason = template.format(label=self.label)

    def key(self) -> Tuple[str, str]:
        return (self.kind, self.label.casefold())

    def target_names(self) -> List[str]:
        """The names a candidate must touch to be *filling* this gap.

        Deliberately narrower than ``symbols``, and the difference is
        load-bearing. ``symbols`` exists to steer RETRIEVAL: it carries the open
        metabolite **plus** the neighbouring reaction's name and its catalysts,
        because all of those help a lexical retriever find the right passage.
        Only the metabolites are the thing that has to be *connected* — which is
        what every ``missing_relationship`` in :data:`_GAP_CONTRACT` actually
        says ("no reaction shares a **substrate or product** with ...").

        Measured on the real ``runs/2026-07-28_0919`` PMC12444477 payload, where
        a separate historical defect had sprayed every enzyme onto every reaction:
        thirteen imported *phospholipid* reactions (``PA -> CDP-DAG``, ``PG
        synthesis``, ``Acetyl-CoA -> malonyl-CoA``, ...) list ``WaaA`` or ``LpxM``
        among their catalysts. With ``symbols`` as the target, all thirteen read
        as "fills the WaaA/LpxM reaction gap **directly**" on the strength of a
        shared catalyst name — exactly the wrong-pathway import this gate exists
        to stop. Restricting the target to substrates and products drops all
        thirteen.

        A **reaction-shaped** gap still needs more than its label: that label is a
        reaction NAME, which no candidate's participant list will ever contain, so
        its own substrates and products are the target. ``detect_gaps`` supplies
        them in ``target_symbols``; the fallback to ``symbols`` covers a
        hand-built gap that did not.
        """
        if self.target_symbols:
            return list(self.target_symbols)
        if self.kind in (GAP_DANGLING_REACTION, GAP_MISSING_PRECURSOR):
            return list(self.symbols or [self.label])
        return [self.label]

    def to_dict(self) -> Dict[str, Any]:
        """The gap contract as a plain dict (for reports / diagnostics)."""
        return {
            "gap_id": self.gap_id,
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "node": self.node,
            "source": self.source,
            "symbols": list(self.symbols),
            "target_symbols": self.target_names(),
            "missing_relationship": self.missing_relationship,
            "adjacent_entities": list(self.adjacent_entities),
            "adjacent_metabolites": list(self.adjacent_metabolites),
            "missing_side": self.missing_side,
            "expected_type": self.expected_type,
            "target_entity_bucket": self.target_entity_bucket,
            "target_entity_type": self.target_entity_type,
            "requested_pathway": self.requested_pathway,
            "requested_organism": self.requested_organism,
            "reason": self.reason,
        }

    def describe(self) -> str:
        """The full contract rendered as text (reports, diagnostics, prompts)."""
        lines = [
            f"Gap-ID: {self.gap_id}",
            f"Missing relationship: {self.missing_relationship}",
            f"Expected type: {self.expected_type}",
        ]
        if self.target_entity_type:
            lines.append(f"Target entity type: {self.target_entity_type}")
        if self.adjacent_entities:
            lines.append(
                "Adjacent existing entities: " + ", ".join(self.adjacent_entities)
            )
        if self.requested_pathway:
            lines.append(f"Requested pathway: {self.requested_pathway}")
        if self.requested_organism:
            lines.append(f"Requested organism: {self.requested_organism}")
        lines.append(f"Why it matters: {self.reason}")
        return "\n".join(lines)

    def query_header(self) -> str:
        """The contract lines a RETRIEVAL QUERY carries — deliberately leaner.

        A retrieval query is scored, not read: the lexical half of the hybrid
        retriever is a Jaccard overlap over the query's tokens, so every word
        added to it that cannot match a passage dilutes the score of every
        passage. Only the lines that either identify the gap (``Gap-ID``, so the
        query is attributable) or genuinely describe the biology being searched
        for (the missing relationship, the adjacent entities, the requested
        pathway/organism) are included. ``expected_type`` and ``reason`` are
        contract *bookkeeping* — they belong in :meth:`describe` and the
        admission report, not in a scored query.
        """
        lines = [
            f"Gap-ID: {self.gap_id}",
            f"Missing relationship: {self.missing_relationship}",
        ]
        if self.adjacent_entities:
            lines.append(
                "Adjacent existing entities: "
                + ", ".join(self.adjacent_entities[:_MAX_QUERY_ADJACENT])
            )
        request = " ".join(
            part
            for part in (self.requested_pathway, self.requested_organism)
            if str(part or "").strip()
        )
        if request:
            lines.append(f"Requested scope: {request}")
        return "\n".join(lines)


@dataclass
class EvidenceBundle:
    """The evidence retrieved for a single :class:`Gap`.

    ``hits`` are WP3 :class:`~t2pw.rag.store.Retrieved` records; each keeps its
    chunk's ``source_id`` / ``source_uri`` provenance, ready for WP5 to attach.
    A bundle is always gap-scoped — there is no "whole pathway" bundle — so
    everything transcribed from ``hits`` inherits :attr:`gap_id`.
    """

    gap: Gap
    query: str
    hits: List[Retrieved] = field(default_factory=list)

    @property
    def gap_id(self) -> str:
        return str(getattr(self.gap, "gap_id", "") or "")


# ---------------------------------------------------------------------------
# Small read-only helpers.
# ---------------------------------------------------------------------------
def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _get_report(reports: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """Pull a named sub-report defensively (read-only)."""
    for name in names:
        candidate = reports.get(name)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _parse_node(node: Any) -> Tuple[str, str]:
    """``"reaction:#1"`` -> ``("reaction", "#1")``; unknown -> ``("", str)``."""
    text = str(node or "").strip()
    if not text:
        return ("", "")
    if ":" in text:
        kind, name = text.split(":", 1)
        return (kind.strip().casefold(), name.strip())
    return ("", text)


def _reaction_index(name: str) -> Optional[int]:
    """``"#1"`` -> ``0`` (0-based). Returns ``None`` if not a ``#<n>`` token."""
    token = str(name or "").strip()
    if token.startswith("#"):
        token = token[1:]
    if token.isdigit():
        idx = int(token) - 1
        return idx if idx >= 0 else None
    return None


def _reaction_rows(payload: Dict[str, Any]) -> List[Any]:
    return _safe_list(_safe_dict(payload.get("processes")).get("reactions"))


def _participant_name(token: Any) -> str:
    """Return a participant's display name, or ``""`` (read-only).

    Reaction participants are either bare strings (``"caffeine"``) or dicts in
    the real payload (``{"name": "caffeine"}`` / ``{"name": ..., "stoichiometry":
    ...}``); both shapes must yield the exact compound symbol so it is never
    dropped from a gap query.
    """
    if isinstance(token, str):
        return token.strip()
    if isinstance(token, dict):
        for field_name in (
            "name",
            "entity",
            "compound",
            "element",
            "element_collection",
            "nucleic_acid",
        ):
            val = token.get(field_name)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _enzyme_symbols(row: Any) -> List[str]:
    """Return the enzyme / modifier names of a reaction row (read-only)."""
    symbols: List[str] = []
    if not isinstance(row, dict):
        return symbols
    for key in ("enzymes", "modifiers"):
        for actor in _safe_list(row.get(key)):
            if isinstance(actor, dict):
                for field_name in ("entity", "protein", "protein_complex", "name"):
                    val = actor.get(field_name)
                    if isinstance(val, str) and val.strip():
                        symbols.append(val.strip())
                        break
    return symbols


def _empty_reaction_side(row: Any) -> str:
    """Which side of a reaction row is empty: ``"inputs"``, ``"outputs"`` or ``""``.

    ``""`` covers both "neither side is empty" and "both are" — in either case
    there is no single missing side to repair, and a caller that appended to a
    guessed side would be inventing chemistry rather than completing it.
    """
    if not isinstance(row, dict):
        return ""
    has_in = any(_participant_name(t) for t in _safe_list(row.get("inputs")))
    has_out = any(_participant_name(t) for t in _safe_list(row.get("outputs")))
    if has_in and not has_out:
        return "outputs"
    if has_out and not has_in:
        return "inputs"
    return ""


def _reaction_participants(row: Any) -> List[str]:
    """The reaction's substrates and products only (no name, no catalysts).

    This is the FILL target of a reaction-shaped gap — connecting to a reaction
    means sharing a metabolite with it, not sharing its catalyst. Kept separate
    from :func:`_reaction_symbols`, whose wider set is what steers retrieval.
    """
    if not isinstance(row, dict):
        return []
    out: List[str] = []
    for side in ("inputs", "outputs"):
        for token in _safe_list(row.get(side)):
            participant = _participant_name(token)
            if participant:
                out.append(participant)
    return out


def _reaction_symbols(row: Any) -> Tuple[str, List[str]]:
    """Return ``(display_name, symbols)`` from a reaction row (read-only)."""
    if not isinstance(row, dict):
        return ("", [])
    name = str(row.get("name") or "").strip()
    symbols: List[str] = []
    if name:
        symbols.append(name)
    symbols.extend(_reaction_participants(row))
    symbols.extend(_enzyme_symbols(row))
    return (name, symbols)


def _cofactor_names() -> frozenset:
    """The ubiquitous-cofactor set, reused from WP5 synthesis (single source).

    Imported lazily (the same pattern as
    :func:`t2pw.rag.provenance._cofactor_names`) so this module stays
    import-cheap and free of any circular dependency. Falls back to an empty set
    if synthesis is unavailable, in which case a terminal cofactor is simply
    reported as a connectivity gap — noisier, never *wrong*.
    """
    try:
        from t2pw.rag.synthesize import COFACTOR_NAMES

        return frozenset(COFACTOR_NAMES)
    except Exception:  # pragma: no cover - defensive; synthesize is a sibling
        return frozenset()


def _dedupe(symbols: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for sym in symbols:
        token = str(sym or "").strip()
        if token and token.casefold() not in seen:
            seen.add(token.casefold())
            out.append(token)
    return out


# ---------------------------------------------------------------------------
# detect_gaps — read qa_graph + gate + mapping reports (READ-ONLY, seam S4).
# ---------------------------------------------------------------------------
def _adjacent_entities(
    label: str, reactions: List[Any], *, metabolites_only: bool = False
) -> List[str]:
    """Entities ALREADY in the pathway that sit next to the gap at ``label``.

    Every reaction that mentions ``label`` (as a participant, as its name, or as
    a catalyst) contributes its *other* entities. That set is the anchor an
    admitted candidate has to connect to — a retrieved reaction that touches none
    of them is not filling this gap, it is unrelated chemistry that happened to be
    retrieved alongside it.

    ``metabolites_only`` restricts the result to substrates and products. Both
    forms are produced by :func:`detect_gaps` and they are used for different
    things: the full set describes the gap for a human and steers retrieval, the
    chemical set is the only one the admission graph may use. Feeding catalysts
    into the graph is how thirteen phospholipid reactions read as connected to
    the lipid A pathway in the ``2026-07-28_0919`` replay — they shared a sprayed
    ``WaaA``, and nothing else.

    Read-only, order-preserving, cofactor-free (a shared H2O is not adjacency).
    """
    folded = str(label or "").strip().casefold()
    if not folded:
        return []
    cofactors = _cofactor_names()
    out: List[str] = []
    seen: set = set()
    for row in reactions:
        if not isinstance(row, dict):
            continue
        name, symbols = _reaction_symbols(row)
        pool = list(symbols)
        if name:
            pool.append(name)
        if folded not in {s.casefold() for s in pool}:
            continue
        contributed = _reaction_participants(row) if metabolites_only else pool
        for symbol in contributed:
            key = symbol.casefold()
            if key == folded or key in seen or key in cofactors:
                continue
            seen.add(key)
            out.append(symbol)
    return out


def detect_gaps(
    payload: Dict[str, Any],
    reports: Optional[Dict[str, Any]] = None,
    *,
    requested_pathway: str = "",
    requested_organism: str = "",
) -> List[Gap]:
    """Classify the current pathway's gaps from the core's read-only reports.

    ``reports`` is a mapping that may carry any of ``qa_graph`` (connectivity /
    degree output from ``qa_graph.py`` — either the ``generate_qa_report``
    ``{"flags": ...}`` shape or the CLI ``{"dangling_nodes": ...,
    "missing_links_suspected": ...}`` shape), ``gate`` (the strict Stage-3 gate
    details with an ``errors`` list), and ``mapping`` (entities with
    ``status="unmapped"``). Each gap is classified as one of
    ``dangling_reaction`` / ``orphan_metabolite`` / ``unmapped_enzyme`` /
    ``missing_precursor`` / ``missing_compartment``.

    Connectivity ("dangling end") gaps are additionally derived **directly from
    ``payload``** by :func:`_connectivity_gaps`, so they no longer depend on the
    caller supplying a particular report shape — ``reports={}`` / ``None`` still
    yields the terminal-product / unfed-substrate gaps that drive cross-paper
    stitching. Report-derived gaps are emitted first and deduplicated against, so
    the existing report behaviour is unchanged.

    Nothing here mutates ``payload`` or any report — they are inspected only
    (seam S4). Reaction gaps are enriched with the reaction's participant/enzyme
    symbols pulled from ``payload`` so the query can hit exact symbols.

    ``requested_pathway`` / ``requested_organism`` are the REQUEST — what this
    run asked for — and are stamped onto every emitted gap. They are never read
    back out of the payload or a retrieved paper: the whole point of carrying
    them on the gap is that a candidate's own (observed) pathway/organism can be
    compared against the request instead of against itself.

    Every returned gap satisfies the gap contract documented on :class:`Gap`:
    a stable ``gap_id``, the missing relationship, the adjacent existing
    entities, the expected type, the request, and why the gap matters.
    """
    payload = _safe_dict(payload)
    reports = _safe_dict(reports)
    reactions = _reaction_rows(payload)

    qa = _get_report(reports, "qa_graph", "qa", "graph", "qa_report")
    gate = _get_report(reports, "gate", "strict_gate", "gates", "gate_report")
    mapping = _get_report(reports, "mapping", "map", "mapping_report")

    gaps: List[Gap] = []
    seen: set = set()

    def _add(gap: Gap) -> None:
        if not gap.label:
            return
        if gap.key() in seen:
            return
        seen.add(gap.key())
        gap.symbols = _dedupe(gap.symbols or [gap.label])
        # Stamp the request-side half of the contract. It is not derivable from
        # the payload — it is what the RUN asked for — so it is injected here
        # rather than in ``Gap.__post_init__``.
        gap.requested_pathway = gap.requested_pathway or str(requested_pathway or "")
        gap.requested_organism = gap.requested_organism or str(requested_organism or "")
        if not gap.adjacent_entities:
            gap.adjacent_entities = _adjacent_entities(gap.label, reactions)
        if not gap.adjacent_metabolites:
            gap.adjacent_metabolites = _adjacent_entities(
                gap.label, reactions, metabolites_only=True
            )
        # Which BUCKET the target row is in, resolved once here against the
        # payload that raised the gap. Every downstream consumer then works from
        # the real type rather than re-deriving it (or, worse, assuming the
        # per-kind ``expected_type`` describes the row).
        if not gap.target_entity_bucket:
            bucket, etype = entity_bucket_of(payload, gap.label)
            gap.target_entity_bucket = bucket
            gap.target_entity_type = gap.target_entity_type or etype
        gaps.append(gap)

    def _add_node_gap(node: Any, source: str, detail: str) -> None:
        kind, name = _parse_node(node)
        if kind in _REACTION_KINDS or (kind == "" and str(name).startswith("#")):
            idx = _reaction_index(name)
            row = reactions[idx] if idx is not None and idx < len(reactions) else None
            display, symbols = _reaction_symbols(row)
            label = display or str(node)
            _add(
                Gap(
                    kind=GAP_DANGLING_REACTION,
                    label=label,
                    detail=detail,
                    node=str(node),
                    symbols=symbols or [label],
                    source=source,
                    target_symbols=_reaction_participants(row) or [label],
                )
            )
        elif kind in _METABOLITE_KINDS:
            _add(Gap(GAP_ORPHAN_METABOLITE, name, detail, str(node), [name], source))
        elif kind in _ENZYME_KINDS:
            _add(Gap(GAP_UNMAPPED_ENZYME, name, detail, str(node), [name], source))

    # --- qa_graph: CLI-shape degree/connectivity signals ---
    for row in _safe_list(qa.get("dangling_nodes")):
        if isinstance(row, dict):
            _add_node_gap(row.get("node"), "qa_graph", "dangling node (degree <= 1)")
    for row in _safe_list(qa.get("missing_links_suspected")):
        if isinstance(row, dict):
            _add_node_gap(
                row.get("node"), "qa_graph", "entity not connected to any process"
            )
    for comp in _safe_list(qa.get("orphan_components")):
        if isinstance(comp, dict):
            for node in _safe_list(comp.get("nodes")):
                _add_node_gap(node, "qa_graph", "in an orphan (disconnected) component")

    # --- qa_graph: generate_qa_report flags-shape ---
    flags = _safe_dict(qa.get("flags"))
    for row in _safe_list(flags.get("orphan_nodes")):
        if isinstance(row, dict) and row.get("entity"):
            name = str(row["entity"]).strip()
            _add(
                Gap(
                    GAP_ORPHAN_METABOLITE,
                    name,
                    "orphan node (degree 0)",
                    "",
                    [name],
                    "qa_graph",
                )
            )
    for row in _safe_list(flags.get("missing_compartments")):
        if isinstance(row, dict) and row.get("entity"):
            name = str(row["entity"]).strip()
            _add(
                Gap(
                    GAP_MISSING_COMPARTMENT,
                    name,
                    str(row.get("reason") or "no subcellular location recorded"),
                    "",
                    [name],
                    "qa_graph",
                )
            )
    for row in _safe_list(flags.get("empty_reactions")):
        if isinstance(row, dict) and row.get("reaction"):
            name = str(row["reaction"]).strip()
            # A reaction missing inputs/outputs is a missing precursor/product.
            # WHICH side is empty is read off the payload here, once, so a repair
            # never has to guess it (see ``Gap.missing_side``).
            source_row = next(
                (
                    r
                    for r in reactions
                    if isinstance(r, dict)
                    and str(r.get("name") or "").strip().casefold() == name.casefold()
                ),
                None,
            )
            gap = Gap(
                GAP_MISSING_PRECURSOR,
                name,
                "reaction missing inputs or outputs",
                "",
                [name],
                "qa_graph",
            )
            gap.missing_side = _empty_reaction_side(source_row)
            if source_row is not None:
                gap.target_symbols = _reaction_participants(source_row) or [name]
            _add(gap)
    for row in _safe_list(flags.get("missing_ids")):
        if isinstance(row, dict) and row.get("entity"):
            name = str(row["entity"]).strip()
            etype = str(row.get("type") or "").strip().casefold()
            if etype in _ENZYME_KINDS:
                _add(
                    Gap(
                        GAP_UNMAPPED_ENZYME,
                        name,
                        "entity has no external identifier",
                        "",
                        [name],
                        "qa_graph",
                    )
                )

    # --- gate: strict Stage-3 report errors ---
    for err in _safe_list(gate.get("errors")):
        if not isinstance(err, dict):
            continue
        reason = str(err.get("reason") or "")
        low = reason.casefold()
        name = _extract_name_from_reason(reason)
        if not name:
            continue
        if "compartment" in low:
            _add(Gap(GAP_MISSING_COMPARTMENT, name, reason, "", [name], "gate"))
        elif any(
            key in low
            for key in (
                "uniprot",
                "drugbank",
                "unknown protein",
                "unknown transporter",
                "modifier reference",
                "does not resolve to a declared protein",
                "missing species/organism",
            )
        ):
            _add(Gap(GAP_UNMAPPED_ENZYME, name, reason, "", [name], "gate"))

    # --- mapping report: entities with status="unmapped" ---
    for entry in _iter_mapping_entries(mapping):
        name = str(
            entry.get("name") or entry.get("query") or entry.get("entity") or ""
        ).strip()
        if not name:
            continue
        etype = str(
            entry.get("type") or entry.get("entity_type") or ""
        ).strip().casefold()
        if etype in _METABOLITE_KINDS or etype in ("compound", "metabolite"):
            _add(
                Gap(
                    GAP_ORPHAN_METABOLITE,
                    name,
                    "unmapped compound (no external id)",
                    "",
                    [name],
                    "mapping",
                )
            )
        else:
            _add(
                Gap(
                    GAP_UNMAPPED_ENZYME,
                    name,
                    "unmapped enzyme/protein (no external id)",
                    "",
                    [name],
                    "mapping",
                )
            )

    # --- payload-derived connectivity gaps (no report required) ---
    for gap in _connectivity_gaps(reactions):
        _add(gap)

    return gaps


# ---------------------------------------------------------------------------
# Payload-derived connectivity ("dangling end") detection.
# ---------------------------------------------------------------------------
def _connectivity_gaps(reactions: List[Any]) -> List[Gap]:
    """Derive the pathway's open ends straight from the reaction list (read-only).

    The reaction graph is walked to find the two shapes that make a pathway
    *extendable* — the ones a report may or may not surface, so they are computed
    here instead of depending on any particular caller-supplied report shape:

    * **terminal product** — a metabolite produced by some reaction and consumed
      by none (e.g. ``Kdo2-lipid A`` at the end of the lipid A pathway);
    * **unfed substrate** — a metabolite consumed by some reaction and produced
      by none.

    Each such metabolite becomes a :data:`GAP_ORPHAN_METABOLITE` gap, whose
    :func:`query_for_gap` intent is exactly "find a reaction that produces or
    consumes '<metabolite>' so it links into the pathway" — i.e. the search that
    stitches a *second* paper's reaction onto this pathway. The reaction sitting
    on that open end additionally becomes a :data:`GAP_DANGLING_REACTION` gap so
    the connecting step can be searched for from the reaction side too.

    Ubiquitous cofactors (:func:`_cofactor_names`) are excluded: a terminal
    H2O / ATP / CO2 is not a pathway gap. Nothing here mutates ``reactions``;
    ordering follows the payload so the result is deterministic.
    """
    cofactors = _cofactor_names()
    produced: Dict[str, str] = {}
    consumed: Dict[str, str] = {}
    for row in reactions:
        if not isinstance(row, dict):
            continue
        for side, bucket in (("inputs", consumed), ("outputs", produced)):
            for token in _safe_list(row.get(side)):
                name = _participant_name(token)
                if name:
                    bucket.setdefault(name.casefold(), name)

    def _is_open(name: str, other_side: Dict[str, str]) -> bool:
        folded = name.casefold()
        return folded not in other_side and folded not in cofactors

    gaps: List[Gap] = []
    for idx, row in enumerate(reactions):
        if not isinstance(row, dict):
            continue
        display, symbols = _reaction_symbols(row)
        enzymes = _enzyme_symbols(row)
        for side, other_side, detail_fmt in (
            (
                "outputs",
                consumed,
                "terminal product: produced by '{rxn}' but consumed by no reaction",
            ),
            (
                "inputs",
                produced,
                "unfed substrate: consumed by '{rxn}' but produced by no reaction",
            ),
        ):
            open_names = [
                name
                for name in (
                    _participant_name(token) for token in _safe_list(row.get(side))
                )
                if name and _is_open(name, other_side)
            ]
            if not open_names:
                continue
            rxn = display or f"#{idx + 1}"
            detail = detail_fmt.format(rxn=rxn)
            for name in open_names:
                gaps.append(
                    Gap(
                        kind=GAP_ORPHAN_METABOLITE,
                        label=name,
                        detail=detail,
                        node=f"compound:{name}",
                        symbols=[name] + ([display] if display else []) + enzymes,
                        source="payload",
                    )
                )
            if display:
                gaps.append(
                    Gap(
                        kind=GAP_DANGLING_REACTION,
                        label=display,
                        detail=(
                            f"{detail_fmt.split(':')[0]}(s) "
                            f"{', '.join(open_names)} not linked to any other reaction"
                        ),
                        node=f"reaction:#{idx + 1}",
                        symbols=symbols or [display],
                        source="payload",
                        target_symbols=_reaction_participants(row) or [display],
                    )
                )
    return gaps


def _extract_name_from_reason(reason: str) -> str:
    """Best-effort pull of the offending entity name out of a gate reason string."""
    text = str(reason or "")
    # Prefer a quoted token: ``Protein 'NdmA' is missing ...``.
    if "'" in text:
        parts = text.split("'")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()
    # ``Unknown protein/modifier reference: NdmA`` -> trailing token after ':'.
    if ":" in text:
        tail = text.rsplit(":", 1)[-1].strip().rstrip(".")
        if tail and " " not in tail:
            return tail
    return ""


def _iter_mapping_entries(mapping: Any) -> List[Dict[str, Any]]:
    """Collect unmapped mapping entries from a variety of report shapes (read-only)."""
    out: List[Dict[str, Any]] = []

    def _consider(entry: Any) -> None:
        if (
            isinstance(entry, dict)
            and str(entry.get("status") or "").strip().casefold() == "unmapped"
        ):
            out.append(entry)

    if isinstance(mapping, list):
        for entry in mapping:
            _consider(entry)
    elif isinstance(mapping, dict):
        # Flat status dict, or nested {"unmapped": [...]} / per-section lists.
        _consider(mapping)
        for value in mapping.values():
            if isinstance(value, list):
                for entry in value:
                    _consider(entry)
            elif isinstance(value, dict):
                _consider(value)
    return out


# ---------------------------------------------------------------------------
# query_for_gap — natural-language + exact-symbol query.
# ---------------------------------------------------------------------------
def query_for_gap(gap: Gap, seed_context: str = "") -> str:
    """Build a retrieval query: a natural-language ask + the exact symbols.

    The natural-language sentence steers the semantic half of the hybrid
    retriever; the explicit symbol list steers the lexical half so an exact
    gene/compound symbol is retrieved even when embeddings are unavailable.
    ``seed_context`` (pathway name / organism / key terms) is appended when
    given.

    **Every query names one gap.** The rendered gap contract (:meth:`Gap.describe`)
    leads the query, so the retrieval that produced a passage is always
    attributable to a specific ``gap_id`` and the reactions transcribed from that
    passage inherit it. A gap whose ``gap_id`` has been blanked raises
    :class:`GapContractError` rather than silently issuing a broad,
    unattributable pathway query — "retrieve everything about the pathway, then
    merge whatever comes back" is exactly the shape this refuses to build.
    """
    if not str(getattr(gap, "gap_id", "") or "").strip():
        raise GapContractError(
            "retrieval query refused: the gap carries no gap_id, so any reaction "
            "retrieved by it could not be attributed to a detected gap "
            f"(kind={getattr(gap, 'kind', '')!r}, label={getattr(gap, 'label', '')!r})"
        )
    label = gap.label
    symbols = _dedupe(gap.symbols or [label])
    intents = {
        GAP_DANGLING_REACTION: (
            f"Find the reaction, enzyme, or transport step that connects '{label}' "
            f"to the rest of the pathway (its missing substrate, product, or catalyst)."
        ),
        GAP_ORPHAN_METABOLITE: (
            f"Find a reaction that produces or consumes the metabolite '{label}' "
            f"so it links into the pathway."
        ),
        GAP_UNMAPPED_ENZYME: (
            f"Identify the enzyme or protein '{label}': its gene, UniProt identifier, "
            f"organism, and the reaction it catalyzes."
        ),
        GAP_MISSING_PRECURSOR: (
            f"Find the precursor or product that reaction '{label}' is missing, and the "
            f"upstream/downstream reaction supplying or consuming it."
        ),
        GAP_MISSING_COMPARTMENT: (
            f"Determine the subcellular compartment / location where '{label}' occurs."
        ),
    }
    parts: List[str] = [
        gap.query_header(),
        intents.get(gap.kind, f"Find evidence about '{label}'."),
    ]
    if symbols:
        parts.append("Key symbols: " + ", ".join(symbols))
    ctx = str(seed_context or "").strip()
    if ctx:
        parts.append("Pathway context: " + ctx)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# retrieve_evidence — hybrid retrieval via the WP3 scorer.
# ---------------------------------------------------------------------------
def retrieve_evidence(
    gap: Gap,
    store: VectorStore,
    *,
    top_k: Optional[int] = None,
    seed_context: str = "",
    scorer: Optional[Any] = None,
) -> EvidenceBundle:
    """Retrieve top-``top_k`` evidence for ``gap`` using the WP3 hybrid scorer.

    ``top_k`` defaults to ``rag_config()["retrieve_top_k"]`` (read at call time).
    ``scorer`` is the WP3 ``build_hybrid_scorer(store)`` callable by default; it
    can be injected for testing. Each returned hit is a
    :class:`~t2pw.rag.store.Retrieved` whose chunk keeps its ``source_id`` /
    ``source_uri`` provenance.
    """
    k = int(top_k) if top_k is not None else int(rag_config()["retrieve_top_k"])
    k = max(1, k)
    query = query_for_gap(gap, seed_context)
    retriever = scorer if scorer is not None else build_hybrid_scorer(store)
    hits = list(retriever(query, top_k=k))
    return EvidenceBundle(gap=gap, query=query, hits=hits)


# ---------------------------------------------------------------------------
# format_retrieval_context — mirror / wrap build_retrieval_context.
# ---------------------------------------------------------------------------
def _render_hit_block(ordinal: int, gap: Gap, retrieved: Retrieved) -> str:
    """Render one hit in the ``build_retrieval_context`` block shape + provenance.

    Wraps the existing renderer: a synthetic single-entry motif index is fed to
    ``build_retrieval_context`` (with a self-matching query so it always renders),
    its ``[Example i]`` header is swapped for a gap-tagged ``[Evidence i]`` header,
    and the mandatory additive provenance line (``source_id`` / ``source_uri`` /
    ``chunk_id``) is appended.
    """
    chunk = retrieved.chunk
    text = str(chunk.text or "").strip()
    tokens = sorted(set(_lexical_tokenize(text))) if text else []
    entry = {
        "source_name": chunk.source_id or chunk.id,
        "model_name": chunk.source_title or chunk.source_type,
        "compartments": [chunk.section] if chunk.section else [],
        "species": [chunk.organism] if chunk.organism else [],
        "reaction_patterns": [text] if text else [],
        # Token fields so the wrapped lexical scorer matches this single entry.
        "tokens": tokens,
        "species_tokens": tokens,
        "reaction_tokens": tokens,
        "compartment_tokens": [],
    }
    query = " ".join(tokens) if tokens else (text or "evidence")
    block, _meta = build_retrieval_context(
        query, {"entries": [entry]}, top_k=1, max_chars=100000
    )
    provenance = (
        f"Provenance: source_id={chunk.source_id or chunk.id} "
        f"source_uri={chunk.source_uri} "
        f"source_type={chunk.source_type} "
        f"section={chunk.section} chunk_id={chunk.id}"
    )
    header = (
        f"[Evidence {ordinal}] gap_id={getattr(gap, 'gap_id', '')} gap={gap.kind} "
        f"target={gap.label} score={round(float(retrieved.score), 6)}"
    )
    lines = block.splitlines() if block.strip() else []
    if lines and lines[0].startswith("[Example"):
        lines[0] = header
    else:
        # Renderer produced nothing (e.g. empty passage): fall back to a minimal
        # but shape-compatible block so provenance is never dropped.
        lines = [
            header,
            f"Source: {chunk.source_id or chunk.id}",
            f"Model: {chunk.source_title or chunk.source_type or 'n/a'}",
            "Reactions:",
            f"- {text or 'n/a'}",
        ]
    lines.append(provenance)
    return "\n".join(lines).strip()


def format_retrieval_context(
    bundles: List[EvidenceBundle], *, max_chars: int = 6000
) -> str:
    """Render evidence bundles to the plain-text shape the prompts expect.

    Mirrors ``t2pw.sbml.examples.build_retrieval_context`` (wrapped, not edited)
    so the returned **string** is shape-compatible with what the extraction /
    audit prompts already parse, and carries the additive provenance every hit
    needs. This string is what rides seam **S1** (folded into
    ``pathway_context`` / ``user_task_context``) and seam **S2** (passed to
    ``run_audit(..., retrieval_context=...)``); WP4 returns it, WP7 wires it.

    Returns ``""`` when there is nothing to inject (so an empty context is a
    no-op for the prompts, exactly like today).
    """
    bundles = [b for b in (bundles or []) if isinstance(b, EvidenceBundle) and b.hits]
    if not bundles:
        return ""

    gap_counts: Dict[str, int] = {}
    for bundle in bundles:
        gap_counts[bundle.gap.kind] = gap_counts.get(bundle.gap.kind, 0) + 1
    summary = ", ".join(f"{kind}x{n}" for kind, n in sorted(gap_counts.items()))
    header = (
        f"RETRIEVED EVIDENCE (RAG) - gap-targeted passages with provenance ({summary})."
    )

    blocks: List[str] = [header, ""]
    ordinal = 1
    for bundle in bundles:
        for hit in bundle.hits:
            block = _render_hit_block(ordinal, bundle.gap, hit)
            if block:
                blocks.append(block)
                blocks.append("")
                ordinal += 1
        if len("\n".join(blocks)) > max_chars:
            break

    text = "\n".join(blocks).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


__all__ = [
    "Gap",
    "GapContractError",
    "EvidenceBundle",
    "make_gap_id",
    "detect_gaps",
    "query_for_gap",
    "retrieve_evidence",
    "format_retrieval_context",
    "GAP_DANGLING_REACTION",
    "GAP_ORPHAN_METABOLITE",
    "GAP_UNMAPPED_ENZYME",
    "GAP_MISSING_PRECURSOR",
    "GAP_MISSING_COMPARTMENT",
]
