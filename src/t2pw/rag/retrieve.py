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
    """

    kind: str
    label: str
    detail: str = ""
    node: str = ""
    symbols: List[str] = field(default_factory=list)
    source: str = ""

    def key(self) -> Tuple[str, str]:
        return (self.kind, self.label.casefold())


@dataclass
class EvidenceBundle:
    """The evidence retrieved for a single :class:`Gap`.

    ``hits`` are WP3 :class:`~t2pw.rag.store.Retrieved` records; each keeps its
    chunk's ``source_id`` / ``source_uri`` provenance, ready for WP5 to attach.
    """

    gap: Gap
    query: str
    hits: List[Retrieved] = field(default_factory=list)


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


def _reaction_symbols(row: Any) -> Tuple[str, List[str]]:
    """Return ``(display_name, symbols)`` from a reaction row (read-only)."""
    if not isinstance(row, dict):
        return ("", [])
    name = str(row.get("name") or "").strip()
    symbols: List[str] = []
    if name:
        symbols.append(name)
    for side in ("inputs", "outputs"):
        for token in _safe_list(row.get(side)):
            if isinstance(token, str) and token.strip():
                symbols.append(token.strip())
            elif isinstance(token, dict):
                # Reaction participants are dicts in the real payload
                # (``{"name": "caffeine"}``); pull the compound/entity name so the
                # gap query still carries its exact substrate/product symbols.
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
                        symbols.append(val.strip())
                        break
    for key in ("enzymes", "modifiers"):
        for actor in _safe_list(row.get(key)):
            if isinstance(actor, dict):
                for field_name in ("entity", "protein", "protein_complex", "name"):
                    val = actor.get(field_name)
                    if isinstance(val, str) and val.strip():
                        symbols.append(val.strip())
                        break
    return (name, symbols)


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
def detect_gaps(
    payload: Dict[str, Any], reports: Optional[Dict[str, Any]] = None
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

    Nothing here mutates ``payload`` or any report — they are inspected only
    (seam S4). Reaction gaps are enriched with the reaction's participant/enzyme
    symbols pulled from ``payload`` so the query can hit exact symbols.
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
            _add(
                Gap(
                    GAP_MISSING_PRECURSOR,
                    name,
                    "reaction missing inputs or outputs",
                    "",
                    [name],
                    "qa_graph",
                )
            )
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
    """
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
    parts: List[str] = [intents.get(gap.kind, f"Find evidence about '{label}'.")]
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
        f"[Evidence {ordinal}] gap={gap.kind} target={gap.label} "
        f"score={round(float(retrieved.score), 6)}"
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
    "EvidenceBundle",
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
