"""Stage R3 — ingest & index (WP3).

Turns the WP2-selected papers, structured DB reaction records, and the existing
on-disk example corpus (``reference/*.pwml`` / ``reference/*.sbml``) into a
populated, persisted :class:`~t2pw.rag.store.VectorStore`, and exposes the hybrid
(semantic + lexical) scorer that WP4 will call.

What this module does (and only this):

* :func:`chunk_paper` — section-aware splitting of a :class:`CandidatePaper`
  (abstract / introduction / results / methods / discussion / figure captions),
  ~500-1000 tokens per chunk with overlap, carrying ``source_uri`` / ``source_id``
  provenance. The chunk ``id`` is a stable hash of ``(source_id, section,
  offset)`` so re-ingesting an unchanged paper is idempotent.
* :func:`chunk_db_reactions` — one :class:`Chunk` per structured reaction /
  record, tagged ``source_type`` ``"pathbank"`` / ``"kegg"``.
* :func:`chunk_corpus` — the ``reference/`` example corpus, one-or-more chunks
  per file, tagged ``source_type="pwml_example"`` so example-pathway retrieval
  becomes semantic too.
* :func:`ingest` — chunk -> embed (the WP0 :class:`Embedder`, which caches and
  never re-embeds an unchanged chunk) -> ``upsert`` to the store -> ``persist``.
* :func:`build_hybrid_scorer` — the WP4-facing callable that blends the store's
  semantic score with the existing lexical motif score
  (``t2pw.sbml.examples._score_entry``) at ``0.7 * semantic + 0.3 * lexical``.

Separation invariant (docs/rag/03_separation_invariant.md)
----------------------------------------------------------
All of this lives in ``t2pw.rag``. The dependency arrow points **RAG -> core
only**: this module imports the WP0 store/embedder, the WP1 ``CandidatePaper``,
and *wraps* the existing lexical layer ``t2pw.sbml.examples`` (``parse_sbml`` /
``build_motif_entry`` for corpus text, ``_score_entry`` for the lexical half of
the hybrid scorer). Nothing in ``t2pw.sbml`` (or any stage module) imports
``t2pw.rag``, and ``t2pw.sbml.examples`` is **not** edited — the lexical scorer is
wrapped from here, never modified. WP3 uses no core seam and changes no pipeline
behavior; it only builds the store/scorer that WP4 consumes.

Offline-first / determinism
---------------------------
Embedding degrades gracefully to the :class:`Embedder`'s deterministic lexical
fallback when no endpoint is reachable, and the lexical half of the hybrid scorer
guarantees an exact gene/compound symbol (e.g. ``NdmA``) is still retrieved when
embeddings are unavailable. Stable chunk ids + the embedding cache make a
re-ingest of unchanged inputs a no-op. Corpus parsing is fail-safe: an unreadable
file contributes nothing rather than raising.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence
from xml.etree import ElementTree

from t2pw.config import rag_config
from t2pw.paths import PROJECT_ROOT
from t2pw.rag.acquire import CandidatePaper
from t2pw.rag.embed import Embedder
from t2pw.rag.provenance import SEED_SOURCE_ID
from t2pw.rag.store import Chunk, Retrieved, VectorStore, get_vector_store

# Wrap (never edit) the existing lexical layer. ``_score_entry`` / ``_tokenize``
# are the token-overlap scorer + tokenizer the pipeline already uses; reusing
# them keeps the hybrid scorer's lexical half identical to motif retrieval.
from t2pw.sbml.examples import (
    _score_entry,
    _tokenize as _lexical_tokenize,
    build_motif_entry,
    parse_sbml,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids importing the LLM client
    from t2pw.rag.select import Selection

# Hybrid blend (starting weights per docs/rag/02_vector_store.md; tunable via the
# ``build_hybrid_scorer`` parameters).
DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_LEXICAL_WEIGHT = 0.3

# Chunk sizing (~500-1000 tokens with overlap).
_TARGET_TOKENS = 800
_OVERLAP_TOKENS = 150

# Corpus (example-pathway) reference directory and the file suffixes we index.
_CORPUS_DIRNAME = "reference"
_SBML_SUFFIXES = {".sbml", ".xml"}
_PWML_SUFFIXES = {".pwml"}
_MAX_CORPUS_NAMES = 1500  # bound the name bag pulled from one big pwml file
_MAX_FIGURE_CHUNKS = 12

_WORD_RE = re.compile(r"\S+")

# Section-header keywords used to split a flattened full text. Only the FIRST
# occurrence of each canonical label becomes a boundary, so the word "results"
# reappearing in prose does not spuriously start a new section.
_HEADER_KEYWORDS: List[tuple] = [
    ("introduction", r"\bintroduction\b"),
    ("introduction", r"\bbackground\b"),
    ("methods", r"\bmaterials and methods\b"),
    ("methods", r"\bexperimental procedures\b"),
    ("methods", r"\bmethods\b"),
    ("results", r"\bresults\b"),
    ("discussion", r"\bdiscussion\b"),
    ("conclusion", r"\bconclusions?\b"),
]

# Back-matter headers, matched separately from the body ones above. A reference
# list is not prose: it is a run of cited titles, and an extractor reading it
# transcribes those titles as reactions (a cited title like "An Escherichia coli
# gene (FabZ) encoding (3R)-hydroxymyristoyl acyl carrier protein dehydrase"
# becomes a reaction with no substrate or product). Without a label for it the
# bibliography was absorbed into whichever body section matched last and mined
# like any other passage.
#
# These match on the LAST occurrence and only in the document tail: the words
# occur in ordinary prose ("see references therein"), and binding the boundary to
# an early mention would discard the entire body.
_BACK_MATTER_KEYWORDS: List[tuple] = [
    ("references", r"\b(?:references|bibliography|literature cited)\b"),
    ("acknowledgments", r"\backnowledge?ments?\b"),
]

# Fraction of the document a back-matter header must start beyond to count.
_BACK_MATTER_MIN_POSITION = 0.55

# Sections that are never handed to retrieval or reaction extraction.
_NON_EXTRACTABLE_SECTIONS = frozenset({"references", "acknowledgments"})

_FIGURE_RE = re.compile(r"(?i)\b(?:fig(?:ure)?\.?)\s*\d+")


# ---------------------------------------------------------------------------
# Report shape.
# ---------------------------------------------------------------------------
@dataclass
class IngestReport:
    """Summary of one :func:`ingest` run.

    ``embedded`` counts vectors computed *this run* (cache misses) and
    ``cache_hits`` counts chunks served from the embedding cache — so a re-ingest
    of unchanged inputs shows ``embedded == 0`` (a no-op).
    """

    papers: int = 0
    corpus_files: int = 0
    db_records: int = 0
    chunks: int = 0
    embedded: int = 0
    cache_hits: int = 0
    upserted: int = 0
    skipped: int = 0
    store_total: int = 0
    backend: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "papers": self.papers,
            "corpus_files": self.corpus_files,
            "db_records": self.db_records,
            "chunks": self.chunks,
            "embedded": self.embedded,
            "cache_hits": self.cache_hits,
            "upserted": self.upserted,
            "skipped": self.skipped,
            "store_total": self.store_total,
            "backend": self.backend,
        }


# ---------------------------------------------------------------------------
# Low-level splitting helpers.
# ---------------------------------------------------------------------------
def _chunk_id(source_id: str, section: str, offset: Any) -> str:
    """Stable id = hash of (source_id, section, offset) -> idempotent re-ingest."""
    raw = f"{source_id}\x00{section}\x00{offset}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _iter_windows(
    text: str, *, target: int = _TARGET_TOKENS, overlap: int = _OVERLAP_TOKENS
) -> List[str]:
    """Split ``text`` into overlapping ~``target``-token windows (whitespace tokens)."""
    words = _WORD_RE.findall(text or "")
    if not words:
        return []
    target = max(1, int(target))
    overlap = max(0, min(int(overlap), target - 1))
    step = max(1, target - overlap)
    windows: List[str] = []
    start = 0
    n = len(words)
    while start < n:
        window = words[start : start + target]
        windows.append(" ".join(window))
        if start + target >= n:
            break
        start += step
    return windows


def _split_sections(full_text: str) -> List[tuple]:
    """Split flattened full text into ``(section_label, text)`` spans.

    Best-effort: uses the first occurrence of each canonical header keyword (in
    document order) as a boundary. Text before the first header is ``"body"``.
    """
    text = full_text or ""
    if not text.strip():
        return []
    lowered = text.casefold()
    boundaries: Dict[str, int] = {}
    for label, pattern in _HEADER_KEYWORDS:
        if label in boundaries:
            continue
        match = re.search(pattern, lowered)
        if match:
            boundaries[label] = match.start()

    # Back matter binds to its last mention, and only in the document tail.
    tail_start = int(len(lowered) * _BACK_MATTER_MIN_POSITION)
    for label, pattern in _BACK_MATTER_KEYWORDS:
        if label in boundaries:
            continue
        last_start = -1
        for match in re.finditer(pattern, lowered):
            last_start = match.start()
        if last_start >= tail_start:
            boundaries[label] = last_start

    points = sorted(boundaries.items(), key=lambda kv: kv[1])
    if not points:
        return [("body", text)]

    spans: List[tuple] = []
    first_pos = points[0][1]
    if first_pos > 0:
        head = text[:first_pos].strip()
        if head:
            spans.append(("body", head))
    for idx, (label, pos) in enumerate(points):
        end = points[idx + 1][1] if idx + 1 < len(points) else len(text)
        body = text[pos:end].strip()
        if body:
            spans.append((label, body))
    return spans


def _figure_captions(full_text: str) -> List[str]:
    """Extract short figure-caption windows (best-effort, bounded)."""
    text = full_text or ""
    out: List[str] = []
    for match in _FIGURE_RE.finditer(text):
        snippet = text[match.start() : match.start() + 300].strip()
        if snippet:
            out.append(snippet)
        if len(out) >= _MAX_FIGURE_CHUNKS:
            break
    return out


# ---------------------------------------------------------------------------
# Public chunkers.
# ---------------------------------------------------------------------------
def seed_candidate(
    text: str,
    *,
    title: str = "",
    organism: str = "",
) -> Optional[CandidatePaper]:
    """Present the uploaded seed paper as a ``CandidatePaper`` so it can be chunked.

    The seed is the one document guaranteed to be about the pathway, yet it was
    the only one never indexed: :func:`ingest` chunked ``selection.selected``
    only. That left every seed-derived claim citable as "the seed paper" with no
    section and no quotable passage, and unable to corroborate anything, because
    corroboration is counted over *identified* sources and the seed had no chunk.

    Returns ``None`` for empty text so callers can pass it unconditionally.
    """

    body = str(text or "").strip()
    if not body:
        return None
    return CandidatePaper(
        id=SEED_SOURCE_ID,
        source="seed",
        title=str(title or "").strip() or "uploaded seed paper",
        organism=str(organism or "").strip(),
        full_text=body,
    )


def _observed_list(value: Any) -> List[str]:
    """Clean a candidate's observed-metadata list (read-only, order-preserving)."""
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def chunk_paper(
    candidate: CandidatePaper,
    *,
    target_tokens: int = _TARGET_TOKENS,
    overlap_tokens: int = _OVERLAP_TOKENS,
) -> List[Chunk]:
    """Section-aware split of ``candidate`` into provenance-carrying chunks.

    The abstract field becomes ``section="abstract"`` chunk(s); the full text (a
    flattened plain-text body) is split into introduction / methods / results /
    discussion / body spans plus figure-caption chunks. Every chunk carries the
    candidate's ``source_id`` / ``source_uri`` / ``organism`` so provenance
    survives retrieval, and its ``id`` is a stable hash of
    ``(source_id, section, offset)`` for idempotent re-ingest.

    The candidate's eligibility-derived ``observed_pathways`` /
    ``observed_organisms`` ride along too. Without them the admission gate
    downstream has only the chunk's free text to judge scope by, so a paper the
    eligibility screen already read as being about a *different* pathway or a
    *different* species arrives at the gate looking merely "unknown" — the
    permissive verdict. These are the PAPER's own reading; the request travels
    separately and is never copied into them (see ``acquire.CandidatePaper``,
    where stamping the requested organism onto every row is the documented bug
    this split exists to prevent).
    """
    source_id = str(candidate.id or "").strip()
    if not source_id:
        return []
    title = str(candidate.title or "").strip()
    uri = str(candidate.source_uri or "").strip()
    organism = str(candidate.organism or "").strip()
    observed_pathways = _observed_list(getattr(candidate, "observed_pathways", None))
    observed_organisms = _observed_list(getattr(candidate, "observed_organisms", None))

    def _emit(section: str, text: str, out: List[Chunk]) -> None:
        for ordinal, window in enumerate(
            _iter_windows(text, target=target_tokens, overlap=overlap_tokens)
        ):
            out.append(
                Chunk(
                    id=_chunk_id(source_id, section, ordinal),
                    text=window,
                    source_id=source_id,
                    source_title=title,
                    source_type="paper",
                    source_uri=uri,
                    organism=organism,
                    section=section,
                    observed_pathways=list(observed_pathways),
                    observed_organisms=list(observed_organisms),
                )
            )

    chunks: List[Chunk] = []
    _emit("abstract", str(candidate.abstract or "").strip(), chunks)

    full_text = str(candidate.full_text or "").strip()
    if full_text:
        body_spans = [
            (section, span)
            for section, span in _split_sections(full_text)
            if section not in _NON_EXTRACTABLE_SECTIONS
        ]
        for section, span in body_spans:
            _emit(section, span, chunks)
        # Caption scanning reads the retained body only, so "Fig." mentions inside
        # cited titles do not smuggle the bibliography back in.
        caption_source = "\n".join(span for _section, span in body_spans) or full_text
        for ordinal, caption in enumerate(_figure_captions(caption_source)):
            chunks.append(
                Chunk(
                    id=_chunk_id(source_id, "figure", ordinal),
                    text=caption,
                    source_id=source_id,
                    source_title=title,
                    source_type="paper",
                    source_uri=uri,
                    organism=organism,
                    section="figure",
                )
            )
    return chunks


def _reaction_record_text(record: Dict[str, Any]) -> str:
    """Assemble a compact text blob from a structured reaction record."""
    parts: List[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            parts.extend(str(v).strip() for v in value if str(v).strip())
        elif value is not None and str(value).strip():
            parts.append(str(value).strip())

    for key in ("name", "reaction", "equation", "definition", "description"):
        _add(record.get(key))
    for key in ("enzyme", "enzymes", "ec", "ec_number", "gene", "genes", "proteins"):
        _add(record.get(key))
    for key in ("reactants", "inputs", "substrates", "products", "outputs"):
        _add(record.get(key))
    for key in ("pathway", "pathway_name"):
        _add(record.get(key))
    # De-dup while preserving order.
    seen: set = set()
    ordered: List[str] = []
    for part in parts:
        if part.casefold() not in seen:
            seen.add(part.casefold())
            ordered.append(part)
    return " ; ".join(ordered)


def _record_tags(record: Dict[str, Any]) -> List[str]:
    tags = record.get("pathway_tags") or record.get("tags")
    if isinstance(tags, (list, tuple)):
        return [str(t).strip() for t in tags if str(t).strip()]
    if isinstance(tags, str) and tags.strip():
        return [tags.strip()]
    return []


def chunk_db_reactions(
    records: Sequence[Dict[str, Any]], *, source_type: Optional[str] = None
) -> List[Chunk]:
    """One :class:`Chunk` per structured reaction / record.

    ``source_type`` is taken per-record from ``record["source_type"]`` /
    ``record["source"]`` (normalized to ``"pathbank"`` / ``"kegg"``) when present,
    else from the ``source_type`` argument, else defaults to ``"pathbank"``. This
    formats already-fetched records; it does **no** DB/HTTP work itself (callers
    use the existing ``t2pw.mapping`` / ``t2pw.stoich`` helpers), keeping ingest
    offline-safe and deterministic.
    """
    chunks: List[Chunk] = []
    for idx, record in enumerate(records or []):
        if not isinstance(record, dict):
            continue
        text = _reaction_record_text(record)
        if not text:
            continue
        rec_type = str(
            record.get("source_type") or record.get("source") or source_type or ""
        ).strip().lower()
        if rec_type not in {"pathbank", "kegg"}:
            rec_type = (source_type or "pathbank").strip().lower()
        rec_id = str(
            record.get("id")
            or record.get("reaction_id")
            or record.get("entry")
            or f"{rec_type}:{idx}"
        ).strip()
        chunks.append(
            Chunk(
                id=_chunk_id(rec_id, "reaction", rec_type),
                text=text,
                source_id=rec_id,
                source_title=str(
                    record.get("name") or record.get("pathway") or ""
                ).strip(),
                source_type=rec_type,
                source_uri=str(record.get("source_uri") or record.get("uri") or "").strip(),
                organism=str(record.get("organism") or "").strip(),
                section="reaction",
                pathway_tags=_record_tags(record),
            )
        )
    return chunks


def _corpus_text_for_file(path: Path) -> str:
    """Extract a retrievable text blob from one reference-corpus file.

    SBML/XML files reuse the existing ``parse_sbml`` + ``build_motif_entry``
    lexical helpers; PWML files are flattened to their name/description bag.
    Fail-safe: any parse error yields ``""`` (the file is skipped).
    """
    suffix = path.suffix.lower()
    try:
        if suffix in _SBML_SUFFIXES:
            entry = build_motif_entry(parse_sbml(path))
            pieces: List[str] = []
            if entry.get("model_name"):
                pieces.append(str(entry["model_name"]))
            pieces.extend(str(x) for x in entry.get("compartments", []))
            pieces.extend(str(x) for x in entry.get("species", []))
            pieces.extend(str(x) for x in entry.get("reaction_patterns", []))
            return " ; ".join(p for p in pieces if p.strip())
        if suffix in _PWML_SUFFIXES:
            return _extract_pwml_text(path)
    except Exception:  # noqa: BLE001 - a bad corpus file must not break ingest
        return ""
    return ""


def _extract_pwml_text(path: Path) -> str:
    root = ElementTree.parse(str(path)).getroot()
    names: List[str] = []
    seen: set = set()
    for tag in ("cached-name", "cached-description", "cached-subject"):
        el = root.find(f".//{tag}")
        if el is not None and el.text and el.text.strip():
            value = el.text.strip()
            if value.casefold() not in seen:
                seen.add(value.casefold())
                names.append(value)
    for el in root.iter():
        local = el.tag.rsplit("}", 1)[-1]
        if local in ("name", "description") and el.text and el.text.strip():
            value = el.text.strip()
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                names.append(value)
        if len(names) >= _MAX_CORPUS_NAMES:
            break
    return " ; ".join(names)


def chunk_corpus(corpus_dir: Optional[Path] = None) -> List[Chunk]:
    """Chunk the ``reference/`` example corpus, tagged ``source_type="pwml_example"``.

    One-or-more chunks per ``*.pwml`` / ``*.sbml`` file so example-pathway
    retrieval becomes semantic. Chunk ids are stable per ``(filename, "example",
    ordinal)`` so re-indexing the corpus is idempotent.
    """
    root = (
        Path(corpus_dir)
        if corpus_dir is not None
        else (PROJECT_ROOT / _CORPUS_DIRNAME)
    )
    if not root.exists():
        return []
    chunks: List[Chunk] = []
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in (_SBML_SUFFIXES | _PWML_SUFFIXES)
    )
    for path in files:
        text = _corpus_text_for_file(path)
        if not text.strip():
            continue
        source_id = path.name
        for ordinal, window in enumerate(_iter_windows(text)):
            chunks.append(
                Chunk(
                    id=_chunk_id(source_id, "example", ordinal),
                    text=window,
                    source_id=source_id,
                    source_title=path.stem,
                    source_type="pwml_example",
                    source_uri=path.as_uri(),
                    section="example",
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Ingest orchestration.
# ---------------------------------------------------------------------------
def ingest(
    selection: "Selection",
    *,
    store: Optional[VectorStore] = None,
    embedder: Optional[Embedder] = None,
    corpus_dir: Optional[Path] = None,
    index_corpus: bool = True,
    db_records: Optional[Sequence[Dict[str, Any]]] = None,
    persist: bool = True,
    seed: Optional[CandidatePaper] = None,
) -> IngestReport:
    """Chunk -> embed -> upsert -> persist the selected papers (+ corpus + records).

    Chunks every ``selection.selected`` paper (:func:`chunk_paper`), optionally the
    ``reference/`` example corpus (:func:`chunk_corpus`) and any structured DB
    reaction records (:func:`chunk_db_reactions`), embeds them through the WP0
    :class:`Embedder` (which caches, so unchanged chunks are never re-embedded),
    upserts them to ``store`` and persists. Returns an :class:`IngestReport`.

    Offline-first: with no embedding endpoint the embedder falls back to its
    deterministic lexical vectors, so ingest never hard-fails. Re-ingesting
    unchanged inputs is a no-op (stable chunk ids + embedding-cache hits ->
    ``embedded == 0``).
    """
    embedder = embedder if embedder is not None else Embedder(config=rag_config())
    if store is None:
        store = get_vector_store(embed_fn=embedder.embed)

    papers = list(getattr(selection, "selected", None) or [])
    # The seed is indexed alongside the retrieved papers (see seed_candidate) so
    # its passages are retrievable and quotable like any other source. It is
    # counted separately from ``papers`` because it was not acquired or selected.
    if seed is not None and str(getattr(seed, "id", "") or ""):
        papers = [seed, *papers]
    chunks: List[Chunk] = []
    for paper in papers:
        chunks.extend(chunk_paper(paper))

    corpus_files = 0
    if index_corpus:
        corpus_chunks = chunk_corpus(corpus_dir)
        corpus_files = len({c.source_id for c in corpus_chunks})
        chunks.extend(corpus_chunks)

    db_chunks: List[Chunk] = []
    if db_records:
        db_chunks = chunk_db_reactions(db_records)
        chunks.extend(db_chunks)

    hits_before = int(embedder.stats.get("hits", 0))
    misses_before = int(embedder.stats.get("misses", 0))

    # Embed via the WP0 embedder so its cache decides what is (re)computed; the
    # store's upsert then sees embeddings already set and does not re-embed.
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = list(vector)

    report = store.upsert(chunks)

    if hasattr(embedder, "save"):
        embedder.save()
    if persist:
        store.persist()

    embedded_new = int(embedder.stats.get("misses", 0)) - misses_before
    cache_hits = int(embedder.stats.get("hits", 0)) - hits_before
    return IngestReport(
        papers=len(papers),
        corpus_files=corpus_files,
        db_records=len(db_chunks),
        chunks=len(chunks),
        embedded=embedded_new,
        cache_hits=cache_hits,
        upserted=report.upserted,
        skipped=report.skipped,
        store_total=report.total,
        backend=report.backend,
    )


# ---------------------------------------------------------------------------
# Hybrid (semantic + lexical) scorer — the WP4-facing seam.
# ---------------------------------------------------------------------------
def _minmax_normalize(scores: List[float]) -> List[float]:
    """Scale a list of scores into [0, 1]; a flat list maps to all-1.0."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [1.0 for _ in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


def _chunk_to_motif_entry(chunk: Chunk) -> Dict[str, Any]:
    """Adapt a :class:`Chunk` to the entry shape ``_score_entry`` consumes.

    The chunk's content tokens are placed in the ``tokens`` / ``species_tokens``
    fields so the wrapped ``_score_entry`` (which the pipeline already uses for
    motif retrieval) scores exact gene/compound-symbol overlap against them.
    """
    tokens = sorted(set(_lexical_tokenize(chunk.text)))
    return {
        "tokens": tokens,
        "species_tokens": tokens,
        "reaction_tokens": [],
        "compartment_tokens": [],
    }


def build_hybrid_scorer(
    store: VectorStore,
    *,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    pool_k: Optional[int] = None,
) -> Callable[..., List[Retrieved]]:
    """Return the WP4 hybrid retriever: ``0.7 * semantic + 0.3 * lexical``.

    The returned callable ``scorer(query, *, top_k=8, filters=None)`` pulls a
    candidate pool from ``store.query`` (semantic half), rescores each candidate
    with the wrapped lexical scorer ``t2pw.sbml.examples._score_entry`` (lexical
    half), min-max normalizes each half, blends with the given weights, and
    returns the top ``top_k`` :class:`Retrieved`.

    The lexical half guarantees an exact gene/compound symbol (e.g. ``NdmA``) is
    never lost to embedding fuzz — and still retrieves it when embeddings are
    unavailable (``store.query`` then returns its own lexical-overlap pool, over
    which the exact-token match still wins). Weights are tunable via the keyword
    arguments.
    """

    def scorer(
        query: str, *, top_k: int = 8, filters: Optional[dict] = None
    ) -> List[Retrieved]:
        k = max(1, int(top_k))
        pool = max(int(pool_k) if pool_k else 0, k * 8, 40)
        retrieved = store.query(query, top_k=pool, filters=filters)
        if not retrieved:
            return []

        query_tokens = set(_lexical_tokenize(query))
        semantic = _minmax_normalize([float(r.score) for r in retrieved])
        lexical_raw = [
            _score_entry(query_tokens, _chunk_to_motif_entry(r.chunk))
            for r in retrieved
        ]
        lexical = _minmax_normalize(lexical_raw)

        blended: List[Retrieved] = []
        for retrieved_item, sem, lex in zip(retrieved, semantic, lexical):
            score = semantic_weight * sem + lexical_weight * lex
            blended.append(
                Retrieved(chunk=retrieved_item.chunk, score=round(score, 6))
            )
        blended.sort(key=lambda r: r.score, reverse=True)
        return blended[:k]

    return scorer


__all__ = [
    "IngestReport",
    "chunk_paper",
    "chunk_db_reactions",
    "chunk_corpus",
    "ingest",
    "build_hybrid_scorer",
]
