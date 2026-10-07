"""Passage indexing and retrieval for a mining run.

Reuses the existing RAG infrastructure -- ``t2pw.rag.ingest.chunk_paper`` for
section-aware chunking, ``t2pw.rag.embed.Embedder`` for cached embeddings and
``t2pw.rag.store`` for the vector index -- and adds the two mining-specific
things:

1. every retrieved chunk becomes an :class:`~t2pw.mining.models.EvidenceSpan`
   carrying the query and frontier item that surfaced it, so provenance runs
   backwards from a claim all the way to the question that asked it;
2. the index is **per-run** and defaults to the in-memory backend, so mining
   cannot write into the production RAG index.

Embeddings honesty
------------------
``Embedder`` degrades to deterministic lexical vectors when no embedding
endpoint answers. That degradation is legitimate for offline work but must never
be invisible: :attr:`PassageIndex.embedding_mode` records which mode a run was
actually in, the preflight reports it, and a run configured to require real
embeddings refuses to start without them. "RAG added nothing" is not an
acceptable description of an embeddings outage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from t2pw.mining.acquisition import LoadedPaper
from t2pw.mining.config import MiningConfig
from t2pw.mining.ids import evidence_id as make_evidence_id
from t2pw.mining.models import EvidenceSpan

EMBEDDING_API = "api"
EMBEDDING_LEXICAL = "lexical_fallback"
EMBEDDING_UNKNOWN = "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Text hygiene: what of a full text is the paper's own prose
# ---------------------------------------------------------------------------
# A EuropePMC full text is the whole JATS document flattened to one line: journal
# front matter (ids, ISSNs, authors, affiliations, e-mails, dates, licence, the
# abstract again, processing flags), the body, back matter, the reference list,
# and in older layouts the figure and table legends AFTER the references
# (PMC8508710). rag's chunker labels sections by keyword and cuts a reference list
# only when its heading falls in the last 45% of the text. Pilot-003 therefore
# indexed 25 kB of the GH3.10 paper's bibliography as "results" (claims #31/#32
# cited it), every paper's front matter as "body#0", and an MDPI disclaimer as a
# 61-character "methods" passage. None of that may become an evidence span. The
# section labels cannot be trusted, so these rules read the text itself.

#: A reference-list heading as printed: a heading, not "see references therein".
_REF_HEADING = re.compile(
    r"(?<![\w-])(?:References|REFERENCES|Literature Cited|LITERATURE CITED|Bibliography|BIBLIOGRAPHY)(?![\w-])")
#: One bibliographic mark: a DOI, a PMCID, a bare 6-8 digit PMID, a "( 2019 )" or
#: "2020;298" citation year, or a "14 , 171 – 178" volume and page range.
_CITATION_MARK = re.compile(
    r"\b10\.\d{4,9}/[^\s,;]+"
    r"|\bPMC\d{5,9}\b"
    r"|(?<![\w.,/:+-])\d{6,8}(?![\w.,/:+-])"
    r"|\(\s*(?:19|20)\d\d[a-z]?\s*\)"
    r"|\b(?:19|20)\d\d\s*;\s*\d+"
    r"|\b\d+\s*[,:]\s*\d+\s*[–‐-]\s*\d+\b")
#: Marks per 1000 characters (a text under 1000 characters counts as 1000).
#: ponytail: fit on the seven pilot-002/003 full texts: every reference list reads
#: >= 8.7 in the 1500 characters after its heading and every pilot-003 reference
#: chunk >= 13, while no 5000-character window of body text reads above 1.6.
_CITATION_DENSITY = 5.0
#: How far apart a reference list's marks may be (pilot maximum: 462 characters).
_CITATION_GAP = 600
#: Where the legends resume after a reference list: the next figure or table label OR mention
#: ("Figure 1 In silico", "FIG. 3", "Fig 4", "(Table 1 )"). A mention ends a list early (the
#: pepper paper's at "(Supplemental Table S3) ,") but never late: a stricter label-only rule
#: let a FALSE list -- a table's "References" column with author-year cells -- run past the
#: table's own mentions to the end of the text and drop the body prose after it (PMC8510960,
#: PMC12925505), and lost legends that open "(A)", "|", a dash, "jar1", "13-LOX" or "Δ".
_CAPTION = re.compile(r"(?<![\w.])(?:Figure|FIGURE|Fig\.?|FIG\.?|Table|TABLE|Scheme)\s*S?\d+")
#: The processing flags that close EuropePMC's flattened front matter:
#: "pmc-prop-open-access yes", "is-retracted no", "pmc-license-ref", "Issue date 2019 Jul 9.",
#: and ACS's "document-id-old-9 id5c00566", "production-flag-si-avail yes", "document-id-alt-64 <hash>".
_FRONT_FLAG = re.compile(
    r"\b(?:pmc|is)-[a-z][a-z-]* (?:yes|no|\d+)\b|\bpmc-license-ref\b|\bdisplay-pdf (?:yes|no)\b"
    r"|\b(?:document-id|production-flag)-[\w-]+ \S+"
    r"|\bCollection date (?:19|20)\d\d\b(?: [A-Z][a-z]{2}\.?)?"
    r"|\bIssue date (?:19|20)\d\d [A-Z][a-z]{2}\.? \d{1,2}\.?")
#: Flags are only looked for in this leading share of the text.
_FRONT_SHARE = 0.4
#: A first section heading this close after the last flag also ends the front matter,
#: taking a short tail ("CC BY 1.", "Editor: F. Loreto") with it. Case-sensitive: a
#: heading, not the word in prose.
_FRONT_TAIL = 300
_FIRST_HEADING = re.compile(
    r"(?<![\w-])(?:Introduction|INTRODUCTION|Background|BACKGROUND|Results|RESULTS"
    r"|Materials? and Methods|MATERIALS? AND METHODS|Methods|METHODS)(?![\w-])")
#: Publisher disclaimers: the two MDPI notes in the pilot-002/003 texts and Frontiers'
#: standard one. ponytail: three endings; another publisher's note passes until one is seen.
_PUBLISHER_NOTE = re.compile(
    r"(?:Disclaimer/)?Publisher[’']s [Nn]ote:?.{0,600}?"
    r"(?:referred to in the content|institutional affiliations|endorsed by the publisher)\.")
#: A chunk trimmed to less than this is a fragment, not a passage.
_MIN_TRIMMED_CHARS = 100
#: A reference list with no heading (EuropePMC's flattening of ACS and IUCr articles) is
#: found from its density: the first 1500-character window, anchored on a citation mark past
#: the first third of the text, that is citation-dense AND names this many authors with
#: initials ("Andrade F. F.", "Aragão, D.,", "Moskowitz SM,", "Siegel RL Kratzer TB"). The
#: authors keep out DOI-dense data and supplement lists (PMC13233763's key resources table,
#: PMC12782028's supplement list).
_AUTHOR = re.compile(
    r"\b[A-Z][a-z'’‐-]+(?:,? (?:[A-Z]\.[ -]?)+| [A-Z]{1,3}(?: ?[,.]|(?= [A-Z][a-z'’‐-]+ [A-Z]{1,3}\b)))")
_MIN_AUTHORS = 3
#: The list's first entry starts before its first citation mark: the start backs over
#: author names no further apart than this.
_ENTRY_HEAD = 300
#: The abstract inside EuropePMC's front matter: from its heading to the keyword, funding
#: or flag block that follows it, cut back to its last full stop.
_ABSTRACT_HEADING = re.compile(r"(?<![\w-])(?:Abstract|ABSTRACT)(?![\w-])")
_ABSTRACT_END = re.compile(
    r"(?<![\w-])(?:Keywords|KEYWORDS|Key words|Key Words)(?![\w-])|(?:https?://dx\.doi\.org/)?\b10\.13039/")


def citation_dense(text: str) -> bool:
    """Is ``text`` mostly bibliography? Marks per 1000 characters >= the density bar."""
    return len(_CITATION_MARK.findall(text)) * 1000 >= _CITATION_DENSITY * max(len(text), 1000)


def text_regions(text: str) -> List[Tuple[int, int, str]]:
    """``(start, end, reason)`` for each span of ``text`` that is not the paper's own prose.

    ``text`` is whitespace-normalised. Reasons: ``front_matter`` (from the start to
    the last processing flag, plus a short tail up to the first section heading),
    ``publisher_note``, ``reference_list`` (a heading followed by citation-dense
    text, to the end of its run of citation marks and then to the next figure or
    table label or mention, or to the end of the text). When no heading is found, the list is
    found from its density (:func:`_unheaded_list_start`, A2: PMC12624714's ACS
    bibliography leaked into the chunk that straddles it). Without a flag block there
    is no front-matter region: ponytail: the rule knows EuropePMC's flattening only; a
    PDF upload's title page is not recognised.
    """
    regions: List[Tuple[int, int, str]] = []
    flags = list(_FRONT_FLAG.finditer(text, 0, int(len(text) * _FRONT_SHARE)))
    if flags:
        end = flags[-1].end()
        heading = _FIRST_HEADING.search(text, end, end + _FRONT_TAIL)
        regions.append((0, heading.start() if heading else end, "front_matter"))
    regions += [(m.start(), m.end(), "publisher_note") for m in _PUBLISHER_NOTE.finditer(text)]
    for heading in _REF_HEADING.finditer(text):
        if any(start <= heading.start() < end for start, end, _ in regions):
            continue
        if not citation_dense(text[heading.end():heading.end() + 1500]):
            continue
        regions.append((heading.start(), _list_end(text, heading.end()), "reference_list"))
    if not any(reason == "reference_list" for _, _, reason in regions):
        front_end = max((end for _, end, reason in regions if reason == "front_matter"), default=0)
        start = _unheaded_list_start(text, max(front_end, len(text) // 3))
        while start is not None:
            # A mention ends a list early, so look again after it: PMC13418202's run opened at
            # its Supporting Information DOI, ended at "(Table S1, PDF)", and its ACS
            # bibliography ("De Vreese R. D'hooghe M. ...") followed with no heading.
            end = _list_end(text, start)
            regions.append((start, end, "reference_list"))
            start = _unheaded_list_start(text, end)
    return sorted(regions)


def _list_end(text: str, start: int) -> int:
    """Where a reference list starting at ``start`` ends: past its run of citation marks
    (no two further apart than ``_CITATION_GAP``), at the next figure or table label or
    mention (:data:`_CAPTION`), or at the end of the text."""
    end = start
    for mark in _CITATION_MARK.finditer(text, start):
        if mark.start() - end > _CITATION_GAP:
            break
        end = mark.end()
    caption = _CAPTION.search(text, end)
    return caption.start() if caption else len(text)


def _unheaded_list_start(text: str, after: int) -> Optional[int]:
    """The start of a reference list that has no heading, or ``None``."""
    for mark in _CITATION_MARK.finditer(text, after):
        window = text[mark.start():mark.start() + 1500]
        if citation_dense(window) and len(_AUTHOR.findall(window)) >= _MIN_AUTHORS:
            start = mark.start()
            while True:
                names = list(_AUTHOR.finditer(text, max(after, start - _ENTRY_HEAD), start))
                if not names:
                    return start
                start = names[0].start()
    return None


def front_matter_abstract(text: str) -> str:
    """The "Abstract ..." passage inside EuropePMC's front matter, or ``""`` (A5).

    For a record with no abstract of its own: the front-matter region is never indexed,
    and it may hold the paper's only copy of its abstract (GH3.10 supplied as a bare
    PMCID: "In flowers and wounded leaves of gh3.10 jar1 knockout lines JA-Ile was
    below the detection limit."). The passage runs from the heading to the keyword,
    funding or flag block that follows it, cut back to its last full stop; it needs
    an "Abstract" heading (PMC8508710's front matter has none).
    """
    norm = " ".join(str(text or "").split())
    front = next((end for _, end, reason in text_regions(norm) if reason == "front_matter"), 0)
    heading = _ABSTRACT_HEADING.search(norm, 0, front)
    if not heading:
        return ""
    stops = [m.start() for m in (_ABSTRACT_END.search(norm, heading.end(), front),
                                  _FRONT_FLAG.search(norm, heading.end(), front)) if m]
    passage = norm[heading.end():min(stops, default=front)]
    return passage[:passage.rfind(".") + 1].strip()


def body_text(text: str) -> str:
    """``text`` whitespace-normalised, without the :func:`text_regions` spans."""
    norm = " ".join(str(text or "").split())
    pieces: List[str] = []
    at = 0
    for start, end, _ in text_regions(norm):
        pieces.append(norm[at:start])
        at = max(at, end)
    pieces.append(norm[at:])
    return " ".join(p.strip() for p in pieces if p.strip())


def _outside(start: int, end: int, regions: Sequence[Tuple[int, int, str]]) -> Tuple[int, int]:
    """The longest part of ``[start, end)`` that no region covers. ``(0, 0)`` if none is left.

    ponytail: one piece. A chunk with a region in its middle keeps its longer side
    rather than gluing two pieces into a passage the paper never printed.
    """
    pieces = [(start, end)]
    for r_start, r_end, _ in regions:
        pieces = [p for s, e in pieces for p in ((s, min(e, r_start)), (max(s, r_end), e)) if p[1] > p[0]]
    return max(pieces, key=lambda p: p[1] - p[0], default=(0, 0))


class EmbeddingsUnavailable(RuntimeError):
    """The configured embeddings service did not answer and was required."""


@dataclass
class IndexedPaper:
    """Bookkeeping for one paper's passages."""

    paper_id: str
    chunk_ids: List[str] = field(default_factory=list)
    chars: int = 0
    sections: Dict[str, int] = field(default_factory=dict)
    #: What was kept out of the index and why: reason -> {"count", "chars"}. Text
    #: regions (``front_matter``, ``publisher_note``, ``reference_list``) count
    #: regions; ``chunk_inside_excluded_text``, ``citation_dense_chunk``,
    #: ``duplicate_chunk`` and ``over_passage_cap`` count chunks.
    excluded: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def exclude(self, reason: str, chars: int) -> None:
        row = self.excluded.setdefault(reason, {"count": 0, "chars": 0})
        row["count"] += 1
        row["chars"] += int(chars)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "chunks": len(self.chunk_ids),
            "chars": self.chars,
            "sections": dict(self.sections),
            "excluded": {k: dict(v) for k, v in self.excluded.items()},
        }


class PassageIndex:
    """A per-run passage store with provenance-carrying retrieval."""

    def __init__(
        self,
        config: MiningConfig,
        *,
        embedder: Optional[Any] = None,
        store: Optional[Any] = None,
        require_embeddings: bool = False,
    ) -> None:
        self.config = config
        self.require_embeddings = bool(require_embeddings)
        self.papers: Dict[str, IndexedPaper] = {}
        self.texts: Dict[str, str] = {}
        self._chunk_meta: Dict[str, Dict[str, Any]] = {}
        self._embedder = embedder
        self._store = store
        self.embedding_mode = EMBEDDING_UNKNOWN
        self.warnings: List[str] = []

    # -- lazy construction --------------------------------------------------
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from t2pw.rag.embed import Embedder

            self._embedder = Embedder()
        return self._embedder

    @property
    def store(self) -> Any:
        if self._store is None:
            from t2pw.rag.store import get_vector_store

            self._store = get_vector_store(
                backend=self.config.vector_backend, embed_fn=self.embedder.embed
            )
        return self._store

    def _record_embedding_mode(self) -> None:
        stats = getattr(self.embedder, "stats", None)
        if not isinstance(stats, dict):
            return
        api = int(stats.get("api", 0) or 0)
        lexical = int(stats.get("lexical", 0) or 0)
        if api and not lexical:
            self.embedding_mode = EMBEDDING_API
        elif lexical:
            self.embedding_mode = EMBEDDING_LEXICAL
            message = (
                f"embeddings degraded to the deterministic lexical fallback for {lexical} "
                "passage(s): the configured embedding endpoint did not answer. Retrieval "
                "quality is materially lower; this is an infrastructure state, not a "
                "finding about the literature."
            )
            if message not in self.warnings:
                self.warnings.append(message)
            if self.require_embeddings:
                raise EmbeddingsUnavailable(message)

    # -- indexing -----------------------------------------------------------
    def add_paper(self, paper: LoadedPaper) -> IndexedPaper:
        """Chunk, embed and index one paper. Idempotent per paper id."""
        record = paper.record
        if record.paper_id in self.papers:
            return self.papers[record.paper_id]

        indexed = IndexedPaper(paper_id=record.paper_id, chars=len(paper.text))
        # Hygiene BEFORE the cap: a reference chunk must not displace a body chunk.
        chunks = self._keep_prose(self._chunk(paper), paper.text, indexed)
        cap = max(1, int(self.config.budgets.max_passages_per_paper or len(chunks) or 1))
        for chunk in chunks[cap:]:
            indexed.exclude("over_passage_cap", len(chunk.text))
        chunks = chunks[:cap]
        if indexed.excluded:
            note = "indexing kept out: " + ", ".join(
                f"{reason} x{row['count']} ({row['chars']} chars)" for reason, row in indexed.excluded.items())
            if note not in record.notes:
                record.notes.append(note)

        ordinals: Dict[str, int] = {}
        for chunk in chunks:
            section = str(getattr(chunk, "section", "") or "body")
            ordinal = ordinals.get(section, 0)
            ordinals[section] = ordinal + 1
            indexed.sections[section] = ordinals[section]
            indexed.chunk_ids.append(chunk.id)
            start = paper.text.find(chunk.text) if paper.text else -1
            self._chunk_meta[chunk.id] = {
                "paper_id": record.paper_id,
                "section": section,
                "ordinal": ordinal,
                "char_start": start if start >= 0 else None,
                "char_end": (start + len(chunk.text)) if start >= 0 else None,
            }

        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = list(vector)
            self.store.upsert(chunks)
            self._record_embedding_mode()

        self.papers[record.paper_id] = indexed
        self.texts[record.paper_id] = paper.text
        record.full_text_chars = len(paper.text)
        return indexed

    @staticmethod
    def _keep_prose(chunks: List[Any], text: str, indexed: IndexedPaper) -> List[Any]:
        """The chunks that are the paper's own prose, each once (F-4, A4).

        rag chunks the full text as it is (so an untouched chunk keeps its id and
        text); here each chunk loses what falls in a :func:`text_regions` span,
        keeping its longest remaining piece. A chunk left with a fragment, a chunk
        that is still citation-dense (a reference list without a recognised
        heading), and a chunk whose whitespace-normalised text equals or is
        contained in a chunk already kept (an abstract-only paper's ``abstract#0``
        and ``body#0``; a figure snippet inside its section) are dropped. Abstract
        chunks come from the record, not the full text, so no region trims them.
        """
        norm = " ".join(str(text or "").split())
        regions = text_regions(norm)
        for start, end, reason in regions:
            indexed.exclude(reason, end - start)
        kept: List[Any] = []
        bodies: List[str] = []
        for chunk in chunks:
            body = " ".join(str(chunk.text or "").split())
            at = norm.find(body) if regions and chunk.section != "abstract" else -1
            if at >= 0:
                start, end = _outside(at, at + len(body), regions)
                if (start, end) != (at, at + len(body)):
                    piece = norm[start:end].strip()
                    if len(piece) < _MIN_TRIMMED_CHARS:
                        indexed.exclude("chunk_inside_excluded_text", len(body))
                        continue
                    chunk.text = body = piece
            if citation_dense(body):
                indexed.exclude("citation_dense_chunk", len(body))
            elif any(body in other for other in bodies):
                indexed.exclude("duplicate_chunk", len(body))
            else:
                bodies.append(body)
                kept.append(chunk)
        return kept

    def _chunk(self, paper: LoadedPaper) -> List[Any]:
        """Section-aware chunking through the existing RAG chunker."""
        from t2pw.rag.acquire import CandidatePaper
        from t2pw.rag.ingest import chunk_paper

        record = paper.record
        candidate = CandidatePaper(
            id=record.paper_id,
            source=record.acquisition_source or "mining",
            title=record.title,
            abstract=record.abstract or front_matter_abstract(paper.text),
            full_text=paper.text,
            source_uri=record.url,
            year=record.year,
            observed_organisms=list(record.observed_organisms),
            observed_pathways=list(record.observed_pathways),
        )
        chunks = chunk_paper(candidate)
        if chunks:
            return chunks
        # A very short document (an abstract-only record, a tiny fixture) can
        # fall through the section splitter. Index it whole rather than losing it.
        from t2pw.rag.store import Chunk

        body = paper.text or record.abstract
        if not body.strip():
            return []
        return [
            Chunk(
                id=f"{record.paper_id}::whole",
                text=body,
                source_id=record.paper_id,
                source_title=record.title,
                source_type="paper",
                source_uri=record.url,
                section="body",
            )
        ]

    # -- retrieval ----------------------------------------------------------
    def retrieve(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        round_index: int = 0,
        query_id: str = "",
        frontier_item_id: str = "",
        paper_ids: Optional[Sequence[str]] = None,
    ) -> List[EvidenceSpan]:
        """Retrieve passages and return them as evidence spans.

        ``paper_ids`` restricts retrieval to a subset -- used when extracting
        from one specific paper, so passages from a different paper can never be
        cited as that paper's evidence. Each requested paper's OWN chunks are
        ranked (the store's ``source_id`` filter): a store-wide top ``3k`` filtered
        afterwards gave a paper whose chunks rank below other papers' nothing at
        all (A7: pilot-002's PMC5104190, "no passages retrieved").
        """
        if not self.papers:
            return []
        k = int(top_k or self.config.retrieve_top_k or 8)
        query = str(query_text or "")
        try:
            if paper_ids:
                hits = sorted(
                    (hit for pid in dict.fromkeys(paper_ids)
                     for hit in self.store.query(query, top_k=k, filters={"source_id": pid})),
                    key=lambda hit: float(getattr(hit, "score", 0.0) or 0.0), reverse=True)
            else:
                hits = self.store.query(query, top_k=max(k * 3, k))
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"retrieval failed: {type(exc).__name__}: {exc}")
            return []
        self._record_embedding_mode()

        allowed = set(paper_ids) if paper_ids else None
        spans: List[EvidenceSpan] = []
        for hit in hits:
            chunk = getattr(hit, "chunk", hit)
            source = str(getattr(chunk, "source_id", "") or "")
            if allowed is not None and source not in allowed:
                continue
            meta = self._chunk_meta.get(str(getattr(chunk, "id", "")), {})
            text = str(getattr(chunk, "text", "") or "").strip()
            if not text:
                continue
            section = str(meta.get("section") or getattr(chunk, "section", "") or "body")
            ordinal = meta.get("ordinal")
            spans.append(
                EvidenceSpan(
                    evidence_id=make_evidence_id(source, str(getattr(chunk, "id", "")), text),
                    paper_id=source,
                    text=text,
                    section=section,
                    paragraph=(int(ordinal) + 1) if isinstance(ordinal, int) else None,
                    locator=f"{section}#{ordinal}" if ordinal is not None else section,
                    chunk_id=str(getattr(chunk, "id", "")),
                    char_start=meta.get("char_start"),
                    char_end=meta.get("char_end"),
                    context_ref=str(getattr(chunk, "source_title", "") or ""),
                    extraction_round=int(round_index),
                    retrieval_query_id=query_id,
                    frontier_item_id=frontier_item_id,
                    retrieval_score=float(getattr(hit, "score", 0.0) or 0.0),
                    retrieved_at=_utcnow(),
                )
            )
            if len(spans) >= k:
                break
        return spans

    def spans_for_paper(
        self,
        paper_id: str,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        round_index: int = 0,
        query_id: str = "",
        frontier_item_id: str = "",
    ) -> List[EvidenceSpan]:
        return self.retrieve(
            query_text,
            top_k=top_k,
            round_index=round_index,
            query_id=query_id,
            frontier_item_id=frontier_item_id,
            paper_ids=[paper_id],
        )

    def stats(self) -> Dict[str, Any]:
        excluded: Dict[str, Dict[str, int]] = {}
        for paper in self.papers.values():
            for reason, row in paper.excluded.items():
                total = excluded.setdefault(reason, {"count": 0, "chars": 0})
                total["count"] += row["count"]
                total["chars"] += row["chars"]
        return {
            "papers_indexed": len(self.papers),
            "chunks": sum(len(p.chunk_ids) for p in self.papers.values()),
            "excluded": excluded,
            "excluded_by_paper": {pid: {k: dict(v) for k, v in p.excluded.items()}
                                  for pid, p in self.papers.items() if p.excluded},
            "embedding_mode": self.embedding_mode,
            "vector_backend": self.config.vector_backend,
            "warnings": list(self.warnings),
        }


__all__ = [
    "EMBEDDING_API",
    "EMBEDDING_LEXICAL",
    "EMBEDDING_UNKNOWN",
    "EmbeddingsUnavailable",
    "IndexedPaper",
    "PassageIndex",
    "body_text",
    "citation_dense",
    "front_matter_abstract",
    "text_regions",
]
