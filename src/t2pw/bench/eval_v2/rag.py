"""Evaluation v2 RAG metrics (ORCH-742c). EVALUATION-ONLY; admission rules untouched.

Two independent measurements, never blended into one "RAG accuracy":

ARCHIVED (:func:`archived_failure_split`) -- over committed ``rag_admission_report.json``
files of legs whose paper has a COMPLETE v2 gold case. For every in-scope gold reaction
that the leg's own strict Stage-1 payload did NOT already contain (the reactions RAG was
there to recover), one outcome, exactly the brief's classes:

  A  correct evidence not retrieved   no persisted retrieved span names every defining
                                      participant, and no candidate matches
  B  retrieved, LLM did not use it    some persisted span names every defining
                                      participant, but no candidate matches the reaction
  C  proposed, admission rejected     a candidate matches (strict v2 matcher) and every
                                      matching candidate was rejected
  ok admitted                         a matching candidate was accepted

and, per ACCEPTED candidate, D = "admission accepted an unsupported candidate": its own
retrieved span does not name all of its inputs and outputs, so nothing persisted states
the reaction. An accepted candidate that simply is not the target paper's chemistry is
counted apart as ``accepted_external_biology`` and is NOT an error (D-093).

Honest limit: the pipeline persists only candidate records (with a clipped span), not
the full retrieved chunk list, and reports truncate at ``max_report_entries``. So "not
retrieved" means "not visible in what was persisted"; truncated legs are a separate
population, as in ``evidence/rd093_rag_metrics.py``.

LIVE (:func:`live_retrieval`) -- the production retriever itself (``ingest.chunk_paper``,
``embed.Embedder``, ``store.MemoryVectorStore``, ``ingest.build_hybrid_scorer``,
``retrieve.query_for_gap``) over the COMMITTED ``01_source_text.txt`` of every gold paper.
Every number is labelled with the embedding mode the embedder actually used. No chat or
completion call is made anywhere in this module.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from t2pw.bench.eval_v2 import matcher as M
from t2pw.bench.eval_v2.gold import CaseV2, Reaction
from t2pw.bench.goldset import fold_for_quote

#: ``source_id`` shapes accepted as a valid source reference.
_VALID_SOURCE_ID = re.compile(r"^(PMC\d+|\d{5,9}|10\.\d{4,9}/\S+|seed_paper)$", re.IGNORECASE)


def valid_source_reference(source_id: Any, source_uri: Any) -> bool:
    sid, uri = str(source_id or "").strip(), str(source_uri or "").strip()
    return bool(_VALID_SOURCE_ID.match(sid)) or uri.startswith(("http://", "https://"))


def chunk_provenance(source_id: Any, source_uri: Any) -> Dict[str, Any]:
    """Per retrieved chunk: provenance present, validate_provenance, valid reference.

    ``validate_provenance`` is the production WP6 tripwire; it takes a payload, so the
    chunk is presented as the single reaction a transcription from it would become.
    """

    from t2pw.rag.provenance import validate_provenance

    present = bool(str(source_id or "").strip() or str(source_uri or "").strip())
    report = validate_provenance({"processes": {"reactions": [
        {"name": "probe", "rag_provenance": {"source_id": source_id or "", "source_uri": source_uri or ""}}]}})
    return {"provenance_present": present, "validate_provenance_ok": bool(report.ok),
            "valid_source_reference": valid_source_reference(source_id, source_uri)}


def _names_in_span(names: Sequence[str], keyed_span: str) -> bool:
    return bool(keyed_span.strip()) and bool(names) and all(M.mentions(n, keyed_span) for n in names)


def archived_failure_split(case: CaseV2, admission: Mapping[str, Any],
                           stage1: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    from t2pw.bench.eval_v2.reactions import match_reaction

    recs = []
    for group in ("accepted", "rejected"):
        for r in admission.get(group) or ():
            if isinstance(r, dict):
                ev = r.get("evidence") or {}
                recs.append({"group": group, "row": {"inputs": r.get("inputs") or [], "outputs": r.get("outputs") or []},
                             "span": M.text_key(ev.get("span") or ""),
                             "prov": chunk_provenance(ev.get("source_id"), ev.get("source_uri")),
                             "score": ev.get("score")})
    s1_rows = [r for r in ((stage1 or {}).get("processes") or {}).get("reactions") or [] if isinstance(r, dict)]
    outcomes: Dict[str, str] = {}
    for g in case.reactions:
        if any(match_reaction(g, r)[0] == "full" for r in s1_rows):
            outcomes[g.id] = "already_in_stage1"
            continue
        matching = [x for x in recs if match_reaction(g, x["row"])[0] == "full"]
        if any(x["group"] == "accepted" for x in matching):
            outcomes[g.id] = "ok_admitted"
        elif matching:
            outcomes[g.id] = "C_proposed_but_rejected"
        elif any(all(M.mentions(t.name, x["span"], t.aliases) for t in (*g.defining("inputs"), *g.defining("outputs")))
                 for x in recs):
            outcomes[g.id] = "B_retrieved_not_used"
        else:
            outcomes[g.id] = "A_not_retrieved"
    accepted = [x for x in recs if x["group"] == "accepted"]
    d_unsupported = [x for x in accepted
                     if not _names_in_span(M.participant_names(x["row"]["inputs"])
                                           + M.participant_names(x["row"]["outputs"]), x["span"])]
    external = [x for x in accepted
                if not any(match_reaction(g, x["row"])[0] == "full"
                           for g in (*case.reactions, *case.off_pathway, *case.summaries))]
    prov = collections.Counter()
    for x in recs:
        for k, v in x["prov"].items():
            prov[f"{k}={v}"] += 1
    trunc = admission.get("truncated") or {}
    return {
        "per_gold_reaction": outcomes,
        "counts": dict(collections.Counter(outcomes.values())),
        "accepted_candidates": len(accepted),
        "D_accepted_unsupported": len(d_unsupported),
        "accepted_external_biology": len(external),
        "candidates_persisted": len(recs),
        "truncated": bool(trunc.get("accepted") or trunc.get("rejected")),
        "chunk_provenance": dict(prov),
    }


# --------------------------------------------------------------------------- live retrieval
def _relevant(chunk_text: str, quotes: Sequence[str]) -> bool:
    """A chunk is relevant when it holds a gold quote, or either half of one that a
    chunk boundary split (quotes are folded; halves must be >= 30 folded chars)."""

    folded = fold_for_quote(chunk_text)
    for q in quotes:
        fq = fold_for_quote(q)
        if fq and fq in folded:
            return True
        half = len(fq) // 2
        if half >= 30 and (fq[:half] in folded or fq[half:] in folded):
            return True
    return False


def _names_all_participants(chunk_text: str, g: Reaction) -> bool:
    """Second, paper-agnostic relevance label: the chunk names EVERY defining participant
    of ``g`` on token boundaries, in any paper. Looser than the quote label; reported
    beside it, never instead of it."""

    keyed = M.text_key(chunk_text)
    return all(M.mentions(t.name, keyed, t.aliases) for t in (*g.defining("inputs"), *g.defining("outputs")))


def gap_query(g: Reaction, pathway: str, organism: str) -> str:
    """The production query text for a gap describing ``g`` as missing."""

    from t2pw.rag.retrieve import GAP_DANGLING_REACTION, Gap, query_for_gap

    names = [t.name for t in (*g.defining("inputs"), *g.defining("outputs"))]
    symbols = names + ([g.enzyme.name] if g.enzyme else [])
    gap = Gap(kind=GAP_DANGLING_REACTION, label=" + ".join(t.name for t in g.defining("inputs")),
              symbols=symbols, adjacent_entities=names,
              requested_pathway=pathway, requested_organism=organism)
    return query_for_gap(gap)


def rank_metrics(ranked_relevance: Sequence[Sequence[bool]], k_values=(1, 3, 5)) -> Dict[str, Any]:
    """Recall@k, Precision@5 and MRR over positive queries. Pure; unit-tested."""

    n = len(ranked_relevance)
    out: Dict[str, Any] = {"positive_queries": n}
    for k in k_values:
        hits = sum(1 for rel in ranked_relevance if any(rel[:k]))
        out[f"recall_at_{k}"] = round(hits / n, 4) if n else None
        out[f"recall_at_{k}_hits"] = hits
    p5 = [sum(rel[:5]) / 5 for rel in ranked_relevance]
    out["precision_at_5"] = round(sum(p5) / n, 4) if n else None
    rr = [next((1.0 / (i + 1) for i, r in enumerate(rel) if r), 0.0) for rel in ranked_relevance]
    out["mrr"] = round(sum(rr) / n, 4) if n else None
    return out


def live_retrieval(cases: Mapping[str, CaseV2], texts: Mapping[str, str], gold_meta: Mapping[str, Mapping[str, str]],
                   cache_path: Path, top_k: int = 8) -> Dict[str, Any]:
    """Run the production retriever over committed paper texts. ``cache_path`` MUST be
    outside the repo: the embedding cache is never written into ``data/``."""

    from t2pw.config import rag_config
    from t2pw.rag.acquire import CandidatePaper
    from t2pw.rag.embed import Embedder
    from t2pw.rag.ingest import build_hybrid_scorer, chunk_paper
    from t2pw.rag.store import MemoryVectorStore

    cfg = rag_config()
    embedder = Embedder(cache_path=Path(cache_path), config=cfg)
    chunks = []
    for pid, text in sorted(texts.items()):
        meta = gold_meta.get(pid, {})
        cp = CandidatePaper(id=pid, source="committed_01_source_text", title=meta.get("title", ""),
                            full_text=text, source_uri=f"https://europepmc.org/article/PMC/{pid}",
                            organism=meta.get("organism", ""))
        chunks.extend(chunk_paper(cp))
    vectors = embedder.embed([c.text for c in chunks])
    for c, v in zip(chunks, vectors):
        c.embedding = list(v)
    store = MemoryVectorStore(embed_fn=embedder.embed)
    store.upsert(chunks)
    scorer = build_hybrid_scorer(store)

    positives: List[List[bool]] = []
    positives_any_paper: List[List[bool]] = []
    per_query, prov = [], collections.Counter()
    negatives: List[Dict[str, Any]] = []
    for pid, case in sorted(cases.items()):
        meta = gold_meta.get(pid, {})
        for g in case.reactions:
            q = gap_query(g, meta.get("pathway", ""), meta.get("organism", ""))
            hits = scorer(q, top_k=top_k)
            rel = [h.chunk.source_id == pid and _relevant(h.chunk.text, g.quotes()) for h in hits]
            n_rel = sum(1 for c in chunks if c.source_id == pid and _relevant(c.text, g.quotes()))
            for h in hits:
                for k, v in chunk_provenance(h.chunk.source_id, h.chunk.source_uri).items():
                    prov[f"{k}={v}"] += 1
            if n_rel:
                positives.append(rel)
            rel_any = [_names_all_participants(h.chunk.text, g) for h in hits]
            if any(_names_all_participants(c.text, g) for c in chunks):
                positives_any_paper.append(rel_any)
            per_query.append({"paper": pid, "gold_id": g.id, "relevant_chunks_in_corpus": n_rel,
                              "first_relevant_rank": next((i + 1 for i, r in enumerate(rel) if r), None),
                              "top_sources": [h.chunk.source_id for h in hits[:5]]})
            # NEGATIVE: the same query restricted to each OTHER complete case whose
            # exhaustive gold states no matching reaction -- zero relevant chunks by gold.
            for other_id, other in sorted(cases.items()):
                if other_id == pid:
                    continue
                if any(_same_chemistry(g, o) for o in (*other.reactions, *other.off_pathway, *other.summaries)):
                    continue
                neg_hits = scorer(q, top_k=top_k, filters={"source_id": other_id})
                negatives.append({"query_paper": pid, "gold_id": g.id, "against": other_id,
                                  "hits_returned": len(neg_hits),
                                  "top_score": neg_hits[0].score if neg_hits else None})
    metrics = rank_metrics(positives)
    metrics_any_paper = rank_metrics(positives_any_paper)
    rejected = sum(1 for n in negatives if n["hits_returned"] == 0)
    stats = dict(embedder.stats)
    mode = ("api" if stats.get("api") and not stats.get("lexical") else
            "lexical_fallback" if stats.get("lexical") and not stats.get("api") else
            "mixed" if stats.get("api") and stats.get("lexical") else "cache_only")
    return {
        "embedding_mode": mode,
        "embedding_model": embedder.model,
        "embedder_stats": stats,
        "corpus_papers": sorted(texts),
        "corpus_chunks": len(chunks),
        "top_k": top_k,
        "relevance_label": ("target_quote: a chunk of the TARGET paper holding the gold quote (or half of "
                            "one split by a chunk boundary). Strict: the same chemistry stated by another "
                            "corpus paper counts as NOT relevant."),
        "metrics": metrics,
        "relevance_label_any_paper": ("participants_named: a chunk of ANY corpus paper naming every defining "
                                      "participant on token boundaries. Looser; reported beside, never instead."),
        "metrics_any_paper": metrics_any_paper,
        "score_note": ("build_hybrid_scorer min-max normalises both halves WITHIN each query's pool, so the "
                       "top hit scores ~1.0 whatever its relevance; the returned score cannot support a "
                       "rejection floor, and the production retriever applies none."),
        "queries": per_query,
        "negative_queries": len(negatives),
        "negative_query_rejection_rate": round(rejected / len(negatives), 4) if negatives else None,
        "negative_query_rejected": rejected,
        "negatives": negatives,
        "chunk_provenance": dict(prov),
    }


def _same_chemistry(a: Reaction, b: Reaction) -> bool:
    ka = {M.strict_key(t.name) for t in (*a.defining("inputs"), *a.defining("outputs"))}
    kb = set()
    for t in (*b.defining("inputs"), *b.defining("outputs")):
        kb |= M.term_keys(t.name, t.aliases)
    return bool(ka) and ka <= kb


__all__ = ["archived_failure_split", "chunk_provenance", "gap_query", "live_retrieval",
           "rank_metrics", "valid_source_reference"]
