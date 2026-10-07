"""The iterative frontier-expansion controller.

This is the part of pathway mining that is genuinely a different controller, not
a reconfigured one. The production RAG loop is gap-driven: it knows the pathway
it is trying to complete and retrieves evidence for a fixed vocabulary of gaps.
This loop is expansion-driven: each round's questions come from what the previous
round just discovered, so the network grows outward until a limit or convergence
stops it.

One round
---------
1. select frontier items
2. formulate traceable literature queries for them
3. acquire, screen, cache and index papers
4. retrieve passages
5. extract structured claims, every field cited to a passage
6. normalize entities within the run
7. deduplicate claims without losing independent supporting evidence
8. classify: accepted / candidate / conflicting / incomplete / rejected /
   hypothesis-only
9. place admitted reactions by chemistry -- primary graph or quarantined
   candidate component
10. record the complete graph delta
11. generate the next frontier from what was just found
12. stop at the first limit reached, naming it

Everything expensive is sequential. There is no parallelism around the shared
caches, the vector store or the budget ledger.
"""

from __future__ import annotations

import traceback
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from t2pw.mining import artifacts as art
from t2pw.mining import frontier as frontier_mod
from t2pw.mining import queries as queries_mod
from t2pw.mining import viz
from t2pw.mining.acquisition import Acquirer, LoadedPaper, PaperSource
from t2pw.mining.admission import ADMITTED_STATUSES, clear_disputed_attributes, judge
from t2pw.mining.artifacts import (
    INCOMPLETE_STATES,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_PREFLIGHT_FAILED,
    STATE_RUNNING,
    STATE_STOPPED,
    RunDirectory,
    RunManifest,
    build_manifest,
)
from t2pw.mining.budget import (
    BudgetLedger,
    STOP_COMPLETED,
    STOP_ERROR,
    STOP_FRONTIER_EXHAUSTED,
    STOP_PREFLIGHT_FAILED,
)
from t2pw.mining.claims import ClaimStore
from t2pw.mining.config import MiningConfig
from t2pw.mining.extraction import extract_claims
from t2pw.mining.flags import require_mining_enabled
from t2pw.mining.graph import ResearchGraph, load_starting_graph
from t2pw.mining.models import (
    ClaimDecision,
    ConflictRecord,
    EvidenceSpan,
    ExtractedClaim,
    FrontierItem,
    GraphDelta,
    LiteratureQuery,
    ModelCallRecord,
    PaperRecord,
    RoundCheckpoint,
    STATUS_REJECTED,
    provenance_chain_complete,
)
from t2pw.mining.modelclient import MiningModelClient
from t2pw.mining.normalize import EntityRegistry
from t2pw.mining.preflight import PreflightReport, run_preflight
from t2pw.mining.report import build_report, build_summary
from t2pw.mining.resolver import EntityResolver, default_resolver


@dataclass
class MiningState:
    """Everything a run holds, and everything a resume must restore."""

    graph: ResearchGraph
    store: ClaimStore = field(default_factory=ClaimStore)
    papers: Dict[str, PaperRecord] = field(default_factory=dict)
    paper_texts: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, EvidenceSpan] = field(default_factory=dict)
    queries: List[LiteratureQuery] = field(default_factory=list)
    frontier: List[FrontierItem] = field(default_factory=list)
    exhausted_frontier: List[str] = field(default_factory=list)
    model_calls: List[ModelCallRecord] = field(default_factory=list)
    deltas: List[GraphDelta] = field(default_factory=list)
    round_records: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "store": self.store.to_dict(),
            "papers": {k: v.to_dict() for k, v in self.papers.items()},
            # Paper text is carried so a resumed run can rebuild its passage
            # index without re-fetching (and without a second round of charges).
            "paper_texts": dict(self.paper_texts),
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "queries": [q.to_dict() for q in self.queries],
            "frontier": [f.to_dict() for f in self.frontier],
            "exhausted_frontier": list(self.exhausted_frontier),
            "model_calls": [m.to_dict() for m in self.model_calls],
            "deltas": [d.to_dict() for d in self.deltas],
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MiningState":
        payload = dict(data or {})
        state = cls(graph=ResearchGraph.from_dict(payload.get("graph") or {}))
        state.store = ClaimStore.from_dict(payload.get("store"))
        state.papers = {
            str(k): PaperRecord.from_dict(v) for k, v in (payload.get("papers") or {}).items()
        }
        state.paper_texts = {
            str(k): str(v) for k, v in (payload.get("paper_texts") or {}).items()
        }
        state.evidence = {
            str(k): EvidenceSpan.from_dict(v) for k, v in (payload.get("evidence") or {}).items()
        }
        state.queries = [LiteratureQuery.from_dict(q) for q in payload.get("queries") or []]
        state.frontier = [FrontierItem.from_dict(f) for f in payload.get("frontier") or []]
        state.exhausted_frontier = [str(f) for f in payload.get("exhausted_frontier") or []]
        state.model_calls = [ModelCallRecord.from_dict(m) for m in payload.get("model_calls") or []]
        state.deltas = [GraphDelta.from_dict(d) for d in payload.get("deltas") or []]
        state.errors = [str(e) for e in payload.get("errors") or []]
        return state


@dataclass
class MiningOutcome:
    """What a finished (or stopped, or cancelled) run produced."""

    run_dir: RunDirectory
    manifest: RunManifest
    state: MiningState
    ledger: BudgetLedger
    preflight: Optional[PreflightReport] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def completed_normally(self) -> bool:
        return self.manifest.completion_state == STATE_COMPLETED

    @property
    def stop_reason(self) -> str:
        return self.manifest.stop_reason


ProgressFn = Callable[[str, Dict[str, Any]], None]


class MiningController:
    """Drives one mining run. The only entry point the CLI and the UI both use."""

    def __init__(
        self,
        config: MiningConfig,
        *,
        source: Optional[PaperSource] = None,
        chat_fn: Optional[Callable[..., Any]] = None,
        index: Optional[Any] = None,
        resolver: Optional[EntityResolver] = None,
        progress: Optional[ProgressFn] = None,
        enabled_override: Optional[bool] = None,
        run_dir: Optional[RunDirectory] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        require_mining_enabled(enabled_override)
        self.config = config
        self.enabled_override = enabled_override
        self.ledger = BudgetLedger(config.budgets, clock=clock)
        self.resolver = resolver or default_resolver()
        self._progress = progress
        self._chat_fn = chat_fn
        self._source = source
        self._index = index
        self.run_dir = run_dir or RunDirectory.create(config)
        self.manifest = build_manifest(
            config, run_id=self.run_dir.run_id, enabled_via=config.enabled_via
        )
        self.state = MiningState(
            graph=ResearchGraph(registry=EntityRegistry(), topic=config.topic)
        )
        self.client = MiningModelClient(config.model, self.ledger, chat_fn=chat_fn, clock=clock)
        self.acquirer = Acquirer(config, source=source)
        self.preflight: Optional[PreflightReport] = None
        self._round_index = 0
        #: A resumed run must not re-adopt its starting graph or re-load its
        #: explicit papers; both are already in the restored state.
        self._setup_done = False

    # -- plumbing -----------------------------------------------------------
    def emit(self, event: str, **payload: Any) -> None:
        if self._progress is not None:
            try:
                self._progress(event, dict(payload))
            except Exception:  # noqa: BLE001 - a UI callback must never kill a run
                pass

    @property
    def index(self) -> Any:
        if self._index is None:
            from t2pw.mining.retrieval import PassageIndex

            self._index = PassageIndex(self.config)
        return self._index

    def cancel(self) -> None:
        """Request a graceful stop after the current round is flushed."""
        self.ledger.cancel()

    # -- setup --------------------------------------------------------------
    def _adopt_starting_graph(self) -> None:
        for item in self.config.inputs_of("pathway_graph"):
            payload = None
            if item.source_path:
                payload = art.read_json(art.Path(item.source_path))
            if payload is None:
                self.state.errors.append(
                    f"starting graph {item.raw!r} could not be read; it was NOT used"
                )
                continue
            adopted = load_starting_graph(payload, registry=self.state.graph.registry)
            # Read-only: the supplied network seeds the graph and is never
            # written back, never exported, and never converted to PWML.
            for reaction in adopted.reactions.values():
                self.state.graph.add_reaction(reaction, round_index=0)
            self.state.graph.registry = adopted.registry
            self.state.graph.select_primary_component(round_index=0, topic=self.config.topic)
            self.emit("starting_graph_adopted", reactions=len(adopted.reactions))

    def _load_explicit_papers(self) -> List[LoadedPaper]:
        loaded, errors = self.acquirer.load_explicit(
            self.config.inputs_of("paper"), round_index=0
        )
        self.state.errors.extend(errors)
        for item in self.config.inputs_of("paper"):
            if item.id_type != "file":
                continue
            self.state.errors.append(
                f"uploaded file {item.file_name!r} must be supplied through "
                "add_uploaded_paper(); it was declared but no bytes were provided"
            )
        return loaded

    def add_uploaded_paper(self, paper: LoadedPaper) -> PaperRecord:
        """Register an uploaded paper (Streamlit's upload path).

        Deterministic on duplicates: the same bytes re-register the same
        ``paper_id`` and merge rather than creating a second citation.
        """
        record = paper.record
        existing = self.state.papers.get(record.paper_id)
        if existing is not None:
            existing.notes.append(f"duplicate upload of {record.file_name!r} ignored")
            return existing
        self.state.papers[record.paper_id] = record
        self.state.paper_texts[record.paper_id] = paper.text
        self.acquirer.seen_paper_ids.add(record.paper_id)
        self.index.add_paper(paper)
        return record

    # -- the run ------------------------------------------------------------
    def run(
        self,
        *,
        probe_model: bool = True,
        probe_embeddings: bool = True,
        uploaded: Optional[Sequence[LoadedPaper]] = None,
    ) -> MiningOutcome:
        """Preflight, then rounds until a limit, convergence or cancellation."""
        self.run_dir.write_config(self.config)
        self.manifest.completion_state = STATE_RUNNING
        self.run_dir.write_manifest(self.manifest)

        self.preflight = run_preflight(
            self.config,
            probe_model=probe_model,
            probe_embeddings=probe_embeddings,
            enabled_override=self.enabled_override,
            model_client_injected=self._chat_fn is not None,
        )
        self.run_dir.write_preflight(self.preflight)
        self.emit("preflight", ok=self.preflight.ok, report=self.preflight.to_dict())
        if not self.preflight.ok:
            self.ledger.stop(
                STOP_PREFLIGHT_FAILED,
                "; ".join(f"{c.name}: {c.detail}" for c in self.preflight.failures),
            )
            return self._finish(STATE_PREFLIGHT_FAILED)

        try:
            if not self._setup_done:
                self._adopt_starting_graph()
                for paper in uploaded or []:
                    self.add_uploaded_paper(paper)
                for paper in self._load_explicit_papers():
                    self._register_paper(paper)
                self._setup_done = True
            if not self.state.frontier:
                self.state.frontier = frontier_mod.seed_frontier(self.config)

            while True:
                blocked = self.ledger.check_before_round(
                    primary_reactions=len(self.state.graph.primary_reactions),
                    candidate_reactions=len(self.state.graph.candidate_reactions),
                    candidate_components=len(self.state.graph.candidate_components),
                )
                if blocked:
                    return self._finish(STATE_STOPPED)
                if not self.state.frontier:
                    self.ledger.stop(
                        STOP_FRONTIER_EXHAUSTED, "no unexhausted frontier items remain"
                    )
                    return self._finish(STATE_STOPPED)
                self._round_index += 1
                self._run_round(self._round_index)
        except KeyboardInterrupt:
            self.ledger.cancel()
            self.ledger.stop("cancelled_by_user", "KeyboardInterrupt")
            return self._finish(STATE_CANCELLED)
        except Exception as exc:  # noqa: BLE001
            self.state.errors.append(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            self.ledger.stop(STOP_ERROR, f"{type(exc).__name__}: {exc}")
            return self._finish(STATE_FAILED)

    # -- one round ----------------------------------------------------------
    def _register_paper(self, paper: LoadedPaper) -> None:
        record = paper.record
        if record.paper_id in self.state.papers:
            return
        self.state.papers[record.paper_id] = record
        self.state.paper_texts[record.paper_id] = paper.text
        if not record.exclusion_reasons:
            self.index.add_paper(paper)

    def _run_round(self, index: int) -> None:
        self.emit("round_start", round=index, frontier=len(self.state.frontier))
        graph_before = self.state.graph.graph_hash()
        delta = GraphDelta(round_index=index, graph_hash_before=graph_before)

        round_queries: List[LiteratureQuery] = []
        round_papers: List[str] = []
        round_spans: List[EvidenceSpan] = []
        round_claims: List[ExtractedClaim] = []
        round_decisions: List[ClaimDecision] = []
        round_rejections: List[Dict[str, Any]] = []
        round_errors: List[str] = []
        round_raw: List[Dict[str, Any]] = []
        entities_before = {e.entity_id for e in self.state.graph.registry.all()}

        active = list(self.state.frontier)
        processed_this_round = 0

        def processing_capped() -> bool:
            # max_papers_processed is a hard bound for the run (the ledger records the
            # stop reason when it is reached) and for each round.
            return (bool(self.ledger.check_papers_processed())
                    or processed_this_round >= max(1, self.config.budgets.max_papers_processed))

        # Seeds and target_species_supported reactions only: never a quarantined lead's names.
        run_terms = queries_mod.rescue_terms(self.config, self.state.graph.reactions.values())

        for item in active:
            if self.ledger.check_before_call():
                break
            if self.ledger.check_papers_discovered():
                break
            if processing_capped():
                break

            # -- search: up to max_query_attempts attempts -----------------------
            # A newly included paper ends the search for this item this round.
            # The item is exhausted only when every attempt has been tried
            # without one, or at once when a query returns no result at all.
            # When no distinct formulation exists for an attempt (the topic is
            # already one of the item's terms: every "X | <topic>" bridge, the
            # topic seed) but the last query filled its window, the attempt asks
            # the same question over a wider window instead.
            # A window the discovery budget cut short is not a search that found
            # nothing: an item asked only that way is never marked exhausted
            # (pilot-002 exhausted 21 items on 1-result windows).
            found: List[Any] = []
            exhausted = False
            budget_cut = False
            budgets = self.config.budgets
            attempts = max(1, budgets.max_query_attempts)
            per_query = max(1, budgets.max_results_per_query)
            asked = returned = 0  # the previous attempt's window and result count
            for attempt in range(1, attempts + 1):
                remaining = max(0, budgets.max_papers_discovered - self.ledger.usage.papers_discovered)
                window = min(per_query, remaining)
                query = queries_mod.build_query(item, self.config, round_index=index, attempt=attempt)
                deeper = False
                if query is None and asked and returned >= asked:
                    wide = min(2 * per_query, remaining)
                    if wide > asked:
                        deeper, window = True, wide
                        query = queries_mod.build_query(item, self.config, round_index=index,
                                                        attempt=attempt, deeper=True)
                    if wide < 2 * per_query:
                        budget_cut = True  # the wider window is, or would be, cut by the budget
                if query is None:
                    exhausted = True  # no further formulation, and no wider window, exists
                    break
                if attempt > 1 and self.ledger.check_papers_discovered():
                    break
                context = queries_mod.build_context(item, self.config, attempt=1 if deeper else attempt)
                round_queries.append(query)
                self.state.queries.append(query)
                item.attempts += 1

                result = self.acquirer.acquire(
                    query,
                    context,
                    round_index=index,
                    max_papers=window,
                    run_terms=run_terms,
                )
                asked, returned = window, query.result_count
                if window < (2 * per_query if deeper else per_query):
                    budget_cut = True
                round_errors.extend(result.errors)
                self.ledger.record_papers_discovered(len(result.papers) + len(result.excluded))
                for excluded in result.excluded:
                    self.state.papers.setdefault(excluded.paper_id, excluded)
                for loaded in result.papers:
                    self._register_paper(loaded)
                    round_papers.append(loaded.record.paper_id)
                    found.append((loaded, query))
                if result.papers:
                    break
                if not query.result_count or attempt == attempts:
                    exhausted = True
                    break

            if exhausted and not found and not budget_cut:
                item.exhausted = True
                if item.frontier_item_id not in self.state.exhausted_frontier:
                    self.state.exhausted_frontier.append(item.frontier_item_id)

            # -- retrieve + extract, one paper at a time ---------------------
            retrieval_text = queries_mod.retrieval_query_text(item, self.config)
            for loaded, query in found:
                if self.ledger.check_before_call() or processing_capped():
                    break
                record = loaded.record
                if record.processed_round >= 0:
                    continue
                spans = self.index.spans_for_paper(
                    record.paper_id,
                    retrieval_text,
                    top_k=self.config.budgets.max_passages_per_paper,
                    round_index=index,
                    query_id=query.query_id,
                    frontier_item_id=item.frontier_item_id,
                )
                if not spans:
                    record.notes.append(f"round {index}: no passages retrieved for this query")
                    continue
                for span in spans:
                    self.state.evidence.setdefault(span.evidence_id, span)
                round_spans.extend(spans)
                self.ledger.record_passages(len(spans))

                # One extraction call per batch of max_passages_per_call passages
                # (evidence 032). Each batch is its own logical call; its claims
                # merge through the claim store like any other paper's.
                per_call = max(1, self.config.budgets.max_passages_per_call)
                paper_claims: List[ExtractedClaim] = []
                for start in range(0, len(spans), per_call):
                    if start and self.ledger.check_before_call():
                        record.notes.append(
                            f"round {index}: extraction stopped after {start} of {len(spans)} "
                            "passages: the call budget is spent"
                        )
                        break
                    extraction = extract_claims(
                        client=self.client,
                        spans=spans[start:start + per_call],
                        paper_id=record.paper_id,
                        paper_title=record.title,
                        target_species=self.config.target_species,
                        target_species_normalized=self.config.target_species_normalized,
                        round_index=index,
                        topic=self.config.topic,
                        paper_observed_organisms=record.observed_organisms,
                        # Settled at inclusion (Acquirer._settle_included), before any
                        # extraction: a review is read by the review rule (F-1).
                        paper_evidence_type=record.evidence_type,
                        # False after two failed EuropePMC lookups: an "unknown" grade is
                        # then read by the review rule (CS-B repair 1, A1-R1).
                        paper_evidence_type_verified=record.evidence_type_verified,
                    )
                    if extraction.outcome is not None:
                        self.state.model_calls.append(extraction.outcome.record)
                        round_raw.extend(extraction.outcome.lineage_rows())
                    if extraction.notes:
                        record.notes.extend(extraction.notes)
                    if extraction.dropped_fields:
                        round_errors.extend(extraction.dropped_fields)
                    self.ledger.record_claims(len(extraction.claims))
                    round_claims.extend(extraction.claims)
                    paper_claims.extend(extraction.claims)

                # Register EVERY batch's claims of this paper before judging any of them
                # (CS-B repair 2, R1-N1): the same-paper union (F-6) is then complete when
                # a claim is judged, so a later batch can never downgrade a reaction an
                # earlier batch of the same paper admitted. Each distinct stored claim is
                # judged once.
                stored_claims = list({c.claim_id: c for c in map(self._register, paper_claims)}.values())
                for stored in stored_claims:
                    self._judge(
                        stored,
                        index=index,
                        delta=delta,
                        decisions=round_decisions,
                        rejections=round_rejections,
                    )
                record.processed_round = index
                processed_this_round += 1
                self.ledger.record_paper_processed()

        # -- close the round ----------------------------------------------
        if not self.state.graph.primary_component_id and self.state.graph.components:
            self.state.graph.select_primary_component(
                round_index=index, at=art.utcnow(), topic=self.config.topic
            )
            for reaction in self.state.graph.primary_reactions:
                if reaction.reaction_id not in delta.added_reaction_ids:
                    delta.promoted_reaction_ids.append(reaction.reaction_id)

        entities_after = {e.entity_id for e in self.state.graph.registry.all()}
        delta.added_entity_ids = sorted(entities_after - entities_before)
        delta.graph_hash_after = self.state.graph.graph_hash()
        delta.new_conflict_ids = sorted(
            c.conflict_id for c in self.state.store.conflicts.values()
            if c.last_updated_round == index
        )
        for component in self.state.graph.components.values():
            if component.promoted_round == index:
                delta.merged_component_ids.append(component.component_id)
                for rid in component.reaction_ids:
                    if rid not in delta.promoted_reaction_ids:
                        delta.promoted_reaction_ids.append(rid)
            elif component.created_round == index:
                delta.created_component_ids.append(component.component_id)
        self.state.deltas.append(delta)

        # The per-round cap counts reactions newly admitted to the target graph
        # only. Quarantined candidates and leads are bounded by the candidate,
        # discovery, call, cost and wall-clock caps instead (ORCH-745 B1).
        self.ledger.record_round(useful=delta.useful, new_reactions=len(delta.added_reaction_ids))
        self.state.errors.extend(round_errors)

        cut: List[FrontierItem] = []
        next_frontier = frontier_mod.generate(
            graph=self.state.graph,
            store=self.state.store,
            delta=delta,
            config=self.config,
            round_index=index,
            previous=self.state.frontier,
            exhausted_ids=set(self.state.exhausted_frontier),
            cut=cut,
        )
        self.state.frontier = frontier_mod.merge_frontier(
            [i for i in self.state.frontier if not i.exhausted],
            next_frontier,
            cap=self.config.budgets.max_frontier_size,
            cut=cut,
        )
        # Every item the cap dropped, once, unless it survived anyway (an item cut
        # from the new list can be the same question carried forward). Pilot-003's
        # cap silently dropped 137 items (ORCH-745 F-3).
        kept_ids = {i.frontier_item_id for i in self.state.frontier}
        frontier_cut = [
            {"frontier_item_id": i.frontier_item_id, "kind": i.kind, "target": i.target,
             "priority": i.priority, "relevance": i.relevance}
            for i in {c.frontier_item_id: c for c in cut}.values()
            if i.frontier_item_id not in kept_ids
        ]

        record = {
            "round": index,
            "input_frontier": [i.to_dict() for i in active],
            "queries": [q.to_dict() for q in round_queries],
            "papers_considered": [
                self.state.papers[p].to_dict() for p in dict.fromkeys(round_papers)
                if p in self.state.papers
            ],
            "passages_retrieved": [s.to_dict() for s in round_spans],
            "claims_extracted": [c.to_dict() for c in round_claims],
            "decisions": [d.to_dict() for d in round_decisions],
            "rejections": round_rejections,
            "graph_delta": delta.to_dict(),
            "new_frontier": [i.to_dict() for i in self.state.frontier],
            "frontier_cut": frontier_cut,
            "metrics": {
                "queries": len(round_queries),
                "papers_discovered": len(set(round_papers)),
                "papers_processed": processed_this_round,
                "passages": len(round_spans),
                "claims": len(round_claims),
                "decisions": len(round_decisions),
                "rejections": len(round_rejections),
                "useful": delta.useful,
                "usage": self.ledger.usage.to_dict(),
                "graph_hash_before": delta.graph_hash_before,
                "graph_hash_after": delta.graph_hash_after,
                "primary_reactions": len(self.state.graph.primary_reactions),
                "candidate_reactions": len(self.state.graph.candidate_reactions),
                "candidate_components": len(self.state.graph.candidate_components),
                "frontier_cut": {
                    "count": len(frontier_cut),
                    "by_kind": dict(sorted(Counter(c["kind"] for c in frontier_cut).items())),
                },
            },
            "errors": round_errors,
            "model_raw": round_raw,
        }
        self.state.round_records.append(record)
        self.run_dir.write_round(index, record)

        checkpoint = RoundCheckpoint(
            round_index=index,
            completed=True,
            written_at=art.utcnow(),
            graph_hash=delta.graph_hash_after,
            frontier=[i.to_dict() for i in self.state.frontier],
            processed_paper_ids=[
                p for p, r in self.state.papers.items() if r.processed_round >= 0
            ],
            discovered_paper_ids=sorted(self.state.papers),
            seen_query_ids=[q.query_id for q in self.state.queries],
            unproductive_streak=self.ledger.usage.unproductive_streak,
            usage=self.ledger.usage.to_dict(),
        )
        self.run_dir.write_checkpoint(checkpoint, round_index=index)
        self.run_dir.write_state(self.state.to_dict())
        self.manifest.rounds_completed = index
        self._flush()
        self.emit("round_end", round=index, metrics=record["metrics"])

    def _admit(
        self,
        claim: ExtractedClaim,
        *,
        index: int,
        delta: GraphDelta,
        decisions: List[ClaimDecision],
        rejections: List[Dict[str, Any]],
    ) -> None:
        """Register one claim and judge it at once (a single-claim convenience; a
        paper's batches go through :meth:`_register` then :meth:`_judge`)."""
        self._judge(self._register(claim), index=index, delta=delta, decisions=decisions,
                    rejections=rejections)

    def _register(self, claim: ExtractedClaim) -> ExtractedClaim:
        stored, _new_support = self.state.store.register(claim)
        return stored

    def _judge(
        self,
        stored: ExtractedClaim,
        *,
        index: int,
        delta: GraphDelta,
        decisions: List[ClaimDecision],
        rejections: List[Dict[str, Any]],
    ) -> None:
        outcome = judge(
            stored,
            store=self.state.store,
            registry=self.state.graph.registry,
            papers=self.state.papers,
            evidence=self.state.evidence,
            round_index=index,
            decided_at=art.utcnow(),
        )
        decision = self.state.store.record_decision(outcome.decision)
        decisions.append(decision)

        # A claim that is not support for an identity admitted on other claims gets an
        # admitted decision for the identity; its own verdict is in the detail.
        verdict = str(decision.detail.get("claim_verdict") or decision.status)
        if verdict == STATUS_REJECTED or not outcome.admitted or "claim_verdict" in decision.detail:
            rejections.append(
                {
                    "claim_id": stored.claim_id,
                    "claim_identity": stored.claim_identity,
                    "status": verdict,
                    "reason": str(decision.detail.get("claim_reason") or decision.reason),
                    "paper_id": stored.paper_id,
                    "species_state": stored.species_state,
                    "stripped_fields": outcome.stripped_fields,
                }
            )

        if outcome.reaction is None:
            return

        existing = self.state.graph.by_identity(stored.claim_identity)
        stored_reaction = self.state.graph.add_reaction(
            outcome.reaction, round_index=index, at=art.utcnow()
        )
        # A non-admitted decision never relabels an admitted node (ORCH-745 A3): the
        # graph cannot un-admit a reaction, so its status must not say "candidate".
        if decision.status in ADMITTED_STATUSES or not stored_reaction.admitted:
            stored_reaction.decision_status = decision.status
            stored_reaction.decision_reason = decision.reason
            stored_reaction.evidence_level = decision.evidence_level
        for cid in outcome.conflict_ids:
            if cid not in stored_reaction.conflict_ids:
                stored_reaction.conflict_ids.append(cid)
        # A reaction admitted in an earlier round with one attribution must stop
        # displaying it once independent papers disagree. Applied to the STORED
        # node, after the merge, so a round-1 value cannot survive a round-2
        # contradiction.
        cleared = clear_disputed_attributes(
            stored_reaction, self.state.store.conflicts_for(stored.claim_identity)
        )
        if cleared:
            stored_reaction.note(
                index,
                "disputed_attribute_cleared",
                ", ".join(cleared) + " (positions retained in conflicts.json)",
                art.utcnow(),
            )

        if existing is None:
            if outcome.admitted:
                delta.added_reaction_ids.append(stored_reaction.reaction_id)
            else:
                delta.added_candidate_reaction_ids.append(stored_reaction.reaction_id)
        else:
            delta.updated_reaction_ids.append(stored_reaction.reaction_id)

    # -- finishing ----------------------------------------------------------
    def _flush(self) -> None:
        self.run_dir.snapshot(
            papers=list(self.state.papers.values()),
            queries=self.state.queries,
            frontier=self.state.frontier,
            evidence=list(self.state.evidence.values()),
            claims=list(self.state.store.claims.values()),
            decisions=self.state.store.decisions(),
            model_calls=self.state.model_calls,
            conflicts=list(self.state.store.conflicts.values()),
            costs=self.ledger.costs_to_dict(),
            primary_graph=self.state.graph.primary_to_dict(),
            candidate_subgraphs=self.state.graph.candidates_to_dict(),
        )
        self.manifest.stop_reason = self.ledger.stop_reason
        self.manifest.stop_detail = self.ledger.stop_detail
        self.run_dir.write_manifest(self.manifest)

    def verify_provenance(self) -> List[str]:
        """Every admitted reaction must have an intact provenance chain."""
        problems: List[str] = []
        for reaction in self.state.graph.reactions.values():
            if reaction.decision_status not in ADMITTED_STATUSES:
                continue
            broken = provenance_chain_complete(
                reaction=reaction,
                claims=self.state.store.claims,
                evidence=self.state.evidence,
                papers=self.state.papers,
            )
            problems.extend(f"{reaction.reaction_id}: {p}" for p in broken)
        return problems

    def _write_views(self) -> None:
        conflicts: Dict[str, ConflictRecord] = dict(self.state.store.conflicts)
        reaction_view = viz.reaction_network(
            self.state.graph, papers=self.state.papers, conflicts=conflicts
        )
        provenance_view = viz.provenance_network(
            graph=self.state.graph,
            papers=self.state.papers,
            claims=self.state.store.claims,
            evidence=self.state.evidence,
            conflicts=conflicts,
        )
        self.run_dir.write_graph_view("reaction_network.json", reaction_view)
        self.run_dir.write_graph_view("provenance_network.json", provenance_view)
        self.run_dir.write_graph_view(
            "reaction_network.dot", viz.reaction_network_dot(reaction_view)
        )
        self.run_dir.write_graph_view(
            "provenance_network.dot", viz.provenance_network_dot(provenance_view)
        )
        self.run_dir.write_graph_view(
            "evidence_table.json",
            viz.evidence_table(
                graph=self.state.graph,
                papers=self.state.papers,
                claims=self.state.store.claims,
                evidence=self.state.evidence,
                decisions=self.state.store.decisions(),
            ),
        )

    def _finish(self, completion_state: str) -> MiningOutcome:
        if not self.ledger.stop_reason and completion_state == STATE_COMPLETED:
            self.ledger.stop(STOP_COMPLETED, "all configured rounds completed")
        if self.ledger.cancelled and completion_state not in {STATE_FAILED, STATE_PREFLIGHT_FAILED}:
            completion_state = STATE_CANCELLED

        problems = self.verify_provenance()
        if problems:
            self.state.errors.extend(problems)
            self.ledger.warn(
                f"{len(problems)} admitted reaction(s) have an incomplete provenance chain"
            )

        self.manifest.completion_state = completion_state
        self.manifest.finished_at = art.utcnow()
        self.manifest.stop_reason = self.ledger.stop_reason
        self.manifest.stop_detail = self.ledger.stop_detail
        if completion_state in INCOMPLETE_STATES:
            self.manifest.notes.append(
                f"this run did NOT complete normally (state={completion_state}); every "
                "completed round's artifacts are present and nothing is presented as final"
            )

        self._flush()
        self._write_views()

        summary = build_summary(
            config=self.config,
            manifest=self.manifest,
            state=self.state,
            ledger=self.ledger,
            preflight=self.preflight,
            index_stats=getattr(self._index, "stats", lambda: {})() if self._index else {},
            provenance_problems=problems,
        )
        self.run_dir.write_summary(summary)
        self.run_dir.write_report(build_report(summary, state=self.state, config=self.config))
        self.run_dir.write_manifest(self.manifest)
        self.emit("finished", state=completion_state, stop_reason=self.ledger.stop_reason)

        return MiningOutcome(
            run_dir=self.run_dir,
            manifest=self.manifest,
            state=self.state,
            ledger=self.ledger,
            preflight=self.preflight,
            summary=summary,
        )

    # -- resume -------------------------------------------------------------
    @classmethod
    def resume(
        cls,
        run_path: Any,
        *,
        source: Optional[PaperSource] = None,
        chat_fn: Optional[Callable[..., Any]] = None,
        progress: Optional[ProgressFn] = None,
        enabled_override: Optional[bool] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "MiningController":
        """Rebuild a controller from a run directory's checkpoint and state.

        The graph, claims, decisions, papers, evidence and frontier all come back;
        the token, call and cost budgets are cumulative across the resume so a
        resumed run cannot spend the budget twice. Wall clock restarts, because
        yesterday's elapsed time is not a resource this process is consuming.
        """
        run_dir = RunDirectory.open(art.Path(str(run_path)))
        config_payload = run_dir.read_config() or {}
        state_payload = run_dir.read_state() or {}
        checkpoint = run_dir.read_checkpoint() or {}

        from t2pw.mining.config import MiningBudgets, MiningModelConfig, StartingInput

        config = MiningConfig(
            target_species=str(config_payload.get("target_species") or "unknown"),
            target_species_normalized=str(config_payload.get("target_species_normalized") or ""),
            topic=str(config_payload.get("topic") or ""),
            starting_inputs=[
                StartingInput(**{k: v for k, v in dict(i).items()
                                 if k in StartingInput.__dataclass_fields__})  # type: ignore[attr-defined]
                for i in config_payload.get("starting_inputs") or []
            ],
            budgets=MiningBudgets(**{
                k: v for k, v in (config_payload.get("budgets") or {}).items()
                if k in MiningBudgets.__dataclass_fields__  # type: ignore[attr-defined]
            }),
            model=MiningModelConfig(**{
                k: v for k, v in (config_payload.get("model") or {}).items()
                if k in MiningModelConfig.__dataclass_fields__  # type: ignore[attr-defined]
            }),
            acquisition_sources=list(config_payload.get("acquisition_sources") or ["europepmc"]),
            retrieve_top_k=int(config_payload.get("retrieve_top_k") or 8),
            vector_backend=str(config_payload.get("vector_backend") or "memory"),
            offline=bool(config_payload.get("offline")),
            screen_papers=bool(config_payload.get("screen_papers", True)),
            runs_dir=str(config_payload.get("runs_dir") or "runs_mining"),
            run_id=run_dir.run_id,
            enabled_via=str(config_payload.get("enabled_via") or "environment"),
        )
        for key, value in dict(overrides or {}).items():
            if hasattr(config.budgets, key):
                setattr(config.budgets, key, value)

        controller = cls(
            config,
            source=source,
            chat_fn=chat_fn,
            progress=progress,
            enabled_override=enabled_override,
            run_dir=run_dir,
        )
        controller.state = MiningState.from_dict(state_payload)
        controller._setup_done = True
        controller.acquirer.seen_paper_ids.update(controller.state.papers)
        for record in controller.state.papers.values():
            if not record.exclusion_reasons:
                controller.acquirer.remember(record)  # its PMCID / PMID / DOI too (IT2-N3)
        controller.ledger.restore(dict(checkpoint.get("usage") or {}))
        controller._round_index = int(checkpoint.get("round_index") or 0)
        manifest = run_dir.read_manifest()
        if manifest is not None:
            controller.manifest = manifest
            controller.manifest.resumed_from_round = controller._round_index
            controller.manifest.completion_state = STATE_RUNNING
            controller.manifest.notes.append(
                f"resumed from round {controller._round_index} at {art.utcnow()}"
            )
        # Re-index the papers whose text survived, so retrieval works again.
        for pid, record in controller.state.papers.items():
            text = controller.state.paper_texts.get(pid, "")
            if text and not record.exclusion_reasons:
                controller.index.add_paper(LoadedPaper(record=record, text=text))
        return controller

    def continue_run(self, **kwargs: Any) -> MiningOutcome:
        """Resume the loop from the restored state (no second preflight probe)."""
        kwargs.setdefault("probe_model", False)
        kwargs.setdefault("probe_embeddings", False)
        return self.run(**kwargs)


__all__ = ["MiningController", "MiningOutcome", "MiningState", "ProgressFn"]
