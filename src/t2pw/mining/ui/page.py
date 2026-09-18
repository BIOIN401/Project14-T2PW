"""The Streamlit Pathway Mining page.

Completely separate from the PWML pipeline page: its own session-state
namespace (``mining_*``), its own run directories, its own controller. Nothing
here touches the production pipeline's state, and with the feature disabled the
page renders one explanatory panel and stops.

It calls exactly the same :class:`~t2pw.mining.controller.MiningController` and
artifact layer the CLI calls. There is no second implementation.

Sections, in order: run setup, service preflight, starting inputs, species and
seeds, budgets and stop conditions, live progress, reaction network, paper /
evidence network, claims and evidence table, candidate subgraphs, conflicts and
rejected claims, and logs / cost / exports.

No secret is ever displayed. The page shows whether a key is present, never its
value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from t2pw.mining.config import (
    DEFAULT_MINING_MODEL,
    DEFAULT_RUNS_DIR,
    MiningConfig,
    StartingInput,
)
from t2pw.mining.flags import FEATURE_ENV_VAR, mining_enabled
from t2pw.mining.ui.network import render_network_html
from t2pw.paths import PROJECT_ROOT

_SS = "mining_"  # session-state namespace, isolated from the pipeline page


def _state(st: Any, key: str, default: Any) -> Any:
    full = _SS + key
    if full not in st.session_state:
        st.session_state[full] = default
    return st.session_state[full]


def _set(st: Any, key: str, value: Any) -> None:
    st.session_state[_SS + key] = value


def _runs_dir(st: Any) -> Path:
    raw = str(_state(st, "runs_dir", str(PROJECT_ROOT / DEFAULT_RUNS_DIR)))
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


# ---------------------------------------------------------------------------
# Disabled state
# ---------------------------------------------------------------------------
def _render_disabled(st: Any) -> None:
    st.info(
        "**Pathway mining is an experimental feature and is switched off.**\n\n"
        f"Enable it in one of three ways:\n\n"
        f"1. set `{FEATURE_ENV_VAR}=true` in your environment or `.env`;\n"
        "2. run the CLI with `--enable-pathway-mining`;\n"
        "3. tick the box below, which enables it for this browser session only.\n\n"
        "Enabling it does **not** start a run, does not change any existing pipeline "
        "behaviour, and produces no PWML."
    )
    enabled = st.checkbox(
        "Enable pathway mining for this session",
        value=bool(_state(st, "session_enabled", False)),
        key=_SS + "session_enabled_box",
        help="A per-session override. The environment flag stays as it is.",
    )
    _set(st, "session_enabled", enabled)
    if enabled:
        st.rerun()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _collect_inputs(st: Any) -> Dict[str, Any]:
    st.subheader("1. Starting inputs")
    st.caption(
        "Composable: a topic, a species, seed entities, explicit papers and an existing "
        "pathway graph may all be supplied together."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        species = st.text_input(
            "Target species (required)",
            value=str(_state(st, "species", "")),
            key=_SS + "species_in",
            help=(
                "Every run declares one target species. Evidence from another species is "
                "retained as a labelled lead and never silently converted into a "
                "target-species claim."
            ),
        )
        topic = st.text_input(
            "Topic",
            value=str(_state(st, "topic", "")),
            key=_SS + "topic_in",
            placeholder="e.g. jasmonate biosynthesis",
        )
    with col_b:
        seed_metabolites = st.text_area(
            "Seed metabolites (one per line)",
            value=str(_state(st, "seed_metabolites", "")),
            key=_SS + "seed_met_in",
            height=80,
        )
        seed_enzymes = st.text_area(
            "Seed enzymes / proteins (one per line)",
            value=str(_state(st, "seed_enzymes", "")),
            key=_SS + "seed_enz_in",
            height=80,
        )

    col_c, col_d = st.columns(2)
    with col_c:
        seed_reactions = st.text_area(
            "Seed reactions (one per line, free text)",
            value=str(_state(st, "seed_reactions", "")),
            key=_SS + "seed_rxn_in",
            height=80,
        )
    with col_d:
        papers = st.text_area(
            "Papers (one per line: PMID:123, PMC123, a DOI, or a URL)",
            value=str(_state(st, "papers", "")),
            key=_SS + "papers_in",
            height=80,
        )

    uploads = st.file_uploader(
        "Upload papers (PDF or text)",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "xml"],
        key=_SS + "uploads_in",
        help=(
            "Each upload is hashed; re-uploading the same bytes merges into the existing "
            "record rather than creating a second citation for one paper."
        ),
    )
    starting_graph = st.text_input(
        "Existing pathway / research graph JSON (optional, read-only)",
        value=str(_state(st, "starting_graph", "")),
        key=_SS + "start_graph_in",
        help=(
            "Used as a read-only starting network. No PWML is produced from it and no "
            "PathWhiz importer gate runs."
        ),
    )

    for key, value in (
        ("species", species), ("topic", topic), ("seed_metabolites", seed_metabolites),
        ("seed_enzymes", seed_enzymes), ("seed_reactions", seed_reactions),
        ("papers", papers), ("starting_graph", starting_graph),
    ):
        _set(st, key, value)

    return {
        "species": species,
        "topic": topic,
        "seed_metabolites": seed_metabolites,
        "seed_enzymes": seed_enzymes,
        "seed_reactions": seed_reactions,
        "papers": papers,
        "starting_graph": starting_graph,
        "uploads": uploads or [],
    }


def _collect_budgets(st: Any) -> Dict[str, Any]:
    st.subheader("2. Budgets and stop conditions")
    st.caption(
        "The run stops at the FIRST limit reached and records which one. Every value here "
        "is written into the run manifest."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        max_rounds = st.number_input("Max rounds", 1, 50, int(_state(st, "max_rounds", 8)))
        max_discovered = st.number_input(
            "Max papers discovered", 1, 2000, int(_state(st, "max_discovered", 150))
        )
        max_processed = st.number_input(
            "Max papers fully processed", 1, 500, int(_state(st, "max_processed", 40))
        )
        max_passages = st.number_input(
            "Max passages per paper", 1, 100, int(_state(st, "max_passages", 20))
        )
    with col_b:
        max_primary = st.number_input(
            "Max primary-graph reactions", 1, 5000, int(_state(st, "max_primary", 200))
        )
        max_candidates = st.number_input(
            "Max candidate reactions", 1, 5000, int(_state(st, "max_candidates", 300))
        )
        max_frontier = st.number_input(
            "Max frontier size", 1, 500, int(_state(st, "max_frontier", 60))
        )
        unproductive = st.number_input(
            "Stop after N unproductive rounds", 1, 10, int(_state(st, "unproductive", 2))
        )
    with col_c:
        max_calls = st.number_input("Max LLM calls", 1, 5000, int(_state(st, "max_calls", 200)))
        max_cost = st.number_input(
            "Max estimated cost (USD)", 0.01, 500.0, float(_state(st, "max_cost", 5.0)), step=0.5
        )
        max_seconds = st.number_input(
            "Max wall clock (seconds)", 30, 86400, int(_state(st, "max_seconds", 3600))
        )
        max_out_tokens = st.number_input(
            "Max output tokens", 1000, 10_000_000, int(_state(st, "max_out_tokens", 400_000)),
            step=10_000,
        )

    for key, value in (
        ("max_rounds", max_rounds), ("max_discovered", max_discovered),
        ("max_processed", max_processed), ("max_passages", max_passages),
        ("max_primary", max_primary), ("max_candidates", max_candidates),
        ("max_frontier", max_frontier), ("unproductive", unproductive),
        ("max_calls", max_calls), ("max_cost", max_cost),
        ("max_seconds", max_seconds), ("max_out_tokens", max_out_tokens),
    ):
        _set(st, key, value)

    return {
        "max_rounds": int(max_rounds),
        "max_papers_discovered": int(max_discovered),
        "max_papers_processed": int(max_processed),
        "max_passages_per_paper": int(max_passages),
        "max_primary_reactions": int(max_primary),
        "max_candidate_reactions": int(max_candidates),
        "max_frontier_size": int(max_frontier),
        "stop_after_unproductive_rounds": int(unproductive),
        "max_llm_calls": int(max_calls),
        "max_cost_usd": float(max_cost),
        "max_wall_clock_seconds": float(max_seconds),
        "max_output_tokens": int(max_out_tokens),
    }


def _collect_engine(st: Any) -> Dict[str, Any]:
    st.subheader("3. Model, retrieval and acquisition")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        model = st.text_input(
            "Model slug", value=str(_state(st, "model", DEFAULT_MINING_MODEL)),
            key=_SS + "model_in",
            help="The model REQUESTED. What actually answered is logged per call.",
        )
        fallback = st.text_input(
            "Explicit fallback model (optional)",
            value=str(_state(st, "fallback", "")),
            key=_SS + "fallback_in",
            help="Leave empty for no fallback: a failure then stays loud instead of "
                 "being answered quietly by a different model.",
        )
    with col_b:
        sources = st.multiselect(
            "Literature sources",
            ["europepmc", "ncbi", "crossref", "semanticscholar", "biorxiv"],
            default=list(_state(st, "sources", ["europepmc", "ncbi"])),
            key=_SS + "sources_in",
            help="Paper DISCOVERY only. No biochemical database is consulted for "
                 "admission in version 1.",
        )
        top_k = st.number_input("Passages retrieved per query", 1, 50,
                                int(_state(st, "top_k", 8)))
    with col_c:
        backend = st.selectbox(
            "Vector backend", ["memory", "chroma"],
            index=["memory", "chroma"].index(str(_state(st, "backend", "memory"))),
            key=_SS + "backend_in",
            help="`memory` keeps the run's index out of the production RAG index.",
        )
        screen = st.checkbox("Screen papers with the eligibility gate",
                             value=bool(_state(st, "screen", True)), key=_SS + "screen_in")
        offline = st.checkbox("Offline (never contact a literature service)",
                              value=bool(_state(st, "offline", False)), key=_SS + "offline_in")
        runs_dir = st.text_input("Runs directory",
                                 value=str(_state(st, "runs_dir", str(PROJECT_ROOT / DEFAULT_RUNS_DIR))),
                                 key=_SS + "runs_dir_in")

    for key, value in (
        ("model", model), ("fallback", fallback), ("sources", sources), ("top_k", top_k),
        ("backend", backend), ("screen", screen), ("offline", offline), ("runs_dir", runs_dir),
    ):
        _set(st, key, value)

    return {
        "model": {"model": model, "fallback_model": fallback},
        "acquisition_sources": sources or ["europepmc"],
        "retrieve_top_k": int(top_k),
        "vector_backend": backend,
        "screen_papers": bool(screen),
        "offline": bool(offline),
        "runs_dir": runs_dir,
    }


def build_config(st: Any, inputs: Dict[str, Any], budgets: Dict[str, Any],
                 engine: Dict[str, Any]) -> Optional[MiningConfig]:
    """Assemble a :class:`MiningConfig` from the form, or ``None`` with an error."""
    from t2pw.mining.acquisition import PaperReferenceError, make_paper_starting_input

    species = str(inputs["species"] or "").strip()
    if not species:
        st.error("A target species is required: the species guardrail cannot classify "
                 "evidence without one.")
        return None

    starting: List[StartingInput] = []
    for kind, blob in (
        ("seed_metabolite", inputs["seed_metabolites"]),
        ("seed_enzyme", inputs["seed_enzymes"]),
        ("seed_reaction", inputs["seed_reactions"]),
    ):
        for line in str(blob or "").splitlines():
            value = line.strip()
            if value:
                starting.append(StartingInput(kind=kind, raw=value, value=value))
    for line in str(inputs["papers"] or "").splitlines():
        ref = line.strip()
        if not ref:
            continue
        try:
            starting.append(make_paper_starting_input(ref))
        except PaperReferenceError as exc:
            st.error(str(exc))
            return None
    if str(inputs["topic"] or "").strip():
        starting.append(
            StartingInput(kind="topic", raw=inputs["topic"], value=inputs["topic"].strip())
        )
    if str(inputs["starting_graph"] or "").strip():
        path = Path(str(inputs["starting_graph"]).strip())
        if not path.exists():
            st.error(f"Starting graph not found: {path}")
            return None
        starting.append(
            StartingInput(kind="pathway_graph", raw=str(path), value=path.stem,
                          source_path=str(path.resolve()))
        )

    overrides: Dict[str, Any] = dict(engine)
    overrides["budgets"] = budgets
    overrides["enabled_via"] = "streamlit"
    return MiningConfig.build(
        target_species=species,
        topic=str(inputs["topic"] or "").strip(),
        starting_inputs=starting,
        overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def _run(st: Any, config: MiningConfig, uploads: List[Any], *, probe: bool,
         override: Optional[bool]) -> None:
    from t2pw.mining.acquisition import record_from_upload
    from t2pw.mining.controller import MiningController

    loaded = []
    for upload in uploads:
        data = upload.read()
        loaded.append(record_from_upload(data=data, file_name=upload.name))

    status = st.empty()
    progress = st.progress(0.0)
    log_area = st.empty()
    lines: List[str] = []
    max_rounds = max(1, config.budgets.max_rounds)

    def on_progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "preflight":
            lines.append(f"preflight: {'PASSED' if payload.get('ok') else 'FAILED'}")
        elif event == "round_start":
            status.info(f"Round {payload.get('round')} of at most {max_rounds} - "
                        f"frontier {payload.get('frontier')} items")
        elif event == "round_end":
            metrics = payload.get("metrics") or {}
            round_no = int(payload.get("round") or 0)
            progress.progress(min(1.0, round_no / max_rounds))
            lines.append(
                f"round {round_no}: {metrics.get('papers_processed', 0)} papers, "
                f"{metrics.get('claims', 0)} claims, "
                f"primary {metrics.get('primary_reactions', 0)}, "
                f"candidates {metrics.get('candidate_reactions', 0)}, "
                f"useful={'yes' if metrics.get('useful') else 'no'}"
            )
        elif event == "finished":
            status.success(f"Run {payload.get('state')} ({payload.get('stop_reason')})")
            progress.progress(1.0)
        log_area.code("\n".join(lines[-25:]) or "starting...", language="text")

    controller = MiningController(
        config, progress=on_progress, enabled_override=override
    )
    _set(st, "active_run", str(controller.run_dir.root))
    with st.spinner("Mining. Heavy work is sequential; this blocks until a limit is reached."):
        outcome = controller.run(
            probe_model=probe, probe_embeddings=probe, uploaded=loaded
        )
    _set(st, "selected_run", str(outcome.run_dir.root))
    if outcome.completed_normally:
        st.success(f"Completed. Artifacts in `{outcome.run_dir.root}`")
    else:
        st.warning(
            f"Run state `{outcome.manifest.completion_state}` "
            f"(stop reason `{outcome.stop_reason}`). Every completed round's artifacts "
            f"are present in `{outcome.run_dir.root}`; nothing is presented as final."
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _pick_run(st: Any) -> Optional[Path]:
    from t2pw.mining.artifacts import list_runs

    runs_dir = _runs_dir(st)
    rows = list_runs(runs_dir)
    if not rows:
        st.info(f"No runs under `{runs_dir}` yet.")
        return None
    labels = [
        f"{r['run_id']}  -  {r['completion_state']}  -  {r['species']} / {r['topic'] or 'no topic'}"
        for r in rows
    ]
    selected = str(_state(st, "selected_run", rows[0]["path"]))
    default = next((i for i, r in enumerate(rows) if r["path"] == selected), 0)
    choice = st.selectbox("Run", labels, index=default, key=_SS + "run_pick")
    path = rows[labels.index(choice)]["path"]
    _set(st, "selected_run", path)
    return Path(path)


def _render_overview(st: Any, bundle: Dict[str, Any], run_path: Path) -> None:
    summary = bundle["summary"]
    manifest = bundle["manifest"]
    if not summary:
        st.warning("This run has no summary.json yet.")
        return
    if not summary.get("completed_normally"):
        st.warning(
            f"**This run did not complete normally.** State `{summary.get('completion_state')}`, "
            f"stop reason `{summary.get('stop_reason')}`. "
            f"{summary.get('stop_detail', '')}"
        )
    st.caption(
        "This mode produces a research graph. No PWML was generated and external "
        "biochemical database validation is deferred (every entity reads `not_evaluated`)."
    )
    graph = summary.get("graph") or {}
    papers = summary.get("papers") or {}
    costs = summary.get("costs") or {}
    cols = st.columns(5)
    cols[0].metric("Primary reactions", graph.get("primary_reactions", 0))
    cols[1].metric("Candidate reactions", graph.get("candidate_reactions", 0))
    cols[2].metric("Papers included", papers.get("included", 0))
    cols[3].metric("Unresolved conflicts",
                   (summary.get("conflicts") or {}).get("unresolved", 0))
    cols[4].metric("Est. cost (USD)", f"{costs.get('estimated_cost_usd', 0.0):.4f}")

    st.markdown(
        f"- **Rounds completed:** {summary.get('rounds_completed', 0)}  \n"
        f"- **Stop reason:** `{summary.get('stop_reason', '')}`  \n"
        f"- **Target species:** {summary.get('target_species', '')}  \n"
        f"- **Git commit:** `{(manifest.get('git') or {}).get('commit', '')[:12]}` "
        f"({'dirty' if (manifest.get('git') or {}).get('dirty') else 'clean'})  \n"
        f"- **Reactions supported by more than one paper:** "
        f"{graph.get('multi_paper_reactions', 0)}"
    )
    if graph.get("primary_selection_rationale"):
        st.caption("Primary component: " + graph["primary_selection_rationale"])

    report = (run_path / "report.md")
    if report.exists():
        with st.expander("Full report.md", expanded=False):
            st.markdown(report.read_text(encoding="utf-8"))


def _render_reaction_view(st: Any, bundle: Dict[str, Any]) -> None:
    from t2pw.mining.viz import filter_reaction_network

    view = bundle["reaction_network"]
    if not view.get("nodes"):
        st.info("This run produced no reaction nodes.")
        return
    rounds = view.get("rounds") or [0]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        max_round = st.slider(
            "Show the network as it stood at the end of round",
            min_value=int(min(rounds)), max_value=int(max(rounds)),
            value=int(max(rounds)), key=_SS + "rx_round",
            help="The growth slider: drag it back to watch the network build up.",
        )
        status = st.selectbox("Graph status", ["(all)", "primary", "candidate"],
                              key=_SS + "rx_status")
    with col_b:
        reactions = [n for n in view["nodes"] if n.get("type") == "reaction"]
        paper_ids = sorted({
            p.get("paper_id", "")
            for n in reactions for p in n.get("supporting_papers", [])
            if p.get("paper_id")
        })
        paper = st.selectbox("Supporting paper", ["(all)", *paper_ids], key=_SS + "rx_paper")
        species_states = sorted({n.get("species_state", "") for n in reactions if n.get("species_state")})
        species_state = st.selectbox("Species state", ["(all)", *species_states],
                                     key=_SS + "rx_species")
    with col_c:
        enzymes = sorted({e for n in reactions for e in n.get("enzymes", []) if e})
        enzyme = st.selectbox("Enzyme", ["(all)", *enzymes], key=_SS + "rx_enzyme")
        multi_only = st.checkbox("Only reactions supported by >1 paper", key=_SS + "rx_multi")

    filtered = filter_reaction_network(
        view,
        max_round=int(max_round),
        graph_status=None if status == "(all)" else status,
        paper_id="" if paper == "(all)" else paper,
        enzyme="" if enzyme == "(all)" else enzyme,
        species_state="" if species_state == "(all)" else species_state,
        only_multi_paper=bool(multi_only),
    )
    shown = sum(1 for n in filtered["nodes"] if n.get("type") == "reaction")
    st.caption(
        f"{shown} reaction node(s) shown of {len(reactions)}. Enzymes are annotations "
        "(dashed edges) and never connect two reactions."
    )
    st.components.v1.html(
        render_network_html(filtered, height=640, uid="rx"), height=1060, scrolling=True
    )


def _render_provenance_view(st: Any, bundle: Dict[str, Any]) -> None:
    view = bundle["provenance_network"]
    if not view.get("nodes"):
        st.info("This run produced no provenance nodes.")
        return
    st.warning(
        "Edges in this view are **bibliographic**, not chemical. Two reactions sharing a "
        "paper are not connected. Chemistry lives in the reaction-network view."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        include_excluded = st.checkbox(
            "Include papers that were retrieved but EXCLUDED",
            value=False, key=_SS + "pv_excluded",
            help="Excluded papers are kept behind this filter rather than mixed into "
                 "accepted evidence.",
        )
        layers = st.multiselect(
            "Layers to show",
            ["paper", "evidence", "claim", "reaction"],
            default=["paper", "claim", "reaction"],
            key=_SS + "pv_layers",
            help="Evidence spans are numerous; hide them for a readable overview and "
                 "use the evidence table for exact auditing.",
        )
    with col_b:
        bridges_only = st.checkbox("Only papers involved in a bridge", key=_SS + "pv_bridge")
        candidates_only = st.checkbox("Only papers supporting a candidate subgraph",
                                      key=_SS + "pv_cand")

    keep = set(layers or ["paper", "claim", "reaction"])
    nodes = []
    for node in view["nodes"]:
        if node.get("type") not in keep:
            continue
        if node.get("type") == "paper":
            if node.get("excluded") and not include_excluded:
                continue
            if bridges_only and not node.get("is_bridge_paper"):
                continue
            if candidates_only and not node.get("supports_candidate_subgraph"):
                continue
        nodes.append(node)
    ids = {n["id"] for n in nodes}
    edges = [e for e in view["edges"] if e.get("source") in ids and e.get("target") in ids]
    st.caption(
        f"{len(nodes)} node(s), {len(edges)} bibliographic edge(s). "
        f"{view.get('counts', {}).get('papers_excluded', 0)} paper(s) were retrieved and excluded."
    )
    st.components.v1.html(
        render_network_html({**view, "nodes": nodes, "edges": edges}, height=640, uid="pv"),
        height=1060,
        scrolling=True,
    )


def _render_table(st: Any, bundle: Dict[str, Any]) -> None:
    rows = bundle["evidence_table"]
    if not rows:
        st.info("No evidence rows.")
        return
    st.caption(
        "One row per (reaction, claim, evidence span). A diagram is not enough for exact "
        "auditing; this is the authoritative view."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        status = st.selectbox(
            "Decision status",
            ["(all)", *sorted({r.get("decision_status", "") for r in rows if r.get("decision_status")})],
            key=_SS + "tb_status",
        )
    with col_b:
        graph_status = st.selectbox("Graph status", ["(all)", "primary", "candidate"],
                                    key=_SS + "tb_graph")
    with col_c:
        query = st.text_input("Text search (reaction, paper, quote)", key=_SS + "tb_q")

    filtered = [
        r for r in rows
        if (status == "(all)" or r.get("decision_status") == status)
        and (graph_status == "(all)" or r.get("graph_status") == graph_status)
        and (
            not query
            or query.lower()
            in f"{r.get('reaction','')} {r.get('paper_title','')} {r.get('quote','')}".lower()
        )
    ]
    st.write(f"{len(filtered)} of {len(rows)} rows")
    try:
        import pandas as pd

        st.dataframe(pd.DataFrame(filtered), use_container_width=True, height=460)
    except Exception:  # noqa: BLE001 - pandas is optional for this view
        st.json(filtered[:200])


def _render_candidates(st: Any, bundle: Dict[str, Any]) -> None:
    payload = bundle["candidate_subgraphs"]
    components = payload.get("components") or []
    if not components:
        st.success("No quarantined candidate subgraphs: everything relevant is either in the "
                   "primary graph or was rejected with a recorded reason.")
        return
    st.caption(
        "Paper-supported findings that are relevant but not admitted to the primary graph. "
        "They keep full provenance, stay eligible for bridge-seeking queries, and are never "
        "displayed as connected."
    )
    for component in components:
        reason = component.get("quarantine_reason") or "not chemically connected to the primary graph"
        header = (
            f"`{component.get('component_id')}`  -  {len(component.get('reactions', []))} "
            f"reaction(s)  -  {reason}"
        )
        with st.expander(header, expanded=False):
            st.markdown(
                f"- Discovered in round **{component.get('created_round')}**\n"
                f"- Promoted: **{'yes, round ' + str(component.get('promoted_round')) if component.get('promoted') else 'no'}**\n"
                f"- Relevance score: {component.get('relevance_score')}\n"
                f"- Rationale: {component.get('rationale')}"
            )
            if component.get("promotion_history"):
                st.write("Promotion history:")
                st.json(component["promotion_history"])
            for reaction in component.get("reactions", []):
                st.markdown(
                    f"**{reaction.get('label')}** - `{reaction.get('decision_status')}` - "
                    f"species `{reaction.get('species_state') or 'n/a'}` - "
                    f"papers {', '.join(reaction.get('paper_ids', [])) or 'none'}"
                )
                st.caption(reaction.get("decision_reason", ""))


def _render_conflicts(st: Any, bundle: Dict[str, Any], run_path: Path) -> None:
    from t2pw.mining.artifacts import read_jsonl

    conflicts = (bundle["conflicts"] or {}).get("conflicts") or []
    st.subheader("Conflicts")
    if not conflicts:
        st.info("No conflicting evidence was recorded.")
    else:
        st.caption("Both positions are retained. Nothing was resolved by choosing a side.")
        for conflict in conflicts:
            with st.expander(
                f"`{conflict.get('attribute')}` on `{conflict.get('claim_identity')}` "
                f"({len(conflict.get('paper_ids', []))} paper(s))"
            ):
                st.write(conflict.get("resolution_note", ""))
                for position in conflict.get("positions", []):
                    st.markdown(
                        f"- **{position.get('value')}** - papers "
                        f"{', '.join(position.get('paper_ids', []))}, claims "
                        f"{', '.join(position.get('claim_ids', []))}"
                    )

    st.subheader("Rejected, incomplete and quarantined claims")
    decisions = read_jsonl(run_path / "decisions.jsonl")
    held = [
        d for d in decisions
        if d.get("status") in {"rejected", "incomplete", "hypothesis_only", "candidate"}
    ]
    if not held:
        st.info("Nothing was rejected or held back.")
        return
    st.caption(
        "Nothing here is permanently closed: a claim held as incomplete or rejected is "
        "reconsidered when a later round supplies stronger evidence."
    )
    try:
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "round": d.get("round_index"),
                        "status": d.get("status"),
                        "species_state": d.get("species_state"),
                        "identity": d.get("claim_identity"),
                        "reason": d.get("reason"),
                        "papers": ", ".join(d.get("supporting_paper_ids", [])),
                    }
                    for d in held
                ]
            ),
            use_container_width=True,
            height=380,
        )
    except Exception:  # noqa: BLE001
        st.json(held[:100])


def _render_logs(st: Any, bundle: Dict[str, Any], run_path: Path) -> None:
    from t2pw.mining.artifacts import read_jsonl

    summary = bundle["summary"]
    costs = bundle["costs"]
    st.subheader("Cost and token usage")
    st.caption(costs.get("note", ""))
    st.json(costs)

    st.subheader("Model calls")
    calls = read_jsonl(run_path / "model_calls.jsonl")
    failed = [c for c in calls if c.get("status") != "ok"]
    st.write(f"{len(calls)} call(s), {len(failed)} that produced no admissible claim.")
    if failed:
        st.caption(
            "An empty, truncated or unparseable completion never adds a reaction. Each one "
            "is recorded here instead of silently reading as 'the literature says nothing'."
        )
        st.json(failed[:50])

    st.subheader("Round metrics")
    st.json(summary.get("additions_by_round") or [])

    if summary.get("warnings") or summary.get("errors"):
        st.subheader("Warnings and errors")
        for warning in (summary.get("warnings") or [])[:40]:
            st.warning(warning)
        for error in (summary.get("errors") or [])[:20]:
            st.error(str(error).splitlines()[0][:400])

    st.subheader("Exports")
    for name, label in (
        ("report.md", "report.md"),
        ("summary.json", "summary.json"),
        ("primary_graph.json", "primary_graph.json"),
        ("candidate_subgraphs.json", "candidate_subgraphs.json"),
        ("claims.jsonl", "claims.jsonl"),
        ("evidence_spans.jsonl", "evidence_spans.jsonl"),
        ("decisions.jsonl", "decisions.jsonl"),
        ("model_calls.jsonl", "model_calls.jsonl"),
        ("graph_views/reaction_network.json", "reaction_network.json"),
        ("graph_views/provenance_network.json", "provenance_network.json"),
        ("graph_views/reaction_network.dot", "reaction_network.dot (Graphviz)"),
        ("graph_views/provenance_network.dot", "provenance_network.dot (Graphviz)"),
        ("graph_views/evidence_table.json", "evidence_table.json"),
    ):
        path = run_path / name
        if path.exists():
            st.download_button(
                label,
                data=path.read_bytes(),
                file_name=path.name,
                key=_SS + "dl_" + name.replace("/", "_"),
            )
    st.caption(f"All artifacts live under `{run_path}`.")


def _render_results(st: Any) -> None:
    from t2pw.mining.artifacts import RunDirectory
    from t2pw.mining.regenerate import load_views

    run_path = _pick_run(st)
    if run_path is None:
        return
    try:
        bundle = load_views(RunDirectory.open(run_path))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load this run: {type(exc).__name__}: {exc}")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Reaction network",
            "Paper / evidence network",
            "Claims & evidence",
            "Candidate subgraphs",
            "Conflicts & rejections",
            "Logs, cost & exports",
        ]
    )
    with tabs[0]:
        _render_overview(st, bundle, run_path)
    with tabs[1]:
        _render_reaction_view(st, bundle)
    with tabs[2]:
        _render_provenance_view(st, bundle)
    with tabs[3]:
        _render_table(st, bundle)
    with tabs[4]:
        _render_candidates(st, bundle)
    with tabs[5]:
        _render_conflicts(st, bundle, run_path)
    with tabs[6]:
        _render_logs(st, bundle, run_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render_page(st: Any = None) -> None:
    """Render the whole Pathway Mining page. Safe to call when disabled."""
    if st is None:
        import streamlit as st  # type: ignore

    st.title("Pathway Mining (experimental)")
    st.caption(
        "Literature-driven research-graph mining. This mode does **not** produce PWML, does "
        "**not** invoke the PWML exporter, and does **not** require PathWhiz identifiers."
    )

    override = True if bool(_state(st, "session_enabled", False)) else None
    if not mining_enabled(override):
        _render_disabled(st)
        return

    st.success(
        f"Pathway mining is enabled. Enabling it does not start a run. "
        f"(`{FEATURE_ENV_VAR}` "
        f"{'is set' if mining_enabled(None) else 'is not set; enabled for this session only'}.)"
    )

    setup_tab, results_tab = st.tabs(["Set up and run", "Explore results"])

    with setup_tab:
        inputs = _collect_inputs(st)
        budgets = _collect_budgets(st)
        engine = _collect_engine(st)

        st.subheader("4. Service preflight")
        st.caption(
            "Run this before an expensive run. An embeddings or model outage must not be "
            "reported as 'the literature had nothing'."
        )
        probe = st.checkbox("Probe live services (one tiny model call, one embedding)",
                            value=True, key=_SS + "probe")
        col_a, col_b = st.columns(2)
        if col_a.button("Run preflight", key=_SS + "btn_preflight"):
            from t2pw.mining.preflight import run_preflight

            config = build_config(st, inputs, budgets, engine)
            if config is not None:
                report = run_preflight(
                    config, probe_model=probe, probe_embeddings=probe,
                    enabled_override=override,
                )
                for check in report.checks:
                    line = f"**{check.name}** - {check.detail}"
                    if check.status == "fail":
                        st.error(line)
                    elif check.status == "warn":
                        st.warning(line)
                    else:
                        st.write(("OK " if check.status == "ok" else "SKIP ") + line)
                (st.success if report.ok else st.error)(
                    "Preflight passed." if report.ok else "Preflight FAILED - a run would not start."
                )

        st.subheader("5. Start")
        if col_b.button("Start mining run", type="primary", key=_SS + "btn_run"):
            config = build_config(st, inputs, budgets, engine)
            if config is not None:
                _run(st, config, list(inputs["uploads"]), probe=probe, override=override)

        with st.expander("Resume an interrupted run", expanded=False):
            st.caption(
                "A cancelled or interrupted run keeps every completed round. Resuming "
                "restores the graph, claims, decisions, papers and frontier; the token, "
                "call and cost budgets stay cumulative."
            )
            resume_path = st.text_input("Run directory", key=_SS + "resume_path")
            if st.button("Resume", key=_SS + "btn_resume") and resume_path.strip():
                from t2pw.mining.controller import MiningController

                controller = MiningController.resume(
                    resume_path.strip(), enabled_override=override
                )
                with st.spinner("Resuming..."):
                    outcome = controller.continue_run()
                _set(st, "selected_run", str(outcome.run_dir.root))
                st.success(
                    f"Resumed run finished with state `{outcome.manifest.completion_state}` "
                    f"({outcome.stop_reason})."
                )

    with results_tab:
        _render_results(st)


__all__ = ["render_page", "build_config"]
