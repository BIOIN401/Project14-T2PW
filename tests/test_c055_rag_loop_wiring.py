"""C-055 — the RAG round loop, wired into the app (D-008, PRODUCT_CONTRACT §10/§9).

NEW ACCEPTANCE TESTS. ``t2pw.rag.controller`` had **zero production callers** at this
card's base SHA `32f3a57`, so the behaviour these assert — a bounded, deadline-aware,
checkpointing loop whose every round re-enters normalization, mapping, gates, persistence
and classification — has never existed in this app and replaces no prior observable
behaviour. That is G9 clause 4 (new capability), and it needs no fabricated base failure.
The proof that would be tempting and is FORBIDDEN — "at base nothing imports
``controller.py``" — is symbol absence, which G9 rejects by name; no test here asserts an
ImportError or the absence of a symbol.

ONE test in this file is a PRESERVATION claim, and it is labelled as such:
``test_apptest_rag_off_takes_the_untouched_single_paper_path`` passes at the base SHA
**and** at the tip. It is not dressed as a base failure.

What is proven, and why each is here:

* ``test_apptest_*`` drive the REAL app script through ``streamlit.testing.v1.AppTest``.
  ``MASTER_PLAN.md:165`` makes AppTest-driven focused tests mandatory for this hotspot,
  and at base no test anywhere combined a real AppTest run with the RAG path.
* The loop is the CONTROLLER's. ``test_the_app_contains_no_hand_rolled_round_loop`` walks
  the app's AST: a ``while`` in ``maybe_run_rag`` or in ``run_rag_rounds`` is a reject
  under the separation invariant (``docs/rag/03_separation_invariant.md``,
  ``streamlit_app.py:161``), and the app must call ``run_rag_loop``.
* Round N re-runs the chain against round N-1's RECOMPUTED reports. At base only
  ``qa_graph`` was ever populated and it was frozen to the Stage-1 payload, so a second
  round would re-detect the identical gap set and converge for the wrong reason.
* A stage failure surfaces as ``RoundAborted`` with its checkpoint and its PARTIAL
  re-entry map, never as a swallowed exception and never as a termination reason.
* TRAP 11: the loop never writes ``data/id_mapping_cache.json``.

Offline and deterministic: no network, no LLM, no database. Every stage function that
would reach one is stubbed at its SOURCE module before ``AppTest`` execs the app (the app
binds them with ``from ... import``), which is the pattern
``tests/test_streamlit_quarantine_boundary.py`` established.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_PATH = SRC / "t2pw" / "app" / "streamlit_app.py"

#: The app's import graph is large; the default 3 s is not enough on a cold run.
APP_TIMEOUT = 180.0

#: A Streamlit harness fault, not an app fault — see test_streamlit_quarantine_boundary.
_HARNESS_FAULT = "FragmentThreadState not initialized"

GAP_ID = "reaction_gap::c055"
SOURCE_ID = "PMC_C055"

_LINEAGE = [
    {
        "stage": "rag_retrieval",
        "origin": "rag_literature",
        "support": "direct",
        "paper_explicit": "not_explicit",
        "reason": "retrieved to fill a detected typed gap",
        "review_required": False,
        "sources": [{"source_id": SOURCE_ID, "source_type": "paper", "locator": "results"}],
    }
]
_PROVENANCE = {
    "source_id": SOURCE_ID,
    "source_title": "A C-055 fixture paper",
    "source_type": "paper",
    "chunk_id": "c" * 32,
    "gap_id": GAP_ID,
    "gap_ids": [GAP_ID],
}


def _seed_payload() -> Dict[str, Any]:
    """A small, structurally complete Stage-1 payload."""

    return {
        "metadata": {"pathway_name": "Menaquinone biosynthesis", "organism": "Escherichia coli"},
        "entities": {
            "compounds": [{"name": "menaquinone"}, {"name": "demethylmenaquinone"}],
            "proteins": [{"name": "MenG"}],
            "protein_complexes": [],
            "nucleic_acids": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": "menaquinone demethylation",
                    "inputs": ["demethylmenaquinone"],
                    "outputs": ["menaquinone"],
                    "enzymes": ["MenG"],
                    "evidence": "the seed paper states this reaction",
                }
            ]
        },
    }


def _rag_reaction(name: str, product: str) -> Dict[str, Any]:
    """One synthesized reaction carrying everything §10 requires of an addition."""

    return {
        "name": name,
        "inputs": ["menaquinone"],
        "outputs": [product],
        "enzymes": ["MenA"],
        "evidence": f"the retrieved paper states {name}",
        "evidence_span": f"the retrieved paper states {name}",
        "scope_membership": "core",
        "rag_provenance": dict(_PROVENANCE),
        "source_refs": [SOURCE_ID],
        "source_papers": [{"source_id": SOURCE_ID, "source_type": "paper"}],
        "provenance_lineage": deepcopy(_LINEAGE),
    }


def _rag_payload(name: str = "menaquinone prenylation", product: str = "prenylmenaquinone"):
    return {
        "metadata": {"pathway_name": "Menaquinone biosynthesis"},
        "entities": {
            "compounds": [{"name": product, "rag_provenance": dict(_PROVENANCE),
                           "provenance_lineage": deepcopy(_LINEAGE)}],
            "proteins": [{"name": "MenA", "rag_provenance": dict(_PROVENANCE),
                          "provenance_lineage": deepcopy(_LINEAGE)}],
        },
        "processes": {"reactions": [_rag_reaction(name, product)]},
    }


def _claim(name: str, product: str) -> Dict[str, Any]:
    """An admission-report row, in the shape ``RagReactionCandidate.to_dict`` emits."""

    return {
        "gap_id": GAP_ID,
        "gap_ids": [GAP_ID],
        "name": name,
        "inputs": ["menaquinone"],
        "outputs": [product],
        "enzymes": ["MenA"],
        "reversible": False,
    }


def _rag_result(
    name: str = "menaquinone prenylation", product: str = "prenylmenaquinone", *, synthesized=True
):
    """A ``RagOrchestrationResult``-shaped stand-in for one chain run."""

    payload = _rag_payload(name, product)
    synthesis = types.SimpleNamespace(
        payload=payload,
        admission={
            "accepted": [_claim(name, product)],
            "rejected": [],
            "counts": {"accepted": 1, "rejected": 0, "considered": 1},
            "reason_counts": {},
        },
        stitched=[],
        unresolved_gaps=[],
    )
    return types.SimpleNamespace(
        decision=types.SimpleNamespace(run=True, signals=("user_flag",)),
        evidence_context="C-055 fixture evidence",
        payload=payload,
        synthesized=bool(synthesized),
        selection=None,
        candidates=[],
        ingest_report=None,
        synthesis=synthesis,
        provenance={},
        gaps=[types.SimpleNamespace(gap_id=GAP_ID, kind="dangling_reaction", label=product)],
        bundles=[object()],
        diagnostics={},
        error="",
    )


def _merge_site(final_payload, rag_envelope, pathway_context):
    """The RAG merge statement, for the module-level tests.

    The production statement lives in the app's script body so that C-060a's AST lift
    (``tests/test_c060a_requested_pathway_wiring.py:68-104``) can still reach it; a
    nested ``def`` inside ``if submit:`` cannot be imported, so these tests supply the
    same call. The REAL statement is exercised end-to-end by the AppTest below and is
    separately protected by C-060a's own lift, so nothing here is the only cover for it.
    """

    from t2pw.pipeline.entity_admission import pathway_context_from_stage_zero
    from t2pw.pipeline.pipeline import merge_additions

    return merge_additions(
        final_payload,
        rag_envelope,
        pathway_context=pathway_context_from_stage_zero(pathway_context),
    )


# ---------------------------------------------------------------------------
# The app module, imported WITHOUT streamlit for the non-AppTest tests.
# ---------------------------------------------------------------------------
@pytest.fixture()
def app_module():
    """``t2pw.app.streamlit_app`` with ``streamlit`` stubbed, as the rag tests do.

    The module executes a Streamlit script at import, so ``streamlit`` is replaced by a
    MagicMock exactly the way ``tests/test_rag_triage_orchestration.py`` does it. Only
    the module-level helpers are exercised here; the script body is exercised through
    the real ``AppTest`` below.
    """

    from unittest.mock import MagicMock

    saved = {n: sys.modules[n] for n in list(sys.modules)
             if n == "streamlit" or n.startswith("streamlit.")}
    if not isinstance(sys.modules.get("streamlit"), types.ModuleType) or hasattr(
        sys.modules.get("streamlit"), "__path__"
    ):
        for name in list(saved):
            sys.modules.pop(name, None)
    stub = MagicMock(name="streamlit")
    stub.columns.side_effect = lambda spec=1, **_: [
        MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))
    ]
    stub.tabs.side_effect = lambda labels, **_: [MagicMock() for _ in labels]
    stub.form_submit_button.return_value = False
    stub.checkbox.return_value = False
    stub.session_state = MagicMock()
    stub.session_state.get.return_value = None
    stub.session_state.__contains__ = lambda _s, _k: False
    sys.modules["streamlit"] = stub
    sys.modules.pop("t2pw.app.streamlit_app", None)
    before = set(sys.modules)
    import t2pw.app.streamlit_app as app  # noqa: PLC0415

    try:
        yield app
    finally:
        # Undo EVERY t2pw module this fixture caused to be imported under the stub.
        # Restoring ``sys.modules["streamlit"]`` alone is not enough: a t2pw module
        # that ran ``import streamlit as st`` while the stub was installed keeps the
        # MagicMock forever, and a later real ``AppTest`` run then execs an app whose
        # helper modules draw widgets on a mock -- the script never completes and the
        # run sits on its timeout. Only modules absent before the fixture are removed,
        # so nothing another test module already held is disturbed.
        for name in sorted(set(sys.modules) - before, reverse=True):
            if name == "t2pw" or name.startswith("t2pw."):
                del sys.modules[name]
        sys.modules.pop("t2pw.app.streamlit_app", None)
        for name in [n for n in list(sys.modules)
                     if n == "streamlit" or n.startswith("streamlit.")]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture()
def offline_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, List[Any]]:
    """Count every real D-008 stage call, and keep mapping off the network.

    ``normalize_process_payload``, ``run_strict_post_normalization_gates``,
    ``build_and_save_draft_graph`` and ``quarantine_and_close`` stay REAL — they are
    pure and offline. Only ``map_payload`` is replaced, because it reaches UniProt,
    ChEBI and MySQL. Each wrapper records its call, so an adapter that quietly skipped
    its stage would fail these tests rather than pass them vacuously.
    """

    import t2pw.mapping.map_ids as map_ids
    import t2pw.pipeline.pipeline as pipeline_module
    import t2pw.pipeline.process_normalizer as normalizer
    import t2pw.pipeline.strict_quarantine as quarantine

    calls: Dict[str, List[Any]] = {n: [] for n in
                                   ("normalization", "mapping", "gates", "persistence",
                                    "classification")}

    real_normalize = normalizer.normalize_process_payload
    real_gates = normalizer.run_strict_post_normalization_gates
    real_persist = pipeline_module.build_and_save_draft_graph
    real_quarantine = quarantine.quarantine_and_close

    def _normalize(payload, **kwargs):
        calls["normalization"].append(payload)
        return real_normalize(payload, **kwargs)

    def _gates(payload, **kwargs):
        calls["gates"].append(payload)
        return real_gates(payload, **kwargs)

    def _persist(payload, **kwargs):
        # Record WHICH path the caller asked for — that is the assertion — but never
        # let a test perform the app's authoritative write. ``output_path=None`` means
        # ``tmp/draft_graph.json`` + ``tmp/qa_report.json`` + ``tmp/reaction_summary.txt``,
        # three of the seven PROTECTED scratch files. An AppTest that drives the real
        # ``if submit:`` block reaches that write, so it is redirected here instead.
        asked = kwargs.get("output_path")
        calls["persistence"].append(asked)
        if asked is None:
            redirected = tmp_path / "authoritative"
            redirected.mkdir(parents=True, exist_ok=True)
            kwargs["output_path"] = redirected / "draft_graph.json"
        return real_persist(payload, **kwargs)

    def _quarantine(payload, **kwargs):
        calls["classification"].append(payload)
        return real_quarantine(payload, **kwargs)

    def _map(payload, **kwargs):
        calls["mapping"].append(kwargs.get("cache_path"))
        return {"payload": deepcopy(payload), "report": {"summary": {"mapped": 0},
                                                         "unmapped": []}}

    monkeypatch.setattr(normalizer, "normalize_process_payload", _normalize)
    monkeypatch.setattr(normalizer, "run_strict_post_normalization_gates", _gates)
    monkeypatch.setattr(pipeline_module, "build_and_save_draft_graph", _persist)
    monkeypatch.setattr(quarantine, "quarantine_and_close", _quarantine)
    monkeypatch.setattr(map_ids, "map_payload", _map)
    return calls


def _rebind(app, calls_source: str = "") -> None:
    """Re-bind the app module's own names to the patched source functions.

    ``streamlit_app`` binds every stage with ``from ... import``, so a monkeypatch on
    the source module reaches the app only when the app is exec'd afterwards (which is
    what ``AppTest`` does). For the module-level tests the module is already imported,
    so the same names are re-pointed here — at the app, not at the owned module.
    """

    import t2pw.mapping.map_ids as map_ids
    import t2pw.pipeline.pipeline as pipeline_module
    import t2pw.pipeline.process_normalizer as normalizer
    import t2pw.pipeline.strict_quarantine as quarantine

    app.normalize_process_payload = normalizer.normalize_process_payload
    app.run_strict_post_normalization_gates = normalizer.run_strict_post_normalization_gates
    app.build_and_save_draft_graph = pipeline_module.build_and_save_draft_graph
    app.quarantine_and_close = quarantine.quarantine_and_close
    app.map_payload = map_ids.map_payload


# ---------------------------------------------------------------------------
# D-008 — every round re-enters all five stages, and the loop is the controller's.
# ---------------------------------------------------------------------------
def test_one_round_re_enters_all_five_stages_and_merges(app_module, offline_stages, tmp_path):
    """NEW acceptance. A single round runs every D-008 stage and lands the merge."""

    _rebind(app_module)
    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    assert record["aborted"] is None
    assert len(record["rounds"]) == 1
    assert record["rounds"][0]["reentry"] == {
        "normalization": "true", "mapping": "true", "gates": "true",
        "persistence": "true", "classification": "true",
    }
    # Every stage genuinely ran: these counters come from the REAL functions, so an
    # adapter that recorded re-entry without performing it would fail here.
    # ``gates`` is 2 for one round and that is correct: ``normalize_process_payload``
    # runs the strict gate itself (``process_normalizer.py:5501``) and the gates adapter
    # then runs it explicitly on the payload normalization produced — the same
    # normalize-then-gate-again shape ``run_post_pipeline_sbml_artifacts`` already uses.
    assert [len(offline_stages[name]) for name in
            ("normalization", "mapping", "gates", "persistence", "classification")] == [
        1, 1, 2, 1, 1]
    # The merge landed, and the canonical graph advanced.
    names = [r["name"] for r in record["payload"]["processes"]["reactions"]]
    assert "menaquinone prenylation" in names
    assert record["rounds"][0]["accepted"] is True
    assert record["rounds"][0]["graph_delta"] >= 1


def test_the_checkpoint_is_written_before_the_round(app_module, offline_stages, tmp_path):
    """NEW acceptance (§9). The last good graph, the claim history and the budget are on
    disk BEFORE the round's retrieval, so a deadline or a crash still leaves them."""

    _rebind(app_module)
    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    written = json.loads((tmp_path / "rag_rounds" / "round_00" / "checkpoint.json")
                         .read_text(encoding="utf-8"))
    assert written["loop"]["round_index"] == 0
    assert written["budget"]["stage"] == "rag_loop"
    assert written["budget"]["remaining_seconds"] > 0
    graph = json.loads((tmp_path / "rag_rounds" / "round_00" / "checkpoint_graph.json")
                       .read_text(encoding="utf-8"))
    # The checkpoint holds the graph as of BEFORE the round: one reaction, not two.
    assert [r["name"] for r in graph["processes"]["reactions"]] == ["menaquinone demethylation"]
    assert record["checkpoints"][0]["reactions"] == 1


def test_round_two_re_runs_the_chain_on_recomputed_reports(app_module, offline_stages, tmp_path,
                                                           monkeypatch):
    """NEW acceptance, and the reason the five stages are re-entered at all.

    ``retrieve.detect_gaps`` reads three report channels (``rag/retrieve.py:699-701``).
    At base the app supplied ONE, ``qa_graph``, computed once from the Stage-1 payload,
    so a second round would re-detect the identical gap set and converge for the wrong
    reason. Here round 2's chain must receive a ``qa_graph`` recomputed from round 1's
    merged payload PLUS the ``gate`` and ``mapping`` channels, which no session key
    carried at base.
    """

    _rebind(app_module)
    seen: List[Dict[str, Any]] = []
    seeds: List[Any] = []

    def _chain(**kwargs):
        seen.append(dict(kwargs.get("reports") or {}))
        seeds.append(kwargs.get("seed_payload"))
        return _rag_result("menaquinone methylation", "methylmenaquinone")

    monkeypatch.setattr(app_module, "maybe_run_rag", _chain)
    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=2,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    assert len(record["rounds"]) == 2
    assert [r["reentry"] for r in record["rounds"]] == [
        {name: "true" for name in ("normalization", "mapping", "gates",
                                   "persistence", "classification")}
    ] * 2
    # Round 0 reused the pre-computed chain result; round 1 re-entered the chain once.
    assert len(seen) == 1
    assert set(seen[0]) == {"qa_graph", "gate", "mapping"}
    # ... against round 0's MERGED graph, not the frozen Stage-1 seed.
    assert any(r["name"] == "menaquinone prenylation"
               for r in seeds[0]["processes"]["reactions"])


def test_the_app_contains_no_hand_rolled_round_loop(app_module):
    """NEW acceptance. The separation invariant (``streamlit_app.py:161``,
    ``docs/rag/03_separation_invariant.md``) makes the orchestrator wiring only: the loop
    must be ``run_rag_loop``. A ``while`` in either RAG function is a reject.

    Non-vacuity: the same walk also asserts the call it is protecting is present, so a
    version that removed the loop entirely would fail here rather than pass.
    """

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef)}
    for name in ("maybe_run_rag", "run_rag_rounds", "build_rag_round_stages"):
        assert name in functions, f"{name} is missing from the app"
        assert not [n for n in ast.walk(functions[name]) if isinstance(n, (ast.While,))], (
            f"{name} hand-rolls a loop; the loop must be rag.controller.run_rag_loop")
    called = {node.func.id for node in ast.walk(functions["run_rag_rounds"])
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "run_rag_loop" in called
    # And the four inputs the invariant allows, and only those, reach the controller.
    call = next(node for node in ast.walk(functions["run_rag_rounds"])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "run_rag_loop")
    assert {kw.arg for kw in call.keywords} == {
        "stages", "round_runner", "deadline", "max_rounds",
        "next_round_reserve_seconds", "checkpoint"}


# ---------------------------------------------------------------------------
# §9 / §11 — an operational failure is never converted into a scientific verdict.
# ---------------------------------------------------------------------------
def test_a_failed_stage_aborts_the_round_and_keeps_its_checkpoint(app_module, offline_stages,
                                                                  tmp_path, monkeypatch):
    """NEW acceptance (trap 3). ``RoundAborted`` carries the failing stage, the PARTIAL
    re-entry map and the checkpoint. Un-run stages report ``not_evaluated``, which §11
    forbids collapsing into ``false``, and an abort carries NO termination reason."""

    _rebind(app_module)
    import t2pw.pipeline.pipeline as pipeline_module

    def _explode(*_a, **_k):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(pipeline_module, "build_and_save_draft_graph", _explode)
    app_module.build_and_save_draft_graph = _explode

    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    assert record["aborted"]["stage"] == "persistence"
    assert record["aborted"]["reentry"] == {
        "normalization": "true", "mapping": "true", "gates": "true",
        "persistence": "false", "classification": "not_evaluated",
    }
    # No termination reason: an infrastructure failure is not one of D-005's six.
    assert record["reason"] is None
    # The last good graph survived, unmerged.
    assert [r["name"] for r in record["payload"]["processes"]["reactions"]] == [
        "menaquinone demethylation"]
    assert any(kind == "error" for kind, _ in record["messages"])


def test_a_budget_that_cannot_afford_a_round_runs_none(app_module, offline_stages, tmp_path):
    """NEW acceptance (§9). The loop is deadline-aware by REFUSING TO START a round whose
    finalization reserve does not fit — it does not discover the deadline by overrunning
    it, and it does not call a single stage."""

    _rebind(app_module)
    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=4,
        deadline_seconds=1.0,  # far below the 120 s finalization reserve
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    assert record["rounds"] == []
    assert record["checkpoints"] == []
    assert record["reason"] == "budget_exhausted"
    assert all(not calls for calls in offline_stages.values())
    assert record["payload"]["processes"]["reactions"][0]["name"] == "menaquinone demethylation"


def test_a_refused_delta_does_not_advance_the_graph_and_says_which_rule(app_module,
                                                                        offline_stages,
                                                                        tmp_path):
    """NEW acceptance. §10: a retrieved reaction enters only when it fills a specific
    DETECTED typed gap. A round whose additions name no detected gap is refused by
    ``validate_graph_delta``, the canonical graph does not advance, and the rule that
    refused it is reported rather than the RAG output silently disappearing."""

    _rebind(app_module)
    result = _rag_result()
    result.gaps = []  # nothing was detected, so nothing may be attributed

    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=result,
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    assert record["rounds"][0]["accepted"] is False
    assert "gap_not_detected" in record["rounds"][0]["rules"]
    assert [r["name"] for r in record["payload"]["processes"]["reactions"]] == [
        "menaquinone demethylation"]
    assert any("validator refused" in str(body) for _, body in record["messages"])
    # The stages still ran: D-008 binds every round the loop runs, refused or not.
    assert record["rounds"][0]["reentry"]["classification"] == "true"


# ---------------------------------------------------------------------------
# TRAP 11 — the loop must not multiply the unlocked 4.2 MB cache overwrite.
# ---------------------------------------------------------------------------
def test_per_round_mapping_never_writes_the_authoritative_cache(app_module, offline_stages,
                                                                tmp_path):
    """NEW acceptance. ``MappingCache.save`` (``map_ids.py:780-783``) is a non-atomic,
    unlocked, whole-file overwrite, and a T-100 leg died on exactly that write on
    2026-08-18 after producing real biology. Re-entering mapping once per round would
    turn one such overwrite of the shared file into N; the rounds map against a seeded
    copy under ``tmp/`` instead, so the authoritative file takes ZERO writes from here."""

    _rebind(app_module)
    authoritative = app_module.PROJECT_ROOT / "data" / "id_mapping_cache.json"
    before = authoritative.stat().st_mtime_ns if authoritative.exists() else None

    app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    used = offline_stages["mapping"][0]
    assert Path(used) == tmp_path / "rag_rounds" / "id_mapping_cache.json"
    assert Path(used) != authoritative
    after = authoritative.stat().st_mtime_ns if authoritative.exists() else None
    assert after == before, "the loop wrote the authoritative id mapping cache"


def test_per_round_persistence_never_writes_the_protected_scratch_files(app_module,
                                                                       offline_stages,
                                                                       tmp_path):
    """NEW acceptance. ``build_and_save_draft_graph`` writes ``draft_graph.json``,
    ``qa_report.json`` and ``reaction_summary.txt`` next to its ``output_path``; three of
    the seven protected scratch files live at ``tmp/``. Every round writes into its own
    directory, so the authoritative write stays the one the existing flow makes."""

    _rebind(app_module)
    app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    written = Path(offline_stages["persistence"][0])
    assert written == tmp_path / "rag_rounds" / "round_00" / "draft_graph.json"
    assert written.parent != app_module.PROJECT_ROOT / "tmp"


def test_the_loop_record_touches_no_refinement_state_key(app_module, offline_stages, tmp_path):
    """NEW acceptance (trap 7). ``refinement_round`` and ``refinement_checkpoints`` belong
    to the post-mapping review UI's undo stack. A RAG round counter written there would
    corrupt it, so the loop's record shares no key with that subsystem."""

    _rebind(app_module)
    record = app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    session = app_module.rag_loop_session_record(record)
    assert "payload" not in session and "messages" not in session
    assert not (set(session) & set(app_module.REFINEMENT_STATE_DEFAULTS))
    assert set(session) >= {"rounds", "checkpoints", "reason", "budget", "max_rounds"}


def test_the_round_leaves_the_quarantine_boundary_session_record_untouched(app_module,
                                                                            offline_stages,
                                                                            tmp_path):
    """NEW acceptance. The classification stage re-enters the app's ONE deciding
    function, ``run_quarantine_boundary`` — ``tests/test_streamlit_quarantine_boundary.py``
    forbids a third ``quarantine_and_close`` caller in the app — so the round has to
    leave that function's session record exactly as it found it. A stored boundary
    decision describes the payload version it judged, and a round's intermediate graph
    is not that payload.

    Non-vacuity: the pre-existing values are DIFFERENT sentinels, and one key is absent
    on purpose, so a round that wrote the boundary's real report through would fail here
    rather than pass by coincidence.
    """

    _rebind(app_module)
    # A REAL dict, not the MagicMock the app fixture installs: the assertions below
    # are about what is stored and what is absent, and a MagicMock answers both
    # questions the same way whatever happened.
    session = {"quarantine_report": {"sentinel": "before"}, "quarantine_ok": "sentinel-ok"}
    app_module.st.session_state = session

    app_module.run_rag_rounds(
        _seed_payload(),
        first_result=_rag_result(),
        merge_site=_merge_site,
        pathway_context={"pathway_name": "Menaquinone biosynthesis"},
        user_flag=True,
        strict_db=False,
        max_rounds=1,
        scratch_root=tmp_path,
        locked_manifest_path=tmp_path / "no_manifest.json",
        quarantine_output_path=tmp_path / "quarantined.json",
    )

    assert session["quarantine_report"] == {"sentinel": "before"}
    assert session["quarantine_ok"] == "sentinel-ok"
    assert "quarantine_coverage" not in session
    assert "quarantine_artifacts" not in session
    assert "quarantine_decision_controls" not in session
    # And the round's own artifacts went to the round's directory, never ``outputs/``.
    assert (tmp_path / "rag_rounds" / "round_00" / "quarantine").is_dir()


def test_the_default_round_budget_is_one_round(app_module, monkeypatch):
    """NEW acceptance. Wiring the loop must not change what an unchanged deployment does,
    so the round ceiling defaults to 1 and only an explicit env var raises it."""

    monkeypatch.delenv(app_module.RAG_LOOP_MAX_ROUNDS_ENV, raising=False)
    assert app_module.rag_loop_max_rounds() == 1
    monkeypatch.setenv(app_module.RAG_LOOP_MAX_ROUNDS_ENV, "not a number")
    assert app_module.rag_loop_max_rounds() == 1
    monkeypatch.setenv(app_module.RAG_LOOP_MAX_ROUNDS_ENV, "0")
    assert app_module.rag_loop_max_rounds() == 1
    monkeypatch.setenv(app_module.RAG_LOOP_MAX_ROUNDS_ENV, "3")
    assert app_module.rag_loop_max_rounds() == 3


# ---------------------------------------------------------------------------
# The REAL app, through AppTest — MASTER_PLAN.md:165's mandatory instrument.
# ---------------------------------------------------------------------------
def _is_real_streamlit(module: object) -> bool:
    return isinstance(module, types.ModuleType) and hasattr(module, "__path__")


@pytest.fixture()
def real_streamlit():
    """Guarantee the genuine ``streamlit`` package, whatever collection order did.

    Identical in shape to ``tests/test_streamlit_quarantine_boundary.py``'s fixture and
    restores whatever it found, so the modules that install a MagicMock keep it.
    """

    def _keys():
        return [n for n in list(sys.modules) if n == "streamlit" or n.startswith("streamlit.")]

    saved = {name: sys.modules[name] for name in _keys()}
    stubbed = not _is_real_streamlit(sys.modules.get("streamlit"))
    if stubbed:
        for name in saved:
            del sys.modules[name]
    import streamlit  # noqa: F401,PLC0415
    import streamlit.testing.v1  # noqa: F401,PLC0415

    assert _is_real_streamlit(sys.modules["streamlit"]), "the real streamlit did not load"
    try:
        yield
    finally:
        if stubbed:
            for name in _keys():
                del sys.modules[name]
            sys.modules.update(saved)


@pytest.fixture()
def offline_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Stage 0, Stage 1 and Stage 2, stubbed at their SOURCE modules.

    The app binds each with ``from ... import`` at exec time, which ``AppTest`` performs
    after this fixture runs, so patching the source module is enough.
    """

    import t2pw.pipeline.pipeline as pipeline_module
    import t2pw.pipeline.preprocessor as preprocessor
    import t2pw.pipeline.stage_one_boundary as boundary

    monkeypatch.setattr(
        preprocessor, "preprocess",
        lambda *_a, **_k: {"pathway_name": "Menaquinone biosynthesis",
                           "likely_organism": "Escherichia coli",
                           "scope_clarity_score": 0.9},
    )
    monkeypatch.setattr(
        pipeline_module, "run_stage_one_with_chunking",
        lambda *_a, **_k: (deepcopy(_seed_payload()), []),
    )
    monkeypatch.setattr(
        pipeline_module, "run_stage_two_with_feedback_loop",
        lambda *_a, **_k: ({}, [], []),
    )

    def _settle(payload, **_kwargs):
        return types.SimpleNamespace(
            ok=True, payload=payload, failure=None, reconstruction=None,
            incomplete_reason="", to_summary=lambda: {"ok": True},
        )

    monkeypatch.setattr(boundary, "settle_stage_one", _settle)


@pytest.fixture()
def offline_rag_chain(monkeypatch: pytest.MonkeyPatch):
    """The whole R1–R5 chain, stubbed at ``t2pw.rag.*``. No network, no LLM."""

    from t2pw.rag import acquire, identity, ingest, provenance, retrieve, select
    from t2pw.rag import synonyms, synthesize, triage

    monkeypatch.setattr(triage, "should_run_rag",
                        lambda *_a, **_k: types.SimpleNamespace(run=True, signals=("user_flag",)))
    monkeypatch.setattr(acquire, "search_candidates", lambda *_a, **_k: [])
    monkeypatch.setattr(acquire, "fetch_full_text", lambda *_a, **_k: "")
    monkeypatch.setattr(
        select, "select",
        lambda *_a, **_k: types.SimpleNamespace(
            selected=[], kept_entries=lambda: [], dropped_entries=lambda: []),
    )
    monkeypatch.setattr(ingest, "Embedder", lambda **_k: types.SimpleNamespace(embed=lambda t: t))
    monkeypatch.setattr(ingest, "get_vector_store", lambda **_k: object())
    monkeypatch.setattr(ingest, "seed_candidate", lambda *_a, **_k: object())
    monkeypatch.setattr(ingest, "ingest",
                        lambda *_a, **_k: types.SimpleNamespace(chunks=1))
    monkeypatch.setattr(
        retrieve, "detect_gaps",
        lambda *_a, **_k: [types.SimpleNamespace(gap_id=GAP_ID, kind="dangling_reaction",
                                                 label="prenylmenaquinone")],
    )
    monkeypatch.setattr(retrieve, "retrieve_evidence", lambda *_a, **_k: object())
    monkeypatch.setattr(retrieve, "format_retrieval_context",
                        lambda *_a, **_k: "C-055 fixture evidence")
    monkeypatch.setattr(synonyms, "build_offline_synonym_resolver", lambda *_a, **_k: None)
    monkeypatch.setattr(identity, "fail_closed_verifier", lambda *_a, **_k: None)
    monkeypatch.setattr(provenance, "validate_provenance", lambda *_a, **_k: {})

    fixture = _rag_result()
    monkeypatch.setattr(synthesize, "synthesize_with_report",
                        lambda *_a, **_k: fixture.synthesis)


def _app_exceptions(at) -> List[str]:
    return [str(e.value) for e in at.exception if _HARNESS_FAULT not in str(e.value)]


def _run_submit(monkeypatch: pytest.MonkeyPatch, *, rag_enabled: bool):
    from streamlit.testing.v1 import AppTest  # noqa: PLC0415

    monkeypatch.setenv("RAG_ENABLED", "1" if rag_enabled else "0")
    monkeypatch.setenv("RAG_ELIGIBILITY_ENABLED", "0")
    monkeypatch.setenv("RAG_LOOP_MAX_ROUNDS", "1")
    at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    at.run()
    at.session_state["pwml_strict_db"] = False
    at.text_area(key="pathway_text_0").set_value(
        "MenG demethylates demethylmenaquinone to menaquinone in Escherichia coli."
    )
    at.button[0].click().run()
    return at


@pytest.mark.usefixtures("real_streamlit", "offline_pipeline", "offline_stages",
                         "offline_rag_chain")
def test_apptest_rag_on_runs_a_round_that_re_enters_all_five_stages(monkeypatch, offline_stages):
    """NEW acceptance, and this card's central test obligation.

    ``MASTER_PLAN.md:165`` makes AppTest-driven focused tests mandatory for the
    ``streamlit_app.py`` script body, and at base NO test anywhere combined a real
    ``streamlit.testing.v1.AppTest`` run with the RAG path. This drives the real app
    script: paste text, click Run pipeline, and read what the real ``if submit:`` block
    left in the real ``st.session_state``.
    """

    at = _run_submit(monkeypatch, rag_enabled=True)

    assert _app_exceptions(at) == []
    loop = at.session_state["rag_loop"]
    assert loop["max_rounds"] == 1
    assert len(loop["rounds"]) == 1
    assert loop["rounds"][0]["reentry"] == {
        "normalization": "true", "mapping": "true", "gates": "true",
        "persistence": "true", "classification": "true",
    }
    assert loop["checkpoints"][0]["round_index"] == 0
    assert loop["budget"]["checkpoints"] == 1
    # The five stages were re-entered by the REAL functions, inside the real script run,
    # and persistence records BOTH the round-scoped write and the authoritative one.
    assert all(offline_stages[name] for name in offline_stages)
    assert offline_stages["persistence"][0] is not None
    assert offline_stages["persistence"][-1] is None
    # And the merged pathway is what the rest of the pipeline received.
    names = [r["name"] for r in at.session_state["final_payload"]["processes"]["reactions"]]
    assert "menaquinone demethylation" in names and "menaquinone prenylation" in names


@pytest.mark.usefixtures("real_streamlit", "offline_pipeline", "offline_stages")
def test_apptest_rag_off_takes_the_untouched_single_paper_path(monkeypatch, offline_stages):
    """PRESERVATION claim, labelled as such: this passes at the base SHA **and** at the
    tip, and is deliberately NOT presented as a base failure.

    With ``RAG_ENABLED`` off the app must take exactly today's branch — no loop record,
    no round, not one of the five stages re-entered in the ``if submit:`` pass, and a
    ``final_payload`` that is the Stage-1 payload merged with Stage 2 and nothing else.
    """

    at = _run_submit(monkeypatch, rag_enabled=False)

    assert _app_exceptions(at) == []
    assert "rag_loop" not in at.session_state
    assert "rag_result" not in at.session_state
    # Not one round-scoped stage ran. ``persistence`` is the exception that proves it:
    # the ONE call recorded is the app's own authoritative ``build_and_save_draft_graph``
    # at :5659, which passes no ``output_path`` — a round's call always passes one.
    for name in ("normalization", "mapping", "gates", "classification"):
        assert offline_stages[name] == [], f"{name} ran with RAG off"
    assert offline_stages["persistence"] == [None]
    assert [r["name"] for r in at.session_state["final_payload"]["processes"]["reactions"]] == [
        "menaquinone demethylation"]
