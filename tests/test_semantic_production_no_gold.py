"""C-017 / D-006: semantic evaluation of a produced payload with NO gold case.

Guards, in order: a clean payload passes; a real defect fails and the failure names it;
``not_evaluated`` is never `false` and never a pass; the five D-006 states stay distinct;
the module drags in no benchmark harness; and no path touches a socket, a database or an
LLM client.
"""

import importlib
import json
import os
import subprocess
import sys
from types import MappingProxyType

from t2pw.bench import semantic as bench_semantic
from t2pw.bench import semantic_production as sp

HARNESS = ("t2pw.bench.acceptance", "t2pw.bench.metrics", "t2pw.bench.render", "t2pw.bench.preflight")
REMOTE = ("openai", "requests", "httpx", "urllib.request", "http.client", "socket", "ssl", "sqlite3")

PROBE = """\
import json, sys
before = set(sys.modules)
import t2pw.bench
package = set(sys.modules)
import t2pw.bench.semantic_production
print(json.dumps({"package": sorted(package - before), "own": sorted(set(sys.modules) - package),
                  "t2pw": sorted(n for n in sys.modules if n.startswith("t2pw")),
                  "loaded": sorted(sys.modules)}))
"""


def payload():
    """A small lipid A payload that is semantically clean: connected, sourced, mapped."""
    return {
        "pathway": {"name": "lipid A biosynthesis"},
        "entities": {
            "compounds": [{"name": "UDP-GlcNAc"}, {"name": "lipid X"}, {"name": "lipid A"}],
            "proteins": [{"name": "LpxA", "uniprot": "P0A722", "identity_status": "verified"}],
        },
        "processes": {"reactions": [
            {"name": "acylation", "inputs": ["UDP-GlcNAc"], "outputs": ["lipid X"], "enzymes": ["LpxA"],
             "organism": "Escherichia coli", "evidence": "the paper states this step"},
            {"name": "condensation", "inputs": ["lipid X"], "outputs": ["lipid A"], "enzymes": ["LpxA"],
             "organism": "Escherichia coli", "evidence": "the paper states this step"},
        ]},
    }


def evaluate(case, **kw):
    kw.setdefault("requested_pathway", "lipid A biosynthesis")
    kw.setdefault("requested_organism", "Escherichia coli")
    return sp.evaluate_production_semantics(case, **kw)


def axes(report):
    """The facts D-006 forbids collapsing, read straight off the report."""
    return (report.evaluated, report.ok, report.confirmed,
            tuple(report.failed_checks), tuple(report.inapplicable_checks))


def test_a_clean_payload_passes_with_no_gold_case():
    report = evaluate(payload(), min_connected_reactions=2)
    assert (report.evaluated, report.ok, report.failed_checks) == (True, True, [])


def test_b_real_semantic_defects_fail_and_the_failure_names_them():
    # C-080 / F-108 narrowed the collision half of this obligation to a
    # CROSS-KIND claim, exactly as C-076 narrowed the acceptance scorer's own
    # ``test_one_accession_claimed_by_two_entities_is_a_conflict``. The node id
    # and the assertion shape are kept, so the obligation stays where readers of
    # this file expect it and no test count moves.
    #
    # It used to append a second PROTEIN -- ``LpxD`` -- carrying LpxA's UniProt
    # accession. That is the predicate C-073's review rejected in the pipeline for
    # contradicting **D-035 clause 3c**: a matching stable external identifier is
    # *proof* that two differently-named rows are the same entity, so a name
    # difference alone is agreement, not collision. The product owner's ruling of
    # 2026-08-23 rejected it in the scorer, and this card rejects it in the
    # production release gate. What survives is the one true defect C-073 fixed
    # upstream, in its natural form: the mapper has given the METABOLITE
    # ``lipid X`` the accession of the ENZYME ``LpxA``, fusing a protein and a
    # compound under one identifier. The within-kind half is pinned in
    # ``tests/test_c080_production_gate_kind_aware.py``.
    broken = payload()
    broken["entities"]["compounds"][1]["uniprot"] = "P0A722"
    broken["processes"]["reactions"][1]["organism"] = "Listeria monocytogenes"
    report = evaluate(broken)
    assert report.ok is False
    assert bench_semantic.CHECK_ID_CONFLICT in report.failed_checks
    assert bench_semantic.CHECK_ORGANISM in report.failed_checks
    collision = report.checks[bench_semantic.CHECK_ID_CONFLICT].findings[-1]
    assert collision["kind"] == "accession_claimed_by_multiple_entities"
    assert collision["identifier"] == "p0a722"
    assert sorted(collision["entities"]) == ["LpxA", "lipid X"]
    cross = report.checks[bench_semantic.CHECK_ORGANISM].findings[0]
    assert cross["pointer"] == "/processes/reactions/1"
    assert cross["observed_organism"] == "Listeria monocytogenes"


def test_c_gold_only_check_is_not_evaluated_and_is_neither_a_pass_nor_a_failure():
    report = evaluate(payload(), min_connected_reactions=2)
    name = bench_semantic.CHECK_SUPPORTED_REACTIONS
    check = report.checks[name]
    assert check.applicable is False
    assert check.inapplicable_reason == sp.NO_GOLD_SIGNATURES
    assert check.ok is not False
    assert name in report.inapplicable_checks
    assert name not in report.failed_checks
    assert report.ok is True
    assert report.confirmed is False
    assert report.to_dict()["checks"][name]["applicable"] is False
    # The error count it would have fed is ABSENT, not a zero: a zero would read as
    # "measured, clean" for a question production cannot ask.
    assert bench_semantic.ERR_UNSUPPORTED_REACTIONS not in report.scientific_errors


def test_d_five_d006_states_are_independently_reachable():
    no_payload = sp.evaluate_production_semantics(None)
    assert no_payload.evaluated is False
    assert no_payload.not_evaluated_reason == sp.NO_PAYLOAD
    assert no_payload.checks == {} and no_payload.failed_checks == []
    gated = payload()
    gated["strict_gate_passed"] = True
    gated["release_status"] = "release_ready"
    gated["processes"]["reactions"][1]["organism"] = "Listeria monocytogenes"
    failed = evaluate(gated, min_connected_reactions=2)
    assert failed.ok is False
    assert not {"release_status", "strict_gate_passed"} & set(failed.to_dict())
    passed = evaluate(payload(), min_connected_reactions=2)
    unasked = sp.evaluate_production_semantics(payload())
    assert axes(passed)[:3] == (True, True, False)
    assert axes(failed)[:3] == (True, False, False)
    # C-071 appended CHECK_ACTOR_EVIDENCE. It is inapplicable here for a reason that is
    # this list's own subject: ``payload()``'s ``enzymes`` are bare strings, so no actor
    # row cites a span of its own and there is nothing to compare. Reporting it as
    # PASSED on a payload it never examined is the one answer D-006 forbids, so the
    # honest disposition puts it exactly here.
    assert unasked.inapplicable_checks == [
        bench_semantic.CHECK_ANCHORS, bench_semantic.CHECK_SUPPORTED_REACTIONS,
        bench_semantic.CHECK_ORGANISM, bench_semantic.CHECK_RAG_REINTRODUCTION,
        bench_semantic.CHECK_CONNECTED_CORE, bench_semantic.CHECK_ACTOR_EVIDENCE,
    ]
    states = [axes(no_payload), axes(passed), axes(failed), axes(unasked)]
    assert len(set(states)) == len(states)
    # A real payload is never reported as "no payload": NO_PAYLOAD is state 5, and
    # attaching it to a mapping that IS a payload would be a false reason on state 1.
    assert sp.evaluate_production_semantics(MappingProxyType(payload())).evaluated is True


def test_e_import_graph_carries_no_benchmark_harness(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True,
                         timeout=120, check=True, env=env)
    graph = json.loads(out.stdout.strip().splitlines()[-1])
    assert set(graph["t2pw"]) == {"t2pw", "t2pw.bench", "t2pw.bench.goldset",
                                  "t2pw.bench.semantic", "t2pw.bench.semantic_production"}
    # goldset arrives from ``t2pw.bench.__init__``, NOT from this module: its own imports
    # add only the sanctioned helper module and itself.
    assert "t2pw.bench.goldset" in graph["package"]
    assert set(graph["own"]) == {"t2pw.bench.semantic", "t2pw.bench.semantic_production"}
    for name in HARNESS + REMOTE:
        assert name not in graph["loaded"]


def test_h_anchor_match_needs_whole_tokens_and_floors_both_operands():
    """CHECK_ANCHORS is this module's ONLY "is this the pathway that was asked for?"
    question and C-056a wires it to release_status, so a permissive match ships a pathway
    the run never produced. Every pair below is a false anchor that a raw substring test
    accepts and ``goldset.contains_term`` rejects."""

    for requested, name in (("alanine biosynthesis", "alan"), ("ALA", "alanine"),
                            ("menaquinone biosynthesis", "quinone"),
                            ("glycolysis", "lysis"), ("AA", "isoamyl aacid")):
        check = sp.evaluate_production_semantics(
            {"entities": {"compounds": [{"name": name}]}},
            requested_pathway=requested).checks[bench_semantic.CHECK_ANCHORS]
        assert check.ok is False, f"{name!r} must not anchor the request {requested!r}"
        assert check.findings[0]["missing_anchor"] == requested
    # ...while a genuine whole-token anchor still matches, in both directions.
    for requested, name in (("lipid A biosynthesis", "lipid A"), ("lipid A", "lipid A biosynthesis")):
        check = sp.evaluate_production_semantics(
            {"entities": {"compounds": [{"name": name}]}},
            requested_pathway=requested).checks[bench_semantic.CHECK_ANCHORS]
        assert check.ok is True, f"{name!r} must anchor the request {requested!r}"


def test_g_unknown_backed_protein_is_counted_never_adjudicated():
    """TRAP-3 / PRODUCT_CONTRACT section 13: whether an Unknown-backed protein should be
    exported is a standing policy disagreement. A placeholder that admits what it is
    passes, and the count is still reported so the disagreement stays visible. This module
    takes neither side, and this test fails if a later edit makes it take one."""

    honest = payload()
    honest["entities"]["proteins"].append({
        "name": "Unknown", "identity_status": "placeholder",
        "mapping_meta": {"fallback_used": True},  # a correctly-formed placeholder records WHY
    })
    report = evaluate(honest, min_connected_reactions=2)
    assert report.ok is True
    assert bench_semantic.CHECK_PLACEHOLDER_IDENTITY not in report.failed_checks
    assert report.scientific_errors[bench_semantic.ERR_PLACEHOLDER_BACKED_PROTEINS] == 1


def test_f_no_network_database_or_llm_on_any_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("semantic_production reached the network or a database")

    for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo",
                   "sqlite3.connect", "urllib.request.urlopen"):
        module, _, attribute = target.rpartition(".")
        monkeypatch.setattr(importlib.import_module(module), attribute, forbidden)
    forger = payload()
    forger["entities"]["proteins"] = [
        {"name": "Unknown", "identity_status": "placeholder", "uniprot": "P99999"}]
    for case in (None, {}, payload(), forger):
        report = evaluate(case, min_connected_reactions=1)
        assert isinstance(report, bench_semantic.SemanticReport)
