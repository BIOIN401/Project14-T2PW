"""Scoped pinned acquisition, production-shaped artifacts, and partial-run rates.

Why this file exists
--------------------
Three defects reached a staged run before anything noticed, and each one was
invisible in the place a reader would look:

1. **Scope loss.** ``runs/2026-08-01_1724`` fetched all ten pinned papers,
   skipped none, and reported a clean acquisition funnel -- with
   ``requested_pathway: ""`` and ``requested_organism: ""`` on every row, because
   a bare pinned id yields ``TopicSpec(pinned_id=..., topic="", organism="")``.
   Extraction would have been handed no scope at all.

2. **A driver reading a shape production never produces.**
   ``rag_admission_report.json`` was written from ``rag_result.admission``. The
   report actually hangs off ``rag_result.synthesis.admission`` (see
   ``streamlit_app.render_rag_panels``), so against a real run the artifact would
   simply never appear -- and the reintroduction check would report "not
   evaluated" forever while looking wired up.

3. **Rates over a population that never ran.** Scoring a 7-of-10 run against the
   full gold set charged the exporter for four strict legs when only two had been
   attempted.

Offline and deterministic: no network, no LLM, no Streamlit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.fetch import BatchPaper, TopicSpec, parse_topics  # noqa: E402
from t2pw.bench.acceptance import score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.metrics import (  # noqa: E402
    BLOCKERS_EXTRACTION,
    BLOCKERS_RESEARCH,
    BLOCKERS_STRICT,
    DENOM_RELEVANCE,
    DENOM_RESEARCH_CONFIRMED,
    DENOM_SEMANTIC,
    DENOM_STRICT,
)
from t2pw.bench.preflight import (  # noqa: E402
    EXPECTED_OUTCOME,
    PreflightError,
    verify_plan,
    verify_rows,
    verify_specs,
)


# ---------------------------------------------------------------------------
# 1. Scoped pinned acquisition.
# ---------------------------------------------------------------------------
def test_a_bare_pinned_id_still_parses_but_carries_no_scope():
    # Kept working on purpose: the batch runner is a general tool and an
    # exploratory pinned run is legitimate. It is the BENCHMARK that must refuse.
    specs = parse_topics("PMC4412817")
    assert specs[0].is_pinned
    assert specs[0].topic == ""
    assert specs[0].organism == ""


def test_a_scoped_pinned_line_preserves_the_request_exactly():
    specs = parse_topics("PMC12444477 | lipid A biosynthesis | Escherichia coli")
    assert len(specs) == 1
    spec = specs[0]
    assert spec.is_pinned and spec.pinned_id == "PMC12444477"
    assert spec.topic == "lipid A biosynthesis"
    assert spec.organism == "Escherichia coli"
    assert spec.count == 1, "a pinned line asks for exactly one paper"


def test_a_search_topic_is_not_mistaken_for_a_pinned_line():
    specs = parse_topics("lipid A biosynthesis | Escherichia coli | 6")
    assert not specs[0].is_pinned
    assert specs[0].topic == "lipid A biosynthesis"
    assert specs[0].organism == "Escherichia coli"
    assert specs[0].count == 6


def test_comments_and_blanks_still_ignored_on_scoped_lines():
    specs = parse_topics("\n# a comment\n\nPMC1 | p | o   # trailing\n")
    assert len(specs) == 1
    assert specs[0].pinned_id == "PMC1"
    assert specs[0].organism == "o"


def test_the_scope_reaches_batchpaper_and_the_plan_row():
    """The end of the chain: spec -> BatchPaper -> plan row.

    ``_as_batch_paper`` reads ``spec.topic`` / ``spec.organism`` directly into
    ``requested_pathway`` / ``requested_organism``, which is exactly why an empty
    spec produced an empty plan.
    """
    spec = TopicSpec(pinned_id="PMC12444477", topic="lipid A biosynthesis", organism="Escherichia coli")
    paper = BatchPaper(
        paper_id=spec.pinned_id,
        requested_pathway=spec.topic,
        requested_organism=spec.organism,
        topic=spec.topic,
        query="",
        eligibility={"outcome": EXPECTED_OUTCOME},
    )
    row = paper.to_dict()
    assert row["requested_pathway"] == "lipid A biosynthesis"
    assert row["requested_organism"] == "Escherichia coli"
    assert row["query"] == "", "a pinned paper is fetched by id and records no query"


def test_preflight_accepts_the_generated_topics_file():
    gold = load_gold_set()
    report = verify_specs(parse_topics("\n".join(gold.topics_lines())), gold)
    assert report.ok, report.render()
    assert report.search_calls == 0
    assert len(report.triples) == len(gold)
    for triple, case in zip(report.triples, gold.cases):
        assert triple["paper_id"] == case.paper_id
        assert triple["requested_pathway"] == case.requested_pathway
        assert triple["requested_organism"] == case.requested_organism
        assert triple["outcome"] == EXPECTED_OUTCOME


def test_preflight_refuses_a_scopeless_plan():
    gold = load_gold_set()
    rows = [
        {"paper_id": case.paper_id, "requested_pathway": "", "requested_organism": "",
         "query": "", "eligibility": {"outcome": EXPECTED_OUTCOME}}
        for case in gold
    ]
    report = verify_rows(rows, gold, source="test")
    assert not report.ok
    assert len(report.findings) == 2 * len(gold)
    with pytest.raises(PreflightError, match="requested_pathway"):
        report.raise_for_status()


def test_preflight_refuses_a_changed_scope():
    gold = load_gold_set()
    case = gold.cases[0]
    rows = [
        {"paper_id": c.paper_id,
         "requested_pathway": "something else" if c is case else c.requested_pathway,
         "requested_organism": c.requested_organism,
         "query": "", "eligibility": {"outcome": EXPECTED_OUTCOME}}
        for c in gold
    ]
    report = verify_rows(rows, gold, source="test")
    assert not report.ok
    assert any(f.paper_id == case.paper_id and f.field_name == "requested_pathway" for f in report.findings)


def test_preflight_refuses_a_recorded_search_query():
    gold = load_gold_set()
    rows = [
        {"paper_id": c.paper_id, "requested_pathway": c.requested_pathway,
         "requested_organism": c.requested_organism,
         "query": '"lipid A biosynthesis" AND "Escherichia coli"' if c is gold.cases[0] else "",
         "eligibility": {"outcome": EXPECTED_OUTCOME}}
        for c in gold
    ]
    report = verify_rows(rows, gold, source="test")
    assert not report.ok
    assert report.search_calls == 1


def test_preflight_refuses_admission_by_score_rather_than_override():
    gold = load_gold_set()
    rows = [
        {"paper_id": c.paper_id, "requested_pathway": c.requested_pathway,
         "requested_organism": c.requested_organism, "query": "",
         "eligibility": {"outcome": "eligible" if c is gold.cases[0] else EXPECTED_OUTCOME}}
        for c in gold
    ]
    report = verify_rows(rows, gold, source="test")
    assert not report.ok
    assert any(f.field_name == "eligibility.outcome" for f in report.findings)


def test_preflight_refuses_a_missing_or_extra_paper(tmp_path):
    gold = load_gold_set()
    rows = [
        {"paper_id": c.paper_id, "requested_pathway": c.requested_pathway,
         "requested_organism": c.requested_organism, "query": "",
         "eligibility": {"outcome": EXPECTED_OUTCOME}}
        for c in gold.cases[1:]
    ] + [{"paper_id": "PMC00000", "requested_pathway": "x", "requested_organism": "y"}]
    report = verify_rows(rows, gold, source="test")
    assert report.missing == [gold.cases[0].paper_id]
    assert report.unexpected == ["PMC00000"]
    assert not report.ok


def test_verify_plan_reads_a_stored_plan_json(tmp_path):
    gold = load_gold_set()
    plan = {
        # `modes` is required: a plan that does not schedule exactly strict and
        # research is the wrong shape for an acceptance run, and is refused by
        # verify_plan's shape checks.
        "modes": ["strict", "research"],
        "papers": [
            {"paper_id": c.paper_id, "requested_pathway": c.requested_pathway,
             "requested_organism": c.requested_organism, "query": "",
             "eligibility": {"outcome": EXPECTED_OUTCOME}}
            for c in gold
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert verify_plan(path, gold).ok
    assert verify_plan(tmp_path, gold).ok, "a run directory should resolve to its plan.json"

    with pytest.raises(PreflightError, match="no plan to verify"):
        verify_plan(tmp_path / "nope", gold)


# ---------------------------------------------------------------------------
# 2. Production-shaped RAG admission artifact.
# ---------------------------------------------------------------------------
ADMISSION = {
    "accepted": [{"gap_id": "g1", "name": "chorismate to isochorismate"}],
    "rejected": [
        {
            "gap_id": "g2",
            "name": "invented step",
            "inputs": ["a"],
            "outputs": ["b"],
            "enzymes": ["SomeEnzyme"],
            "reasons": ["organism_mismatch"],
        }
    ],
    "counts": {"accepted": 1, "rejected": 1},
    "reason_counts": {"organism_mismatch": 1},
    "resolutions": [{"name": "SomeEnzyme", "outcome": "unverified_no_resolver"}],
    "identity_outcomes": {"unverified_no_resolver": 1},
}


def _rag_result(*, admission=ADMISSION, payload=None):
    """The shape production builds: admission hangs off `.synthesis`."""

    return SimpleNamespace(
        synthesis=SimpleNamespace(admission=dict(admission) if admission else None),
        payload=payload if payload is not None else {"entities": {}, "processes": {}},
    )


class _AppTest:
    """The two attributes `driver._ss` actually reads."""

    def __init__(self, state):
        self.session_state = state


def _artifacts_for(rag_result, artifacts=None):
    from t2pw.batch import driver

    out = {}
    at = _AppTest({"rag_result": rag_result})
    driver._add_identity_artifacts(at, artifacts or {}, out)
    return out


def test_admission_is_read_from_the_production_synthesis_attribute():
    out = _artifacts_for(_rag_result())
    assert "rag_admission_report.json" in out
    blob = json.loads(out["rag_admission_report.json"])
    for key in ("accepted", "rejected", "counts", "reason_counts", "resolutions", "identity_outcomes"):
        assert key in blob, f"the artifact must carry {key}"
    assert blob["rejected"][0]["reasons"] == ["organism_mismatch"]


def test_a_flattened_admission_attribute_is_still_accepted():
    # Fallback only, so a hand-built double or a future flattening does not
    # silently stop producing the artifact.
    out = _artifacts_for(SimpleNamespace(synthesis=None, admission=dict(ADMISSION)))
    assert "rag_admission_report.json" in out


def test_no_rag_result_writes_no_admission_artifact_and_does_not_raise():
    assert _artifacts_for(None) == {}
    assert _artifacts_for(SimpleNamespace()) == {}
    assert _artifacts_for(SimpleNamespace(synthesis=None)) == {}
    assert _artifacts_for(_rag_result(admission=None)) == {}


@pytest.mark.parametrize(
    "scenario, artifacts",
    [
        # A strict run that died at the Stage-3 gate: mapping ran, export never did.
        ("strict failure", {"final_mapped": {"entities": {"proteins": [{"name": "MenD"}]}}}),
        # A research run that succeeded: no PWML, but RAG synthesized.
        ("research success", {"final_mapped_db": {"entities": {"proteins": []}}}),
        # A failure after RAG synthesis but before PWML export: no mapped payload
        # at all, and the admission report is the only surviving RAG evidence.
        ("failure after synthesis, before export", {}),
    ],
)
def test_the_admission_artifact_survives_every_failure_path(scenario, artifacts):
    out = _artifacts_for(_rag_result(), artifacts)
    assert "rag_admission_report.json" in out, scenario
    if artifacts:
        assert "final_mapped.json" in out, scenario
    else:
        assert "final_mapped.json" not in out, scenario


def test_the_mapped_payload_prefers_the_db_resolved_copy():
    out = _artifacts_for(
        _rag_result(),
        {"final_mapped_db": {"entities": {"tag": "db"}}, "final_mapped": {"entities": {"tag": "plain"}}},
    )
    assert json.loads(out["final_mapped.json"])["entities"]["tag"] == "db"


# ---------------------------------------------------------------------------
# 3. Partial-run denominators.
# ---------------------------------------------------------------------------
def _run_with(tmp_path: Path, rows) -> Path:
    run_dir = tmp_path / "2026-01-01_0000"
    (run_dir / "papers").mkdir(parents=True)
    run_dir.joinpath("manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return run_dir


def test_an_unattempted_paper_is_missing_coverage_not_a_failure(tmp_path):
    gold = load_gold_set()
    strict_ids = list(gold.strict_expected_ids)
    assert len(strict_ids) >= 2

    # Attempt the strict leg of exactly one strict-exportable paper, and fail it.
    rows = [
        {
            "paper_id": strict_ids[0],
            "slug": strict_ids[0],
            "mode": "strict",
            "status": "fail",
            "stage": "post_pipeline",
            "failure_kind": "contract",
            "issue_codes": ["gate.protein_x_is_missing_a_unipro"],
            "files": [],
        }
    ]
    report = score_run(_run_with(tmp_path, rows), gold)

    strict = report.denominators[DENOM_STRICT]
    assert strict.denominator == 1, "only the attempted strict leg is in the denominator"
    assert strict.numerator == 0
    excluded = {e["paper_id"] for e in strict.excluded}
    for paper_id in strict_ids[1:]:
        assert paper_id in excluded
    assert any("never attempted" in e["reason"] for e in strict.excluded)

    completion = report.completion()
    assert completion["planned_gold_cases"] == len(gold)
    assert completion["papers_attempted"] == 1
    assert completion["strict_legs_attempted"] == 1
    assert completion["research_legs_attempted"] == 0
    assert completion["fully_completed_cases"] == 0
    assert completion["complete"] is False
    assert len(completion["papers_with_no_attempted_leg"]) == len(gold) - 1
    assert completion["papers_with_only_one_mode_attempted"] == [strict_ids[0]]
    assert any("PARTIAL RUN" in note for note in report.notes)


def test_the_research_denominator_only_counts_attempted_research_legs(tmp_path):
    gold = load_gold_set()
    relevant = [c.paper_id for c in gold if c.mechanistic_relevance != "context_only"]
    rows = [
        {"paper_id": relevant[0], "slug": relevant[0], "mode": "research",
         "status": "fail", "stage": "stage1", "failure_kind": "contract",
         "issue_codes": ["processes_required"], "files": []}
    ]
    report = score_run(_run_with(tmp_path, rows), gold)
    research = report.denominators[DENOM_RESEARCH_CONFIRMED]
    assert research.denominator == 1
    assert research.numerator == 0
    assert all(p != relevant[0] for p in (e["paper_id"] for e in research.excluded))


def test_gold_relevance_prevalence_is_not_called_an_eligibility_yield(tmp_path):
    gold = load_gold_set()
    report = score_run(_run_with(tmp_path, []), gold)
    prevalence = report.denominators[DENOM_RELEVANCE]
    assert prevalence.key == "gold_relevance_prevalence"
    assert "GOLD SET" in prevalence.question
    # It is a property of the gold set, so it does not move with what ran.
    assert prevalence.denominator == len(gold)


# ---------------------------------------------------------------------------
# 4. Semantic success requires `confirmed`, not `ok`.
# ---------------------------------------------------------------------------
def test_a_leg_with_an_unavailable_check_never_enters_the_semantic_numerator(tmp_path):
    """Every applicable check passes, but reintroduction could not be evaluated.

    ``ok`` is True here -- it only considers *applicable* checks -- so scoring on
    ``ok`` would count this as a semantic success while one question was never
    asked. ``confirmed`` additionally requires that every check WAS evaluable.
    """
    gold = load_gold_set()
    case = next(c for c in gold if c.supported_reactions and c.mechanistic_relevance != "context_only")

    signature = case.supported_reactions[0]
    payload = {
        "entities": {
            "compounds": [{"name": t.name} for t in (*signature.inputs, *signature.outputs)],
            "proteins": ([{"name": signature.enzyme.name, "organism": case.requested_organism}]
                         if signature.enzyme else []),
            "protein_complexes": [],
        },
        "processes": {
            "reactions": [
                {
                    "name": signature.label or "supported step",
                    "inputs": [t.name for t in signature.inputs],
                    "outputs": [t.name for t in signature.outputs],
                    "enzymes": ([{"protein": signature.enzyme.name}] if signature.enzyme else []),
                    "evidence": signature.quote,
                }
            ],
            "transports": [],
            "interactions": [],
        },
    }
    for term in case.expected_pathway_anchors:
        payload["entities"]["compounds"].append({"name": term.name})

    run_dir = _run_with(
        tmp_path,
        [{"paper_id": case.paper_id, "slug": case.paper_id, "mode": "strict",
          "status": "fail", "stage": "post_pipeline", "failure_kind": "contract",
          "issue_codes": [], "files": []}],
    )
    leg = run_dir / "papers" / case.paper_id / "strict"
    leg.mkdir(parents=True)
    (leg / "final_mapped.json").write_text(json.dumps(payload), encoding="utf-8")

    report = score_run(run_dir, gold)
    semantic = report.papers[[p.paper_id for p in report.papers].index(case.paper_id)].leg("strict").semantic
    assert semantic is not None and semantic.evaluated
    assert "no_rejected_rag_reaction_reintroduced" in semantic.inapplicable_checks
    assert not semantic.confirmed, "an unevaluated check must block confirmation"
    assert case.paper_id not in report.denominators[DENOM_SEMANTIC].numerator_names


# ---------------------------------------------------------------------------
# 5. Blocker rankings stay separate.
# ---------------------------------------------------------------------------
def test_a_partial_only_paper_never_appears_in_the_strict_blocker_ranking(tmp_path):
    """PMC12312563 is partial_only: fixing its strict gate cannot raise the rate.

    A merged ranking presented it as the top strict blocker, which sends the next
    day's work at a fix that no benchmark denominator can collect.
    """
    gold = load_gold_set()
    partial = next(c for c in gold if c.expected_export == "partial_only" and not c.is_negative_control)
    strict_case = next(c for c in gold if c.expected_export == "strict_exportable")

    rows = []
    for case in (partial, strict_case):
        rows.append(
            {"paper_id": case.paper_id, "slug": case.paper_id, "mode": "strict",
             "status": "fail", "stage": "post_pipeline", "failure_kind": "contract",
             "issue_codes": ["gate.protein_x_is_missing_a_unipro"], "files": []}
        )
    run_dir = _run_with(tmp_path, rows)
    for case in (partial, strict_case):
        leg = run_dir / "papers" / case.paper_id / "strict"
        leg.mkdir(parents=True)
        (leg / "merged_payload.json").write_text(
            json.dumps(
                {
                    "entities": {"compounds": [{"name": "a"}, {"name": "b"}], "proteins": [], "protein_complexes": []},
                    "processes": {"reactions": [{"name": "r", "inputs": ["a"], "outputs": ["b"], "evidence": "q"}],
                                  "transports": [], "interactions": []},
                }
            ),
            encoding="utf-8",
        )

    report = score_run(run_dir, gold)
    strict_papers = {p for b in report.blockers[BLOCKERS_STRICT] for p in b.papers}
    assert strict_case.paper_id in strict_papers
    assert partial.paper_id not in strict_papers, (
        "a partial_only paper is not in the strict denominator, so it cannot be a strict blocker"
    )
    assert BLOCKERS_RESEARCH in report.blockers
    assert BLOCKERS_EXTRACTION in report.blockers
