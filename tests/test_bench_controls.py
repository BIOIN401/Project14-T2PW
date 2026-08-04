"""Benchmark control gaps: completeness, plan shape, stage-only, and honest rates.

Why this file exists
--------------------
Each test here pins a way the benchmark could report a *better* result than it
earned, without any gold label being touched:

* a 19/20 run exiting 0, because every error count is computed over the legs that
  ran and the twentieth leg is exactly where the missing error would be;
* a plan that lost its acquisition outcome, duplicated a paper, dropped a mode or
  shrank by one paper -- all of which score cleanly on whatever they do contain;
* a "staged" run that actually started a leg because the deadline it relied on is
  evaluated inside the loop, after a cache-warm acquisition;
* a negative control ranked as an ``extraction_empty`` blocker for correctly
  declining to invent a pathway;
* a research citation report counted as scientific success;
* ``placeholder_backed == 0`` rendered as "every protein has a real identity"
  when in fact nothing resolved at all.

Offline and deterministic: no network, no LLM, no Streamlit.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.bench.acceptance import MODES, score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set  # noqa: E402
from t2pw.bench.metrics import (  # noqa: E402
    BLOCKERS_EXTRACTION,
    DENOM_RESEARCH_CONFIRMED,
    DENOM_RESEARCH_DELIVERABLE,
)
from t2pw.bench.preflight import (  # noqa: E402
    EXPECTED_OUTCOME,
    REQUIRED_MODES,
    verify_plan,
    verify_rows,
)
from t2pw.bench.render import render_text  # noqa: E402
from t2pw.bench.semantic import CHECK_PLACEHOLDER_IDENTITY, validate_semantic_coverage  # noqa: E402

GOLD = load_gold_set()
CLI = ROOT / "scripts" / "bench_acceptance.py"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _payload(reactions=1):
    rows = [
        {"name": f"step {i}", "inputs": [f"m{i}"], "outputs": [f"m{i + 1}"], "evidence": "quoted"}
        for i in range(reactions)
    ]
    names = {n for r in rows for n in (*r["inputs"], *r["outputs"])}
    return {
        "entities": {
            "compounds": [{"name": n} for n in sorted(names)],
            "proteins": [],
            "protein_complexes": [],
        },
        "processes": {"reactions": rows, "transports": [], "interactions": []},
    }


def _build_run(tmp_path: Path, legs, *, payloads=True) -> Path:
    """``legs`` is a sequence of ``(paper_id, mode)`` that were attempted."""

    run_dir = tmp_path / "2026-01-01_0000"
    (run_dir / "papers").mkdir(parents=True)
    rows = []
    for paper_id, mode in legs:
        rows.append(
            {
                "paper_id": paper_id,
                "slug": paper_id,
                "mode": mode,
                "status": "fail",
                "stage": "post_pipeline",
                "failure_kind": "contract",
                "issue_codes": ["gate.protein_x_is_missing_a_unipro"],
                "files": [],
            }
        )
        if payloads:
            leg = run_dir / "papers" / paper_id / mode
            leg.mkdir(parents=True, exist_ok=True)
            (leg / "merged_payload.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (run_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return run_dir


def _all_legs():
    return [(case.paper_id, mode) for case in GOLD for mode in MODES]


def _run_cli(run_dir: Path, tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(CLI), "--run-dir", str(run_dir), "--json", str(tmp_path / "r.json")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    return result


# ---------------------------------------------------------------------------
# 1. A partial run must never exit accepted.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attempted", [0, 1, 19])
def test_an_incomplete_run_exits_nonzero(tmp_path, attempted):
    """Even when every leg that ran was clean.

    Error counts are computed over the legs that ran, so fewer legs means fewer
    chances to be wrong -- 0/20 clears priorities 1-3 perfectly, and 19/20 is the
    same failure in miniature.
    """
    run_dir = _build_run(tmp_path, _all_legs()[:attempted])
    report = score_run(run_dir, GOLD)
    assert not report.is_complete
    assert report.completion()["legs_attempted"] == attempted

    result = _run_cli(run_dir, tmp_path)
    assert result.returncode == 1, result.stdout[-2000:]
    assert "INCOMPLETE" in result.stderr
    # The reports are still produced -- a partial run is worth reading.
    assert (tmp_path / "r.json").is_file()
    assert "PARTIAL" in result.stdout


def test_a_complete_run_is_judged_on_its_priorities_not_on_completeness(tmp_path):
    run_dir = _build_run(tmp_path, _all_legs())
    report = score_run(run_dir, GOLD)
    completion = report.completion()
    assert completion["legs_attempted"] == 20
    assert completion["fully_completed_cases"] == len(GOLD)
    assert report.is_complete

    result = _run_cli(run_dir, tmp_path)
    assert "INCOMPLETE" not in result.stderr
    # This synthetic payload violates priorities, so it still exits 1 -- but for
    # a scientific reason, which is the point of the distinction.
    assert result.returncode == 1
    assert any(not e["ok"] for e in report.priorities() if e["rank"] <= 3)


def test_a_partial_report_says_partial_in_both_formats(tmp_path):
    run_dir = _build_run(tmp_path, _all_legs()[:4])
    report = score_run(run_dir, GOLD)
    assert "PARTIAL" in render_text(report)
    assert report.to_dict()["completion"]["complete"] is False


# ---------------------------------------------------------------------------
# 2. Plan preflight.
# ---------------------------------------------------------------------------
def _good_rows():
    return [
        {
            "paper_id": case.paper_id,
            "requested_pathway": case.requested_pathway,
            "requested_organism": case.requested_organism,
            "query": "",
            "eligibility": {"outcome": EXPECTED_OUTCOME},
        }
        for case in GOLD
    ]


def _plan(tmp_path: Path, rows, *, modes=None) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps({"papers": rows, "modes": list(REQUIRED_MODES if modes is None else modes)}),
        encoding="utf-8",
    )
    return path


def test_a_well_formed_plan_passes(tmp_path):
    assert verify_plan(_plan(tmp_path, _good_rows()), GOLD).ok


def test_a_missing_acquisition_outcome_is_a_failure(tmp_path):
    rows = _good_rows()
    del rows[0]["eligibility"]
    report = verify_plan(_plan(tmp_path, rows), GOLD)
    assert not report.ok
    finding = next(f for f in report.findings if f.field_name == "eligibility.outcome")
    assert finding.observed == "(absent)"


def test_a_duplicate_paper_id_is_refused_not_silently_deduped(tmp_path):
    rows = _good_rows()
    rows.append(dict(rows[0]))
    report = verify_plan(_plan(tmp_path, rows), GOLD)
    assert not report.ok
    assert report.duplicates == [GOLD.cases[0].paper_id]


def test_a_missing_mode_is_refused(tmp_path):
    report = verify_plan(_plan(tmp_path, _good_rows(), modes=["strict"]), GOLD)
    assert not report.ok
    assert any("modes" in problem for problem in report.shape)
    assert any("pair" in problem for problem in report.shape)


def test_an_extra_mode_is_refused(tmp_path):
    report = verify_plan(_plan(tmp_path, _good_rows(), modes=["strict", "research", "draft"]), GOLD)
    assert not report.ok
    assert any("modes" in problem for problem in report.shape)


def test_a_missing_case_is_refused(tmp_path):
    rows = _good_rows()[:-1]
    report = verify_plan(_plan(tmp_path, rows), GOLD)
    assert not report.ok
    assert report.missing == [GOLD.cases[-1].paper_id]
    assert any("paper row" in problem for problem in report.shape)


def test_an_extra_case_is_refused(tmp_path):
    rows = _good_rows()
    rows.append(
        {"paper_id": "PMC00000", "requested_pathway": "x", "requested_organism": "y",
         "query": "", "eligibility": {"outcome": EXPECTED_OUTCOME}}
    )
    report = verify_plan(_plan(tmp_path, rows), GOLD)
    assert not report.ok
    assert report.unexpected == ["PMC00000"]


def test_the_required_pair_count_is_papers_times_modes(tmp_path):
    report = verify_plan(_plan(tmp_path, _good_rows()), GOLD)
    assert report.ok
    assert len(GOLD) == 10
    assert len(GOLD) * len(REQUIRED_MODES) == 20


def test_verify_specs_still_validates_before_any_fetch():
    from t2pw.batch.fetch import parse_topics
    from t2pw.bench.preflight import verify_specs

    # No acquisition outcome exists yet, so requiring one here would refuse every
    # valid topics file.
    report = verify_specs(parse_topics("\n".join(GOLD.topics_lines())), GOLD)
    assert report.ok, report.render()


def test_verify_rows_can_be_told_not_to_require_an_outcome():
    rows = [{k: v for k, v in row.items() if k != "eligibility"} for row in _good_rows()]
    assert verify_rows(rows, GOLD, require_outcome=False).ok
    assert not verify_rows(rows, GOLD, require_outcome=True).ok


# ---------------------------------------------------------------------------
# 3. Stage-only executes zero legs by construction.
# ---------------------------------------------------------------------------
def test_stage_only_never_launches_a_child_even_with_an_instant_fetch(tmp_path):
    """The integration test for the defect a tiny --deadline could not prevent.

    ``fetch_fn`` returns immediately, which is what a warm paper cache looks
    like. Under ``--deadline 0.0003`` that is precisely when the budget is not
    yet spent and the first child is spawned. ``child_fn`` here raises, so any
    launch at all fails the test loudly.
    """
    from t2pw.batch import runner
    from t2pw.batch.fetch import BatchPaper

    launched = []

    def _never(command, timeout):  # pragma: no cover - must never run
        launched.append(command)
        raise AssertionError(f"stage-only launched a leg: {command}")

    def _instant_fetch(_topics_text, **_kwargs):
        papers = [
            BatchPaper(
                paper_id=case.paper_id,
                title=case.title,
                requested_pathway=case.requested_pathway,
                requested_organism=case.requested_organism,
                topic=case.requested_pathway,
                query="",
                full_text="cached body text",
                slug=case.paper_id,
                eligibility={"outcome": EXPECTED_OUTCOME},
            )
            for case in GOLD
        ]
        return papers, []

    out = tmp_path / "runs"
    code = runner.run_batch(
        topics_path=tmp_path / "topics.txt",
        out=out,
        modes="strict,research",
        fresh=True,
        stage_only=True,
        fetch_fn=_instant_fetch,
        child_fn=_never,
        data_dir=tmp_path / "data",
        echo=False,
        # A generous deadline: stage-only must not depend on it at all.
        deadline_hours=10.0,
    )
    assert code == 0
    assert launched == []

    run_dir = next(p for p in out.iterdir() if p.is_dir())
    assert (run_dir / "plan.json").is_file()
    assert (run_dir / "skipped.json").is_file()
    assert (run_dir / "SUMMARY.txt").is_file()
    # Zero manifest rows and zero leg directories.
    assert not (run_dir / "manifest.jsonl").exists()
    assert [d.name for d in (run_dir / "papers").iterdir() if d.is_dir()]
    leg_dirs = [
        child
        for paper in (run_dir / "papers").iterdir()
        if paper.is_dir()
        for child in paper.iterdir()
        if child.is_dir() and child.name in MODES
    ]
    assert leg_dirs == []

    # And the staged plan is a valid acceptance plan.
    assert verify_plan(run_dir, GOLD).ok


def test_the_cli_exposes_stage_only():
    from scripts import batch_run  # noqa: PLC0415 - imported for its parser only

    args = batch_run.build_parser().parse_args(["--stage-only"])
    assert args.stage_only is True
    assert batch_run.build_parser().parse_args([]).stage_only is False


def test_no_staging_instruction_still_recommends_a_tiny_deadline():
    """The deadline trick must be gone from every place a human reads."""

    targets = [
        ROOT / "docs" / "scientific_benchmark.md",
        ROOT / "scripts" / "bench_acceptance.py",
        ROOT / "bench" / "pinned_topics.txt",
        ROOT / "src" / "t2pw" / "bench" / "goldset.py",
    ]
    offenders = []
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for marker in ("--deadline 0.0003", "--deadline 1e-9"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, "staging must be --stage-only, not a deadline: " + "; ".join(offenders)


# ---------------------------------------------------------------------------
# 4. Negative controls are not extraction blockers.
# ---------------------------------------------------------------------------
def test_an_empty_negative_control_is_correct_and_an_empty_relevant_paper_is_a_blocker(tmp_path):
    """The paired regression. Same empty output, opposite meaning."""

    control = next(c for c in GOLD if c.is_negative_control)
    relevant = next(c for c in GOLD if c.mechanistic_relevance != "context_only")

    run_dir = tmp_path / "2026-01-01_0000"
    (run_dir / "papers").mkdir(parents=True)
    rows = [
        {
            "paper_id": case.paper_id,
            "slug": case.paper_id,
            "mode": "research",
            "status": "fail",
            "stage": "stage1",
            "failure_kind": "contract",
            "issue_codes": ["processes_required"],
            "files": [],
        }
        for case in (control, relevant)
    ]
    (run_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )

    report = score_run(run_dir, GOLD)
    blocked = {p for b in report.blockers[BLOCKERS_EXTRACTION] for p in b.papers}
    assert control.paper_id not in blocked, (
        "declining to invent a pathway from a paper that contains none is the correct outcome"
    )
    assert relevant.paper_id in blocked, "an empty relevant paper IS an extraction blocker"


def test_a_negative_control_that_retains_reactions_is_reported_as_unsupported_retention(tmp_path):
    control = next(c for c in GOLD if c.is_negative_control)
    run_dir = tmp_path / "2026-01-01_0000"
    leg = run_dir / "papers" / control.paper_id / "strict"
    leg.mkdir(parents=True)
    (leg / "merged_payload.json").write_text(json.dumps(_payload(reactions=5)), encoding="utf-8")
    (run_dir / "manifest.jsonl").write_text(
        json.dumps(
            {"paper_id": control.paper_id, "slug": control.paper_id, "mode": "strict",
             "status": "fail", "stage": "post_pipeline", "failure_kind": "contract",
             "issue_codes": [], "files": []}
        )
        + "\n",
        encoding="utf-8",
    )

    report = score_run(run_dir, GOLD)
    # Reported as hallucinated retention...
    assert report.errors.totals["unsupported_reactions"] == 5
    assert control.paper_id in report.errors.papers_affected["unsupported_reactions"]
    # ...and still not as an extraction blocker.
    blocked = {p for b in report.blockers[BLOCKERS_EXTRACTION] for p in b.papers}
    assert control.paper_id not in blocked


# ---------------------------------------------------------------------------
# 5. Research output is not research correctness.
# ---------------------------------------------------------------------------
def test_a_deliverable_with_an_unevaluable_check_is_produced_but_not_confirmed(tmp_path):
    case = next(c for c in GOLD if c.mechanistic_relevance != "context_only")
    run_dir = tmp_path / "2026-01-01_0000"
    leg = run_dir / "papers" / case.paper_id / "research"
    leg.mkdir(parents=True)
    (leg / "merged_payload.json").write_text(json.dumps(_payload()), encoding="utf-8")
    (run_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "paper_id": case.paper_id,
                "slug": case.paper_id,
                "mode": "research",
                "status": "pass",
                "stage": "research_report",
                "failure_kind": "",
                "issue_codes": [],
                # The deliverable exists.
                "files": [{"name": f"papers/{case.paper_id}/research/research_pathway_report.txt",
                           "bytes": 100}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = score_run(run_dir, GOLD)
    produced = report.denominators[DENOM_RESEARCH_DELIVERABLE]
    confirmed = report.denominators[DENOM_RESEARCH_CONFIRMED]
    assert case.paper_id in produced.numerator_names, "the report was produced"
    assert case.paper_id not in confirmed.numerator_names, (
        "no admission report on disk means the reintroduction check never ran, "
        "so the science is not confirmed"
    )
    assert produced.denominator == confirmed.denominator == 1


def test_the_two_research_rates_are_reported_separately():
    text = render_text(score_run(ROOT / "runs" / "2026-07-28_2122", GOLD))
    assert "RESEARCH DELIVERABLE PRODUCED" in text
    assert "RESEARCH SEMANTICALLY CONFIRMED" in text


# ---------------------------------------------------------------------------
# 6. Identity wording.
# ---------------------------------------------------------------------------
def test_zero_placeholders_with_unresolved_rows_is_not_called_fully_resolved():
    case = next(c for c in GOLD if c.mechanistic_relevance != "context_only")
    payload = _payload()
    payload["entities"]["proteins"] = [{"name": "MenD"}, {"name": "MenF"}]
    report = validate_semantic_coverage(case, payload)
    census = report.identity_census
    assert census["placeholder"] == 0
    assert census["unresolved"] == 2

    summary = report.checks[CHECK_PLACEHOLDER_IDENTITY].summary
    assert "0 verified" in summary
    assert "2 unresolved" in summary
    assert "every protein" not in summary or "NOT" in summary


def test_a_genuinely_resolved_payload_may_say_so():
    case = next(c for c in GOLD if c.mechanistic_relevance != "context_only")
    payload = _payload()
    payload["entities"]["proteins"] = [
        {"name": "MenD", "uniprot": "P23966"},
        {"name": "MenF", "uniprot": "P38051"},
    ]
    report = validate_semantic_coverage(case, payload)
    assert report.identity_census["verified"] == 2
    assert report.identity_census["unresolved"] == 0
    assert "every protein row carries a real external identity" in (
        report.checks[CHECK_PLACEHOLDER_IDENTITY].summary
    )
