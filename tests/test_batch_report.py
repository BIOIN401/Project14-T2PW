"""Tests for ``t2pw.batch.report`` -- the human-facing side of an overnight run.

These tests care about two things a formatting module can silently get wrong.
First, triage correctness: the whole value of running every paper twice is that
the *pair* of outcomes classifies the failure, so a paper landing in the wrong
bucket sends the morning's debugging at the wrong layer. Second, survivability:
this report is often the only evidence left of a run that died at 3am, so a
truncated manifest line, a missing field, or a completely empty manifest must
degrade into a readable report rather than a traceback.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch.report import (  # noqa: E402
    CLASS_BROKEN,
    CLASS_FORMAT_BLOCKED,
    CLASS_OK,
    CLASS_RESEARCH_DEFECT,
    DEFECT_MARKER,
    WARNING_MARKER,
    WIDTH,
    build_failures_by_code,
    build_summary,
    group_papers,
    load_manifest,
    status_line,
)


# ---------------------------------------------------------------------------
# Row builders.
# ---------------------------------------------------------------------------
def _row(
    paper_id: str,
    mode: str,
    status: str,
    **overrides: Any,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "paper_id": paper_id,
        "slug": f"{paper_id.lower()}-slug",
        "title": f"{paper_id} pathway study",
        "source_uri": f"https://example.org/{paper_id}",
        "topic": "lipid biosynthesis",
        "mode": mode,
        "status": status,
        "stage": "complete" if status == "pass" else "pwml-export",
        "seconds": 12.5,
        "counts": {"reactions": 7, "entities": 19},
        "files": [f"{paper_id}/{mode}/report.txt"],
    }
    row.update(overrides)
    return row


def _pair(
    paper_id: str,
    strict_status: str,
    research_status: str,
    *,
    strict_codes: List[str] | None = None,
    research_codes: List[str] | None = None,
) -> List[Dict[str, Any]]:
    strict_extra: Dict[str, Any] = {}
    research_extra: Dict[str, Any] = {}
    if strict_status != "pass":
        strict_extra = {
            "failure_kind": "format-rule",
            "message": f"{paper_id} strict failed",
            "issue_codes": strict_codes or ["PW_FORMAT_COMPOUND_NAME"],
        }
    if research_status != "pass":
        research_extra = {
            "failure_kind": "crash",
            "message": f"{paper_id} research failed",
            "issue_codes": research_codes or ["T2PW_RESEARCH_CRASH"],
        }
    return [
        _row(paper_id, "strict", strict_status, **strict_extra),
        _row(paper_id, "research", research_status, **research_extra),
    ]


def _four_class_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows += _pair("PMC_OK", "pass", "pass")
    rows += _pair("PMC_BLOCKED", "fail", "pass")
    rows += _pair("PMC_BROKEN", "fail", "fail")
    rows += _pair("PMC_DEFECT", "pass", "fail")
    return rows


def _classes(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    return {paper.label: paper.triage_class for paper in group_papers(rows)}


# ---------------------------------------------------------------------------
# 1. Triage classification.
# ---------------------------------------------------------------------------
def test_four_triage_classes_land_in_the_right_bucket() -> None:
    classes = _classes(_four_class_rows())
    assert classes["PMC_OK"] == CLASS_OK
    assert classes["PMC_BLOCKED"] == CLASS_FORMAT_BLOCKED
    assert classes["PMC_BROKEN"] == CLASS_BROKEN
    assert classes["PMC_DEFECT"] == CLASS_RESEARCH_DEFECT


def test_triage_matrix_reports_each_class_with_its_paper() -> None:
    summary = build_summary(_four_class_rows())
    assert "TRIAGE MATRIX" in summary
    for name in (CLASS_OK, CLASS_FORMAT_BLOCKED, CLASS_BROKEN, CLASS_RESEARCH_DEFECT):
        assert name in summary
    matrix = summary.split("TRIAGE MATRIX", 1)[1].split("PER-PAPER BREAKDOWN", 1)[0]
    for paper_id in ("PMC_OK", "PMC_BLOCKED", "PMC_BROKEN", "PMC_DEFECT"):
        assert paper_id in matrix


def test_timeout_counts_as_a_failure_and_is_tallied_separately() -> None:
    rows = _pair("PMC_SLOW", "timeout", "pass")
    assert _classes(rows)["PMC_SLOW"] == CLASS_FORMAT_BLOCKED
    summary = build_summary(rows)
    assert "timeouts         : 1" in summary


def test_a_missing_mode_is_incomplete_not_a_pass() -> None:
    rows = [_row("PMC_HALF", "strict", "pass")]
    assert _classes(rows)["PMC_HALF"] == "incomplete"


# ---------------------------------------------------------------------------
# 2. Research-mode failures are defects, listed first and loudest.
# ---------------------------------------------------------------------------
def test_research_failure_is_listed_first_and_labelled_a_defect() -> None:
    summary = build_summary(_four_class_rows())
    assert DEFECT_MARKER in summary
    defect_at = summary.index(DEFECT_MARKER)

    # Loudest: the defect banner precedes the per-class tallies, the per-paper
    # breakdown and the fix list.
    assert defect_at < summary.index("PER-PAPER BREAKDOWN")
    assert defect_at < summary.index("WHAT TO FIX FIRST")
    assert defect_at < summary.index(CLASS_OK + " ")

    # First: inside the matrix, the research failures come before the ok paper.
    matrix = summary.split("TRIAGE MATRIX", 1)[1].split("PER-PAPER BREAKDOWN", 1)[0]
    assert matrix.index("PMC_DEFECT") < matrix.index("PMC_OK")
    assert "fail-open" in matrix

    # And "fix first" means literally first in the fix list.
    fixes = summary.split("WHAT TO FIX FIRST", 1)[1]
    assert fixes.index("PMC_DEFECT") < fixes.index("PMC_BLOCKED")


def test_both_modes_failing_is_broken_and_still_flagged_as_a_research_defect() -> None:
    papers = {p.label: p for p in group_papers(_pair("PMC_BROKEN", "fail", "fail"))}
    broken = papers["PMC_BROKEN"]
    assert broken.triage_class == CLASS_BROKEN
    assert broken.research_defect is True

    summary = build_summary(_pair("PMC_BROKEN", "fail", "fail"))
    banner = summary.split(DEFECT_MARKER, 1)[1].split("PER-PAPER BREAKDOWN", 1)[0]
    assert "PMC_BROKEN" in banner


def test_no_research_failures_says_so_explicitly() -> None:
    summary = build_summary(_pair("PMC_OK", "pass", "pass"))
    assert DEFECT_MARKER not in summary
    assert "research-mode defects: NONE" in summary


# ---------------------------------------------------------------------------
# 3. Per-paper breakdown.
# ---------------------------------------------------------------------------
def test_per_paper_breakdown_names_every_file() -> None:
    files_strict = [
        {"path": "PMC1/strict/pathway.pwml", "bytes": 20480},
        {"path": "PMC1/strict/mapping_report.json", "bytes": 512},
    ]
    files_research = [
        "PMC1/research/report.txt",
        "PMC1/research/citations.csv",
        {"path": "PMC1/research/tiers.json"},
    ]
    rows = [
        _row("PMC1", "strict", "pass", files=files_strict),
        _row("PMC1", "research", "pass", files=files_research),
    ]
    summary = build_summary(rows)
    breakdown = summary.split("PER-PAPER BREAKDOWN", 1)[1]
    for entry in files_strict:
        assert entry["path"] in breakdown
    assert "PMC1/research/report.txt" in breakdown
    assert "PMC1/research/citations.csv" in breakdown
    assert "PMC1/research/tiers.json" in breakdown
    # Sizes are rendered when provided.
    assert "20.0 kB" in breakdown
    assert "512 B" in breakdown


def test_per_paper_breakdown_shows_identity_counts_and_failure_detail() -> None:
    rows = [
        _row(
            "PMC2",
            "strict",
            "fail",
            stage="stage8-export",
            failure_kind="format-rule",
            message="Compound name beta-hydroxymyristoyl-ACP violates FORMAT rule 4",
            issue_codes=["PW_FORMAT_COMPOUND_NAME", "PW_FORMAT_ARROW"],
        ),
        _row(
            "PMC2",
            "research",
            "pass",
            counts={"reactions": 11, "entities": 30, "tiers": {"A": 4, "D": 2}},
        ),
    ]
    summary = build_summary(rows)
    breakdown = summary.split("PER-PAPER BREAKDOWN", 1)[1]
    assert "PMC2 pathway study" in breakdown
    assert "https://example.org/PMC2" in breakdown
    assert "lipid biosynthesis" in breakdown
    assert "pmc2-slug" in breakdown
    assert "stage8-export" in breakdown
    assert "format-rule" in breakdown
    assert "FORMAT rule 4" in breakdown
    assert "PW_FORMAT_COMPOUND_NAME" in breakdown
    assert "tiers.A=4" in breakdown
    assert "tiers.D=2" in breakdown


def test_every_paper_gets_its_own_block() -> None:
    summary = build_summary(_four_class_rows())
    breakdown = summary.split("PER-PAPER BREAKDOWN", 1)[1]
    for position in ("[1]", "[2]", "[3]", "[4]"):
        assert position in breakdown


def test_report_stays_within_the_column_budget() -> None:
    long_message = "a very long failure message " * 12
    rows = [
        _row(
            "PMC_WIDE",
            "strict",
            "fail",
            title="An extremely long paper title " * 6,
            source_uri="https://example.org/" + "x" * 180,
            message=long_message,
            issue_codes=["PW_FORMAT_" + "Y" * 140],
            files=[{"path": "PMC_WIDE/strict/" + "z" * 160, "bytes": 3}],
        ),
        _row("PMC_WIDE", "research", "pass"),
    ]
    for text in (build_summary(rows), build_failures_by_code(rows)):
        for line in text.splitlines():
            assert len(line) <= WIDTH, line


# ---------------------------------------------------------------------------
# 4. failures_by_code.txt -- the fix-list.
# ---------------------------------------------------------------------------
def test_failures_by_code_ranks_by_paper_count() -> None:
    rows: List[Dict[str, Any]] = []
    # Three papers hit CODE_WIDE once each.
    for paper_id in ("PMC_A", "PMC_B", "PMC_C"):
        rows += _pair(paper_id, "fail", "pass", strict_codes=["CODE_WIDE"])
    # One paper hits CODE_DEEP in both modes: more runs, fewer papers.
    rows += _pair(
        "PMC_D",
        "fail",
        "fail",
        strict_codes=["CODE_DEEP"],
        research_codes=["CODE_DEEP"],
    )
    text = build_failures_by_code(rows)
    assert text.index("CODE_WIDE") < text.index("CODE_DEEP")

    wide = text.split("CODE_WIDE", 1)[1].split("CODE_DEEP", 1)[0]
    assert "papers: 3" in wide
    assert "runs: 3" in wide
    assert "modes: strict" in wide
    for paper_id in ("PMC_A", "PMC_B", "PMC_C"):
        assert paper_id in wide

    deep = text.split("CODE_DEEP", 1)[1]
    assert "papers: 1" in deep
    assert "runs: 2" in deep
    assert "research, strict" in deep


def test_failures_without_issue_codes_are_grouped_by_failure_kind() -> None:
    rows = [
        _row(
            "PMC_E",
            "strict",
            "fail",
            failure_kind="llm-refusal",
            message="model declined",
            issue_codes=[],
        ),
        _row("PMC_E", "research", "pass"),
        _row("PMC_F", "strict", "timeout", failure_kind="", message="", issue_codes=[]),
        _row("PMC_F", "research", "pass"),
    ]
    text = build_failures_by_code(rows)
    tail = text.split("FAILURES WITH NO ISSUE CODES", 1)[1]
    assert "llm-refusal" in tail
    assert "PMC_E" in tail
    # A timeout with no failure_kind still shows up, keyed by its status.
    assert "timeout" in tail
    assert "PMC_F" in tail


def test_failures_by_code_is_sane_with_no_failures() -> None:
    text = build_failures_by_code(_pair("PMC_OK", "pass", "pass"))
    assert "failing runs: 0" in text
    assert "(no issue codes were reported)" in text
    assert "FAILURES WITH NO ISSUE CODES" in text


# ---------------------------------------------------------------------------
# 5. Tolerance: malformed and empty manifests.
# ---------------------------------------------------------------------------
def test_malformed_manifest_line_is_skipped_and_counted_unreadable(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    good = _row("PMC_GOOD", "strict", "pass")
    lines = [
        json.dumps(good),
        '{"paper_id": "PMC_TRUNC", "mode": "resear',  # half-written line
        "[1, 2, 3]",  # valid JSON, wrong shape
        "",  # blank
        json.dumps(_row("PMC_GOOD", "research", "pass")),
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = load_manifest(manifest)
    assert len(rows) == 4  # blank dropped, two unreadable retained as markers
    papers = group_papers(rows)
    assert [p.label for p in papers] == ["PMC_GOOD"]

    summary = build_summary(rows)  # must not raise
    assert "unreadable: 2" in summary
    assert "PMC_GOOD" in summary
    assert "PMC_TRUNC" not in summary
    assert "2 unreadable" in status_line(rows, total_expected=1)


def test_rows_with_missing_fields_do_not_raise() -> None:
    rows: List[Dict[str, Any]] = [
        {},
        {"mode": "Strict PWML"},
        {"paper_id": "PMC_X", "mode": "Research (relaxed)", "status": "PASS"},
        {
            "paper_id": "PMC_X",
            "mode": "strict",
            "status": "fail",
            "counts": "not-a-dict",
            "files": "solo/file.txt",
            "issue_codes": "SINGLE_CODE",
            "seconds": "not-a-number",
        },
    ]
    summary = build_summary(rows)
    assert "PMC_X" in summary
    assert "SINGLE_CODE" in summary
    assert "solo/file.txt" in summary
    assert "n/a" in summary  # unparseable duration
    assert "SINGLE_CODE" in build_failures_by_code(rows)
    assert status_line(rows)


def test_non_ascii_entity_names_survive_formatting() -> None:
    rows = _pair("PMC_UTF", "fail", "pass")
    rows[0]["message"] = "beta-hydroxymyristoyl -> 3–oxo–acyl (β form)"
    summary = build_summary(rows)
    assert "β form" in summary


def test_load_manifest_handles_missing_and_empty_files(tmp_path) -> None:
    assert load_manifest(tmp_path / "nope.jsonl") == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_manifest(empty) == []


def test_empty_manifest_produces_a_sane_summary() -> None:
    summary = build_summary([], started_at="2026-07-27T21:00", finished_at="2026-07-28T02:00")
    assert "papers attempted : 0" in summary
    assert "2026-07-27T21:00" in summary
    assert "nothing was recorded" in summary
    # No crash, no bogus triage sections claiming success.
    assert DEFECT_MARKER not in summary
    assert summary.endswith("\n")
    assert build_failures_by_code([]).strip()
    assert status_line([]) == "0/0 done | strict 0 pass | research 0 pass"


# ---------------------------------------------------------------------------
# 6. status_line.
# ---------------------------------------------------------------------------
def test_status_line_reports_progress_failures_and_current_paper() -> None:
    rows: List[Dict[str, Any]] = []
    for paper_id in ("P1", "P2", "P3", "P4"):
        rows += _pair(paper_id, "pass", "pass")
    rows += _pair("P5", "fail", "pass")
    rows += _pair("P6", "timeout", "pass")

    line = status_line(rows, total_expected=10, running="PMC12648303")
    assert line.startswith("6/10 done")
    assert "strict 4 pass 2 fail" in line
    assert "research 6 pass" in line
    assert "1 timeout" in line
    assert line.endswith("running: PMC12648303")
    assert len(line.splitlines()) == 1


def test_status_line_shouts_about_research_defects() -> None:
    rows = _pair("P1", "pass", "fail")
    assert "1 RESEARCH DEFECT" in status_line(rows, total_expected=2)


# ---------------------------------------------------------------------------
# 7. A pass that produced nothing. Research mode returns a "RAG did not trigger"
#    note and passes -- correctly, since that is not a code defect -- but the
#    folder then has no report, no citations and no tiers. The pass/fail columns
#    cannot express that, so the warning has to be reported on its own.
# ---------------------------------------------------------------------------
_NO_RAG = "no research report: RAG did not trigger for this paper"


def test_a_pass_with_a_warning_is_labelled_not_hidden() -> None:
    rows = [
        _row("PMC_W", "strict", "pass"),
        _row("PMC_W", "research", "pass", warnings=[_NO_RAG], files=[]),
    ]

    papers = group_papers(rows)
    research = papers[0].runs["research"]
    assert research.passed is True  # still a pass: the pipeline really ran
    assert research.warned is True
    assert research.verdict == "PASS (no research deliverable)"
    assert research.warnings == [_NO_RAG]
    assert papers[0].research_defect is False  # a warning is not a defect

    summary = build_summary(rows)
    assert "PASS (no research deliverable)" in summary
    assert WARNING_MARKER in summary
    assert _NO_RAG in summary
    assert "pass, no deliv.  : 1" in summary
    # The clean-run wording must not survive alongside a warning.
    assert "research-mode defects: NONE" in summary  # true, and still said
    assert "1 NO DELIVERABLE" in status_line(rows, total_expected=2)


def test_a_clean_pass_says_nothing_about_deliverables() -> None:
    summary = build_summary(_pair("PMC_CLEAN", "pass", "pass"))
    assert WARNING_MARKER not in summary
    assert "pass, no deliv.  : 0" in summary
    assert "NO DELIVERABLE" not in status_line(_pair("PMC_CLEAN", "pass", "pass"))


def test_a_failed_artifact_write_is_named_at_the_file() -> None:
    """``write_artifacts`` records ``{"bytes": 0, "error": ...}`` and keeps going.
    Rendered as a plain zero-byte file, that is how a run with no .pwml on disk
    reads as a clean one."""

    rows = [
        _row(
            "PMC_DISK",
            "strict",
            "pass",
            files=[
                {"name": "papers/x/strict/pathway.pwml", "bytes": 0,
                 "error": "OSError: [Errno 28] No space left on device"},
                {"name": "papers/x/strict/notes.txt", "bytes": 12},
            ],
        ),
        _row("PMC_DISK", "research", "pass"),
    ]

    run = group_papers(rows)[0].runs["strict"]
    assert run.file_errors == [
        ("papers/x/strict/pathway.pwml", "OSError: [Errno 28] No space left on device")
    ]
    assert run.warned is True

    summary = build_summary(rows)
    assert "artifact writes  : 1 FAILED" in summary
    assert "WRITE FAILED" in summary
    assert "No space left on device" in summary
    assert WARNING_MARKER in summary


def test_retried_run_last_row_wins() -> None:
    rows = [
        _row("PMC_R", "strict", "fail", message="first attempt"),
        _row("PMC_R", "strict", "pass", stage="complete"),
        _row("PMC_R", "research", "pass"),
    ]
    assert _classes(rows)["PMC_R"] == CLASS_OK
    assert "first attempt" not in build_summary(rows)
