"""W6b: the result page says why a review_required run needs review (counts, unplaced
terms, omitted processes) from the result's own fields, and withholds the fields the
E2E-2 audit found misreporting."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from t2pw.service import organisms as organism_data
from t2pw.service.pathway import PathwayResult
from t2pw.web import views
from t2pw.web.app import create_app
from t2pw.web.jobs import JobStore, result_to_dict

# Synthetic, shaped like docs/webserver/evidence/E2E-2/job_result.json.
E2E2 = dict(
    status="pass",
    counts={"reactions": 0, "transports": 6, "proteins": 6, "compounds": 4,
            "pwml": {"compounds": 4, "proteins": 6, "reactions": 0, "edges": 18}},
    release_status={
        "status": "review_required", "completeness": 0.8,
        "missing_anchors": ["PKD2L1", "SCN2A"], "retrieval_attempts": 0,
        "expansion_blocked_reason": "BLOCKED-REASON-TEXT",
        "reasons": ["semantic_evaluation_failed:requested_pathway_anchors_present",
                    "omitted_from_pwml:1_processes+0_actors"]},
)


def _page(tmp_path, result):
    db, work = tmp_path / "jobs.sqlite", tmp_path / "work"
    app = create_app(db, work, "https://t2pw.test", rate_secret=b"s", rate_limit=100)
    c = TestClient(app, follow_redirects=False)
    org = organism_data.get_by_label("Homo sapiens")
    r = c.post("/submit", data={"mode": "name", "name_1": "taste", "organism_1": org.label,
                                "consent": "1"})
    sid = re.search(r"/job/([0-9a-f]{32})$", r.headers["location"]).group(1)
    st = JobStore(db)
    item = st.claim_next(3600)
    d = result_to_dict(PathwayResult())
    d.update(result)
    st.finish(item["id"], item["attempt"], d, 0, requeue=False)
    st.close()
    return c.get(f"/job/{sid}/0")


def test_e2e2_case_states_why(tmp_path):
    r = _page(tmp_path, E2E2)
    assert r.status_code == 200
    html = r.text
    assert "Why this needs review" in html
    assert "Reactions: <strong>0</strong>" in html and "Transports: <strong>6</strong>" in html
    assert "The PWML contains no reactions." in html and "found in this text" not in html
    assert "PKD2L1" in html and "SCN2A" in html and "did not place them" in html
    assert "1 process found in the sources left out of the PWML" in html
    # withheld: audit F4/F5
    assert "BLOCKED-REASON-TEXT" not in html and "0.8" not in html
    assert "retrieval_attempts" not in html and "Retrieval" not in html and "Completeness" not in html
    assert "semantic_evaluation_failed" not in html and "requested_pathway_anchors_present" not in html
    assert "See why this needs review below." in html  # detail line points at the card


def test_not_review_required_has_no_section(tmp_path):
    ok = dict(E2E2, release_status={"status": "release_ready", "missing_anchors": ["X1"]})
    assert "Why this needs review" not in _page(tmp_path, ok).text


@pytest.mark.parametrize("res", [
    {}, {"counts": None, "release_status": {"missing_anchors": "PKD2L1", "reasons": 7}},
    {"counts": {"reactions": "3", "pwml": 5, "proteins": True}, "release_status": []},
    {"release_status": {"missing_anchors": [1, None, {"a": 1}, "  "], "reasons": [None, 3]}},
])
def test_malformed_report_data_is_left_out(res):
    assert views.review_why(res) is None


def test_zero_missing_terms_and_plurals():
    w = views.review_why({"counts": {"reactions": 4},
                          "release_status": {"reasons": ["omitted_from_pwml:2_processes+1_actors"]}})
    assert w["anchors"] == [] and not w["no_reactions"] and w["counts"] == [{"label": "Reactions", "n": 4}]
    assert w["omitted"].startswith("2 processes and 1 actor found")


def test_report_text_is_escaped_and_paths_scrubbed(tmp_path):
    evil = dict(E2E2, release_status=dict(E2E2["release_status"],
                missing_anchors=["<script>alert(1)</script>", "C:/secret/dir/x.json"]))
    html = _page(tmp_path, evil).text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "secret" not in html and "[server path]" in html


def test_unknown_reactions_count_is_not_zero():
    w = views.review_why({"counts": {"transports": 3}})
    assert w["counts"] == [{"label": "Transports", "n": 3}] and not w["no_reactions"]


def test_huge_digit_run_in_omitted_token_cannot_raise():
    big = "9" * 5000
    assert views.review_why({"release_status": {"reasons": [f"omitted_from_pwml:{big}_processes+0_actors"]}}) is None
