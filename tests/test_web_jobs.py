"""NEW ACCEPTANCE TEST (W3, new capability): the SQLite job store, ``t2pw.web.jobs``.

No process is spawned here; the worker has its own file (``test_web_worker.py``).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest

from t2pw.pipeline.release_status import RELEASE_READY, ReleaseStatus
from t2pw.rag.eligibility import RequestedScope
from t2pw.service.pathway import (
    KIND_CRASH,
    STATUS_ERROR,
    STATUS_PASS,
    PathwayRequest,
    PathwayResult,
)
from t2pw.web import jobs
from t2pw.web.jobs import JobStore, QueueFull


def _req(name: str = "serine biosynthesis") -> PathwayRequest:
    return PathwayRequest(scope=RequestedScope(requested_pathway=name,
                                               requested_organism="Homo sapiens"))


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def test_submit_accepts_1_to_5_and_rejects_0_and_6(store):
    """NEW ACCEPTANCE TEST (W3, new capability): 1-5 items per submission."""
    for n in (1, 5):
        sid = store.submit([_req(f"p{i}") for i in range(n)])
        sub = store.submission(sid)
        assert [it["position"] for it in sub["items"]] == list(range(n))
        assert all(it["state"] == jobs.QUEUED and it["attempt"] == 0 for it in sub["items"])
    for n in (0, 6):
        with pytest.raises(ValueError):
            store.submit([_req()] * n)
    with pytest.raises(TypeError):
        store.submit(["not a request"])


def test_consent_defaults_on_and_email_is_stored(store):
    """NEW ACCEPTANCE TEST (W3, new capability): consent default on; email optional."""
    a = store.submission(store.submit([_req()]))
    assert a["consent"] is True and a["email"] is None
    b = store.submission(store.submit([_req()], consent=False, email="x@example.org"))
    assert b["consent"] is False and b["email"] == "x@example.org"


def test_queue_cap_rejects_whole_submission_and_frees_on_finish(store):
    """NEW ACCEPTANCE TEST (W3, new capability): total-queue cap, all or nothing."""
    assert jobs.MAX_QUEUED_ITEMS == 20
    for _ in range(4):
        store.submit([_req()] * 5)
    with pytest.raises(QueueFull):
        store.submit([_req()])
    (count,) = store._db.execute("SELECT COUNT(*) FROM items").fetchone()
    assert count == 20  # nothing partial was inserted
    # running items still count against the cap; finished ones do not
    item = store.claim_next(60)
    with pytest.raises(QueueFull):
        store.submit([_req()])
    store.finish(item["id"], item["attempt"], jobs.result_to_dict(PathwayResult()))
    store.submit([_req()])


def test_state_transitions_and_result_persisted(store):
    """NEW ACCEPTANCE TEST (W3, new capability): queued -> running -> finished, with
    the deadline set at dequeue and every PathwayResult field persisted."""
    before = time.time()
    sid = store.submit([_req("a"), _req("b")])
    first = store.claim_next(100)
    assert first["state"] == jobs.RUNNING and first["position"] == 0
    assert first["attempt"] == 1
    assert first["deadline"] == pytest.approx(first["started_at"] + 100)
    assert first["started_at"] >= before
    assert [a["attempt"] for a in first["attempts"]] == [1]
    assert first["attempts"][0]["deadline"] == first["deadline"]

    rs = ReleaseStatus(status=RELEASE_READY, pipeline_executed=True, strict_gates_passed=True)
    res = PathwayResult(status=STATUS_PASS, stage="pwml_export", message="ok",
                        warnings=["w"], issue_codes=["c"], counts={"reactions": 3},
                        release_status=rs, pwml_path=Path("o/pathway.pwml"),
                        report_paths={"pwml_qa.json": Path("o/pwml_qa.json")},
                        attempt_log=[{"attempt": 1}])
    assert not store.finish(first["id"], 2, jobs.result_to_dict(res))  # stale attempt
    assert store.finish(first["id"], 1, jobs.result_to_dict(res), exit_code=0)
    assert not store.finish(first["id"], 1, jobs.result_to_dict(res))  # already finished

    done = store.item(first["id"])
    assert done["state"] == jobs.FINISHED and done["status"] == STATUS_PASS
    r = done["result"]
    assert set(r) == {f.name for f in dataclasses.fields(PathwayResult)}
    assert r["release_status"] == rs.to_dict()
    assert r["pwml_path"] == str(Path("o/pathway.pwml"))
    assert r["counts"] == {"reactions": 3} and r["warnings"] == ["w"]
    assert done["attempts"][0]["exit_code"] == 0
    assert done["attempts"][0]["status"] == STATUS_PASS

    second = store.claim_next(100)
    assert second["position"] == 1 and second["submission_id"] == sid
    assert store.claim_next(100) is None


def test_request_round_trips_every_field(store):
    """NEW ACCEPTANCE TEST (W3, new capability): the stored request is the submitted one."""
    req = PathwayRequest(texts=("t1", "t2"), pdf_paths=(Path("a.pdf"),), pmids=("123",),
                         scope=RequestedScope("glycolysis", "Escherichia coli", True),
                         extraction_focus="focus", export_mode="research",
                         out_dir=Path("out"), deadline=5.0)
    assert set(jobs.request_to_dict(req)) == {f.name for f in dataclasses.fields(PathwayRequest)}
    sid = store.submit([req])
    assert store.submission(sid)["items"][0]["request"] == req


def test_recover_running_marks_orphan_as_crash(tmp_path):
    """NEW ACCEPTANCE TEST (W3, new capability): an item left running by a dead worker
    is finished as error/crash on recovery; queued items are untouched."""
    db = tmp_path / "jobs.sqlite"
    s1 = JobStore(db)
    s1.submit([_req("a"), _req("b")])
    orphan = s1.claim_next(100)
    s1.close()  # the "crash"

    s2 = JobStore(db)
    assert s2.recover_running() == [orphan["id"]]
    item = s2.item(orphan["id"])
    assert item["state"] == jobs.FINISHED
    assert item["status"] == STATUS_ERROR and item["failure_kind"] == KIND_CRASH
    assert item["attempts"][0]["failure_kind"] == KIND_CRASH
    assert item["attempts"][0]["finished_at"] is not None
    assert s2.recover_running() == []
    assert s2.claim_next(100)["position"] == 1  # the queued one is still there
    s2.close()


def test_concurrent_claims_exactly_one_wins(tmp_path):
    """NEW ACCEPTANCE TEST (W3, new capability, review F9): two connections racing for
    the only queued item: exactly one gets it, with one attempts row."""
    import threading

    db = tmp_path / "jobs.sqlite"
    seed = JobStore(db)
    seed.submit([_req()])
    seed.close()
    barrier, got = threading.Barrier(2), [None, None]

    def claim(i):  # one connection per thread, as two processes would have
        s = JobStore(db)
        barrier.wait()
        got[i] = s.claim_next(100)
        s.close()

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    winners = [g for g in got if g is not None]
    assert len(winners) == 1
    check = JobStore(db)
    assert len(check.item(winners[0]["id"])["attempts"]) == 1
    check.close()
