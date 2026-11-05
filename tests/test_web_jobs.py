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
    KIND_LLM,
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


# ── W4: retry support in the store ─────────────────────────────────────────


def _fail(kind: str = KIND_CRASH) -> dict:
    return jobs.result_to_dict(PathwayResult(status=STATUS_ERROR, failure_kind=kind, message="m"))


def test_requeue_keeps_the_attempt_and_clears_the_item_result(store):
    """NEW ACCEPTANCE TEST (W4, new capability): finish(requeue=True) closes the attempt
    row but puts the item back to queued with no item-level result; the next claim is
    attempt 2 with its own attempts row; a queued retry still counts against the cap."""
    store.submit([_req()])
    first = store.claim_next(100)
    assert store.finish(first["id"], 1, _fail(), exit_code=3, requeue=True)
    item = store.item(first["id"])
    assert item["state"] == jobs.QUEUED and item["attempt"] == 1
    assert item["status"] is None and item["failure_kind"] is None and item["result"] is None
    assert item["finished_at"] is None
    assert item["attempts"][0]["failure_kind"] == KIND_CRASH
    assert item["attempts"][0]["finished_at"] is not None and item["attempts"][0]["exit_code"] == 3
    assert not store.finish(first["id"], 1, _fail(), requeue=True)  # no longer running
    second = store.claim_next(100)
    assert second["id"] == first["id"] and second["attempt"] == 2
    assert [a["attempt"] for a in second["attempts"]] == [1, 2]
    for _ in range(3):
        store.submit([_req()] * 5)
    store.finish(second["id"], 2, _fail(), requeue=True)
    store.submit([_req()] * 4)  # 20 open: the requeued item is one of them
    with pytest.raises(QueueFull):
        store.submit([_req()])


def test_claim_shapes_the_deadline_by_the_remaining_wall_budget(store):
    """NEW ACCEPTANCE TEST (W4, new capability): each attempt gets a fresh absolute
    deadline at dequeue, capped by what is left of the per-pathway wall cap; time
    spent queued between attempts is not charged."""
    store.submit([_req()])
    a1 = store.claim_next(100, wall_cap_s=250)
    assert a1["deadline"] == pytest.approx(a1["started_at"] + 100)  # cap not binding
    store.finish(a1["id"], 1, _fail(), requeue=True)
    # pretend attempt 1 ran for 100 s, ended 1000 s ago (queue wait is not charged)
    store._db.execute("UPDATE attempts SET started_at = started_at - 1100, "
                      "finished_at = finished_at - 1000, deadline = deadline - 1000")
    a2 = store.claim_next(100, wall_cap_s=250)
    assert a2["attempt"] == 2 and a2["deadline"] == pytest.approx(a2["started_at"] + 100)
    store.finish(a2["id"], 2, _fail(), requeue=True)
    store._db.execute("UPDATE attempts SET started_at = started_at - 100 WHERE attempt = 2")
    assert store.wall_used(a1["id"]) == pytest.approx(100 + 100 + 0, abs=2)
    a3 = store.claim_next(100, wall_cap_s=250)
    assert a3["deadline"] - a3["started_at"] == pytest.approx(50, abs=2)  # 250 - 200 left


def test_wall_used_counts_an_open_attempt_to_its_deadline_at_most(store):
    """NEW ACCEPTANCE TEST (W4, new capability): an attempt left open by a dead worker
    is charged up to its own deadline, not for the whole downtime."""
    store.submit([_req()])
    item = store.claim_next(100)
    store._db.execute("UPDATE attempts SET started_at = started_at - 5000, deadline = started_at - 4900")
    assert store.wall_used(item["id"]) == pytest.approx(100)
    assert store.wall_used(item["id"], now=time.time() - 4950) == pytest.approx(50, abs=1)


def test_attempt_history_for_the_job_page(store):
    """NEW ACCEPTANCE TEST (W4, new capability): attempt_history lists number, start,
    end, status, failure_kind and whether each attempt was or will be retried."""
    store.submit([_req()])
    assert store.attempt_history(999) == []
    a1 = store.claim_next(100)
    h = store.attempt_history(a1["id"])
    assert [(r["attempt"], r["finished_at"], r["retried"], r["will_retry"]) for r in h] \
        == [(1, None, False, False)]                      # running: undecided
    store.finish(a1["id"], 1, _fail(KIND_LLM), requeue=True)
    h = store.attempt_history(a1["id"])
    assert (h[0]["status"], h[0]["failure_kind"]) == (STATUS_ERROR, KIND_LLM)
    assert h[0]["finished_at"] >= h[0]["started_at"]
    assert (h[0]["retried"], h[0]["will_retry"]) == (False, True)   # queued for attempt 2
    a2 = store.claim_next(100)
    store.finish(a2["id"], 2, _fail(KIND_CRASH))                    # final
    h = store.attempt_history(a1["id"])
    assert [(r["attempt"], r["failure_kind"], r["retried"], r["will_retry"]) for r in h] == [
        (1, KIND_LLM, True, False), (2, KIND_CRASH, False, False)]
    assert set(h[0]) >= {"attempt", "started_at", "finished_at", "status", "failure_kind",
                         "retried", "will_retry"}


def test_recover_running_requeues_when_the_policy_says_so(tmp_path):
    """NEW ACCEPTANCE TEST (W4, new capability): recover_running(should_retry) records the
    crash in the attempts row and re-queues the item only for a True answer."""
    db = tmp_path / "jobs.sqlite"
    s1 = JobStore(db)
    s1.submit([_req("a"), _req("b")])
    a, b = s1.claim_next(100), s1.claim_next(100)
    s1.close()
    s2 = JobStore(db)
    seen = []

    def policy(item_id, attempt, result):
        seen.append((item_id, attempt, result["failure_kind"]))
        return item_id == a["id"]

    assert s2.recover_running(policy) == [a["id"], b["id"]]
    assert seen == [(a["id"], 1, KIND_CRASH), (b["id"], 1, KIND_CRASH)]
    assert s2.item(a["id"])["state"] == jobs.QUEUED
    assert s2.item(a["id"])["attempts"][0]["failure_kind"] == KIND_CRASH
    assert s2.item(b["id"])["state"] == jobs.FINISHED
    s2.close()
