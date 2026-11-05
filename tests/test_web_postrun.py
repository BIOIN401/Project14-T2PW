"""NEW ACCEPTANCE TEST (W7, new capability): the worker's post-final actions,
``t2pw.web.postrun`` and the hook in ``t2pw.web.worker``. Fake results and a fake notify:
no LLM, no network, no SMTP, no child process.
"""

from __future__ import annotations

import json
import logging

import pytest

from t2pw.pipeline.release_status import RELEASE_READY, REVIEW_REQUIRED, ReleaseStatus
from t2pw.rag.eligibility import RequestedScope
from t2pw.service.pathway import (
    KIND_LLM,
    STAGE_PWML_EXPORT,
    STAGE_STAGE1,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    PathwayRequest,
    PathwayResult,
)
from t2pw.web import curator_queue, postrun
from t2pw.web.jobs import JobStore, result_to_dict
from t2pw.web.postrun import PostRunConfig
from t2pw.web.views import job_url
from t2pw.web.worker import Worker

EMAIL, BASE = "someone@example.org", "https://t2pw.example.org"


class FakeNotify:
    def __init__(self, ret=True, exc=None):
        self.calls, self.ret, self.exc = [], ret, exc

    def __call__(self, to, url, *, outcome, base_url=None):
        self.calls.append((to, url, outcome, base_url))
        if self.exc:
            raise self.exc
        return self.ret


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


@pytest.fixture
def notify():
    return FakeNotify()


@pytest.fixture
def cfg(tmp_path, notify):
    return PostRunConfig(queue_dir=tmp_path / "queue", work_root=tmp_path / "work",
                         base_url=BASE, notify=notify)


def submit(store, n=1, *, email=EMAIL, consent=True):
    return store.submit([PathwayRequest(scope=RequestedScope(requested_pathway=f"p{i}"))
                         for i in range(n)], consent=consent, email=email)


def result(tmp_path, item_id, *, good=True, kind=""):
    if not good:
        return result_to_dict(PathwayResult(status=STATUS_FAIL, stage=STAGE_STAGE1,
                                            failure_kind=kind or KIND_LLM, message="x"))
    out = tmp_path / "work" / f"item-{item_id}" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pathway.pwml").write_text("<pwml/>", encoding="utf-8")
    (out / "final_mapped.json").write_text("{}", encoding="utf-8")
    return result_to_dict(PathwayResult(
        status=STATUS_PASS, stage=STAGE_PWML_EXPORT, pwml_path=out / "pathway.pwml",
        report_paths={"final_mapped.json": out / "final_mapped.json"},
        release_status=ReleaseStatus(status=RELEASE_READY, pipeline_executed=True,
                                     strict_gates_passed=True)))


def settle(store, tmp_path, cfg, *, good=True, requeue=False):
    """Claim the next item and finish it as the worker would; then run the hook."""
    item = store.claim_next(60)
    res = result(tmp_path, item["id"], good=good)
    assert store.finish(item["id"], item["attempt"], res, requeue=requeue)
    return item["id"], res, postrun.run(store, item["id"], cfg)


def test_final_qualifying_item_is_enqueued_and_single_item_job_is_notified(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): one post-final call enqueues the item for
    the curator and sends the one email, with a link under base_url."""
    sid = submit(store)
    iid, _, out = settle(store, tmp_path, cfg)
    assert out == {"enqueued": True, "notified": True}
    assert notify.calls == [(EMAIL, job_url(BASE, sid), "finished", BASE)]
    q = curator_queue.CuratorQueue(store.path, cfg.queue_dir)
    assert [r["item_id"] for r in q.list_pending()] == [iid] and q.verify(iid)


def test_notify_once_per_submission_only_after_all_items_final(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): a 3-item submission emails once, when
    the last item is final, not after the first or second."""
    sid = submit(store, 3)
    for n in (1, 2):
        assert settle(store, tmp_path, cfg)[2]["notified"] is None
        assert notify.calls == [], f"emailed after item {n}"
    assert settle(store, tmp_path, cfg)[2]["notified"] is True
    assert [c[1] for c in notify.calls] == [job_url(BASE, sid)]


def test_no_email_while_a_retry_is_pending(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): an item re-queued for a retry is not
    final: no email, no enqueue; the email comes when the retried item is final."""
    submit(store)
    _, _, out = settle(store, tmp_path, cfg, good=False, requeue=True)
    assert out == {"enqueued": False, "notified": None} and notify.calls == []
    _, _, out = settle(store, tmp_path, cfg)
    assert out["notified"] is True and len(notify.calls) == 1


def test_not_notified_again_after_restart_or_repeat(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): the claim row makes the email
    exactly-once: the hook run again, from a fresh store and config (a worker
    restart), sends nothing and enqueues nothing more."""
    submit(store)
    iid, _, _ = settle(store, tmp_path, cfg)
    assert postrun.run(store, iid, cfg)["notified"] is None
    store2 = JobStore(store.path)
    try:
        cfg2 = PostRunConfig(queue_dir=cfg.queue_dir, work_root=cfg.work_root, base_url=BASE, notify=notify)
        assert postrun.run(store2, iid, cfg2) == {"enqueued": True, "notified": None}
    finally:
        store2.close()
    assert len(notify.calls) == 1
    assert len(curator_queue.CuratorQueue(store.path, cfg.queue_dir).list_entries()) == 1


def test_recovery_at_worker_start_settles_a_crashed_final_item(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): an item the worker died on, finished as
    a crash at the next start with no retries left, triggers the email once; a second
    start does not repeat it."""
    sid = submit(store)
    store.claim_next(60)
    for _ in range(2):
        w = Worker(store, tmp_path / "work", max_retries=0, postrun_cfg=cfg)
        w.start()
        w.close()
    assert [c[1:3] for c in notify.calls] == [(job_url(BASE, sid), "failed")]


def test_outcome_failed_when_no_item_passed(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): all items failed -> outcome 'failed';
    any item passing -> 'finished'."""
    submit(store, 2)
    settle(store, tmp_path, cfg, good=False)
    settle(store, tmp_path, cfg, good=False)
    assert notify.calls[0][2] == "failed"
    submit(store, 2)
    settle(store, tmp_path, cfg, good=False)
    settle(store, tmp_path, cfg, good=True)
    assert notify.calls[1][2] == "finished"


@pytest.mark.parametrize("email,base", [(None, BASE), (EMAIL, None)], ids=["no_email", "no_base_url"])
def test_no_email_without_address_or_base_url(store, tmp_path, cfg, notify, email, base):
    """NEW ACCEPTANCE TEST (W7, new capability): no address or no safe base link means no
    notify call (and no claim row); the curator enqueue still happens."""
    submit(store, email=email)
    cfg.base_url = base
    out = settle(store, tmp_path, cfg)[2]
    assert out == {"enqueued": True, "notified": None} and notify.calls == []


def test_unconsented_review_required_item_still_gets_its_email_but_no_queue(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): consent controls only the curator queue,
    not the job email."""
    submit(store, consent=False)
    out = settle(store, tmp_path, cfg)[2]
    assert out == {"enqueued": False, "notified": True} and not cfg.queue_dir.exists()


@pytest.mark.parametrize("ret,exc,state", [
    (False, None, "not_sent"), (True, None, "sent"), (False, RuntimeError(EMAIL), "not_sent")])
def test_a_failed_or_raising_send_is_recorded_and_not_repeated(store, tmp_path, cfg, caplog, ret, exc, state):
    """NEW ACCEPTANCE TEST (W7, new capability): the send result is recorded; a failed
    send raises nothing, is not retried, and the log holds neither the address nor the
    exception text."""
    cfg.notify = FakeNotify(ret=ret, exc=exc)
    submit(store)
    with caplog.at_level(logging.DEBUG):
        iid, _, _ = settle(store, tmp_path, cfg)
        postrun.run(store, iid, cfg)
    assert len(cfg.notify.calls) == 1
    (got,) = store._db.execute("SELECT state FROM web_notifications").fetchone()
    assert got == state and EMAIL not in caplog.text


def test_hook_failure_changes_nothing_and_does_not_crash_the_worker(store, tmp_path, cfg, notify, monkeypatch, caplog):
    """NEW ACCEPTANCE TEST (W7, new capability): enqueue raising, the store read raising
    and notify raising are all absorbed: Worker._finish returns normally and the item's
    persisted result is byte-identical to a run without the hook."""
    submit(store)
    item = store.claim_next(60)
    res = result(tmp_path, item["id"])
    w = Worker(store, tmp_path / "work", max_retries=0, postrun_cfg=cfg)

    def boom(*a, **k):
        raise OSError("disk full " + EMAIL)

    monkeypatch.setattr(curator_queue.CuratorQueue, "enqueue", boom)
    cfg.notify = FakeNotify(exc=ValueError(EMAIL))
    with caplog.at_level(logging.DEBUG):
        w._finish(item["id"], item["attempt"], res)
    got = store.item(item["id"])
    assert got["state"] == "finished" and got["result"] == json.loads(json.dumps(res))
    assert got["status"] == STATUS_PASS and EMAIL not in caplog.text and "disk full" not in caplog.text
    assert len(cfg.notify.calls) == 1  # the email step ran despite the enqueue failure

    monkeypatch.setattr(store, "item", boom)  # even the read failing must not escape
    assert postrun.run(store, item["id"], cfg) == {"enqueued": False, "notified": None}


def test_worker_finish_runs_the_hook_for_final_not_for_requeued(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): Worker._finish with the real retry
    policy: an LLM failure with retries left is re-queued (no hook effect); the final
    attempt enqueues and notifies once."""
    submit(store)
    w = Worker(store, tmp_path / "work", max_retries=1, postrun_cfg=cfg)
    item = store.claim_next(3600)
    w._finish(item["id"], item["attempt"], result(tmp_path, item["id"], good=False))
    assert store.item(item["id"])["state"] == "queued" and notify.calls == []
    item = store.claim_next(3600)
    w._finish(item["id"], item["attempt"], result(tmp_path, item["id"]))
    assert store.item(item["id"])["state"] == "finished" and len(notify.calls) == 1
    assert curator_queue.CuratorQueue(store.path, cfg.queue_dir).get(item["id"]) is not None


def test_review_required_result_is_not_enqueued_by_the_hook(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability): a review_required result is final and
    emailed but never copied to the curator queue."""
    submit(store)
    item = store.claim_next(60)
    res = result(tmp_path, item["id"])
    res["release_status"]["status"] = REVIEW_REQUIRED
    store.finish(item["id"], item["attempt"], res)
    assert postrun.run(store, item["id"], cfg) == {"enqueued": False, "notified": True}
    assert not cfg.queue_dir.exists()


def test_config_from_env(tmp_path):
    """NEW ACCEPTANCE TEST (W7, new capability): T2PW_BASE_URL and T2PW_CURATOR_QUEUE_DIR
    configure the worker; the default queue sits beside the jobs DB."""
    c = PostRunConfig.from_env(tmp_path / "j.sqlite", env={"T2PW_BASE_URL": " https://x.org "})
    assert c.base_url == "https://x.org" and c.queue_dir == tmp_path / "curator_queue"
    c = PostRunConfig.from_env(tmp_path / "j.sqlite", env={"T2PW_CURATOR_QUEUE_DIR": str(tmp_path / "q")})
    assert c.base_url is None and c.queue_dir == tmp_path / "q"


def test_startup_sweep_enqueues_an_item_the_hook_missed_and_sends_no_email(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability; review N2): the process died between
    store.finish and the hook. At the next start the curator enqueue happens exactly once
    (a second start adds nothing) and no email is sent."""
    submit(store)
    item = store.claim_next(60)
    assert store.finish(item["id"], item["attempt"], result(tmp_path, item["id"]))  # hook skipped
    q = curator_queue.CuratorQueue(store.path, cfg.queue_dir)
    assert q.list_entries() == []
    for _ in range(2):
        w = Worker(store, tmp_path / "work", postrun_cfg=cfg)
        assert w.start() == []
        w.close()
    assert [r["item_id"] for r in q.list_entries()] == [item["id"]] and q.verify(item["id"])
    assert notify.calls == []


def test_startup_recovery_with_retries_left_does_neither(store, tmp_path, cfg, notify):
    """NEW ACCEPTANCE TEST (W7, new capability; review N5): a crashed running item that the
    retry policy re-queues is not final: start() enqueues nothing and sends no email."""
    submit(store)
    item = store.claim_next(60)
    w = Worker(store, tmp_path / "work", max_retries=2, postrun_cfg=cfg)
    assert w.start() == [item["id"]]
    w.close()
    assert store.item(item["id"])["state"] == "queued"
    assert notify.calls == [] and not cfg.queue_dir.exists()


def test_email_link_is_views_job_url_and_the_site_serves_it(store, tmp_path, cfg, notify):
    """NEW (W7b): the e-mailed link is exactly views.job_url and the W6 app serves it."""
    from urllib.parse import urlsplit

    from fastapi.testclient import TestClient

    from t2pw.web.app import create_app

    sid = submit(store)
    settle(store, tmp_path, cfg)
    link = notify.calls[0][1]
    assert link == job_url(BASE, sid)
    app = create_app(store.path, tmp_path / "work2", BASE, rate_secret=b"k")
    r = TestClient(app, follow_redirects=False).get(urlsplit(link).path)
    assert r.status_code == 200 or (r.status_code == 303 and r.headers["location"] == f"/job/{sid}")


def test_postrun_import_does_not_pull_fastapi():
    """NEW (W7b): importing postrun (and so views.job_url) must not import FastAPI."""
    import subprocess
    import sys

    code = "import sys, t2pw.web.postrun; sys.exit(1 if any(m == 'fastapi' or m.startswith('fastapi.') for m in sys.modules) else 0)"
    assert subprocess.run([sys.executable, "-c", code], timeout=120).returncode == 0
