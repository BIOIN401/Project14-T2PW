"""NEW ACCEPTANCE TEST (W3, new capability): the job worker, ``t2pw.web.worker``.

The run function is one of the fakes below, imported by name in the child process
(``--run-fn test_web_worker:<fake>``), so no LLM, no network and no pipeline runs.
Every test that spawns a process asserts that each PID it learned of is dead before
it returns.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from t2pw.pipeline.release_status import RELEASE_READY, ReleaseStatus
from t2pw.rag.eligibility import RequestedScope
from t2pw.service.pathway import (
    KIND_CONTRACT,
    KIND_CRASH,
    KIND_LLM,
    KIND_TIMEOUT,
    STAGE_PWML_EXPORT,
    STAGE_STAGE1,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_TIMEOUT,
    PathwayRequest,
    PathwayResult,
)
from t2pw.web import jobs
from t2pw.web.jobs import JobStore
from t2pw.web import worker as worker_mod
from t2pw.web.retry import MAX_RETRIES
from t2pw.web.worker import Worker

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
MOD = Path(__file__).stem


# ── fakes, run in the child ────────────────────────────────────────────────


def fake_ok(req: PathwayRequest) -> PathwayResult:
    req.out_dir.mkdir(parents=True, exist_ok=True)
    pwml = req.out_dir / "pathway.pwml"
    pwml.write_text("<fake/>", encoding="utf-8")
    return PathwayResult(
        status=STATUS_PASS, stage=STAGE_PWML_EXPORT,
        release_status=ReleaseStatus(status=RELEASE_READY, pipeline_executed=True,
                                     strict_gates_passed=True),
        pwml_path=pwml,
        counts={"deadline": req.deadline, "out_dir": str(req.out_dir),
                "pathway": req.scope.requested_pathway})


def fake_hang(req: PathwayRequest) -> PathwayResult:
    """Starts a grandchild as its very first action, records both PIDs, never returns."""
    grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (req.out_dir.parent / "pids.txt").write_text(f"{os.getpid()} {grandchild.pid}")
    time.sleep(120)
    raise AssertionError("unreachable")


def fake_unserialisable(req: PathwayRequest) -> PathwayResult:
    res = fake_ok(req)
    res.counts["odd"] = object()
    res.attempt_log.append({"when": Path("x")})
    return res


def fake_result_then_hang(req: PathwayRequest) -> PathwayResult:
    """Returns a result, so it is written, but a non-daemon thread keeps the process
    alive past the deadline."""
    import threading
    threading.Thread(target=time.sleep, args=(120,)).start()
    return fake_ok(req)


def fake_raise(req: PathwayRequest) -> PathwayResult:
    raise RuntimeError("boom")


def fake_exit(req: PathwayRequest) -> PathwayResult:
    os._exit(3)


def fake_llm(req: PathwayRequest) -> PathwayResult:
    return PathwayResult(status=STATUS_FAIL, stage=STAGE_STAGE1, failure_kind=KIND_LLM,
                         message="empty LLM reply")


def fake_llm_then_ok(req: PathwayRequest) -> PathwayResult:
    """Fails with an LLM error on attempt 1 only (out_dir is .../attempt-<n>/out)."""
    return fake_llm(req) if req.out_dir.parent.name == "attempt-1" else fake_ok(req)


def fake_contract(req: PathwayRequest) -> PathwayResult:
    return PathwayResult(status=STATUS_FAIL, stage=STAGE_STAGE1, failure_kind=KIND_CONTRACT,
                         message="stage boundary contract violated")


# ── helpers ────────────────────────────────────────────────────────────────


def alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32")
        k32.OpenProcess.restype = ctypes.c_void_p
        h = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not h:
            return False
        try:
            return k32.WaitForSingleObject(ctypes.c_void_p(h), 0) == 0x102  # WAIT_TIMEOUT
        finally:
            k32.CloseHandle(ctypes.c_void_p(h))
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def wait_dead(pids, within: float = 10.0) -> list:
    end = time.time() + within
    while time.time() < end and any(alive(p) for p in pids):
        time.sleep(0.1)
    return [p for p in pids if alive(p)]


def force_kill(pids) -> None:
    """Last-resort cleanup of PIDs this test itself caused, only if an assert failed."""
    for p in pids:
        if alive(p):
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(p)], capture_output=True, check=False)
            else:
                os.kill(p, 9)


@pytest.fixture
def child_env(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(SRC_DIR), str(TESTS_DIR)]))


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def _req(name: str = "serine biosynthesis", **kw) -> PathwayRequest:
    return PathwayRequest(scope=RequestedScope(requested_pathway=name), **kw)


# Deadline tests use 10 s: a child imports this module (and pytest) before the fake
# runs, which measured 2-5 s per child on the dev machine under load.
# W4: the W3 tests assert the outcome of a single attempt, so they run with retries off
# (max_retries=0); the W4 tests below pass retry.MAX_RETRIES.
def _worker(store, tmp_path, fake, timeout_s=60.0, max_retries=0) -> Worker:
    w = Worker(store, tmp_path / "work", run_fn=f"{MOD}:{fake}", timeout_s=timeout_s,
               max_retries=max_retries)
    assert w.start() == []
    return w


# ── tests ──────────────────────────────────────────────────────────────────


def test_normal_run_persists_result(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W3, new capability): a run in a child process; its
    PathwayResult (release_status included) is persisted and the child is gone. A
    submitted out_dir is ignored: the worker always owns where the run writes."""
    w = _worker(store, tmp_path, "fake_ok")
    store.submit([_req(out_dir=tmp_path / "elsewhere")])
    item_id = w.run_one()
    w.close()
    item = store.item(item_id)
    child_pid = item["attempts"][0]["child_pid"]
    assert wait_dead([child_pid]) == []
    assert item["state"] == jobs.FINISHED and item["status"] == STATUS_PASS
    r = item["result"]
    assert r["release_status"]["status"] == RELEASE_READY
    assert Path(r["pwml_path"]).read_text(encoding="utf-8") == "<fake/>"
    assert r["counts"]["pathway"] == "serine biosynthesis"
    out_dir = tmp_path / "work" / f"item-{item_id}" / "attempt-1" / "out"
    assert r["counts"]["out_dir"] == str(out_dir)
    assert item["attempts"][0]["exit_code"] == 0
    assert not (tmp_path / "elsewhere").exists()


def test_unserialisable_result_values_keep_the_run(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W3, new capability, review F4): a non-JSON value in counts
    or attempt_log is stringified; the run stays a pass and keeps its PWML."""
    w = _worker(store, tmp_path, "fake_unserialisable")
    store.submit([_req()])
    item = store.item(w.run_one())
    w.close()
    assert wait_dead([item["attempts"][0]["child_pid"]]) == []
    assert item["status"] == STATUS_PASS
    r = item["result"]
    assert Path(r["pwml_path"]).read_text(encoding="utf-8") == "<fake/>"
    assert isinstance(r["counts"]["odd"], str)
    assert r["attempt_log"][-1]["when"] == str(Path("x"))


def test_result_written_before_deadline_kill_is_kept(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W3, new capability, review F8): a child that wrote its
    result but did not exit by the deadline is killed, and the result is kept."""
    w = _worker(store, tmp_path, "fake_result_then_hang", timeout_s=10.0)
    store.submit([_req()])
    item = store.item(w.run_one())
    w.close()
    assert wait_dead([item["attempts"][0]["child_pid"]]) == []
    assert item["status"] == STATUS_PASS and item["result"]["pwml_path"]


@pytest.mark.skipif(os.name != "nt", reason="Job Object containment is Windows-only")
def test_uncontainable_child_never_runs(store, tmp_path, child_env, monkeypatch):
    """NEW ACCEPTANCE TEST (W3, new capability, review F2): if the Job Object cannot be
    made, the suspended child is killed before it runs and the item is error/crash."""
    monkeypatch.setattr(worker_mod, "_new_job", lambda handle: None)
    w = _worker(store, tmp_path, "fake_ok")
    store.submit([_req()])
    item_id = w.run_one()
    w.close()
    item = store.item(item_id)
    assert item["status"] == STATUS_ERROR and item["failure_kind"] == KIND_CRASH
    assert "Job Object" in item["result"]["message"]
    assert not (tmp_path / "work" / f"item-{item_id}" / "attempt-1" / "out").exists()


@pytest.mark.skipif(os.name != "nt", reason="Job Object containment is Windows-only")
def test_exception_during_containment_kills_suspended_child(store, tmp_path, child_env,
                                                             monkeypatch):
    """NEW ACCEPTANCE TEST (W3, new capability, review round 2 item 9): an exception
    between Popen and the job assignment kills the suspended child; none is leaked."""
    import ctypes
    seen = []

    def boom(handle):
        seen.append(ctypes.WinDLL("kernel32").GetProcessId(ctypes.c_void_p(handle)))
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_mod, "_new_job", boom)
    w = _worker(store, tmp_path, "fake_ok")
    store.submit([_req()])
    try:
        with pytest.raises(KeyboardInterrupt):
            w.run_one()
    finally:
        w.close()
    assert len(seen) == 1 and seen[0] > 0
    try:
        assert wait_dead(seen) == []
    finally:
        force_kill(seen)


def test_deadline_is_set_at_dequeue_not_submit(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W3, new capability): the child sees the deadline set at
    dequeue (now + timeout); a deadline on the submitted request is overwritten."""
    w = _worker(store, tmp_path, "fake_ok", timeout_s=500.0)
    store.submit([_req(deadline=1.0)])
    submitted = time.time()
    time.sleep(0.3)  # time spent queued must not count
    item_id = w.run_one()
    w.close()
    item = store.item(item_id)
    assert wait_dead([item["attempts"][0]["child_pid"]]) == []
    seen = item["result"]["counts"]["deadline"]
    assert seen == item["deadline"]
    assert seen == pytest.approx(item["started_at"] + 500.0)
    assert seen >= submitted + 0.3 + 500.0
    assert item["request"].deadline == 1.0  # stored request untouched


def test_deadline_kills_whole_tree_and_marks_timeout(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W3, new capability): past its deadline the child AND a
    grandchild it spawned as its very first action are killed; the item is
    status=timeout, failure_kind=timeout."""
    w = _worker(store, tmp_path, "fake_hang", timeout_s=10.0)
    store.submit([_req()])
    t0 = time.time()
    item_id = w.run_one()
    elapsed = time.time() - t0
    w.close()
    pids = [int(p) for p in
            (tmp_path / "work" / f"item-{item_id}" / "attempt-1" / "pids.txt").read_text().split()]
    item = store.item(item_id)
    # On Windows a venv python.exe is a launcher that starts the real interpreter as
    # ITS child, so the recorded child PID and the run's os.getpid() can differ.
    pids.append(item["attempts"][0]["child_pid"])
    try:
        assert wait_dead(pids) == []
    finally:
        force_kill(pids)
    assert item["status"] == STATUS_TIMEOUT and item["failure_kind"] == KIND_TIMEOUT
    assert item["result"]["status"] == STATUS_TIMEOUT
    assert item["attempts"][0]["failure_kind"] == KIND_TIMEOUT
    assert 10.0 <= elapsed < 40.0


@pytest.mark.parametrize("fake", ["fake_raise", "fake_exit"])
def test_child_failure_is_a_crash(store, tmp_path, child_env, fake):
    """NEW ACCEPTANCE TEST (W3, new capability): a raising run, or a child that exits
    without a result, is persisted as status=error, failure_kind=crash."""
    w = _worker(store, tmp_path, fake)
    store.submit([_req()])
    item = store.item(w.run_one())
    w.close()
    assert wait_dead([item["attempts"][0]["child_pid"]]) == []
    assert item["status"] == STATUS_ERROR and item["failure_kind"] == KIND_CRASH
    if fake == "fake_raise":
        assert "boom" in item["result"]["message"]
    else:
        assert item["attempts"][0]["exit_code"] == 3


def test_killed_worker_takes_its_child_tree_and_restart_recovers(tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W3, new capability): a worker killed mid-run leaves no
    child or grandchild behind, and the next worker start finishes the orphaned
    ``running`` item as error/crash without re-running it."""
    db, work = tmp_path / "jobs.sqlite", tmp_path / "work"
    s = JobStore(db)
    s.submit([_req("a"), _req("b")])
    s.close()
    proc = subprocess.Popen([sys.executable, "-m", "t2pw.web.worker", "--db", str(db),
                             "--work-root", str(work), "--run-fn", f"{MOD}:fake_hang",
                             "--poll", "0.1"])
    pids_file = work / "item-1" / "attempt-1" / "pids.txt"
    pids = []
    try:
        end = time.time() + 30
        while not pids_file.exists() and time.time() < end and proc.poll() is None:
            time.sleep(0.1)
        time.sleep(0.2)
        pids = [int(p) for p in pids_file.read_text().split()]
        probe = JobStore(db)
        pids.append(probe.item(1)["attempts"][0]["child_pid"])
        probe.close()
        proc.kill()  # the worker only: simulates a crash
        proc.wait(timeout=30)
        assert wait_dead(pids) == []
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        force_kill(pids)

    s = JobStore(db)
    w = Worker(s, work)
    assert w.start() == [1]
    item = s.item(1)
    # W4: the recovered crash now goes through the retry policy and is re-queued (W3
    # asserted FINISHED here). It is still not re-run: attempt stays 1.
    assert item["state"] == jobs.QUEUED and item["result"] is None
    assert item["attempts"][0]["status"] == STATUS_ERROR
    assert item["attempts"][0]["failure_kind"] == KIND_CRASH
    assert item["attempt"] == 1 and len(item["attempts"]) == 1
    assert s.item(2)["state"] == jobs.QUEUED
    w.close()
    s.close()


def test_single_worker_lock(store, tmp_path):
    """NEW ACCEPTANCE TEST (W3, new capability, review F1): a second worker on the same
    database refuses to start, even with a different work root, so start-up recovery
    never touches a live worker's item."""
    w1 = Worker(store, tmp_path / "work-a")
    w1.start()
    other = JobStore(store.path)
    try:
        for root in ("work-a", "work-b"):
            with pytest.raises(RuntimeError):
                Worker(other, tmp_path / root).start()
    finally:
        w1.close()
    w2 = Worker(other, tmp_path / "work-b")
    w2.start()
    w2.close()
    other.close()


def test_worker_imports_no_streamlit_or_app():
    """NEW ACCEPTANCE TEST (W3, new capability): import isolation, in a fresh process."""
    code = ("import sys, t2pw.web.worker, t2pw.web.jobs; "
            "bad = [m for m in sys.modules if m.split('.')[0] == 'streamlit' "
            "or m == 't2pw.app' or m.startswith('t2pw.app.')]; print(bad); sys.exit(bool(bad))")
    env = dict(os.environ, PYTHONPATH=str(SRC_DIR))
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


# ── W4: auto-retry ─────────────────────────────────────────────────────────


def test_infrastructure_failure_is_retried_with_a_fresh_attempt(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W4, new capability): an LLM failure re-queues the item; the
    next run_one is attempt 2 with its own directory and deadline, and passes."""
    w = _worker(store, tmp_path, "fake_llm_then_ok", timeout_s=500.0, max_retries=MAX_RETRIES)
    store.submit([_req()])
    item_id = w.run_one()
    mid = store.item(item_id)
    assert mid["state"] == jobs.QUEUED and mid["result"] is None and mid["attempt"] == 1
    assert store.attempt_history(item_id)[0]["will_retry"] is True
    assert w.run_one() == item_id
    w.close()
    item = store.item(item_id)
    assert item["state"] == jobs.FINISHED and item["status"] == STATUS_PASS and item["attempt"] == 2
    assert item["result"]["counts"]["out_dir"].endswith(os.path.join("attempt-2", "out"))
    assert item["result"]["counts"]["deadline"] == pytest.approx(item["started_at"] + 500.0)
    hist = store.attempt_history(item_id)
    assert [(h["attempt"], h["failure_kind"], h["status"], h["retried"]) for h in hist] == [
        (1, KIND_LLM, STATUS_FAIL, True), (2, "", STATUS_PASS, False)]
    assert wait_dead([a["child_pid"] for a in item["attempts"]]) == []


def test_content_verdict_is_final_on_the_first_attempt(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W4, new capability): a contract verdict is persisted as the
    item's outcome and never re-queued."""
    w = _worker(store, tmp_path, "fake_contract", max_retries=MAX_RETRIES)
    store.submit([_req()])
    item_id = w.run_one()
    assert w.run_one() is None
    w.close()
    item = store.item(item_id)
    assert item["state"] == jobs.FINISHED and item["failure_kind"] == KIND_CONTRACT
    assert len(item["attempts"]) == 1
    assert store.attempt_history(item_id)[0]["retried"] is False


def test_stops_at_three_attempts(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W4, new capability): a run that keeps failing is attempted
    three times (MAX_RETRIES = 2) and the third failure is the final outcome."""
    w = _worker(store, tmp_path, "fake_llm", max_retries=MAX_RETRIES)
    store.submit([_req()])
    ids = [w.run_one() for _ in range(4)]
    w.close()
    item = store.item(ids[0])
    assert ids[:3] == [ids[0]] * 3 and ids[3] is None
    assert item["state"] == jobs.FINISHED and item["attempt"] == 3
    assert item["status"] == STATUS_FAIL and item["failure_kind"] == KIND_LLM
    assert [h["retried"] for h in store.attempt_history(ids[0])] == [True, True, False]
    assert wait_dead([a["child_pid"] for a in item["attempts"]]) == []


def test_wall_cap_shapes_the_retry_deadline_and_stops_retries(store, tmp_path, child_env):
    """NEW ACCEPTANCE TEST (W4, new capability): with 7000 s of the 7200 s wall cap used
    by attempt 1, attempt 2 gets a deadline of the ~200 s left (not RUN_TIMEOUT_S), and
    its failure is final because less than MIN_RETRY_BUDGET_S remains."""
    w = _worker(store, tmp_path, "fake_llm", timeout_s=3600.0, max_retries=MAX_RETRIES)
    store.submit([_req()])
    item_id = w.run_one()
    assert store.item(item_id)["state"] == jobs.QUEUED
    store._db.execute("UPDATE attempts SET started_at = started_at - 7000")  # 7000 s used
    w.run_one()
    w.close()
    item = store.item(item_id)
    assert 100 < item["deadline"] - item["started_at"] < 200
    assert item["state"] == jobs.FINISHED and item["attempt"] == 2
    assert item["failure_kind"] == KIND_LLM
    assert store.wall_used(item_id) > 7000


def test_restart_crash_goes_through_the_policy_and_is_bounded(store, tmp_path):
    """NEW ACCEPTANCE TEST (W4, new capability): an item the worker died on is a crash
    that the policy re-queues; a run that keeps killing the worker is final after the
    third attempt, not an endless loop."""
    store.submit([_req()])
    for n in (1, 2, 3):
        item = store.claim_next(60)            # a worker takes it ...
        assert item["attempt"] == n
        w = Worker(store, tmp_path / "work")   # ... and dies; the next start recovers
        assert w.start() == [item["id"]]
        w.close()
        state = store.item(item["id"])["state"]
        assert state == (jobs.QUEUED if n < 3 else jobs.FINISHED)
    final = store.item(item["id"])
    assert final["status"] == STATUS_ERROR and final["failure_kind"] == KIND_CRASH
    assert [a["failure_kind"] for a in final["attempts"]] == [KIND_CRASH] * 3
    assert store.claim_next(60) is None
