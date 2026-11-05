"""NEW ACCEPTANCE TEST (W7, new capability): the PathBank curator queue,
``t2pw.web.curator_queue``. Fake results only: no LLM, no network, no SMTP, no pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from t2pw.pipeline.release_status import RELEASE_READY, REVIEW_REQUIRED, ReleaseStatus
from t2pw.rag.eligibility import RequestedScope
from t2pw.service.pathway import (
    MODE_RESEARCH,
    MODE_STRICT,
    STAGE_PWML_EXPORT,
    STATUS_PASS,
    PathwayRequest,
    PathwayResult,
)
from t2pw.web.curator_queue import CITATION_KEYS, PROVENANCE_KEYS, CuratorQueue, qualify
from t2pw.web.jobs import JobStore, result_to_dict


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


@pytest.fixture
def work(tmp_path):
    return tmp_path / "work"


def make_final(store, work, *, mode=MODE_STRICT, release=RELEASE_READY, consent=True,
               pwml=True, email=None, reports=PROVENANCE_KEYS, requeue=False):
    """Submit one item, claim it, write fake artifacts, finish it. Returns (sid, item_id)."""
    sid = store.submit([PathwayRequest(
        scope=RequestedScope(requested_pathway="serine biosynthesis", requested_organism="Homo sapiens"),
        export_mode=mode)], consent=consent, email=email)
    item = store.claim_next(60)
    out = work / f"item-{item['id']}" / "attempt-1" / "out"
    out.mkdir(parents=True)
    pwml_path = None
    if pwml:
        pwml_path = out / ("pathway.pwml" if release == RELEASE_READY else "pathway.review_required.pwml")
        pwml_path.write_text("<pwml>" + "x" * 3000 + "</pwml>", encoding="utf-8")
    rp = {}
    for k in tuple(reports) + ("stage0_attempts.json", "gate_fail_report.json"):  # last two: not provenance
        (out / k).write_text(json.dumps({"k": k}), encoding="utf-8")
        rp[k] = out / k
    res = result_to_dict(PathwayResult(
        status=STATUS_PASS, stage=STAGE_PWML_EXPORT, pwml_path=pwml_path, report_paths=rp,
        release_status=ReleaseStatus(status=release, pipeline_executed=True, strict_gates_passed=True)))
    assert store.finish(item["id"], item["attempt"], res, requeue=requeue)
    return sid, item["id"]


def parts(store, sid, item_id):
    return store.submission(sid), store.item(item_id)


def test_qualifying_item_is_staged_with_matching_sha256(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): a consented strict release_ready item
    with a PWML is copied with its provenance reports; the manifest records ids, organism,
    pathway, release_status, consent and a sha256 of every copied file."""
    sid, iid = make_final(store, work)
    q = CuratorQueue(store.path, tmp_path / "queue")
    row = q.enqueue(*parts(store, sid, iid), work)
    assert row["status"] == "pending" and row["organism"] == "Homo sapiens"
    entry = tmp_path / "queue" / row["entry_dir"]
    man = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert man["item_id"] == iid and man["submission_id"] == sid and man["job_id"] == sid
    assert man["pathway_name"] == "serine biosynthesis" and man["organism"] == "Homo sapiens"
    assert man["release_status"]["status"] == RELEASE_READY and man["consent"] is True
    assert man["enqueued_at"] > 0 and man["provenance_not_available"] == []
    names = {f["path"] for f in man["files"]}
    assert names == {"pathway.pwml"} | {f"provenance/{k}" for k in PROVENANCE_KEYS}
    for f in man["files"]:
        data = (entry / f["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == f["sha256"] and len(data) == f["bytes"]
    src = work / f"item-{iid}" / "attempt-1" / "out" / "pathway.pwml"
    assert (entry / "pathway.pwml").read_bytes() == src.read_bytes()
    assert not (entry / "provenance" / "stage0_attempts.json").exists()  # diagnostics are not provenance
    assert q.verify(iid) and not list((tmp_path / "queue").glob(".tmp-*"))
    (entry / "pathway.pwml").write_text("tampered", encoding="utf-8")
    assert not q.verify(iid)


def test_citation_report_copied_when_present_and_missing_provenance_is_recorded(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): the citation report goes in when the
    result has one; a provenance report the run did not write is listed, not fatal."""
    sid, iid = make_final(store, work, reports=PROVENANCE_KEYS[:2] + CITATION_KEYS)
    q = CuratorQueue(store.path, tmp_path / "queue")
    q.enqueue(*parts(store, sid, iid), work)
    man = q.manifest(iid)
    paths = {f["path"] for f in man["files"]}
    assert "citations/research_pathway_citations.json" in paths
    assert man["provenance_not_available"] == list(PROVENANCE_KEYS[2:])


@pytest.mark.parametrize("kw", [
    {"mode": MODE_RESEARCH}, {"release": REVIEW_REQUIRED}, {"pwml": False}, {"consent": False},
    {"requeue": True},
], ids=["research_mode", "review_required", "no_pwml", "consent_off", "retry_pending"])
def test_non_qualifying_items_are_not_queued(store, tmp_path, work, kw):
    """NEW ACCEPTANCE TEST (W7, new capability): research mode, review_required, no PWML,
    consent off, and an item re-queued for a retry never enter the queue, and nothing
    is created on disk for them."""
    sid, iid = make_final(store, work, **kw)
    q = CuratorQueue(store.path, tmp_path / "queue")
    assert q.enqueue(*parts(store, sid, iid), work) is None
    assert q.list_entries() == [] and not (tmp_path / "queue").exists()


def test_files_outside_the_work_root_are_not_copied(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): a result naming a PWML or report outside
    the worker's work root is not followed."""
    sid, iid = make_final(store, work)
    sub, item = parts(store, sid, iid)
    assert qualify(sub, item, work)[0] and not qualify(sub, item, tmp_path / "elsewhere")[0]
    secret = tmp_path / "secret.txt"
    secret.write_text("s", encoding="utf-8")
    item["result"]["report_paths"]["final_mapped.json"] = str(secret)
    q = CuratorQueue(store.path, tmp_path / "queue")
    q.enqueue(sub, item, work)
    assert "final_mapped.json" in q.manifest(iid)["provenance_not_available"]
    assert not (tmp_path / "queue" / f"entry-{iid}" / "provenance" / "final_mapped.json").exists()


def test_enqueue_is_idempotent(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): enqueueing the same item again (a re-run,
    a retry, a restart) leaves one row, one folder and an unchanged manifest."""
    sid, iid = make_final(store, work)
    q = CuratorQueue(store.path, tmp_path / "queue")
    first = q.enqueue(*parts(store, sid, iid), work)
    before = (tmp_path / "queue" / first["entry_dir"] / "manifest.json").read_bytes()
    q2 = CuratorQueue(store.path, tmp_path / "queue")  # a fresh process would build a new object
    assert q2.enqueue(*parts(store, sid, iid), work) == first
    assert len(q.list_entries()) == 1 and len(list((tmp_path / "queue").iterdir())) == 1
    assert (tmp_path / "queue" / first["entry_dir"] / "manifest.json").read_bytes() == before


def test_orphan_folder_from_a_crash_is_adopted_not_duplicated(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): a crash after the folder was published but
    before the row was inserted: the next enqueue adds the row for the same folder."""
    sid, iid = make_final(store, work)
    q = CuratorQueue(store.path, tmp_path / "queue")
    q.enqueue(*parts(store, sid, iid), work)
    q._db.execute("DELETE FROM curator_queue")
    row = q.enqueue(*parts(store, sid, iid), work)
    assert row["item_id"] == iid and len(list((tmp_path / "queue").iterdir())) == 1 and q.verify(iid)


def test_curator_read_api(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): list pending, mark imported / rejected
    with a note; a decided entry cannot be decided again and leaves the pending list."""
    q = CuratorQueue(store.path, tmp_path / "queue")
    ids = []
    for _ in range(3):
        sid, iid = make_final(store, work)
        q.enqueue(*parts(store, sid, iid), work)
        ids.append(iid)
    assert [r["item_id"] for r in q.list_pending()] == ids
    assert q.mark_imported(ids[0], "imported as PW123456") is True
    assert q.mark_rejected(ids[1], "duplicate of an existing pathway") is True
    assert q.mark_rejected(ids[0], "late change of mind") is False
    assert [r["item_id"] for r in q.list_pending()] == [ids[2]]
    done = {r["item_id"]: r for r in q.list_entries()}
    assert done[ids[0]]["status"] == "imported" and done[ids[0]]["note"] == "imported as PW123456"
    assert done[ids[1]]["status"] == "rejected" and done[ids[1]]["decided_at"] > 0
    assert [r["item_id"] for r in q.list_entries("rejected")] == [ids[1]]


def snapshot(root: Path, skip: Path):
    return {p: (p.stat().st_size, p.read_bytes() if p.suffix != ".sqlite" else b"")
            for p in root.rglob("*")
            if p.is_file() and skip not in p.parents and "jobs.sqlite" not in p.name}


def test_nothing_is_written_outside_the_queue_directory(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability): enqueueing adds files only under the
    queue directory (plus the jobs DB, which holds the queue table); sources are
    untouched."""
    sid, iid = make_final(store, work)
    queue = tmp_path / "queue"
    before = snapshot(tmp_path, queue)
    CuratorQueue(store.path, queue).enqueue(*parts(store, sid, iid), work)
    assert snapshot(tmp_path, queue) == before
    added = [p for p in tmp_path.rglob("*") if p.is_file() and queue in p.parents]
    assert added and all(not p.name.endswith(".tmp") for p in added)


def test_stale_folder_from_another_job_is_moved_aside_not_adopted(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability; review N1): after a DB reset ids restart; an
    entry-<id> folder whose manifest names another submission is moved aside (never
    deleted) and the entry is rebuilt from this job's files."""
    queue = tmp_path / "queue"
    sid, iid = make_final(store, work)
    stale = queue / f"entry-{iid}"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text(json.dumps({"item_id": iid, "submission_id": "other"}), encoding="utf-8")
    (stale / "pathway.pwml").write_text("OTHER JOB", encoding="utf-8")
    q = CuratorQueue(store.path, queue)
    q.enqueue(*parts(store, sid, iid), work)
    assert q.manifest(iid)["submission_id"] == sid and q.verify(iid)
    assert "OTHER JOB" not in (queue / f"entry-{iid}" / "pathway.pwml").read_text(encoding="utf-8")
    (moved,) = queue.glob(".stale-*")
    assert (moved / "pathway.pwml").read_text(encoding="utf-8") == "OTHER JOB"


def test_non_pass_status_does_not_qualify(store, tmp_path, work):
    """NEW ACCEPTANCE TEST (W7, new capability; review N4): a result that is not status
    pass never qualifies, whatever else it carries."""
    sid, iid = make_final(store, work)
    sub, item = parts(store, sid, iid)
    assert qualify(sub, item, work)[0]
    item["result"]["status"] = "fail"
    assert qualify(sub, item, work) == (False, "status not pass")
