"""Local production-like deployment glue for W9-local (this Windows PC). Not product code.

Two entry points, loaded through PYTHONPATH (``deploy/local``):

* ``t2pw_deploy:create_app`` - uvicorn ``--factory``. The stock ``create_app_from_env`` plus
  the deployment's caps: per-IP submissions per 24 h (``T2PW_RATE_LIMIT``) and the open-item
  queue cap (``T2PW_MAX_QUEUED``). The module constants are set before the app is built so
  the limits shown on the site are the limits enforced.
* ``t2pw_deploy:run_pathway`` - worker ``--run-fn``. Runs the real
  ``t2pw.service.pathway:run_pathway`` as its OWN process under
  ``bounded_run.py --heavy-lock``, so a web job never overlaps other heavy work on this PC.
  While the lock is held by someone else it waits and retries; the inner wrapper's timeout
  ends before the worker's deadline, so the wrapper (not a hard kill) releases the lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOUNDED_RUN = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "bounded_run.py"
REAL_RUN_FN = "t2pw.service.pathway:run_pathway"
HOLDER = "W9-local-job"
LOCK_RETRY_S = 30
MARGIN_S = 120       # inner timeout ends this long before the worker's deadline kill
MIN_RUN_S = 600      # less than this left after waiting for the lock: give up as a timeout


def create_app():
    from t2pw.web import app as web_app, jobs, limits

    limits.RATE_LIMIT_SUBMISSIONS = int(os.environ.get("T2PW_RATE_LIMIT", "2"))
    jobs.MAX_QUEUED_ITEMS = int(os.environ.get("T2PW_MAX_QUEUED", "5"))
    return web_app.create_app(os.environ["T2PW_DB"], os.environ["T2PW_WORK_ROOT"],
                              os.environ.get("T2PW_BASE_URL", ""),
                              rate_limit=limits.RATE_LIMIT_SUBMISSIONS,
                              trusted_proxies=os.environ.get("T2PW_TRUSTED_PROXIES"))


class _Release:
    """``result_to_dict`` only calls ``to_dict()`` on the release status."""

    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


def _result(d):
    from t2pw.service.pathway import PathwayResult

    return PathwayResult(
        status=d["status"], stage=d["stage"], failure_kind=d["failure_kind"],
        message=d["message"], warnings=list(d["warnings"]), issue_codes=list(d["issue_codes"]),
        counts=dict(d["counts"]),
        release_status=None if d["release_status"] is None else _Release(d["release_status"]),
        pwml_path=None if d["pwml_path"] is None else Path(d["pwml_path"]),
        image_path=None if d["image_path"] is None else Path(d["image_path"]),
        report_paths={k: Path(v) for k, v in d["report_paths"].items()},
        attempt_log=list(d["attempt_log"]))


def run_pathway(req):
    from t2pw.service.pathway import (KIND_CRASH, KIND_TIMEOUT, STAGE_INIT, STATUS_ERROR,
                                      STATUS_TIMEOUT, PathwayResult)
    from t2pw.web.jobs import request_to_dict

    attempt_dir = Path(req.out_dir).parent
    inner_req, inner_res = attempt_dir / "inner_request.json", attempt_dir / "inner_result.json"
    inner_req.write_text(json.dumps(request_to_dict(req)), encoding="utf-8")
    deadline = req.deadline or (time.time() + 90 * 60)
    while True:
        budget = deadline - time.time() - MARGIN_S
        if budget < MIN_RUN_S:
            return PathwayResult(status=STATUS_TIMEOUT, stage=STAGE_INIT, failure_kind=KIND_TIMEOUT,
                                 message="the server stayed busy with other work; try again later")
        rc = subprocess.run(
            [sys.executable, str(BOUNDED_RUN), "--timeout", f"{budget:.0f}", "--label",
             f"t2pw-web-{attempt_dir.parent.name}-{attempt_dir.name}", "--heavy-lock", HOLDER,
             "--json", str(attempt_dir / "bounded_run.json"), "--",
             sys.executable, "-m", "t2pw.web.worker", "--child", REAL_RUN_FN,
             str(inner_req), str(inner_res)],
            cwd=str(ROOT), stdin=subprocess.DEVNULL).returncode
        print(f"[t2pw_deploy] bounded_run exit {rc}", flush=True)
        if rc != 95:  # 95 = heavy lock held elsewhere; the run never started
            break
        time.sleep(LOCK_RETRY_S)
    try:
        return _result(json.loads(inner_res.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return PathwayResult(status=STATUS_ERROR, stage=STAGE_INIT, failure_kind=KIND_CRASH,
                             message=f"run wrapper exited {rc} without a result")
