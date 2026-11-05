# W7b: completion-email link uses W6's canonical job_url

Branch `ws/W7b-email-link` from `webserver/refactor` @ d60f1690.

## Change
- `src/t2pw/web/postrun.py`: the link passed to `notify_job` is now `t2pw.web.views.job_url(base_url, sid)`
  (was `<base>/jobs/<sid>`, which the site only 303-redirected to `/job/<sid>`). Email links now equal the
  canonical `<base>/job/<sid>`.
- `views.py` has no FastAPI import (imports only `t2pw.service.pathway`, `t2pw.web.retry`); not edited.

## Tests (`tests/test_web_postrun.py`)
Updated (link assertion only, 3 lines): `test_final_qualifying_item_is_enqueued_and_single_item_job_is_notified`,
`test_notify_once_per_submission_only_after_all_items_final`, `test_recovery_at_worker_start_settles_a_crashed_final_item`
now expect `job_url(BASE, sid)` instead of `f"{BASE}/jobs/{sid}"`; added `job_url` import.
New (labelled in docstrings):
- `test_email_link_is_views_job_url_and_the_site_serves_it`: link == `job_url`, and a W6 `create_app` TestClient answers its path with 200 (or 303 to the job page).
- `test_postrun_import_does_not_pull_fastapi`: subprocess import of `t2pw.web.postrun`; no `fastapi*` in `sys.modules`.

## Evidence
`evidence/W7b/focused.json` + `focused.log`: test_web_postrun + test_web_app + test_web_curator_queue via bounded_run/pinned_pytest,
68 passed, survivors 0. SMOKE not run (per brief).

## Behaviour change
Email URL path `/jobs/<sid>` -> `/job/<sid>`. The legacy `/jobs/<sid>` route is untouched, so already-sent links still work.

## Open questions
None.

## Dead ends

None.
