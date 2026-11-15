# RELEASE-P6 reviews (ws-reviewer, opus / high; the reviewer has no write tool, verdicts transcribed by the orchestrator)

## Review 1: `49551d19..7e258805`, CHANGES REQUESTED
Gates: boundary ok, no-biology ok, behaviour preserved ok (default flip authorised by D-W23 b), tests NOT ok (F1), evidence and result doc missing (F7), secrets ok.
- F1 [blocking] `tests/test_equivalence_default_path.py:102`: the v2 check `b"<membrane-visualization" in xml` also matches v1 output, because the writer always writes the empty `<membrane-visualizations/>` tag. Fix: parse the XML and require items.
- F2 [note] `tests/test_final_payload_default_seam.py:105`: `inner.calls == sum(per_host.values())` is always true. Fix: cap the number of hosts.
- F3 [note] the default-path test never compares the output with the recorded payload. Fix: compare, allowing only the documented flag-on fields.
- F4 [note] `final_payload.py:25` docstring: "a dead network costs seconds" is now N failed requests per host.
- F5 [note] `layout.py:74`: `0`, `off`, `false` give v2; document it.
- F6 [note] `streamlit_app.py:2664-2665`: under v2 the Width/Height inputs no longer set the canvas.
- F7 [note] no RELEASE-P6 doc, no committed evidence.
Answers: the default-path test runs the real defaults through export on both arms, offline (`refused == []`, `misses == []`), needs its own process (stated); per-host breaker correct (host = `urlsplit(url).netloc.casefold()`, success resets only its host, N = 5, deadline global, no threads); only `v1` selects v1, nothing in `src/` sets the flag; no v1 pin hides a v2 defect (the 8 `test_pwml_writer.py` tests assert v1 coordinates; the writer-fixes test compares whole files); the fixture was recorded 2026-11-05/08 before the layout code, so its `pwml_sha256` is v1 geometry and the harness pin is justified; merge `4c6e8dfb` clean (one auto-merged import hunk, no extra edits); protected files and fixtures unedited; no biology; no secrets.

## Review 2: `7e258805..24fdbe73`, APPROVE WITH NOTES
F1 fixed (XML parse; items required under membrane-visualizations and label-locations; bite proof `impl/eq_bite.log` fails on exactly that assertion, `assert 0 > 0`, exit 1, 0 survivors). F2 fixed (`len(per_host) <= 3`). F3 fixed and narrow: every recorded leaf must exist with the same value; new leaves only under `/final_payload_pathway/`; changed values only for paths ending `/retrieved_at` (15 of 2221 leaves in the sour_taste fixture, all compound provenance stamps; the same stamp exception `test_equivalence_app_vs_service.py:155` uses under the 2026-11-05 ruling). F4, F5 fixed (docstrings).
Notes, not fixed in this release (non-blocking):
- N1 the stamp exception does not also check that the replay stamps equal `harness.FROZEN_UTC` (as `test_equivalence_app_vs_service.py:175` does).
- N2 host cap is 3 while the comment says 2 hosts; `== 2` would be tighter.
- N3 record how the hook was disabled for the bite proof: `impl/bite_toggle.py on` replaced `if layout_version() == "v2":` in `writer.py` with `if False and ...`, `impl/bite_run.sh` ran the test file and restored it (`bite_toggle.py off`); the tree was clean afterwards.
