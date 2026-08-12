"""Focused parser/runner selftest for ``chunk_d_gate.py`` (task H-007).

The Chunk D gate decides whether a merge may proceed, and it decides it by
reading pytest's own text. Every case below is a way that reading has been, or
could be, wrong in the direction that matters -- a run reported as a pass when
it was not:

* a skip, an xfail, an xpass or a deselection counted as coverage;
* an error line beside a full ``passed`` count;
* a summary that never appeared because the job was killed at its bound;
* a nonzero process exit underneath an apparently valid summary;
* the wrapper's merged shutdown noise read as the verdict;
* a basetemp whose parent does not exist, which errors every test in setup.

Run it under the wrapper like any other test::

    <py> evidence/bounded_run.py --label chunkd-selftest --timeout 900 \\
         --json evidence/g11/<TASK>/<SEQ>-chunkd-selftest.json \\
         -- <py> -u evidence/chunk_d_gate_selftest.py

``CHUNK_D_SELFTEST_TMP`` overrides where the one real end-to-end case puts its
scratch tree, for a checkout whose temp path is close to ``MAX_PATH``.
"""

from __future__ import annotations

import itertools
import os
import shutil
import tempfile
from typing import Dict

from chunk_d_gate import CORE, _collect_one, _job, job_verdict, parse_summary

#: A wrapper report from a child that exited cleanly with nothing left alive.
CLEAN: Dict[str, object] = {
    "exit_reason": "completed", "cleanup_success": True, "final_surviving_count": 0,
}
ZEROS = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0,
         "xfailed": 0, "xpassed": 0, "deselected": 0}


def _counts(**over: int) -> Dict[str, int]:
    return {**ZEROS, **over}


def test_ordinary_passing_summary() -> None:
    assert parse_summary("collecting ...\n150 passed in 0.93s\n") == _counts(passed=150)
    assert job_verdict(150, 0, "150 passed in 0.93s", CLEAN) == (True, "ok")


def test_failures_and_errors() -> None:
    assert parse_summary("2 failed, 21 passed in 400.12s") == _counts(failed=2, passed=21)
    ok, detail = job_verdict(23, 1, "2 failed, 21 passed in 400.12s", CLEAN)
    assert not ok and "failed=2" in detail and "passed=21/23" in detail, detail
    # An error line beside a FULL passed count is still a failure.
    ok, detail = job_verdict(150, 0, "150 passed, 5 errors in 3.10s", CLEAN)
    assert not ok and "errors=5" in detail, detail
    assert parse_summary("1 error in 0.20s") == _counts(errors=1)  # singular folds in


def test_skips_are_not_passes() -> None:
    ok, detail = job_verdict(4, 0, "3 passed, 1 skipped in 0.50s", CLEAN)
    assert not ok and "passed=3/4" in detail and "skipped=1" in detail, detail


def test_xfail_and_xpass_are_not_passes() -> None:
    # 'xfailed' must not be read as 'failed', and neither counts as coverage.
    assert parse_summary("1 xfailed in 0.10s") == _counts(xfailed=1)
    for word in ("xfailed", "xpassed"):
        ok, detail = job_verdict(23, 0, f"22 passed, 1 {word} in 9.00s", CLEAN)
        assert not ok and f"{word}=1" in detail, detail


def test_deselection_is_not_coverage() -> None:
    ok, detail = job_verdict(23, 0, "1 passed, 22 deselected in 0.40s", CLEAN)
    assert not ok and "deselected=22" in detail and "passed=1/23" in detail, detail


def test_malformed_or_missing_summary_is_never_an_implicit_zero() -> None:
    assert parse_summary("") is None
    assert parse_summary("collecting ...\nINTERNALERROR> boom\n") is None
    assert parse_summary("150 passed") is None  # no "in <n>s": not a summary line
    ok, detail = job_verdict(150, 0, "", CLEAN)
    assert not ok and "no_summary" in detail, detail


def test_trailing_merged_stream_shutdown_noise() -> None:
    noisy = (
        "150 passed in 0.93s\n"
        "Exception ignored in: <_io.TextIOWrapper name='<stdout>' encoding='utf-8'>\n"
        "timeout                 : 2400 s\n"
        "duration                : 400.12 s\n"
        "exit code (real)        : 0\n"
        "FINAL SURVIVING COUNT   : 0\n"
    )
    assert parse_summary(noisy) == _counts(passed=150)
    # ... while a genuinely later summary still wins over an earlier one.
    assert parse_summary("1 passed in 0.10s\n2 passed in 0.20s\n") == _counts(passed=2)


def test_timeout_is_a_failure_not_a_silent_pass() -> None:
    killed = {**CLEAN, "exit_reason": "timeout"}
    ok, detail = job_verdict(23, 1, "collecting ... collected 23 items\n", killed)
    assert not ok and "no_summary" in detail and "exit_reason=timeout" in detail, detail


def test_nonzero_exit_or_an_unaccounted_child_beats_a_valid_summary() -> None:
    green = "150 passed in 0.93s"
    ok, detail = job_verdict(150, 1, green, CLEAN)
    assert not ok and "exit=1" in detail, detail
    ok, detail = job_verdict(150, 0, green, {**CLEAN, "cleanup_success": False})
    assert not ok and "cleanup_not_successful" in detail, detail
    ok, detail = job_verdict(150, 0, green, {**CLEAN, "final_surviving_count": 2})
    assert not ok and "survivors=2" in detail, detail
    ok, detail = job_verdict(150, 0, green, {})  # no report written at all
    assert not ok and "report_unreadable" in detail, detail


def test_absent_basetemp_parent_and_one_real_isolated_node() -> None:
    """End to end: a missing basetemp parent is created, and one real core node
    run ALONE in a fresh process returns a clean 1-passed verdict.

    Without this the parser cases above would all be assertions about strings.
    """

    box = tempfile.mkdtemp(prefix="cd_", dir=os.environ.get("CHUNK_D_SELFTEST_TMP"))
    try:
        seq = itertools.count()
        reports = os.path.join(box, "r")
        os.makedirs(reports)

        def alloc(label: str) -> str:
            return os.path.join(reports, f"{next(seq):02d}-{label}.json")

        tmp = os.path.join(box, "t")  # deliberately NOT created
        assert not os.path.isdir(tmp)
        ids, ok, note = _collect_one(alloc, "sel-collect", tmp, [CORE[-1]], 300.0)
        assert ok, note
        assert os.path.isdir(tmp), "the runner did not create the basetemp parent"
        node = sorted(ids)[0]
        code, out, report = _job(alloc, "sel-node", tmp, [node], 300.0)
        good, detail = job_verdict(1, code, out, report)
        assert good, f"{node}: {detail}"
    finally:
        shutil.rmtree(box, ignore_errors=True)


TESTS = [
    test_ordinary_passing_summary,
    test_failures_and_errors,
    test_skips_are_not_passes,
    test_xfail_and_xpass_are_not_passes,
    test_deselection_is_not_coverage,
    test_malformed_or_missing_summary_is_never_an_implicit_zero,
    test_trailing_merged_stream_shutdown_noise,
    test_timeout_is_a_failure_not_a_silent_pass,
    test_nonzero_exit_or_an_unaccounted_child_beats_a_valid_summary,
    test_absent_basetemp_parent_and_one_real_isolated_node,
]


def main() -> int:
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"ok   {fn.__name__}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
