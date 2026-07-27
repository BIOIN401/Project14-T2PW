"""Tests for the overnight batch orchestrator (``t2pw.batch.runner`` + the CLI).

Nothing here launches the real app, spawns a pipeline, or touches the network.
The orchestration seams are injected instead:

* ``child_fn`` stands in for the subprocess call, so a pass / a timeout / a
  garbage-printing crash are all one-line fixtures;
* ``fetch_fn`` stands in for paper acquisition, so the "papers" are literals;
* ``run_fn`` stands in for ``driver.run_one`` when the *child* half is under test.

The things actually asserted are the promises the user made the runner keep:
resume happens with no flag and never redoes or skips work, one stuck paper does
not eat the night, a child that dies without a result line is an error rather
than a hang, and the manifest stays valid append-only JSONL throughout.

The single test that does spawn a real process spawns ``python -c "sleep"`` to
prove the timeout path kills a child tree -- no app, no network.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.batch import runner  # noqa: E402
from t2pw.batch.driver import MODE_RESEARCH, MODE_STRICT, RunOutcome  # noqa: E402
from t2pw.batch.fetch import BatchPaper  # noqa: E402

CLI_PATH = ROOT / "scripts" / "batch_run.py"


# ---------------------------------------------------------------------------
# Fixtures / helpers.
# ---------------------------------------------------------------------------
def _make_run_dir(
    tmp_path: Path,
    *,
    name: str = "2020-01-01_0000",
    slugs: Sequence[str] = ("PMC1__alpha", "PMC2__beta"),
    modes: Sequence[str] = (MODE_STRICT, MODE_RESEARCH),
) -> Path:
    """Lay out a run directory exactly as a fresh run would have left it."""
    run_dir = tmp_path / name
    (run_dir / runner.PAPERS_DIRNAME).mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for index, slug in enumerate(slugs):
        folder = run_dir / runner.PAPERS_DIRNAME / slug
        folder.mkdir(parents=True, exist_ok=True)
        (folder / runner.PAPER_DESC_NAME).write_text(f"PAPER {slug}\n", encoding="utf-8")
        (folder / runner.SOURCE_TEXT_NAME).write_text(f"source text for {slug}", encoding="utf-8")
        records.append(
            {
                "paper_id": slug.split("__")[0],
                "slug": slug,
                "title": f"Title {index}",
                "source_uri": f"https://example.org/{slug}",
                "topic": "lipid A biosynthesis",
            }
        )
    runner.save_plan(
        run_dir,
        {
            "created_at": "2020-01-01T00:00:00",
            "modes": list(modes),
            "papers": records,
        },
    )
    return run_dir


def _row(slug: str, mode: str, status: str = "pass", **over: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "paper_id": slug.split("__")[0],
        "slug": slug,
        "mode": mode,
        "status": status,
        "stage": "pwml_export",
        "seconds": 12.5,
        "failure_kind": "",
        "message": "ok",
        "detail": "",
        "issue_codes": [],
        "counts": {"reactions": 3},
        "files": [{"name": "pathway.pwml", "bytes": 42}],
    }
    row.update(over)
    return row


class _FakeChild:
    """A stand-in for :func:`runner.launch_child`, scripted per call."""

    def __init__(self, script: Optional[Dict[Any, Any]] = None, default: str = "pass") -> None:
        self.script = dict(script or {})
        self.default = default
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, cmd: Sequence[str], timeout: float) -> runner.ChildResult:
        argv = list(cmd)
        slug = argv[argv.index("--slug") + 1]
        mode = argv[argv.index("--mode") + 1]
        self.calls.append({"slug": slug, "mode": mode, "timeout": timeout, "cmd": argv})
        behaviour = self.script.get((slug, mode), self.script.get(slug, self.default))
        if behaviour == "timeout":
            return runner.ChildResult(None, "partial output\n", "", True)
        if behaviour == "garbage":
            return runner.ChildResult(1, "not json at all\nTraceback (most recent call last)\n", "boom", False)
        if behaviour == "silent":
            return runner.ChildResult(0, "", "", False)
        status = "pass" if behaviour == "pass" else behaviour
        payload = json.dumps(_row(slug, mode, status=status))
        return runner.ChildResult(0, f"noise line\n{runner.OUTCOME_SENTINEL} {payload}\n", "", False)


def _fetch(*papers: BatchPaper, skipped: Optional[List[Dict[str, Any]]] = None):
    calls: List[Dict[str, Any]] = []

    def fetch_fn(text: str, limit: Any = None) -> Any:
        calls.append({"text": text, "limit": limit})
        return list(papers), list(skipped or [])

    fetch_fn.calls = calls  # type: ignore[attr-defined]
    return fetch_fn


def _no_fetch(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("a resumed run must never re-fetch papers")


def _paper(slug: str = "PMC9__gamma", text: str = "some paper body") -> BatchPaper:
    return BatchPaper(
        paper_id=slug.split("__")[0],
        title="A real title",
        source_uri="https://example.org/x",
        topic="menaquinone biosynthesis",
        full_text=text,
        slug=slug,
    )


def _manifest_rows(run_dir: Path) -> List[Dict[str, Any]]:
    raw = (run_dir / runner.MANIFEST_NAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------
def test_parse_modes_accepts_labels_slugs_and_junk() -> None:
    assert runner.parse_modes("strict,research") == [MODE_STRICT, MODE_RESEARCH]
    assert runner.parse_modes("research, strict") == [MODE_RESEARCH, MODE_STRICT]
    assert runner.parse_modes("Research (relaxed)") == [MODE_RESEARCH]
    assert runner.parse_modes("Strict PWML") == [MODE_STRICT]
    assert runner.parse_modes("strict strict") == [MODE_STRICT]
    # Unrecognised or empty must never mean "run nothing".
    assert runner.parse_modes("nonsense") == list(runner.ALL_MODES)
    assert runner.parse_modes(None) == list(runner.ALL_MODES)
    assert runner.parse_modes("") == list(runner.ALL_MODES)


def test_resolve_out_dir_is_project_relative_not_cwd_relative(tmp_path: Path) -> None:
    assert runner.resolve_out_dir(tmp_path) == tmp_path
    assert runner.resolve_out_dir("runs") == runner.PROJECT_ROOT / "runs"
    assert runner.resolve_out_dir(None) == runner.PROJECT_ROOT / "runs"


def test_new_run_dir_creates_layout_and_avoids_collisions(tmp_path: Path) -> None:
    from datetime import datetime

    when = datetime(2026, 7, 27, 22, 10)
    first = runner.new_run_dir(tmp_path, now=when)
    second = runner.new_run_dir(tmp_path, now=when)
    assert first.name == "2026-07-27_2210"
    assert second.name == "2026-07-27_2210-2"
    assert (first / runner.PAPERS_DIRNAME).is_dir()


def test_child_command_shape_and_grace_period(tmp_path: Path) -> None:
    cmd = runner.child_command(CLI_PATH, tmp_path, "PMC1__a", MODE_STRICT, 2700.0)
    assert cmd[0] == sys.executable
    assert cmd[1] == str(CLI_PATH)
    assert "--single" in cmd
    assert cmd[cmd.index("--slug") + 1] == "PMC1__a"
    assert cmd[cmd.index("--mode") + 1] == MODE_STRICT
    # The child gets a smaller budget than the parent's kill timeout so it can
    # write artifacts instead of being killed a second before it finishes.
    assert float(cmd[cmd.index("--timeout") + 1]) < 2700.0


def test_parse_child_output_prefers_the_sentinel(tmp_path: Path) -> None:
    good = json.dumps({"status": "pass", "mode": "strict"})
    stdout = f"streamlit chatter\n{{\"status\": \"fail\"}}\n{runner.OUTCOME_SENTINEL} {good}\n"
    assert runner.parse_child_output(stdout) == {"status": "pass", "mode": "strict"}
    # A bare JSON object still works (hand-run child), but garbage is None.
    assert runner.parse_child_output('{"status": "fail"}')["status"] == "fail"
    assert runner.parse_child_output("no json here") is None
    assert runner.parse_child_output("") is None
    assert runner.parse_child_output(runner.OUTCOME_SENTINEL + " {truncated") is None


def test_write_artifacts_is_utf8_and_cannot_escape_its_folder(tmp_path: Path) -> None:
    written = runner.write_artifacts(
        tmp_path,
        {
            "report.txt": "beta-hydroxymyristoyl → β-oxidation",
            "pathway.pwml": b"<pwml/>",
            "../escape.txt": "nope",
        },
    )
    names = {entry["name"] for entry in written}
    assert names == {"report.txt", "pathway.pwml", "escape.txt"}
    assert not (tmp_path.parent / "escape.txt").exists()
    assert "β-oxidation" in (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert (tmp_path / "pathway.pwml").read_bytes() == b"<pwml/>"


def test_snapshot_caches_copies_what_exists_and_ignores_what_does_not(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "id_mapping_cache.json").write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    copied = runner.snapshot_caches(run_dir, data_dir=data)
    assert copied == ["id_mapping_cache.json"]
    assert (run_dir / runner.CACHE_SNAPSHOT_DIRNAME / "id_mapping_cache.json").is_file()
    assert runner.snapshot_caches(tmp_path / "other", data_dir=tmp_path / "missing") == []


def test_result_text_states_verdict_stage_time_and_everything_about_a_failure() -> None:
    text = runner.result_text(
        _row(
            "PMC1__a",
            MODE_STRICT,
            status="fail",
            stage="post_pipeline",
            failure_kind="contract",
            message="post-pipeline validation failed",
            detail="Traceback: boom",
            issue_codes=["gate.forbidden_complex_reference"],
        )
    )
    assert text.startswith("RESULT: FAIL")
    assert "post_pipeline" in text
    assert "12.5s" in text
    assert "contract" in text
    assert "gate.forbidden_complex_reference" in text
    assert "Traceback: boom" in text
    assert runner.result_text(_row("PMC1__a", MODE_STRICT)).startswith("RESULT: PASS")


# ---------------------------------------------------------------------------
# Resume state.
# ---------------------------------------------------------------------------
def test_pending_and_completed_pairs_track_the_manifest(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    assert len(runner.pending_pairs(run_dir)) == 4
    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_STRICT))
    assert runner.completed_pairs(run_dir) == {("PMC1__alpha", MODE_STRICT)}
    assert ("PMC1__alpha", MODE_STRICT) not in runner.pending_pairs(run_dir)
    assert len(runner.pending_pairs(run_dir)) == 3


def test_find_resumable_skips_finished_and_plan_less_directories(tmp_path: Path) -> None:
    # A directory with no plan cannot be continued -- there is no work list.
    (tmp_path / "2019-01-01_0000").mkdir()
    finished = _make_run_dir(tmp_path, name="2021-01-01_0000", slugs=("PMC7__x",))
    for mode in (MODE_STRICT, MODE_RESEARCH):
        runner.append_manifest(finished, _row("PMC7__x", mode))
    assert runner.find_resumable(tmp_path) is None

    pending = _make_run_dir(tmp_path, name="2022-01-01_0000", slugs=("PMC8__y",))
    assert runner.find_resumable(tmp_path) == pending


def test_find_resumable_prefers_the_most_recent_incomplete_run(tmp_path: Path) -> None:
    _make_run_dir(tmp_path, name="2021-01-01_0000", slugs=("PMC1__a",))
    newer = _make_run_dir(tmp_path, name="2023-01-01_0000", slugs=("PMC2__b",))
    assert runner.find_resumable(tmp_path) == newer


# ---------------------------------------------------------------------------
# The child half.
# ---------------------------------------------------------------------------
def test_run_single_writes_the_mode_folder_and_returns_a_relative_row(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    seen: Dict[str, Any] = {}

    def fake_run_one(paper: Any, mode: Any, **kwargs: Any) -> RunOutcome:
        seen["paper"] = paper
        seen["mode"] = mode
        seen["timeout"] = kwargs.get("timeout")
        return RunOutcome(
            paper_id="PMC1",
            mode=MODE_STRICT,
            status="pass",
            stage="pwml_export",
            seconds=3.0,
            message="strict run completed",
            counts={"reactions": 4},
            artifacts={"pathway.pwml": b"<pwml/>", "notes.txt": "β-oxidation"},
        )

    row = runner.run_single(run_dir, "PMC1__alpha", MODE_STRICT, timeout=900.0, run_fn=fake_run_one)

    # The driver was handed the paper's real text, read back off disk.
    assert seen["paper"]["full_text"] == "source text for PMC1__alpha"
    assert seen["timeout"] == 900.0

    mode_dir = run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / MODE_STRICT
    assert (mode_dir / "pathway.pwml").read_bytes() == b"<pwml/>"
    assert "β" in (mode_dir / "notes.txt").read_text(encoding="utf-8")
    assert (mode_dir / runner.RESULT_NAME).read_text(encoding="utf-8").startswith("RESULT: PASS")

    assert row["slug"] == "PMC1__alpha"
    assert row["status"] == "pass"
    names = {entry["name"] for entry in row["files"]}
    assert names == {
        f"{runner.PAPERS_DIRNAME}/PMC1__alpha/{MODE_STRICT}/pathway.pwml",
        f"{runner.PAPERS_DIRNAME}/PMC1__alpha/{MODE_STRICT}/notes.txt",
    }


def test_emit_outcome_prints_exactly_one_parseable_line(capsys: Any) -> None:
    runner.emit_outcome(_row("PMC1__a", MODE_STRICT))
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    assert runner.parse_child_output(out[0])["slug"] == "PMC1__a"


# ---------------------------------------------------------------------------
# The parent loop: resume, fresh, timeout, crash, layout, exit codes.
# ---------------------------------------------------------------------------
def test_auto_continue_skips_done_work_and_never_refetches(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    for mode in (MODE_STRICT, MODE_RESEARCH):
        runner.append_manifest(run_dir, _row("PMC1__alpha", mode))

    child = _FakeChild()
    code = runner.run_batch(
        out=tmp_path,
        child_fn=child,
        fetch_fn=_no_fetch,  # asserts on call: a resume must not fetch
        script_path=CLI_PATH,
        echo=False,
    )

    assert code == 0
    assert [(call["slug"], call["mode"]) for call in child.calls] == [
        ("PMC2__beta", MODE_STRICT),
        ("PMC2__beta", MODE_RESEARCH),
    ]
    rows = _manifest_rows(run_dir)
    assert len(rows) == 4
    assert runner.pending_pairs(run_dir) == []
    # It continued the existing directory rather than creating a second one.
    assert [d.name for d in runner.list_run_dirs(tmp_path)] == [run_dir.name]
    assert "CONTINUING" in (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")


def test_fresh_starts_a_new_run_dir_and_leaves_the_old_one_alone(tmp_path: Path) -> None:
    old = _make_run_dir(tmp_path)
    old_manifest_before = (old / runner.MANIFEST_NAME).exists()

    fetch_fn = _fetch(_paper())
    child = _FakeChild()
    code = runner.run_batch(
        out=tmp_path,
        modes="strict",
        fresh=True,
        child_fn=child,
        fetch_fn=fetch_fn,
        script_path=CLI_PATH,
        data_dir=tmp_path / "no-data-here",
        echo=False,
    )

    assert code == 0
    dirs = [d for d in runner.list_run_dirs(tmp_path) if d != old]
    assert len(dirs) == 1
    new_dir = dirs[0]
    assert fetch_fn.calls  # type: ignore[attr-defined]
    assert [(c["slug"], c["mode"]) for c in child.calls] == [("PMC9__gamma", MODE_STRICT)]
    assert (old / runner.MANIFEST_NAME).exists() is old_manifest_before
    assert "fresh run" in (new_dir / runner.LOG_NAME).read_text(encoding="utf-8")


def test_a_stuck_paper_becomes_a_timeout_and_the_loop_keeps_going(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    child = _FakeChild(script={("PMC1__alpha", MODE_STRICT): "timeout"})

    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, timeout=1234.0, echo=False
    )

    assert code == 1
    # Every remaining pair still ran: one stuck paper must not eat the night.
    assert len(child.calls) == 4
    rows = {(r["slug"], r["mode"]): r for r in _manifest_rows(run_dir)}
    stuck = rows[("PMC1__alpha", MODE_STRICT)]
    assert stuck["status"] == "timeout"
    assert stuck["failure_kind"] == "timeout"
    assert "1234" in stuck["message"]
    assert rows[("PMC2__beta", MODE_RESEARCH)]["status"] == "pass"
    result = (run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / MODE_STRICT / runner.RESULT_NAME)
    assert result.read_text(encoding="utf-8").startswith("RESULT: FAIL")
    assert "timeout" in result.read_text(encoding="utf-8")


def test_a_child_that_prints_garbage_is_an_error_not_a_hang(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",), modes=(MODE_STRICT,))
    child = _FakeChild(script={"PMC1__alpha": "garbage"})

    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )

    assert code == 1
    row = _manifest_rows(run_dir)[0]
    assert row["status"] == "error"
    assert row["failure_kind"] == "crash"
    assert "exited with code 1" in row["message"]
    assert "not json at all" in row["detail"]
    assert "boom" in row["detail"]


def test_a_silent_child_is_also_an_error(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",), modes=(MODE_STRICT,))
    code = runner.run_batch(
        out=tmp_path,
        child_fn=_FakeChild(script={"PMC1__alpha": "silent"}),
        fetch_fn=_no_fetch,
        script_path=CLI_PATH,
        echo=False,
    )
    assert code == 1
    assert _manifest_rows(run_dir)[0]["status"] == "error"


def test_a_spawn_failure_is_one_bad_pair_not_a_dead_batch(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    calls: List[str] = []

    def child(cmd: Sequence[str], timeout: float) -> runner.ChildResult:
        argv = list(cmd)
        slug = argv[argv.index("--slug") + 1]
        mode = argv[argv.index("--mode") + 1]
        calls.append(slug)
        if len(calls) == 1:
            raise OSError("could not spawn python")
        return runner.ChildResult(
            0, f"{runner.OUTCOME_SENTINEL} {json.dumps(_row(slug, mode))}", "", False
        )

    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )
    assert code == 1
    rows = _manifest_rows(run_dir)
    assert len(rows) == 4
    assert rows[0]["status"] == "error"
    assert "could not spawn python" in rows[0]["detail"]


def test_the_run_directory_layout_is_exactly_the_agreed_one(tmp_path: Path) -> None:
    fetch_fn = _fetch(_paper(), skipped=[{"paper_id": "PMC5", "reason": "no_full_text"}])
    data = tmp_path / "data"
    data.mkdir()
    (data / "id_mapping_cache.json").write_text("{}", encoding="utf-8")
    (data / "enrichment_cache.json").write_text("{}", encoding="utf-8")

    code = runner.run_batch(
        out=tmp_path / "runs",
        child_fn=_FakeChild(),
        fetch_fn=fetch_fn,
        script_path=CLI_PATH,
        data_dir=data,
        echo=False,
    )
    assert code == 0

    run_dir = runner.latest_run_dir(tmp_path / "runs")
    assert run_dir is not None
    for name in (
        runner.SUMMARY_NAME,
        runner.FAILURES_NAME,
        runner.MANIFEST_NAME,
        runner.LOG_NAME,
        runner.SKIPPED_NAME,
        runner.PLAN_NAME,
    ):
        assert (run_dir / name).is_file(), name
    assert (run_dir / runner.CACHE_SNAPSHOT_DIRNAME / "id_mapping_cache.json").is_file()
    assert (run_dir / runner.CACHE_SNAPSHOT_DIRNAME / "enrichment_cache.json").is_file()

    paper_folder = run_dir / runner.PAPERS_DIRNAME / "PMC9__gamma"
    assert (paper_folder / runner.PAPER_DESC_NAME).is_file()
    assert (paper_folder / runner.SOURCE_TEXT_NAME).read_text(encoding="utf-8") == "some paper body"
    for mode in (MODE_STRICT, MODE_RESEARCH):
        assert (paper_folder / mode / runner.RESULT_NAME).is_file()

    assert json.loads((run_dir / runner.SKIPPED_NAME).read_text(encoding="utf-8"))[0]["reason"] == "no_full_text"
    summary = (run_dir / runner.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "SUMMARY" in summary
    assert "PMC9" in summary


def test_reports_are_regenerated_after_every_pair(tmp_path: Path, monkeypatch: Any) -> None:
    run_dir = _make_run_dir(tmp_path)
    snapshots: List[int] = []
    real_refresh = runner.refresh_reports

    def counting_refresh(target: Any, **kwargs: Any) -> None:
        real_refresh(target, **kwargs)
        path = Path(target) / runner.MANIFEST_NAME
        snapshots.append(len(_manifest_rows(Path(target))) if path.exists() else 0)

    monkeypatch.setattr(runner, "refresh_reports", counting_refresh)
    runner.run_batch(
        out=tmp_path, child_fn=_FakeChild(), fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )

    # Once before the loop, once per pair, once at the end.
    assert snapshots == [0, 1, 2, 3, 4, 4]
    assert (run_dir / runner.FAILURES_NAME).read_text(encoding="utf-8").strip()


def test_the_manifest_stays_valid_append_only_jsonl_across_sessions(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    first_child = _FakeChild(script={("PMC1__alpha", MODE_RESEARCH): "timeout"})

    # Session 1 runs both PMC1 pairs, then we pretend the machine died.
    runner.run_batch(
        out=tmp_path, child_fn=first_child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )
    first_lines = (run_dir / runner.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    assert len(first_lines) == 4

    # Session 2 must add nothing (all pairs recorded) and touch no earlier line.
    second_child = _FakeChild()
    runner.run_batch(
        out=tmp_path, child_fn=second_child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )
    second_lines = (run_dir / runner.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    assert second_lines[: len(first_lines)] == first_lines
    assert second_child.calls == []
    for line in second_lines:
        assert isinstance(json.loads(line), dict)


def test_partial_manifest_resumes_only_the_missing_pairs(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_STRICT))
    runner.append_manifest(run_dir, _row("PMC2__beta", MODE_RESEARCH, status="fail"))

    child = _FakeChild()
    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )
    assert {(c["slug"], c["mode"]) for c in child.calls} == {
        ("PMC1__alpha", MODE_RESEARCH),
        ("PMC2__beta", MODE_STRICT),
    }
    # The pre-existing failure still makes the whole run non-zero.
    assert code == 1


def test_modes_flag_overrides_the_plan_on_resume(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    child = _FakeChild()
    runner.run_batch(
        out=tmp_path, modes="research", child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )
    assert [c["mode"] for c in child.calls] == [MODE_RESEARCH]
    assert runner.load_plan(run_dir)["modes"] == [MODE_RESEARCH]


def test_no_papers_fetched_is_a_failed_night_not_a_silent_success(tmp_path: Path) -> None:
    code = runner.run_batch(
        out=tmp_path,
        fresh=True,
        child_fn=_FakeChild(),
        fetch_fn=_fetch(),
        script_path=CLI_PATH,
        echo=False,
    )
    assert code == 1
    run_dir = runner.latest_run_dir(tmp_path)
    assert run_dir is not None
    assert "NOTHING RAN" in (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")


def test_a_fetch_crash_is_recorded_rather_than_propagated(tmp_path: Path) -> None:
    def exploding_fetch(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("europepmc melted")

    code = runner.run_batch(
        out=tmp_path, fresh=True, child_fn=_FakeChild(), fetch_fn=exploding_fetch,
        script_path=CLI_PATH, echo=False,
    )
    assert code == 1
    run_dir = runner.latest_run_dir(tmp_path)
    skipped = json.loads((run_dir / runner.SKIPPED_NAME).read_text(encoding="utf-8"))
    assert skipped[0]["reason"] == "fetch_crashed"
    assert "europepmc melted" in skipped[0]["detail"]


def test_ctrl_c_leaves_a_valid_resumable_manifest(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    seen: List[str] = []

    def child(cmd: Sequence[str], timeout: float) -> runner.ChildResult:
        argv = list(cmd)
        slug = argv[argv.index("--slug") + 1]
        mode = argv[argv.index("--mode") + 1]
        seen.append(slug)
        if len(seen) == 2:
            raise KeyboardInterrupt
        return runner.ChildResult(
            0, f"{runner.OUTCOME_SENTINEL} {json.dumps(_row(slug, mode))}", "", False
        )

    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )
    assert code == 1
    rows = _manifest_rows(run_dir)  # would raise if a line were half-written
    assert len(rows) == 1
    assert len(runner.pending_pairs(run_dir)) == 3
    assert "Ctrl+C" in (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Timeout plumbing at the subprocess seam.
# ---------------------------------------------------------------------------
def test_launch_child_translates_timeoutexpired_and_kills_the_tree(monkeypatch: Any) -> None:
    killed: List[Any] = []

    class FakePopen:
        pid = 4242

        def __init__(self, *_a: Any, **_k: Any) -> None:
            self.returncode = None
            self._drained = False

        def communicate(self, timeout: Any = None) -> Any:
            if not self._drained:
                self._drained = True
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)
            return "leftover", "stderr tail"

        def kill(self) -> None:
            killed.append("kill")

    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runner, "_kill_tree", lambda proc: killed.append(proc.pid))

    result = runner.launch_child([sys.executable, "-c", "pass"], 1.0)
    assert result.timed_out is True
    assert result.returncode is None
    assert result.stdout == "leftover"
    assert killed == [4242]


def test_launch_child_really_kills_a_hung_process() -> None:
    """The one test that spawns a process -- ``python -c sleep``, no app, no net."""
    result = runner.launch_child(
        [sys.executable, "-c", "import time; time.sleep(120)"], 2.0
    )
    assert result.timed_out is True
    assert result.returncode is None


def test_launch_child_returns_a_real_childs_stdout() -> None:
    result = runner.launch_child(
        [sys.executable, "-c", "print('hello from the child')"], 60.0
    )
    assert result.timed_out is False
    assert result.returncode == 0
    assert "hello from the child" in result.stdout


# ---------------------------------------------------------------------------
# --status
# ---------------------------------------------------------------------------
def test_print_status_reports_the_most_recent_run_without_running_anything(tmp_path: Path) -> None:
    lines: List[str] = []
    assert runner.print_status(tmp_path, echo=lines.append) == 0
    assert "no runs found" in lines[0]

    run_dir = _make_run_dir(tmp_path)
    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_STRICT))
    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_RESEARCH, status="fail"))

    lines = []
    assert runner.print_status(tmp_path, echo=lines.append) == 0
    joined = "\n".join(lines)
    assert run_dir.name in joined
    assert "1/2 done" in joined
    assert "pending" in joined
    assert "RESEARCH DEFECT" in joined  # strict passed, research failed


# ---------------------------------------------------------------------------
# The CLI script itself.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cli() -> Any:
    spec = importlib.util.spec_from_file_location("t2pw_batch_run_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_match_the_documented_behaviour(cli: Any) -> None:
    args = cli.build_parser().parse_args([])
    assert args.topics is None and args.out is None and args.modes is None
    assert args.limit is None
    assert args.fresh is False and args.status is False and args.single is False
    assert args.timeout == runner.DEFAULT_PAPER_TIMEOUT
    assert args.deadline == runner.DEFAULT_DEADLINE_HOURS


def test_cli_passes_the_deadline_through(cli: Any, monkeypatch: Any) -> None:
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(runner, "run_batch", lambda **kwargs: captured.update(kwargs) or 0)
    assert cli.main(["--deadline", "6.5"]) == 0
    assert captured["deadline_hours"] == 6.5


def test_cli_forces_utf8_stdio_before_anything_can_print(cli: Any, monkeypatch: Any) -> None:
    """The child's result line is lost forever if it is printed to a cp1252 pipe."""

    order: List[str] = []
    monkeypatch.setattr(runner, "force_utf8_stdio", lambda: order.append("stdio"))
    monkeypatch.setattr(runner, "run_batch", lambda **_k: order.append("batch") or 0)
    assert cli.main([]) == 0
    assert order == ["stdio", "batch"]


def test_cli_passes_every_flag_through_to_run_batch(cli: Any, monkeypatch: Any, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(runner, "run_batch", lambda **kwargs: captured.update(kwargs) or 0)
    code = cli.main(
        [
            "--topics", "topics.txt",
            "--limit", "10",
            "--modes", "strict,research",
            "--out", str(tmp_path),
            "--timeout", "2700",
        ]
    )
    assert code == 0
    assert captured["topics_path"] == "topics.txt"
    assert captured["limit"] == 10
    assert captured["modes"] == "strict,research"
    assert captured["out"] == str(tmp_path)
    assert captured["timeout"] == 2700.0
    assert captured["fresh"] is False
    assert Path(captured["script_path"]).name == "batch_run.py"


def test_cli_status_exits_zero_without_running(cli: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(runner, "run_batch", lambda **_k: pytest.fail("--status must not run the batch"))
    monkeypatch.setattr(runner, "print_status", lambda out: 0)
    assert cli.main(["--status"]) == 0


def test_cli_single_requires_its_internal_flags(cli: Any) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--single"])
    assert excinfo.value.code == 2


def test_cli_single_runs_one_pair_and_emits_one_line(cli: Any, monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    monkeypatch.setattr(
        runner,
        "run_single",
        lambda rd, slug, mode, timeout=None: _row(slug, mode, status="fail"),
    )
    code = cli.main(["--single", "--run-dir", str(run_dir), "--slug", "PMC1__alpha", "--mode", "strict"])
    assert code == 1  # a failing pair makes the child exit non-zero
    printed = capsys.readouterr().out.strip().splitlines()
    assert len(printed) == 1
    assert runner.parse_child_output(printed[0])["status"] == "fail"


# ---------------------------------------------------------------------------
# Encoding at the pipe. The child's stdout is ALWAYS a pipe, and a piped
# sys.stdout.encoding is cp1252 on this machine -- so these tests must use a real
# subprocess. ``capsys`` is UTF-8 in-process and never caught the bug.
# ---------------------------------------------------------------------------
_BETA_ROW = {
    "paper_id": "PMC1",
    "slug": "PMC1__lipid_a",
    "mode": MODE_STRICT,
    "status": "pass",
    # The exact shape that killed it: a Greek beta in the canonical lipid A
    # intermediate, plus an em dash from a journal title.
    "message": "strict run completed for β-hydroxymyristoyl-ACP — 4210 bytes",
    "files": [{"name": "pathway.pwml", "bytes": 4210}],
}


def _cp1252_env() -> Dict[str, str]:
    """The parent environment, forced back to the codepage that breaks things."""

    env = dict(os.environ)
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def _run_child_script(body: str, tmp_path: Path, env: Dict[str, str]) -> subprocess.CompletedProcess:
    """Run ``body`` in a real child with real pipes and no text decoding."""

    script = tmp_path / "child_probe.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(SRC)!r})\nfrom t2pw.batch import runner\n{body}\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=120,
        check=False,
    )


def test_a_cp1252_pipe_really_does_break_a_plain_print(tmp_path: Path) -> None:
    """The reproduction itself, kept as a test so the fix cannot be "proved" by
    an environment that was never broken."""

    result = _run_child_script(
        "print('β-hydroxymyristoyl — em dash')", tmp_path, _cp1252_env()
    )
    assert result.returncode == 1
    assert b"UnicodeEncodeError" in result.stderr


def test_emit_outcome_survives_a_real_cp1252_pipe(tmp_path: Path) -> None:
    """The child row must reach the parent even with NO other mitigation.

    This is the belt-and-braces case: no ``force_utf8_stdio``, no ``PYTHONUTF8``
    in the environment. If this ever regresses, a paper that PASSED (artifacts and
    RESULT.txt already on disk) is recorded as a crash -- and because a row exists
    the pair is marked done and never retried.
    """

    result = _run_child_script(
        f"runner.emit_outcome({_BETA_ROW!r})", tmp_path, _cp1252_env()
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    row = runner.parse_child_output(result.stdout.decode("utf-8", "replace"))
    assert row is not None
    assert row["status"] == "pass"
    assert row["slug"] == "PMC1__lipid_a"
    assert "β-hydroxymyristoyl" in row["message"]
    assert "—" in row["message"]


def test_force_utf8_stdio_makes_a_piped_child_printable(tmp_path: Path) -> None:
    result = _run_child_script(
        "runner.force_utf8_stdio()\nprint('β-hydroxymyristoyl — em dash')",
        tmp_path,
        _cp1252_env(),
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "β-hydroxymyristoyl" in result.stdout.decode("utf-8", "replace")


def test_emit_outcome_falls_back_to_ascii_json_when_there_is_no_buffer(
    monkeypatch: Any,
) -> None:
    """A wrapped/captured stdout may expose no ``.buffer``.

    ``\\uXXXX`` escapes are still valid JSON, so the parent recovers the exact
    string -- it must be IMPOSSIBLE to lose a valid row to encoding.
    """

    import io

    class NoBuffer(io.StringIO):
        """A text stream with no ``.buffer`` attribute."""

    stream = NoBuffer()
    monkeypatch.setattr(sys, "stdout", stream)
    runner.emit_outcome(dict(_BETA_ROW))
    text = stream.getvalue()

    assert "β" not in text  # pure ASCII on the wire
    row = runner.parse_child_output(text)
    assert row is not None
    assert row["message"] == _BETA_ROW["message"]  # ...and it decodes back exactly


def test_child_env_forces_utf8_without_discarding_the_parents_environment(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("T2PW_SOME_API_KEY", "from-dotenv")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    env = runner.child_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    # The child needs every .env-derived variable: replacing the environment
    # would make it fail for a completely unrelated reason.
    assert env["T2PW_SOME_API_KEY"] == "from-dotenv"
    assert os.environ["PYTHONIOENCODING"] == "cp1252"  # ours is untouched


def test_launch_child_gives_the_child_a_utf8_stdout(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    result = runner.launch_child(
        [sys.executable, "-c", "print('\\u03b2-oxidation')"], 60.0
    )
    assert result.returncode == 0, result.stderr
    assert "β-oxidation" in result.stdout


# ---------------------------------------------------------------------------
# Logging may never be fatal.
# ---------------------------------------------------------------------------
def _cp1252_print(monkeypatch: Any) -> List[str]:
    """Replace ``print`` with one that refuses non-cp1252 text, as a console does."""

    printed: List[str] = []

    def fake_print(*args: Any, **_kwargs: Any) -> None:
        text = " ".join(str(item) for item in args)
        text.encode("cp1252")  # UnicodeEncodeError -- which is a ValueError
        printed.append(text)

    monkeypatch.setattr("builtins.print", fake_print)
    return printed


def test_logger_survives_a_console_that_cannot_encode_the_message(
    tmp_path: Path, monkeypatch: Any
) -> None:
    printed = _cp1252_print(monkeypatch)
    log = runner.Logger(tmp_path / "run", echo=True)

    log("  + PMC1__lipid_a  β-oxidation of 3-hydroxymyristoyl—ACP")  # must not raise

    assert printed, "the fallback never printed anything"
    assert "oxidation" in printed[-1]
    assert "\\u03b2" in printed[-1]  # ascii-safe rendering, not the raw beta
    # The log FILE is utf-8, so it keeps the real characters.
    assert "β-oxidation" in (tmp_path / "run" / runner.LOG_NAME).read_text(encoding="utf-8")


def test_a_greek_paper_title_cannot_kill_the_night_before_it_starts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """``_plan_for_fresh_run`` logs every fetched title verbatim.

    Before the fix that print died on a Greek letter, and it died *after*
    ``new_run_dir`` and *before* ``save_plan`` -- so the directory had no
    plan.json, ``find_resumable`` refused it, and the next launch made another
    empty one. Zero papers run, every night, forever.
    """

    _cp1252_print(monkeypatch)
    paper = BatchPaper(
        paper_id="PMC9",
        title="β-oxidation of 3-hydroxymyristoyl—ACP in lipid A biosynthesis",
        source_uri="https://example.org/x",
        topic="lipid A biosynthesis",
        full_text="some paper body",
        slug="PMC9__lipid_a",
    )
    child = _FakeChild()

    code = runner.run_batch(
        out=tmp_path,
        fresh=True,
        modes="strict",
        child_fn=child,
        fetch_fn=_fetch(paper),
        script_path=CLI_PATH,
        data_dir=tmp_path / "no-data-here",
        echo=True,  # the dangerous path: echo is what raised
    )

    assert code == 0
    run_dir = runner.latest_run_dir(tmp_path)
    assert run_dir is not None
    assert (run_dir / runner.PLAN_NAME).is_file(), "the run directory was orphaned"
    assert runner.load_plan(run_dir)["papers"][0]["slug"] == "PMC9__lipid_a"
    assert [call["slug"] for call in child.calls] == ["PMC9__lipid_a"]
    assert "β-oxidation" in (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")


def test_an_unexpected_crash_leaves_a_resumable_dir_not_an_orphan(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("something nobody predicted")

    monkeypatch.setattr(runner, "_plan_for_fresh_run", boom)
    code = runner.run_batch(
        out=tmp_path, fresh=True, child_fn=_FakeChild(), fetch_fn=_fetch(_paper()),
        script_path=CLI_PATH, echo=False,
    )
    assert code == 1
    dead = runner.latest_run_dir(tmp_path)
    assert dead is not None
    assert (dead / runner.PLAN_NAME).is_file()
    log = (dead / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "UNEXPECTED ERROR" in log
    assert "something nobody predicted" in log

    # And the next launch does real work instead of finding an orphan it refuses.
    monkeypatch.undo()
    child = _FakeChild()
    runner.run_batch(
        out=tmp_path, modes="strict", child_fn=child, fetch_fn=_fetch(_paper()),
        script_path=CLI_PATH, data_dir=tmp_path / "nope", echo=False,
    )
    assert [call["slug"] for call in child.calls] == ["PMC9__gamma"]


# ---------------------------------------------------------------------------
# Manifest durability.
# ---------------------------------------------------------------------------
def test_append_manifest_repairs_a_truncated_tail_instead_of_fusing_onto_it(
    tmp_path: Path,
) -> None:
    """A hard reboot mid-write leaves the last line without its newline.

    Appending onto it fused the two rows: the previous row became unparseable AND
    the new one was invisible to ``completed_pairs``, so the pair was re-run on
    every resume, appending another mangled row each time.
    """

    run_dir = _make_run_dir(tmp_path)
    manifest = run_dir / runner.MANIFEST_NAME
    first = _row("PMC1__alpha", MODE_STRICT)
    manifest.write_text(json.dumps(first), encoding="utf-8")  # no trailing newline
    assert not manifest.read_text(encoding="utf-8").endswith("\n")

    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_RESEARCH))

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]  # BOTH rows must load
    assert rows[0]["mode"] == MODE_STRICT
    assert rows[1]["mode"] == MODE_RESEARCH
    # And both are visible as resume state, so neither pair is re-run.
    assert runner.completed_pairs(run_dir) == {
        ("PMC1__alpha", MODE_STRICT),
        ("PMC1__alpha", MODE_RESEARCH),
    }
    assert len(runner.pending_pairs(run_dir)) == 2


def test_append_manifest_does_not_add_a_blank_line_to_a_healthy_file(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)
    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_STRICT))
    runner.append_manifest(run_dir, _row("PMC1__alpha", MODE_RESEARCH))
    raw = (run_dir / runner.MANIFEST_NAME).read_text(encoding="utf-8")
    assert raw.count("\n") == 2
    assert "\n\n" not in raw


# ---------------------------------------------------------------------------
# Resume must not reach back into the archive.
# ---------------------------------------------------------------------------
def _age_dir(run_dir: Path, *, hours: float) -> None:
    when = time.time() - hours * 3600.0
    for target in (run_dir / runner.MANIFEST_NAME, run_dir):
        if target.exists():
            os.utime(target, (when, when))


def test_find_resumable_resumes_the_newest_recent_pending_run(tmp_path: Path) -> None:
    recent = _make_run_dir(tmp_path, name="2020-01-01_0000", slugs=("PMC1__a",))
    notes: List[str] = []
    assert runner.find_resumable(tmp_path, now=time.time(), log=notes.append) == recent
    joined = "\n".join(notes)
    assert f"resuming {recent.name}" in joined
    assert "pending" in joined


def test_find_resumable_ignores_a_run_older_than_the_max_age(tmp_path: Path) -> None:
    stale = _make_run_dir(tmp_path, name="2020-01-01_0000", slugs=("PMC1__a",))
    _age_dir(stale, hours=30 * 24)
    notes: List[str] = []

    assert runner.find_resumable(tmp_path, now=time.time(), log=notes.append) is None

    joined = "\n".join(notes)
    assert f"ignoring {stale.name}" in joined
    assert "older than 24h" in joined


def test_find_resumable_starts_fresh_when_the_newest_run_is_complete(tmp_path: Path) -> None:
    """The exact trap: an ancient aborted run still owes work, so it used to beat
    last night's finished run and every night re-ran one stale paper forever."""

    ancient = _make_run_dir(tmp_path, name="2020-01-01_0000", slugs=("PMC_OLD__x",))
    _age_dir(ancient, hours=40 * 24)
    newest = _make_run_dir(tmp_path, name="2020-06-01_0100", slugs=("PMC_NEW__y",))
    for mode in (MODE_STRICT, MODE_RESEARCH):
        runner.append_manifest(newest, _row("PMC_NEW__y", mode))
    notes: List[str] = []

    assert runner.find_resumable(tmp_path, now=time.time(), log=notes.append) is None

    joined = "\n".join(notes)
    assert f"the newest run {newest.name} is complete" in joined
    assert ancient.name not in joined  # it is never even considered


def test_run_batch_starts_fresh_rather_than_continuing_a_stale_run(tmp_path: Path) -> None:
    stale = _make_run_dir(tmp_path, name="2020-01-01_0000", slugs=("PMC_OLD__x",))
    _age_dir(stale, hours=48)
    child = _FakeChild()

    code = runner.run_batch(
        out=tmp_path, modes="strict", child_fn=child, fetch_fn=_fetch(_paper()),
        script_path=CLI_PATH, data_dir=tmp_path / "nope", echo=False,
    )

    assert code == 0
    assert [call["slug"] for call in child.calls] == ["PMC9__gamma"]
    fresh = runner.latest_run_dir(tmp_path)
    assert fresh is not None and fresh != stale
    log = (fresh / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "older than 24h" in log
    assert stale.name in log
    # The stale run was left completely alone.
    assert not (stale / runner.MANIFEST_NAME).exists()


# ---------------------------------------------------------------------------
# A valid child row is never discarded.
# ---------------------------------------------------------------------------
def test_a_row_printed_just_before_the_kill_is_ground_truth(tmp_path: Path) -> None:
    """The child finished, wrote its artifacts and printed its row a moment before
    the parent's stopwatch expired. That row is the truth; a synthesized timeout
    row would mark the pair done with a fabricated failure, never to be retried."""

    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",), modes=(MODE_STRICT,))
    payload = json.dumps(_row("PMC1__alpha", MODE_STRICT, status="pass"))

    def child(cmd: Sequence[str], timeout: float) -> runner.ChildResult:
        return runner.ChildResult(
            None, f"chatter\n{runner.OUTCOME_SENTINEL} {payload}\n", "", True
        )

    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )

    assert code == 0
    row = _manifest_rows(run_dir)[0]
    assert row["status"] == "pass"
    assert row["failure_kind"] == ""
    assert row["files"], "the child's own artifact list was thrown away"
    log = (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "had already printed its result line" in log


# ---------------------------------------------------------------------------
# A pass must be a pass: artifacts on disk, deliverables named.
# ---------------------------------------------------------------------------
def _outcome(mode: str, **over: Any) -> RunOutcome:
    kwargs: Dict[str, Any] = {
        "paper_id": "PMC1",
        "mode": mode,
        "status": "pass",
        "stage": "pwml_export",
        "seconds": 3.0,
        "message": "run completed",
        "artifacts": {"pathway.pwml": b"<pwml/>", "notes.txt": "fine"},
    }
    kwargs.update(over)
    return RunOutcome(**kwargs)


def _break_writes(monkeypatch: Any, *names: str) -> None:
    """Make ``write_artifacts`` report a failed write for ``names``."""

    real = runner.write_artifacts

    def broken(target: Any, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = real(target, {k: v for k, v in artifacts.items() if k not in names})
        return rows + [
            {"name": name, "bytes": 0, "error": "OSError: [Errno 28] No space left on device"}
            for name in names
            if name in artifacts
        ]

    monkeypatch.setattr(runner, "write_artifacts", broken)


def test_a_failed_required_artifact_write_demotes_the_pass_to_a_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    _break_writes(monkeypatch, "pathway.pwml")

    row = runner.run_single(
        run_dir, "PMC1__alpha", MODE_STRICT,
        run_fn=lambda *_a, **_k: _outcome(MODE_STRICT),
    )

    assert row["status"] == "fail"
    assert row["failure_kind"] == "crash"
    assert "pathway.pwml" in row["message"]
    assert "No space left" in row["message"]
    result = (
        run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / MODE_STRICT / runner.RESULT_NAME
    ).read_text(encoding="utf-8")
    assert result.startswith("RESULT: FAIL")
    assert "WRITE FAILED" in result
    assert "No space left" in result


def test_a_failed_optional_artifact_write_is_reported_but_not_a_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    _break_writes(monkeypatch, "notes.txt")

    row = runner.run_single(
        run_dir, "PMC1__alpha", MODE_STRICT,
        run_fn=lambda *_a, **_k: _outcome(MODE_STRICT),
    )

    assert row["status"] == "pass"
    result = (
        run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / MODE_STRICT / runner.RESULT_NAME
    ).read_text(encoding="utf-8")
    assert "WRITE FAILED" in result  # still never silent


def test_a_failed_research_report_write_demotes_the_research_pass(
    tmp_path: Path, monkeypatch: Any
) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    _break_writes(monkeypatch, "research_pathway_report.txt")

    row = runner.run_single(
        run_dir, "PMC1__alpha", MODE_RESEARCH,
        run_fn=lambda *_a, **_k: _outcome(
            MODE_RESEARCH,
            stage="research_report",
            artifacts={"research_pathway_report.txt": "report body", "review_flags.json": "{}"},
        ),
    )

    assert row["status"] == "fail"
    assert "research_pathway_report.txt" in row["message"]


def test_required_artifacts_only_demands_what_the_driver_produced(tmp_path: Path) -> None:
    # Research mode legitimately has no PWML, and a research run with no RAG
    # synthesis legitimately has no report -- neither absence is a write failure.
    assert runner.required_artifacts(MODE_STRICT, {"pathway.pwml": b"x"}) == ["pathway.pwml"]
    assert runner.required_artifacts(MODE_STRICT, {"notes.txt": "x"}) == []
    assert runner.required_artifacts(MODE_RESEARCH, {"review_flags.json": "{}"}) == []


def test_a_pass_with_warnings_is_never_reported_as_all_green(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",), modes=(MODE_RESEARCH,))
    warning = "no research report: RAG did not trigger for this paper"

    def child(cmd: Sequence[str], timeout: float) -> runner.ChildResult:
        payload = json.dumps(
            _row("PMC1__alpha", MODE_RESEARCH, warnings=[warning], files=[], counts={})
        )
        return runner.ChildResult(0, f"{runner.OUTCOME_SENTINEL} {payload}\n", "", False)

    code = runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    )

    # The pipeline genuinely ran, so the pair passed and the exit code stays 0 --
    # but nothing in the reports may call this night clean.
    assert code == 0
    assert _manifest_rows(run_dir)[0]["warnings"] == [warning]
    log = (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "ALL GREEN" not in log
    assert "PASSED WITH WARNINGS" in log
    assert warning in log
    result = (
        run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / MODE_RESEARCH / runner.RESULT_NAME
    ).read_text(encoding="utf-8")
    assert result.startswith("RESULT: PASS (with warnings)")
    assert warning in result
    summary = (run_dir / runner.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "no research deliverable" in summary


# ---------------------------------------------------------------------------
# A missing input file is infrastructure, not biology.
# ---------------------------------------------------------------------------
def test_a_missing_source_text_file_is_an_infrastructure_failure(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    (run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / runner.SOURCE_TEXT_NAME).unlink()
    started: List[str] = []

    row = runner.run_single(
        run_dir, "PMC1__alpha", MODE_STRICT,
        run_fn=lambda *_a, **_k: started.append("app") or _outcome(MODE_STRICT),
    )

    assert started == [], "the app must never be started with no input file"
    assert row["status"] == "error"
    assert row["failure_kind"] == "crash"  # infrastructure, not "no_reactions"
    assert runner.SOURCE_TEXT_NAME in row["message"]
    result = (
        run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / MODE_STRICT / runner.RESULT_NAME
    ).read_text(encoding="utf-8")
    assert "could not be read" in result


def test_an_empty_but_present_source_text_is_still_a_biology_outcome(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    (run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / runner.SOURCE_TEXT_NAME).write_text(
        "", encoding="utf-8"
    )
    seen: List[Any] = []

    runner.run_single(
        run_dir, "PMC1__alpha", MODE_STRICT,
        run_fn=lambda paper, mode, **_k: seen.append(paper) or _outcome(MODE_STRICT),
    )

    # Present-but-empty is a paper with no usable text: that judgement belongs to
    # the driver, so it is still handed the paper.
    assert seen and seen[0]["full_text"] == ""
    assert "source_text_error" not in seen[0]


def test_paper_from_plan_separates_unreadable_from_empty(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, slugs=("PMC1__alpha",))
    source = run_dir / runner.PAPERS_DIRNAME / "PMC1__alpha" / runner.SOURCE_TEXT_NAME
    assert "source_text_error" not in runner.paper_from_plan(run_dir, "PMC1__alpha")
    source.unlink()
    record = runner.paper_from_plan(run_dir, "PMC1__alpha")
    assert record["full_text"] == ""
    assert record["source_text_error"]
    assert record["source_text_path"].endswith(runner.SOURCE_TEXT_NAME)


# ---------------------------------------------------------------------------
# The whole-night deadline.
# ---------------------------------------------------------------------------
def _stepping_clock(step: float) -> Any:
    """A monotonic-looking clock that advances ``step`` seconds per reading."""

    state = {"t": 0.0}

    def clock() -> float:
        now = state["t"]
        state["t"] += step
        return now

    return clock


def test_the_deadline_stops_the_night_cleanly_and_leaves_it_resumable(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)  # 2 papers x 2 modes = 4 pairs
    child = _FakeChild()

    code = runner.run_batch(
        out=tmp_path,
        child_fn=child,
        fetch_fn=_no_fetch,
        script_path=CLI_PATH,
        echo=False,
        deadline_hours=2.0,
        clock=_stepping_clock(3600.0),  # one hour per clock reading
    )

    assert code == 1
    assert len(child.calls) == 1, "the deadline did not stop the loop"
    log = (run_dir / runner.LOG_NAME).read_text(encoding="utf-8")
    assert "deadline reached, 3 pair(s) not attempted" in log
    assert "DEADLINE REACHED" in log
    # Stopping cleanly means the summary is written and the rest can finish later.
    assert (run_dir / runner.SUMMARY_NAME).is_file()
    assert len(runner.pending_pairs(run_dir)) == 3

    # And the remainder really does finish on the next launch.
    second = _FakeChild()
    assert runner.run_batch(
        out=tmp_path, child_fn=second, fetch_fn=_no_fetch, script_path=CLI_PATH, echo=False
    ) == 0
    assert len(second.calls) == 3
    assert runner.pending_pairs(run_dir) == []


def test_a_generous_deadline_does_not_interfere(tmp_path: Path) -> None:
    _make_run_dir(tmp_path)
    child = _FakeChild()
    assert runner.run_batch(
        out=tmp_path, child_fn=child, fetch_fn=_no_fetch, script_path=CLI_PATH,
        echo=False, deadline_hours=runner.DEFAULT_DEADLINE_HOURS,
    ) == 0
    assert len(child.calls) == 4


def test_repo_has_the_double_click_entry_point() -> None:
    bat = (ROOT / "run_overnight.bat").read_text(encoding="utf-8", errors="replace").lower()
    assert "scripts\\batch_run.py" in bat
    assert "pause" in bat
    assert 'cd /d "%~dp0"' in bat
    topics = (ROOT / "topics.txt").read_text(encoding="utf-8")
    assert topics.lstrip().startswith("#")
    from t2pw.batch.fetch import parse_topics

    assert len(parse_topics(topics)) >= 2
