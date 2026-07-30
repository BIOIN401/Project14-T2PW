"""ADVERSARIAL GROUP 4 — two runs at once, and one enormous value.

TWO PROBLEMS, ONE FILE, because they share a cause: the diagnostics recorder is
reached ambiently and fed arbitrary caller data, so both "whose run is this?" and
"how big is this?" have to be answered *at the recorder*, not at each call site.

ISOLATION. Streamlit serves every browser session from one process, on its own
ScriptRunner thread. A module-level recorder — the first implementation — would
have been shared by all of them: session B's ``activate`` discarding session A's
boundaries mid-run, both writing the same filenames, and the artifacts ending up
a blend of two papers with nothing saying so. That is worse than no artifact,
because it looks authoritative. The recorder is a ``ContextVar`` and every
invocation gets its own directory; this file runs two extractions concurrently,
synchronised on a barrier so their writes genuinely interleave, and proves
neither can see or overwrite the other.

BOUNDING. ``record_boundary(**extra)``, ``record_cleaning``,
``record_mapped_failure`` and ``record_iteration`` all take caller-shaped data
this module never sees the shape of. One recursive sanitizer runs at every one of
those doors. The test here is deliberately hostile: a 150 KB string buried three
levels down in every entry point at once, asserted against the bytes actually
written to disk.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    AUDIT_ITERATIONS_NAME,
    BOUNDARY_REPORT_NAME,
    BOUNDARY_STAGE1_EXTRACTION,
    CLEANING_REPORT_NAME,
    GAP_ITERATIONS_NAME,
    MAPPED_FAILURE_SNAPSHOT_NAME,
    MAX_BOUND_DEPTH,
    MAX_BOUND_ITEMS,
    MAX_BOUND_KEYS,
    STAGE0_ATTEMPTS_NAME,
    ExtractionDiagnostics,
    activate,
    bound,
    current,
    deactivate,
    descriptor,
)


@pytest.fixture(autouse=True)
def _clean_recorder():
    deactivate()
    yield
    deactivate()


# ---------------------------------------------------------------------------
# Isolation.
# ---------------------------------------------------------------------------
def test_two_concurrent_runs_cannot_see_or_overwrite_each_other(tmp_path: Path) -> None:
    """The two-session case, with the interleaving forced rather than hoped for.

    Both threads activate, then wait on a barrier, then record. The barrier is
    what makes this a real test: without it thread A could finish before thread B
    starts and a shared global would pass. With it, both recorders are live at
    the same moment, which is exactly the situation a process-global variable
    cannot survive.
    """

    base = tmp_path / "diagnostics"
    barrier = threading.Barrier(2)
    results: Dict[str, Dict[str, Any]] = {}
    errors: List[BaseException] = []

    def _run(label: str) -> None:
        try:
            recorder = activate(run_id=label, artifact_dir=base, unique=True)
            barrier.wait(timeout=10)
            for index in range(20):
                recorder.record_boundary(
                    stage="extraction",
                    boundary=BOUNDARY_STAGE1_EXTRACTION,
                    model=f"model-{label}",
                    attempts=index,
                    note=f"{label}-{index}",
                )
            barrier.wait(timeout=10)
            results[label] = {
                "dir": recorder.artifact_dir,
                "boundaries": recorder.boundaries,
                # Read back through the AMBIENT accessor, not the local handle:
                # this is what every recording site in the pipeline actually
                # uses, so it is what has to be isolated.
                "ambient_is_mine": current() is recorder,
                "ambient_boundaries": len(current().boundaries),
            }
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(label,)) for label in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert set(results) == {"A", "B"}

    # Each thread saw its own recorder through the ambient accessor.
    assert results["A"]["ambient_is_mine"] is True
    assert results["B"]["ambient_is_mine"] is True
    assert results["A"]["ambient_boundaries"] == 20
    assert results["B"]["ambient_boundaries"] == 20

    # No boundary crossed over.
    for label, other in (("A", "B"), ("B", "A")):
        models = {record["model"] for record in results[label]["boundaries"]}
        assert models == {f"model-{label}"}
        assert f"model-{other}" not in models
        assert len(results[label]["boundaries"]) == 20

    # And no file was shared: two directories, each with its own report.
    assert results["A"]["dir"] != results["B"]["dir"]
    report_a = json.loads(
        (Path(results["A"]["dir"]) / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8")
    )
    report_b = json.loads(
        (Path(results["B"]["dir"]) / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8")
    )
    assert report_a["run_id"] == "A"
    assert report_b["run_id"] == "B"
    assert len(report_a["boundaries"]) == 20
    assert "model-B" not in json.dumps(report_a)
    assert "model-A" not in json.dumps(report_b)


def test_a_worker_thread_cannot_write_into_the_parent_runs_artifacts(tmp_path: Path) -> None:
    """A ContextVar is not inherited by a new thread, and that is the point.

    A helper that offloads work to its own thread records into a fresh detached
    recorder rather than the run's. The cost is a lost diagnostic from that
    helper; the alternative is a worker writing into whichever run happened to be
    active, which is the contamination this design refuses.
    """

    recorder = activate(run_id="parent", artifact_dir=tmp_path)
    recorder.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION, note="parent")

    seen: Dict[str, Any] = {}

    def _worker() -> None:
        child = current()
        child.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION, note="child")
        seen["is_parent"] = child is recorder
        seen["artifact_dir"] = child.artifact_dir

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join(timeout=10)

    assert seen["is_parent"] is False
    assert seen["artifact_dir"] is None  # detached: it writes nothing
    notes = [record.get("note") for record in recorder.boundaries]
    assert notes == ["parent"]
    assert "child" not in (tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8")


def test_every_invocation_gets_its_own_directory(tmp_path: Path) -> None:
    """Same base, two runs, two directories. No file is ever a shared target."""

    first = activate(run_id="one", artifact_dir=tmp_path, unique=True)
    second = activate(run_id="two", artifact_dir=tmp_path, unique=True)

    assert first.artifact_dir != second.artifact_dir
    assert first.artifact_dir.parent == tmp_path
    assert second.artifact_dir.parent == tmp_path

    first.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION, note="one")
    second.record_boundary(stage="extraction", boundary=BOUNDARY_STAGE1_EXTRACTION, note="two")

    assert "one" in (first.artifact_dir / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8")
    assert "one" not in (second.artifact_dir / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8")


def test_a_fixed_directory_is_still_available_for_a_caller_that_wants_one(
    tmp_path: Path,
) -> None:
    """``unique`` is opt-in: a replay or a test naming a path still gets it."""

    recorder = activate(run_id="fixed", artifact_dir=tmp_path)

    assert recorder.artifact_dir == tmp_path


def test_deactivate_only_affects_the_calling_context(tmp_path: Path) -> None:
    recorder = activate(run_id="mine", artifact_dir=tmp_path)
    seen: Dict[str, Any] = {}

    def _worker() -> None:
        deactivate()
        seen["worker_recorder"] = current()

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join(timeout=10)

    assert current() is recorder
    assert seen["worker_recorder"] is not recorder


# ---------------------------------------------------------------------------
# Recursive bounding.
# ---------------------------------------------------------------------------
#: 150 KB, and shaped like the thing that actually gets this big: a payload
#: evidence field carrying verbatim source prose.
HUGE = "Lipid A biosynthesis proceeds via LpxA in Escherichia coli. " * 2600


def _nested(payload: Any, depth: int = 3) -> Any:
    """``payload`` buried ``depth`` levels down, under innocuous keys."""

    node = payload
    for level in range(depth):
        node = {f"level_{level}": node}
    return node


def test_the_fixture_really_is_150kb() -> None:
    """A bounding test that quietly stopped being hostile proves nothing."""

    assert len(HUGE) > 150_000


def test_a_150kb_value_nested_three_deep_is_bounded_in_every_artifact(
    tmp_path: Path,
) -> None:
    """The hostile case: the same blob through every entry point at once.

    Asserted on the BYTES ON DISK, not on an in-memory object, because the file
    is what costs disk and what a reviewer opens. Each of these doors takes
    caller-shaped data, and before one shared sanitizer existed each one bounded
    only the shapes somebody had thought of.
    """

    recorder = activate(run_id="huge", artifact_dir=tmp_path)

    recorder.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_EXTRACTION,
        error=HUGE,
        raw_preview=HUGE,
        raw_chars=len(HUGE),
        # The open door: an arbitrary extra key holding a deeply nested blob.
        rejected_sample=_nested({"evidence": HUGE, "note": HUGE}),
        attempt_log=[{"attempt": 1, "raw": HUGE, "detail": _nested(HUGE)}],
    )
    recorder.record_stage0_attempt({"status": "ok", "raw": HUGE, "context": _nested(HUGE)})
    recorder.record_cleaning({"label": "merged", "discarded_sample": _nested({"evidence": HUGE})})
    recorder.record_mapped_failure({"stage": "post_mapping", "payload": _nested(HUGE)})
    recorder.record_iteration("audit", {"round": 1, "candidates": _nested(HUGE)})
    recorder.record_iteration("gap", {"round": 1, "db": _nested({"text": HUGE})})

    written = sorted(path.name for path in tmp_path.iterdir())
    assert set(written) == {
        STAGE0_ATTEMPTS_NAME,
        BOUNDARY_REPORT_NAME,
        CLEANING_REPORT_NAME,
        MAPPED_FAILURE_SNAPSHOT_NAME,
        AUDIT_ITERATIONS_NAME,
        GAP_ITERATIONS_NAME,
    }
    for name in written:
        size = (tmp_path / name).stat().st_size
        assert size < 8_000, f"{name} is {size} bytes"
        text = (tmp_path / name).read_text(encoding="utf-8")
        # The blob itself never appears; only a bounded head of it can.
        assert HUGE not in text
        assert len(text) < 8_000


def test_the_size_of_what_was_elided_is_still_reported(tmp_path: Path) -> None:
    """Bounded is not the same as hidden.

    Every elision states what it dropped, so a reviewer can tell a 200-character
    reply from a 150 KB one without either being stored.
    """

    recorder = activate(artifact_dir=tmp_path)
    recorder.record_boundary(
        stage="extraction",
        boundary=BOUNDARY_STAGE1_EXTRACTION,
        raw_chars=len(HUGE),
        rejected_sample={"evidence": HUGE},
    )

    record = json.loads((tmp_path / BOUNDARY_REPORT_NAME).read_text(encoding="utf-8"))[
        "boundaries"
    ][0]
    assert record["raw_chars"] == len(HUGE)
    assert record["rejected_sample"]["evidence"]["chars"] == len(HUGE)
    assert record["rejected_sample"]["evidence"]["hash"].startswith("sha256:")
    assert record["rejected_sample"]["evidence"]["preview"]


def test_a_bulk_key_becomes_a_descriptor_at_any_depth() -> None:
    """Depth does not launder an evidence field."""

    for depth in range(0, MAX_BOUND_DEPTH):
        result = bound(_nested({"evidence": HUGE}, depth))
        rendered = json.dumps(result)
        assert HUGE not in rendered
        assert "sha256:" in rendered


def test_a_long_string_under_an_ordinary_key_is_clipped_not_described() -> None:
    """A human-readable reason must stay readable.

    ``incomplete_reason`` runs to a few hundred characters and is the single most
    useful field in these artifacts; turning it into a hash would bound it and
    destroy it.
    """

    result = bound({"incomplete_reason": "x" * 900})

    assert isinstance(result["incomplete_reason"], str)
    assert result["incomplete_reason"].startswith("xxx")
    assert "elided" in result["incomplete_reason"]
    assert len(result["incomplete_reason"]) < 400


def test_width_is_capped_on_mappings_and_sequences() -> None:
    wide_map = bound({f"k{index}": index for index in range(200)})
    wide_list = bound([index for index in range(200)])

    assert len(wide_map) == MAX_BOUND_KEYS + 1  # +1 for the elision marker
    assert "more key(s) elided" in wide_map["…"]
    assert len(wide_list) == MAX_BOUND_ITEMS + 1
    assert "more item(s) elided" in wide_list[-1]


def _structural_depth(value: Any) -> int:
    """How many nested containers deep ``value`` goes."""

    if isinstance(value, dict):
        return 1 + max((_structural_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_structural_depth(item) for item in value), default=0)
    return 0


def test_depth_is_capped_and_the_tail_becomes_a_descriptor() -> None:
    """Ten levels in, at most ``MAX_BOUND_DEPTH`` levels of container out.

    Measured on the STRUCTURE, not on the rendered text: the descriptor that
    replaces the tail carries a preview of what it replaced, so the elided keys
    still appear as characters. That is intended -- the preview is what makes the
    shape recognisable -- and it is not nesting.
    """

    result = bound(_nested({"a": 1}, MAX_BOUND_DEPTH + 4))

    assert "hash" in json.dumps(result)
    # +1 for the descriptor dict that terminates the walk.
    assert _structural_depth(result) <= MAX_BOUND_DEPTH + 1
    assert _structural_depth(_nested({"a": 1}, MAX_BOUND_DEPTH + 4)) > MAX_BOUND_DEPTH + 1


def test_a_self_referential_structure_cannot_hang_the_recorder(tmp_path: Path) -> None:
    """A cycle is a plausible accident in a report built by a loop.

    The depth cap is what stops it, and a diagnostics writer that hangs on one
    takes out the run it was diagnosing.
    """

    cycle: Dict[str, Any] = {"name": "loop"}
    cycle["self"] = cycle

    recorder = activate(artifact_dir=tmp_path)
    recorder.record_cleaning({"label": "cycle", "row": cycle})

    text = (tmp_path / CLEANING_REPORT_NAME).read_text(encoding="utf-8")
    assert "loop" in text
    assert len(text) < 4_000


def test_the_sanitizer_never_raises_on_an_unserializable_value(tmp_path: Path) -> None:
    class _Odd:
        def __repr__(self) -> str:
            return "<odd " + "x" * 5000 + ">"

    recorder = activate(artifact_dir=tmp_path)
    recorder.record_iteration("audit", {"round": 1, "thing": _Odd(), "set": {1, 2, 3}})

    text = (tmp_path / AUDIT_ITERATIONS_NAME).read_text(encoding="utf-8")
    assert len(text) < 4_000
    assert "odd" in text


def test_descriptor_reports_container_shape_too() -> None:
    assert descriptor({"a": 1, "b": 2})["keys"] == 2
    assert descriptor([1, 2, 3])["items"] == 3
    assert descriptor("abc")["chars"] == 3


def test_bounding_leaves_a_small_report_alone(tmp_path: Path) -> None:
    """The sanitizer must be invisible on the payloads it is not there for.

    A cleaning report's counts, flags and sample entries are the whole point of
    the artifact; a depth cap that censured them would make every file useless to
    protect against a case that never happens in it.
    """

    report = {
        "label": "stage_one_merged",
        "raw_process_counts": {"reactions": 3, "total": 3},
        "cleaned_process_counts": {"total": 0},
        "all_processes_discarded": True,
        "discarded_by_reason": {"reaction_no_usable_outputs": 3},
        "discarded_sample": {
            "reaction_no_usable_outputs": [
                {"pointer": "/processes/reactions/0", "name": "r1"},
                {"pointer": "/processes/reactions/1", "name": "r2"},
            ]
        },
    }

    recorder = activate(artifact_dir=tmp_path)
    stored = recorder.record_cleaning(report)

    assert stored == report
    on_disk = json.loads((tmp_path / CLEANING_REPORT_NAME).read_text(encoding="utf-8"))
    assert on_disk["passes"][0] == report


def test_a_detached_recorder_bounds_just_the_same() -> None:
    """Bounding is a property of the recorder, not of having a directory."""

    recorder = ExtractionDiagnostics()
    recorder.record_cleaning({"label": "x", "blob": {"evidence": HUGE}})

    rendered = json.dumps(recorder.artifacts(), default=str)
    assert HUGE not in rendered
    assert len(rendered) < 4_000
