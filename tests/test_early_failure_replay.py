"""Offline replay of stored early-failure shapes through the real Stage-0/1 path.

WHAT THIS HARNESS IS. The unit suites around it check one component each. This
one runs the WHOLE early pipeline -- ``preprocess`` -> ``run_stage_one_with_chunking``
-> ``settle_stage_one`` -- against a canned provider, once per stored shape in
``tests/fixtures/early_failures/cases.json``, and asserts two things per shape:

  * the deliverable: does the run recover a usable pathway, or report a clear
    incomplete outcome? Both are correct results; a fabricated reaction is not,
    so every case pins its reaction count.
  * the diagnostics: do ``stage0_attempts.json``,
    ``extraction_boundary_report.json`` and ``cleaning_report.json`` exist on
    disk, with the model, finish_reason, attempt count, counts and reasons that
    say WHY -- including on the shapes that produce no payload at all.

WHY REPLAY RATHER THAN A LIVE RUN. The two legs lost in ``runs/2026-07-28_0919``
wrote ``files: []`` and nothing but ``RESULT.txt``, so there is no recorded
transcript to replay and no way to re-run them deterministically. What can be
stored is the SHAPE of each failure, and a shape replayed offline is reproducible
on every machine, costs nothing, and cannot pass because a provider happened to
behave that day.

Nothing here touches the network, streamlit, or a real model.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ``t2pw.llm.client`` builds a provider client at import time. Per the repo
# convention, any test that pulls it in stubs ``openai`` itself so it runs in
# isolation rather than only when an earlier test happened to stub it first.
if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *_: object, **__: object) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **__: None)
            )

    openai_stub.OpenAI = _OpenAI
    openai_stub.RateLimitError = RuntimeError
    openai_stub.APIError = RuntimeError
    openai_stub.APITimeoutError = RuntimeError
    openai_stub.AuthenticationError = RuntimeError
    openai_stub.BadRequestError = RuntimeError
    sys.modules["openai"] = openai_stub

from t2pw.llm import client as client_mod  # noqa: E402
from t2pw.pipeline.extraction_diagnostics import (  # noqa: E402
    BOUNDARY_REPORT_NAME,
    CLEANING_REPORT_NAME,
    STAGE0_ATTEMPTS_NAME,
    activate,
    deactivate,
)
from t2pw.pipeline.pipeline import (  # noqa: E402
    PipelineFailure,
    run_stage_one_with_chunking,
)
from t2pw.pipeline.preprocessor import PREPROCESS_STATUS_KEY, preprocess  # noqa: E402
from t2pw.pipeline.stage_one_boundary import settle_stage_one  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "early_failures" / "cases.json"
CASES: List[Dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
CASES_BY_ID = {case["id"]: case for case in CASES}


# ---------------------------------------------------------------------------
# The canned provider.
# ---------------------------------------------------------------------------
def _resp(content: Any, *, finish_reason: str = "stop") -> Any:
    """A minimal stand-in for an openai ChatCompletion.

    ``usage=None`` so ``_record_usage`` returns early and the module's global
    token counters stay untouched by these replays.
    """

    message = types.SimpleNamespace(content=content, tool_calls=None)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=None)


class _Completions:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                f"the replay made {len(self.calls)} provider call(s) but the case "
                f"scripted only {len(self.calls) - 1}"
            )
        return self._responses.pop(0)


class _Client:
    def __init__(self, responses: List[Any]) -> None:
        self.completions = _Completions(responses)
        self.chat = types.SimpleNamespace(completions=self.completions)


class Replay:
    """One case's result: what the pipeline produced and what it wrote down."""

    def __init__(
        self,
        *,
        context: Dict[str, Any],
        payload: Dict[str, Any],
        boundary: Any,
        failure: Any,
        provider_calls: int,
        artifact_dir: Path,
    ) -> None:
        self.context = context
        self.payload = payload
        self.boundary = boundary
        self.failure = failure
        self.provider_calls = provider_calls
        self.artifact_dir = artifact_dir

    def artifact(self, name: str) -> Dict[str, Any]:
        path = self.artifact_dir / name
        assert path.exists(), f"{name} was not written; found {sorted(p.name for p in self.artifact_dir.iterdir())}"
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def written(self) -> List[str]:
        return sorted(p.name for p in self.artifact_dir.iterdir() if p.is_file())

    @property
    def reactions(self) -> List[Dict[str, Any]]:
        processes = self.payload.get("processes") if isinstance(self.payload, dict) else None
        rows = (processes or {}).get("reactions") if isinstance(processes, dict) else None
        return rows if isinstance(rows, list) else []

    @property
    def recovers(self) -> bool:
        """A run 'recovers' when the boundary is settled AND a pathway exists.

        Both halves are required. A settled boundary over an empty payload is not
        a recovery -- it is the empty result reported honestly -- and conflating
        the two is how a pipeline starts calling nothing a success.
        """

        return bool(self.boundary is not None and self.boundary.ok and self.reactions)


def _run_case(case: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Replay:
    """Replay one stored shape through the real Stage-0 -> Stage-1 -> boundary path."""

    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_RETRY_BASE_SLEEP", "0")
    monkeypatch.setenv("LLM_RETRY_MAX_SLEEP", "0")
    monkeypatch.setenv("LLM_CALL_SPACING", "0")
    # Replace the module's own `time` reference rather than stdlib's, so nothing
    # outside t2pw.llm.client is affected and the replay runs at full speed.
    monkeypatch.setattr(client_mod, "time", types.SimpleNamespace(sleep=lambda *_: None))

    client = _Client(
        [
            _resp(reply.get("content", ""), finish_reason=reply.get("finish_reason", "stop"))
            for reply in case["replies"]
        ]
    )
    monkeypatch.setattr(client_mod, "_client", client)

    artifact_dir = tmp_path / "diagnostics"
    activate(run_id=case["id"], artifact_dir=artifact_dir)

    text = case["source_text"]
    context = preprocess(text)

    payload: Dict[str, Any] = {}
    boundary = None
    failure: Any = None
    try:
        payload, _chunks = run_stage_one_with_chunking(
            text,
            pathway_context=context,
            enable_chunking=False,
            max_attempts=2,
        )
    except PipelineFailure as exc:
        failure = exc

    if failure is None:
        boundary = settle_stage_one(
            payload,
            cleaning_report=_last_cleaning_pass(artifact_dir),
            repair_rows=False,
        )
        payload = boundary.payload

    return Replay(
        context=context,
        payload=payload,
        boundary=boundary,
        failure=failure,
        provider_calls=len(client.completions.calls),
        artifact_dir=artifact_dir,
    )


def _last_cleaning_pass(artifact_dir: Path) -> Dict[str, Any]:
    path = artifact_dir / CLEANING_REPORT_NAME
    if not path.exists():
        return {}
    passes = json.loads(path.read_text(encoding="utf-8")).get("passes") or []
    return passes[-1] if passes else {}


@pytest.fixture(autouse=True)
def _clean_recorder():
    deactivate()
    yield
    deactivate()


# ---------------------------------------------------------------------------
# The properties every stored shape must satisfy.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_every_stored_shape_replays_to_its_recorded_verdict(
    case: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery, reaction count and provider cost, per stored shape."""

    replay = _run_case(case, tmp_path, monkeypatch)
    expect = case["expect"]

    assert replay.recovers is expect["recovers"], (
        f"{case['id']}: expected recovers={expect['recovers']}, "
        f"got boundary_ok="
        f"{None if replay.boundary is None else replay.boundary.ok} "
        f"with {len(replay.reactions)} reaction(s)"
    )
    assert len(replay.reactions) == expect["reactions"]
    # The cost matters as much as the verdict: a shape that recovers by spending
    # LLM_MAX_RETRIES draws on an identical prompt has not been fixed.
    assert replay.provider_calls == expect["provider_calls"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_every_stored_shape_leaves_its_diagnostics_on_disk(
    case: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifacts exist for the shapes that fail, not only the ones that pass.

    This is the 2026-07-28 property inverted: those legs died with ``files: []``.
    """

    replay = _run_case(case, tmp_path, monkeypatch)

    for name in case["expect"]["artifacts"]:
        assert name in replay.written, f"{case['id']} did not write {name}"

    stage0 = replay.artifact(STAGE0_ATTEMPTS_NAME)["attempts"][0]
    assert stage0["status"] == case["expect"]["stage0_status"]
    # The facts the 2026-07-28 postmortem had to infer are now recorded.
    for key in ("model", "finish_reason", "attempts", "response_status", "boundary"):
        assert key in stage0, key

    boundaries = replay.artifact(BOUNDARY_REPORT_NAME)["boundaries"]
    assert boundaries, f"{case['id']} recorded no extraction boundary"
    for record in boundaries:
        for key in ("stage", "boundary", "outcome", "attempts"):
            assert key in record, key


@pytest.mark.parametrize(
    "case", CASES, ids=[case["id"] for case in CASES]
)
def test_no_stored_shape_ever_fabricates_a_reaction(
    case: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every delivered reaction traces to a reaction the provider actually sent.

    The rule requirement 6 states: repair may fail, and when it does the run
    reports an incomplete outcome rather than padding the payload to satisfy
    ``entities_required`` or ``processes_required``. Checked by name against the
    canned replies, which are the only source of truth in a replay.
    """

    replay = _run_case(case, tmp_path, monkeypatch)
    scripted = " ".join(reply.get("content", "") for reply in case["replies"])

    for reaction in replay.reactions:
        assert reaction.get("name", "") in scripted
        for role in ("inputs", "outputs"):
            for participant in reaction.get(role, []):
                assert participant in scripted, (
                    f"{case['id']} delivered participant {participant!r} that no "
                    "provider reply ever mentioned"
                )


# ---------------------------------------------------------------------------
# Shape-specific assertions: what makes each one worth storing.
# ---------------------------------------------------------------------------
def test_an_empty_stage0_draw_is_retried_and_the_cost_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CASES_BY_ID["stage0_empty_completion_then_retry"]
    replay = _run_case(case, tmp_path, monkeypatch)

    stage0 = replay.artifact(STAGE0_ATTEMPTS_NAME)["attempts"][0]
    assert stage0["attempts"] == case["expect"]["stage0_attempts"]
    assert stage0["status"] == "ok"
    assert replay.context["pathway_name"] == "Glutathione biosynthesis"


def test_a_stage0_moderation_stop_spends_one_draw_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable-exit property, end to end.

    Before 2026-07-29 this shape cost LLM_MAX_RETRIES identical draws and left no
    record of why. Now it costs one, the reason reaches the artifact, and Stage 1
    still runs unguided rather than the run dying at Stage 0.
    """

    case = CASES_BY_ID["stage0_content_filter_is_durable"]
    replay = _run_case(case, tmp_path, monkeypatch)

    stage0 = replay.artifact(STAGE0_ATTEMPTS_NAME)["attempts"][0]
    assert stage0["attempts"] == 1
    assert stage0["terminal_reason"] == "content_filter"
    assert stage0["finish_reason"] == "content_filter"
    assert stage0["status"] == "empty_reply"
    assert replay.context[PREPROCESS_STATUS_KEY]["status"] == "empty_reply"
    # Unguided, but not dead: the extraction still produced the reaction.
    assert len(replay.reactions) == 1


def test_a_malformed_reaction_row_is_repaired_rather_than_salvaged_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One missing comma must not cost the reaction that follows it.

    The free prefix salvage returns this body's entities and drops its processes
    entirely -- measured, not assumed. Localized repair recovers the reaction, and
    the boundary report shows both the ``invalid_json`` attempt and the repair
    that followed, so the salvage-shaped loss is visible even when it is fixed.
    """

    case = CASES_BY_ID["stage1_invalid_json_localized_repair"]
    replay = _run_case(case, tmp_path, monkeypatch)

    assert len(replay.reactions) == 1
    assert replay.reactions[0]["outputs"] == ["gamma-glutamylcysteine"]

    boundaries = replay.artifact(BOUNDARY_REPORT_NAME)["boundaries"]
    kinds = [record["boundary"] for record in boundaries]
    assert "stage1_extraction" in kinds
    assert "stage1_json_repair" in kinds
    invalid = next(r for r in boundaries if r["outcome"] == "invalid_json")
    assert "JSONDecodeError" in invalid["error"]
    # Exactly one repair draw: the cap held.
    assert kinds.count("stage1_json_repair") == 1


def test_zero_processes_and_all_discarded_are_told_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two shapes behind the identical terminal message, now separated.

    ``Payload must include a processes object`` was the last word on both of the
    2026-07-28 legs. These two cases reach zero reactions from opposite causes,
    and every distinguishing fact now lives in the artifacts.

    Both now CLEAR the structural contract, and that is the point rather than a
    regression. ``processes: {}`` is structurally valid -- the payload has the
    key, of the right type -- so a structural gate is the wrong place to reject
    it, and rejecting there is what made one message stand for five different
    causes. Emptiness is judged downstream, where the cause is known: the batch
    driver fails the leg as ``no_reactions``, and strict quarantine refuses it as
    ``no_surviving_process``. What must survive here is that the two causes stay
    distinguishable, which they now are at the level of the boundary OUTCOME --
    previously both collapsed to ``contract_failed_after_cleaning``.
    """

    empty = _run_case(CASES_BY_ID["stage1_valid_json_zero_processes"], tmp_path / "a", monkeypatch)
    discarded = _run_case(
        CASES_BY_ID["stage1_reactions_all_discarded_by_cleaning"], tmp_path / "b", monkeypatch
    )

    # Structurally valid either way, and empty either way. Neither invents a row
    # to satisfy a gate -- that is the property requirement 6 actually protects.
    assert empty.boundary.ok is True
    assert discarded.boundary.ok is True
    assert empty.reactions == []
    assert discarded.reactions == []

    # ...and the causes are told apart, which is what this test is for.
    assert empty.boundary.outcome == "valid_json_zero_processes"
    assert discarded.boundary.outcome == "processes_discarded_by_cleaning"

    # The model declared nothing: raw count zero, nothing discarded.
    empty_clean = _last_cleaning_pass(empty.artifact_dir)
    assert empty_clean["raw_process_counts"]["total"] == 0
    assert empty_clean["all_processes_discarded"] is False
    # The flag that CAN fire for zero-in-zero-out; all_processes_discarded
    # cannot, because it requires raw > 0.
    assert empty_clean["no_processes_declared"] is True
    assert any(
        record["outcome"] == "valid_json_zero_processes"
        for record in empty.artifact(BOUNDARY_REPORT_NAME)["boundaries"]
    )

    # The model declared three and cleaning removed all three, with the rules named.
    discarded_clean = _last_cleaning_pass(discarded.artifact_dir)
    assert discarded_clean["raw_process_counts"]["reactions"] == 3
    assert discarded_clean["cleaned_process_counts"]["total"] == 0
    assert discarded_clean["all_processes_discarded"] is True
    for reason in CASES_BY_ID["stage1_reactions_all_discarded_by_cleaning"]["expect"][
        "discard_reasons"
    ]:
        assert discarded_clean["discarded_by_reason"][reason] == 1
    assert discarded_clean["no_processes_declared"] is False

    # The rules that removed the rows are named for the discarded case and there
    # are none to name for the empty one. This used to be read off
    # ``incomplete_reason``, which only existed because the boundary failed; the
    # cleaning report carries the same facts whether or not anything failed, so
    # the distinction no longer depends on something going wrong.
    assert empty_clean["discarded_by_reason"] == {}
    assert sum(discarded_clean["discarded_by_reason"].values()) == 3


def test_an_empty_result_is_delivered_empty_rather_than_padded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 6, end to end: no fabrication, artifacts intact.

    Renamed from ``..._unrecoverable_shape_reports_incomplete_...``. The shape is
    no longer unrecoverable at the stage contract: a reply that reads the source
    and finds no chemistry is a legitimate answer, and the source here is one
    sentence about glutathione that describes no reaction at all. The payload is
    now delivered as an empty pathway instead of being rejected as malformed.

    What requirement 6 actually forbids is UNCHANGED and is what this asserts:
    the run must not pad the payload with an invented reaction to satisfy
    ``processes_required``. Zero reactions in, zero reactions out.
    """

    case = CASES_BY_ID["stage1_valid_json_zero_processes"]
    replay = _run_case(case, tmp_path, monkeypatch)

    assert replay.boundary.ok is True
    assert replay.boundary.outcome == case["expect"]["boundary_outcome"]
    # The container is present and empty -- the shape that used to be
    # unrepresentable, and the reason the negative control could not pass.
    assert replay.boundary.payload["processes"] == {}
    assert replay.boundary.payload["entities"]
    # Nothing was invented to fill it.
    assert replay.reactions == []
    for name in (STAGE0_ATTEMPTS_NAME, BOUNDARY_REPORT_NAME, CLEANING_REPORT_NAME):
        assert name in replay.written


def test_missing_registry_rows_become_unresolved_shells_not_lost_reactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent loss, closed.

    Both reactions are correct and evidenced; only the registry was short. The
    shells carry no identifier and no function, so every identity gate downstream
    still applies -- the incompleteness is made explicit, not laundered away.
    """

    case = CASES_BY_ID["stage1_participants_missing_from_the_registry"]
    replay = _run_case(case, tmp_path, monkeypatch)

    assert len(replay.reactions) == 2
    entities = replay.payload["entities"]
    shells = {
        row["name"]: row
        for bucket in entities.values()
        if isinstance(bucket, list)
        for row in bucket
        if isinstance(row, dict) and row.get("resolution_status") == "unresolved"
    }
    assert set(case["expect"]["reconstructed_shells"]) == set(shells)
    for row in shells.values():
        assert row["provenance"] == "reconstructed_from_process_participant"
        assert "mapped_ids" not in row
        assert "uniprot" not in row
        assert "ec_number" not in row
    # The name that WAS declared is untouched -- no duplicate shell for it.
    assert entities["compounds"][0] == {"name": "L-glutamate"}


def test_the_stored_shapes_stay_small_enough_to_read() -> None:
    """A fixture that grows into a payload dump stops being a fixture.

    The real merged payloads from that run are 306 KB - 4.7 MB and carry verbatim
    source passages; the point of storing shapes is that they do not.
    """

    assert FIXTURE.stat().st_size < 24_000
    assert len(CASES) >= 6
    assert len({case["id"] for case in CASES}) == len(CASES)
    for case in CASES:
        assert len(case["source_text"]) < 600, case["id"]
        for key in ("title", "why_it_matters", "observed_in", "expect"):
            assert case.get(key), f"{case['id']} is missing {key}"
