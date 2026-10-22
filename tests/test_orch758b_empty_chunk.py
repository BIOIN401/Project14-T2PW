"""ORCH-758b (defect D3): a chunk that correctly says nothing no longer fails the paper.

WHY. ``ORCH-758-RESULT.md`` § 4 D3: PMC10425814 is chunked into 4 and chunk 3 is 90 %
reference list. Every call on every draw (``normal``, then ``structurally_empty_repair``)
returned valid JSON with 0 entities and 0 processes, ``finish_reason=stop``. The draw
raised "structurally empty payload", ``run_ensemble`` re-raised once every draw had, and
``run_stage_one_with_chunking`` turned that one chunk into a Stage-1 failure for the whole
paper: "Chunk 3 failed to produce valid JSON". Two of four WNT legs died that way.

Every test drives the real ``run_stage_one_with_chunking`` -> ``run_extraction_pipeline``
-> (``run_ensemble``) -> ``_run_json_stage`` path with only the provider call faked
(``chat_detailed``, in ``pipeline`` and, for the localized JSON repair, in the client).
Each sentence of the input is its own chunk (``chunk_word_limit=5``, no overlap), and the
fake answers by the marker word in the chunk text.

G9, BEHAVIOURAL (``test_g9_*``): fail on base ``3582b0d7`` because the empty chunk raises
"Chunk 2 failed to produce valid JSON."; pass at the tip.

NEW ACCEPTANCE TEST, D-104 AMENDMENT 2 (``test_llm_inputs_*``): the empty chunk's record
reaches no LLM input. With the empty chunk first, in the middle or last, the merged Stage-1
payload, the locked-reaction manifest and every Stage-1 and Stage-2 prompt of the other
chunks are byte-identical to a run of the same paper without that chunk's text, and no
prompt carries the record. It cannot pass on base: there the run dies at the empty chunk.

PRESERVATION (``test_still_*``): pass on base and at the tip. Invalid JSON on every
attempt, a ``length`` stop, an empty draw mixed with a broken one, every chunk empty, and
the unchunked path all fail exactly as before. Repair round 1 added three that the round-0
tip ``a9179eb8`` failed: a ``content_filter`` stop, a retry refused as ``budget_exhausted``
by a live leg deadline (both stay fatal), and a failed all-empty run leaving no record of
an "accepted" empty chunk.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import t2pw.llm.client as client_mod  # noqa: E402
import t2pw.pipeline.extraction_diagnostics as diag  # noqa: E402
import t2pw.pipeline.pipeline as pipeline_mod  # noqa: E402
from t2pw.llm.client import CompletionDiagnostics, CompletionResult  # noqa: E402
from t2pw.pipeline.pipeline import (  # noqa: E402
    NOTE_EMPTY_PAYLOAD_EXHAUSTED,
    NOTE_EMPTY_PAYLOAD_RETRY,
    PipelineFailure,
    clean_stage_one,
    merge_stage_one_outputs,
    run_stage_one_with_chunking,
    run_stage_two_with_chunking,
)
from t2pw.pipeline.reaction_lock_manifest import MANIFEST_FILENAME, write_stage1_lock_artifacts  # noqa: E402
from t2pw.pipeline.stage_one_ensemble import ENV_VAR, REPORT_NAME  # noqa: E402
from t2pw.pipeline.deadline import BUDGET_EXHAUSTED, LegDeadline, price_llm_call  # noqa: E402
from t2pw.pipeline.extraction_ladder import activate_leg_deadline, deactivate_leg_deadline  # noqa: E402

SENTENCE = {
    "MARKA": "Glucose is phosphorylated MARKA here.",
    "MARKB": "Only references follow MARKB here.",
    "MARKC": "Fructose is phosphorylated MARKC here.",
}


def _reaction_reply(substrate: str, product: str, name: str, marker: str) -> str:
    return json.dumps({
        "entities": {"compounds": [
            {"name": substrate, "evidence": SENTENCE[marker]},
            {"name": product, "evidence": SENTENCE[marker]},
        ]},
        "processes": {"reactions": [
            {"name": name, "inputs": [substrate], "outputs": [product], "evidence": SENTENCE[marker]},
        ]},
    })


REPLY_A = _reaction_reply("glucose", "glucose 6-phosphate", "glucose phosphorylation", "MARKA")
REPLY_C = _reaction_reply("fructose 6-phosphate", "fructose 1,6-bisphosphate", "fructose phosphorylation", "MARKC")
#: Two different valid, structurally empty replies, as production's 1,153-1,249-char ones were.
EMPTY_1 = json.dumps({"entities": {"proteins": [], "compounds": []}, "processes": {"reactions": []}})
EMPTY_2 = json.dumps({"entities": {}, "processes": {}})
NOT_JSON = "the reply is not json at all"
STOP, LENGTH = "stop", "length"
EMPTY_DRAW = [(EMPTY_1, STOP), (EMPTY_2, STOP)]
#: Strings only the empty-chunk record carries; none may reach a model.
RECORD_STRINGS = ("empty_chunk", "stage1_empty_chunk", "structurally empty", "valid_json_zero_processes")


class Transport:
    """Fake ``chat_detailed``. Answers each call from the script of the marker its user
    message contains (replies served in order, the last one repeated); a call naming no
    marker -- a localized JSON repair -- gets unparseable text."""

    def __init__(self, script: Dict[str, Sequence[Tuple[str, str]]]) -> None:
        self.script = {marker: list(replies) for marker, replies in script.items()}
        self.calls: List[Tuple[str, str, str]] = []
        #: Called with the marker after each call is recorded (the budget test's clock).
        self.on_call: Any = None

    def __call__(self, messages: Any, **_kwargs: Any) -> CompletionResult:
        user = str(messages[-1]["content"])
        system = str(messages[0]["content"]) if len(messages) > 1 else ""
        marker = next((m for m in SENTENCE if m in user), "")
        served = sum(1 for call in self.calls if call[0] == marker)
        self.calls.append((marker, system, user))
        if self.on_call is not None:
            self.on_call(marker)
        replies = self.script.get(marker) or [("still not json", STOP)]
        text, finish = replies[min(served, len(replies) - 1)]
        return CompletionResult(
            text, CompletionDiagnostics(model="test-model", stage="test", attempts=1, finish_reason=finish))

    def count(self, marker: str) -> int:
        return sum(1 for call in self.calls if call[0] == marker)

    def prompts(self, marker: str) -> List[Tuple[str, str]]:
        return [(system, user) for m, system, user in self.calls if m == marker]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    diag.deactivate()
    yield
    diag.deactivate()


def _install(monkeypatch: pytest.MonkeyPatch, script: Dict[str, Sequence[Tuple[str, str]]]) -> Transport:
    transport = Transport(script)
    monkeypatch.setattr(pipeline_mod, "chat_detailed", transport)
    # The localized JSON repair imports the client's chat_detailed lazily: fake it too, so
    # no test here can reach a provider.
    monkeypatch.setattr(client_mod, "chat_detailed", transport)
    return transport


def _text(*markers: str) -> str:
    return " ".join(SENTENCE[m] for m in markers)


def _stage_one(text: str, artifact_dir: Any = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return run_stage_one_with_chunking(
        text, enable_chunking=True, chunk_word_limit=5, chunk_overlap=0, max_attempts=2,
        artifact_dir=artifact_dir)


def _reaction_names(payload: Dict[str, Any]) -> List[str]:
    return [row["name"] for row in (payload.get("processes") or {}).get("reactions") or []]


DRAWS = pytest.mark.parametrize("draws", [1, 3], ids=["1-draw", "3-draws"])


# ----------------------------------------------------------------------------- G9


@DRAWS
def test_g9_an_empty_chunk_contributes_nothing_and_the_paper_goes_on(
        draws: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, str(draws))
    transport = _install(monkeypatch, {"MARKA": [(REPLY_A, STOP)], "MARKB": EMPTY_DRAW * draws,
                                       "MARKC": [(REPLY_C, STOP)]})
    recorder = diag.activate(run_id="orch758b", artifact_dir=tmp_path)

    merged, chunks = _stage_one(_text("MARKA", "MARKB", "MARKC"))

    assert _reaction_names(merged) == ["glucose phosphorylation", "fructose phosphorylation"]
    assert [c["chunk_id"] for c in chunks] == [1, 2, 3]
    assert (transport.count("MARKA"), transport.count("MARKB"), transport.count("MARKC")) == (
        draws, 2 * draws, draws)  # no extra call: retries and budgets are unchanged

    empty = chunks[1]
    assert "MARKB" in empty["text"]
    assert _reaction_names(empty["output"]) == []
    assert [a["note"] for a in empty["attempts"]] == [
        NOTE_EMPTY_PAYLOAD_RETRY, NOTE_EMPTY_PAYLOAD_EXHAUSTED] * draws
    if draws > 1:
        assert [a["draw"] for a in empty["attempts"]] == [1, 1, 2, 2, 3, 3]
    record = empty["empty_chunk"]
    assert record["chunk_id"] == 2 and record["draws"] == draws
    assert [a["finish_reason"] for a in record["attempts"]] == [STOP] * (2 * draws)
    assert "structurally empty payload" in record["failure"]
    assert all("empty_chunk" not in c for c in (chunks[0], chunks[2]))

    [boundary] = [b for b in recorder.boundaries if b["boundary"] == "stage1_empty_chunk"]
    assert boundary["empty_chunk"]["chunk_id"] == 2
    assert boundary["outcome"] == "valid_json_zero_processes"
    assert boundary["attempts"] == 2 * draws

    if draws > 1:
        entries = json.loads((tmp_path / REPORT_NAME).read_text(encoding="utf-8"))
        # Repair round 1: written once the run is known not to fail, so after the other chunks'.
        assert [e["draws_succeeded"] for e in entries] == [3, 3, 0]
        assert entries[2]["chunk_id"] == 2 and entries[2]["empty_chunk"] is True
        assert [d["draw"] for d in entries[2]["failed_draws"]] == [1, 2, 3]
        assert "0 of 3 succeeded" in (tmp_path / REPORT_NAME).with_suffix(".txt").read_text(encoding="utf-8")
    else:
        assert not (tmp_path / REPORT_NAME).exists()


@DRAWS
def test_g9_every_chunk_empty_fails_as_an_empty_extraction_after_every_chunk_was_read(
        draws: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(draws))
    transport = _install(monkeypatch, {m: EMPTY_DRAW * draws for m in SENTENCE})

    with pytest.raises(PipelineFailure) as excinfo:
        _stage_one(_text("MARKA", "MARKB", "MARKC"))

    failure = excinfo.value
    assert failure.stage == "extraction"
    assert str(failure).startswith("Extraction stage returned a structurally empty payload")
    assert "every one of the 3 chunks" in str(failure)
    assert [transport.count(m) for m in SENTENCE] == [2 * draws] * 3
    assert len(failure.attempts) == 3 * 2 * draws


# ----------------------------------------------------------------------------- new acceptance: D-104 amendment 2


@pytest.mark.parametrize("order", [("MARKB", "MARKA", "MARKC"), ("MARKA", "MARKB", "MARKC"),
                                   ("MARKA", "MARKC", "MARKB")], ids=["first", "middle", "last"])
@DRAWS
def test_llm_inputs_are_those_of_the_paper_without_the_empty_chunk(
        order: Tuple[str, ...], draws: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, str(draws))
    script = {"MARKA": [(REPLY_A, STOP)], "MARKB": EMPTY_DRAW * draws, "MARKC": [(REPLY_C, STOP)]}
    without = tuple(m for m in order if m != "MARKB")

    with_empty = _install(monkeypatch, script)
    diag.activate(run_id="with", artifact_dir=tmp_path / "with")
    merged, chunks = _stage_one(_text(*order))
    without_empty = _install(monkeypatch, script)
    diag.activate(run_id="without", artifact_dir=tmp_path / "without")
    merged_ref, chunks_ref = _stage_one(_text(*without))

    # The Stage-2 prompt, the RAG seed and reaction_summary.txt are all built from this.
    assert json.dumps(merged) == json.dumps(merged_ref)
    assert json.dumps(merged) == json.dumps(clean_stage_one(merge_stage_one_outputs(
        [c["output"] for c in chunks_ref])))
    for marker in without:
        assert with_empty.prompts(marker) == without_empty.prompts(marker)

    # The Stage-2 round-1 prompts, built from the chunk records themselves.
    stage_two: Dict[str, Transport] = {}
    for label, text, payload, details in (("with", _text(*order), merged, chunks),
                                          ("without", _text(*without), merged_ref, chunks_ref)):
        stage_two[label] = _install(monkeypatch, {m: [('{"additions": {}}', STOP)] for m in SENTENCE})
        run_stage_two_with_chunking(text, payload, details, enable_chunking=True, chunk_word_limit=5,
                                    chunk_overlap=0, max_attempts=1, retry_on_failure=False)
    for marker in without:
        assert stage_two["with"].prompts(marker) == stage_two["without"].prompts(marker)
    for transport in (with_empty, stage_two["with"]):
        for _marker, system, user in transport.calls:
            assert not any(s in system + user for s in RECORD_STRINGS)


def test_llm_inputs_the_lock_manifest_gains_nothing_from_the_empty_chunk(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``audit_json_llm`` and ``gap_resolver`` read the manifest into LLM prompts."""

    _install(monkeypatch, {"MARKA": [(REPLY_A, STOP)], "MARKB": EMPTY_DRAW, "MARKC": [(REPLY_C, STOP)]})
    _stage_one(_text("MARKA", "MARKB", "MARKC"), artifact_dir=tmp_path / "run")
    write_stage1_lock_artifacts(
        [{"payload": json.loads(REPLY_A), "source_chunk": "chunk_001"},
         {"payload": json.loads(REPLY_C), "source_chunk": "chunk_003"}], tmp_path / "ref")
    assert (tmp_path / "run" / MANIFEST_FILENAME).read_bytes() == (tmp_path / "ref" / MANIFEST_FILENAME).read_bytes()


# ----------------------------------------------------------------------------- preservation


@pytest.mark.parametrize("bad", [
    [(NOT_JSON, STOP)],
    [("", LENGTH)],
    [(EMPTY_1, LENGTH)],  # valid and empty, but cut off: not an answer
    [(EMPTY_1, STOP), (NOT_JSON, STOP)],
    # Repair round 1 (B1b): the client returns a content_filter reply as a partial answer.
    [(EMPTY_1, "content_filter")],
], ids=["invalid-json", "empty-length", "empty-json-at-length", "empty-then-invalid",
        "empty-json-content-filter"])
@DRAWS
def test_still_fatal_when_any_attempt_broke(bad: List[Tuple[str, str]], draws: int,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(draws))
    # Every draw but the last says nothing; the last one breaks. One broken draw is enough.
    transport = _install(monkeypatch, {"MARKA": [(REPLY_A, STOP)],
                                       "MARKB": EMPTY_DRAW * (draws - 1) + bad, "MARKC": [(REPLY_C, STOP)]})

    with pytest.raises(PipelineFailure) as excinfo:
        _stage_one(_text("MARKA", "MARKB", "MARKC"))

    _assert_chunk_2_fatal_as_at_base(excinfo.value, transport)


def _assert_chunk_2_fatal_as_at_base(failure: PipelineFailure, transport: Transport) -> None:
    assert failure.stage == "extraction chunk 2"
    assert str(failure) == "Chunk 2 failed to produce valid JSON."
    assert isinstance(failure.__cause__, PipelineFailure)
    assert failure.attempts is failure.__cause__.attempts
    assert transport.count("MARKC") == 0


@DRAWS
def test_still_fatal_when_the_budget_refused_the_retry(draws: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repair round 1 (B1a). A live leg deadline admits the last draw's first call on chunk 2,
    which says nothing, then refuses its retry as ``budget_exhausted``. That chunk never got
    its retry, so its silence is not an answer."""

    monkeypatch.setenv(ENV_VAR, str(draws))
    price = price_llm_call()
    assert math.isfinite(price)
    now = [0.0]
    transport = _install(monkeypatch, {"MARKA": [(REPLY_A, STOP)], "MARKB": EMPTY_DRAW * draws,
                                       "MARKC": [(REPLY_C, STOP)]})

    def tick(marker: str) -> None:
        if marker == "MARKB" and transport.count("MARKB") == 2 * draws - 1:
            now[0] += 20.0  # the leg had price + 10 s: now one more call does not fit

    transport.on_call = tick
    token = activate_leg_deadline(LegDeadline(price + 10.0, reserve_seconds=0.0, clock=lambda: now[0]))
    try:
        with pytest.raises(PipelineFailure) as excinfo:
            _stage_one(_text("MARKA", "MARKB", "MARKC"))
    finally:
        deactivate_leg_deadline(token)

    assert excinfo.value.__cause__.terminal_reason == BUDGET_EXHAUSTED  # the case is the one meant
    assert transport.count("MARKB") == 2 * draws - 1
    _assert_chunk_2_fatal_as_at_base(excinfo.value, transport)


@DRAWS
def test_still_no_accepted_record_when_every_chunk_is_empty(
        draws: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Repair round 1 (reviewer note 1). A run that fails leaves no chunk recorded as accepted."""

    monkeypatch.setenv(ENV_VAR, str(draws))
    _install(monkeypatch, {m: EMPTY_DRAW * draws for m in SENTENCE})
    recorder = diag.activate(run_id="orch758b", artifact_dir=tmp_path)

    with pytest.raises(PipelineFailure):
        _stage_one(_text("MARKA", "MARKB", "MARKC"))

    assert [b for b in recorder.boundaries if b["boundary"] == "stage1_empty_chunk"] == []
    assert not (tmp_path / REPORT_NAME).exists()


@DRAWS
def test_still_fails_when_every_chunk_is_empty(draws: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(draws))
    _install(monkeypatch, {m: EMPTY_DRAW * draws for m in SENTENCE})

    with pytest.raises(PipelineFailure) as excinfo:
        _stage_one(_text("MARKA", "MARKB", "MARKC"))

    chain = [excinfo.value, excinfo.value.__cause__]
    assert any(isinstance(e, PipelineFailure) and "structurally empty payload" in str(e) for e in chain)


@DRAWS
def test_still_unchanged_unchunked(draws: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(draws))
    text = _text("MARKA", "MARKB", "MARKC")
    transport = _install(monkeypatch, {"MARKA": EMPTY_DRAW * draws})
    with pytest.raises(PipelineFailure) as excinfo:
        run_stage_one_with_chunking(text, enable_chunking=False, max_attempts=2)
    assert excinfo.value.stage == "extraction"
    assert str(excinfo.value).startswith(
        "Extraction stage returned a structurally empty payload (no entities and no processes) on all 2 attempts.")
    assert transport.count("MARKA") == 2 * draws

    _install(monkeypatch, {"MARKA": [(REPLY_A, STOP)]})
    merged, chunks = run_stage_one_with_chunking(text, enable_chunking=False, max_attempts=2)
    assert _reaction_names(merged) == ["glucose phosphorylation"]
    assert [sorted(c) for c in chunks] == [["attempts", "chunk_id", "end_word", "output", "start_word", "text"]]


def test_still_unchanged_chunked_without_an_empty_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, {"MARKA": [(REPLY_A, STOP)], "MARKC": [(REPLY_C, STOP)]})
    merged, chunks = _stage_one(_text("MARKA", "MARKC"))
    assert _reaction_names(merged) == ["glucose phosphorylation", "fructose phosphorylation"]
    assert [sorted(c) for c in chunks] == [sorted(
        ["chunk_id", "section", "relevance_score", "text", "word_count", "start_word", "end_word",
         "output", "attempts"])] * 2
