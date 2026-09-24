"""ORCH-743 item 1: every extraction call's raw completion is kept, linked by call id.

Pilot-001 kept only a response hash, so nobody could see why 59 of 59 enzyme /
gene / species fields came back uncited. G9 correction: at the base SHA the raw
completion text is recoverable from nowhere in the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import helpers_mining as H  # noqa: E402
from t2pw.mining.artifacts import read_jsonl  # noqa: E402
from t2pw.mining.controller import MiningController  # noqa: E402

MARK_OK = "LINEAGE-MARK-7f3a"
MARK_BAD = "LINEAGE-UNPARSEABLE-91c2"


def _diag(finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(finish_reason=finish_reason, attempts=1, model="scripted",
                           request_hash="req", response_hash="res-hash", attempt_log=[])


def _run(tmp_path: Path):
    scripted = H.ScriptedModel()

    def chat(messages, **kwargs):
        user = "".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        if "maize lipoxygenase" in user:
            return SimpleNamespace(text=f"I cannot answer in JSON. {MARK_BAD}", diagnostics=_diag())
        result = scripted(messages, **kwargs)
        payload = json.loads(result.text) if result.text.strip().startswith("{") else None
        if not isinstance(payload, dict) or "claims" not in payload:
            return result
        # A key the parser ignores: it can only survive in a raw-completion record.
        payload["lineage_note"] = MARK_OK
        return SimpleNamespace(text=json.dumps(payload), diagnostics=_diag())

    # One attempt per call, so the unparseable paper ends the call as a parse error.
    config = H.make_config(tmp_path, budgets={"max_rounds": 1}, model={"max_attempts": 1})
    source = H.ScriptedPaperSource()

    def progress(event, payload):
        if event == "round_start":
            source.round = int(payload.get("round") or 1)

    controller = MiningController(config, source=source, chat_fn=chat,
                                  index=H.offline_index(config, tmp_path),
                                  progress=progress, enabled_override=True)
    return controller.run(probe_model=False, probe_embeddings=False)


def test_the_raw_completion_is_persisted_and_linked_to_its_model_call(tmp_path):
    outcome = _run(tmp_path)
    root = outcome.run_dir.root

    holders = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and MARK_OK in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert holders, "the raw completion text is recoverable from nowhere in the run directory"

    rows = read_jsonl(root / "rounds" / "round_001" / "model_raw.jsonl")
    calls = {r["call_id"]: r for r in read_jsonl(root / "model_calls.jsonl")}
    ok_rows = [r for r in rows if MARK_OK in r.get("raw_text", "")]
    assert ok_rows
    for row in ok_rows:
        assert row["call_id"] in calls, "a raw row must link to its model_calls.jsonl record"
        assert row["purpose"] == "extract_claims"
        assert row["finish_reason"] == "stop"
        assert row["parse_outcome"] == "ok"
        assert row["response_hash"] == "res-hash"
        assert json.loads(row["raw_text"])["lineage_note"] == MARK_OK  # verbatim, not re-serialised claims
        assert calls[row["call_id"]]["response_hash"] == "res-hash"  # the hash is still kept


def test_an_unparseable_completion_keeps_its_text_and_its_parse_error(tmp_path):
    outcome = _run(tmp_path)
    root = outcome.run_dir.root
    rows = [
        r for r in read_jsonl(root / "rounds" / "round_001" / "model_raw.jsonl")
        if MARK_BAD in r.get("raw_text", "")
    ]
    assert rows, "a failed parse must still leave the text that failed"
    calls = {r["call_id"]: r for r in read_jsonl(root / "model_calls.jsonl")}
    for row in rows:
        assert row["parse_outcome"] == "parse_error"
        assert row["error"]
        assert calls[row["call_id"]]["status"] == "parse_error"
    assert [r["attempt"] for r in rows] == list(range(1, len(rows) + 1))


def test_the_lineage_file_holds_no_credentials_or_request(tmp_path):
    outcome = _run(tmp_path)
    text = (outcome.run_dir.root / "rounds" / "round_001" / "model_raw.jsonl").read_text(
        encoding="utf-8").lower()
    for token in ("api_key", "authorization", "openrouter_api_key", "bearer", "[evidence_id:"):
        assert token not in text
