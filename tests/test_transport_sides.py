"""T2PW_TRANSPORT_SIDES (D-W21c, test only): offline, fake chat function, no network."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict

import pytest

from t2pw.service import transport_sides as ts

TEXT = (
    "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment by a "
    "TolC‐dependent process. Following secretion, enterobactin chelates extracellular ferric iron, "
    "and the ferric‐enterobactin complex is then imported via TonB‐dependent uptake."
)


def _payload(**transport: Any) -> Dict[str, Any]:
    row = {"name": "enterobactin secretion", "cargo": "enterobactin",
           "evidence": "Enterobactin is produced in the cytoplasm and secreted to the extracellular environment"}
    row.update(transport)
    return {"entities": {}, "biological_states": [{"name": "Cytoplasm", "species": "Escherichia coli"}],
            "processes": {"transports": [row]}}


class Chat:
    def __init__(self, reply: Any = None, boom: bool = False) -> None:
        self.reply, self.boom, self.calls = reply, boom, 0

    def __call__(self, messages, **kwargs):
        self.calls += 1
        if self.boom:
            raise RuntimeError("provider down")
        return json.dumps(self.reply)


FROM_CYT = {"state": "cytoplasm", "compartment": "cytoplasm", "quote": "produced in the cytoplasm"}
TO_EXT = {"state": "extracellular", "compartment": "extracellular", "quote": "secreted to the extracellular environment"}


def _row(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload["processes"]["transports"][0]


def test_repairs_both_missing_sides_from_quotes_with_dash_folding():
    chat = Chat({"rows": [{"id": 0, "from": FROM_CYT, "to": TO_EXT}]})
    payload = _payload()
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=chat)
    assert chat.calls == 1
    assert _row(fixed)["from_biological_state"] == "Cytoplasm"  # existing state reused, case-insensitively
    assert _row(fixed)["to_biological_state"] == "extracellular"
    assert [s["name"] for s in fixed["biological_states"]] == ["Cytoplasm", "extracellular"]
    assert fixed["biological_states"][1]["species"] == "Escherichia coli"
    assert "from_biological_state" not in _row(payload)  # the input is not mutated
    (rep,) = report["repairs"]
    assert rep["old"] == ["", ""] and rep["new"] == ["Cytoplasm", "extracellular"] and rep["quotes"]["to_biological_state"]


def test_quote_with_unicode_hyphen_matches_ascii_quote():
    payload = _payload(from_biological_state="Cytoplasm", evidence="x")
    reply = {"rows": [{"id": 0, "to": {"state": "periplasm", "quote": "ferric-enterobactin complex is then imported"}}]}
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat(reply))
    assert _row(fixed)["to_biological_state"] == "periplasm" and not report["rejected"]


def test_quote_not_in_text_is_rejected_and_row_unchanged():
    bad = {"state": "extracellular", "quote": "is exported into the growth medium"}
    payload = _payload()
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat({"rows": [{"id": 0, "from": bad, "to": bad}]}))
    assert fixed is payload and fixed == _payload()
    assert report["rejected"] and not report["repairs"]


def test_one_word_or_missing_quote_is_rejected():
    reply = {"rows": [{"id": 0, "to": {"state": "extracellular", "quote": "extracellular"}, "from": {"state": "cytoplasm"}}]}
    fixed, report = ts.repair_transport_sides(_payload(), TEXT, chat_fn=Chat(reply))
    assert not report["repairs"] and "from_biological_state" not in _row(fixed)


def test_never_writes_an_equal_pair():
    payload = _payload(from_biological_state="E. coli cytoplasmic state", to_biological_state="E. coli cytoplasmic state")
    same = {"state": "e. coli cytoplasmic state", "quote": "produced in the cytoplasm"}
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat({"rows": [{"id": 0, "from": same, "to": same}]}))
    assert fixed is payload and report["rejected"][0]["reason"] == "result would be an equal pair"
    # one new distinct side fixes the equal pair
    fixed2, rep2 = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat({"rows": [{"id": 0, "to": TO_EXT}]}))
    assert _row(fixed2)["to_biological_state"] == "extracellular"
    assert _row(fixed2)["from_biological_state"] == "E. coli cytoplasmic state"


def test_one_side_named_other_stays_empty():
    fixed, report = ts.repair_transport_sides(_payload(), TEXT, chat_fn=Chat({"rows": [{"id": 0, "from": None, "to": TO_EXT}]}))
    assert _row(fixed)["to_biological_state"] == "extracellular" and "from_biological_state" not in _row(fixed)
    assert report["repairs"][0]["new"] == ["", "extracellular"]


def test_stated_side_is_kept_and_new_side_may_not_equal_it():
    payload = _payload(from_biological_state="Cytoplasm")
    reply = {"rows": [{"id": 0, "from": {"state": "periplasm", "quote": "produced in the cytoplasm"},
                       "to": {"state": "cytoplasm", "quote": "produced in the cytoplasm"}}]}
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat(reply))
    assert fixed is payload and report["rejected"]


def test_skips_repair_that_would_duplicate_twin_transport():
    payload = _payload(from_biological_state="Cytoplasm", to_biological_state="Cytoplasm")
    twin = {"name": "enterobactin secretion", "cargo": "Enterobactin", "from_biological_state": "cytoplasm",
            "to_biological_state": "extracellular", "evidence": "..."}
    payload["processes"]["transports"].append(twin)
    before = copy.deepcopy(payload)
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat({"rows": [{"id": 0, "from": FROM_CYT, "to": TO_EXT}]}))
    assert fixed is payload and payload == before
    assert report["rejected"][0]["reason"].startswith("would duplicate")


def test_clean_payload_makes_zero_calls_and_returns_same_object():
    payload = _payload(from_biological_state="cytoplasm", to_biological_state="extracellular")
    chat = Chat({"rows": []})
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=chat)
    assert chat.calls == 0 and fixed is payload and not report["deficient"]


def test_failure_fails_open_and_is_recorded():
    payload = _payload()
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=Chat(boom=True))
    assert fixed is payload and "provider down" in report["error"]
    fixed, report = ts.repair_transport_sides(payload, TEXT, chat_fn=lambda *a, **k: "not json")
    assert fixed is payload and report["error"]


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(ts.ENV_FLAG, raising=False)
    assert not ts.enabled()
    for off in ("", "0", "no"):
        monkeypatch.setenv(ts.ENV_FLAG, off)
        assert not ts.enabled()
    monkeypatch.setenv(ts.ENV_FLAG, "1")
    assert ts.enabled()


def test_run_stage_one_hook_flag_off_makes_no_call_and_flag_on_records(monkeypatch, tmp_path):
    """The hook sits in service.extraction.run_stage_one; here only its helper is exercised."""
    payload = _payload()
    state: Dict[str, Any] = {}
    monkeypatch.setattr(ts, "repair_transport_sides", lambda p, t, **k: (p, {"calls": 0, "repairs": []}))
    assert ts.apply_to_stage_one(payload, TEXT, state, tmp_path) is payload
    assert state["transport_sides"] == {"calls": 0, "repairs": []}
    assert (tmp_path / ts.REPORT_FILE).exists()


# -- the hook in service.extraction.run_stage_one (reuses the R3c fixture) --------------------
from test_service_extraction_stage1 import _run, world  # noqa: E402,F401


def test_hook_off_calls_nothing_and_on_records_and_repairs(world, monkeypatch):
    seen = []

    def fake(payload, text, **kw):
        seen.append(text)
        return dict(payload, repaired=True), {"calls": 1, "repairs": [{"pointer": "/x"}]}

    monkeypatch.setattr(ts, "repair_transport_sides", fake)
    monkeypatch.delenv(ts.ENV_FLAG, raising=False)
    res, _, state = _run(world)
    assert not seen and "transport_sides" not in state and "repaired" not in res.stage_one

    monkeypatch.setenv(ts.ENV_FLAG, "1")
    res, _, state = _run(world)
    assert seen == ["the paper"] and res.stage_one["repaired"] and state["transport_sides"]["calls"] == 1
