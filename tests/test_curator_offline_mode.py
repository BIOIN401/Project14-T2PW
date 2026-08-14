"""NEW ACCEPTANCE TESTS (C-050b) -- deterministic offline curator mode.

G9 labelling, explicit: this file tests a **new capability** and contains no
fabricated base failure. At the base SHA the flag does not exist and is inert;
that is *measured* by the A6 differential below and reported as an honest
inertness observation, never as a G9 base-failure proof.

Clauses pinned: A1 spellings and documentation; A2 zero model calls and zero
per-call provider work; A3 byte-identical payload out of a deliberately
NON-canonical input, no patch log; A4 the three outcomes are distinguishable;
A5 three repeats byte-identical; A6 the default path preserved exactly; A7 blast
radius of one module, one function.

A6 is a differential, not an assertion, and needs two source trees, so it is a
CLI in this same file -- run once per tree with ``PYTHONPATH`` pointed at that
tree, then compared: ``... run --base-src D --tip-src D --work W --out J``. Four
arms in :data:`ARMS`, flag absent except ``flag_on``, which is EXPECTED to
differ; a deterministic stub replaces ``chat_with_tools`` throughout, so **no
provider is ever contacted**. Control flow is the ordered sequence of ``t2pw``
frames entered, keyed ``module::function``; line ranges are unused because
inserting a branch shifts every line below it (S9.1). On the default path the
one admissible flow change is the new guard being evaluated -- see
``control_flow_is_base_plus_guard_only`` in :func:`compare`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import t2pw
from t2pw.curation import pathway_curator as pc
from t2pw.curation.apply_audit_patch import (
    APPLIED_PATCH_LOG_FILENAME,
    REJECTED_PATCH_LOG_FILENAME,
)
from t2pw.llm import client as llm_client

# Deliberately NOT what ``json.dumps(payload, indent=2, ensure_ascii=False)``
# emits: tab indent, irregular spacing, a \u escape, unsorted keys, trailing
# newline. Round-tripping a producer's own output proves nothing (S9.4).
NON_CANONICAL_INPUT = (
    '{"metadata":{"pathway_name":"Test \\u00fcber pathway","organism":"E. coli"},\n'
    '\t"entities" : {"compounds":[{"name":"NAD+"},{"name":"ATP"}],"proteins":[]},\n'
    '  "processes":{"reactions":[{"name":"R1","inputs":["NAD"],'
    '"outputs":["NADH"]}],"transports":[]}}\n'
)
ACCEPTED = ["1", "true", "TRUE", "True", "yes", "YES", "on", "y", "t", " 1 ", "\ttrue\n"]
REJECTED = [None, "", "   ", "0", "false", "FALSE", "no", "off", "n", "f", "2", "enabled"]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def _stub_patches(**kw):
    name = "/entities/compounds/0/name"
    for op, path, value, why, conf in (
            ("replace", name, "NAD", "mismatch", 0.9),
            ("replace", name, "NADX", "duplicate", 0.5),   # same (op, path): dedup
            ("add", "/reaction_order", ["R1"], "order", 0.8)):
        kw["tool_executor"]("propose_patch", {"op": op, "path": path, "value": value,
                                              "evidence": why, "confidence": conf})
    kw["tool_executor"]("no_such_tool", {})
    return "proposed three patches"


def _stub_raises(**_kw):
    raise RuntimeError("Bad request (400): deterministic stub, no provider contacted")


def _stub_forbidden(**_kw):
    raise AssertionError("MODEL CALLED WITH OFFLINE FLAG SET")


ARMS = {"quiet": (None, lambda **_kw: "no changes needed"),
        "patches": (None, _stub_patches), "raises": (None, _stub_raises),
        "flag_on": ("1", _stub_forbidden)}
STABLE_ARMS = ("quiet", "patches", "raises")
TRACE_KEYS = ("trace_sha256", "trace_len", "curator_frames")
GUARD = "curation/pathway_curator.py::curator_offline_mode_enabled"


class _ExplodingClient:
    """Any attribute touch is a provider request being prepared."""

    def __getattr__(self, name: str):  # pragma: no cover - must never run
        raise AssertionError(f"provider client was touched: {name}")


def _set_flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(pc.OFFLINE_CURATOR_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(pc.OFFLINE_CURATOR_ENV_VAR, value)


def _instrument(monkeypatch, chat=None) -> dict:
    """Count every route to a model; ``chat`` (if given) replaces the call."""
    seen: dict = {"chat": [], "resolve": 0}

    def _chat(**kwargs):
        seen["chat"].append(kwargs)
        if chat is None:
            raise AssertionError("chat_with_tools was called in offline mode")
        return chat(**kwargs)

    def _resolve(**_kwargs):  # pragma: no cover - counted, not exercised
        seen["resolve"] += 1
        return "unused-model"

    monkeypatch.setattr(pc, "chat_with_tools", _chat)
    monkeypatch.setattr(llm_client, "chat_with_tools", _chat)
    monkeypatch.setattr(llm_client, "_resolve_model", _resolve)
    monkeypatch.setattr(llm_client, "_client", _ExplodingClient())
    return seen


def _run(tmp_path, monkeypatch, flag, chat=None, tag=""):
    """One curator run under *flag*; returns (report, seen, src, out, report_path)."""
    seen = _instrument(monkeypatch, chat=chat)
    _set_flag(monkeypatch, flag)
    src = tmp_path / "audited.json"
    src.write_text(NON_CANONICAL_INPUT, encoding="utf-8", newline="")
    out, rep = tmp_path / f"curated{tag}.json", tmp_path / f"report{tag}.json"
    report = pc.run_pathway_curator(src, out, rep, pathway_name="P", organism="O",
                                    llm_temperature=0.2, llm_max_tokens=2000)
    return report, seen, src, out, rep


@pytest.mark.parametrize("value", ACCEPTED)
def test_a1_accepted_truthy_spellings(monkeypatch, value):
    _set_flag(monkeypatch, value)
    assert pc.curator_offline_mode_enabled() is True


@pytest.mark.parametrize("value", REJECTED)
def test_a1_everything_else_leaves_the_curator_online(monkeypatch, value):
    _set_flag(monkeypatch, value)
    assert pc.curator_offline_mode_enabled() is False


def test_a1_flag_name_tokens_and_documentation_are_pinned():
    assert pc.OFFLINE_CURATOR_ENV_VAR == "T2PW_OFFLINE_CURATOR"
    assert set(pc.OFFLINE_CURATOR_TRUE_TOKENS) == {"1", "true", "yes", "on", "y", "t"}
    assert pc.OFFLINE_CURATOR_SKIP_REASON == "offline_mode_env_flag"
    assert "Offline mode" in (pc.__doc__ or "")
    assert pc.OFFLINE_CURATOR_ENV_VAR in (pc.__doc__ or "")


def test_a2_a3_a4_offline_run_is_zero_call_zero_mutation_and_explicit(tmp_path, monkeypatch):
    report, seen, src, out, rep = _run(tmp_path, monkeypatch, "1")
    original = src.read_bytes()

    # A2 -- zero calls local and remote, no per-call model/provider selection.
    assert seen["chat"] == [] and seen["resolve"] == 0
    assert report["skipped"]["llm_calls"] == 0

    # A3 -- byte-identical payload, on the file, not merely on the dict.
    assert out.read_bytes() == original and src.read_bytes() == original
    canonical = json.dumps(json.loads(original), indent=2, ensure_ascii=False)
    assert canonical.encode("utf-8") != original  # re-serialization WOULD show
    assert report["patches"] == [] and report["apply_summary"] == {}
    assert report["summary"] == {
        "patches_proposed": 0, "patches_accepted": 0, "patches_rejected": 0}
    for name in (APPLIED_PATCH_LOG_FILENAME, REJECTED_PATCH_LOG_FILENAME):
        assert not (rep.parent / name).exists()

    # A4 -- the whole diagnostic, and it is not the error key.
    assert report["skipped"] == {
        "intentional": True, "reason": pc.OFFLINE_CURATOR_SKIP_REASON,
        "env_var": pc.OFFLINE_CURATOR_ENV_VAR, "llm_calls": 0, "payload_mutated": False}
    assert "error" not in report and "raw" not in report
    assert json.loads(rep.read_text(encoding="utf-8")) == report


def test_a4_the_three_outcomes_are_distinguishable(tmp_path, monkeypatch):
    """off-on-purpose vs ran-and-found-nothing vs raised-and-swallowed."""
    off = _run(tmp_path, monkeypatch, "1", tag="off")[0]
    nothing = _run(tmp_path, monkeypatch, None, chat=lambda **_kw: "nil", tag="nil")[0]
    raised = _run(tmp_path, monkeypatch, None, chat=_stub_raises, tag="err")[0]

    assert "skipped" in off and "error" not in off
    assert "skipped" not in nothing and "error" not in nothing and "raw" in nothing
    assert "skipped" not in raised and "error" in raised
    # Zero accepted patches in all three: counts alone cannot tell them apart.
    assert {r["summary"]["patches_accepted"] for r in (off, nothing, raised)} == {0}


def test_a5_three_repeats_are_byte_identical(tmp_path, monkeypatch):
    runs = []
    for i in range(3):
        report, _seen, _src, out, rep = _run(tmp_path, monkeypatch, "1", tag=str(i))
        runs.append((out.read_bytes(), rep.read_bytes(), json.dumps(report, sort_keys=True)))

    assert runs[0] == runs[1] == runs[2]


# --- A6: in-process complement; the differential CLI below is authoritative -

@pytest.mark.parametrize("value", [None, "0", "false", "", "off"])
def test_a6_flag_absent_or_falsy_still_calls_the_model_unchanged(tmp_path, monkeypatch, value):
    report, seen, _src, _out, _rep = _run(
        tmp_path, monkeypatch, value, chat=lambda **_kw: "done")

    assert len(seen["chat"]) == 1
    call = seen["chat"][0]
    assert call["temperature"] == 0.2 and call["max_tokens"] == 2000
    assert call["max_tool_rounds"] == 20
    assert call["model_env_var"] == "OPENROUTER_CURATOR_MODEL"
    assert call["stage_name"] == "curator"
    assert "skipped" not in report and "raw" in report


def test_a7_flag_is_read_in_exactly_one_module():
    package = Path(t2pw.__file__).resolve().parent

    def _scan(token: str) -> list:
        return sorted(p.relative_to(package).as_posix() for p in package.rglob("*.py")
                      if token in p.read_text(encoding="utf-8", errors="ignore"))

    assert _scan(pc.OFFLINE_CURATOR_ENV_VAR) == ["curation/pathway_curator.py"]
    assert _scan("curator_offline_mode_enabled") == ["curation/pathway_curator.py"]


# --- A6 differential CLI (not collected by pytest) --------------------------

def capture(work: Path) -> dict:
    """Run every arm against whichever ``t2pw`` is on ``sys.path``."""
    package = Path(t2pw.__file__).resolve().parent
    work.mkdir(parents=True, exist_ok=True)
    src = work / "audited.json"
    src.write_text(NON_CANONICAL_INPUT, encoding="utf-8", newline="")
    result: dict = {"curator_file": str(package / "curation" / "pathway_curator.py"),
                    "arms": {}}
    real_chat = pc.chat_with_tools
    for arm, (flag, stub) in ARMS.items():
        calls: list = []
        frames: list = []

        def _chat(_stub=stub, _calls=calls, **kw):
            msgs = kw.get("messages") or []
            _calls.append(dict(
                roles=[m.get("role") for m in msgs],
                content_sha=[_sha((m.get("content") or "").encode("utf-8")) for m in msgs],
                tools_sha=_sha(_norm(kw.get("tools")).encode("utf-8")),
                **{k: kw.get(k) for k in ("temperature", "max_tokens", "max_tool_rounds",
                                          "model_env_var", "stage_name")}))
            return _stub(**kw)

        def _tracer(frame, event, _arg, _frames=frames, _root=package):
            if event == "call":
                try:
                    rel = Path(frame.f_code.co_filename).resolve().relative_to(_root)
                except (ValueError, OSError):
                    return None
                _frames.append(f"{rel.as_posix()}::{frame.f_code.co_name}")
            return None

        pc.chat_with_tools = _chat
        os.environ.pop("T2PW_OFFLINE_CURATOR", None)
        if flag is not None:
            os.environ["T2PW_OFFLINE_CURATOR"] = flag
        out, rep = work / f"{arm}.out.json", work / f"{arm}.rep.json"
        try:
            sys.settrace(_tracer)
            returned = pc.run_pathway_curator(
                src, out, rep, reaction_summary="R1: NAD -> NADH", pathway_name="P",
                organism="O", llm_temperature=0.2, llm_max_tokens=2000)
        finally:
            sys.settrace(None)
            pc.chat_with_tools = real_chat
            os.environ.pop("T2PW_OFFLINE_CURATOR", None)
        result["arms"][arm] = {
            "chat_calls": len(calls), "call_kwargs": calls,
            "output_sha256": _sha(out.read_bytes()),
            "output_equals_input": out.read_bytes() == src.read_bytes(),
            "report_sha256": _sha(rep.read_bytes()), "returned_norm": _norm(returned),
            "trace_sha256": _sha(_norm(frames).encode("utf-8")), "trace_len": len(frames),
            "curator_frames": [f for f in frames
                               if f.startswith("curation/pathway_curator.py")]}
    return result


def _child(src_dir: Path, work: Path, out: Path, timeout: float) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir)
    env.pop("T2PW_OFFLINE_CURATOR", None)
    # Never text=True: these trees carry em dashes and cp1252 would decode them
    # into a false difference (S9.2).
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "capture",
         "--work", str(work), "--out", str(out)], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout)
    sys.stdout.write(proc.stdout.decode("utf-8", "replace"))
    if proc.returncode != 0:
        raise SystemExit(f"capture failed for {src_dir}: rc={proc.returncode}")


def compare(base: dict, tip: dict) -> dict:
    v: dict = {"base_curator_file": base["curator_file"],
               "tip_curator_file": tip["curator_file"], "arms": {},
               "a6_default_path_identical": True}
    for arm in ARMS:
        b, t = base["arms"][arm], tip["arms"][arm]
        differing = sorted(k for k in b if k not in TRACE_KEYS and b[k] != t[k])
        # The ONLY flow change the default path may show is the new guard being
        # evaluated: same frames, same order, plus GUARD entries and nothing else.
        flow_ok = ([f for f in t["curator_frames"] if f != GUARD] == b["curator_frames"]
                   and t["trace_len"] - b["trace_len"] == 1)
        v["arms"][arm] = {"identical_artifacts": not differing, "differing_fields": differing,
                          "control_flow_is_base_plus_guard_only": flow_ok,
                          "base_chat_calls": b["chat_calls"], "tip_chat_calls": t["chat_calls"],
                          "base_trace_len": b["trace_len"], "tip_trace_len": t["trace_len"]}
        if arm in STABLE_ARMS and (differing or not flow_ok):
            v["a6_default_path_identical"] = False
    ft = tip["arms"]["flag_on"]
    v["flag_inert_at_base"] = base["arms"]["flag_on"]["chat_calls"] == 1
    v["flag_honoured_at_tip"] = ft["chat_calls"] == 0 and ft["output_equals_input"]
    v["pass"] = bool(v["a6_default_path_identical"] and v["flag_inert_at_base"]
                     and v["flag_honoured_at_tip"])
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C-050b A6 base-vs-tip differential")
    ap.add_argument("mode", choices=("capture", "run"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--base-src")
    ap.add_argument("--tip-src")
    a = ap.parse_args(argv)
    out, work = Path(a.out).resolve(), Path(a.work).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"T2PW: {t2pw.__file__}")
    if a.mode == "capture":
        out.write_text(_norm(capture(work)), encoding="utf-8")
        return 0
    base_json, tip_json = work / "base.capture.json", work / "tip.capture.json"
    _child(Path(a.base_src).resolve(), work / "base", base_json, 600.0)
    _child(Path(a.tip_src).resolve(), work / "tip", tip_json, 600.0)
    verdict = compare(json.loads(base_json.read_text(encoding="utf-8")),
                      json.loads(tip_json.read_text(encoding="utf-8")))
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.path[0] = os.getcwd()
    raise SystemExit(main())
