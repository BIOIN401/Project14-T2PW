"""ORCH-734 -- prove the Stage-1 alternate-model rung is LIVE after configuration.

Read-only with respect to ``src/``. Nothing here modifies production code; the
whole activation is one line of ``.env`` (``OPENROUTER_EXTRACTION_FALLBACK_MODEL``),
which is why ORCH-733 section 6 ruled it a configuration change and not a card.

WHAT THIS PROVES, and what it deliberately does not
---------------------------------------------------
The charter asks for five things:

  1. the primary extraction model is invoked;
  2. a B/C/D-style *recoverable* delivery failure can reach rung 3;
  3. rung 3 now identifies a materially different fallback model;
  4. the alternate-model request actually issues;
  5. the existing semantic safeguards remain active.

Phases A-E drive the real production function ``_run_json_stage`` with a stubbed
provider boundary, so the ladder, the rung admission, the degeneracy test and the
attempt cap are the shipped ones. Phase F is the only phase that touches the
network: ONE real call to the configured fallback model, which is what turns
"rung 3 selects a different selector" into "the alternate-model request actually
issues and the account can reach it".

Phase A is the BASE arm and it is the point of the exercise. It runs the same
input with the variable UNSET and shows the refusal that the whole archived
corpus recorded -- ``strategy_not_materially_different``, 6 legs, 0 issued. A
proof that only showed the tip passing would not distinguish "the configuration
did something" from "this always worked".

NO PWML-YIELD CLAIM FOLLOWS FROM THIS FILE. Rung 3 buys each qualifying leg one
more draw from a different model where today it gets none. Whether that draw
succeeds on a real paper is unmeasured here and is not asserted anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

PRIMARY_VAR = "OPENROUTER_EXTRACTION_MODEL"
FALLBACK_VAR = "OPENROUTER_EXTRACTION_FALLBACK_MODEL"

# A payload that parses but declares nothing. This is mechanism B/D from the
# ORCH-733 census: the provider delivered, the content was degenerate. It is the
# shape that sets ``saw_empty_payload`` and therefore the only shape that can
# reach rung 3 -- class A (wholly empty completion) cannot, which ORCH-733
# section 3 established and this probe does not attempt to change.
DEGENERATE = '{"entities": {}, "processes": {}}'

USABLE = json.dumps(
    {
        "entities": {
            "compounds": [{"name": "chorismate"}, {"name": "isochorismate"}],
            "proteins": [{"name": "MenF"}],
        },
        "processes": {
            "reactions": [
                {
                    "name": "isochorismate synthase",
                    "left": ["chorismate"],
                    "right": ["isochorismate"],
                    "enzyme": "MenF",
                }
            ]
        },
    }
)


# ---------------------------------------------------------------------------
# The stubbed provider boundary.
# ---------------------------------------------------------------------------
class _Diag:
    def __init__(self, model: str, text: str) -> None:
        self.model = model
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        import hashlib

        return {
            "model": self.model,
            "stage": "Stage 1 extraction",
            "attempts": 1,
            "finish_reason": "stop",
            "response_status": "ok" if self.text else "empty",
            "terminal_reason": "",
            "request_hash": "",
            "response_hash": hashlib.sha256(self.text.encode()).hexdigest()[:32],
            "raw_chars": len(self.text),
            "attempt_log": [],
        }


class _Completion:
    def __init__(self, model: str, text: str) -> None:
        self.text = text
        self.diagnostics = _Diag(model, text)


class Boundary:
    """Records every crossing, and answers by which env var was named."""

    def __init__(self, *, fallback_reply: str) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fallback_reply = fallback_reply

    def __call__(self, messages, **kw):  # noqa: ANN001 - mirrors chat_detailed
        env_var = kw.get("model_env_var")
        resolved = (os.getenv(env_var) or "") if env_var else ""
        resolved = resolved or os.getenv("OPENROUTER_MODEL") or ""
        is_fallback = env_var == FALLBACK_VAR
        text = self.fallback_reply if is_fallback else DEGENERATE
        self.calls.append(
            {
                "n": len(self.calls) + 1,
                "model_env_var": env_var,
                "resolved_model": resolved,
                "is_fallback_route": is_fallback,
                "reply_chars": len(text),
                "max_tokens": kw.get("max_tokens"),
                "user_prompt_chars": len(messages[-1]["content"]),
            }
        )
        return _Completion(resolved, text)


def _drive(*, fallback_reply: str, deadline_seconds: float = 1800.0) -> Dict[str, Any]:
    """Run the REAL ``_run_json_stage`` Stage-1 path over the stub."""

    from t2pw.pipeline import pipeline as P
    from t2pw.pipeline.deadline import LegDeadline
    from t2pw.pipeline.extraction_ladder import (
        ExtractionLadder,
        activate_leg_deadline,
        deactivate_leg_deadline,
    )

    boundary = Boundary(fallback_reply=fallback_reply)
    original = P.chat_detailed
    P.chat_detailed = boundary  # type: ignore[assignment]

    deadline = LegDeadline(deadline_seconds)
    token = activate_leg_deadline(deadline)
    ladder = ExtractionLadder(stage="extraction", deadline=deadline)

    def build_user_prompt(prev: Optional[str], reason: Optional[str]) -> str:
        # Mirrors production's two-argument builder. The reason string is what
        # production passes for the section-scoped rung; it is echoed so the
        # prompt genuinely differs and the request hash cannot collide.
        base = "Extract the pathway from the following paper.\n\n" + ("PAPER " * 400)
        if reason:
            base += "\n\nRETRY REASON: " + str(reason)
        return base

    outcome: Dict[str, Any] = {}
    try:
        payload, attempts = P._run_json_stage(
            stage_name="extraction",
            system_prompt="You are a biochemical pathway extractor. Reply with JSON.",
            build_user_prompt=build_user_prompt,
            max_attempts=2,
            temperature=0.0,
            max_tokens=16000,
            model_env_var=PRIMARY_VAR,
            repair_json=False,
            retry_on_empty_payload=True,
            deadline=deadline,
            ladder=ladder,
        )
        outcome["raised"] = None
        outcome["payload_reactions"] = len(
            ((payload or {}).get("processes") or {}).get("reactions") or []
        )
        outcome["attempt_notes"] = [a.get("note") or a.get("phase") for a in attempts]
    except BaseException as exc:  # noqa: BLE001 - a raise IS the result here
        outcome["raised"] = f"{exc.__class__.__name__}: {exc}"
        outcome["payload_reactions"] = 0
        outcome["attempt_notes"] = []
    finally:
        P.chat_detailed = original  # type: ignore[assignment]
        deactivate_leg_deadline(token)

    record = ladder.preservation_record()
    outcome["calls"] = boundary.calls
    outcome["ladder"] = record
    outcome["issued_rungs"] = [a.get("rung") for a in (record.get("attempts") or [])]
    outcome["issued_models"] = [a.get("model") for a in (record.get("attempts") or [])]
    outcome["skipped"] = [
        {"rung": s.get("rung"), "skip_cause": s.get("skip_cause"), "detail": s.get("detail")}
        for s in (record.get("skipped_steps") or [])
    ]
    outcome["fallback_calls"] = [c for c in boundary.calls if c["is_fallback_route"]]
    return outcome


# ---------------------------------------------------------------------------
# Phases.
# ---------------------------------------------------------------------------
def _set(var: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = value


def phase_a_base_unset(primary: str) -> Dict[str, Any]:
    """BASE ARM. Variable unset -> rung 3 refused, zero alternate-model calls."""

    _set(FALLBACK_VAR, None)
    _set(PRIMARY_VAR, primary)
    out = _drive(fallback_reply=USABLE)
    refusals = [s for s in out["skipped"] if s["skip_cause"] == "strategy_not_materially_different"]
    return {
        "phase": "A_base_variable_unset",
        "expectation": "rung 3 REFUSED strategy_not_materially_different; 0 fallback calls",
        "fallback_var_value": os.getenv(FALLBACK_VAR),
        "alternate_env_var_returned": _alt(primary),
        "rung3_refusals": refusals,
        "fallback_calls": len(out["fallback_calls"]),
        "issued_rungs": out["issued_rungs"],
        "raised": out["raised"],
        "pass": bool(refusals) and not out["fallback_calls"],
    }


def phase_b_activated(primary: str, fallback: str) -> Dict[str, Any]:
    """TIP ARM. Variable set to a different model -> rung 3 ISSUES to it."""

    _set(FALLBACK_VAR, fallback)
    _set(PRIMARY_VAR, primary)
    out = _drive(fallback_reply=USABLE)
    fb = out["fallback_calls"]
    return {
        "phase": "B_activated_alternate_model_issues",
        "expectation": "rung 3 ADMITTED and ISSUED to the fallback selector; payload recovered",
        "alternate_env_var_returned": _alt(primary),
        "primary_calls": [c for c in out["calls"] if not c["is_fallback_route"]],
        "fallback_calls": fb,
        "issued_rungs": out["issued_rungs"],
        "skipped": out["skipped"],
        "payload_reactions": out["payload_reactions"],
        "attempt_notes": out["attempt_notes"],
        "raised": out["raised"],
        "pass": (
            len(fb) == 1
            and fb[0]["resolved_model"] == fallback
            and "materially_different_strategy" in out["issued_rungs"]
            and out["payload_reactions"] == 1
            and out["raised"] is None
        ),
    }


def phase_c_same_model_guard(primary: str) -> Dict[str, Any]:
    """SAFEGUARD. Fallback var pointing at the SAME model is still refused."""

    _set(FALLBACK_VAR, primary)
    _set(PRIMARY_VAR, primary)
    out = _drive(fallback_reply=USABLE)
    refusals = [s for s in out["skipped"] if s["skip_cause"] == "strategy_not_materially_different"]
    return {
        "phase": "C_safeguard_same_model_still_refused",
        "expectation": "two variable names resolving to one model is NOT an escalation",
        "alternate_env_var_returned": _alt(primary),
        "rung3_refusals": refusals,
        "fallback_calls": len(out["fallback_calls"]),
        "pass": bool(refusals) and not out["fallback_calls"],
    }


def phase_d_degenerate_fallback(primary: str, fallback: str) -> Dict[str, Any]:
    """SAFEGUARD. A degenerate rung-3 reply is NOT laundered into a success."""

    _set(FALLBACK_VAR, fallback)
    _set(PRIMARY_VAR, primary)
    out = _drive(fallback_reply=DEGENERATE)
    return {
        "phase": "D_safeguard_degenerate_fallback_reply_rejected",
        "expectation": "rung 3 issues, returns nothing usable, and the stage still FAILS",
        "fallback_calls": len(out["fallback_calls"]),
        "issued_rungs": out["issued_rungs"],
        "payload_reactions": out["payload_reactions"],
        "raised": out["raised"],
        "pass": len(out["fallback_calls"]) == 1 and out["raised"] is not None,
    }


def phase_e_attempt_cap(primary: str, fallback: str) -> Dict[str, Any]:
    """SAFEGUARD. The section-9 ceiling of three model attempts still binds."""

    _set(FALLBACK_VAR, fallback)
    _set(PRIMARY_VAR, primary)
    out = _drive(fallback_reply=USABLE)
    total = len(out["calls"])
    rec = out["ladder"]
    return {
        "phase": "E_safeguard_attempt_cap",
        "expectation": "total model calls <= MAX_TOTAL_MODEL_ATTEMPTS (3)",
        "total_model_calls": total,
        "attempts_used": rec.get("attempts_issued"),
        "attempts_remaining": rec.get("attempts_remaining"),
        "pass": total <= 3,
    }


def phase_f_live(fallback: str) -> Dict[str, Any]:
    """THE ONLY NETWORK PHASE. One real call over the configured fallback route.

    Issued through the production client with ``model_env_var=FALLBACK_VAR``, so
    it exercises the same ``_resolve_model`` fallthrough rung 3 uses. Tiny by
    construction: the point is that the request issues and the account can reach
    the model, not what it says.
    """

    _set(FALLBACK_VAR, fallback)
    from t2pw.llm.client import chat_detailed

    try:
        result = chat_detailed(
            [
                {"role": "system", "content": "Reply with a JSON object and nothing else."},
                {
                    "role": "user",
                    "content": (
                        'Return exactly {"reaction": {"left": ["chorismate"], '
                        '"right": ["isochorismate"], "enzyme": "MenF"}}'
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=200,
            response_json=True,
            model_env_var=FALLBACK_VAR,
            stage_name="ORCH-734 fallback liveness",
        )
    except BaseException as exc:  # noqa: BLE001
        return {
            "phase": "F_live_alternate_model_request",
            "expectation": "the configured fallback model answers over the real provider",
            "error": f"{exc.__class__.__name__}: {exc}",
            "pass": False,
        }

    diag = result.diagnostics.to_dict()
    parsed: Any
    try:
        parsed = json.loads(result.text)
    except Exception:  # noqa: BLE001
        parsed = None
    return {
        "phase": "F_live_alternate_model_request",
        "expectation": "the configured fallback model answers over the real provider",
        "model_answered": diag.get("model"),
        "finish_reason": diag.get("finish_reason"),
        "response_status": diag.get("response_status"),
        "raw_chars": diag.get("raw_chars"),
        "parsed_ok": parsed is not None,
        "reply_preview": (result.text or "")[:200],
        "pass": bool(result.text) and diag.get("model") == fallback,
    }


def _alt(primary_var_value: str) -> str:
    """``alternate_model_env_var`` under the CURRENT environment."""

    from t2pw.pipeline.extraction_ladder import alternate_model_env_var

    return alternate_model_env_var(PRIMARY_VAR)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_path", default=None)
    ap.add_argument("--live", action="store_true", help="include phase F (one network call)")
    args = ap.parse_args(argv)

    # Load .env exactly the way the app does, then snapshot what we found so the
    # report states the configuration it actually measured.
    try:
        from dotenv import load_dotenv

        load_dotenv(_ROOT / ".env", override=False)
    except Exception:  # noqa: BLE001
        pass

    # EAGER IMPORT, and it is not cosmetic. ``t2pw.llm.client`` calls
    # ``load_dotenv`` at import time, so importing it lazily inside a phase
    # re-populates the very variable phase A has just unset -- which is exactly
    # how the first run of this probe reported a false failure: the base arm
    # issued a fallback call because .env had been reloaded underneath it. Doing
    # the import here means every phase mutates an environment that is already
    # fully loaded, and ``_set`` is the only thing that changes it afterwards.
    from t2pw.pipeline import pipeline as _P  # noqa: F401
    from t2pw.llm import client as _C  # noqa: F401

    primary = (os.getenv(PRIMARY_VAR) or os.getenv("OPENROUTER_MODEL") or "").strip()
    fallback = (os.getenv(FALLBACK_VAR) or "").strip()
    if not primary:
        print("ABORT: no primary extraction model configured", file=sys.stderr)
        return 2
    if not fallback:
        print(f"ABORT: {FALLBACK_VAR} is not set; nothing to prove", file=sys.stderr)
        return 2

    saved = {v: os.getenv(v) for v in (PRIMARY_VAR, FALLBACK_VAR)}
    phases: List[Dict[str, Any]] = []
    try:
        phases.append(phase_a_base_unset(primary))
        phases.append(phase_b_activated(primary, fallback))
        phases.append(phase_c_same_model_guard(primary))
        phases.append(phase_d_degenerate_fallback(primary, fallback))
        phases.append(phase_e_attempt_cap(primary, fallback))
        if args.live:
            phases.append(phase_f_live(fallback))
    finally:
        for var, val in saved.items():
            _set(var, val)

    report = {
        "task": "ORCH-734",
        "probe": "orch734_rung3_probe.py",
        "primary_extraction_model": primary,
        "fallback_extraction_model": fallback,
        "live_phase_included": bool(args.live),
        "phases": phases,
        "all_pass": all(p.get("pass") for p in phases),
    }

    text = json.dumps(report, indent=2)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(text, encoding="utf-8")
    print(text)

    for p in phases:
        print(
            "  %-7s %s" % ("PASS" if p.get("pass") else "FAIL", p["phase"]),
            file=sys.stderr,
        )
    print("VERDICT: " + ("ALL PHASES PASS" if report["all_pass"] else "FAILED"), file=sys.stderr)
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
