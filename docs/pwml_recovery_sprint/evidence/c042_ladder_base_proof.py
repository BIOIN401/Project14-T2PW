"""C-042 G9 capture: runs unchanged at base ``bcc0bfe`` and at the branch tip.

Two captures, deliberately separated because G9 treats them differently.

``correction`` -- the behavioural base failure. Feeds ``_run_json_stage`` a chat
function that returns a FIXED empty ``{}`` every time, with the REAL extraction
prompt builder, and records how many model calls were issued, whether the third
prompt was byte-identical to the second, and what termination reason the run
reported. At ``bcc0bfe`` the third call happens and its prompt equals the
second's -- the same prompt to the same model after two identical empty
responses, which ``PRODUCT_CONTRACT`` § 9 forbids and which is the measured
PMC12782028 signature -- and the failure names none of the six D-005 reasons, so
nothing downstream can tell an inert model from a spent clock. Neither fact is
"a symbol is missing": both are output.

``preservation`` -- the A7 golden. Four fixtures that all end in a usable
payload (clean JSON, broken-then-repairable, repair-unavailable-then-salvaged,
degenerate-then-recovering) captured as return value + attempt log + every
boundary record. Must be byte-identical at base and at tip, and is checked at
base FIRST: a golden captured at tip proves nothing.

Usage (offline; never issues a real model call)::

    python c042_ladder_base_proof.py --out capture.json
    python c042_ladder_base_proof.py --out a.json --only preservation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import t2pw.pipeline.extraction_diagnostics as diag_mod
import t2pw.pipeline.pipeline as pipeline_mod
from t2pw.llm.client import CompletionDiagnostics, CompletionResult

# --- fixtures --------------------------------------------------------------
PAPER = """Abstract
Meropenem resistance in Pseudomonas aeruginosa was examined.

Introduction
Beta-lactam resistance has been reviewed extensively elsewhere.

Methods
Cells were grown in LB broth at 37 C and lysates assayed for hydrolysis.

Results
LpxC hydrolyses UDP-3-O-acyl-N-acetylglucosamine to UDP-3-O-acyl-glucosamine
and acetate. The reaction consumes one water molecule.

References
1. Somebody et al, 2019.

Acknowledgements
We thank the sequencing core.
"""

WITH_REACTIONS = json.dumps(
    {
        "entities": {"compounds": [{"name": "ATP", "evidence": "ATP is consumed"}]},
        "processes": {
            "reactions": [
                {
                    "name": "r1",
                    "inputs": ["ATP"],
                    "outputs": ["ADP"],
                    "evidence": "ATP is consumed to give ADP",
                }
            ]
        },
    }
)

#: One missing comma inside ``processes``. The prefix salvage recovers the
#: entities and drops every reaction, which is why the repair path exists.
MALFORMED = (
    '{"entities": {"compounds": [{"name": "ATP", "evidence": "ATP is consumed"}]}, '
    '"processes": {"reactions": [{"name": "r1" "inputs": ["ATP"], "outputs": ["ADP"], '
    '"evidence": "ATP is consumed to give ADP"}]}}'
)

#: Malformed, but with a complete reaction BEFORE the syntax error, so the free
#: prefix scan salvages something usable without any model call.
SALVAGEABLE = WITH_REACTIONS[:-1] + ', "trailing": '


class Recorder:
    """A chat function that answers from a script and remembers the prompts."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.prompts: List[str] = []
        self.models: List[Optional[str]] = []

    def __call__(self, messages: Any, **kwargs: Any) -> CompletionResult:
        self.prompts.append(messages[-1]["content"])
        self.models.append(kwargs.get("model_env_var"))
        text = self._texts[min(len(self.prompts) - 1, len(self._texts) - 1)]
        return CompletionResult(
            text,
            CompletionDiagnostics(model="test-model", stage="extraction", attempts=1),
        )

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _build(prev: Optional[str], err: Optional[str]) -> str:
    return pipeline_mod._build_extraction_prompt(PAPER, prev, err)


def _run(
    *,
    chat: Recorder,
    max_attempts: int,
    repair_json: bool = False,
    repair_chat_fn: Any = None,
    stage_name: str = "extraction",
    retry_on_empty_payload: bool = True,
    build: Any = None,
) -> Dict[str, Any]:
    """One ``_run_json_stage`` call with everything captured and nothing live."""

    recorder = diag_mod.activate(run_id="c042")
    original = pipeline_mod.chat_detailed
    pipeline_mod.chat_detailed = chat
    result: Dict[str, Any] = {}
    try:
        payload, attempts = pipeline_mod._run_json_stage(
            stage_name=stage_name,
            system_prompt="SYSTEM",
            build_user_prompt=build or _build,
            max_attempts=max_attempts,
            temperature=0.0,
            max_tokens=100,
            repair_json=repair_json,
            repair_chat_fn=repair_chat_fn,
            retry_on_empty_payload=retry_on_empty_payload,
        )
        result["returned"] = payload
        result["raised"] = None
        result["attempt_log"] = _clean_attempts(attempts)
    except pipeline_mod.PipelineFailure as failure:
        result["returned"] = None
        result["raised"] = str(failure)
        result["attempt_log"] = _clean_attempts(failure.attempts)
    finally:
        pipeline_mod.chat_detailed = original
    result["boundaries"] = [dict(row) for row in recorder.boundaries]
    result["model_calls"] = chat.calls
    diag_mod.deactivate()
    return result


def _clean_attempts(attempts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attempt rows minus the fields a run cannot reproduce byte-for-byte."""

    rows = []
    for row in attempts:
        entry = {k: v for k, v in row.items() if k != "boundary_diagnostics"}
        rows.append(entry)
    return rows


# --- capture 1: the behavioural base failure (A3, A5) ----------------------
def capture_correction() -> Dict[str, Any]:
    """Three identical-empty draws through the real prompt builder."""

    chat = Recorder("{}")
    run = _run(chat=chat, max_attempts=3)
    prompts = chat.prompts
    reasons = sorted(
        {str(row.get("terminal_reason") or "") for row in run["boundaries"]} - {""}
    )
    return {
        "model_calls": run["model_calls"],
        "third_prompt_equals_second": (
            prompts[2] == prompts[1] if len(prompts) >= 3 else None
        ),
        "third_prompt_model_env_var": (
            chat.models[2] if len(chat.models) >= 3 else None
        ),
        "distinct_prompts": len(set(prompts)),
        "failure_message": run["raised"],
        "boundary_terminal_reasons": reasons,
        "boundary_names": sorted({str(row.get("boundary")) for row in run["boundaries"]}),
    }


# --- capture 2: the A7 preservation golden ---------------------------------
def capture_preservation() -> Dict[str, Any]:
    """Four fixtures that all end in a usable payload. Must not move."""

    fixtures: Dict[str, Dict[str, Any]] = {}

    fixtures["clean_json"] = _run(chat=Recorder(WITH_REACTIONS), max_attempts=2)

    repair = Recorder(WITH_REACTIONS)
    fixtures["broken_then_repairable"] = _run(
        chat=Recorder(MALFORMED),
        max_attempts=2,
        repair_json=True,
        repair_chat_fn=repair,
    )

    fixtures["repair_unavailable_then_salvaged"] = _run(
        chat=Recorder(SALVAGEABLE), max_attempts=2, repair_json=False
    )

    fixtures["degenerate_then_recovering"] = _run(
        chat=Recorder("{}", WITH_REACTIONS), max_attempts=2
    )

    fixtures["inference_stage_untouched"] = _run(
        chat=Recorder("{}"),
        max_attempts=2,
        stage_name="inference",
        retry_on_empty_payload=False,
    )
    return fixtures


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", choices=("correction", "preservation"), default=None)
    args = parser.parse_args(argv)

    import t2pw

    capture: Dict[str, Any] = {"t2pw_file": t2pw.__file__.replace("\\", "/")}
    if args.only in (None, "correction"):
        capture["correction"] = capture_correction()
    if args.only in (None, "preservation"):
        capture["preservation"] = capture_preservation()

    Path(args.out).write_text(
        json.dumps(capture, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("T2PW:", t2pw.__file__)
    print(json.dumps(capture.get("correction", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
