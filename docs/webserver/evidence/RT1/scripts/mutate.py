"""Apply / restore one named revert on the scratch tree C:/t/ev-RT1/rev/base (never on the worktree)."""
import pathlib
import subprocess
import sys

SCRATCH = pathlib.Path("C:/t/ev-RT1/rev/base")
CLIENT = "src/t2pw/llm/client.py"
PATHWAY = "src/t2pw/service/pathway.py"

# name -> (file, old, new)
MUT = {
    "M01_no_web_fields": (CLIENT, "    if _WEB_RUN:\n        # Same row as", "    if False:\n        # Same row as"),
    "M02_curator_silent": (CLIENT, "            if _WEB_RUN\n            else None\n        )\n\n        def _trace", "            if False\n            else None\n        )\n\n        def _trace"),
    "M03_no_trace_activation": (PATHWAY, "    if own_trace:\n        leg_trace.activate(", "    if False:\n        leg_trace.activate("),
    "M04_trace_is_a_download": (PATHWAY, "    result = bridge._to_result(outcome, out_dir)\n",
                                "    result = bridge._to_result(outcome, out_dir)\n    result.report_paths[leg_trace.LEG_TRACE_NAME] = out_dir / leg_trace.LEG_TRACE_NAME\n"),
    "M05_no_cap": (CLIENT, "    if not (_WEB_RUN and stage in LLM_CALL_CAP_STAGES):\n", "    if True:\n"),
    "M06_late_result_recorded": (CLIENT, '            box["resp"] = _client.chat.completions.create(**request)\n',
                                 '            box["resp"] = _client.chat.completions.create(**request)\n            _record_usage(box["resp"])\n'),
    "M07_every_stage_capped": (CLIENT, "    if not (_WEB_RUN and stage in LLM_CALL_CAP_STAGES):\n", "    if not _WEB_RUN:\n"),
    "M08_cap_outside_web": (CLIENT, "    if not (_WEB_RUN and stage in LLM_CALL_CAP_STAGES):\n", "    if not (stage in LLM_CALL_CAP_STAGES):\n"),
    "M09_cap_does_not_retry": (CLIENT, '            logger.warning("%s; retrying (attempt %d/%d).", e, attempt + 1, max_retries)\n',
                              '            logger.warning("%s; retrying (attempt %d/%d).", e, attempt + 1, max_retries)\n            break\n'),
    "M10_worker_not_daemon": (CLIENT, 'name="t2pw-llm-call", daemon=True', 'name="t2pw-llm-call", daemon=False'),
    "M11_scope_not_restored": (CLIENT, "    finally:\n        _WEB_RUN = previous\n", "    finally:\n        pass\n"),
    "M12_web_scope_on_by_default": (CLIENT, "_WEB_RUN = False\n# Bound at import", "_WEB_RUN = True\n# Bound at import"),
    "M13_cap_is_not_720": (CLIENT, "\nLLM_CALL_CAP_SECONDS = 720.0\n", "\nLLM_CALL_CAP_SECONDS = 600.0\n"),
    "M14_diagnostics_gain_fields": (CLIENT, '            "attempt_log": list(self.attempt_log),\n        }',
                                    '            "attempt_log": list(self.attempt_log),\n            "duration_s": 0.0,\n        }'),
    "M15_run_paper_enters_scope": (PATHWAY, "    started = time.monotonic()\n    try:\n        _drive(request, paper, outcome, state, CollectingReporter(), deadline)",
                                   "    started = time.monotonic()\n    from t2pw.llm import client as _c\n    _c._WEB_RUN = True\n    try:\n        _drive(request, paper, outcome, state, CollectingReporter(), deadline)"),
}


def main() -> int:
    action, name = sys.argv[1], sys.argv[2]
    rel, old, new = MUT[name]
    target = SCRATCH / rel
    if action == "restore":
        target.write_bytes(subprocess.check_output(["git", "-C", "C:/t/ws-RT1", "show", f"HEAD:{rel}"]))
        return 0
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        print(f"{name}: pattern found {text.count(old)} times", file=sys.stderr)
        return 3
    target.write_text(text.replace(old, new), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
