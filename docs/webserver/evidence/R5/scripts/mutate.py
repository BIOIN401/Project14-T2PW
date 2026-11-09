"""R5 mutation check on a SCRATCH COPY of the tip (git archive, no .git). Each single mutation of the
mapping, isolation or deadline code must make the new tests fail. Every pytest run goes through
bounded_run + pinned_pytest with the heavy lock; the file is restored after every mutation.

usage: mutate.py SCRATCH_TREE OUTDIR
"""
import os
import subprocess
import sys
from pathlib import Path

TREE = sys.argv[1].replace("\\", "/")
EV = Path(sys.argv[2])
EV.mkdir(parents=True, exist_ok=True)
PY = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
PW = Path(TREE, "src/t2pw/service/pathway.py")
EX = Path(TREE, "src/t2pw/service/extraction.py")
PR = Path(TREE, "src/t2pw/service/post_run.py")
APP = Path(TREE, "src/t2pw/app/streamlit_app.py")
DF = Path(TREE, "src/t2pw/service/defaults.py")
NEW = "tests/test_service_run_pathway.py"
CLOB = "tests/test_stage0_case_c_retry_clobber.py"

M = {
    # -- failure mapping
    "input-status-fail": (PW, 'driver._fail(outcome, status=STATUS_ERROR, kind=KIND_CONTRACT, message=failure.message,',
                          'driver._fail(outcome, status=STATUS_FAIL, kind=KIND_CONTRACT, message=failure.message,'),
    "input-stage-not-set": (PW, '            outcome.stage = STAGE_INPUT\r\n            driver._fail(',
                            '            driver._fail('),
    "input-drops-no-text": (PW, 'ex.KIND_PDF_FAILED, ex.KIND_NO_TEXT)', 'ex.KIND_PDF_FAILED)'),
    "input-drops-no-pdf": (PW, '(ex.KIND_NO_PDF, ex.KIND_PDF_FAILED,', '(ex.KIND_PDF_FAILED,'),
    "contract-drops-incomplete": (PW, 'ex.KIND_STAGE_ONE_BOUNDARY_ERROR, ex.KIND_STAGE_ONE_INCOMPLETE)',
                                  'ex.KIND_STAGE_ONE_BOUNDARY_ERROR,)'),
    "contract-drops-boundary": (PW, '(ex.KIND_STAGE_ONE_BOUNDARY_ERROR, ', '('),
    "contract-status-error": (PW, 'driver._fail(outcome, status=STATUS_FAIL, kind=KIND_CONTRACT, message=message,',
                              'driver._fail(outcome, status=STATUS_ERROR, kind=KIND_CONTRACT, message=message,'),
    "ambiguous-kind-contract": (PW, 'driver._fail(outcome, status=STATUS_FAIL, kind=KIND_AMBIGUOUS,',
                                'driver._fail(outcome, status=STATUS_FAIL, kind=KIND_CONTRACT,'),
    "ambiguous-no-code": (PW, 'codes=[KIND_AMBIGUOUS])', 'codes=[])'),
    "ambiguous-branch-gone": (PW, 'if failure is not None and failure.kind == ex.KIND_AMBIGUOUS_REVIEW_SCOPE:',
                              'if False:'),
    "classify-unknown": (PW, 'kind=classify(text=joined, issue_codes=guard_codes, crashed=bool(exceptions)),\r\n                         message=message',
                         'kind=KIND_UNKNOWN,\r\n                         message=message'),
    "exception-not-reported": (PW, '        reporter.exception(exc)\r\n        extraction = None',
                               '        extraction = None'),
    "exception-status-fail": (PW, 'driver._fail(outcome, status=STATUS_ERROR if exceptions else STATUS_FAIL,\r\n                         kind=classify(text=joined, issue_codes=guard_codes,',
                              'driver._fail(outcome, status=STATUS_FAIL,\r\n                         kind=classify(text=joined, issue_codes=guard_codes,'),
    "drive-crash-not-caught": (PW, '    except Exception:  # noqa: BLE001 - a crash here is a result, never an unhandled error in the worker',
                               '    except ZeroDivisionError if False else KeyError:  # mutated'),
    "export-exception-escapes": (PW, '    except Exception as exc:  # noqa: BLE001 - as the app: the failure becomes the result',
                                 '    except KeyError as exc:  # mutated'),
    "scope-reconcile-gone": (PW, 'if driver._reconcile_stage0_scope(at, paper, outcome):', 'if False:'),
    "no-reactions-gone": (PW, 'if not merged_counts["reactions"] and not merged_counts["transports"]:', 'if False:'),
    "payload-guard-gone": (PW, 'if not isinstance(final_payload, dict) or not final_payload:', 'if False:'),
    "no-artifacts-gone": (PW, '    if not artifacts:\r\n        codes.extend', '    if False:\r\n        codes.extend'),
    "gate-failure-gone": (PW, 'if blocking_gate or error_count:', 'if False:'),
    "review-guard-gone": (PW, 'if state.get("refinement_working_json") is None:', 'if False:'),
    "research-runs-export": (PW, '    if research:\r\n        outcome.stage = STAGE_RESEARCH_REPORT', '    if False:\r\n        outcome.stage = STAGE_RESEARCH_REPORT'),
    "stale-export-error-message": (PW, 'reporter.error(f"PWML export failed: {pwml_result.get(\'error\', \'unknown\')}")', 'pass'),
    # -- isolation
    "no-start-reset": (PW, '    pr.reset_quarantine_state(outputs_dir, state=state)\r\n    outcome.stage = STAGE_INPUT',
                       '    outcome.stage = STAGE_INPUT'),
    "no-pre-audit-reset": (PW, '    state.pop("post_pipeline_artifacts", None)\r\n    pr.reset_quarantine_state(outputs_dir, state=state)\r\n',
                           '    state.pop("post_pipeline_artifacts", None)\r\n'),
    "no-pop": (PW, '    state.pop("post_pipeline_artifacts", None)\r\n', ''),
    "reset-skips-keys": (PR, '    for key in QUARANTINE_STATE_KEYS:\r\n        state.pop(key, None)\r\n', ''),
    "reset-skips-disk": (PR, '            clear_quarantine_artifacts(outputs_dir)\r\n', '            pass\r\n'),
    "shared-state": (PW, '    state: Dict[str, Any] = {}\r\n    started = time.monotonic()',
                     '    state = _SHARED\r\n    started = time.monotonic()'),
    # -- deadline
    "no-grace": (PW, 'deadline = float(request.deadline) - bridge.GRACE_S', 'deadline = float(request.deadline)'),
    "no-start-deadline-check": (PW, 'if float(request.deadline) - time.time() <= 0:', 'if False:'),
    "audit-timeout-uncapped": (PW, 'audit_timeout = max(1, int(min(audit_timeout, deadline - time.time())))', 'pass'),
    "audit-timeout-always-left": (PW, 'audit_timeout = max(1, int(min(audit_timeout, deadline - time.time())))',
                                  'audit_timeout = max(1, int(deadline - time.time()))'),
    "no-pre-audit-deadline": (PW, 'if out_of_time("audit and DB mapping did not finish inside the time budget"):', 'if False:'),
    "no-pre-export-deadline": (PW, 'if out_of_time("PWML export did not finish inside the time budget"):', 'if False:'),
    "deadline-branch-gone": (PW, 'if failure is not None and failure.kind == ex.KIND_DEADLINE_EXCEEDED:', 'if False:'),
    "deadline-not-published": (PW, '    state[ex.DEADLINE_STATE_KEY] = deadline\r\n', ''),
    "no-deadline-before-stage1": (EX, '    expired = _deadline_failure(state)\r\n    if expired is not None:\r\n        return ExtractionResult(failure=expired)\r\n    stage_one = run_stage_one(',
                                  '    stage_one = run_stage_one('),
    "no-deadline-before-stage2": (EX, '    expired = _deadline_failure(state)\r\n    if expired is not None:\r\n        return ExtractionResult(failure=expired)\r\n    return run_stage_two(',
                                  '    return run_stage_two('),
    "rag-deadline-never-passed": (EX, 'return {} if deadline is None else {"deadline_seconds": max(1.0, float(deadline) - time.time())}',
                                  'return {}'),
    # -- gate, delivery, defaults
    "run-extraction-skips-gate": (EX, '    gated = ambiguous_review_gate(stage_zero.pathway_context, state=state, reporter=reporter)\r\n    if gated is not None:\r\n        return ExtractionResult(failure=gated)\r\n', ''),
    "gate-no-pipeline-ready": (EX, '    state["pipeline_ready"] = False\r\n    state["pathway_context"] = pathway_context\r\n    message = (', '    state["pathway_context"] = pathway_context\r\n    message = ('),
    "scrub-disabled": (PW, 'outcome.artifacts = _scrub_server_paths(outcome.artifacts, out_dir)', 'pass'),
    "scrub-by-shape": (PW, 'return {name: pattern.sub("", blob) if isinstance(blob, str) else blob for name, blob in artifacts.items()}',
                       'return {name: re.sub(r"(/[A-Za-z_0-9.]+)+", "", blob) if isinstance(blob, str) else blob for name, blob in artifacts.items()}'),
    "image-not-saved": (PW, '            result.image_path = out_dir / IMAGE_NAME\r\n', ''),
    "form-hardcodes-chunk": (APP, 'value=_defaults.CHUNK_SIZE,', 'value=10000,'),
    "constant-drifts": (DF, 'AUDIT_MAX_ROUNDS = 4', 'AUDIT_MAX_ROUNDS = 5'),
}
SHARED_LINE = "IMAGE_NAME = \"pathway.png\"\r\n"


def run(name, tests):
    cmd = ["C:/Program Files/Git/usr/bin/bash.exe", "C:/t/ws-R5-tmp/rt.sh", TREE, f"mut-{name}", EV.as_posix(), "120", *tests, "--no-header",
           "-p", "no:cacheprovider", "-x"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", stdin=subprocess.DEVNULL, timeout=500).stdout


LO, HI = (int(x) for x in sys.argv[3].split(":"))
results = []
for name, (path, old, new) in list(M.items())[LO:HI]:
    text = path.read_text(encoding="utf-8", newline="") if False else open(path, encoding="utf-8", newline="").read()
    if text.count(old) != 1:
        results.append((name, f"NOT APPLIED count={text.count(old)}"))
        continue
    mutated = text.replace(old, new)
    if name == "shared-state":
        mutated = mutated.replace(SHARED_LINE, SHARED_LINE + "_SHARED = {}\r\n")
    open(path, "w", encoding="utf-8", newline="").write(mutated)
    try:
        tests = [NEW, CLOB] if path == EX and "gate" in name else [NEW]
        out = run(name, tests)
    finally:
        open(path, "w", encoding="utf-8", newline="").write(text)
    last = [l for l in out.splitlines() if l.startswith("[mut-")]
    verdict = "KILLED" if last and "exit=1" in last[-1] else ("SURVIVED" if last and "exit=0" in last[-1] else "ERROR")
    results.append((name, verdict))
    print(name, verdict, flush=True)
lines = [f"{v}\t{n}" for n, v in results]
tally = {k: sum(1 for _, v in results if v == k) for k in sorted({v.split()[0] for _, v in results})}
(EV / f"tally-{LO}.txt").write_text("\n".join(lines + [str(tally)]) + "\n", encoding="utf-8")
print(tally)
