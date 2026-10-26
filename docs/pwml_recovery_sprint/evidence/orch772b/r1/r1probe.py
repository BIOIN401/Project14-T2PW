"""ORCH-772b round 1 probe: the reader's cause and its label on the two live legs review r0 named (B-1, N-1)."""
import importlib.util
import sys
from pathlib import Path

from t2pw.app.reaction_evidence import known_issue_id

EV = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("cause", EV / "orch772b_no_pwml_cause.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
P = Path("C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
for leg in [P / "runs_verify/2026-08-25_1216/papers/PMC12856317/strict",
            *sorted(P.glob("runs_papers/ORCH-759*/legs/menaquinone-strict/*/papers/*/strict"))]:
    cause = mod.recorded_no_pwml_cause(leg)
    print(f"{leg.relative_to(P).as_posix()}\n    cause: {cause}\n    KI: {known_issue_id(cause)}")
