"""ORCH-772 dev helper: render the Reaction Evidence .txt of the given leg dirs into C:/t/orch772/peek/."""
import sys
from pathlib import Path

from t2pw.app.reaction_evidence import build_report, load_run_artifacts, render_text

out = Path("C:/t/orch772/peek")
out.mkdir(parents=True, exist_ok=True)
for i, leg in enumerate(sys.argv[1:]):
    report = build_report(load_run_artifacts(leg))
    text = render_text(report)
    (out / f"{i:02d}.txt").write_text(text, encoding="utf-8")
    print(i, leg)
    print("\n".join(text.splitlines()[:6]))
    print("omitted:", len([e for e in report["omitted"] if e["counted"]]), "groups:",
          sorted({e["group"] for e in report["omitted"]}))
