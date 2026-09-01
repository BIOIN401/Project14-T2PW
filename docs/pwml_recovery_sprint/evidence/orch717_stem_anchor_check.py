import sys
sys.path.insert(0, r"C:/t/rev107/src")
from t2pw.curation.apply_audit_patch import _span_licenses_actor
print("Measured against C:/t/rev107 @ b569205 (C-107 correction round 1 tip)")
print()
print("LEGITIMATE CATALYSIS SPANS -- do they still license?")
cases = [
    ("the reductase P4X catalyses the conversion of A to B", "P4X"),
    ("the nitroreductase NfsB catalyses the conversion of A to B", "NfsB"),
    ("the oxidoreductase YkgC catalyses the conversion of A to B", "YkgC"),
    ("the purified reductase NfsB catalyses the conversion of A to B", "NfsB"),
    ("the blocker protein P4X catalyses the conversion of A to B", "P4X"),
    ("the silencer complex P4X catalyses the conversion of A to B", "P4X"),
    ("interferon IRF3 catalyses the conversion of A to B", "IRF3"),
    ("the hydrolase P4X catalyses the conversion of A to B", "P4X"),
]
refused = 0
for span, actor in cases:
    lic = _span_licenses_actor(span, actor, "catalysis")
    if not lic:
        refused += 1
    tag = "ACCEPT " if lic else "REFUSED"
    print(f"  {tag}  actor={actor:6s} {span!r}")
print(f"\n  falsely refused: {refused} of {len(cases)}  (the last one is the control and must ACCEPT)")
print()
print("THE INTENDED REFUSAL must still hold:")
for span, actor in [("the reduction of NDM-1 activity is mediated by PSA", "NDM-1"),
                    ("the reduction of NDM-1 is mediated by PSA", "NDM-1")]:
    print(f"  {'ACCEPT ' if _span_licenses_actor(span, actor, 'catalysis') else 'REFUSED'}  {span!r}")
