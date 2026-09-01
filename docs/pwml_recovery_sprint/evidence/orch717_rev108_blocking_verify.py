"""ORCH-717: verify REV-108's blocking finding MYSELF before spending a correction round.

The handoff's discipline: the Lead verified REV-107's blocking finding at 7 of 8
before sending the author back. Same here.

REV-108 claims four rephrasings in which the actor IS the thing being shut down
go base=REFUSE -> tip=ACCEPT, i.e. the catalysis contra was weakened. That is
merge rule 6 and would be a reject.

Driven through the REAL public seam ``apply_patch_with_policy``, exactly as
``c107_battery.py`` drives it -- never against the private regex.

Usage::  <python> orch717_rev108_blocking_verify.py <code-root>
Exit 0 always; read the printed verdicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CODE / "src"))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402
import t2pw.curation.apply_audit_patch as _m  # noqa: E402

print("code loaded from:", _m.__file__)


def licenses(container: str, name: str, evidence: str) -> bool:
    """True if the guard ADMITS this actor row.

    CORRECTED. The first version of this probe built ``{"reactions": [...]}`` and
    addressed ``/reactions/0/<container>/-``. That path does NOT match the
    actor-role path pattern, so the guard was never reached at all and EVERY span
    came back ACCEPTed -- at base as well as tip, including cases F-155 records as
    REFUSED at base. The disagreement with two independent records is what exposed
    it; the failed run is preserved beside this one.

    This mirrors ``c107_battery.py``'s ``run()`` exactly: the ``processes``
    envelope, the ``/processes/<bucket>/0/<container>/-`` path, ``stage="probe"``,
    and the verdict read from ``summary.accepted_count``.
    """
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": name}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {"reactions": [proc]}}
    op = {"op": "add", "path": f"/processes/reactions/0/{container}/-",
          "value": name, "confidence": 1.0, "evidence": evidence}
    _out, rep = apply_patch_with_policy(payload, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


# REV-108's four blocking spans. In every one the ACTOR is the thing being shut
# down, so a CATALYST row for it must be REFUSED.
BLOCKING = [
    "P4X is a target of the inhibitor and catalyses the conversion of A to B",
    "P4X was subject to inhibitors during the assay, yet catalyses A to B",
    "P4X, whose inhibitor was characterised, catalyses the conversion of A to B",
    "the repressor bound P4X and the catalysis of A to B stopped",
]

# Controls that must NOT move: appositive namings, which member (d) exists to
# stop falsely refusing. These SHOULD be ACCEPTed at the tip.
APPOSITIVE = [
    "the repressor complex P4X catalyses the conversion of A to B",
    "the suppressor protein P4X catalyses the conversion of A to B",
    "the inhibitor protein P4X catalyses the conversion of A to B",
]

# The pinned property: a span saying the actor's activity is attenuated must stay
# REFUSED. This is C5's own case.
PINNED = [
    "A significantly inhibited P4X activity in the assay",
]

print()
print("=" * 78)
print("BLOCKING CANDIDATES -- actor IS shut down; a CATALYST row must be REFUSED")
print("=" * 78)
bad = 0
for span in BLOCKING:
    acc = licenses("enzymes", "P4X", span)
    flag = "  <<< ADMITTED (contra did not fire)" if acc else ""
    if acc:
        bad += 1
    print(f"  {'ACCEPT' if acc else 'REFUSE'}{flag}\n     {span!r}")
print(f"\n  ADMITTED (want 0): {bad} / {len(BLOCKING)}")

print()
print("=" * 78)
print("APPOSITIVE NAMINGS -- member (d)'s purpose; these SHOULD be ACCEPTed")
print("=" * 78)
missed = 0
for span in APPOSITIVE:
    acc = licenses("enzymes", "P4X", span)
    if not acc:
        missed += 1
    print(f"  {'ACCEPT' if acc else 'REFUSE'}{'  <<< falsely refused' if not acc else ''}"
          f"\n     {span!r}")
print(f"\n  REFUSED (want 0): {missed} / {len(APPOSITIVE)}")

print()
print("=" * 78)
print("PINNED -- attenuation of the actor's activity must stay REFUSED")
print("=" * 78)
leaked = 0
for span in PINNED:
    acc = licenses("enzymes", "P4X", span)
    if acc:
        leaked += 1
    print(f"  {'ACCEPT' if acc else 'REFUSE'}{'  <<< LEAKED' if acc else ''}"
          f"\n     {span!r}")
print(f"\n  ADMITTED (want 0): {leaked} / {len(PINNED)}")

print()
print("=" * 78)
print(f"SUMMARY  blocking_admitted={bad}  appositive_refused={missed}  pinned_leaked={leaked}")
print("=" * 78)
