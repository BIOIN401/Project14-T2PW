"""ORCH-717: re-check my own 1a measurement after the C-107 author corrected it.

The C-107 implementer reports that my `orch717_c107_claims_probe.py` misstates 1a
in both directions. This checks that claim against the BASE tree, because a
correction to the orchestrator's own measurement is exactly the kind that nobody
else will make.

My probe reported two things:

  (a) 11 of 12 near-synonyms DEFEAT the contra-cue      -- word-level
  (b) 'the reduction of NDM-1 activity by PSA' licenses -- end-to-end

Both are true. The author's point is that (a) and (b) do not compose the way my
card's prose implied: defeating the contra is NECESSARY but not SUFFICIENT. A
window still needs a catalysis cue to license anything, and in the bare frame
'the <word> of X activity by Y' only `reduction of` supplies one -- because
`reduction of` is itself in the catalysis set. The other ten defeat the contra
while carrying no catalysis cue at all, so the bare frame refuses them for a
DIFFERENT reason than the guard working.

Put a real catalysis cue in the window and the author says all eleven admit.
That is the shape the card must close, and it is a sharper statement of the
finding than mine.

Usage::  <python> orch717_1a_frame_recheck.py <base-tree-root>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))

from t2pw.curation.apply_audit_patch import (      # noqa: E402
    _ROLE_CUE_RES, _span_licenses_actor,
)

WORDS = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
         "sequestration", "depletion", "ablation", "interference", "quenching"]

print(f"tree under measurement: {ROOT}")
print()
print("=" * 74)
print("FRAME 1 -- the bare frame my probe implied (no independent catalysis cue)")
print("=" * 74)
print(f"  {'word':16s} {'contra?':8s} {'catalysis?':11s} licenses NDM-1 as CATALYST")
bare_admits = 0
for w in WORDS:
    span = f"the {w} of NDM-1 activity by PSA"
    lic = _span_licenses_actor(span, "NDM-1", "catalysis")
    bare_admits += bool(lic)
    print(f"  {w:16s} {str(bool(_ROLE_CUE_RES['inhibition'].search(span))):8s} "
          f"{str(bool(_ROLE_CUE_RES['catalysis'].search(span))):11s} {lic}")
print(f"\n  admits the defect in the BARE frame : {bare_admits} of {len(WORDS)}")

print()
print("=" * 74)
print("FRAME 2 -- with a real catalysis cue in the window ('is mediated by')")
print("=" * 74)
cued_admits = 0
for w in WORDS:
    span = f"the {w} of NDM-1 activity is mediated by PSA"
    lic = _span_licenses_actor(span, "NDM-1", "catalysis")
    cued_admits += bool(lic)
    print(f"  {w:16s} licenses NDM-1 as CATALYST: {lic}")
print(f"\n  admits the defect WITH a cue present : {cued_admits} of {len(WORDS)}")

print()
print("=" * 74)
print("CONTROL -- the sentence the paper actually contains must still refuse")
print("=" * 74)
for span in ["PSA significantly inhibited NDM-1 enzyme activity",
             "NDM-1 catalyses the hydrolysis of meropenem"]:
    print(f"  {str(_span_licenses_actor(span, 'NDM-1', 'catalysis')):5s} <- {span!r}")

print()
print("VERDICT ON MY OWN PROBE:")
print("  '11 of 12 defeat the contra-cue' -- CORRECT as stated (a word-level fact).")
print("  But defeating the contra is NECESSARY, NOT SUFFICIENT: a window still")
print("  needs a catalysis cue. My card's prose ran the two together.")
print(f"  Bare frame admits {bare_admits}; with a cue present, {cued_admits}.")
print("  The author's refinement is sharper and it is the one to charter against.")
