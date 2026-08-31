"""ORCH-717: verify REV-105's routed claims against the SHIPPED code, before C-107 acts on them.

"Verify a subagent's load-bearing claims yourself" -- the wave lesson. These are
the claims C-107 is chartered on, exercised through the real production
predicates rather than read out of a report. Asserts nothing, changes nothing.

Usage::  <python> orch717_c107_claims_probe.py <repo-root>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.curation.apply_audit_patch import (      # noqa: E402
    _ROLE_CUE_RES, _NON_ENZYME_ASE_WORDS, _span_licenses_actor,
)

print("=" * 76)
print("1a. INHIBITION NEAR-SYNONYMS vs the catalysis contra-cue")
print("=" * 76)
contra = _ROLE_CUE_RES["inhibition"]
cat = _ROLE_CUE_RES["catalysis"]
words = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
         "sequestration", "depletion", "ablation", "interference", "quenching",
         "inhibition"]
print(f"\n  {'word':16s} {'contra fires':13s} {'ALSO a catalysis cue':21s} verdict")
defeats = 0
for w in words:
    c = bool(contra.search(w))
    k = bool(cat.search(w))
    if not c:
        defeats += 1
    verdict = "DEFEATS contra" if not c else "caught"
    if not c and k:
        verdict = "DEFEATS + IS A CATALYSIS CUE"
    print(f"  {w:16s} {str(c):13s} {str(k):21s} {verdict}")
print(f"\n  {defeats} of {len(words)} defeat the contra-cue")

print("\n  The live sentence, through the REAL predicate:")
for span in [
    "the reduction of NDM-1 activity by PSA",
    "PSA significantly inhibited NDM-1 enzyme activity",
    "NDM-1 catalyses the hydrolysis of meropenem",
]:
    lic = _span_licenses_actor(span, "NDM-1", "catalysis")
    print(f"      licenses NDM-1 as CATALYST: {str(lic):5s}  <- {span!r}")

print()
print("=" * 76)
print("1c. THE -ase STOPLIST PLURAL BYPASS")
print("=" * 76)
listed = set(_NON_ENZYME_ASE_WORDS)
singular_only = []
for w in sorted(listed):
    if w.endswith("s"):
        continue
    if w + "s" not in listed and w + "es" not in listed:
        singular_only.append(w)
print(f"\n  stoplist entries              : {len(listed)}")
print(f"  entries with NO plural listed : {len(singular_only)}")
print(f"      {singular_only}")
print("\n  does the unlisted plural slip through the enzyme-noun rule?")
for w in singular_only:
    plural = w + "s"
    sing = bool(_ROLE_CUE_RES["catalysis"].search(w))
    plur = bool(_ROLE_CUE_RES["catalysis"].search(plural))
    flag = "  <-- BYPASS" if (plur and not sing) else ""
    print(f"      {w:12s} cue={str(sing):5s}   {plural:13s} cue={str(plur):5s}{flag}")

print()
print("=" * 76)
print("1f. 'mediat' MATCHES INSIDE 'intermediate'")
print("=" * 76)
for probe in ["intermediate", "intermediates", "mediates", "an intermediate of the pathway"]:
    print(f"  catalysis cue on {probe!r:34s} -> {bool(_ROLE_CUE_RES['catalysis'].search(probe))}")
print("\n  through the real predicate, a span that names a protein beside 'intermediate':")
span = "EntB is an intermediate carrier in this pathway"
print(f"      licenses EntB as CATALYST: {_span_licenses_actor(span, 'EntB', 'catalysis')}  <- {span!r}")

print()
print("=" * 76)
print("1d / 1e. TRANSPORT ENZYME-NOUN, and the cofactor role")
print("=" * 76)
for span, actor, fam in [
    ("MsbA is the flippase for lipid A", "MsbA", "transport"),
    ("MsbA transports lipid A across the membrane", "MsbA", "transport"),
    ("NDM-1 requires Zn2+ as a cofactor", "Zn2+", "other"),
    ("Zn2+ is the cofactor of this enzyme", "Zn2+", "other"),
]:
    print(f"  family={fam:10s} licenses {actor:6s}: "
          f"{str(_span_licenses_actor(span, actor, fam)):5s}  <- {span!r}")
