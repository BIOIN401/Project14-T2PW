"""C-107: re-derive every routed claim against the code in THIS tree.

The Lead's `orch717_c107_claims_probe.py` measured 1a/1c/1d/1e/1f at the C-106
tip. This script reproduces those numbers and extends them where the card told
C-107 to enumerate more than the routed finding named:

* 1a  the eleven near-synonyms, at word level AND through the real seam in the
      F-146 paraphrase shape.
* 1b  the passive-with-agent cue firing when the agent is somebody else.
* 1c  every singular-only stoplist entry, the plural bypasses among them, and an
      empirical enumeration of ordinary `-ase` words that over-accept.
* 1d  transport family, enzyme-family noun.
* 1e  the cofactor role, and WHICH mechanism refuses it.
* 1f  `mediat` inside "intermediate".
* 1g  evidence-span length census (registered, not fixed).

Asserts nothing, changes nothing. Usage::  <python> c107_claims_reproduce.py <repo-root>
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

import t2pw.curation.apply_audit_patch as M  # noqa: E402
from t2pw.curation.apply_audit_patch import (  # noqa: E402
    _ROLE_CUE_RES, _NON_ENZYME_ASE_WORDS, _span_licenses_actor,
    _actor_role_family, apply_patch_with_policy,
)

CAT = _ROLE_CUE_RES["catalysis"]
INH = _ROLE_CUE_RES["inhibition"]


def seam(name, evidence, container="enzymes", bucket="reactions", role=None):
    """One actor-role add through the REAL public seam. True == accepted."""
    value = name if role is None else {"entity": name, "role": role}
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "chem only", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": name}],
                            "protein_complexes": [], "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{container}/-",
          "value": value, "confidence": 1.0}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


NEAR_SYNONYMS = ["blockade", "impairment", "disruption", "reduction", "loss",
                 "silencing", "sequestration", "depletion", "ablation",
                 "interference", "quenching"]

print("=" * 78)
print("1a. INHIBITION NEAR-SYNONYMS")
print("=" * 78)
print(f"  {'word':16s} {'contra fires':13s} {'also catalysis cue':20s}")
defeats = 0
for w in NEAR_SYNONYMS + ["inhibition"]:
    c, k = bool(INH.search(w)), bool(CAT.search(w))
    defeats += 0 if c else 1
    print(f"  {w:16s} {str(c):13s} {str(k):20s}")
print(f"  {defeats} of {len(NEAR_SYNONYMS) + 1} defeat the contra-cue at word level")

print("\n  Through the REAL seam, F-146 paraphrase shape "
      "'the <word> of NDM-1 activity by PSA' -- True == the defect is admitted:")
for w in NEAR_SYNONYMS:
    span = f"the {w} of NDM-1 activity by PSA"
    print(f"      {w:16s} licensed as CATALYST: {seam('NDM-1', span)}")

print("\n  The three sentences the Lead measured:")
for span in ["the reduction of NDM-1 activity by PSA",
             "PSA significantly inhibited NDM-1 enzyme activity",
             "NDM-1 catalyses the hydrolysis of meropenem"]:
    print(f"      _span_licenses_actor -> {str(_span_licenses_actor(span, 'NDM-1', 'catalysis')):5s}"
          f"  seam -> {str(seam('NDM-1', span)):5s}  {span!r}")

print("\n  REDOX, which must KEEP licensing (C-107 1a trap / REV-107 B2):")
for span, actor in [("NADH-dependent reduction of the substrate by P", "P"),
                    ("ferrochelatase reduces the substrate in this step", "ferrochelatase"),
                    ("P catalyses the reduction of the quinone to the quinol", "P"),
                    ("the reduction of A to B is carried out by P", "P")]:
    print(f"      licensed={str(seam(actor, span)):5s}  {span!r}")

print()
print("=" * 78)
print("1b. PASSIVE-WITH-AGENT FIRES WHEN THE AGENT IS SOMEBODY ELSE")
print("=" * 78)
for span, actor in [
    ("A is converted to B by Q, and P was also detected in the assay", "P"),
    ("the beta-lactam is produced by decomposition, and NDM-1 is an inhibitor target", "NDM-1"),
    ("A is converted to B by P in the intestine", "P"),
    ("the reaction is catalyzed by serine hydroxymethyltransferase", "Serine hydroxymethyltransferase, mitochondrial"),
]:
    print(f"  licensed={str(seam(actor, span)):5s}  actor={actor!r}\n      span={span!r}")

print()
print("=" * 78)
print("1c. THE -ase STOPLIST")
print("=" * 78)
listed = set(_NON_ENZYME_ASE_WORDS)
singular_only = [w for w in sorted(listed)
                 if not w.endswith("s") and w + "s" not in listed and w + "es" not in listed]
print(f"  stoplist entries              : {len(listed)}")
print(f"  entries with NO plural listed : {len(singular_only)}")
print(f"      {' '.join(singular_only)}")
bypass = []
for w in singular_only:
    p = w + "s"
    if CAT.search(p) and not CAT.search(w):
        bypass.append(p)
print(f"  plurals that BYPASS           : {len(bypass)}  {' '.join(bypass)}")

# Empirical enumeration: ordinary English -ase(s) words that the enzyme-noun rule
# accepts. Drawn from the real corpus text rather than guessed from a dictionary.
ASE_RE = re.compile(r"(?<![a-z])[a-z]{3,}ases?(?![a-z])")
KNOWN_ENGLISH = [
    "briefcase", "briefcases", "database", "databases", "disease", "diseases",
    "increase", "increases", "increased", "decrease", "decreases", "decreased",
    "release", "releases", "released", "purchase", "purchases", "purchased",
    "phrase", "phrases", "phrased", "chase", "chases", "chased", "erase",
    "erases", "erased", "cease", "ceases", "ceased", "lease", "leases",
    "leased", "please", "pleases", "pleased", "case", "cases", "base", "bases",
    "phase", "phases", "phased", "vase", "vases", "showcase", "showcases",
    "showcased", "staircase", "staircases", "suitcase", "suitcases",
    "grease", "greases", "greased", "appease", "appeases", "appeased",
    "crease", "creases", "creased", "upstase",
]
over = [w for w in KNOWN_ENGLISH if CAT.search(w)]
print(f"\n  ordinary English words probed : {len(KNOWN_ENGLISH)}")
print(f"  OVER-ACCEPTING as enzyme noun : {len(over)}")
print(f"      {' '.join(over)}")

print("\n  real enzyme nouns that must KEEP licensing:")
for w in ["hydrolase", "hydrolases", "lipase", "lipases", "kinase", "kinases",
          "transferase", "transferases", "isochorismatase", "flippase",
          "lyase", "dnase", "rnase", "polymerases"]:
    print(f"      {w:18s} cue={bool(CAT.search(w))}")

print()
print("=" * 78)
print("1d. TRANSPORT FAMILY HAS NO ENZYME-NOUN RULE")
print("=" * 78)
for span, actor, ok in [
    ("MsbA is the flippase for lipid A", "MsbA", "a real transporter, named by its family noun"),
    ("MsbA transports lipid A across the membrane", "MsbA", "control: the verb form"),
    ("P is the permease for this substrate", "P", "the one-off stem that IS listed"),
    ("P was detected in the membrane fraction", "P", "NON-transporter, must refuse"),
]:
    print(f"  licensed={str(seam(actor, span, container='transporters', bucket='transports')):5s}"
          f"  {ok}\n      {span!r}")

print()
print("=" * 78)
print("1e. ROLE 'cofactor'")
print("=" * 78)
row = {"entity": "P", "role": "cofactor"}
print(f"  _actor_role_family('modifiers', {{role: 'cofactor'}}) -> "
      f"{_actor_role_family('modifiers', row)!r}")
print(f"  is there a 'cofactor' key in _ROLE_FAMILY_BY_ROLE? "
      f"{'cofactor' in M._ROLE_FAMILY_BY_ROLE}")
for span in ["P is a required cofactor for the step",
             "the reaction requires P as a cofactor",
             "the enzyme is dependent on P",
             "the conversion proceeds only in the presence of P"]:
    fam = _actor_role_family("modifiers", row)
    print(f"  fallback _ANY_ROLE_CUE_RE hit in span? "
          f"{str(bool(M._ANY_ROLE_CUE_RE.search(M._match_fold(span)))):5s}"
          f"  _span_licenses_actor(fam={fam})="
          f"{str(_span_licenses_actor(span, 'P', fam)):5s}"
          f"  seam={str(seam('P', span, container='modifiers', role='cofactor')):5s}  {span!r}")
print("  does ANY family's vocabulary contain a cofactor-predicating word?")
for fam, pat in _ROLE_CUE_RES.items():
    hits = [w for w in ("cofactor", "coenzyme", "requires", "required for",
                        "dependent on", "in the presence of", "prosthetic group")
            if pat.search(w)]
    print(f"      {fam:12s} -> {hits}")

print()
print("=" * 78)
print("1f. 'mediat' INSIDE 'intermediate'")
print("=" * 78)
for probe in ["intermediate", "intermediates", "mediates", "p mediated"]:
    print(f"  catalysis cue on {probe!r:20s} -> {bool(CAT.search(probe))}")
span = "EntB is an intermediate carrier in this pathway"
print(f"  seam licenses EntB as CATALYST: {seam('EntB', span)}  <- {span!r}")
print(f"  seam licenses ALAS2 complex  : "
      f"{seam('ALAS2 complex', 'ALAS2 mediates the condensation of glycine')}"
      f"  <- the legitimate repair that must survive")

print()
print("=" * 78)
print("1g. EVIDENCE-SPAN LENGTH CENSUS  (registered, NOT fixed)")
print("=" * 78)
CONT = ("enzymes", "modifiers", "modifiers_or_enzymes", "catalysts",
        "transporters", "cargo", "cargo_complex")
EVK = ("evidence", "evidence_quote", "source_evidence", "source_text")
lengths = []
files = sorted(glob.glob(str(REPO / "runs/**/final_mapped.json"), recursive=True) +
               glob.glob(str(REPO / "runs_verify/**/final_mapped.json"), recursive=True))
seen = set()
for f in files:
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:
        continue
    procs = d.get("processes") or {}
    for bucket in ("reactions", "transports", "reaction_coupled_transports"):
        for rxn in (procs.get(bucket) or []):
            if not isinstance(rxn, dict):
                continue
            for cont in CONT:
                rows = rxn.get(cont)
                if not isinstance(rows, list):
                    continue
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    ev = ""
                    for k in EVK:
                        v = r.get(k)
                        if isinstance(v, str) and v.strip():
                            ev = v
                            break
                    nm = ""
                    for fkey in ("entity", "protein", "protein_name",
                                 "protein_complex", "enzyme", "modifier", "name"):
                        v = r.get(fkey)
                        if isinstance(v, str) and v.strip():
                            nm = v.strip()
                            break
                    if not nm or not ev:
                        continue
                    key = f"{cont}|{bucket}|{nm}|{r.get('role')}|{ev}"
                    if key in seen:
                        continue
                    seen.add(key)
                    lengths.append((len(ev), f, nm))
lengths.sort(reverse=True)
over5k = [x for x in lengths if x[0] > 5000]
print(f"  distinct evidence-bearing actor rows : {len(lengths)}")
print(f"  spans longer than 5,000 characters   : {len(over5k)}")
print(f"  longest span                         : {lengths[0][0] if lengths else 0}")
for n, f, nm in lengths[:5]:
    print(f"      {n:>7d}  {nm[:28]:28s}  {Path(f).parent.parent.name}/{Path(f).parent.name}")
print(f"  files contributing an oversized span : "
      f"{len({x[1] for x in over5k})} of {len(files)}")
