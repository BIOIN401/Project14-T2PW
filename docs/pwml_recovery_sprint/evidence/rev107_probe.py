"""REV-107 independent probe. Runs identically at BASE and TIP; the only
difference is PYTHONPATH. Everything that can go through the real production
seam does -- apply_patch_with_policy -- rather than the predicate alone.

Usage:  <python> rev107_probe.py <label>
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy

from t2pw.curation.apply_audit_patch import (
    apply_patch_with_policy,
    _span_licenses_actor,
    _actor_role_family,
    _match_fold,
    _NON_ENZYME_ASE_WORDS,
    _SHORT_ENZYME_NOUNS,
    _ENZYME_NOUN_RE_SRC,
    _ROLE_CUE_RES,
    _ANY_ROLE_CUE_RE,
    _ROLE_FAMILY_BY_ROLE,
    UNEVIDENCED_ACTOR_ROLE_REASON_PREFIX,
)

LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"
FAILS = []


def head(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


PAYLOAD = {
    "paper_id": "PMCTEST",
    "entities": {
        "proteins": [{"name": "NDM-1"}, {"name": "MsbA"}, {"name": "Fur"},
                     {"name": "ALAS2"}, {"name": "ferrochelatase"}],
        "compounds": [{"name": "phthalylsulfacetamide"}, {"name": "sulfacetamide"}],
    },
    "processes": {
        "reactions": [{
            "id": "R1",
            "name": "phthalylsulfacetamide decomposition to sulfacetamide",
            "inputs": ["phthalylsulfacetamide"],
            "outputs": ["sulfacetamide"],
            "enzymes": [],
            "modifiers": [{"entity": "NDM-1", "role": "inhibitor",
                           "evidence": "PSA significantly inhibited NDM-1 enzyme activity"}],
            "evidence": "PSA is decomposed in the intestine, resulting in an antibacterial effect",
        }],
        "transports": [{
            "id": "T1", "name": "lipid A flipping",
            "transporters": [], "cargo": [],
        }],
    },
}


def run_op(op, payload=None):
    """Run ONE op through the real production seam. Returns (accepted, reason)."""
    src = deepcopy(payload if payload is not None else PAYLOAD)
    _, report = apply_patch_with_policy(src, [deepcopy(op)])
    rejected = report.get("rejected") or []
    accepted = report.get("accepted") or []
    if rejected:
        return False, str(rejected[0].get("reason", ""))
    if accepted:
        return True, "accepted"
    return None, json.dumps(report)[:400]


def expect(name, got, want):
    ok = got == want
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILS.append(f"{name}: got {got!r} want {want!r}")
    return ok


# ---------------------------------------------------------------- B1  F-146
head("B1 -- the F-146 patch, constructed by REV-107 from C-105.md section 1")

F146_RATIONALE = (
    "add NDM-1 as an enzyme to the decomposition reaction to resolve the "
    "structural inconsistency where an inhibitor is listed without a target enzyme."
)

F146_VARIANTS = [
    ("F146/enzymes/-  bare string value",
     {"op": "add", "path": "/processes/reactions/0/enzymes/-",
      "value": "NDM-1", "confidence": 0.95, "reason": F146_RATIONALE}),
    ("F146/enzymes/-  dict value",
     {"op": "add", "path": "/processes/reactions/0/enzymes/-",
      "value": {"entity": "NDM-1"}, "confidence": 0.95, "reason": F146_RATIONALE}),
    ("F146/modifiers/- role=catalyst",
     {"op": "add", "path": "/processes/reactions/0/modifiers/-",
      "value": {"entity": "NDM-1", "role": "catalyst"}, "confidence": 0.95,
      "reason": F146_RATIONALE}),
    ("F146 evidence field not reason",
     {"op": "add", "path": "/processes/reactions/0/enzymes/-",
      "value": "NDM-1", "confidence": 0.95, "evidence": F146_RATIONALE}),
    ("F146 row-carried evidence",
     {"op": "add", "path": "/processes/reactions/0/enzymes/-",
      "value": {"entity": "NDM-1", "evidence": F146_RATIONALE},
      "confidence": 0.95}),
    ("F146 paraphrase: reduction of activity",
     {"op": "add", "path": "/processes/reactions/0/enzymes/-",
      "value": "NDM-1", "confidence": 0.95,
      "reason": "the reduction of NDM-1 activity by PSA shows NDM-1 acts on the decomposition"}),
    ("F146 paraphrase: blockade",
     {"op": "add", "path": "/processes/reactions/0/enzymes/-",
      "value": "NDM-1", "confidence": 0.95,
      "reason": "the blockade of NDM-1 by PSA mediates the decomposition of PSA"}),
    ("F146 via cofactor role (C-107 1e, new family)",
     {"op": "add", "path": "/processes/reactions/0/modifiers/-",
      "value": {"entity": "NDM-1", "role": "cofactor"}, "confidence": 0.95,
      "reason": "add NDM-1 as a cofactor to resolve the structural inconsistency"}),
    ("F146 via transporter container (C-107 1d, enzyme-noun rule now on transport)",
     {"op": "add", "path": "/processes/transports/0/transporters/-",
      "value": "NDM-1", "confidence": 0.95,
      "reason": "add NDM-1 as a transporter to resolve the structural inconsistency"}),
    ("F146-class: cofactor rationale using 'requires'",
     {"op": "add", "path": "/processes/reactions/0/modifiers/-",
      "value": {"entity": "NDM-1", "role": "cofactor"}, "confidence": 0.95,
      "reason": "the reaction requires a cofactor, so NDM-1 is added to resolve the structural inconsistency"}),
]

for name, op in F146_VARIANTS:
    acc, reason = run_op(op)
    print(f"  {name}\n      accepted={acc}  reason={reason[:150]}")

# The pinned one, asserted:
acc, reason = run_op(F146_VARIANTS[0][1])
expect("B1 PINNED F-146 rejected", acc, False)
expect("B1 PINNED F-146 reason prefix",
       reason.startswith(UNEVIDENCED_ACTOR_ROLE_REASON_PREFIX), True)


# ---------------------------------------------------------------- B2  redox
head("B2 -- 'reduction of' closed WITHOUT breaking redox")

print("  catalysis pattern still contains 'reduces|reducing|reduction of': "
      f"{'reduction of' in _ROLE_CUE_RES['catalysis'].pattern}")
print("  'reduces' present: "
      f"{'reduces' in _ROLE_CUE_RES['catalysis'].pattern}")
print("  'reducing' present: "
      f"{'reducing' in _ROLE_CUE_RES['catalysis'].pattern}")

B2 = [
    # (span, actor, must_license)
    ("the reduction of NDM-1 activity by PSA", "NDM-1", False),
    ("PSA caused a reduction of NDM-1 enzyme activity", "NDM-1", False),
    ("NADH-dependent reduction of the substrate by ferrochelatase", "ferrochelatase", True),
    ("ferrochelatase reduces protoporphyrin IX to heme", "ferrochelatase", True),
    ("nitroreductase catalyses the reduction of the nitro group", "nitroreductase", True),
    ("the reduction of the disulfide bond is mediated by thioredoxin", "thioredoxin", True),
    ("reducing equivalents are transferred by ferredoxin reductase", "ferredoxin reductase", True),
]
for span, actor, want in B2:
    got = _span_licenses_actor(span, actor, "catalysis")
    expect(f"B2 {span[:52]!r} / {actor}", got, want)

print("\n  -- REV-107 additional redox stress (residual hunt, not in the card) --")
for span, actor in [
    ("the enzyme catalyses the reduction of the substrate level in vitro", "enzyme"),
    ("ferrochelatase reduces the cellular level of protoporphyrin", "ferrochelatase"),
    ("NADPH-dependent reduction of flavin is required for enzyme function", "flavin reductase"),
]:
    print(f"     {span[:70]!r} / {actor} -> {_span_licenses_actor(span, actor, 'catalysis')}")


# ------------------------------------------------- B3 eleven near-synonyms
head("B3 -- all eleven near-synonyms, INDIVIDUALLY, in three frames")

NEAR = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
        "sequestration", "depletion", "ablation", "interference", "quenching"]

print("\n  Frame A -- bare Lead frame: 'the <word> of NDM-1 activity by PSA'")
for w in NEAR:
    span = f"the {w} of NDM-1 activity by PSA"
    print(f"     {w:<14} -> {_span_licenses_actor(span, 'NDM-1', 'catalysis')}")

print("\n  Frame B -- with a REAL catalysis cue in the window:")
print("            '<word> of NDM-1 activity is mediated by PSA'")
for w in NEAR:
    span = f"the {w} of NDM-1 activity is mediated by PSA"
    print(f"     {w:<14} -> {_span_licenses_actor(span, 'NDM-1', 'catalysis')}")

print("\n  Frame C -- object BEFORE the stem: 'NDM-1 activity is <word>ed, mediated by PSA'")
FRAME_C = {
    "blockade": "NDM-1 activity showed blockade, and hydrolysis is mediated by NDM-1",
    "impairment": "NDM-1 activity showed impairment, and hydrolysis is mediated by NDM-1",
    "disruption": "NDM-1 activity showed disruption, and hydrolysis is mediated by NDM-1",
    "reduction": "NDM-1 activity showed reduction, and hydrolysis is mediated by NDM-1",
    "loss": "NDM-1 activity showed loss, and hydrolysis is mediated by NDM-1",
    "silencing": "NDM-1 activity showed silencing, and hydrolysis is mediated by NDM-1",
    "sequestration": "NDM-1 activity showed sequestration, and hydrolysis is mediated by NDM-1",
    "depletion": "NDM-1 activity showed depletion, and hydrolysis is mediated by NDM-1",
    "ablation": "NDM-1 activity showed ablation, and hydrolysis is mediated by NDM-1",
    "interference": "NDM-1 activity showed interference, and hydrolysis is mediated by NDM-1",
    "quenching": "NDM-1 activity showed quenching, and hydrolysis is mediated by NDM-1",
}
for w in NEAR:
    print(f"     {w:<14} -> {_span_licenses_actor(FRAME_C[w], 'NDM-1', 'catalysis')}")

print("\n  Frame D -- no activity/level/expression object at all:")
print("            'the <word> of NDM-1 by PSA, and hydrolysis is mediated by NDM-1'")
for w in NEAR:
    span = f"the {w} of NDM-1 by PSA, and hydrolysis is mediated by NDM-1"
    print(f"     {w:<14} -> {_span_licenses_actor(span, 'NDM-1', 'catalysis')}")


# ------------------------------------------------------------ C4 adjudication
head("C4 -- the Lead's word-level probe, adjudicated")
print("  bare word 'reduction' alone as a catalysis cue in the shipped pattern:")
print(f"     _ROLE_CUE_RES['catalysis'].search('reduction') = "
      f"{bool(_ROLE_CUE_RES['catalysis'].search('reduction'))}")
print(f"     _ROLE_CUE_RES['catalysis'].search('reduction of') = "
      f"{bool(_ROLE_CUE_RES['catalysis'].search('reduction of'))}")
print("\n  For each of the eleven, does the BARE frame carry ANY catalysis cue?")
for w in NEAR:
    span = _match_fold(f"the {w} of NDM-1 activity by PSA")
    m = _ROLE_CUE_RES["catalysis"].search(span)
    print(f"     {w:<14} cue={m.group(0) if m else None!r}")


# ------------------------------------------------------ B6 / B7 stoplist
head("B6/B7 -- the closed stoplist and the plural bypass")
print(f"  _NON_ENZYME_ASE_WORDS entries : {len(_NON_ENZYME_ASE_WORDS)}")
print(f"  _SHORT_ENZYME_NOUNS           : {_SHORT_ENZYME_NOUNS}")
print(f"  stoplist                      : {list(_NON_ENZYME_ASE_WORDS)}")

import re as _re
ENZ = _re.compile(_ENZYME_NOUN_RE_SRC)
singles = [w for w in _NON_ENZYME_ASE_WORDS if w + "s" not in _NON_ENZYME_ASE_WORDS]
print(f"\n  entries with NO explicit plural listed: {len(singles)}")
print(f"     {singles}")
print("\n  does each singular-only entry's -s plural BYPASS (match as enzyme noun)?")
bypass = []
for w in singles:
    p = w + "s"
    hit = ENZ.search(p)
    if hit and hit.group(0) == p:
        bypass.append(p)
    print(f"     {p:<14} match={hit.group(0) if hit else None!r}")
print(f"\n  BYPASSING PLURALS: {bypass}")
expect("B7 no singular-only plural bypasses", bypass, [])

print("\n  every entry, singular and plural, must NOT match:")
leaks = []
for w in _NON_ENZYME_ASE_WORDS:
    for form in (w, w + "s", w + "es"):
        hit = ENZ.search(form)
        if hit and hit.group(0) == form:
            leaks.append(form)
print(f"     leaks: {sorted(set(leaks))}")

print("\n  real enzymes must STILL match (preservation, incl. -ases plurals):")
enzymes = ["hydrolase", "hydrolases", "lactamase", "lactamases", "flippase", "flippases",
           "translocase", "translocases", "permease", "permeases", "kinase", "kinases",
           "protease", "proteases", "reductase", "reductases", "lyase", "lyases",
           "dnase", "rnase", "atpase", "atpases", "polymerase", "synthase", "oxidase"]
missed = []
for e in enzymes:
    hit = ENZ.search(e)
    if not (hit and hit.group(0) == e):
        missed.append(e)
print(f"     enzymes NOT matched: {missed}")
expect("B6/B7 no real enzyme lost", missed, [])

print("\n  B6 -- is the stoplist still a CLOSED STOPLIST of English words?")
print(f"     _ENZYME_NOUN_RE_SRC = {_ENZYME_NOUN_RE_SRC[:120]}...")
print(f"     contains a negative lookahead (stoplist) : "
      f"{'(?!' in _ENZYME_NOUN_RE_SRC}")
print(f"     generic [a-z]{{3,}}ases? rule retained    : "
      f"{'[a-z]{3,}ases?' in _ENZYME_NOUN_RE_SRC}")


# ------------------------------------------------------------ 1f mediat
head("1f / B10 -- 'mediat' anchoring")
for span, actor, note in [
    ("ALAS2 mediates the condensation of glycine and succinyl-CoA", "ALAS2", "legit, must license"),
    ("NDM-1 is an intermediate carrier in this pathway", "NDM-1", "intermediate, must NOT license"),
    ("hydrolysis is NDM-1-mediated in the periplasm", "NDM-1", "hyphenated, must license"),
    ("the intermediates accumulate near NDM-1 in this pathway", "NDM-1", "must NOT license"),
]:
    print(f"     [{note}] {span[:60]!r} -> {_span_licenses_actor(span, actor, 'catalysis')}")

print("\n  the cancelling pair, isolated:")
CANCEL = ("suppressor mutations were mapped, and NDM-1 appears as an intermediate "
          "in the discussion of this pathway")
print(f"     {CANCEL[:90]!r}\n       -> {_span_licenses_actor(CANCEL, 'NDM-1', 'catalysis')}")
print(f"     'suppress' still unanchored in inhibition set: "
      f"{bool(_ROLE_CUE_RES['inhibition'].search('suppressor mutations'))}")


# ------------------------------------------------------------ 1d transport
head("1d -- transport enzyme-noun rule")
for span, actor, want in [
    ("MsbA is the flippase that translocates lipid A across the inner membrane", "MsbA", True),
    ("MsbA is the flippase of the inner membrane", "MsbA", True),
    ("MsbA was detected in the lysate", "MsbA", False),
    ("Fur is a transcriptional regulator of iron genes", "Fur", False),
]:
    got = _span_licenses_actor(span, actor, "transport")
    expect(f"1d {span[:50]!r}", got, want)
print("\n  cost of the general rule (author admits it):")
print(f"     'P is a hydrolase' cues TRANSPORT for P: "
      f"{_span_licenses_actor('P4X is a hydrolase', 'P4X', 'transport')}")


# ------------------------------------------------------------ 1e cofactor
head("1e / B8 -- cofactor role")
print(f"  _ROLE_FAMILY_BY_ROLE.get('cofactor') = {_ROLE_FAMILY_BY_ROLE.get('cofactor')!r}")
print(f"  'cofactor' in _ROLE_CUE_RES          = {'cofactor' in _ROLE_CUE_RES}")
for span, actor, want in [
    ("PLP is the cofactor for ALAS2 in this condensation", "PLP", True),
    ("the enzyme requires Zn2+ for activity", "Zn2+", True),
    ("add PLP as a cofactor to resolve the structural inconsistency", "PLP", False),
    ("PLP was detected in the assay", "PLP", False),
]:
    got = _span_licenses_actor(span, actor, "cofactor")
    expect(f"1e {span[:52]!r}", got, want)

print("\n  N1 -- REV-107 residual: does a bare rationale with a cofactor stem self-license?")
for span in [
    "add NDM-1 as a cofactor because the reaction requires a cofactor",
    "NDM-1 is required for the reaction to proceed",
    "the reaction is dependent on NDM-1",
    "the decomposition proceeds in the presence of NDM-1",
    "NDM-1 depends on the payload structure being consistent",
]:
    print(f"     {span[:70]!r} -> {_span_licenses_actor(span, 'NDM-1', 'cofactor')}")

print("\n  N2 -- does the widened _ANY_ROLE_CUE_RE leak to EVERY unmapped role?")
print(f"     _actor_role_family('modifiers', {{'role':'chaperone'}}) = "
      f"{_actor_role_family('modifiers', {'role': 'chaperone'})}")
for span in ["NDM-1 is required for the reaction to proceed",
             "the reaction proceeds in the presence of NDM-1"]:
    print(f"     role=chaperone (family 'other'): {span[:60]!r} -> "
          f"{_span_licenses_actor(span, 'NDM-1', 'other')}")


# ------------------------------------------------------------ 1b passive
head("1b / N4 -- passive-with-agent, actor-anchored")
for span, actor, want in [
    ("phthalylsulfacetamide is converted to sulfacetamide by NDM-1", "NDM-1", True),
    ("phthalylsulfacetamide is converted to sulfacetamide by Q9X, and NDM-1 was also detected",
     "NDM-1", False),
    ("the substrate is converted to product by the purified recombinant enzyme NDM-1",
     "NDM-1", True),
    ("the substrate is converted to product by Q9X; NDM-1 was present", "NDM-1", False),
]:
    got = _span_licenses_actor(span, actor, "catalysis")
    expect(f"1b {span[:58]!r}", got, want)

print("\n  N4 -- is the contra seen when it sits BEFORE the passive verb?")
for span in [
    "PSA inhibited NDM-1 and the substrate is converted to product by NDM-1",
    "the substrate is converted to product by NDM-1, which PSA inhibited",
]:
    print(f"     {span[:74]!r} -> {_span_licenses_actor(span, 'NDM-1', 'catalysis')}")


# ------------------------------------------------------------ C6 claims
head("C6 -- the two new findings the author registered")
print("  (a) transport family bare schema noun self-licenses:")
span_a = "add MsbA as a transporter to resolve the structural inconsistency"
print(f"     {span_a!r}\n       family=transport -> {_span_licenses_actor(span_a, 'MsbA', 'transport')}")
acc, reason = run_op({"op": "add", "path": "/processes/transports/0/transporters/-",
                      "value": "MsbA", "confidence": 0.95, "reason": span_a})
print(f"       through the production seam: accepted={acc} reason={reason[:120]}")
print(f"     'transport' is a bare stem in the transport family: "
      f"{bool(_ROLE_CUE_RES['transport'].search('transporter'))}")
print("\n     control -- the catalysis family does NOT do this:")
span_ctl = "add MsbA as an enzyme to resolve the structural inconsistency"
print(f"       {span_ctl!r} -> {_span_licenses_actor(span_ctl, 'MsbA', 'catalysis')}")

print("\n  (b) [^.] is a no-op because _match_fold strips periods:")
sample = "One sentence here. A second sentence there."
print(f"     _match_fold({sample!r})\n        = {_match_fold(sample)!r}")
print(f"     '.' in folded haystack: {'.' in _match_fold(sample)}")
cross = ("PSA inhibited the target. NDM-1 catalyses the hydrolysis of meropenem")
print(f"     cross-sentence contra reaches across the period: "
      f"{_span_licenses_actor(cross, 'NDM-1', 'catalysis')}  (False => contra crossed the '.')")


# ------------------------------------------------------------ summary
head(f"REV-107 PROBE SUMMARY [{LABEL}]")
print(f"  assertion failures: {len(FAILS)}")
for f in FAILS:
    print(f"    - {f}")
print("REV107_PROBE_DONE")
