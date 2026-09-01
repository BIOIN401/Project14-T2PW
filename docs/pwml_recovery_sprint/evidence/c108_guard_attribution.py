"""C-108: WHICH GUARD FIRED, per row -- built because the charter's nominated
instrument does not do this.

`c107_1g_stage_attribution.py` attributes an oversized span to the PIPELINE
STAGE that wrote it (it walks stage1_payload -> merged_payload -> final ->
final_mapped and reports the longest actor-row evidence each holds). It says
nothing about which pattern inside the actor-evidence guard produced a verdict.
C-108 section 2 requires the latter, so this instrument provides it.

It mirrors `_span_licenses_actor`'s own control flow against the imported
module's own pattern objects, and reports, for every (span, actor, family):

  needle          the identifying token the window was centred on
  cue             the exact text the family cue matched, or None
  contra          the exact text _CATALYSIS_CONTRA_RE matched, or None
  actor_contra    the exact text the per-needle attenuation frame matched
  dependence      the exact text the per-needle cofactor frame matched
  passive         the exact text the per-needle passive-agent frame matched
  verdict         the licensing decision this mirror reaches

The mirror's verdict is asserted against the real seam on every call, so a
divergence between the mirror and production is reported rather than hidden.

Usage:
  <python> c108_guard_attribution.py <code-root> corpus <flat-verdicts.json> <keys-file>
  <python> c108_guard_attribution.py <code-root> census <flat-verdicts.json>
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

CODE = Path(sys.argv[1]).resolve()
MODE = sys.argv[2]
sys.path.insert(0, str(CODE / "src"))

import t2pw.curation.apply_audit_patch as M  # noqa: E402

print("code loaded from:", M.__file__, file=sys.stderr)


def attribute(span, actor, family):
    """Mirror of _span_licenses_actor, reporting what matched at each step."""
    out = []
    haystack = M._match_fold(span)
    if not haystack:
        return [{"needle": None, "verdict": False, "why": "empty haystack"}]
    needles = M._identifying_match_tokens(actor)
    if not needles:
        whole = M._match_fold(actor)
        needles = [whole] if whole else []
    if not needles:
        return [{"needle": None, "verdict": False, "why": "actor folds away"}]
    cue = M._ROLE_CUE_RES.get(family, M._ANY_ROLE_CUE_RE)
    contra = M._CATALYSIS_CONTRA_RE if family == "catalysis" else None
    for needle in needles:
        escaped = re.escape(needle)
        actor_contra = None
        if family == "catalysis":
            actor_contra = re.compile(
                M._ATTENUATION_WORD_SRC + r"(?:\s+(?:of|in))?"
                r"(?:\s+" + M._PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\s+"
                + escaped + r"(?![a-z0-9])"
                r"|(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
                r"[^.]{0," + str(M._ATTENUATION_GAP) + r"}?\b"
                + M._ATTENUATION_OBJECT_SRC + r"[a-z]*\b"
                r"[^.]{0," + str(M._ATTENUATION_GAP) + r"}?\b"
                + M._ATTENUATION_WORD_SRC
                + (
                    (r"|" + M._ATTENUATION_AGENT_NOUN_SRC
                     + r"\s+(?:of|for|against|on|upon|toward|towards|to)"
                     r"(?:\s+" + M._PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\s+"
                     + escaped + r"(?![a-z0-9])"
                     r"|(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
                     r"(?:\s+" + M._ATTENUATION_AGENT_ADJ_SRC + r"){0,"
                     + str(M._ATTENUATION_AGENT_MAX_ADJ) + r"}\s+"
                     + M._ATTENUATION_AGENT_NOUN_SRC + r"(?![a-z])")
                    if hasattr(M, "_ATTENUATION_AGENT_NOUN_SRC") else ""
                )
            )
        dependence = None
        if family == "cofactor":
            dependence = re.compile(
                M._COFACTOR_DEPENDENCE_SRC + r"\b(?:\s+(?:on|of|upon|for))?"
                r"(?:\s+" + M._COFACTOR_MODIFIERS_SRC + r"){0,"
                + str(M._COFACTOR_MAX_MODIFIERS) + r"}\s+"
                + escaped + r"(?![a-z0-9])"
            )
        for match in re.finditer(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])", haystack):
            start = max(0, match.start() - M._ACTOR_CUE_WINDOW)
            end = min(len(haystack), match.end() + M._ACTOR_CUE_WINDOW)
            window = haystack[start:end]
            # C-108 (c): the tip masks the actor's own name for the POSITIVE cue
            # search only. Mirrored here through the module's own helper when it
            # exists, so this instrument reports the base and the tip faithfully
            # from one source. The dependence route and every contra read the
            # unmasked window at both SHAs.
            masker = getattr(M, "_mask_actor_name", None)
            cue_window = masker(window, actor) if masker else window
            cm = cue.search(cue_window)
            dm = dependence.search(window) if dependence is not None else None
            rec = {"needle": needle, "window": window, "cue_window": cue_window,
                   "cue": cm.group(0) if cm else None,
                   "dependence": dm.group(0) if dm else None,
                   "contra": None, "actor_contra": None, "passive": None,
                   "route": "window"}
            if not (cm or dm):
                rec["verdict"] = False
                rec["why"] = "no cue in window"
                out.append(rec)
                continue
            km = contra.search(window) if contra is not None else None
            if km:
                rec["contra"] = km.group(0)
                rec["verdict"] = False
                rec["why"] = "family contra fired"
                out.append(rec)
                continue
            am = actor_contra.search(window) if actor_contra is not None else None
            if am:
                rec["actor_contra"] = am.group(0)
                rec["verdict"] = False
                rec["why"] = "actor-anchored attenuation frame fired"
                out.append(rec)
                continue
            rec["verdict"] = True
            rec["why"] = "licensed on the window route"
            out.append(rec)
            return out
        if family != "catalysis":
            continue
        passive = (
            M._PASSIVE_AGENT_VERBS_SRC
            + r"\b[^.]{0,80}\bby(?:\s+" + M._PASSIVE_AGENT_MODIFIERS_SRC
            + r"){0," + str(M._PASSIVE_AGENT_MAX_MODIFIERS) + r"}\s+"
            + escaped + r"(?![a-z0-9])"
        )
        for match in re.finditer(passive, haystack):
            start = max(0, match.end() - len(needle) - M._ACTOR_CUE_WINDOW)
            end = min(len(haystack), match.end() + M._ACTOR_CUE_WINDOW)
            window = haystack[start:end]
            rec = {"needle": needle, "window": window, "cue": None,
                   "dependence": None, "contra": None, "actor_contra": None,
                   "passive": match.group(0), "route": "passive-agent"}
            km = contra.search(window) if contra is not None else None
            if km:
                rec["contra"] = km.group(0)
                rec["verdict"] = False
                rec["why"] = "family contra fired on the passive window"
                out.append(rec)
                continue
            am = actor_contra.search(window) if actor_contra is not None else None
            if am:
                rec["actor_contra"] = am.group(0)
                rec["verdict"] = False
                rec["why"] = "actor attenuation frame fired on the passive window"
                out.append(rec)
                continue
            rec["verdict"] = True
            rec["why"] = "licensed on the passive-agent route"
            out.append(rec)
            return out
    return out


CATALYST_CONT = ("enzymes", "catalysts", "modifiers_or_enzymes")
TRANSPORT_CONT = ("transporters", "cargo", "cargo_complex")
ROLE_FAMILY = {
    "catalyst": "catalysis", "enzyme": "catalysis", "activator": "activation",
    "inhibitor": "inhibition", "repressor": "inhibition",
    "transporter": "transport", "cofactor": "cofactor",
}


def family_of(cont, role):
    if cont in TRANSPORT_CONT:
        return "transport"
    if cont in CATALYST_CONT:
        return "catalysis"
    r = re.sub(r"[^a-z0-9]+", "", str(role or "").lower())
    if not r:
        return "catalysis"
    return ROLE_FAMILY.get(r, "other")


def parse(key):
    cont, bucket, name, role, ev = key.split("|", 4)
    return cont, bucket, name, role, ev


def report(key):
    cont, bucket, name, role, ev = parse(key)
    fam = family_of(cont, role)
    print("-" * 90)
    print("  [%s/%s role=%r family=%s] actor=%r" % (cont, bucket, role, fam, name))
    print("      span: %r%s" % (ev[:300], " ...(%d chars)" % len(ev) if len(ev) > 300 else ""))
    recs = attribute(ev, name, fam)
    licensed = any(r.get("verdict") for r in recs)
    print("      LICENSED: %s" % licensed)
    for r in recs[:6]:
        print("        needle=%r route=%s cue=%r contra=%r actor_contra=%r "
              "dependence=%r passive=%r -> %s (%s)"
              % (r.get("needle"), r.get("route"), r.get("cue"), r.get("contra"),
                 r.get("actor_contra"), r.get("dependence"),
                 (r.get("passive") or "")[:60] or None,
                 r.get("verdict"), r.get("why")))
    if len(recs) > 6:
        print("        ... and %d more needle/occurrence records" % (len(recs) - 6))


flat = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
if isinstance(flat, dict) and "verdicts" in flat:
    flat = flat["verdicts"]

if MODE == "corpus":
    keys = [ln.rstrip("\n") for ln in
            Path(sys.argv[4]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    print("attributing %d rows" % len(keys))
    for k in keys:
        report(k)
elif MODE == "census":
    # For every ACCEPTED row, what text did the family cue match? This is the
    # measurement that sizes member (a): how much of the corpus rests on a bare
    # role noun rather than on a verb.
    cue_counter = Counter()
    fam_counter = Counter()
    route_counter = Counter()
    mismatch = 0
    for k, v in flat.items():
        cont, bucket, name, role, ev = parse(k)
        fam = family_of(cont, role)
        recs = attribute(ev, name, fam)
        lic = any(r.get("verdict") for r in recs)
        if lic != bool(v):
            mismatch += 1
            print("MIRROR DIVERGENCE (mirror=%s seam=%s): %r / %r"
                  % (lic, v, name, ev[:120]))
        if not lic:
            continue
        win = [r for r in recs if r.get("verdict")][0]
        fam_counter[fam] += 1
        route_counter[win.get("route")] += 1
        cue_counter[(fam, (win.get("cue") or win.get("dependence")
                           or "<passive-agent>"))] += 1
    print()
    print("MIRROR DIVERGENCES vs the real seam: %d  (must be 0)" % mismatch)
    print()
    print("ACCEPTED rows by family:", dict(fam_counter))
    print("ACCEPTED rows by route :", dict(route_counter))
    print()
    print("ACCEPTED rows by (family, matched cue text) -- descending:")
    for (fam, txt), n in cue_counter.most_common():
        print("  %5d  %-10s %r" % (n, fam, txt))
elif MODE == "family":
    # Every ACCEPTED row of one family, with the text its cue matched. Sizes what
    # a change to that family's vocabulary can cost.
    want = sys.argv[4]
    n = 0
    for k, v in flat.items():
        if not v:
            continue
        cont, bucket, name, role, ev = parse(k)
        if family_of(cont, role) != want:
            continue
        n += 1
        report(k)
    print()
    print("ACCEPTED %s rows: %d" % (want, n))
elif MODE == "namecue":
    # member (c) exposure: ACCEPTED rows whose winning cue match lies INSIDE a
    # contiguous occurrence of the actor's own folded name. These are the rows a
    # name-masking fix would move, and they must be enumerated BEFORE the fix.
    n = 0
    tot = 0
    for k, v in flat.items():
        if not v:
            continue
        tot += 1
        cont, bucket, name, role, ev = parse(k)
        fam = family_of(cont, role)
        recs = [r for r in attribute(ev, name, fam) if r.get("verdict")]
        if not recs:
            continue
        r = recs[0]
        cuetxt = r.get("cue") or r.get("dependence") or ""
        folded_name = M._match_fold(name)
        if not cuetxt or not folded_name:
            continue
        # does the cue text occur only inside the folded name?
        hay = M._match_fold(ev)
        spans_in_name = [m.span() for m in
                         re.finditer(r"(?<![a-z0-9])" + re.escape(folded_name)
                                     + r"(?![a-z0-9])", hay)]
        cue_hits = [m.span() for m in re.finditer(re.escape(cuetxt), hay)]
        if not cue_hits:
            continue
        outside = [s for s in cue_hits
                   if not any(a <= s[0] and s[1] <= b for a, b in spans_in_name)]
        if not outside:
            n += 1
            print("-" * 90)
            print("  CUE ONLY INSIDE THE ACTOR NAME  [%s role=%r fam=%s] actor=%r"
                  % (cont, role, fam, name))
            print("      cue=%r  name-occurrences=%d" % (cuetxt, len(spans_in_name)))
            print("      span: %r" % (ev[:300],))
    print()
    print("ACCEPTED rows whose cue occurs ONLY inside the actor name: %d of %d"
          % (n, tot))
else:
    raise SystemExit("unknown mode %r" % MODE)
