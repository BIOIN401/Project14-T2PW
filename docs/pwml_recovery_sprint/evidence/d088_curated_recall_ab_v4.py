"""D-088 step 6, FOURTH PASS — an undetailed branch is tested by its MEMBERS, not its NAME.

VERSIONS 1, 2 AND 3 ARE PRESERVED beside this one (G11 `ORCH-719/06`, `/07`, `/08`).
**All four are the record.** The iteration is deliberate and auditable, because a rule
adjusted until a number comes out right is what D-088 clause 10 forbids, and the only
defence against doing it accidentally is to leave every pass where a reviewer can see it.

THE SAME BUG TWICE, WHICH IS WHY THE APPROACH CHANGES RATHER THAN THE THRESHOLD
-------------------------------------------------------------------------------
v1 scored `PMC12782028`'s undetailed upstream arm COVERED on the token `cholesterol`.
v3, after excluding the pathway's own tokens, scored it COVERED on the token `lanosterol`
-- taken from the branch's own name, *"the upstream, **pre-lanosterol** segment"*.

**A named-but-undetailed branch cannot be tested by its name.** These names are relative
descriptors: they locate the branch against the chemistry beside it, so they necessarily
contain that chemistry's words. Every further token exclusion is a patch on an approach
that cannot work.

WHAT REPLACES IT
----------------
The curators supplied `member_entities` for every undetailed chemical branch -- the concrete
enzymes and metabolites **the paper itself assigns to that branch** -- with a
`member_entities_source` saying where each came from, and an explicit instruction that an
EMPTY list is a correct answer for a branch the paper names but never populates. They were
told, again, not to import members from their own knowledge, and reported refusing exactly
that: iron for the ferrochelatase step, succinyl-CoA for the ALAS2 step, and HMG-CoA /
mevalonate / IPP / FPP / squalene for the upstream sterol arm, none of which their papers
name as free compounds.

So the test becomes what "is this branch present?" actually means: **are the branch's own
members in the extraction?**

THE BOUNDARY-ENTITY RULE, AND WHY IT IS NOT A KNOB
---------------------------------------------------
One member of `PMC12782028` S5 is `2,3-oxidosqualene`, which is **also the substrate of S1,
a stage that IS covered.** The curator flagged this shape unprompted, as the "neighbour"
hazard its instructions warned about.

**A branch is not evidenced by an entity it shares with a stage already counted.** If it
were, every branch adjoining a recovered stage would score covered through the metabolite
they have in common -- which is the identical failure as matching on a shared name token,
arrived at from the other side. So members appearing in an already-covered stage of the
same paper are excluded before the test, and coverage requires a member that belongs to
**this** branch alone.

This is the same logic already used for detailed reactions, where a shared cofactor cannot
carry a reaction and the rule demands an enzyme, or a substrate AND a product.

WHAT WOULD FALSIFY THIS PASS
-----------------------------
If `PMC12782028` cleared, the correction removed the measurement and the line of work is a
reject under `HANDOFF.md` § 5.2a step 6. Its upstream arm must be found ABSENT on the
strength of its own nine unrecovered enzymes -- `ACAT2, HMGCS1, HMGCR, MVK, MVD, IDI1,
FDPS, FDFT1, SQLE` -- and if a draw ever recovers them, that draw SHOULD clear, and the
rule reporting so would be right rather than broken.

The v3 header note follows.

---

D-088 step 6, THIRD PASS — regulatory stages are WARNINGS, not recall failures (clause 2).

VERSIONS 1 AND 2 ARE PRESERVED beside this one (`..._ab.py`/`.log`, G11 `ORCH-719/06`;
`..._ab_v2.py`/`.log`, G11 `ORCH-719/07`). **All three results are the record. Read the note
below before quoting any of them.**

WHAT v2 EXPOSED
---------------
v2 fixed `PMC12782028` (HOLDS on all 7 draws) and broke `PMC12096016` (VIOLATED on all 8).
The whole of that break was ONE subprocess: *"feedback regulation of enterobactin
biosynthesis by 2,3-DHB"* -- a stage that groups **no chemical transformation at all.**
`PMC12782028`'s remaining miss beside its real one was the same shape: *"SREBP-mediated
transcriptional regulation."*

**Scoring a regulatory module as a reaction-recall failure is wrong, and D-088 already says
so.** Clause 2: missing *"regulators"* should *"normally produce completeness WARNINGS or
secondary-score deductions -- not automatically remove `release_ready`."* Clause 5's hard
case is *"missing an entire named pathway BRANCH or subprocess"*, and a transcription factor
is not a branch of the chemistry.

THE FIX IS A CURATED FACT, NOT A KNOB
--------------------------------------
The curation schema had **no field** for chemistry-versus-regulation. That was a gap in the
brief, not in the curation -- one curator flagged the distinction in its report and had
nowhere to put it. So the field was added: every one of the 35 subprocesses now carries
`kind` in {`chemical_transformation`, `regulatory`, `other`} with a `kind_rationale`,
**assigned by the curators from the papers and blind to any effect on this measurement.**
Measured distribution: **29 chemical, 6 regulatory.**

This probe then excludes `regulatory` stages from the HARD requirement and **reports them
separately**, which is clauses 2 and 8 applied rather than a rule tuned until the numbers
came out. The two `PMC12782028` stages that share a shape in the old schema were judged
independently and **split**: the upstream mevalonate/squalene arm is
`chemical_transformation`, SREBP is `regulatory`.

WHAT WOULD STILL FALSIFY THIS
------------------------------
If `PMC12782028` now cleared, the correction would have removed the measurement and the
whole line of work would be a reject under `HANDOFF.md` § 5.2a step 6. Its hard miss must
survive on its own chemistry -- the absent upstream arm and, on most draws, the C-4
demethylation -- with no help from the regulatory stage that was just excluded.

The v2 header note follows.

---

D-088 step 6, SECOND PASS — the curated set, with the undetailed-subprocess rule corrected.

VERSION 1 IS PRESERVED at `d088_curated_recall_ab.py` / `.log`, G11 `ORCH-719/06`, per the
sprint rule that a failed measurement is kept beside its correction. **Read this note before
reading either result.**

WHAT V1 GOT WRONG, AND HOW IT SHOWED
------------------------------------
V1 reported `PMC12782028/strict` as VIOLATED on 1 of 7 archived draws:
`runs_verify/2026-08-21_2239` came back `rxn=4/4 sub=6/6`, i.e. complete.

Half of that is real. **That draw genuinely DID extract MSMO1** -- a fourth `core_accepted`
reaction, `["44dimethylcholesta891424trien3ol", "cholesterol", "methylsterol monooxygenase
1", "msmo1 reaction"]` -- so `rxn=4/4` is a correct result about a better draw, not an
artifact.

**`sub=6/6` is false.** That leg's unmatched anchors are still
`SQLE, FDFT1, HMGCR, HMGCS1` -- the mevalonate/squalene arm is absent exactly as it is in
every other draw. The two UNDETAILED subprocesses were scored covered by v1's token rule,
and they were covered on the token **`cholesterol`**, which appears in the `core_terms` of
two admitted reactions because every reaction in cholesterol biosynthesis mentions
cholesterol.

THE CORRECTION, AND WHY IT IS NOT TUNING
----------------------------------------
An undetailed subprocess is now tested only on tokens that are **distinctive to it** --
tokens of its own name minus the tokens of the paper's `requested_pathway`.

**This fix is justified independently of which way it moves the number, and would have been
made either way.** A subprocess named "the upstream segment of <pathway>" being scored
covered because "<pathway>" appears in a process's terms is wrong on its face: every
process in a pathway names that pathway, so the test as written could never fail for any
subprocess whose name contains it. The exclusion set is **derived from the curated
`requested_pathway` field**, not hand-picked, which is what keeps it out of reach of
tuning.

It cannot touch `PMC12096016`: all four of its curated subprocesses carry `reaction_ids`
and are matched through their reactions, so this rule never runs on them.

The original docstring follows.

---

D-088 step 6 — does the CURATED expectation set give the required discrimination?

Read-only A/B over every committed `coverage_summary.json` under **both `runs/` and
`runs_verify/`**, both roots named. Nothing is run, re-run, re-scored or mutated.

WHAT THIS IS THE THIRD ATTEMPT AT, AND WHY THE FIRST TWO FAILED
---------------------------------------------------------------
`HANDOFF.md` § 5.2a step 6 requires proving that a correction makes `PMC12096016` lose
**only** the false entity-anchor cap while `PMC12782028` **remains** a reaction-recall
failure. **A change that clears both has removed the measurement rather than improved
recall and is a reject.**

  * **Candidate A** — production's existing reaction-level thresholds — clears BOTH.
    `PMC12782028/strict` sits at coverage 0.538 against a 0.5 minimum.
  * **Candidate B** — Stage-0's own `main_subprocesses` — gives the discrimination
    perfectly on the T-108 tree and is disqualified by **F-168**: across 83 archived legs,
    **0 of 14** paper/mode pairs named a stable subprocess set, and on
    `runs_verify/2026-08-21_2239` it releases `PMC12782028/strict` because that draw never
    named the arm the pipeline was missing.

This probe tests the **curated** expectation set — `docs/pwml_recovery_sprint/curation/`,
41 core reactions and 35 major subprocesses over the ten papers, every entry carrying a
verbatim quote verified against the paper's full text by
`d088_curation_validator.py` (G11 `ORCH-719/05`: 174 quotes, 0 unverifiable, 0 schema
warnings).

**The property that matters is that it is FIXED.** It does not vary with the draw being
judged, which is precisely what F-168 established the replacement must be.

THE CURATION WAS BLIND, WHICH IS WHAT MAKES THIS A TEST
--------------------------------------------------------
The curators were told to curate the biology in the papers and were **not** told which
papers pass or fail, what any gate does with their output, or that two named legs must
land on opposite sides. `CURATION-BRIEF-D088.md` § 2 says so in terms and explains why. So
the discrimination measured below was **not** built into the input by whoever wrote it. If
it holds, it holds because the papers differ; if it fails, that is a real result about the
proposal and not a curation bug.

**A curation shaped by the answer it produces would launder a policy preference as
biology.** That is the failure this design exists to avoid, and it is the reason this probe
is worth running at all.

THE MATCHING RULE, AND WHY IT IS NOT THE PHRASE MATCH THAT CAME BEFORE
----------------------------------------------------------------------
Candidate B matched Stage-0 subprocess PHRASES against process `core_terms` by shared
content tokens, which needed a stoplist -- a knob. The curated set carries something
better: each reaction names its **enzymes, substrates and products** as discrete entities.
So a curated reaction is matched on its NAMED PARTICIPANTS, not on prose:

    a curated reaction is COVERED when some `core_accepted` process's `core_terms`
    matches its enzyme, OR matches both a substrate and a product.

The enzyme arm alone is sufficient because a named catalyst identifies a step; the
substrate-AND-product arm catches a correctly extracted reaction whose enzyme the paper
left unnamed. **Requiring both a substrate and a product, rather than either, is what stops
a shared cofactor from carrying an unrelated reaction.** No stoplist is involved.

An undetailed subprocess -- one the paper NAMES without giving its chemistry, D-088 clause
5's "entire named branch" -- has no reactions to match, so it is tested on its own name
tokens against the same `core_terms`. That is the one place a token rule survives, and it
is reported separately so a reader can see how much weight it carries.

Usage:  d088_curated_recall_ab.py <repo>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

T108_TREE = "runs_verify/2026-09-01_1612"
MUST_CLEAR = "PMC12096016"
MUST_REMAIN = "PMC12782028"
CORE_ACCEPTED = "core_accepted"

# Only English structure. Deliberately minimal: the curated participants are concrete
# entity names, so unlike candidate B this rule does not lean on a stoplist to work.
STOP = {"a", "an", "and", "the", "of", "to", "in", "on", "by", "for", "from", "with",
        "into", "onto", "via", "at", "or", "its", "their", "pathway", "segment"}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def toks(s):
    return {t for t in re.split(r"[^a-z0-9]+", str(s).lower())
            if len(t) >= 4 and t not in STOP}


def matches(term, term_set):
    """strict_quarantine._term_matches, reproduced: substring either way, normalised."""
    n = norm(term)
    if not n:
        return False
    return any(n == o or n in o or o in n for o in term_set)


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("usage: d088_curated_recall_ab.py <repo>")
        return 2
    repo = os.path.abspath(sys.argv[1])
    cur_dir = os.path.join(repo, "docs", "pwml_recovery_sprint", "curation")

    curated = {}
    for f in sorted(os.listdir(cur_dir)) if os.path.isdir(cur_dir) else []:
        if f.startswith("expected_core_") and f.endswith(".json"):
            doc = load(os.path.join(cur_dir, f))
            if isinstance(doc, dict) and doc.get("paper_id"):
                curated[doc["paper_id"]] = doc

    legs = []
    for root_name in ("runs", "runs_verify"):
        root = os.path.join(repo, root_name)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "coverage_summary.json" in files:
                legs.append(os.path.join(dirpath, "coverage_summary.json"))
    legs.sort()

    print("D-088 STEP 6 (v4) — UNDETAILED BRANCHES TESTED BY MEMBERS, NOT BY NAME")
    print("=" * 96)
    print(f"repo                       : {repo}")
    print(f"roots scanned              : runs/ AND runs_verify/, both named")
    print(f"coverage_summary.json      : {len(legs)}")
    print(f"curated papers             : {len(curated)}")
    print(f"curated core reactions     : {sum(len(d.get('expected_core_reactions') or []) for d in curated.values())}")
    print(f"curated major subprocesses : {sum(len(d.get('expected_major_subprocesses') or []) for d in curated.values())}")
    print("the curation was produced BLIND to the cap and to which legs pass")
    print()

    tot = Counter()
    rows = []
    unannotated = []

    for path in legs:
        cov = load(path)
        if not isinstance(cov, dict) or not cov.get("requested_core_declared"):
            continue
        leg_dir = os.path.dirname(path)
        rel = os.path.relpath(path, repo).replace("\\", "/")
        parts = rel.split("/")
        run_id = "/".join(parts[:2])
        paper = parts[-3]
        mode = parts[-2]
        doc = curated.get(paper)
        if doc is None:
            continue

        rep = load(os.path.join(leg_dir, "quarantine_report.json"))
        if not isinstance(rep, dict) or not isinstance(rep.get("admissions"), list):
            tot["legs_without_report"] += 1
            continue
        core_sets = []
        for row in rep["admissions"]:
            if isinstance(row, dict) and row.get("state") == CORE_ACCEPTED:
                s = {norm(t) for t in (row.get("core_terms") or [])}
                lbl = row.get("label")
                if isinstance(lbl, str):
                    s.add(norm(lbl))
                core_sets.append(s)
        tot["legs_examined"] += 1

        # ---- curated core reactions -------------------------------------
        rxns = doc.get("expected_core_reactions") or []
        covered_r, missing_r = [], []
        for r in rxns:
            if not isinstance(r, dict):
                continue
            enz = [e for e in (r.get("enzymes") or [])]
            subs = [e for e in (r.get("substrates") or [])]
            prods = [e for e in (r.get("products") or [])]
            hit = False
            for cs in core_sets:
                if any(matches(e, cs) for e in enz):
                    hit = True
                    break
                if any(matches(s, cs) for s in subs) and any(matches(p, cs) for p in prods):
                    hit = True
                    break
            (covered_r if hit else missing_r).append(r.get("id"))

        # ---- curated major subprocesses ---------------------------------
        pathway_toks = toks(doc.get("requested_pathway"))
        # Every entity named by a curated reaction that was COVERED. These are the
        # boundary entities an undetailed branch may not be credited for.
        covered_entity_norms = set()
        for r in rxns:
            if isinstance(r, dict) and r.get("id") in covered_r:
                for fld in ("enzymes", "substrates", "products"):
                    for e in (r.get(fld) or []):
                        covered_entity_norms.add(norm(e))
        subs_all = doc.get("expected_major_subprocesses") or []
        # D-088 clause 2: a REGULATORY stage groups no chemical transformation, so its
        # absence is a completeness WARNING and never a reaction-recall failure. Clause
        # 5's hard case is an entire named pathway BRANCH; a transcription factor is not
        # one. Reported separately below -- clause 8 forbids fusing the categories.
        subs_l = [s for s in subs_all
                  if isinstance(s, dict) and s.get("kind") != "regulatory"]
        subs_reg = [s for s in subs_all
                    if isinstance(s, dict) and s.get("kind") == "regulatory"]
        covered_s, missing_s, missing_named = [], [], []
        missing_reg = []
        for s in subs_reg:
            rids = [x for x in (s.get("reaction_ids") or [])]
            hit = any(x in covered_r for x in rids) if rids else False
            if not hit:
                missing_reg.append(s.get("name"))
                tot["regulatory_stages_missing_WARNING_ONLY"] += 1
        tot["regulatory_stages_examined"] += len(subs_reg)
        for s in subs_l:
            if not isinstance(s, dict):
                continue
            rids = [x for x in (s.get("reaction_ids") or [])]
            if rids:
                hit = any(x in covered_r for x in rids)
            else:
                # UNDETAILED branch: tested by its curated MEMBERS, never by its name.
                # See the header note -- name matching failed twice, on 'cholesterol'
                # and then on 'lanosterol' taken from the branch's own name.
                # ABSENT KEY is not an empty list. A branch nobody has annotated yet
                # must never be scored as a branch the paper failed to populate --
                # that would manufacture an uncovered result out of missing metadata.
                if "member_entities" not in s:
                    tot["undetailed_branch_NOT_ANNOTATED"] += 1
                    unannotated.append(f"{paper} {s.get('id')}")
                    continue
                members = [m for m in (s.get("member_entities") or [])]
                # Drop members shared with a stage already counted: a branch is not
                # evidenced by an entity that belongs equally to something else that
                # was recovered. 2,3-oxidosqualene is S5's product AND S1's substrate.
                own = [m for m in members if norm(m) not in covered_entity_norms]
                if not members:
                    # The paper names the branch and never populates it. There is
                    # nothing to test, so this is RECORDED and never counted covered
                    # -- silence is not evidence of recovery.
                    hit = False
                    tot["undetailed_branch_with_no_members"] += 1
                elif not own:
                    hit = False
                    tot["undetailed_branch_only_boundary_members"] += 1
                else:
                    hit = any(any(matches(m, cs) for cs in core_sets) for m in own)
            if hit:
                covered_s.append(s.get("id"))
            else:
                missing_s.append(s.get("id"))
                missing_named.append(s.get("name"))
                if s.get("detailed_in_paper") is False:
                    tot["undetailed_subprocess_missing"] += 1

        old_capped = bool(cov.get("unmatched_terms"))
        new_capped = bool(missing_s)          # THE PROPOSED HARD RULE
        tot["legs_old_capped"] += int(old_capped)
        tot["legs_new_capped"] += int(new_capped)
        tot["curated_reactions_examined"] += len(rxns)
        tot["curated_reactions_missing"] += len(missing_r)
        tot["curated_subprocesses_examined"] += len(subs_l)
        tot["curated_subprocesses_missing"] += len(missing_s)

        rows.append({
            "run": run_id, "paper": paper, "mode": mode,
            "old": old_capped, "new": new_capped,
            "rxn": f"{len(covered_r)}/{len(rxns)}",
            "sub": f"{len(covered_s)}/{len(subs_l)}",
            "missing_named": missing_named,
            "missing_reg": missing_reg,
            "cov": cov.get("coverage_ratio"),
        })

    print("=" * 96)
    print("CORPUS TOTALS")
    print("=" * 96)
    for k in ("legs_examined", "legs_without_report", "legs_old_capped", "legs_new_capped",
              "curated_reactions_examined", "curated_reactions_missing",
              "curated_subprocesses_examined", "curated_subprocesses_missing",
              "undetailed_subprocess_missing",
              "regulatory_stages_examined", "regulatory_stages_missing_WARNING_ONLY",
              "undetailed_branch_with_no_members",
              "undetailed_branch_only_boundary_members",
              "undetailed_branch_NOT_ANNOTATED"):
        print(f"  {k:34s}: {tot[k]}")
    if unannotated:
        print()
        print("  *** UNDETAILED BRANCHES WITH NO member_entities KEY -- these were SKIPPED,")
        print("  *** not counted uncovered. Any verdict below is provisional until annotated:")
        for u in sorted(set(unannotated)):
            print(f"        {u}")
        print()
    flips = [r for r in rows if r["old"] and not r["new"]]
    holds = [r for r in rows if r["old"] and r["new"]]
    newly = [r for r in rows if not r["old"] and r["new"]]
    print()
    print(f"  capped OLD, released NEW : {len(flips)}")
    print(f"  capped OLD, still capped : {len(holds)}")
    print(f"  capped NEW, not OLD      : {len(newly)}   <- the rule is STRICTER here")
    print()

    print("=" * 96)
    print("THE TWO REQUIRED CONSEQUENCES — every archived strict leg of each paper")
    print("=" * 96)
    verdict = {}
    for paper, want_capped in ((MUST_CLEAR, False), (MUST_REMAIN, True)):
        hits = [r for r in rows if r["paper"] == paper and r["mode"] == "strict"]
        print()
        print(f"--- {paper}/strict  (want new_capped={want_capped}) ---")
        bad = []
        for r in sorted(hits, key=lambda r: r["run"]):
            mark = "  <== T-108" if r["run"] == T108_TREE else ""
            ok = (r["new"] == want_capped)
            if not ok:
                bad.append(r)
            print(f"  {'ok ' if ok else 'BAD'} {r['run']:28s} old={str(r['old']):5s} "
                  f"new={str(r['new']):5s} rxn={r['rxn']:6s} sub={r['sub']:6s}{mark}")
            if r["missing_named"]:
                print(f"        HARD missing (chemical): {r['missing_named']}")
            if r["missing_reg"]:
                print(f"        warning only (regulatory): {r['missing_reg']}")
        verdict[paper] = ("HOLDS" if not bad else f"VIOLATED on {len(bad)}/{len(hits)}")

    t108 = {r["paper"]: r for r in rows
            if r["run"] == T108_TREE and r["mode"] == "strict"
            and r["paper"] in (MUST_CLEAR, MUST_REMAIN)}

    print()
    print("=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  ACROSS EVERY ARCHIVED DRAW  {MUST_CLEAR}: {verdict.get(MUST_CLEAR)}")
    print(f"  ACROSS EVERY ARCHIVED DRAW  {MUST_REMAIN}: {verdict.get(MUST_REMAIN)}")
    print()
    a = t108.get(MUST_CLEAR)
    b = t108.get(MUST_REMAIN)
    if a and b:
        print(f"  ON THE T-108 TREE ITSELF:")
        print(f"    {MUST_CLEAR}/strict  new_capped={a['new']}  (want False)  reactions {a['rxn']}  subprocesses {a['sub']}")
        print(f"    {MUST_REMAIN}/strict  new_capped={b['new']}  (want True )  reactions {b['rxn']}  subprocesses {b['sub']}")
        both = (a["new"] is False and b["new"] is True)
        print()
        print(f"    BOTH REQUIRED CONSEQUENCES HOLD SIMULTANEOUSLY: {both}")
        if not both:
            print("    *** REJECT. Step 6 is not satisfied and this input must not be adopted.")
    print()
    print("  Unlike candidate B, this input is FIXED: it does not vary with the draw being")
    print("  judged. The across-every-draw rows above are the check F-168 was registered")
    print("  for, and they are the ones that matter -- a rule that holds only on the tree it")
    print("  was designed against is the rule F-168 already disqualified once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
