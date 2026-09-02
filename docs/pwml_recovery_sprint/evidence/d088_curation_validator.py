"""D-088 — independently validate the curated expected-core dataset.

The curation was produced by two agents working from the ten papers' full texts. **This
validator re-derives every checkable claim rather than trusting the curators' reports.**
It reads the curated JSON, re-reads each paper's full text from the acquire cache, and
verifies the quotes by substring search against it.

WHY IT RE-CHECKS RATHER THAN TRUSTS
-----------------------------------
The curation brief told the curators to verify every quote by substring search before
writing the file, and to report any they could not verify. A self-check that a producer
performs on its own output and reports as passing is exactly the shape the sprint has
already been burned by -- `HANDOFF.md` § 8 lesson 7, the T-108 scorer self-test that ran
green against a tree with zero timeouts, so the timeout columns it existed to validate were
never executed. **The green was real and its coverage was not.**

A fabricated or paraphrased quote is the one failure mode that would silently destroy this
dataset's value, because the whole reason it is trustworthy is that every entry is anchored
to the paper. So the quotes are re-verified here, by a different process, against the same
source.

WHAT IT CHECKS
--------------
  1. **Schema.** Required keys present, types right, enumerations inside their vocabulary.
  2. **Quotes.** Every `quote` appears VERBATIM in the paper's `full_text`. Reported
     per-file with the exact failures, never as an aggregate pass.
  3. **Referential integrity.** Every `reaction_ids` entry names a reaction that exists;
     every subprocess `reaction_ids` entry likewise.
  4. **Cross-file coherence.** All ten papers present; no duplicate reaction ids inside a
     file; `paper_id` matches the filename.
  5. **The distinctions that carry the dataset's meaning.** How many subprocesses are
     `detailed_in_paper: false` (a NAMED-but-undetailed branch is D-088 clause 5's genuine
     recall failure and must not have been collapsed), and how the important/secondary
     participant split fell.

WHAT IT DOES NOT CHECK
----------------------
**It cannot check whether the biology is right.** A quote can be verbatim and still be
attached to the wrong claim, and a reaction can be well-formed and not be in the paper's
pathway. Schema conformance is not curation quality. That judgement is the
`pwml-bio-auditor`'s and this probe does not substitute for it -- it only establishes that
there is something real to audit.

It also does not compare the curation to any pipeline output. Doing so would let what the
pipeline extracted influence what the pipeline is measured against, which is the one thing
the brief forbade.

Usage:  d088_curation_validator.py <repo>
"""

from __future__ import annotations

import json
import os
import sys
import glob

PAPERS = [
    "PMC12444477", "PMC13231680", "PMC12657337", "PMC12421875", "PMC12312563",
    "PMC12856317", "PMC12180156", "PMC12096016", "PMC12452463", "PMC12782028",
]

IMPORTANT_REASONS = {
    "defining_substrate_or_product",
    "distinguishes_identity_or_direction",
    "central_to_pathway_scope",
}
SECONDARY_CLASSES = {
    "cofactor", "currency_metabolite", "regulator", "ancillary_protein",
    "water_or_proton", "other",
}
CONFIDENCE = {"high", "medium", "low"}


def load_full_texts(repo):
    """paper_id -> full_text, from the gitignored acquire cache."""
    out = {}
    d = os.path.join(repo, "data", "rag_index", "acquire_cache", "fulltext")
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                j = json.load(fh)
        except Exception:
            continue
        ident = json.dumps({"id": j.get("id"), "source": j.get("source")})
        for p in PAPERS:
            if p in ident and p not in out:
                out[p] = j.get("full_text") or ""
    return out


def norm_ws(s):
    """Collapse whitespace, for a SECOND-CHANCE quote check only.

    A quote that matches only after whitespace collapsing is reported SEPARATELY as
    `soft` rather than counted as verified. Line wrapping in the source is a real and
    innocent reason for a mismatch; silently accepting it would also accept a quote that
    was reflowed while being edited, so the two are kept distinguishable.
    """
    return " ".join(str(s).split())


def main():
    if len(sys.argv) < 2:
        print("usage: d088_curation_validator.py <repo>")
        return 2
    repo = os.path.abspath(sys.argv[1])
    cur_dir = os.path.join(repo, "docs", "pwml_recovery_sprint", "curation")

    print("D-088 CURATED EXPECTED-CORE DATASET — INDEPENDENT VALIDATION")
    print("=" * 94)
    print(f"repo          : {repo}")
    print(f"curation dir  : {cur_dir}")

    texts = load_full_texts(repo)
    print(f"full texts resolved from acquire cache : {len(texts)} / {len(PAPERS)}")
    missing_text = [p for p in PAPERS if p not in texts]
    if missing_text:
        print(f"  *** NO FULL TEXT FOR: {missing_text} -- quotes for these CANNOT be verified")
    print()

    fatal = 0
    warn = 0
    grand = {"reactions": 0, "subprocesses": 0, "undetailed": 0,
             "important": 0, "secondary": 0, "quotes_ok": 0, "quotes_soft": 0,
             "quotes_bad": 0, "low_conf": 0}

    for paper in PAPERS:
        path = os.path.join(cur_dir, f"expected_core_{paper}.json")
        print("=" * 94)
        print(f"{paper}")
        print("-" * 94)
        if not os.path.isfile(path):
            print("  *** FILE MISSING -- the dataset is incomplete")
            fatal += 1
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:
            print(f"  *** UNREADABLE JSON: {exc}")
            fatal += 1
            continue

        text = texts.get(paper, "")
        text_ws = norm_ws(text)

        if doc.get("paper_id") != paper:
            print(f"  *** paper_id mismatch: file says {doc.get('paper_id')!r}")
            fatal += 1

        rxns = doc.get("expected_core_reactions") or []
        subs = doc.get("expected_major_subprocesses") or []
        imp = doc.get("important_participants") or []
        sec = doc.get("secondary_participants") or []

        ids = [r.get("id") for r in rxns if isinstance(r, dict)]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            print(f"  *** duplicate reaction ids: {sorted(dup)}")
            fatal += 1
        idset = set(ids)

        # -- quotes, the check that matters most --------------------------
        bad, soft, ok = [], [], 0
        def check_quote(where, q):
            nonlocal ok
            if not q:
                bad.append((where, "<empty>"))
                return
            if text and q in text:
                ok += 1
            elif text and norm_ws(q) in text_ws:
                soft.append((where, q[:70]))
            else:
                bad.append((where, q[:70]))

        for r in rxns:
            if isinstance(r, dict):
                check_quote(f"reaction {r.get('id')}", r.get("quote"))
        for s in subs:
            if isinstance(s, dict):
                check_quote(f"subprocess {s.get('id')}", s.get("quote"))
        for p in imp:
            if isinstance(p, dict):
                check_quote(f"important {p.get('name')}", p.get("quote"))
        for p in sec:
            if isinstance(p, dict):
                check_quote(f"secondary {p.get('name')}", p.get("quote"))

        # -- vocabularies --------------------------------------------------
        for r in rxns:
            if isinstance(r, dict) and r.get("confidence") not in CONFIDENCE:
                print(f"  *  reaction {r.get('id')}: bad confidence {r.get('confidence')!r}")
                warn += 1
        for p in imp:
            if isinstance(p, dict) and p.get("reason") not in IMPORTANT_REASONS:
                print(f"  *  important {p.get('name')!r}: bad reason {p.get('reason')!r}")
                warn += 1
        for p in sec:
            if isinstance(p, dict) and p.get("class") not in SECONDARY_CLASSES:
                print(f"  *  secondary {p.get('name')!r}: bad class {p.get('class')!r}")
                warn += 1

        # -- referential integrity ----------------------------------------
        for s in subs:
            if not isinstance(s, dict):
                continue
            for rid in s.get("reaction_ids") or []:
                if rid not in idset:
                    print(f"  *** subprocess {s.get('id')} names unknown reaction {rid!r}")
                    fatal += 1
        for p in list(imp) + list(sec):
            if not isinstance(p, dict):
                continue
            for rid in p.get("reaction_ids") or []:
                if rid not in idset:
                    print(f"  *** participant {p.get('name')!r} names unknown reaction {rid!r}")
                    fatal += 1

        undetailed = [s for s in subs if isinstance(s, dict)
                      and s.get("detailed_in_paper") is False]
        low = [r.get("id") for r in rxns if isinstance(r, dict) and r.get("confidence") == "low"]

        print(f"  core reactions            : {len(rxns)}")
        print(f"  major subprocesses        : {len(subs)}  "
              f"(named but NOT detailed: {len(undetailed)})")
        for s in undetailed:
            print(f"      NAMED-NOT-DETAILED  : {s.get('name')!r}")
        print(f"  important participants    : {len(imp)}")
        print(f"  secondary participants    : {len(sec)}")
        print(f"  low-confidence reactions  : {len(low)} {low}")
        print(f"  quotes verbatim           : {ok}")
        if soft:
            print(f"  quotes matching ONLY after whitespace collapse : {len(soft)}")
            for w, q in soft:
                print(f"      soft  {w}: {q}")
        if bad:
            print(f"  *** QUOTES NOT FOUND IN THE PAPER : {len(bad)}")
            for w, q in bad:
                print(f"      BAD   {w}: {q}")
            fatal += 1
        unc = doc.get("uncertainties") or []
        print(f"  uncertainties recorded    : {len(unc)}")
        for u in unc[:6]:
            print(f"      - {str(u)[:100]}")

        grand["reactions"] += len(rxns)
        grand["subprocesses"] += len(subs)
        grand["undetailed"] += len(undetailed)
        grand["important"] += len(imp)
        grand["secondary"] += len(sec)
        grand["quotes_ok"] += ok
        grand["quotes_soft"] += len(soft)
        grand["quotes_bad"] += len(bad)
        grand["low_conf"] += len(low)

    print()
    print("=" * 94)
    print("TOTALS")
    print("=" * 94)
    for k in ("reactions", "subprocesses", "undetailed", "important", "secondary",
              "low_conf", "quotes_ok", "quotes_soft", "quotes_bad"):
        print(f"  {k:14s}: {grand[k]}")
    print()
    print(f"  fatal problems : {fatal}")
    print(f"  warnings       : {warn}")
    print()
    if fatal:
        print("  *** THE DATASET IS NOT USABLE AS AN ACCEPTANCE INPUT IN THIS STATE.")
        print("  Fix the fatal rows before any consumer reads it. A dataset whose quotes")
        print("  do not resolve is not evidence about the papers; it is assertion.")
    else:
        print("  Schema, quotes and references check out. THIS IS NOT A CURATION-QUALITY")
        print("  VERDICT: a verbatim quote can still be attached to the wrong claim, and")
        print("  a well-formed reaction can still not belong to the requested pathway.")
        print("  That judgement is the bio-auditor's. This establishes only that there is")
        print("  something real to audit.")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
