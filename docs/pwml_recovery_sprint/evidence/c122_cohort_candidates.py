"""C-122 final smoke: find and verify FRESH candidate papers for the 12-leg cohort.

READ-ONLY against the repository. Issues search and full-text GETs to Europe PMC.
Writes only its own `--out` JSON. Touches no run tree, no cache, no production file.

WHY IT EXISTS
-------------
`ORCH-734`'s cohort was hand-built and its verification artifact
(`evidence/orch724_cohort_verification.json`) was committed WITHOUT the script that
produced it, so the selection could not be reproduced or re-run. This tool is that
script, written down.

THE SELECTION RULE, inherited verbatim from `ORCH-734`'s freeze
---------------------------------------------------------------
Ordinary, reasonable pathway papers: clear pathway identity, a known organism,
several reactions or strong pathway anchors, named enzymes, named substrates and
products. **NOT adversarial.** Deliberately excluded: case reports, resistome
surveys, pure omics enrichment, inhibitor-only screening, negative/context-only
papers, multi-organism ambiguity traps, and documents made huge solely to stress
Stage 1.

**This tool does not choose the cohort.** It produces a scored candidate list with
the evidence for each, and refuses candidates that fail a hard rule. A human picks
the twelve. Selecting on the score alone would optimise the cohort against the
metric it is supposed to measure, which section 19 of the card forbids
("Do not manipulate the cohort to achieve the number").

HARD EXCLUSIONS, applied mechanically
-------------------------------------
* any PMC id reachable from the gold set, any `runs*/` tree, or any `topics*.txt`;
* any paper whose title matches a used TOPIC (the sprint's standing topic
  exclusions plus every topic already consumed);
* anything that fails to fetch, or whose full text is shorter than a floor.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
PAUSE = 0.4

# Topics already consumed by the sprint, or standing traps. A candidate whose title
# hits one of these is refused: reusing a topic makes the run a re-measurement of a
# known result rather than a fresh yield estimate.
EXCLUDED_TOPICS = [
    "enterobactin", "lipid a", "heme", "cholesterol", "molybdenum cofactor",
    "terpenoid indole alkaloid", "wall teichoic acid", "pyoverdine", "sphingolipid",
    "ormdl", "menaquinone", "thiamine", "coenzyme a", "riboflavin", "l-serine",
    "coenzyme q", "acarbose", "adp-heptose", "tubercidin", "carnitine",
    "sialic acid", "corticosteroid", "nicotine", "furochromone", "glycoalkaloid",
    "cyanogenic", "cycloclavine", "jasmonate", "psat", "phosphoserine",
]

# Shapes the selection rule refuses outright.
#
# The review/overview block was added after the first pass of this tool, sorted by
# citation count, returned almost nothing BUT famous reviews -- they are the most
# cited documents matching any biology query. A review states that pathways exist
# without establishing individual reactions from evidence, which is exactly what
# the selection rule refuses. Sorting is now by date and the title must itself
# name a biosynthesis.
REFUSED_TITLE_PATTERNS = [
    r"\bcase report\b", r"\bresistome\b", r"\bmeta-analys", r"\bsystematic review\b",
    r"\bscoping review\b", r"\bclinical trial\b", r"\bcohort study\b",
    r"\bepidemiolog", r"\bsurveillance\b", r"\bquestionnaire\b",
    r"\bcrystal structure\b", r"\binhibitor screen", r"\bdocking\b",
    r"\breview\b", r"\boverview\b", r"\bperspectiv", r"\bmechanisms of\b",
    r"\bpast, present\b", r"\bprologue\b", r"\breference resource\b",
    r"\bmulti-omics\b", r"\bimplications\b", r"\brecent advances\b",
    r"\bcurrent status\b", r"\bemerging\b", r"\bupdate\b",
]

# The title must itself name a biosynthesis or pathway. A paper that only mentions
# one somewhere in its body is usually about something else.
REQUIRED_TITLE_PATTERNS = [
    r"\bbiosynthe", r"\bbiosynthetic pathway\b", r"\bpathway\b", r"\bmetabolit",
]

REACTION_CUES = [
    "catalyz", "catalys", "converts", "conversion of", "is converted",
    "substrate", "product of", "intermediate", "biosynthetic step",
    "reaction", "transferase", "synthase", "reductase", "oxidase",
    "dehydrogenase", "hydroxylase", "ligase", "isomerase", "decarboxylase",
]
ENZYME_WORDS = [
    "enzyme", "gene", "encodes", "encoded by", "knockout", "mutant",
    "recombinant", "purified", "assay", "in vitro",
]


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "t2pw-c122-cohort"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("utf-8", "replace")


def used_pmcids(root: Path) -> Set[str]:
    """Every PMC id reachable from gold, any run tree, or any topics file."""
    used: Set[str] = set()
    for f in glob.glob(str(root / "topics_*.txt")):
        used.update(re.findall(r"PMC\d+", Path(f).read_text(encoding="utf-8", errors="replace")))
    for d in ("runs_smoke", "runs_validation", "runs_verify", "runs"):
        for p in glob.glob(str(root / d / "**" / "papers" / "*"), recursive=True):
            b = os.path.basename(p)
            m = re.match(r"(PMC\d+)", b)
            if m:
                used.add(m.group(1))
    for f in glob.glob(str(root / "src" / "t2pw" / "bench" / "gold" / "*.json")):
        used.update(re.findall(r"PMC\d+", Path(f).read_text(encoding="utf-8", errors="replace")))
    return used


def xml_to_text(xml: str) -> str:
    body = re.sub(r"<(ref-list|back|front)\b.*?</\1>", " ", xml, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--query",
        default=('TITLE:"biosynthesis" AND OPEN_ACCESS:Y AND SRC:MED '
                 'AND (PUB_TYPE:"Journal Article") NOT (PUB_TYPE:"Review")'))
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--max-verify", type=int, default=40, help="hard cap on full-text fetches")
    ap.add_argument("--min-chars", type=int, default=18000)
    ap.add_argument("--max-chars", type=int, default=140000)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    used = used_pmcids(root)
    print("=" * 104)
    print("C-122 fresh cohort candidate search -- scored evidence, NOT a selection")
    print("=" * 104)
    print(f"already-used PMC ids excluded : {len(used)}")
    print(f"excluded topics               : {len(EXCLUDED_TOPICS)}")
    print(f"full-text fetch cap           : {args.max_verify}")
    print()

    url = SEARCH + "?" + urllib.parse.urlencode(
        {"query": args.query, "format": "json", "pageSize": str(args.page_size),
         "resultType": "core", "sort": "P_PDATE_D desc"})
    hits = json.loads(get(url)).get("resultList", {}).get("result", []) or []
    print(f"search returned {len(hits)} records\n")

    candidates: List[Dict[str, Any]] = []
    verified = 0
    for h in hits:
        if verified >= args.max_verify:
            break
        pmcid = (h.get("pmcid") or "").strip()
        title = (h.get("title") or "").strip()
        if not pmcid or pmcid in used:
            continue
        low = title.lower()
        topic_hit = next((t for t in EXCLUDED_TOPICS if t in low), "")
        if topic_hit:
            candidates.append({"pmcid": pmcid, "title": title, "ok": False,
                               "refused": f"excluded_topic:{topic_hit}"})
            continue
        if not any(re.search(p, low) for p in REQUIRED_TITLE_PATTERNS):
            candidates.append({"pmcid": pmcid, "title": title, "ok": False,
                               "refused": "title_does_not_name_a_pathway"})
            continue
        shape_hit = next((p for p in REFUSED_TITLE_PATTERNS if re.search(p, low)), "")
        if shape_hit:
            candidates.append({"pmcid": pmcid, "title": title, "ok": False,
                               "refused": f"refused_shape:{shape_hit}"})
            continue

        try:
            text = xml_to_text(get(FULLTEXT.format(pmcid=pmcid)))
            err = ""
        except Exception as exc:  # noqa: BLE001
            text, err = "", f"{type(exc).__name__}: {exc}"[:120]
        verified += 1
        time.sleep(PAUSE)

        if err or not text:
            candidates.append({"pmcid": pmcid, "title": title, "ok": False,
                               "refused": f"no_fulltext:{err or 'empty'}"})
            continue
        low_t = text.lower()
        cues = sum(low_t.count(c) for c in REACTION_CUES)
        enz = sum(low_t.count(w) for w in ENZYME_WORDS)
        ok = args.min_chars <= len(text) <= args.max_chars and cues >= 12 and enz >= 8
        refused = ""
        if len(text) < args.min_chars:
            refused = f"too_short:{len(text)}"
        elif len(text) > args.max_chars:
            refused = f"too_long:{len(text)}"
        elif cues < 12:
            refused = f"weak_reaction_cues:{cues}"
        elif enz < 8:
            refused = f"weak_enzyme_evidence:{enz}"

        row = {"pmcid": pmcid, "title": title, "journal": h.get("journalTitle", ""),
               "year": h.get("pubYear", ""), "chars": len(text),
               "reaction_cue_hits": cues, "enzyme_word_hits": enz,
               "ok": bool(ok), "refused": refused}
        candidates.append(row)
        mark = "OK " if ok else "no "
        print(f"  {mark}{pmcid:<12} chars={len(text):>7} cues={cues:>4} enz={enz:>4}  {title[:70]}")
        if refused:
            print(f"       refused: {refused}")

    passing = [c for c in candidates if c.get("ok")]
    print()
    print("=" * 104)
    print(f"verified {verified} full texts | PASSING CANDIDATES: {len(passing)} | refused: {len(candidates) - len(passing)}")
    print("A human selects twelve from the passing list, balancing organism mix and")
    print("assigning each a scope string taken from the paper's OWN language.")
    print("Selecting purely on score would optimise the cohort against its own metric.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"tool": "c122_cohort_candidates.py", "query": args.query,
         "used_excluded": len(used), "verified": verified,
         "passing": len(passing), "candidates": candidates}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
