"""C-122 pre-charter probe: is the archived identity residue RECOVERABLE at all?

READ-ONLY. No LLM call, no cache read or write, no run artifact touched, no
production import. It issues a small fixed number of GET requests to the public
UniProt REST search endpoint and prints what comes back.

WHY THIS RUNS BEFORE THE CARD IS CHARTERED
------------------------------------------
Every unresolved protein in the two archived legs failed the same way:

    order_step   : name_plausibility_gate
    issue        : implausible_name_match
    failed_check : species_mismatch

and in every case UniProt HAD returned 5-8 candidates. For the ergot cluster the
refused candidate carries the queried gene symbol verbatim in its own protein
name (`EasF` -> "4-dimethylallyltryptophan N-methyltransferase easF") but belongs
to a DIFFERENT fungus. So the refusal is the species safeguard doing its job, and
`C-122` may not weaken it (card sections 7 and 25).

That leaves exactly one way `C-122` could recover these entities without touching
the safeguard: **the requested organism's own record exists and our query did not
reach it.** This probe decides that, per protein, and nothing else.

    hits under the REQUESTED organism   -> recoverable by query construction
    zero under the requested organism   -> coverage. The refusal is correct and
                                           no resolver change can recover it.

`ORCH-725` section 7 ran the same discipline on a different residue and found the
answer was query construction for two narrow conventions and coverage for the
rest. Nothing here assumes that result carries over.

A result from this probe is a scoping input. It is NOT a licence to change the
resolver, and it makes no claim about whether a recovered accession is the
biologically correct one.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
PAUSE = 0.34
# The production fields list, verbatim from map_ids.py:4115 / :4187 / :4250, plus
# organism_id -- which production does NOT request. Its absence is why API-path
# candidates carry no taxonomy_id and the species check degrades to string
# comparison. Requesting it here shows what production is leaving on the table.
FIELDS = "accession,protein_name,gene_names,organism_name,reviewed,organism_id"

# (label, entity name, requested organism, archived outcome)
CASES: List[Tuple[str, str, str, str]] = [
    # C-121 PMC4471609 -- ergot alkaloid cluster, all 8 refused on species_mismatch
    ("C121/PMC4471609", "DmaW", "Aspergillus japonicus", "refused: A. fumigatus Q50EL0"),
    ("C121/PMC4471609", "EasF", "Aspergillus japonicus", "refused: Epichloe A2TBU2"),
    ("C121/PMC4471609", "EasE", "Aspergillus japonicus", "refused: Epichloe A2TBU3"),
    ("C121/PMC4471609", "EasC", "Aspergillus japonicus", "refused: E. nidulans C8VPT2"),
    ("C121/PMC4471609", "EasD", "Aspergillus japonicus", "refused: A. benhamiae D4AK45"),
    ("C121/PMC4471609", "EasA", "Aspergillus japonicus", "refused: E. nidulans C8VPS9"),
    ("C121/PMC4471609", "EasG", "Aspergillus japonicus", "refused: A. fumigatus Q4WZ69"),
    ("C121/PMC4471609", "EasH", "Aspergillus japonicus", "refused: Epichloe A2TBT9"),
    # ORCH-734 PMC13184244 -- nicotine, N. tabacum
    ("ORCH734/PMC13184244", "UGT1", "Nicotiana tabacum", "refused: HUMAN Q9HAW9"),
    ("ORCH734/PMC13184244", "MATE1", "Nicotiana tabacum", "refused: HUMAN Q96FL8"),
    ("ORCH734/PMC13184244", "β-GD1", "Nicotiana tabacum", "no_match, 0 queries tried"),
    # Card section 12 acceptance cases
    ("card-12A", "PSAT", "Homo sapiens", "RESOLVES today -> Q9Y617 (C-120 rung)"),
    ("card-12B", "OPC-8:CoA ligase 1 (OPCL1)", "Arabidopsis thaliana", "manual failure, no archived leg"),
    ("card-12B", "OPCL1", "Arabidopsis thaliana", "the parenthetical symbol alone"),
]


def query(term: str, organism: Optional[str]) -> Tuple[Optional[int], List[Dict[str, Any]], str]:
    if organism:
        q = f'(protein_name:"{term}" OR gene:"{term}") AND organism_name:"{organism}"'
    else:
        q = f'(protein_name:"{term}" OR gene:"{term}")'
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": q, "fields": FIELDS, "size": "10", "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "t2pw-c122-coverage-probe"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8", "replace"))
            return resp.status, list(body.get("results") or []), q
    except urllib.error.HTTPError as exc:
        return exc.code, [], q
    except Exception as exc:  # noqa: BLE001
        return None, [], f"{type(exc).__name__}: {exc}"


def row_summary(r: Dict[str, Any]) -> Dict[str, Any]:
    org = (r.get("organism") or {})
    genes = []
    for g in r.get("genes") or []:
        nm = (g.get("geneName") or {}).get("value")
        if nm:
            genes.append(nm)
        for syn in g.get("synonyms") or []:
            if syn.get("value"):
                genes.append(syn["value"])
    desc = (r.get("proteinDescription") or {})
    rec = ((desc.get("recommendedName") or {}).get("fullName") or {}).get("value") or ""
    if not rec:
        subs = desc.get("submissionNames") or []
        if subs:
            rec = ((subs[0].get("fullName") or {}).get("value")) or ""
    return {
        "accession": r.get("primaryAccession"),
        "protein_name": rec,
        "genes": genes[:6],
        "organism": org.get("scientificName"),
        "taxon_id": org.get("taxonId"),
        "reviewed": r.get("entryType", "").startswith("UniProtKB reviewed"),
    }


def main() -> int:
    out_path = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    print("=" * 108)
    print("C-122 pre-charter coverage probe -- is the archived identity residue RECOVERABLE?")
    print("=" * 108)
    print(f"endpoint : {ENDPOINT}")
    print(f"fields   : {FIELDS}   (production omits organism_id)")
    print(f"cases    : {len(CASES)}, 2 requests each, {PAUSE}s apart. No cache, no LLM, no production import.")
    print()

    records = []
    verdicts = {"query_construction": 0, "coverage": 0, "error": 0}
    for label, name, organism, archived in CASES:
        print("#" * 108)
        print(f"[{label}]  {name!r}   requested organism: {organism!r}")
        print(f"    archived outcome: {archived}")
        st_r, rows_r, q_r = query(name, organism)
        time.sleep(PAUSE)
        st_u, rows_u, q_u = query(name, None)
        time.sleep(PAUSE)

        if st_r is None or st_u is None:
            verdict = "error"
        elif rows_r:
            verdict = "query_construction"
        else:
            verdict = "coverage"
        verdicts[verdict] += 1

        print(f"    species-restricted : HTTP {st_r}  hits={len(rows_r)}")
        for r in rows_r[:3]:
            s = row_summary(r)
            mark = "REVIEWED" if s["reviewed"] else "unreviewed"
            print(f"        {s['accession']}  {str(s['protein_name'])[:52]:<52} {str(s['organism'])[:30]:<30} tax={s['taxon_id']} {mark}")
            print(f"          genes: {s['genes']}")
        print(f"    unrestricted       : HTTP {st_u}  hits={len(rows_u)}")
        for r in rows_u[:2]:
            s = row_summary(r)
            print(f"        {s['accession']}  {str(s['protein_name'])[:52]:<52} {str(s['organism'])[:34]:<34} tax={s['taxon_id']}")
        print(f"    >>> VERDICT: {verdict.upper()}"
              + ("  -- the requested organism HAS a record; our query did not reach it"
                 if verdict == "query_construction"
                 else "  -- no record for the requested organism; the refusal is CORRECT"
                 if verdict == "coverage" else ""))
        print()

        records.append({
            "label": label, "name": name, "organism": organism, "archived": archived,
            "restricted_status": st_r, "restricted_hits": len(rows_r),
            "restricted_rows": [row_summary(r) for r in rows_r[:5]],
            "unrestricted_status": st_u, "unrestricted_hits": len(rows_u),
            "unrestricted_rows": [row_summary(r) for r in rows_u[:3]],
            "verdict": verdict, "query_restricted": q_r,
        })

    print("=" * 108)
    print("SUMMARY -- what each verdict means for C-122's scope")
    print("=" * 108)
    print(f"  query_construction : {verdicts['query_construction']:>3}   recoverable WITHOUT touching the species safeguard")
    print(f"  coverage           : {verdicts['coverage']:>3}   NOT recoverable by any resolver change; refusal is correct")
    print(f"  error              : {verdicts['error']:>3}")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"tool": "c122_coverage_probe.py", "fields": FIELDS,
             "verdicts": verdicts, "cases": records}, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
