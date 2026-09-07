"""ORCH-725: is the residual failure class DATABASE COVERAGE or QUERY CONSTRUCTION?

Read-only diagnostic tooling. Nothing under ``src/`` imports it. It writes no
cache, touches no run artifact and changes no production behaviour. It issues a
small, fixed number of GET requests to the public UniProt REST search endpoint
and prints what comes back.

Why this probe exists
---------------------
``orch725_identity_probe.py`` PART B classifies 12 refusals as *"NO
ORGANISM-MATCHING RECORD IN POOL"* -- the pool the resolver judged contained no
row for the requested organism. That is consistent with TWO different
diagnoses which demand opposite product responses:

* **coverage (class B)** -- UniProt genuinely has no such protein, the refusal
  is correct, and no query change can recover it;
* **query construction (class C)** -- the record exists and our query did not
  reach it, in which case the entity is recoverable.

The run's artifacts cannot separate these. They record the pool that came back,
not what a differently-shaped query would have returned. So each probed name is
asked three ways, exactly as the production ladder shapes them
(``map_ids.py`` records the strings in ``mapping_meta.queries_tried``):

1. the production species-restricted form;
2. the production unrestricted fallback;
3. a control the production ladder does NOT issue -- for organism strings, the
   bare binomial with any strain parenthetical removed; for names, the
   species-prefix-stripped symbol.

Difference between 1 and 3 is query construction. Zero on all three is coverage.

The probe reports counts and the top rows; it does not decide anything, and a
result here is not a licence to change the resolver.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,protein_name,gene_names,organism_name,reviewed"
TIMEOUT = 25
PAUSE = 0.6  # courtesy delay between requests; this is a probe, not a benchmark

#: (label, entity name, organism as the pipeline held it, control name, control organism)
#: Chosen to cover every organism group the pilot touched and every mechanism
#: PART B named, not to maximise hits.
CASES: Tuple[Tuple[str, str, str, Optional[str], Optional[str]], ...] = (
    # --- bacteria: strain parenthetical in the organism string (PMC12071552) ---
    ("bacteria/strain-parenthetical", "DltA", "Staphylococcus aureus (MRSA N315)", None, "Staphylococcus aureus"),
    ("bacteria/strain-parenthetical", "DltC", "Staphylococcus aureus (MRSA N315)", None, "Staphylococcus aureus"),
    # --- fungi: Neurospora, incl. the two-domain nit-9 product (PMC7232280) ---
    ("fungi", "NIT-9G", "Neurospora crassa", "nit-9", "Neurospora crassa"),
    ("fungi", "NIT-9E", "Neurospora crassa", "nit-9", "Neurospora crassa"),
    ("fungi", "NIT-1", "Neurospora crassa", "nit-1", "Neurospora crassa"),
    ("fungi", "NIT-12", "Neurospora crassa", "nit-12", "Neurospora crassa"),
    # --- plants: Catharanthus, the species-abbreviation prefix (PMC8510960) ---
    ("plants", "CrNPF2.9", "Catharanthus roseus", "NPF2.9", "Catharanthus roseus"),
    ("plants", "CrTPT2", "Catharanthus roseus", "TPT2", "Catharanthus roseus"),
    ("plants", "CrGATA1", "Catharanthus roseus", "GATA1", "Catharanthus roseus"),
    ("plants", "CrPIF1", "Catharanthus roseus", "PIF1", "Catharanthus roseus"),
    ("plants", "BIS1", "Catharanthus roseus", None, "Catharanthus roseus"),
    # --- human/mammal control: the margin-rejected reviewed entries ---
    ("human", "ORMDL1", "Homo sapiens", None, "Homo sapiens"),
    ("human", "sphingomyelinase", "Homo sapiens", "SMPD1", "Homo sapiens"),
)


def query(term: str, organism: Optional[str], size: int = 10) -> Tuple[Optional[int], List[Dict[str, Any]], str]:
    """``(status, rows, note)``. ``status`` is None when the request failed."""
    clause = f'(protein_name:"{term}" OR gene:"{term}")'
    if organism:
        clause += f' AND organism_name:"{organism}"'
    url = f"{ENDPOINT}?" + urllib.parse.urlencode(
        {"query": clause, "fields": FIELDS, "size": str(size), "format": "json"}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "T2PW-ORCH725-readonly-probe"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as handle:
            body = json.loads(handle.read().decode("utf-8"))
        return handle.status if hasattr(handle, "status") else 200, body.get("results") or [], ""
    except urllib.error.HTTPError as exc:
        return exc.code, [], f"HTTPError {exc.code}"
    except Exception as exc:  # noqa: BLE001 -- a probe reports transport failures, it does not raise
        return None, [], f"{type(exc).__name__}: {exc}"


def describe(row: Dict[str, Any]) -> str:
    protein = (
        ((row.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName") or {}
    ).get("value") or ""
    if not protein:
        submitted = (row.get("proteinDescription") or {}).get("submissionNames") or []
        if submitted:
            protein = ((submitted[0] or {}).get("fullName") or {}).get("value") or ""
    genes = ",".join(
        ((g.get("geneName") or {}).get("value") or "") for g in (row.get("genes") or [])
    )
    org = (row.get("organism") or {}).get("scientificName") or ""
    reviewed = row.get("entryType") or ""
    return (
        f"{row.get('primaryAccession',''):<12} {protein[:44]:<46} "
        f"gene={genes[:22]:<24} org={org[:34]:<36} {'REVIEWED' if 'Swiss' in reviewed else 'unreviewed'}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    print("=" * 100)
    print("ORCH-725 -- UniProt read-only coverage probe: is the residual class COVERAGE or QUERY?")
    print("=" * 100)
    print(f"endpoint : {ENDPOINT}")
    print(f"cases    : {len(CASES)}   (3 requests each, {PAUSE}s apart; no cache is read or written)")
    print()

    stats = {"requests": 0, "http_ok": 0, "errors": 0, "zero_hit": 0, "with_hits": 0}
    verdicts: List[Tuple[str, str, str]] = []

    for group, name, organism, ctrl_name, ctrl_org in CASES:
        print("#" * 100)
        print(f"[{group}]  {name}   (pipeline organism: {organism!r})")
        results: Dict[str, Tuple[Optional[int], List[Dict[str, Any]], str]] = {}
        plan = [
            ("1 production, species-restricted", name, organism),
            ("2 production, unrestricted", name, None),
        ]
        if ctrl_org and ctrl_org != organism:
            plan.append((f"3 CONTROL, organism -> {ctrl_org!r}", name, ctrl_org))
        if ctrl_name and ctrl_name != name:
            plan.append((f"3 CONTROL, name -> {ctrl_name!r}", ctrl_name, ctrl_org or organism))

        for label, term, org in plan:
            status, rows, note = query(term, org)
            stats["requests"] += 1
            if status is None:
                stats["errors"] += 1
            else:
                stats["http_ok"] += 1
                if rows:
                    stats["with_hits"] += 1
                else:
                    stats["zero_hit"] += 1
            results[label] = (status, rows, note)
            print(f"  {label:<44} http={status} hits={len(rows)} {note}")
            for row in rows[:3]:
                print(f"        {describe(row)}")
            time.sleep(PAUSE)

        restricted = results.get("1 production, species-restricted", (None, [], ""))[1]
        controls = [
            rows for label, (_s, rows, _n) in results.items() if label.startswith("3 CONTROL")
        ]
        best_control = max((len(r) for r in controls), default=0)
        if restricted:
            verdict = "production query already reached it"
        elif best_control:
            verdict = "QUERY CONSTRUCTION -- production query found 0, a control found records"
        else:
            unrestricted = results.get("2 production, unrestricted", (None, [], ""))[1]
            same_org = [
                r
                for r in unrestricted
                if re.sub(r"\(.*?\)", "", (r.get("organism") or {}).get("scientificName") or "")
                .strip()
                .casefold()
                .startswith(re.sub(r"\(.*?\)", "", organism).strip().casefold()[:18])
            ]
            verdict = (
                "QUERY CONSTRUCTION -- the unrestricted pool DOES hold organism-matching rows"
                if same_org
                else "COVERAGE -- no organism-matching record found by any form"
            )
        verdicts.append((group, name, verdict))
        print(f"  => {verdict}")
        print()

    print("=" * 100)
    print("PROBE SUMMARY")
    print("=" * 100)
    print(f"  requests attempted : {stats['requests']}")
    print(f"  HTTP success       : {stats['http_ok']}")
    print(f"  transport errors   : {stats['errors']}")
    print(f"  zero-hit responses : {stats['zero_hit']}")
    print(f"  responses with hits: {stats['with_hits']}")
    print()
    for group, name, verdict in verdicts:
        print(f"  {group:<32} {name:<20} {verdict}")
    print()
    print("  A 'COVERAGE' line is a correct refusal that no query change can recover.")
    print("  A 'QUERY CONSTRUCTION' line is a recoverable entity the resolver did not reach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
