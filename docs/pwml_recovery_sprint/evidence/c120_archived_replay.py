"""C-120 § 9 -- ARCHIVED ORCH-732 REPLAY. Deterministic, no LLM, no live network.

Replays the two blocked ORCH-732 legs through the **production** predicates at the
merged tip and asks one question of each: *does the blocking gate issue disappear?*

Nothing here re-runs a pipeline, contacts an LLM, or bypasses a guard. The one
external dependency -- the NCBI taxonomy lookup that mechanism A's tier 2 needs --
is served by a stub that replays the answer **this same archived run already
recorded** (`PMC11487621` species[0] carries
``taxonomy_backfill: {source: ncbi, taxonomy_id: 1423, classification: Prokaryote}``).
The stub answers only for the expanded binomial, so it also proves the pre-existing
raw-name attempt still runs first and still misses.

Stated limitation, per § 9: a full export replay is not possible from archived
payloads alone, so this proves the **mapping and gate predicates**, not a written
PWML file. The § 12 live validation is what settles the file.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

RUN = _REPO / "runs_smoke" / "2026-09-07_2323" / "papers"

#: The answer the archived run itself recorded for the expanded binomial.
_ARCHIVED_NCBI = {"Bacillus subtilis": ("1423", "Bacteria", "cellular organisms; Bacteria")}


class _ArchivedTaxonomyClient:
    """Replays the archived NCBI answer for the expanded binomial and nothing else.

    Any other term misses, exactly as it did in the real run. Every call is
    recorded so the caller can prove which queries production actually issued.
    """

    def __init__(self) -> None:
        self.calls: List[str] = []

    class _Resp:
        def __init__(self, payload: Any = None, text: str = "") -> None:
            self.status_code = 200
            self._payload = payload
            self.text = text

        def json(self) -> Any:
            if self._payload is None:
                raise ValueError("not json")
            return self._payload

    def get(self, url: str, params: Dict[str, Any] | None = None) -> Any:
        params = params or {}
        if "esearch" in url:
            term = str(params.get("term") or "")
            self.calls.append(f"esearch {term}")
            base = term.replace("[Scientific Name]", "").strip()
            hit = _ARCHIVED_NCBI.get(base)
            idlist = [hit[0]] if hit else []
            return self._Resp(payload={"esearchresult": {"idlist": idlist}})
        taxid = str(params.get("id") or "")
        self.calls.append(f"efetch {taxid}")
        for value in _ARCHIVED_NCBI.values():
            if value[0] == taxid:
                return self._Resp(
                    text=(
                        "<TaxaSet><Taxon>"
                        f"<TaxId>{value[0]}</TaxId>"
                        f"<Division>{value[1]}</Division>"
                        f"<Lineage>{value[2]}</Lineage>"
                        "</Taxon></TaxaSet>"
                    )
                )
        return self._Resp(text="")


def _load(paper: str, name: str) -> Dict[str, Any]:
    with open(RUN / paper / "strict" / name, encoding="utf-8") as handle:
        return json.load(handle)


def _species_errors(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    from t2pw.pwml.ir import validate_required_pwml_contract

    report = validate_required_pwml_contract(payload, strict_db=True)
    return [
        {"code": entry.get("code"), "pointer": entry.get("pointer"), "message": entry.get("message")}
        for entry in report.get("errors", [])
        if isinstance(entry, dict) and str(entry.get("code", "")).startswith("species_")
    ]


def part_a() -> Dict[str, Any]:
    """PMC11487621 -- do the three organism variants converge, and does the gate clear?"""
    from t2pw.mapping.map_ids import backfill_species_taxonomy

    out: Dict[str, Any] = {"paper": "PMC11487621"}
    payload = _load("PMC11487621", "final_mapped.json")

    # Reconstruct the species section as the alias pass actually sees it: the two
    # rows the existing loop already resolved, under their pre-canonical names,
    # and the abbreviation still carrying nothing.
    rows: List[Dict[str, Any]] = []
    for row in payload.get("entities", {}).get("species", []):
        if not isinstance(row, dict):
            continue
        rebuilt: Dict[str, Any] = {"name": row.get("raw_name") or row.get("name")}
        if row.get("taxonomy_id"):
            rebuilt["taxonomy_id"] = row["taxonomy_id"]
        if row.get("classification"):
            rebuilt["classification"] = row["classification"]
        for key in ("pathbank_species_id", "species_id"):
            if row.get(key) is not None:
                rebuilt[key] = row[key]
        rows.append(rebuilt)
    out["input_rows"] = deepcopy(rows)

    client = _ArchivedTaxonomyClient()
    replay = {"entities": {"species": rows}}
    report = backfill_species_taxonomy(replay, client=client, enable_ncbi=True)
    out["backfill_report"] = report
    out["ncbi_calls"] = client.calls
    out["output_rows"] = [
        {
            "name": row.get("name"),
            "taxonomy_id": row.get("taxonomy_id"),
            "classification": row.get("classification"),
            "source": (row.get("mapping_meta") or {}).get("taxonomy_backfill", {}).get("source"),
        }
        for row in rows
    ]

    # The gate, on the archived payload, before and after the repaired species list.
    before = deepcopy(payload)
    out["gate_species_errors_before"] = _species_errors(before)
    after = deepcopy(payload)
    for index, row in enumerate(after.get("entities", {}).get("species", [])):
        if not isinstance(row, dict) or index >= len(rows):
            continue
        if rows[index].get("taxonomy_id"):
            row["taxonomy_id"] = rows[index]["taxonomy_id"]
        if rows[index].get("classification"):
            row["classification"] = rows[index]["classification"]
    out["gate_species_errors_after"] = _species_errors(after)
    return out


def part_b() -> Dict[str, Any]:
    """PMC10031235 -- does PSAT still degrade to Unknown, and does the gate clear?"""
    from t2pw.mapping.map_ids import verify_real_protein_identity
    from t2pw.pwml.ir import validate_required_pwml_contract

    out: Dict[str, Any] = {"paper": "PMC10031235"}

    gate_fail = _load("PMC10031235", "gate_fail_report.json")
    psat: Dict[str, Any] = {}
    for error in gate_fail.get("errors", []):
        detail = (error or {}).get("detail")
        if isinstance(detail, dict) and str(detail.get("name", "")).upper().startswith("PSAT"):
            psat = detail
            break
    meta = psat.get("mapping_meta") or {}
    candidates = [row for row in (meta.get("rejected_candidates") or []) if isinstance(row, dict)]
    out["archived_candidates"] = candidates

    verdicts = []
    for candidate in candidates:
        accession = str(candidate.get("uniprot") or "").strip()
        mapped_ids: Dict[str, Any] = {}
        if accession:
            mapped_ids["uniprot"] = accession
        if candidate.get("pathbank_protein_id"):
            mapped_ids["pathbank_protein_id"] = str(candidate["pathbank_protein_id"])
        verdict = verify_real_protein_identity(
            str(psat.get("name") or "PSAT"),
            candidates=candidates,
            mapped_ids=mapped_ids,
            organism=str(psat.get("organism") or ""),
            source="db",
            resolved_name=str(candidate.get("name") or ""),
            result={"confidence": candidate.get("score")},
        )
        verdicts.append(
            {
                "accession": accession,
                "verified": verdict.get("verified"),
                "reason": verdict.get("reason"),
                "checks": verdict.get("checks"),
                "name_gate_reason": (verdict.get("name_gate") or {}).get("reason"),
            }
        )
    out["replayed_verdicts"] = verdicts

    # Reproduce the blocking gate on the pre-fallback payload, then apply the
    # identity the ladder now verifies and ask the same gate again.
    payload = _load("PMC10031235", "final_mapped.json")
    proteins = payload.get("entities", {}).get("proteins", [])
    before = deepcopy(payload)
    if len(proteins) > 1:
        before["entities"]["proteins"][1] = deepcopy(psat)

    def _protein_errors(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        report = validate_required_pwml_contract(doc, strict_db=True)
        return [
            {"code": entry.get("code"), "pointer": entry.get("pointer")}
            for entry in report.get("errors", [])
            if isinstance(entry, dict) and "protein" in str(entry.get("code", ""))
        ]

    out["gate_protein_errors_before"] = _protein_errors(before)

    after = deepcopy(before)
    verified = next((row for row in verdicts if row.get("verified")), None)
    if verified and len(after["entities"]["proteins"]) > 1:
        row = after["entities"]["proteins"][1]
        row["uniprot"] = verified["accession"]
        row["uniprot_id"] = verified["accession"]
        row.setdefault("mapped_ids", {})["uniprot"] = verified["accession"]
        for candidate in candidates:
            if str(candidate.get("uniprot") or "") == verified["accession"]:
                row["pathbank_protein_id"] = candidate.get("pathbank_protein_id")
    out["gate_protein_errors_after"] = _protein_errors(after)
    out["applied_identity"] = verified["accession"] if verified else None
    return out


def main() -> int:
    report = {"part_a": part_a(), "part_b": part_b()}
    sys.stdout.write(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
