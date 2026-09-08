"""C-120 -- deterministic replay of the two ORCH-732 identity failures.

Read-only. Imports the production functions and replays them on the ARCHIVED
ORCH-732 payload rows. Nothing is re-run through an LLM, no network is touched
by this module itself, and no run directory is written.

PART A -- ``PMC11487621``: three species entities for one organism.
PART B -- ``PMC10031235``: ``PSAT`` -> the literal ``Unknown`` sentinel.

Both parts print the exact production verdict so the mechanism is *measured*,
not inferred.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

RUN = _REPO / "runs_smoke" / "2026-09-07_2323" / "papers"


def _load(paper: str, name: str) -> Dict[str, Any]:
    path = RUN / paper / "strict" / name
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def part_a() -> Dict[str, Any]:
    from t2pw.pwml.ir import _canonicalize_species_offline, _deterministic_species_name

    out: Dict[str, Any] = {"paper": "PMC11487621", "deterministic_species_name": {}}

    # The exact strings the archived payload carries as ``raw_name`` / ``name``.
    for probe in (
        "Bacillus subtilis 168",
        "Bacillus subtilis (strain 168)",
        "Bacillus subtilis (strain",
        "B. subtilis",
        "Escherichia coli (strain K-12)",
        "Herbaspirillum huttiense subsp. huttiense IAM 15032",
    ):
        out["deterministic_species_name"][probe] = _deterministic_species_name(probe)

    payload = _load("PMC11487621", "final_mapped.json")
    rows = payload.get("entities", {}).get("species", [])
    out["archived_species_rows"] = [
        {
            "index": index,
            "name": row.get("name"),
            "raw_name": row.get("raw_name"),
            "taxonomy_id": row.get("taxonomy_id"),
            "classification": row.get("classification"),
            "pathbank_species_id": row.get("pathbank_species_id"),
            "resolution": (row.get("mapping_meta") or {}).get("resolution"),
            "canonicalization": row.get("species_canonicalization"),
        }
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    ]

    # Re-run the production canonicalizer on a fresh copy of the pre-canonical
    # name, exactly as ``resolve_species_prefreeze`` does.
    replays: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        record = {
            "name": row.get("raw_name") or row.get("name"),
            "taxonomy_id": row.get("taxonomy_id"),
        }
        for key in ("pathbank_species_id", "pw_species_id", "pathwhiz_id"):
            if row.get(key) is not None:
                record[key] = row[key]
        report: Dict[str, Any] = {}
        status = _canonicalize_species_offline(record, name_index=None, report=report)
        replays.append(
            {
                "index": index,
                "input_name": record.get("name"),
                "output_name": record.get("name"),
                "status": status,
            }
        )
        # ``record`` is mutated in place, so re-read after the call.
        replays[-1]["output_name"] = record.get("name")
    out["canonicalizer_replay"] = replays

    gate = _load("PMC11487621", "final_stage3_gate_report.json")
    out["final_stage3_gate_errors"] = [
        entry for entry in (gate.get("errors") or []) if isinstance(entry, (dict, str))
    ][:20]
    return out


def part_b() -> Dict[str, Any]:
    from t2pw.mapping.map_ids import (
        _REAL_PROTEIN_MIN_MARGIN,
        _REAL_PROTEIN_MIN_SCORE,
        _name_gate_verdict,
        verify_real_protein_identity,
    )

    out: Dict[str, Any] = {
        "paper": "PMC10031235",
        "thresholds": {
            "min_score": _REAL_PROTEIN_MIN_SCORE,
            "min_margin": _REAL_PROTEIN_MIN_MARGIN,
        },
    }

    gate_fail = _load("PMC10031235", "gate_fail_report.json")
    detail: Dict[str, Any] = {}
    for error in gate_fail.get("errors", []):
        candidate = (error or {}).get("detail")
        if isinstance(candidate, dict) and str(candidate.get("name", "")).upper().startswith("PSAT"):
            detail = candidate
            break
    out["archived_psat_row_found"] = bool(detail)
    meta = (detail.get("mapping_meta") or {}) if detail else {}
    rejected = [row for row in (meta.get("rejected_candidates") or []) if isinstance(row, dict)]
    out["archived_rejected_candidates"] = rejected
    out["archived_ambiguity"] = meta.get("ambiguity")
    out["archived_resolution"] = meta.get("resolution")

    entity_name = str(detail.get("name") or "PSAT")
    organism = str(detail.get("organism") or "Homo sapiens")

    verdicts: List[Dict[str, Any]] = []
    for candidate in rejected:
        accession = str(candidate.get("uniprot") or candidate.get("accession") or "").strip()
        mapped_ids: Dict[str, Any] = {}
        if accession:
            mapped_ids["uniprot"] = accession
        if candidate.get("pathbank_protein_id"):
            mapped_ids["pathbank_protein_id"] = str(candidate["pathbank_protein_id"])
        verdict = verify_real_protein_identity(
            entity_name,
            candidates=rejected,
            mapped_ids=mapped_ids,
            organism=organism,
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
                "score": verdict.get("score"),
                "margin": verdict.get("margin"),
                "verification_status": verdict.get("verification_status"),
            }
        )
        gate = _name_gate_verdict(
            entity_name,
            candidates=rejected,
            mapped_ids=mapped_ids,
            kind="protein",
            organism=organism,
            fallback_name=str(candidate.get("name") or ""),
            source="db",
        )
        verdicts[-1]["name_gate"] = gate
    out["replayed_verdicts"] = verdicts

    payload = _load("PMC10031235", "final_mapped.json")
    out["shipped_proteins"] = [
        {
            "index": index,
            "name": row.get("name"),
            "uniprot": (row.get("mapped_ids") or {}).get("uniprot"),
            "chosen_rule": (row.get("mapping_meta") or {}).get("chosen_rule"),
            "fallback_reason": (row.get("mapping_meta") or {}).get("fallback_reason"),
        }
        for index, row in enumerate(payload.get("entities", {}).get("proteins", []))
        if isinstance(row, dict)
    ]
    return out


def main() -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    report = {"part_a": part_a(), "part_b": part_b()}
    text = json.dumps(report, indent=1, ensure_ascii=False, default=str)
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
